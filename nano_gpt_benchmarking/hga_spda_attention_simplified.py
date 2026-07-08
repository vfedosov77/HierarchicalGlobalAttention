
# Chunk-routed sparse attention for the final (non-paired) layer of the nanoGPT speedrun.
#
# The layer receives a single fully-packed sequence [1, T, H, D] (B == 1). Because the
# stage-3 loader always fills the whole buffer, T is fixed and every token is real, so the
# chunk count C = T // chunk_size is STATIC across steps. That makes the whole path
# traceable under ``torch.compile(fullgraph=True)`` with NO custom ops:
#
#   * ROUTING: per-head windowed causal chunk-vs-chunk scoring over chunk mean-summaries,
#     head-averaged, then top-k of the strictly-past window. NO document masking -- the
#     packed sequence is treated as one stream, so a query chunk may route to ANY of its
#     ``window`` strictly-past chunks. Routing is a pure SELECTION, so it needs no softmax
#     and runs in the training dtype (bf16) under ``no_grad``.
#   * ATTEND: a single block-sparse FlexAttention call whose BlockMask is the own-chunk
#     diagonal (partial, intra-chunk causal) unioned with the routed past chunks (full
#     64x64 blocks). A persistent ``_RoutedFlexMaskState`` owns the BlockMask index buffers
#     and rewrites only the routed near-diagonal band in place each step, avoiding
#     ``BlockMask.from_kv_blocks``'s per-step O(C^2 log C) dense transpose.
#
# Union (no stop-gradient on routed K/V) is the only mode: it is both faster than the
# two-call detached merge and gives the exact gradient. Training-only: routing pools
# within-chunk future tokens (a routing-level causality leak in *which* chunks are picked);
# the attention values stay strictly causal. Use dense attention for validation.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def _qk_norm(x: Tensor) -> Tensor:
    # Match modded-nanogpt's norm(x): F.rms_norm(x, (x.size(-1),)).
    return F.rms_norm(x, (x.size(-1),))


# Inductor's sm90 flex-attention BACKWARD heuristic offers exactly one config for
# head_dim==128 -- FlexBwDConfig(64, 128, 128, 64) -- whose 128-wide tiles do not divide
# our 64-token mask blocks, so every choice is pruned and lowering dies with
# NoValidChoicesError. (kernel_options can't help: the divisibility filter runs on the
# config list BEFORE options are applied.)
#
# For head_dim==128 (our case) we therefore return a SINGLE valid 64-tile config. This
# does two things: (1) it guarantees a lowerable choice, and (2) it collapses the backward
# autotune search to one candidate. The backward lowering emits one autotune choice PER
# config, and at large C (e.g. 4800 chunks = 307K tokens) benchmarking each candidate on
# the full problem costs minutes -- that autotune sweep, not the attention itself, was the
# benchmark's bottleneck. One fixed config compiles in seconds. Other head dims keep the
# stock config list (with the 64-tile config appended so a valid choice always exists).
def _patch_flex_bwd_configs():
    try:
        from torch._inductor.template_heuristics.triton import (
            CUDAConfigHeuristic, FlexBwDConfig,
        )
    except ImportError:
        return
    if getattr(CUDAConfigHeuristic.get_flex_attn_bwd_configs, "_hga_patched", False):
        return
    orig = CUDAConfigHeuristic.get_flex_attn_bwd_configs
    single = FlexBwDConfig(64, 64, 64, 64, 2, 4)

    def patched(self, head_dim, dtype):
        if head_dim == 128:
            return [single]                       # one candidate -> no backward autotune
        cfgs = list(orig(self, head_dim, dtype))
        if single not in cfgs:
            cfgs.append(single)
        return cfgs

    patched._hga_patched = True
    CUDAConfigHeuristic.get_flex_attn_bwd_configs = patched


_patch_flex_bwd_configs()


def _causal_block_mask_mod(b, h, q_idx, kv_idx):
    # Only evaluated inside the diagonal (own-chunk) partial blocks; routed past blocks
    # are passed as full blocks and never call this.
    return q_idx >= kv_idx


def _route_static(q: Tensor, k: Tensor, chunk_size: int,
                  topk: int, softmax_scale: float, window: int):
    """Static-shape chunk routing: per-head windowed causal chunk-vs-chunk scoring over
    chunk mean-summaries, head-averaged, then top-k of the strictly-past window.

    NO document masking: a query chunk may route to ANY of its ``window`` strictly-past
    chunks (the packed sequence is treated as one stream). Routing is only a top-k
    SELECTION, so the per-row softmax an attention router would apply is unnecessary: top-k
    is invariant to softmax's monotonic per-row normalisation and the masking is done BEFORE
    the top-k. We score, head-average the RAW scores, mask out-of-range candidates with
    -inf, and top-k directly.

    Everything runs in the input (bf16) dtype -- no fp32. Requires T % chunk_size == 0 with
    every token real (fully packed buffer). Returns (sel [C, topk+1] long, slot_valid
    [C, topk+1] bool); the own chunk is the last slot and invalid past slots are
    left-compacted out by topk (a real score always beats the -inf sentinel).
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    C = T // CH
    dev = q.device

    # Chunk mean-summaries in the INPUT dtype (bf16 in training). Routing is a pure top-k
    # selection, so it does not need fp32: keeping bf16 makes the scoring einsum below a
    # bf16 tensor-core matmul (no fp32 matmul -> no TF32-fallback warning) and matches the
    # dtype the rest of the layer runs in. ``mean`` accumulates in fp32 internally, so the
    # summary is still accurate.
    q_sum = q.view(C, CH, H, D).mean(dim=1)                      # [C, H, D]
    k_sum = k.view(C, CH, H, D).mean(dim=1)

    # BANDED scores: a query chunk only ever routes within its trailing ``window`` chunks,
    # so score just the [C, W+1] band (offsets r = W..0, r==0 the own chunk) instead of a
    # dense [H, C, C] matrix -- that dense form is multi-GB of transients at large T while
    # the band is a few MB at any T. Column j holds offset r = W - j, so j == W is the own
    # chunk and key chunk index = c - W + j.
    cidx = torch.arange(C, device=dev)
    W = max(0, min(int(window), C - 1))
    kpad = F.pad(k_sum.reshape(C, H * D), (0, 0, W, 0))          # [W+C, H*D] front zero-pad
    kb = kpad.unfold(0, W + 1, 1).view(C, H, D, W + 1)           # [C, H, D, W+1] = k_sum[c-W+j]
    scores = torch.einsum("chd,chdj->hcj", q_sum * float(softmax_scale), kb)  # [H, C, W+1]

    # NO SOFTMAX, NO document mask: head-average the raw scores, mask only the front
    # zero-pad (key chunk index < 0 near the start), then top-k. The -inf sentinel can
    # never be picked and never beats a real (possibly negative) score.
    w = scores.mean(dim=0)                                       # [C, W+1] head-averaged raw
    # Strictly-past candidates only (drop j == W, the own chunk). Past column j (0..W-1)
    # maps to key chunk c - W + j, valid iff >= 0 (in range of the sequence).
    past_key = cidx[:, None] - W + torch.arange(W, device=dev)[None, :]  # [C, W]
    w = w[:, :W].masked_fill(past_key < 0, float("-inf"))        # [C, W]
    topv, topj = w.topk(min(int(topk), W), dim=-1)               # [C, topk]
    valid_past = torch.isfinite(topv)
    topi = (cidx[:, None] - W + topj).clamp(min=0)               # global key chunk index

    sel = torch.cat([topi, cidx[:, None]], dim=-1)               # [C, topk+1] own chunk last
    slot_valid = torch.cat(
        [valid_past, torch.ones(C, 1, dtype=torch.bool, device=dev)], dim=-1,
    )
    return sel, slot_valid


class _RoutedFlexMaskState:
    """Persistent BlockMask index buffers for a fixed chunk count ``C``.

    Owns the reusable block-index tensors and rewrites ONLY the routed near-diagonal band
    (bounded by the routing window ``W``) each step -- avoiding ``BlockMask.from_kv_blocks``'s
    per-step dense transpose (build a dense ``[1,1,C,C]`` mask, transpose it, argsort over
    ``C`` -> O(C^2 log C)) and its two per-step ``[1,1,C,C]`` allocations. The own-chunk
    diagonal is static (depends only on ``C``); the routed band is updated in place from
    ``sel``. The Q-major (backward) side is built directly from the routing band in O(C*W).

    One instance per ``(C, T, chunk_size, device)`` -- see ``_get_flex_mask_state``.
    Allocating the buffers on the first (compiled) forward and mutating them in place on
    later steps is ``torch.compile(fullgraph=True)``-safe: only the band columns are
    written, and the ``num_blocks`` tensors bound how many indices are read, so stale
    entries beyond the current selection are never consulted.
    """

    def __init__(self, C: int, T: int, chunk_size: int, device):
        self.C = int(C)
        self.T = int(T)
        self.CH = int(chunk_size)
        self.device = device
        ci = torch.arange(C, device=device, dtype=torch.int32)
        self._arangeC = torch.arange(C, device=device)
        # Static own-chunk diagonal (partial blocks, intra-chunk causal mask). Its Q-major
        # transpose is itself (one block per row, at column 0 == the diagonal), so the SAME
        # tensors serve both the KV-major and Q-major sides.
        self.diag_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=device)
        self.diag_idx[0, 0, :, 0] = ci
        self.diag_num = torch.ones(1, 1, C, dtype=torch.int32, device=device)
        # Routed band buffers, rewritten in place each step. Only the near-diagonal columns
        # are touched; the remainder stays 0 and is never read (bounded by the num tensors).
        self.full_kv_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=device)
        self.full_q_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=device)
        # Routed-band block COUNTS, also rewritten in place each step. Kept as persistent
        # buffers (not freshly allocated) so the BlockMask below can be built ONCE and
        # reused: flex reads these tensors by reference at call time, so in-place updates are
        # reflected without reconstructing the BlockMask.
        self.full_kv_num = torch.zeros(1, 1, C, dtype=torch.int32, device=device)
        self.full_q_num = torch.zeros(1, 1, C, dtype=torch.int32, device=device)
        # Own diagonal (partial) + routed band (full) BlockMask, built ONCE over the
        # persistent buffers above. ``update_band`` mutates those buffers in place each step,
        # and this same object is handed to flex every step -- no per-step BlockMask
        # construction, no dense transpose.
        self.mask = BlockMask(
            seq_lengths=(self.T, self.T),
            kv_num_blocks=self.diag_num, kv_indices=self.diag_idx,
            full_kv_num_blocks=self.full_kv_num, full_kv_indices=self.full_kv_idx,
            q_num_blocks=self.diag_num, q_indices=self.diag_idx,
            full_q_num_blocks=self.full_q_num, full_q_indices=self.full_q_idx,
            BLOCK_SIZE=(self.CH, self.CH), mask_mod=_causal_block_mask_mod,
        )

    def _band_width(self, window: int | None) -> int:
        # Match ``_route_static``'s effective W = min(window, C-1); None -> full (C-1).
        if window is None:
            return self.C - 1
        return max(1, min(int(window), self.C - 1))

    def update_band(self, sel: Tensor, slot_valid: Tensor, window: int | None) -> BlockMask:
        """Rewrite the routed band + block counts in place; return the cached BlockMask."""
        C = self.C
        S = sel.shape[1]
        W = self._band_width(window)
        # Forward (KV-major): the routed key chunks per query, left-compacted by topk.
        self.full_kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
        self.full_kv_num[0, 0, :] = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32)
        # Backward (Q-major) transpose built directly from the routing band in O(C*W):
        # band[j, r-1] is set iff key chunk j is selected by query chunk j+r (r in 1..W).
        # Vectorized over all past slots at once: scatter_reduce_ with amax is order-
        # independent, so flattening the [C, S-1] (query, slot) grid into one scatter is
        # equivalent to the per-slot loop.
        ci = self._arangeC
        j = sel[:, : S - 1].to(torch.long)                          # [C, S-1] key chunk
        col = ci[:, None] - j - 1                                   # [C, S-1] offset r-1
        # amax (logical OR), NOT overwrite: ``_route_static``'s clamp(min=0) collapses
        # several out-of-range past slots of one query onto key 0 at the same offset; an
        # invalid duplicate must not clobber a valid edge, so reduce by max.
        ok = (slot_valid[:, : S - 1] & (col >= 0) & (col < W)).to(torch.int32)  # [C, S-1]
        band = torch.zeros(C * W, dtype=torch.int32, device=self.device)
        band.scatter_reduce_(
            0, (j * W + col.clamp(0, W - 1)).reshape(-1), ok.reshape(-1),
            reduce="amax", include_self=True,
        )
        band_k = band.view(C, W) > 0
        self.full_q_num[0, 0, :] = band_k.sum(dim=-1, dtype=torch.int32)
        # Left-compact each key row (stable argsort keeps queries increasing, matching
        # ``_dense_to_ordered``): the query chunk index for key j at rank r is j + (r + 1).
        order = torch.argsort(band_k.to(torch.int32), dim=1, descending=True, stable=True)
        self.full_q_idx[0, 0, :, :W] = (ci[:, None] + order + 1).clamp(max=C - 1).to(torch.int32)
        return self.mask


def _get_flex_mask_state(cache, C, T, chunk_size, device) -> _RoutedFlexMaskState:
    """Fetch-or-create the persistent mask state for a given shape (lazy, per shape)."""
    key = (int(C), int(T), int(chunk_size), str(device))
    st = cache.get(key)
    if st is None:
        st = _RoutedFlexMaskState(C, T, chunk_size, device)
        cache[key] = st
    return st


def _flex_attend(q: Tensor, k: Tensor, v: Tensor, sel: Tensor, slot_valid: Tensor,
                 chunk_size: int, softmax_scale: float,
                 window: int | None, state: _RoutedFlexMaskState) -> Tensor:
    """Block-sparse attend over the routed selection via FlexAttention (union mode).

    q/k/v: [T, H, D]. sel/slot_valid: [C, S] with the own chunk in the LAST slot and
    valid routed chunks left-compacted (guaranteed by ``_route_static``'s topk).
    Routed chunks are strictly past -> full blocks (mask_mod never evaluated there); the
    own chunk is the diagonal partial block with an intra-chunk causal mask.

    ``state`` (a held ``_RoutedFlexMaskState``) owns the persistent BlockMask index buffers
    and mutates only the routed band in place, replacing ``from_kv_blocks``'s per-step
    O(C^2 log C) transpose with an O(C*W) direct build. ``window`` bounds the band width.
    Returns y: [T, H, D].
    """
    CH = int(chunk_size)

    mask = state.update_band(sel, slot_valid, window)

    qf = q.transpose(0, 1)[None]                                 # [1, H, T, D]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    # Forward kernel tiles must divide the 64-token mask blocks (the default 128x128 tile
    # fails to lower). fwd_ prefix keeps them away from the backward template, whose tiles
    # come from the (patched, see above) sm90 config list.
   
    y = flex_attention(
        qf, kf, vf, block_mask=mask, scale=float(softmax_scale),
        kernel_options={"fwd_BLOCK_M": CH,
                        "fwd_BLOCK_N": CH,
                        "fwd_num_warps": 4,
                        "fwd_num_stages": 3,
                        "ROWS_GUARANTEED_SAFE": True,
                        "BLOCKS_ARE_CONTIGUOUS": True,
                       },
    )
    
    return y[0].transpose(0, 1)                                  # [T, H, D]


def chunk_routed_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = 64,
    topk: int = 3,
    softmax_scale: float,
    window: int | None = None,
    mask_state: _RoutedFlexMaskState,
) -> Tensor:
    """Static-shape routed sparse attention: torch routing + FlexAttention attend.

    Requires the fully packed buffer (T % chunk_size == 0, no trailing pad) -- the stage-3
    training case. Fully traceable (no custom ops, no FA3): safe under
    ``torch.compile(fullgraph=True)``. Routing runs under ``no_grad`` (pure selection); the
    attend is differentiated by flex's fused backward, and the routed K/V receive the exact
    gradient (union mode). NO document masking -- routing spans the whole packed stream.

    ``mask_state`` is the held ``_RoutedFlexMaskState`` (owned by the attention module)
    whose persistent BlockMask buffers are updated only along the routed band each step.
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    assert T % CH == 0, "flex path requires a fully packed buffer"
    win = (D - 1) if window is None else max(0, min(int(window), D - 1))
   
    with torch.no_grad():
        sel, slot_valid = _route_static(
            q.detach(), k.detach(), CH, topk, softmax_scale, win,
        )
   
    return _flex_attend(q, k, v, sel, slot_valid, CH, softmax_scale, win, mask_state)


class CausalSelfAttentionChunkSDPA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        paired: bool = False,
        *,
        route_chunk_size: int = 64,
        route_topk: int = 3,
        route_window: int = 0,
        **_unused,
    ):
        super().__init__()
        assert not paired, "Chunk-routed attention is only for the non-paired final layer."
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = False
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"

        self.route_chunk_size = int(route_chunk_size)
        self.route_topk = int(route_topk)
        # Routing window (chunks); 0 -> auto (head_dim-1). Bounds the routing span.
        self.route_window = int(route_window)
        # Persistent per-shape FlexAttention mask state: holds the BlockMask index buffers so
        # only the routed near-diagonal band is rewritten in place each step (no per-step
        # from_kv_blocks transpose). Keyed by (C, T, chunk_size, device); lazily populated.
        self._mask_states: dict = {}

    def forward(self, x: Tensor, attn_args, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        assert T % self.route_chunk_size == 0, "fully packed buffer required (T % chunk_size == 0)"

        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        yarn = attn_args.yarn

        q, k, v = F.linear(
            x,
            sa_lambdas[0] * qkvo_w[: self.dim * 3].type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = _qk_norm(q), _qk_norm(k)
        q, k = yarn.rotary(q), yarn.rotary(k)

        
        if key_offset:
            k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

        if aux_v is not None:
            v = v + aux_v.view_as(v)

        mask_state = _get_flex_mask_state(
            self._mask_states, T // self.route_chunk_size, T,
            self.route_chunk_size, x.device,
        )
        
        y = chunk_routed_flex_attention(
            q[0], k[0], v[0],
            chunk_size=self.route_chunk_size,
            topk=self.route_topk,
            softmax_scale=yarn.attn_scale,
            window=(self.route_window or None),
            mask_state=mask_state,
        ).unsqueeze(0)                                          # [1, T, H, D]

        # Thid part take 
        if attn_args.xsa_alpha is not None:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))
       
            
        return y
