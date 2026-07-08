
# Chunk-routed sparse attention for *varlen* packed sequences (FlashAttention-3 routing).
#
# The nanoGPT speedrun feeds attention a single packed sequence [1, T, H, D] (B == 1)
# that concatenates many documents; document boundaries are given by a token-level
# ``cu_seqlens`` tensor. T can be huge (hundreds of thousands of tokens), so the old
# implementation -- which reshaped into fixed-length windows and scored every chunk
# against every other chunk with a dense [C, C] einsum -- is quadratic in the number
# of chunks and does not respect document boundaries.
#
# New design (all routing is *per document*, so it is block-diagonal, not N^2):
#   * Split each document independently into chunks of ``chunk_size`` (=64) tokens.
#     Chunks never straddle a document boundary (each document is padded up to a whole
#     number of chunks), so a chunk belongs to exactly one document.
#   * Mean-pool q and k over each chunk -> per-chunk summaries q_sum / k_sum.
#   * ROUTING: score query-chunk vs key-chunk *within the same document, causally*.
#       - FA3 path: run ``flash_attn_varlen_func`` over the sequence of chunk summaries
#         with ``cu_seqlens`` recomputed in CHUNK units and V a length-1 vector of 1.0.
#         With ``return_attn_probs=True`` the returned attention-weight matrix IS the
#         chunk-level routing distribution; V=1.0 makes the (discarded) output free.
#       - Torch fallback (used on non-Hopper GPUs / for tests): the same per-document
#         causal chunk-vs-chunk softmax, computed directly.
#     Take the top-k strictly-past key chunks per query chunk.
#   * ATTENTION: each query chunk cross-attends (explicit matmul + softmax; faster than
#     SDPA for these tiny gathered windows) to the gathered K/V of its top-k past chunks
#     plus its OWN chunk. Past chunks are entirely before the query chunk (same document),
#     so they are fully visible; the own chunk uses an intra-chunk causal mask. Padding
#     tokens and cross-document keys are masked out, so documents never see each other and
#     causality holds at the token level.
#
# WARNING! Training-only. Routing pools within-chunk future tokens (q_sum), so there is a
# small routing-level causality leak (which key chunks are *selected* can depend on future
# tokens of the query chunk). Per project tests this does not hurt; the actual attention
# values are still strictly causal. Use dense attention for validation.

from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend

# Masked chunk attention backend.
#
# The number of chunks ``C`` (the attention batch dim) is data-dependent: it changes with
# every batch's document layout, so it is effectively different every training step.
#
# cuDNN's fused attention has the fastest sm90 bf16 backward here (~8ms vs the cutlass
# "efficient" backend's ~15ms) BUT it JIT-builds+caches an execution plan keyed on the
# problem shape, and a *new* shape costs ~2 SECONDS to plan. Because ``C`` varies every
# step, cuDNN re-plans almost every step -> catastrophic ~2s/step jitter (a few cached-shape
# steps run fast, the rest rebuild). The cutlass EFFICIENT backend picks precompiled kernels
# by heuristic (no per-shape JIT), so it is rock-steady across varying ``C`` -- for only
# ~10ms/step more than cuDNN's *cached* best. Stability wins: EFFICIENT is the default.
#
# cuDNN is opt-in via ROUTE_SDPA_CUDNN=1 and is ONLY appropriate when the chunk count is
# pinned across steps (e.g. fixed-length non-varlen sequences); otherwise it will thrash.
# FLASH is never usable -- it refuses a non-causal (arbitrary boolean) mask.
_USE_CUDNN_SDPA = os.environ.get("ROUTE_SDPA_CUDNN", "0") != "0"
# Steady backends: EFFICIENT (fused, no per-shape JIT), MATH as a last-resort fallback.
_SDPA_BACKENDS = (
    [SDPBackend.CUDNN_ATTENTION] if _USE_CUDNN_SDPA
    else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
)


def _sdpa_masked(Q_, K_, V_, mask, softmax_scale):
    """Masked SDPA over the routed chunks (see backend notes above)."""
    with sdpa_kernel(_SDPA_BACKENDS):
        return F.scaled_dot_product_attention(
            Q_, K_, V_, attn_mask=mask, scale=float(softmax_scale)
        )

def _qk_norm(x: Tensor) -> Tensor:
    # Match modded-nanogpt's norm(x): F.rms_norm(x, (x.size(-1),)).
    return F.rms_norm(x, (x.size(-1),))


# --------------------------------------------------------------------------------------
# FlashAttention-3 (Hopper only). Loaded defensively so this module stays importable on
# GPUs/machines without the kernel (e.g. Ampere dev boxes, where the flex/torch paths are
# used). ``fa3`` is ``None`` when the kernel (or the ``kernels`` package) is unavailable;
# every FA3 call site already falls back when ``fa3 is None``.
# --------------------------------------------------------------------------------------

try:
    from kernels import get_kernel

    fa3 = get_kernel("kernels-community/flash-attn3", version=1).flash_attn_interface
except Exception:  # noqa: BLE001 -- missing package, non-Hopper GPU, offline, etc.
    fa3 = None


@torch.no_grad()
def _doc_chunk_layout(cu_seqlens: Tensor, T: int, chunk_size: int, device):
    """Build the chunk layout from a token-level ``cu_seqlens``.

    Tokens are chunked *contiguously* over the whole packed sequence (NOT padded per
    document), so the chunk count is always ``ceil(n_real / chunk_size)`` -- independent of
    how many documents are packed. (The old per-document layout padded every document up to
    a whole chunk, so ``C`` grew with the document count and made the step time depend on
    the batch's document layout.) A chunk may therefore straddle a document boundary.
    Because documents are guaranteed to be at least ``chunk_size`` tokens, a chunk spans at
    most two documents.

    ROUTING stays document-isolated: each chunk is assigned to a single document for the
    same-document routing mask. A straddling chunk is assigned to the *later* document (the
    document of its last token). Consequently a document may route to the one boundary chunk
    it shares with its predecessor -- seeing at most the ``< chunk_size``-token tail of the
    previous document -- but it never routes to earlier, fully-foreign chunks. That small,
    strictly-past overlap is the intended optimization; it never violates causality.

    ``cu_seqlens`` (``[0, e1, ..., total, ...]``, possibly trailing-padded) gives the real
    document boundaries; trailing padding tokens in ``q`` (beyond the last document) are
    excluded.

    Returns a dict with:
      lengths       [ND]        token length of each real document
      doc_of_chunk  [C]         document id of each chunk (by its LAST token)
      chunk_in_doc  [C]         chunk index within its assigned document (0-based)
      chunk_counts  [ND]        number of chunks assigned to each document
      cu_chunks     [ND+1]      chunk-level cu_seqlens (prefix sum of chunk_counts)
      tok_index     [C, CH]     source token index for every (chunk, within-chunk pos)
      valid_tok     [C, CH]     True where the (chunk, pos) maps to a real (non-pad) token
    """
    CH = int(chunk_size)
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    seg = cu[1:] - cu[:-1]
    lengths = seg[seg > 0]                                        # real document lengths [ND]
    total = int(lengths.sum().item())
    if total <= 0:
        total = int(T)
        lengths = torch.tensor([total], device=device, dtype=torch.long)
    ND = lengths.numel()

    C = (total + CH - 1) // CH

    # Global contiguous token layout: chunk c owns tokens [c*CH, (c+1)*CH). No per-document
    # padding -> C == ceil(total / CH), independent of the document count.
    ar_ch = torch.arange(CH, device=device)
    tok_index = torch.arange(C, device=device)[:, None] * CH + ar_ch[None, :]  # [C, CH]
    valid_tok = tok_index < total                                             # [C, CH]
    tok_index = torch.where(valid_tok, tok_index, torch.zeros_like(tok_index))

    # Per-chunk document id (by LAST token) -> keeps routing document-isolated. A straddling
    # chunk is owned by the later document, so only that overlap tail can leak (strictly past).
    doc_ends = torch.cumsum(lengths, 0)                                        # [ND] exclusive ends
    last_tok = torch.clamp(
        torch.arange(C, device=device) * CH + (CH - 1), max=total - 1,
    )                                                                          # [C]
    doc_of_chunk = torch.searchsorted(doc_ends, last_tok, right=True).clamp(max=ND - 1)  # [C] non-decreasing

    chunk_counts = torch.bincount(doc_of_chunk, minlength=ND)                  # [ND]
    cu_chunks = torch.empty(ND + 1, device=device, dtype=torch.int32)
    cu_chunks[0] = 0
    cu_chunks[1:] = torch.cumsum(chunk_counts, 0)
    chunk_start = torch.cumsum(chunk_counts, 0) - chunk_counts                 # [ND] first chunk of each doc
    chunk_in_doc = torch.arange(C, device=device) - chunk_start[doc_of_chunk]  # [C]

    return {
        "lengths": lengths,
        "doc_of_chunk": doc_of_chunk,
        "chunk_in_doc": chunk_in_doc,
        "chunk_counts": chunk_counts,
        "cu_chunks": cu_chunks,
        "tok_index": tok_index,
        "valid_tok": valid_tok,
        "C": C,
        "ND": ND,
    }


def _build_onehot_v(chunk_in_doc: Tensor, H: int, D: int, dtype, device) -> Tensor:
    """One-hot positional value for the routing attention: v[c, h, :] = e_{pos(c) % D}.

    ``pos(c)`` is the within-document chunk index. The same one-hot is used on every head,
    so the flattened hidden_dim value has a single 1 repeated once per head_dim block.
    Returns [C, H, D].
    """
    C = chunk_in_doc.shape[0]
    pos = (chunk_in_doc.to(torch.long) % D)                     # [C]
    v = torch.zeros(C, D, dtype=dtype, device=device)
    v.scatter_(1, pos[:, None], 1)                              # one-hot [C, D]
    return v[:, None, :].expand(C, H, D).contiguous()          # [C, H, D]


def _decode_onehot_route(out_chd: Tensor, layout: dict, topk: int, window: int):
    """Recover (sel, slot_valid) from a one-hot-positional-V attention output.

    out_chd: [C, H, D], where out[c, h, d] is the attention weight from query chunk c to
    the (unique, thanks to the window) key chunk whose within-doc position == d (mod D).
    We average over heads, then for each query chunk read off the weights of its <= window
    strictly-past chunks (their one-hot slots) and take the top-k.
    """
    C, H, D = out_chd.shape
    dev = out_chd.device
    chunk_in_doc = layout["chunk_in_doc"].to(torch.long)        # [C]
    W = max(0, min(int(window), D - 1))

    cidx = torch.arange(C, device=dev)
    r = torch.arange(1, W + 1, device=dev)                      # [W] strictly-past chunk offsets
    key_pos = chunk_in_doc[:, None] - r[None, :]                # [C, W] candidate within-doc position
    valid = key_pos >= 0                                        # still inside the same document
    residue = key_pos.clamp(min=0) % D                          # [C, W] its one-hot output slot

    # Gather only the W candidate slots per query chunk *before* reducing over heads, then
    # average. This reduces over C*H*W instead of C*H*D and never materializes a full
    # [C,H,D] fp32 copy of the (bf16) FA3 output. Since W <= D-1 this is strictly less work
    # (and much less when the window is small, e.g. W=26 vs D=64).
    idx = residue.unsqueeze(1).expand(C, H, W)                  # [C, H, W]
    score_r = torch.gather(out_chd, 2, idx).float().mean(dim=1)  # [C, W] head-averaged weights
    score_r = score_r.masked_fill(~valid, float("-inf"))
    key_global = cidx[:, None] - r[None, :]                     # [C, W] global chunk index (same doc)

    topk_eff = min(int(topk), W)
    if topk_eff > 0:
        topv, topj = score_r.topk(topk_eff, dim=-1)
        valid_past = topv > -1e30
        topi = torch.gather(key_global, 1, topj).clamp(min=0)   # clamp guards masked (-1) slots
    else:
        topi = torch.empty(C, 0, dtype=torch.long, device=dev)
        valid_past = torch.empty(C, 0, dtype=torch.bool, device=dev)

    own = cidx.reshape(C, 1)                                    # own chunk index == c
    sel = torch.cat([topi, own], dim=-1)                       # [C, S]
    slot_valid = torch.cat([valid_past, torch.ones(C, 1, dtype=torch.bool, device=dev)], dim=-1)
    return sel, slot_valid


def _onehot_out_torch(q_sum: Tensor, k_sum: Tensor, layout: dict, softmax_scale: float,
                      window: int) -> Tensor:
    """Torch simulation of the FA3 one-hot-V routing OUTPUT (for testing off Hopper).

    Reproduces exactly what ``flash_attn_varlen_func`` would return for the routing pass:
    a per-document, windowed, causal softmax over chunk summaries applied to the one-hot
    positional V. Returns [C, H, D]. Feed the result to ``_decode_onehot_route``.
    """
    C, H, D = q_sum.shape
    dev = q_sum.device
    doc_of_chunk = layout["doc_of_chunk"]
    chunk_in_doc = layout["chunk_in_doc"].to(torch.long)
    W = max(0, min(int(window), D - 1))

    scores = torch.einsum("ihd,jhd->hij", q_sum.float(), k_sum.float()) * float(softmax_scale)  # [H,C,C]
    ii = torch.arange(C, device=dev)
    same_doc = doc_of_chunk[:, None] == doc_of_chunk[None, :]
    causal = ii[None, :] <= ii[:, None]                         # key <= query
    within_win = (ii[:, None] - ii[None, :]) <= W               # key within W past of query
    valid = (same_doc & causal & within_win)[None]              # [1,C,C] (includes the diagonal)
    scores = scores.masked_fill(~valid, float("-inf"))
    attn = torch.softmax(scores, dim=-1)                        # [H, C, C]
    v = _build_onehot_v(chunk_in_doc, H, D, torch.float32, dev)  # [C, H, D]
    return torch.einsum("hck,khd->chd", attn, v)                # [C, H, D]


def _route_onehot_sim(q_sum: Tensor, k_sum: Tensor, layout: dict, softmax_scale: float,
                      topk: int, window: int):
    """Testable (non-Hopper) path: simulate the FA3 one-hot output, then decode it."""
    out = _onehot_out_torch(q_sum, k_sum, layout, softmax_scale, window)
    return _decode_onehot_route(out, layout, topk, window)


def _route_flash(q_sum: Tensor, k_sum: Tensor, layout: dict, softmax_scale: float, topk: int,
                 window: int):
    """FlashAttention-3 chunk routing via a one-hot positional-V decode (Hopper only).

    ``return_attn_probs`` is not exposed by the current FA3 build, so the routing weights
    are recovered indirectly. We run windowed causal attention over the chunk summaries
    with V set to a one-hot vector encoding each key chunk's within-document position
    (``pos % head_dim``, replicated across heads). Since a query attends to at most
    ``window <= head_dim`` past chunks (plus itself), those one-hot slots never collide,
    so the attention OUTPUT is a length-head_dim vector whose d-th entry is the attention
    weight to the unique window chunk at position ``d``. Top-k over that vector -> the
    top-k routed chunks. Falls back to the torch router if FA3 is unavailable.
    """

    doc_of_chunk = layout["doc_of_chunk"]
    if fa3 is None:
        return _get_route_torch()(q_sum, k_sum, doc_of_chunk, topk)

    C, H, D = q_sum.shape
    dev = q_sum.device
    cu_chunks = layout["cu_chunks"]
    max_chunks = int(layout["chunk_counts"].max().item())
    W = max(0, min(int(window), D - 1))                        # window <= head_dim-1 (collision-free)

    q_r = q_sum.to(torch.bfloat16).contiguous()               # [C, H, D]  (routing "tokens" = chunks)
    k_r = k_sum.to(torch.bfloat16).contiguous()
    v_r = _build_onehot_v(layout["chunk_in_doc"], H, D, torch.bfloat16, dev)  # [C, H, D] one-hot pos

    out = fa3.flash_attn_varlen_func(
        q_r, k_r, v_r,
        cu_seqlens_q=cu_chunks, cu_seqlens_k=cu_chunks,
        max_seqlen_q=max_chunks, max_seqlen_k=max_chunks,
        causal=True, softmax_scale=float(softmax_scale),
        window_size=(W, 0),
    )
    out = out[0] if isinstance(out, (tuple, list)) else out    # [C, H, D] attention output
    if out is None or out.dim() != 3:
        return _get_route_torch()(q_sum, k_sum, doc_of_chunk, topk)
    return _decode_onehot_route(out, layout, topk, W)


# --------------------------------------------------------------------------------------
# STATIC (fully traceable) path: routing + FlexAttention block-sparse attend.
#
# In the speedrun's stage-3 the loader always fills the whole packed buffer, so the token
# count ``T`` is fixed (49152) and every token is real -> the chunk count ``C = T // 64``
# is STATIC across steps. That removes every data-dependent shape from the routed
# attention: the routing becomes a plain per-head windowed causal softmax over the C
# chunk summaries (a [H, C, C] einsum -- ~1 GFLOP at C=768, cheaper than launching FA3)
# and the attend becomes a block-sparse FlexAttention call whose BlockMask is built from
# the topk selection with ``BlockMask.from_kv_blocks`` (routed past chunks are FULL
# 64x64 blocks skipping mask evaluation entirely; the own chunk is a partial causal
# block). FlexAttention reads the selected K/V blocks *in-kernel*: no gathered [C,S,CH,H,D]
# copies, no [C,1,CH,Ktot] boolean mask, no multi-GB transients, and a fused backward.
# Everything here is static-shape tensor ops, so it traces under
# ``torch.compile(fullgraph=True)`` with NO custom ops (flex_attention is a HOP).
# --------------------------------------------------------------------------------------

from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

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


def _route_static(q: Tensor, k: Tensor, cu_seqlens: Tensor, chunk_size: int,
                  topk: int, softmax_scale: float, window: int):
    """Static-shape chunk routing: per-head windowed causal chunk-vs-chunk scoring over
    chunk mean-summaries (restricted to the same document), head-averaged, then top-k of
    the strictly-past window.

    Routing is only a top-k SELECTION, so the per-row softmax the FA3 router applies is
    unnecessary here: top-k is invariant to softmax's monotonic per-row normalisation, and
    the masking is done BEFORE the top-k. We therefore score, head-average the RAW scores,
    mask out-of-window/cross-document candidates with -inf, and top-k directly -- one fewer
    softmax and no ``weight >= 0`` sentinel assumption. (For the RMS-normed, 1/sqrt(D)-scaled
    inputs used here, head-averaged softmax and head-averaged raw score pick the same set on
    ~98% of query chunks; see experiments/test_softmax_removal.py.)

    Requires T % chunk_size == 0 with every token real (fully packed buffer).
    Returns (sel [C, topk+1] long, slot_valid [C, topk+1] bool); invalid slots are
    left-compacted out by topk (a real score always beats the -inf sentinel).
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    C = T // CH
    dev = q.device

    q_sum = q.view(C, CH, H, D).float().mean(dim=1)              # [C, H, D]
    k_sum = k.view(C, CH, H, D).float().mean(dim=1)

    # Document id of each chunk (by its LAST token) straight from the token-level
    # cu_seqlens. Trailing padding entries in cu_seqlens equal T, so every last_tok < T
    # lands in a real document; consistent ids are all that matters for same-doc masking.
    cidx = torch.arange(C, device=dev)
    last_tok = cidx * CH + (CH - 1)
    # Equivalent to searchsorted(doc_ends, last_tok, right=True) but with plain
    # elementwise ops ([C, ND_pad] = 768x128 -- trivial); inductor cannot lower
    # searchsorted on a sliced (storage-offset) boundaries tensor.
    doc_ends = cu_seqlens.to(torch.long)[1:]                     # [ND_pad]
    doc_of_chunk = (last_tok[:, None] >= doc_ends[None, :]).sum(dim=-1)  # [C]

    # BANDED scores: a query chunk only ever routes within its trailing ``window`` chunks,
    # so score just the [C, W+1] band (offsets r = W..0, r==0 the own chunk) instead of a
    # dense [H, C, C] matrix -- that dense form is ~2 GB of fp32 transients at T=393K
    # (C=6144) while the band is a few MB at any T. Column j holds offset r = W - j, so
    # j == W is the own chunk and key chunk index = c - W + j.
    W = max(0, min(int(window), C - 1))
    kpad = F.pad(k_sum.reshape(C, H * D), (0, 0, W, 0))          # [W+C, H*D] front zero-pad
    kb = kpad.unfold(0, W + 1, 1).view(C, H, D, W + 1)           # [C, H, D, W+1] = k_sum[c-W+j]
    scores = torch.einsum("chd,chdj->hcj", q_sum * float(softmax_scale), kb)  # [H, C, W+1]

    dpad = F.pad(doc_of_chunk + 1, (W, 0))                       # 0 = out-of-range marker
    d_band = dpad.unfold(0, W + 1, 1)                            # [C, W+1] doc id of key chunk
    allowed = d_band == (doc_of_chunk + 1)[:, None]              # same doc & in range [C, W+1]

    # NO SOFTMAX: head-average the raw scores, then mask + top-k. Top-k is invariant to the
    # softmax's monotonic per-row normalisation, so it is redundant for a pure selection.
    # Mask the strictly-past band with -inf BEFORE the top-k (masking-before-softmax the
    # user asked for): the -inf sentinel can never be picked and never beats a real -- and
    # possibly negative -- score, replacing the old ``weight >= 0`` sentinel.
    w = scores.mean(dim=0)                                       # [C, W+1] head-averaged raw
    # Strictly-past candidates only (drop j == W, the own chunk).
    w = w[:, :W].masked_fill(~allowed[:, :W], float("-inf"))     # [C, W]
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
    per-step dense transpose (``_transpose_ordered``: build a dense ``[1,1,C,C]`` mask,
    transpose it, argsort over ``C`` -> O(C^2 log C)) and its two per-step ``[1,1,C,C]``
    allocations. The own-chunk diagonal is static (depends only on ``C``); the routed band is
    updated in place from ``sel``. The Q-major (backward) side is built directly from the
    routing band in O(C*W), never from a dense transpose.

    One instance per ``(C, T, chunk_size, device)`` -- see ``chunk_routed_flex_attention``'s
    ``mask_state`` cache. Allocating the buffers on the first (compiled) forward and mutating
    them in place on later steps is torch.compile(fullgraph=True)-safe: only the band columns
    are written, and the ``num_blocks`` tensors bound how many indices are read, so stale
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
        # Empty partial side (detached routed-only mask has no diagonal): 0 blocks read.
        self.zero_num = torch.zeros(1, 1, C, dtype=torch.int32, device=device)
        # Routed band buffers, rewritten in place each step. Only the near-diagonal columns
        # are touched; the remainder stays 0 and is never read (bounded by the num tensors).
        self.full_kv_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=device)
        self.full_q_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=device)

    def _band_width(self, window: int | None) -> int:
        # Match ``_route_static``'s effective W = min(window, C-1); None -> full (C-1).
        if window is None:
            return self.C - 1
        return max(1, min(int(window), self.C - 1))

    def update_band(self, sel: Tensor, slot_valid: Tensor, window: int | None):
        """Rewrite the routed band in place; return ``(full_kv_num, full_q_num)`` [1,1,C]."""
        C = self.C
        S = sel.shape[1]
        W = self._band_width(window)
        # Forward (KV-major): the routed key chunks per query, left-compacted by topk.
        self.full_kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
        full_kv_num = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32).view(1, 1, C)
        # Backward (Q-major) transpose built directly from the routing band in O(C*W):
        # band[j, r-1] is set iff key chunk j is selected by query chunk j+r (r in 1..W).
        ci = self._arangeC
        band = torch.zeros(C * W, dtype=torch.int32, device=self.device)
        for slot in range(S - 1):
            j = sel[:, slot].to(torch.long)
            col = ci - j - 1
            ok = (slot_valid[:, slot] & (col >= 0) & (col < W)).to(torch.int32)
            # amax (logical OR), NOT overwrite: ``_route_static``'s clamp(min=0) collapses
            # several out-of-range past slots of one query onto key 0 at the same offset; an
            # invalid duplicate must not clobber a valid edge, so reduce by max.
            band.scatter_reduce_(
                0, j * W + col.clamp(0, W - 1), ok, reduce="amax", include_self=True,
            )
        band_k = band.view(C, W) > 0
        full_q_num = band_k.sum(dim=-1, dtype=torch.int32).view(1, 1, C)
        # Left-compact each key row (stable argsort keeps queries increasing, matching
        # ``_dense_to_ordered``): the query chunk index for key j at rank r is j + (r + 1).
        order = torch.argsort(band_k.to(torch.int32), dim=1, descending=True, stable=True)
        self.full_q_idx[0, 0, :, :W] = (ci[:, None] + order + 1).clamp(max=C - 1).to(torch.int32)
        return full_kv_num, full_q_num

    def union_mask(self, full_kv_num, full_q_num) -> BlockMask:
        """Own diagonal (partial) + routed band (full), as one direct-built BlockMask."""
        return BlockMask(
            seq_lengths=(self.T, self.T),
            kv_num_blocks=self.diag_num, kv_indices=self.diag_idx,
            full_kv_num_blocks=full_kv_num, full_kv_indices=self.full_kv_idx,
            q_num_blocks=self.diag_num, q_indices=self.diag_idx,
            full_q_num_blocks=full_q_num, full_q_indices=self.full_q_idx,
            BLOCK_SIZE=(self.CH, self.CH), mask_mod=_causal_block_mask_mod,
        )

    def own_mask(self) -> BlockMask:
        """Static own-chunk diagonal-only mask (detached path)."""
        return BlockMask(
            seq_lengths=(self.T, self.T),
            kv_num_blocks=self.diag_num, kv_indices=self.diag_idx,
            full_kv_num_blocks=None, full_kv_indices=None,
            q_num_blocks=self.diag_num, q_indices=self.diag_idx,
            full_q_num_blocks=None, full_q_indices=None,
            BLOCK_SIZE=(self.CH, self.CH), mask_mod=_causal_block_mask_mod,
        )

    def routed_mask(self, full_kv_num, full_q_num) -> BlockMask:
        """Routed-band-only mask (no own diagonal) for the detached path."""
        return BlockMask(
            seq_lengths=(self.T, self.T),
            kv_num_blocks=self.zero_num, kv_indices=self.diag_idx,
            full_kv_num_blocks=full_kv_num, full_kv_indices=self.full_kv_idx,
            q_num_blocks=self.zero_num, q_indices=self.diag_idx,
            full_q_num_blocks=full_q_num, full_q_indices=self.full_q_idx,
            BLOCK_SIZE=(self.CH, self.CH), mask_mod=_causal_block_mask_mod,
        )


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
                 window: int | None = None, state: _RoutedFlexMaskState | None = None) -> Tensor:
    """Block-sparse attend over the routed selection via FlexAttention.

    q/k/v: [T, H, D]. sel/slot_valid: [C, S] with the own chunk in the LAST slot and
    valid routed chunks left-compacted (guaranteed by ``_route_static``'s topk).
    Routed chunks are strictly past -> full blocks (mask_mod never evaluated there);
    the own chunk is the diagonal partial block with an intra-chunk causal mask. This
    reproduces ``_attend_chunks``'s masking exactly (it likewise doesn't doc-mask inside
    routed/own blocks; with a fully packed buffer there are no pad tokens either).

    ``state`` (a held ``_RoutedFlexMaskState``) owns the persistent BlockMask index buffers
    and mutates only the routed band in place, replacing ``from_kv_blocks``'s per-step
    O(C^2 log C) transpose with an O(C*W) direct build. ``window`` bounds the band width
    (defaults to C-1, which covers all edges; pass the routing window for the real saving).
    When ``state`` is None a transient one is built (keeps the old positional call working).
    Returns y: [T, H, D].
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    C, S = sel.shape
    if state is None:
        state = _RoutedFlexMaskState(C, T, CH, q.device)

    full_kv_num, full_q_num = state.update_band(sel, slot_valid, window)
    mask = state.union_mask(full_kv_num, full_q_num)

    qf = q.transpose(0, 1)[None]                                 # [1, H, T, D]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    # Forward kernel tiles must divide the 64-token mask blocks (the default 128x128
    # tile fails to lower). fwd_ prefix keeps them away from the backward template,
    # whose tiles come from the (patched, see above) sm90 config list.
    y = flex_attention(
        qf, kf, vf, block_mask=mask, scale=float(softmax_scale),
        kernel_options={"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH},
    )
    return y[0].transpose(0, 1)                                  # [T, H, D]


def _flex_attend_detached(q: Tensor, k: Tensor, v: Tensor, sel: Tensor, slot_valid: Tensor,
                          chunk_size: int, softmax_scale: float,
                          window: int | None = None,
                          state: _RoutedFlexMaskState | None = None) -> Tensor:
    """Detach-routed variant: two flex calls merged exactly by logsumexp.

    Reproduces the old path's ``detach_routed=True`` gradients: the routed (past) chunks'
    K/V receive NO gradient; only the own chunk does. Also balances the backward: the
    routed call's K/V need no grads, so its (per-KV-block, imbalance-prone) dK/dV pass is
    dead code, and the own-chunk call's dK/dV work is perfectly uniform (diagonal only).
    Forward output is numerically identical to the single-call union (same key set).

    Uses the held ``state``'s two masks: a static own-chunk diagonal mask and the routed
    band mask (updated in place). ``window``/``state`` behave as in ``_flex_attend``.
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    C, S = sel.shape
    if state is None:
        state = _RoutedFlexMaskState(C, T, CH, q.device)

    own_mask = state.own_mask()
    full_kv_num, full_q_num = state.update_band(sel, slot_valid, window)
    routed_mask = state.routed_mask(full_kv_num, full_q_num)

    qf = q.transpose(0, 1)[None]                                 # [1, H, T, D]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    opts = {"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH}
    # ``return_aux=AuxRequest(lse=True)`` is the non-deprecated LSE request (the old
    # ``return_lse=True`` emits a ``warnings.warn`` that breaks ``fullgraph=True`` tracing
    # on torch >= 2.9). AuxOutput.lse holds the per-query logsumexp used to merge the two
    # disjoint-key flex calls exactly.
    own_out, own_aux = flex_attention(
        qf, kf, vf, block_mask=own_mask, scale=float(softmax_scale),
        return_aux=AuxRequest(lse=True), kernel_options=opts,
    )
    route_out, route_aux = flex_attention(
        qf, kf.detach(), vf.detach(), block_mask=routed_mask, scale=float(softmax_scale),
        return_aux=AuxRequest(lse=True), kernel_options=opts,
    )
    own_lse, route_lse = own_aux.lse, route_aux.lse

    # Exact merge over disjoint key sets. Query rows with zero routed blocks come back
    # with lse == -inf / out == 0 -> their routed weight is exactly 0.
    route_ok = torch.isfinite(route_lse)
    route_lse = torch.where(route_ok, route_lse, torch.full_like(route_lse, float("-inf")))
    m = torch.maximum(own_lse, route_lse)
    wo = torch.exp(own_lse - m)
    wr = torch.exp(route_lse - m)
    den = wo + wr                                                # own row always finite
    y = (own_out.float() * wo[..., None] + route_out.float() * wr[..., None]) / den[..., None]
    return y.to(q.dtype)[0].transpose(0, 1)                     # [T, H, D]


def chunk_routed_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    *,
    chunk_size: int = 64,
    topk: int = 3,
    softmax_scale: float,
    window: int | None = None,
    detach_routed: bool = False,
    mask_state: _RoutedFlexMaskState | None = None,
) -> Tensor:
    """Static-shape routed sparse attention: torch routing + FlexAttention attend.

    Drop-in replacement for ``chunk_routed_varlen_attention`` when the packed buffer is
    full (T % chunk_size == 0, no trailing pad) -- the stage-3 training case. Fully
    traceable (no custom ops, no FA3): safe under ``torch.compile(fullgraph=True)``.
    Routing runs under no_grad (pure selection, matches the eager router's detached
    summaries); the attend is differentiated by flex's fused backward.

    ``mask_state`` is a held ``_RoutedFlexMaskState`` (owned by the attention module) whose
    persistent BlockMask buffers are updated only along the routed band each step. When None
    the attend functions build a transient state (correct but re-allocates per step).
    """
    T, H, D = q.shape
    CH = int(chunk_size)
    assert T % CH == 0, "flex path requires a fully packed buffer"
    win = (D - 1) if window is None else max(0, min(int(window), D - 1))
    with torch.no_grad():
        sel, slot_valid = _route_static(
            q.detach(), k.detach(), cu_seqlens, CH, topk, softmax_scale, win,
        )
    if detach_routed:
        return _flex_attend_detached(q, k, v, sel, slot_valid, CH, softmax_scale, win, mask_state)
    return _flex_attend(q, k, v, sel, slot_valid, CH, softmax_scale, win, mask_state)


def chunk_routed_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    *,
    chunk_size: int = 64,
    topk: int = 3,
    softmax_scale: float,
    detach_routed: bool = True,
    use_flash: bool = False,
    use_onehot_sim: bool = False,
    window: int | None = None,
    region=None,
) -> Tensor:
    """Chunk-routed sparse attention over a single packed (varlen) sequence.

    q/k/v: [T, H, D] (already RMS-normed and rotary-applied by the caller).
    cu_seqlens: token-level document boundaries, ``[0, e1, ..., T]`` (trailing-padding ok).
    ``window`` bounds the FA3 / one-hot routing span (chunks); defaults to head_dim-1.
    Returns y: [T, H, D].

    NOTE: the routing is inherently data-dependent in *shape* (number of chunks, top-k
    gathers, ``.item()`` sizes, boolean masking), so it cannot be traced into a static
    graph. Under ``torch.compile`` (including ``fullgraph=True``, where a graph break is a
    hard error) it is dispatched through TWO opaque custom ops:

      * ``hga::chunk_route`` (non-differentiable): runs the (FA3) router once and returns
        the ``sel`` / ``slot_valid`` selection tensors.
      * ``hga::chunk_attend`` (differentiable): consumes the *fixed* selection and runs the
        gathered local attention. Its backward reuses the saved ``sel`` / ``slot_valid``
        (tiny int/bool tensors) and re-differentiates ONLY the attention math -- it never
        re-runs the router. (The earlier single-op design recomputed the whole router in
        backward, doubling the FA3 routing cost.)

    In eager mode ``_chunk_routed_impl`` is called directly (routing runs under ``no_grad``,
    so eager autograd already backprops only the saved ``_attend_chunks`` graph and never
    recomputes routing); profiling ``region`` hooks still work there.
    """
    if torch.compiler.is_compiling():
        sel, slot_valid = _route_op(
            q, k, cu_seqlens, int(chunk_size), int(topk), float(softmax_scale),
            bool(use_flash), bool(use_onehot_sim), window,
        )
        return _chunk_attend_op(
            q, k, v, sel, slot_valid, cu_seqlens,
            int(chunk_size), float(softmax_scale), bool(detach_routed),
        )
    return _chunk_routed_impl(
        q, k, v, cu_seqlens,
        chunk_size=chunk_size, topk=topk, softmax_scale=softmax_scale,
        detach_routed=detach_routed, use_flash=use_flash,
        use_onehot_sim=use_onehot_sim, window=window, region=region,
    )


@torch.no_grad()
def _route_chunks(q: Tensor, k: Tensor, layout: dict, chunk_size: int, softmax_scale: float,
                  topk: int, use_flash: bool, use_onehot_sim: bool, window):
    """Non-differentiable chunk routing -> (sel, slot_valid). Uses detached chunk summaries
    only, so it is purely a *selection* (no gradient flows through it). Kept OUT of the
    autograd/functorch path: the FA3 routing kernel is an ``autograd.Function`` that does
    not support functorch transforms, so it must never run inside ``torch.func.vjp``.
    """
    C = layout["C"]
    CH = int(chunk_size)
    _, H, D = q.shape
    flat_idx = layout["tok_index"].reshape(-1)                 # [C*CH]
    valid_tok = layout["valid_tok"]                            # [C, CH]
    vmask = valid_tok[:, :, None, None]
    qc = q.index_select(0, flat_idx).reshape(C, CH, H, D) * vmask
    kc = k.index_select(0, flat_idx).reshape(C, CH, H, D) * vmask
    denom = valid_tok.sum(dim=1).clamp(min=1).to(torch.float32)[:, None, None]  # [C,1,1]
    q_sum = qc.float().sum(dim=1) / denom                      # [C, H, D]
    k_sum = kc.float().sum(dim=1) / denom
    win = (D - 1) if window is None else max(0, min(int(window), D - 1))
    if use_flash:
        return _route_flash(q_sum, k_sum, layout, softmax_scale, topk, win)
    if use_onehot_sim:
        return _route_onehot_sim(q_sum, k_sum, layout, softmax_scale, topk, win)
    return _get_route_torch()(q_sum, k_sum, layout["doc_of_chunk"], topk)


def _attend_chunks(q: Tensor, k: Tensor, v: Tensor, layout: dict, sel: Tensor,
                   slot_valid: Tensor, chunk_size: int, softmax_scale: float,
                   detach_routed: bool, region=None) -> Tensor:
    """Differentiable part: gather the routed chunks' Q/K/V and run the local attention.

    Given a *fixed* routing (``sel`` / ``slot_valid``), this contains only plain torch ops
    (index_select / matmul / softmax / index_copy), so it is safe to differentiate through
    ``torch.func.vjp``. Returns y: [T, H, D].
    """
    if region is None:
        region = lambda _name: contextlib.nullcontext()

    T, H, D = q.shape
    CH = int(chunk_size)
    dev = q.device
    C = layout["C"]
    flat_idx = layout["tok_index"].reshape(-1)                 # [C*CH]
    valid_tok = layout["valid_tok"]                            # [C, CH]

    # Gather tokens into padded per-document chunks: [C, CH, H, D].
    # No pad-zeroing here: padded KEY slots are excluded by the attention mask below, and
    # padded QUERY rows are dropped by the final scatter -- so the (full-size) ``* vmask``
    # elementwise multiplies the old code did on q/k/v were pure redundant memory traffic.
    with region("route.gather_chunks"):
        qc = q.index_select(0, flat_idx).reshape(C, CH, H, D)
        kc = k.index_select(0, flat_idx).reshape(C, CH, H, D)
        vc = v.index_select(0, flat_idx).reshape(C, CH, H, D)

    S = sel.size(1)
    Ktot = S * CH

    # Gather selected chunks' K/V. Detach the routed (past) slots -> cheap/low-memory
    # backward; only the OWN chunk (last slot) carries gradient through K/V.
    with region("route.gather_kv"):
        if detach_routed and S > 1:
            kc_d, vc_d = kc.detach(), vc.detach()
            K_sel = torch.cat([kc_d[sel[:, : S - 1]], kc[sel[:, S - 1:]]], dim=1)  # [C,S,CH,H,D]
            V_sel = torch.cat([vc_d[sel[:, : S - 1]], vc[sel[:, S - 1:]]], dim=1)
        else:
            K_sel = kc[sel]                                     # [C, S, CH, H, D]
            V_sel = vc[sel]

        Q_ = qc.permute(0, 2, 1, 3)                             # [C, H, CH, D]
        K_ = K_sel.reshape(C, Ktot, H, D).permute(0, 2, 1, 3)  # [C, H, Ktot, D]
        V_ = V_sel.reshape(C, Ktot, H, D).permute(0, 2, 1, 3)

    # Attention mask [C, 1, CH, Ktot].
    with region("route.mask"):
        t_ar = torch.arange(CH, device=dev)
        causal = t_ar[None, :] <= t_ar[:, None]                 # [CH(t), CH(kt)] kt <= t
        base = torch.ones(CH, S, CH, dtype=torch.bool, device=dev)
        base[:, S - 1, :] = causal                              # own slot (last) is intra-chunk causal
        base = base.reshape(CH, Ktot)                           # [CH, Ktot]

        # per selected slot: real (non-pad) key token AND slot is genuinely selected
        vsel = valid_tok[sel] & slot_valid[:, :, None]          # [C, S, CH]
        vsel = vsel.reshape(C, Ktot)                            # [C, Ktot]
        mask = base[None, None] & vsel[:, None, None, :]        # [C, 1, CH, Ktot]

    # Fused attention. SDPA (flash/efficient backend) fuses scores+mask+softmax+PV without
    # ever materializing the [C,H,CH,Ktot] score matrix / its fp32 softmax copy -- the old
    # explicit matmul->masked_fill->float-softmax->matmul path did, which was the dominant
    # cost (both time and ~5GB transient). Fully-masked (padded) query rows come out NaN;
    # nan_to_num zeroes them (and their backward grad) before the scatter drops them anyway.
    with region("route.attn"):
        y = _sdpa_masked(Q_, K_, V_, mask, softmax_scale)       # [C, H, CH, D]
        y = torch.nan_to_num(y)
        y = y.permute(0, 2, 1, 3)                               # [C, CH, H, D]

    # Scatter chunk outputs back to the flat token layout. Each real token maps to exactly
    # one (chunk, pos) slot, so index_copy is unambiguous; pad slots are dropped.
    with region("route.scatter"):
        out = torch.zeros(T, H, D, dtype=y.dtype, device=dev)
        vflat = valid_tok.reshape(-1)
        out.index_copy_(0, flat_idx[vflat], y.reshape(C * CH, H, D)[vflat])
    return out


def _chunk_routed_impl(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    *,
    chunk_size: int = 64,
    topk: int = 8,
    softmax_scale: float,
    detach_routed: bool = True,
    use_flash: bool = False,
    use_onehot_sim: bool = False,
    window: int | None = None,
    region=None,
) -> Tensor:
    if region is None:
        region = lambda _name: contextlib.nullcontext()

    T = q.shape[0]
    CH = int(chunk_size)

    with region("route.layout"):
        L = _doc_chunk_layout(cu_seqlens, T, CH, q.device)
    with region("route.select"):
        sel, slot_valid = _route_chunks(
            q, k, L, CH, softmax_scale, topk, use_flash, use_onehot_sim, window,
        )
    return _attend_chunks(q, k, v, L, sel, slot_valid, CH, softmax_scale, detach_routed, region)


# --------------------------------------------------------------------------------------
# Opaque custom ops for torch.compile. The routing body is data-dependent in *shape*
# (number of chunks ``C``, top-k gathers, ``.item()`` sizes, boolean masking), so dynamo
# cannot trace it -- and under ``fullgraph=True`` a graph break is a hard error. We split
# the work into TWO opaque ops so that backward never re-runs the (expensive, FA3) router:
#
#   1. ``hga::chunk_route``  (non-differentiable): layout + router -> (sel, slot_valid).
#      Outputs are int/bool selection tensors, so no gradient flows through it; the router
#      contributes no grad (it is a pure selection). Its ``C`` dim is data-dependent, so the
#      fake kernel allocates an unbacked dynamic size.
#   2. ``hga::chunk_attend`` (differentiable): layout + gathered local attention over the
#      *fixed* selection. Its backward reuses the saved ``sel`` / ``slot_valid`` (tiny
#      [C, topk+1] int/bool tensors) and re-differentiates ONLY ``_attend_chunks`` via
#      ``torch.func.vjp`` -- it recomputes just the cheap token layout (arange/searchsorted,
#      no FA3), never the router.
#
# ``torch.func.vjp`` is used (instead of ``torch.autograd.grad``) because a custom-op kernel
# runs with the Autograd key excluded, so a normal backward graph would not be recorded; the
# functorch grad transform works regardless. The router must never run inside vjp anyway (the
# FA3 routing kernel is an ``autograd.Function`` incompatible with functorch) -- keeping it in
# a separate op guarantees that.
# --------------------------------------------------------------------------------------
@torch.library.custom_op("hga::chunk_route", mutates_args=())
def _route_op(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
    topk: int,
    softmax_scale: float,
    use_flash: bool,
    use_onehot_sim: bool,
    window: Optional[int],
) -> tuple[Tensor, Tensor]:
    layout = _doc_chunk_layout(cu_seqlens, q.shape[0], int(chunk_size), q.device)
    sel, slot_valid = _route_chunks(
        q, k, layout, int(chunk_size), softmax_scale, topk, use_flash, use_onehot_sim, window,
    )
    return sel.contiguous(), slot_valid.contiguous()


@_route_op.register_fake
def _(q, k, cu_seqlens, chunk_size, topk, softmax_scale, use_flash, use_onehot_sim, window):
    # ``C`` (number of chunks) is data-dependent -> unbacked dynamic size. ``S == topk + 1``
    # (the top-k past chunks plus the own chunk); this holds whenever ``topk <= window``,
    # which the caller always satisfies (window is 22-63 chunks, topk is a few).
    ctx = torch.library.get_ctx()
    C = ctx.new_dynamic_size()
    S = int(topk) + 1
    sel = q.new_empty(C, S, dtype=torch.long)
    slot_valid = q.new_empty(C, S, dtype=torch.bool)
    return sel, slot_valid


@torch.library.custom_op("hga::chunk_attend", mutates_args=())
def _chunk_attend_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel: Tensor,
    slot_valid: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
    softmax_scale: float,
    detach_routed: bool,
) -> Tensor:
    layout = _doc_chunk_layout(cu_seqlens, q.shape[0], int(chunk_size), q.device)
    return _attend_chunks(
        q, k, v, layout, sel, slot_valid, int(chunk_size), softmax_scale, detach_routed,
    )


@_chunk_attend_op.register_fake
def _(q, k, v, sel, slot_valid, cu_seqlens, chunk_size, softmax_scale, detach_routed):
    return torch.empty_like(q)


# Backward is *also* a custom op: AOTAutograd traces the ``register_autograd`` backward to
# build the backward graph, so it must not run the data-dependent body under fake tensors.
# As its own opaque op (with a fake kernel) the recompute stays out of the traced graph.
@torch.library.custom_op("hga::chunk_attend_bwd", mutates_args=())
def _chunk_attend_bwd_op(
    grad_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel: Tensor,
    slot_valid: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
    softmax_scale: float,
    detach_routed: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    # Reuse the saved routing (``sel`` / ``slot_valid``) -- no router recompute. Only the
    # cheap token layout (arange / searchsorted, no FA3) is rebuilt, then ``_attend_chunks``
    # is differentiated.
    layout = _doc_chunk_layout(cu_seqlens, q.shape[0], int(chunk_size), q.device)

    def _fwd(qq, kk, vv):
        return _attend_chunks(
            qq, kk, vv, layout, sel, slot_valid, int(chunk_size), softmax_scale, detach_routed,
        )

    _, vjp_fn = torch.func.vjp(_fwd, q, k, v)
    gq, gk, gv = vjp_fn(grad_out)
    return gq, gk, gv


@_chunk_attend_bwd_op.register_fake
def _(grad_out, q, k, v, sel, slot_valid, cu_seqlens, chunk_size, softmax_scale, detach_routed):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _chunk_attend_setup(ctx, inputs, output):
    q, k, v, sel, slot_valid, cu_seqlens = inputs[:6]
    ctx.save_for_backward(q, k, v, sel, slot_valid, cu_seqlens)
    ctx.params = dict(
        chunk_size=inputs[6], softmax_scale=inputs[7], detach_routed=inputs[8],
    )


def _chunk_attend_backward(ctx, grad_out):
    q, k, v, sel, slot_valid, cu_seqlens = ctx.saved_tensors
    gq, gk, gv = _chunk_attend_bwd_op(
        grad_out.contiguous(), q, k, v, sel, slot_valid, cu_seqlens, **ctx.params
    )
    # Grads only for (q, k, v); sel/slot_valid/cu_seqlens + the 3 scalar args are non-diff.
    return gq, gk, gv, None, None, None, None, None, None


_chunk_attend_op.register_autograd(_chunk_attend_backward, setup_context=_chunk_attend_setup)


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
        use_matmul: bool = True,
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
        self.route_topk = int(os.environ.get("ROUTE_TOPK", route_topk))
        self.use_matmul = bool(use_matmul)
        # detach_routed=False (union) is the default optimal mode: on the flex path it is
        # both faster (fwd ~1.6x, fwd+bwd ~1.25x vs the two-call detached merge at C=4800)
        # and gives the *exact* gradient (routed K/V receive grad). Set DETACH_ROUTED=1 to
        # fall back to the stop-gradient path (cheaper backward only on the varlen path).
        self.detach_routed = os.environ.get("DETACH_ROUTED", "0") != "0"
        # Use FlashAttention-3 for routing (Hopper). Off by default; the torch router is
        # numerically the routing target and runs everywhere.
        self.use_flash_route = os.environ.get("ROUTE_FLASH", "1") != "0"
        # Torch simulation of the FA3 one-hot-V routing output (testable off Hopper).
        self.use_onehot_sim = os.environ.get("ROUTE_ONEHOT_SIM", "0") != "0"
        # Routing window (chunks); 0 -> auto (head_dim-1). Bounds the FA3 / one-hot span.
        self.route_window = int(os.environ.get("ROUTE_WINDOW", route_window))
        # Static flex path (torch routing + FlexAttention block-sparse attend). Default ON:
        # fastest, fully traceable, no custom ops. Requires the fully packed stage-3
        # buffer (T % chunk_size == 0). NOTE: gradient flows through routed K/V here
        # (no detach_routed) -- that is the *exact* gradient, richer than the detached one.
        self.use_flex_attend = os.environ.get("USE_FLEX_ATTEND", "1") != "0"
        # Wrap sub-steps in torch.profiler.record_function for per-region attribution.
        # Must stay False under torch.compile(fullgraph=True).
        self.profile = False
        # Persistent per-shape FlexAttention mask state: holds the BlockMask index buffers so
        # only the routed near-diagonal band is rewritten in place each step (no per-step
        # from_kv_blocks transpose). Keyed by (C, T, chunk_size, device); lazily populated.
        self._mask_states: dict = {}

    def _region(self, name: str):
        return torch.profiler.record_function(name) if self.profile else contextlib.nullcontext()

    def forward(self, x: Tensor, attn_args, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0

        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        yarn = attn_args.yarn

        # Document boundaries. Prefer the varlen cu_seqlens; otherwise synthesize
        # fixed-length windows from train_max_seq_len (legacy behaviour).
        seqlens = getattr(attn_args, "seqlens", None)
        if seqlens is None:
            L = int(attn_args.train_max_seq_len)
            assert L > 0 and T % L == 0, f"T={T} not divisible by train_max_seq_len={L}"
            seqlens = torch.arange(0, T + 1, L, device=x.device, dtype=torch.int32)

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

        if self.use_flex_attend and T % self.route_chunk_size == 0:
            mask_state = _get_flex_mask_state(
                self._mask_states, T // self.route_chunk_size, T,
                self.route_chunk_size, x.device,
            )
            y = chunk_routed_flex_attention(
                q[0], k[0], v[0], seqlens,
                chunk_size=self.route_chunk_size,
                topk=self.route_topk,
                softmax_scale=yarn.attn_scale,
                window=(self.route_window or None),
                detach_routed=self.detach_routed,
                mask_state=mask_state,
            ).unsqueeze(0)                                      # [1, T, H, D]
        else:
            y = chunk_routed_varlen_attention(
                q[0], k[0], v[0], seqlens,
                chunk_size=self.route_chunk_size,
                topk=self.route_topk,
                softmax_scale=yarn.attn_scale,
                detach_routed=self.detach_routed,
                use_flash=self.use_flash_route,
                use_onehot_sim=self.use_onehot_sim,
                window=(self.route_window or None),
                region=self._region,
            ).unsqueeze(0)                                      # [1, T, H, D]

        if attn_args.xsa_alpha is not None:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))
        return y
