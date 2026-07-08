# routed_flex_attention_nonpaired_v6.py
# Non-paired final-layer replacement for modded-nanogpt CausalSelfAttention.
# Route: chunk-sum Q/K summaries -> top-k KV chunks per Q chunk -> FlexAttention BlockMask.
# v6 fixes a critical causality leak: routed full blocks are only used for strictly past chunks;
# the current chunk is masked by q_idx >= kv_idx, so tokens cannot see future tokens.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    _HAS_FLEX = True
except Exception:  # pragma: no cover
    BlockMask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    _HAS_FLEX = False

try:
    import flash_attn_interface  # type: ignore
    _HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover
    flash_attn_interface = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False

_FLEX_COMPILED = None


def _get_flex_attention(compile_flex: bool):
    """Return raw or separately compiled flex_attention.

    If the whole GPT model is already wrapped in torch.compile, keep
    compile_flex=False so Dynamo/Inductor captures flex_attention in the model graph.
    """
    global _FLEX_COMPILED
    if not _HAS_FLEX:
        raise RuntimeError("torch.nn.attention.flex_attention is not available")
    if not compile_flex:
        return flex_attention
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


def _qk_norm(x: Tensor) -> Tensor:
    # Match modded-nanogpt norm(x): return F.rms_norm(x, (x.size(-1),))
    return F.rms_norm(x, (x.size(-1),))


def _float_scale(scale) -> Optional[float]:
    if scale is None:
        return None
    if isinstance(scale, (float, int)):
        return float(scale)
    if isinstance(scale, Tensor):
        return float(scale.detach().item())
    return float(scale)


def _causal_mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    # FlexAttention token-level causal mask for partial blocks.
    # This is applied to the current chunk; strictly past chunks are emitted as
    # full blocks and therefore do not pay mask_mod overhead.
    return q_idx >= kv_idx


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        paired: bool = False,
        *,
        route_chunk_size: int = 64,
        route_topk: int = 4,
        route_local_prev: int = 1,
        route_local_next: int = 0,
        route_respect_doc_boundaries: bool = True,
        # If True, top-k routing candidates are limited to current/past chunks.
        # If False, future chunks may affect routing selection, but the final
        # token-level FlexAttention mask still prevents future-token attention.
        route_topk_causal: bool = True,
        route_use_flex: bool = True,
        # Important for your setup: the outer model is already torch.compile'd.
        # Do not nest torch.compile(flex_attention) by default.
        route_compile_flex: bool = False,
        route_fallback_to_flash: bool = True,
    ):
        super().__init__()
        assert not paired, "This routed FlexAttention replacement is only for the non-paired final layer."
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = False
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"

        assert route_chunk_size > 0 and (route_chunk_size & (route_chunk_size - 1)) == 0
        assert route_topk >= 1
        self.route_chunk_size = route_chunk_size
        self.route_topk = route_topk
        self.route_local_prev = route_local_prev
        self.route_local_next = route_local_next
        self.route_respect_doc_boundaries = route_respect_doc_boundaries
        self.route_topk_causal = route_topk_causal
        self.route_use_flex = route_use_flex
        self.route_compile_flex = route_compile_flex
        self.route_fallback_to_flash = route_fallback_to_flash

        # RTX 5090 / Blackwell-friendly: use standard Triton FlexAttention backend,
        # avoid experimental FLASH backend and Hopper-only TMA options.
        self.flex_kernel_options = {
            "BACKEND": "TRITON",
            "BLOCK_M": route_chunk_size,
            "BLOCK_N": route_chunk_size,
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": False,
        }
        self._flex_attn = _get_flex_attention(route_compile_flex) if route_use_flex and _HAS_FLEX else None

    @torch.no_grad()
    def _build_routed_block_mask(self, q: Tensor, k: Tensor, seqlens: Optional[Tensor]) -> BlockMask:
        if BlockMask is None:
            raise RuntimeError("FlexAttention BlockMask is not available")

        # q/k: [1, T, H, D], already RMS-normed and rotary-applied.
        device = q.device
        T = q.size(1)
        H = self.num_heads
        D = self.head_dim
        chunk = self.route_chunk_size
        C = (T + chunk - 1) // chunk
        padded_T = C * chunk
        pad = padded_T - T

        q0 = q[0].detach()
        k0 = k[0].detach()
        if pad:
            q0 = F.pad(q0, (0, 0, 0, 0, 0, pad))
            k0 = F.pad(k0, (0, 0, 0, 0, 0, pad))

        # User-requested routing: sum Q/K by chunks, score q_sum @ k_sum, top-k chunks.
        # Shared route over heads: flatten H*D for a tiny [C,C] matmul.
        q_sum = q0.float().reshape(C, chunk, H, D).sum(dim=1).flatten(1)  # [C,H*D]
        k_sum = k0.float().reshape(C, chunk, H, D).sum(dim=1).flatten(1)  # [C,H*D]
        scores = q_sum @ k_sum.t()  # [qC,kvC]

        ar = torch.arange(C, device=device, dtype=torch.long)
        same_block_doc = None

        if self.route_respect_doc_boundaries and seqlens is not None:
            # Avoid `torch.searchsorted(cu[1:], ...)`: Inductor can lower that sliced
            # boundaries tensor as SliceView and fail. Using the full cu_seqlens gives
            # the same doc id with `right=True` and `-1`, with no sliced boundary tensor.
            cu = seqlens.to(device=device, dtype=torch.long, non_blocking=True)
            block_start = ar * chunk
            block_doc = torch.searchsorted(cu, block_start, right=True) - 1
            same_block_doc = block_doc[:, None] == block_doc[None, :]
            scores = scores.masked_fill(~same_block_doc, -torch.inf)

        # v6 safety: top-k may be chosen causally or non-causally, but the
        # *actual attention* BlockMask below is always block-causal + token-causal.
        # This avoids the v5 leak where selected future/current chunks were put in
        # full_kv_* and therefore bypassed any q_idx >= kv_idx mask.
        if self.route_topk_causal:
            block_causal = ar[:, None] >= ar[None, :]
            scores = scores.masked_fill(~block_causal, -torch.inf)

        topk = min(self.route_topk, C)
        topk_idx = torch.topk(scores, k=topk, dim=-1).indices

        route = torch.zeros((C, C), device=device, dtype=torch.bool)
        route.scatter_(1, topk_idx, True)

        # Always include current chunk and a small local band.
        route[ar, ar] = True
        for off in range(1, self.route_local_prev + 1):
            if off < C:
                route[ar[off:], ar[:-off]] = True
        for off in range(1, self.route_local_next + 1):
            if off < C:
                route[ar[:-off], ar[off:]] = True

        if same_block_doc is not None:
            route = route & same_block_doc
            route[ar, ar] = True

        # Actual attention must be causal at token level.
        # - Strictly past KV chunks are safe as full blocks: every KV token is
        #   before every Q token in the Q chunk.
        # - Current chunks must be partial blocks with mask_mod=q_idx>=kv_idx.
        # - Future chunks are dropped entirely, even if selected by non-causal routing.
        block_past = ar[:, None] > ar[None, :]
        block_current = ar[:, None] == ar[None, :]
        full_route = route & block_past
        partial_route = route & block_current

        # Convert bool block tables to the compact per-Q-block index format.
        # topk on int32 puts 1s before 0s and avoids argsort/slice patterns that
        # have caused Inductor failures on some nightly/stable combinations.
        partial_i32 = partial_route.to(torch.int32)
        full_i32 = full_route.to(torch.int32)

        kv_num_1h = partial_i32.sum(dim=-1).to(torch.int32)
        _, kv_idx_1h = torch.topk(partial_i32, k=C, dim=-1)
        kv_idx_1h = kv_idx_1h.to(torch.int32)

        full_kv_num_1h = full_i32.sum(dim=-1).to(torch.int32)
        _, full_kv_idx_1h = torch.topk(full_i32, k=C, dim=-1)
        full_kv_idx_1h = full_kv_idx_1h.to(torch.int32)

        kv_num = kv_num_1h.view(1, 1, C).expand(1, H, C).contiguous()
        kv_idx = kv_idx_1h.view(1, 1, C, C).expand(1, H, C, C).contiguous()
        full_kv_num = full_kv_num_1h.view(1, 1, C).expand(1, H, C).contiguous()
        full_kv_idx = full_kv_idx_1h.view(1, 1, C, C).expand(1, H, C, C).contiguous()

        return BlockMask.from_kv_blocks(
            kv_num,
            kv_idx,
            full_kv_num,
            full_kv_idx,
            BLOCK_SIZE=(chunk, chunk),
            mask_mod=_causal_mask_mod,
            seq_lengths=(T, T),
        )

    def _flash_fallback(self, q: Tensor, k: Tensor, v: Tensor, seqlens: Tensor, max_len: int, yarn, bm_size: int) -> Tensor:
        if not _HAS_FLASH_ATTN:
            raise RuntimeError("FlexAttention disabled/unavailable and flash_attn_interface is not importable")
        y = flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=seqlens,
            cu_seqlens_k=seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            causal=True,
            softmax_scale=yarn.attn_scale,
            window_size=(bm_size, 0),
        )
        return y.view_as(q)

    def _routed_flex(self, q: Tensor, k: Tensor, v: Tensor, seqlens: Tensor, yarn) -> Tensor:
        block_mask = self._build_routed_block_mask(q, k, seqlens)
        flex = self._flex_attn if self._flex_attn is not None else _get_flex_attention(self.route_compile_flex)

        y = flex(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            block_mask=block_mask,
            scale=_float_scale(yarn.attn_scale),
            kernel_options=self.flex_kernel_options,
        )
        return y.transpose(1, 2).contiguous()

    def forward(self, x: Tensor, attn_args, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0

        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn

        q, k, v = F.linear(
            x,
            sa_lambdas[0] * qkvo_w[: self.dim * 3].type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        max_len = train_max_seq_len if self.training else T

        q, k = _qk_norm(q), _qk_norm(k)
        q, k = yarn.rotary(q), yarn.rotary(k)

        if key_offset:
            k[:, 1:, :, self.head_dim // 2 :] = k[:, :-1, :, self.head_dim // 2 :]

        if aux_v is not None:
            v = v + aux_v.view_as(v)

        C = (T + self.route_chunk_size - 1) // self.route_chunk_size
        route_is_useful = C > (self.route_topk + self.route_local_prev + self.route_local_next + 1)
        use_routed = self.route_use_flex and _HAS_FLEX and q.is_cuda and route_is_useful

        if use_routed:
            y = self._routed_flex(q, k, v, seqlens, yarn)
        else:
            if not self.route_fallback_to_flash:
                raise RuntimeError("Routed FlexAttention disabled/unavailable or route is nearly dense")
            y = self._flash_fallback(q, k, v, seqlens, max_len, yarn, bm_size)

        if attn_args.xsa_alpha is not None:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3 :].type_as(y))
        return y
