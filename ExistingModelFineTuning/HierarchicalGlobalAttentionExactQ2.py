"""Qwen3-0.6B hierarchical attention with encapsulated cache state.

The forward contract follows Qwen3Attention.  Prefill/training stays vectorized;
decode projects only new hidden states, stores compact KV through Qwen Cache.update,
and stores routing tensors at ``past_key_value._hga[layer_idx]``.  With Qwen
StaticCache the HGA tensors are preallocated once and then mutated in-place.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utilities import RMSNorm  # type: ignore
except Exception:  # pragma: no cover - makes this file standalone for tests
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = x.float()
            out = self.weight.float() * x_fp32 * torch.rsqrt(
                x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return out.to(dtype=x.dtype)


CosSin = Tuple[torch.Tensor, torch.Tensor]

_QWEN3_06B_CACHE_IMPL_VERSION = "qwen3-06b-cache-hga-state-fastchunks-2026-06-18-v2"


class _HGAState:
    """Per-layer routing side-cache stored as ``past_key_value._hga[layer_idx]``.

    The tensor shapes are fixed after prefill when Qwen uses StaticCache, so the
    decode step can update them in-place instead of growing Python lists/tensors.
    ``__getitem__``/``__setitem__`` keep older internal helper code compact.
    """

    __slots__ = (
        "seen",
        "max_len",
        "max_chunks",
        "q_chunk",
        "chunk_k",
        "group_k",
        "cur_chunk_raw",
        "cur_chunk_rope",
        "cur_group_raw",
        "cur_group_rope",
    )

    def __init__(
        self,
        *,
        seen: int,
        max_len: int,
        max_chunks: int,
        q_chunk: torch.Tensor,
        chunk_k: torch.Tensor,
        group_k: torch.Tensor,
        cur_chunk_raw: torch.Tensor,
        cur_chunk_rope: torch.Tensor,
        cur_group_raw: torch.Tensor,
        cur_group_rope: torch.Tensor,
    ) -> None:
        self.seen = int(seen)
        self.max_len = int(max_len)
        self.max_chunks = int(max_chunks)
        self.q_chunk = q_chunk
        self.chunk_k = chunk_k
        self.group_k = group_k
        self.cur_chunk_raw = cur_chunk_raw
        self.cur_chunk_rope = cur_chunk_rope
        self.cur_group_raw = cur_group_raw
        self.cur_group_rope = cur_group_rope

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)



def _safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class GlobalAttention(nn.Module):
    """Chunk-routed causal attention with one public ``forward`` path.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 16,
        kv_heads: int = 8,
        dropout: float = 0.0,
        use_bias_q: bool = True,
        use_bias_k: bool = True,
        use_bias_v: bool = True,
        use_bias_o: bool = False,
        causal: bool = True,
        use_global: bool = True,
        chunk_size: int = 64,
        group_size: int = 16,
        topk_chunks: int = 20,
        topk_groups: int = 32,
        return_router_stats: bool = False,
        head_dim: Optional[int] = None,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        theta: float = 1_000_000.0,
        mixed_rope_threshold: float = 0.5,
        mixed_rope_cutoff_pair: Optional[int] = None,
        router_scale_init: float = 1.0,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert causal, "This implementation is causal-only."
        assert nhead % kv_heads == 0
        head_dim = d_model // nhead if head_dim is None else head_dim
        assert head_dim % 2 == 0, "RoPE needs an even head_dim."
        assert chunk_size % group_size == 0
        if mixed_rope_cutoff_pair is not None and not (0 <= mixed_rope_cutoff_pair <= head_dim // 2):
            raise ValueError(f"mixed_rope_cutoff_pair must be in [0, {head_dim // 2}]")

        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout_p = dropout
        self.causal = causal
        self.use_global = use_global
        self.chunk_size = chunk_size
        self.group_size = group_size
        self.groups_per_chunk = chunk_size // group_size
        self.topk_chunks = topk_chunks
        self.topk_groups = topk_groups
        self.theta = float(theta)
        self.return_router_stats = return_router_stats
        self.mixed_rope_threshold = float(mixed_rope_threshold)
        self.mixed_rope_cutoff_pair = mixed_rope_cutoff_pair

        # Original heuristic.  Kept unchanged for checkpoint/behavior continuity.
        self.group_kv_scale = 1.0 / (group_size + math.sqrt(group_size))

        self.q_proj = nn.Linear(d_model, nhead * head_dim, bias=use_bias_q)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_k)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_v)
        self.o_proj = nn.Linear(nhead * head_dim, d_model, bias=use_bias_o)

        make_norm = lambda: RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.q_norm = q_norm if q_norm is not None else make_norm()
        self.k_norm = k_norm if k_norm is not None else make_norm()

        # Qwen3Attention-compatible attributes used by HF generation/cache code.
        self.layer_idx = layer_idx
        self.num_key_value_groups = nhead // kv_heads
        self.scaling = self.scale
        self.attention_dropout = dropout
        self.is_causal = True
        self.sliding_window = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Qwen3Attention-compatible forward with real incremental KV use.

        Prefill/training uses the original vectorized path.  Decode projects only
        the new token(s), updates the Qwen cache, updates tiny chunk/group
        summaries, and runs the same hierarchical routing for the active chunk.
        """
        if hidden_states.ndim != 3:
            raise AssertionError("expected hidden_states with shape [B, S, hidden_size]")
        if hidden_states.shape[-1] != self.d_model:
            raise AssertionError(f"expected hidden dim {self.d_model}, got {hidden_states.shape[-1]}")

        if past_key_value is None:
            past_key_value = kwargs.get("past_key_values", None)

        B, q_len, _ = hidden_states.shape
        if q_len == 0:
            return hidden_states.new_empty(B, 0, self.d_model), None

        cos_qwen, sin_qwen = position_embeddings
        cos, sin = self._normalize_position_embeddings(position_embeddings, hidden_states)

        # This is intentionally only for hidden_states supplied by Qwen.  During
        # generation that tensor is normally [B, 1, D], so old tokens are not
        # re-projected.
        q_raw, k_raw_new, v_new = self._project_qkv_base(hidden_states)
        q_new = self._apply_rotary(q_raw.float(), cos, sin).to(dtype=q_raw.dtype)
        k_new = self._apply_rotary(k_raw_new.float(), cos, sin).to(dtype=k_raw_new.dtype)

        if past_key_value is None:
            k_all = self._repeat_kv(k_new)
            v_all = self._repeat_kv(v_new)
            k_raw_all = self._repeat_kv(k_raw_new)
            return self._forward_full_qkv(q_raw, q_new, k_raw_all, k_all, v_all, cos, sin, attention_mask)

        layer_idx = self._require_layer_idx()
        past_len = self._input_start_pos(past_key_value, layer_idx, cache_position)

        cache_kwargs = {"sin": sin_qwen, "cos": cos_qwen, "cache_position": cache_position}
        try:
            k_cache, v_cache = past_key_value.update(k_new, v_new, layer_idx, cache_kwargs)
        except TypeError:
            k_cache, v_cache = past_key_value.update(k_new, v_new, layer_idx)

        total_len = self._input_end_pos(k_cache, past_len, q_len, cache_position)
        max_cache_len = self._max_cache_len_from_cache(past_key_value, k_cache, total_len)

        if past_len == 0:
            # Prefill: compute exactly as the training path and initialize the
            # side summaries once for following decode calls.  With Qwen
            # StaticCache these side tensors have the same fixed capacity as KV.
            self._set_hga_state(
                past_key_value,
                layer_idx,
                self._build_hga_state_from_prefill(
                    q_new, k_raw_new, k_new, v_new, total_len, max_cache_len, hidden_states.device
                ),
            )
            k_all = self._repeat_kv(k_new)
            v_all = self._repeat_kv(v_new)
            k_raw_all = self._repeat_kv(k_raw_new)
            return self._forward_full_qkv(q_raw, q_new, k_raw_all, k_all, v_all, cos, sin, attention_mask)

        state = self._get_hga_state(past_key_value, layer_idx)
        if state is None or int(state.get("seen", -1)) != past_len:
            # Normally this never runs: prefill creates the state.  It prevents a
            # hard failure if a cache is injected from outside, at the cost of a
            # one-time summary rebuild from the already cached K.
            state = self._rebuild_hga_state_from_k_cache(
                k_cache[:, :, :past_len, :],
                v_cache[:, :, :past_len, :],
                past_len,
                max_cache_len,
                B,
                hidden_states.device,
                hidden_states.dtype,
            )
            self._set_hga_state(past_key_value, layer_idx, state)

        if q_len == 1:
            abs_pos = self._abs_pos_for_token(past_len, 0, cache_position)
            self._append_hga_token(state, q_new[:, :, 0], k_raw_new[:, :, 0], k_new[:, :, 0], v_new[:, :, 0], abs_pos)
            out_seq = self._decode_one_from_state(q_new[:, :, 0], state, k_cache, v_cache, abs_pos).unsqueeze(2)
        elif self._can_use_aligned_suffix_path(past_len, q_len, cache_position):
            # Chunked training/evaluation calls Qwen with e.g. q_len=2048 and
            # an existing KV cache.  Do not run 2048 one-token decodes; update
            # the HGA summaries for this block and process all new chunks at once.
            new_chunk_k, new_group_k = self._append_hga_block_aligned(state, q_new, k_raw_new, k_new, past_len, total_len)
            out_seq = self._forward_aligned_suffix_from_state(
                q_new, state, k_cache, v_cache, past_len, total_len, new_chunk_k, new_group_k
            )
        else:
            # Rare unaligned multi-token suffix fallback.  Generation is q_len=1;
            # the benchmark script uses aligned 2048-token chunks, so this path
            # should only handle unusual caller slicing.
            outs = []
            for i in range(q_len):
                abs_pos = self._abs_pos_for_token(past_len, i, cache_position)
                self._append_hga_token(state, q_new[:, :, i], k_raw_new[:, :, i], k_new[:, :, i], v_new[:, :, i], abs_pos)
                out_h = self._decode_one_from_state(q_new[:, :, i], state, k_cache, v_cache, abs_pos)
                outs.append(out_h.unsqueeze(2))
            out_seq = torch.cat(outs, dim=2)

        out = self.o_proj(out_seq.transpose(1, 2).contiguous().reshape(B, q_len, self.nhead * self.head_dim))
        return out, None

    def _forward_full_qkv(
        self,
        q_raw: torch.Tensor,
        q: torch.Tensor,
        k_raw: torch.Tensor,
        k_rope: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        B, H, S, Dh = q.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = q.device, q.dtype
        stats: Dict[str, Any] = {}

        if not self.use_global:
            attn_mask = self._slice_attention_mask(attention_mask, S, k_rope.shape[-2])
            attn = F.scaled_dot_product_attention(
                q,
                k_rope,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=attn_mask is None and q.shape[-2] == k_rope.shape[-2],
            )
            out = self.o_proj(attn.transpose(1, 2).contiguous().reshape(B, S, H * Dh))
            return out, None

        # Pad only inside this forward so chunk tensors are rectangular.
        N = (S + C - 1) // C
        S_pad = N * C
        valid_flat = torch.arange(S_pad, device=device) < S
        valid_chunks = valid_flat.view(N, C)                         # [N, C]
        chunk_len = valid_chunks.sum(dim=1)                          # [N]
        valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)        # [N, M]
        complete_groups = valid_chunks.view(N, M, gs).all(dim=-1)     # [N, M]

        q_raw_p = self._pad_seq(q_raw, S_pad)
        k_raw_p = self._pad_seq(k_raw, S_pad)
        q_p = self._pad_seq(q, S_pad)
        k_p = self._pad_seq(k_rope, S_pad)
        v_p = self._pad_seq(v, S_pad)

        q_chunks = q_p.reshape(B, H, N, C, Dh)
        k_chunks = k_p.reshape(B, H, N, C, Dh)
        v_chunks = v_p.reshape(B, H, N, C, Dh)
        k_raw_chunks = k_raw_p.reshape(B, H, N, C, Dh)

        chunk_start = torch.arange(N, device=device) * C
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

        group_start = torch.arange(N, device=device)[:, None] * C + torch.arange(M, device=device)[None, :] * gs
        group_len = valid_chunks.view(N, M, gs).sum(dim=-1)
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

        chunk_token_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
        group_token_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)

        chunk_k = self._rope_summary(
            raw=k_raw_chunks,
            rope=k_chunks,
            mask=chunk_token_mask,
            reduce_dim=3,
            anchor_pos=chunk_middle,
            cos=cos,
            sin=sin,
            scale=1.0,
        )

        k_raw_groups = k_raw_chunks.reshape(B, H, N, M, gs, Dh)
        k_rope_groups = k_chunks.reshape(B, H, N, M, gs, Dh)
        v_groups_exact = v_chunks.reshape(B, H, N, M, gs, Dh)
        group_k = self._rope_summary(
            raw=k_raw_groups,
            rope=k_rope_groups,
            mask=group_token_mask,
            reduce_dim=4,
            anchor_pos=group_middle,
            cos=cos,
            sin=sin,
            scale=self.group_kv_scale,
        )
        group_v_base = (v_groups_exact * group_token_mask).sum(dim=4) * self.group_kv_scale

        token_k = k_chunks.reshape(B, H, N, M, gs, Dh)
        token_v = v_chunks.reshape(B, H, N, M, gs, Dh)

        route_q_chunks = self._pad_seq(q, S_pad).reshape(B, H, N, C, Dh)
        neg_inf = -1.0e4
        query_valid = valid_chunks.view(1, 1, N, C, 1)

        # ------------------------------------------------------------------
        # 1) Choose previous chunks and expose their group summaries.
        # ------------------------------------------------------------------
        Kc = min(self.topk_chunks, N) if self.topk_chunks > 0 else 0
        if Kc > 0:
            with torch.no_grad():
                scores_roll = torch.einsum("bhncd,bhmd->bhncm", route_q_chunks, chunk_k) * self.scale
                prev_chunk_mask = torch.arange(N, device=device)[None, :] < torch.arange(N, device=device)[:, None]
                route_mask = prev_chunk_mask.view(1, 1, N, 1, N) & query_valid
                scores_for_candidates = scores_roll.masked_fill(~route_mask, neg_inf)
                req_idx, req_scores = self._route_topk_requests(scores_for_candidates, Kc)
                chunk_scores = self._max_route_scores_from_requests(req_idx, req_scores, N, neg_inf)

            top_chunk_scores, top_chunk_idx = self._topk_scores_indices(chunk_scores, Kc)
            query_chunk_idx = torch.arange(N, device=device).view(1, 1, N, 1)
            prev_chunk_idx = (query_chunk_idx - 1).clamp_min(0).expand(B, H, N, 1)
            has_prev = (query_chunk_idx > 0).expand(B, H, N, 1)
            missing_prev = has_prev & ~(top_chunk_idx == prev_chunk_idx).any(dim=-1, keepdim=True)
            replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
            top_chunk_idx = top_chunk_idx.scatter(
                -1,
                replace_at,
                torch.where(missing_prev, prev_chunk_idx, top_chunk_idx.gather(-1, replace_at)),
            )

            valid_prev = prev_chunk_mask.expand(B, H, N, N).gather(-1, top_chunk_idx)
            chunk_requested = self._requests_for_selected_routes(req_idx, top_chunk_idx) & valid_prev.unsqueeze(3)
            chunk_visible = torch.cumsum(chunk_requested.to(torch.int32), dim=3) > 0
            chunk_visible = chunk_visible & valid_prev.unsqueeze(3) & query_valid

            # Always expose the immediately preceding closed chunk.
            prev_slot = (top_chunk_idx == prev_chunk_idx).unsqueeze(3)
            chunk_visible = chunk_visible | (prev_slot & has_prev.unsqueeze(3) & query_valid).expand(B, H, N, C, Kc)
            chunk_visible = chunk_visible & valid_prev.unsqueeze(3)

            cand_k_groups = self._gather_chunks(group_k, top_chunk_idx)  # [B,H,N,Kc,M,D]
            cand_v_groups = self._gather_chunks(group_v_base, top_chunk_idx)
            cand_group_valid = (
                self._gather_chunks(valid_groups.view(1, 1, N, M).expand(B, H, -1, -1), top_chunk_idx)
                & valid_prev.unsqueeze(-1)
            )
            Tgrp = Kc * M

            cand_k_groups_flat = cand_k_groups.reshape(B, H, N, Tgrp, Dh)
            cand_v_groups_flat = cand_v_groups.reshape(B, H, N, Tgrp, Dh)
            cand_group_valid_flat = cand_group_valid.reshape(B, H, N, Tgrp)

            group_summary_scores = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, cand_k_groups_flat) * self.scale
            group_summary_visible = chunk_visible.unsqueeze(-1).expand(B, H, N, C, Kc, M).reshape(B, H, N, C, Tgrp)
            group_summary_mask = group_summary_visible & cand_group_valid_flat.unsqueeze(3) & query_valid
            group_summary_scores_masked = group_summary_scores.masked_fill(~group_summary_mask, neg_inf)
        else:
            top_chunk_idx = torch.empty(B, H, N, 0, device=device, dtype=torch.long)
            chunk_visible = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)
            Tgrp = 0
            cand_k_groups_flat = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
            cand_v_groups_flat = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
            cand_group_valid_flat = torch.empty(B, H, N, 0, device=device, dtype=torch.bool)
            group_summary_scores_masked = torch.empty(B, H, N, C, 0, device=device, dtype=dtype)
            group_summary_mask = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)

        # ------------------------------------------------------------------
        # 2) Open selected previous groups to exact tokens.
        # ------------------------------------------------------------------
        Kg = min(self.topk_groups, Tgrp) if self.topk_groups > 0 else 0
        Kg_request = min(self.topk_groups // 2, Tgrp) if self.topk_groups > 0 else 0
        if Kg > 0 and Kg_request > 0:
            with torch.no_grad():
                group_scores_roll = torch.einsum("bhncd,bhnrd->bhncr", route_q_chunks, cand_k_groups_flat) * self.scale
                group_scores_roll = group_scores_roll.masked_fill(~group_summary_mask, neg_inf)
                group_req_idx, group_req_scores = self._route_topk_requests(group_scores_roll, Kg_request)
                group_scores = self._max_route_scores_from_requests(group_req_idx, group_req_scores, Tgrp, neg_inf)

            _, top_group_idx = self._topk_scores_indices(group_scores, Kg)
            top_group_valid = cand_group_valid_flat.gather(-1, top_group_idx)
            parent_candidate = top_group_idx // M
            parent_visible = chunk_visible.gather(-1, parent_candidate.unsqueeze(3).expand(B, H, N, C, Kg))
            group_requested = (
                self._requests_for_selected_routes(group_req_idx, top_group_idx)
                & top_group_valid.unsqueeze(3)
                & parent_visible
            )
            group_visible = torch.cumsum(group_requested.to(torch.int32), dim=3) > 0
            group_visible = group_visible & top_group_valid.unsqueeze(3) & parent_visible & query_valid

            source_chunk_idx = top_chunk_idx.gather(-1, parent_candidate)
            source_group_idx = top_group_idx - parent_candidate * M
            opened_k = self._gather_groups(token_k, source_chunk_idx, source_group_idx).reshape(B, H, N, Kg * gs, Dh)
            opened_v_tokens = self._gather_groups(token_v, source_chunk_idx, source_group_idx).reshape(B, H, N, Kg * gs, Dh)
            opened_token_valid = self._gather_groups(
                valid_chunks.view(N, M, gs).view(1, 1, N, M, gs).expand(B, H, -1, -1, -1),
                source_chunk_idx,
                source_group_idx,
            ).reshape(B, H, N, Kg * gs)

            opened_scores = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, opened_k) * self.scale
            opened_visible = group_visible.unsqueeze(-1).expand(B, H, N, C, Kg, gs).reshape(B, H, N, C, Kg * gs)
            opened_visible = opened_visible & opened_token_valid.unsqueeze(3) & query_valid
            opened_scores = opened_scores.masked_fill(~opened_visible, neg_inf)
        else:
            top_group_idx = torch.empty(B, H, N, 0, device=device, dtype=torch.long)
            group_visible = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)
            opened_scores = torch.empty(B, H, N, C, 0, device=device, dtype=dtype)
            opened_v_tokens = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 3) Exact current tokens; group summaries are optional output tokens.
        # ------------------------------------------------------------------
        current_group_scores = torch.einsum("bhncd,bhnmd->bhncm", q_chunks, group_k) * self.scale
        local_t = torch.arange(C, device=device)
        group_end_local = torch.arange(M, device=device) * gs + (gs - 1)
        current_group_mask = (
            complete_groups.view(1, 1, N, 1, M)
            & (group_end_local.view(1, 1, 1, 1, M) <= local_t.view(1, 1, 1, C, 1))
            & query_valid
        )
        current_group_scores = current_group_scores.masked_fill(~current_group_mask, neg_inf)

        current_scores = torch.einsum("bhncd,bhned->bhnce", q_chunks, k_chunks) * self.scale
        causal_in_chunk = local_t.view(1, C) <= local_t.view(C, 1)
        current_token_mask = (
            causal_in_chunk.view(1, 1, 1, C, C)
            & valid_chunks.view(1, 1, N, 1, C)
            & valid_chunks.view(1, 1, N, C, 1)
        )
        current_scores = current_scores.masked_fill(~current_token_mask, neg_inf)

        common_scores = torch.cat(
            [group_summary_scores_masked, current_group_scores, current_scores, opened_scores],
            dim=-1,
        )

        common_scores = torch.where(query_valid, common_scores, torch.zeros_like(common_scores))
        probs = F.dropout(torch.softmax(common_scores.float(), dim=-1).to(dtype), p=self.dropout_p, training=self.training)

        n_prev = group_summary_scores_masked.shape[-1]
        n_cur_summary = current_group_scores.shape[-1]
        n_cur = C
        p_prev = probs[..., :n_prev]
        p_cur_summary = probs[..., n_prev:n_prev + n_cur_summary]
        p_cur = probs[..., n_prev + n_cur_summary:n_prev + n_cur_summary + n_cur]
        p_opened = probs[..., n_prev + n_cur_summary + n_cur:]

        out_chunks = torch.einsum("bhncr,bhnrd->bhncd", p_cur, v_chunks)
        if group_summary_scores_masked.shape[-1] > 0:
            out_chunks = out_chunks + torch.einsum("bhncr,bhnrd->bhncd", p_prev, cand_v_groups_flat)
        out_chunks = out_chunks + torch.einsum("bhncm,bhnmd->bhncd", p_cur_summary, group_v_base)
        if opened_scores.shape[-1] > 0:
            out_chunks = out_chunks + torch.einsum("bhncr,bhnrd->bhncd", p_opened, opened_v_tokens)

        out_seq = out_chunks.reshape(B, H, S_pad, Dh)[:, :, :S, :]
        out = self.o_proj(out_seq.transpose(1, 2).contiguous().reshape(B, S, H * Dh))

        if self.return_router_stats:
            stats = {
                "router_stats": {
                    "mixed_rope_cutoff_pair": torch.tensor(
                        self._mixed_cutoff_pair(), device=device, dtype=torch.int64
                    ),
                    "mean_visible_chunks_per_token": (
                        chunk_visible.float().sum(-1).mean().detach() if Kc > 0 else torch.tensor(0.0, device=device)
                    ),
                    "mean_visible_open_groups_per_token": (
                        group_visible.float().sum(-1).mean().detach() if Kg > 0 else torch.tensor(0.0, device=device)
                    ),
                    "top_chunk_idx": top_chunk_idx.detach(),
                    "top_group_idx": top_group_idx.detach(),
                }
            }
            return out, stats
        return out, None

    # ------------------------------------------------------------------
    # Incremental Qwen-cache decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_cache_len_from_cache(cache: Any, k_cache: torch.Tensor, total_len: int) -> int:
        # StaticCache returns its full preallocated K tensor; DynamicCache returns
        # only the real sequence length.  Prefer explicit cache metadata if the
        # local transformers version exposes it.
        for name in ("get_max_cache_shape", "get_max_length"):
            fn = getattr(cache, name, None)
            if fn is not None:
                try:
                    value = fn()
                    if value is not None:
                        return max(int(value), int(total_len))
                except Exception:
                    pass
        return max(int(k_cache.shape[-2]), int(total_len))

    @staticmethod
    def _mark_hga_static(*tensors: torch.Tensor) -> None:
        # Same idea as HF StaticCache: preserve addresses for cudagraph/compile.
        try:
            is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)()
        except Exception:
            is_compiling = False
        if is_compiling:
            return
        marker = getattr(getattr(torch, "_dynamo", None), "mark_static_address", None)
        if marker is None:
            return
        for tensor in tensors:
            try:
                marker(tensor)
            except Exception:
                pass

    def _grow_hga_state(self, state: _HGAState, min_chunks: int) -> None:
        # DynamicCache fallback only.  StaticCache should have enough capacity,
        # so compiled decode does not hit this path.
        old = state.max_chunks
        new_chunks = max(int(min_chunks), max(1, old * 2))
        B, KVH, _, Dh = state.chunk_k.shape
        M = self.groups_per_chunk
        chunk_k = state.chunk_k.new_zeros(B, KVH, new_chunks, Dh)
        group_k = state.group_k.new_zeros(B, KVH, new_chunks, M, Dh)
        chunk_k[:, :, :old, :].copy_(state.chunk_k)
        group_k[:, :, :old, :, :].copy_(state.group_k)
        state.chunk_k = chunk_k
        state.group_k = group_k
        state.max_chunks = new_chunks
        state.max_len = max(state.max_len, new_chunks * self.chunk_size)

    @staticmethod
    def _input_start_pos(cache: Any, layer_idx: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[0].detach().item())
        return GlobalAttention._cache_seq_length(cache, layer_idx)

    @staticmethod
    def _input_end_pos(k_cache: torch.Tensor, past_len: int, q_len: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[-1].detach().item()) + 1
        # DynamicCache returns the real sequence length; StaticCache can return
        # max-cache length, so prefer the known input range when no position was passed.
        return min(int(k_cache.shape[-2]), past_len + q_len)

    @staticmethod
    def _abs_pos_for_token(past_len: int, i: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[i].detach().item())
        return past_len + i

    @staticmethod
    def _hga_slots(cache: Any, layer_idx: int) -> List[Optional[_HGAState]]:
        slots = getattr(cache, "_hga", None)
        if slots is None or not isinstance(slots, list):
            slots = []
            setattr(cache, "_hga", slots)
        while len(slots) <= int(layer_idx):
            slots.append(None)
        return slots

    def _get_hga_state(self, cache: Any, layer_idx: int) -> Optional[_HGAState]:
        slots = getattr(cache, "_hga", None)
        if isinstance(slots, list) and int(layer_idx) < len(slots):
            return slots[int(layer_idx)]
        if isinstance(slots, dict):
            return slots.get(int(layer_idx))
        return None

    def _set_hga_state(self, cache: Any, layer_idx: int, state: _HGAState) -> None:
        slots = self._hga_slots(cache, layer_idx)
        slots[int(layer_idx)] = state

    def _build_hga_state_from_prefill(
        self,
        q: torch.Tensor,
        k_raw: torch.Tensor,
        k_rope: torch.Tensor,
        v: torch.Tensor,
        total_len: int,
        max_cache_len: int,
        device: torch.device,
    ) -> _HGAState:
        B, H, _, Dh = q.shape
        KVH = self.kv_heads
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        N = max(1, (total_len + C - 1) // C)
        max_chunks = max(N, (max(1, int(max_cache_len)) + C - 1) // C)
        S_pad = N * C
        dtype = k_raw.dtype
        valid = (torch.arange(S_pad, device=device) < total_len).view(N, C)
        km = valid.view(1, 1, N, C, 1).to(dtype)
        gm = valid.view(1, 1, N, M, gs, 1).to(dtype)

        k_raw_p = self._pad_seq(k_raw, S_pad).reshape(B, KVH, N, C, Dh)
        k_rope_p = self._pad_seq(k_rope, S_pad).reshape(B, KVH, N, C, Dh)

        chunk_len = valid.sum(dim=1)
        chunk_start = torch.arange(N, device=device) * C
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(total_len - 1)
        group_start = chunk_start[:, None] + torch.arange(M, device=device)[None, :] * gs
        group_len = valid.view(N, M, gs).sum(dim=-1)
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(total_len - 1)
        cos_all, sin_all = self._build_default_rotary(total_len, device, B)

        chunk_k_used = self._rope_summary(k_raw_p, k_rope_p, km, 3, chunk_middle, cos_all, sin_all, 1.0).contiguous()
        group_raw = (k_raw_p.reshape(B, KVH, N, M, gs, Dh) * gm).sum(dim=4)
        group_rope = (k_rope_p.reshape(B, KVH, N, M, gs, Dh) * gm).sum(dim=4)
        group_k_used = self._rope_summary(
            k_raw_p.reshape(B, KVH, N, M, gs, Dh),
            k_rope_p.reshape(B, KVH, N, M, gs, Dh),
            gm,
            4,
            group_middle,
            cos_all,
            sin_all,
            self.group_kv_scale,
        ).contiguous()
        chunk_k = k_raw.new_zeros(B, KVH, max_chunks, Dh)
        group_k = k_raw.new_zeros(B, KVH, max_chunks, M, Dh)
        with torch.no_grad():
            chunk_k[:, :, :N, :].copy_(chunk_k_used.detach())
            group_k[:, :, :N, :, :].copy_(group_k_used.detach())
        self._mark_hga_static(chunk_k, group_k)

        q_chunk = q.new_zeros(B, H, C, Dh)
        last_start = (N - 1) * C
        last_len = max(0, total_len - last_start)
        if last_len > 0:
            with torch.no_grad():
                q_chunk[:, :, :last_len, :].copy_(q[:, :, last_start:total_len, :].detach())
        last_group = max(0, (last_len - 1) // gs)
        chunk_raw_sum = (k_raw_p * km).sum(dim=3)
        chunk_rope_sum = (k_rope_p * km).sum(dim=3)

        self._mark_hga_static(q_chunk)
        return _HGAState(
            seen=int(total_len),
            max_len=int(max_cache_len),
            max_chunks=int(max_chunks),
            chunk_k=chunk_k,
            group_k=group_k,
            cur_chunk_raw=chunk_raw_sum[:, :, -1, :].detach().clone(),
            cur_chunk_rope=chunk_rope_sum[:, :, -1, :].detach().clone(),
            cur_group_raw=group_raw[:, :, -1, last_group, :].detach().clone(),
            cur_group_rope=group_rope[:, :, -1, last_group, :].detach().clone(),
            q_chunk=q_chunk,
        )

    def _rebuild_hga_state_from_k_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_len: int,
        max_cache_len: int,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _HGAState:
        if total_len == 0:
            Dh = self.head_dim
            max_chunks = max(1, (max(1, int(max_cache_len)) + self.chunk_size - 1) // self.chunk_size)
            q_chunk = torch.zeros(B, self.nhead, self.chunk_size, Dh, device=device, dtype=dtype)
            chunk_k = torch.zeros(B, self.kv_heads, max_chunks, Dh, device=device, dtype=dtype)
            group_k = torch.zeros(B, self.kv_heads, max_chunks, self.groups_per_chunk, Dh, device=device, dtype=dtype)
            self._mark_hga_static(q_chunk, chunk_k, group_k)
            return _HGAState(
                seen=0,
                max_len=int(max_cache_len),
                max_chunks=int(max_chunks),
                q_chunk=q_chunk,
                chunk_k=chunk_k,
                group_k=group_k,
                cur_chunk_raw=torch.zeros(B, self.kv_heads, Dh, device=device, dtype=dtype),
                cur_chunk_rope=torch.zeros(B, self.kv_heads, Dh, device=device, dtype=dtype),
                cur_group_raw=torch.zeros(B, self.kv_heads, Dh, device=device, dtype=dtype),
                cur_group_rope=torch.zeros(B, self.kv_heads, Dh, device=device, dtype=dtype),
            )
        cos, sin = self._build_default_rotary(total_len, device, B)
        k_raw = self._apply_rotary_inverse(k_cache.float(), cos, sin).to(k_cache.dtype)
        q_dummy = torch.zeros(B, self.nhead, total_len, self.head_dim, device=device, dtype=dtype)
        return self._build_hga_state_from_prefill(q_dummy, k_raw, k_cache, v_cache, total_len, max_cache_len, device)

    def _append_hga_token(
        self,
        state: Dict[str, Any],
        q_one: torch.Tensor,
        k_raw_one: torch.Tensor,
        k_rope_one: torch.Tensor,
        v_one: torch.Tensor,
        abs_pos: int,
    ) -> None:
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        n = abs_pos // C
        local = abs_pos - n * C
        g = local // gs
        B, KVH, Dh = k_raw_one.shape

        if n >= state.max_chunks:
            self._grow_hga_state(state, n + 1)

        with torch.no_grad():
            if local == 0:
                state["q_chunk"].zero_()
                state["cur_chunk_raw"].zero_()
                state["cur_chunk_rope"].zero_()
            if local % gs == 0:
                state["cur_group_raw"].zero_()
                state["cur_group_rope"].zero_()

            state["q_chunk"][:, :, local, :].copy_(q_one.detach())
            state["cur_chunk_raw"].add_(k_raw_one.detach())
            state["cur_chunk_rope"].add_(k_rope_one.detach())
            state["cur_group_raw"].add_(k_raw_one.detach())
            state["cur_group_rope"].add_(k_rope_one.detach())

            ccos, csin = self._rotary_at(torch.tensor(n * C + local // 2, device=k_raw_one.device), B, k_raw_one.device)
            gcos, gsin = self._rotary_at(torch.tensor(n * C + g * gs + (local % gs) // 2, device=k_raw_one.device), B, k_raw_one.device)
            chunk_anchor = self._apply_rotary(state["cur_chunk_raw"].float(), ccos, csin).to(k_raw_one.dtype)
            group_anchor = self._apply_rotary((state["cur_group_raw"] * self.group_kv_scale).float(), gcos, gsin).to(k_raw_one.dtype)
            state["chunk_k"][:, :, n, :].copy_(self._mix_tokenwise_and_anchor(state["cur_chunk_rope"], chunk_anchor))
            state["group_k"][:, :, n, g, :].copy_(
                self._mix_tokenwise_and_anchor(state["cur_group_rope"] * self.group_kv_scale, group_anchor)
            )
            state["seen"] = int(abs_pos) + 1

    def _materialize_summary_keys(
        self,
        state: Dict[str, Any],
        total_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        N = max(1, (total_len + C - 1) // C)
        valid_chunks = (torch.arange(N * C, device=device) < total_len).view(N, C)
        valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)
        complete_groups = valid_chunks.view(N, M, gs).all(dim=-1)
        return state["chunk_k"][:, :, :N, :], state["group_k"][:, :, :N, :, :], valid_chunks, valid_groups, complete_groups

    def _can_use_aligned_suffix_path(
        self,
        past_len: int,
        q_len: int,
        cache_position: Optional[torch.Tensor],
    ) -> bool:
        if int(past_len) % self.chunk_size != 0:
            return False
        if cache_position is not None and cache_position.numel() > 0:
            # The fast path needs one contiguous suffix and chunk-aligned start.
            if int(cache_position[0].detach().item()) != int(past_len):
                return False
            if cache_position.numel() != int(q_len):
                return False
            # Avoid expensive checks under compile; eager script path benefits.
            if not bool(torch.equal(cache_position, torch.arange(past_len, past_len + q_len, device=cache_position.device, dtype=cache_position.dtype))):
                return False
        return q_len > 1

    def _rotary_for_positions(
        self,
        pos: torch.Tensor,
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns [B,1,*pos.shape,Dh], directly broadcastable to [B,KVH,*pos.shape,Dh].
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        view = (1, 1, *pos.shape, self.head_dim)
        cos = emb.cos().reshape(view)
        sin = emb.sin().reshape(view)
        if batch != 1:
            cos = cos.expand(batch, *cos.shape[1:])
            sin = sin.expand(batch, *sin.shape[1:])
        return cos, sin

    def _append_hga_block_aligned(
        self,
        state: _HGAState,
        q: torch.Tensor,
        k_raw: torch.Tensor,
        k_rope: torch.Tensor,
        start_pos: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L, Dh = q.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        KVH = self.kv_heads
        device, dtype = q.device, k_raw.dtype
        start_chunk = int(start_pos) // C
        end_chunk = (int(total_len) + C - 1) // C
        n_new = end_chunk - start_chunk
        s_pad = n_new * C
        if end_chunk > state.max_chunks:
            self._grow_hga_state(state, end_chunk)

        valid = (torch.arange(s_pad, device=device) + int(start_pos)) < int(total_len)
        valid = valid.view(n_new, C)
        km = valid.view(1, 1, n_new, C, 1).to(dtype)
        gm = valid.view(1, 1, n_new, M, gs, 1).to(dtype)

        q_p = self._pad_seq(q, s_pad).reshape(B, H, n_new, C, Dh)
        k_raw_p = self._pad_seq(k_raw, s_pad).reshape(B, KVH, n_new, C, Dh)
        k_rope_p = self._pad_seq(k_rope, s_pad).reshape(B, KVH, n_new, C, Dh)

        chunk_raw_sum = (k_raw_p * km).sum(dim=3)
        chunk_rope_sum = (k_rope_p * km).sum(dim=3)
        chunk_len = valid.sum(dim=1)
        chunk_start = (start_chunk + torch.arange(n_new, device=device)) * C
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(max(int(total_len) - 1, 0))
        ccos, csin = self._rotary_for_positions(chunk_middle, B, device)
        chunk_anchor = self._apply_rotary(chunk_raw_sum.float(), ccos, csin).to(dtype)
        chunk_k = self._mix_tokenwise_and_anchor(chunk_rope_sum, chunk_anchor).contiguous()

        k_raw_g = k_raw_p.reshape(B, KVH, n_new, M, gs, Dh)
        k_rope_g = k_rope_p.reshape(B, KVH, n_new, M, gs, Dh)
        group_raw_sum = (k_raw_g * gm).sum(dim=4)
        group_rope_sum = (k_rope_g * gm).sum(dim=4)
        group_len = valid.view(n_new, M, gs).sum(dim=-1)
        group_start = chunk_start[:, None] + torch.arange(M, device=device)[None, :] * gs
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(max(int(total_len) - 1, 0))
        gcos, gsin = self._rotary_for_positions(group_middle, B, device)
        group_anchor = self._apply_rotary((group_raw_sum * self.group_kv_scale).float(), gcos, gsin).to(dtype)
        group_k = self._mix_tokenwise_and_anchor(group_rope_sum * self.group_kv_scale, group_anchor).contiguous()

        with torch.no_grad():
            state.chunk_k[:, :, start_chunk:end_chunk, :].copy_(chunk_k.detach())
            state.group_k[:, :, start_chunk:end_chunk, :, :].copy_(group_k.detach())
            state.q_chunk.zero_()
        last_len = int(total_len) - (end_chunk - 1) * C
        if last_len > 0:
            with torch.no_grad():
                state.q_chunk[:, :, :last_len, :].copy_(q_p[:, :, -1, :last_len, :].detach())
        last_group = max(0, (max(last_len, 1) - 1) // gs)
        state.cur_chunk_raw = chunk_raw_sum[:, :, -1, :].detach().clone()
        state.cur_chunk_rope = chunk_rope_sum[:, :, -1, :].detach().clone()
        state.cur_group_raw = group_raw_sum[:, :, -1, last_group, :].detach().clone()
        state.cur_group_rope = group_rope_sum[:, :, -1, last_group, :].detach().clone()
        state.seen = int(total_len)
        # Return differentiable summaries for the newly supplied block.  The
        # copies stored in state are buffers for future chunks/generation.
        return chunk_k, group_k

    def _forward_aligned_suffix_from_state(
        self,
        q: torch.Tensor,
        state: _HGAState,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start_pos: int,
        total_len: int,
        new_chunk_k: Optional[torch.Tensor] = None,
        new_group_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, L, Dh = q.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = q.device, q.dtype
        start_chunk = int(start_pos) // C
        n_query = (L + C - 1) // C
        n_total = max(1, (int(total_len) + C - 1) // C)
        s_pad = n_query * C
        q_chunks = self._pad_seq(q, s_pad).reshape(B, H, n_query, C, Dh)
        query_abs = start_chunk + torch.arange(n_query, device=device)
        ar_total = torch.arange(n_total, device=device)
        ar_c = torch.arange(C, device=device)
        ar_m = torch.arange(M, device=device)
        query_valid = ((query_abs[:, None] * C + ar_c[None, :]) < int(total_len)).view(1, 1, n_query, C, 1)
        neg_inf = -1.0e4

        if new_chunk_k is not None and new_group_k is not None:
            # Previous loss chunks are stored as detached buffers; the current
            # block summaries stay differentiable so q/k projection training is
            # not silently disabled inside the 2048-token chunk.
            chunk_k_base = torch.cat(
                [state.chunk_k[:, :, :start_chunk, :], new_chunk_k[:, :, :n_total - start_chunk, :]],
                dim=2,
            )
            group_k_base = torch.cat(
                [state.group_k[:, :, :start_chunk, :, :], new_group_k[:, :, :n_total - start_chunk, :, :]],
                dim=2,
            )
        else:
            chunk_k_base = state.chunk_k[:, :, :n_total, :]
            group_k_base = state.group_k[:, :, :n_total, :, :]
        chunk_k = self._repeat_kv(chunk_k_base)
        group_start_abs = ar_total[:, None] * C + ar_m[None, :] * gs
        valid_groups = group_start_abs < int(total_len)
        complete_groups = (group_start_abs + gs) <= int(total_len)

        Kc = min(self.topk_chunks, n_total) if self.topk_chunks > 0 else 0
        if Kc > 0:
            with torch.no_grad():
                scores_roll = torch.einsum("bhncd,bhmd->bhncm", q_chunks, chunk_k) * self.scale
                prev_chunk_mask = ar_total.view(1, -1) < query_abs.view(-1, 1)
                route_mask = prev_chunk_mask.view(1, 1, n_query, 1, n_total) & query_valid
                scores_for_candidates = scores_roll.masked_fill(~route_mask, neg_inf)
                req_idx, req_scores = self._route_topk_requests(scores_for_candidates, Kc)
                chunk_scores = self._max_route_scores_from_requests(req_idx, req_scores, n_total, neg_inf)
                top_chunk_scores, top_chunk_idx = self._topk_scores_indices(chunk_scores, Kc)
                prev_idx = (query_abs - 1).clamp_min(0).view(1, 1, n_query, 1).expand(B, H, n_query, 1)
                has_prev = (query_abs > 0).view(1, 1, n_query, 1).expand(B, H, n_query, 1)
                missing_prev = has_prev & ~(top_chunk_idx == prev_idx).any(dim=-1, keepdim=True)
                replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
                top_chunk_idx = top_chunk_idx.scatter(
                    -1,
                    replace_at,
                    torch.where(missing_prev, prev_idx, top_chunk_idx.gather(-1, replace_at)),
                )
                valid_prev = top_chunk_idx < query_abs.view(1, 1, n_query, 1)
                chunk_requested = self._requests_for_selected_routes(req_idx, top_chunk_idx) & valid_prev.unsqueeze(3)
                chunk_visible = torch.cumsum(chunk_requested.to(torch.int32), dim=3) > 0
                chunk_visible = chunk_visible & valid_prev.unsqueeze(3) & query_valid
                prev_slot = (top_chunk_idx == prev_idx).unsqueeze(3)
                chunk_visible = chunk_visible | (prev_slot & has_prev.unsqueeze(3) & query_valid).expand(B, H, n_query, C, Kc)
                chunk_visible = chunk_visible & valid_prev.unsqueeze(3)

            cand_k_groups = self._gather_kvhead_chunks(group_k_base, top_chunk_idx)
            cand_group_valid = valid_groups[top_chunk_idx] & valid_prev.unsqueeze(-1)
            Tgrp = Kc * M
            cand_k_flat = cand_k_groups.reshape(B, H, n_query, Tgrp, Dh)
            cand_valid_flat = cand_group_valid.reshape(B, H, n_query, Tgrp)
            group_ids = ar_m.view(1, 1, 1, 1, M).expand(B, H, n_query, Kc, M)
            cand_v_flat = self._gather_group_values(
                v_cache,
                top_chunk_idx.unsqueeze(-1).expand(B, H, n_query, Kc, M),
                group_ids,
                int(total_len),
            ).reshape(B, H, n_query, Tgrp, Dh)
            group_summary_scores = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, cand_k_flat) * self.scale
            group_summary_visible = chunk_visible.unsqueeze(-1).expand(B, H, n_query, C, Kc, M).reshape(B, H, n_query, C, Tgrp)
            group_summary_mask = group_summary_visible & cand_valid_flat.unsqueeze(3) & query_valid
            group_summary_scores_masked = group_summary_scores.masked_fill(~group_summary_mask, neg_inf)
        else:
            top_chunk_idx = torch.empty(B, H, n_query, 0, device=device, dtype=torch.long)
            Tgrp = 0
            cand_v_flat = torch.empty(B, H, n_query, 0, Dh, device=device, dtype=dtype)
            cand_valid_flat = torch.empty(B, H, n_query, 0, device=device, dtype=torch.bool)
            group_summary_scores_masked = torch.empty(B, H, n_query, C, 0, device=device, dtype=dtype)
            group_summary_mask = torch.empty(B, H, n_query, C, 0, device=device, dtype=torch.bool)
            chunk_visible = torch.empty(B, H, n_query, C, 0, device=device, dtype=torch.bool)

        Kg = min(self.topk_groups, Tgrp) if self.topk_groups > 0 else 0
        Kg_request = min(self.topk_groups // 2, Tgrp) if self.topk_groups > 0 else 0
        if Kg > 0 and Kg_request > 0:
            with torch.no_grad():
                group_scores_roll = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, cand_k_flat) * self.scale
                group_scores_roll = group_scores_roll.masked_fill(~group_summary_mask, neg_inf)
                group_req_idx, group_req_scores = self._route_topk_requests(group_scores_roll, Kg_request)
                group_scores = self._max_route_scores_from_requests(group_req_idx, group_req_scores, Tgrp, neg_inf)
                _, top_group_idx = self._topk_scores_indices(group_scores, Kg)
                top_group_valid = cand_valid_flat.gather(-1, top_group_idx)
                parent = top_group_idx // M
                parent_visible = chunk_visible.gather(-1, parent.unsqueeze(3).expand(B, H, n_query, C, Kg))
                group_requested = self._requests_for_selected_routes(group_req_idx, top_group_idx)
                group_requested = group_requested & top_group_valid.unsqueeze(3) & parent_visible
                group_visible = torch.cumsum(group_requested.to(torch.int32), dim=3) > 0
                group_visible = group_visible & top_group_valid.unsqueeze(3) & parent_visible & query_valid
                source_chunk_idx = top_chunk_idx.gather(-1, parent)
                source_group_idx = top_group_idx - parent * M

            pos = source_chunk_idx * C + source_group_idx * gs
            pos = pos.unsqueeze(-1) + torch.arange(gs, device=device).view(1, 1, 1, 1, gs)
            pos = pos.reshape(B, H, n_query, Kg * gs)
            opened_k = self._gather_kv_tokens(k_cache, pos.clamp_max(k_cache.shape[-2] - 1).reshape(B, H, n_query * Kg * gs)).reshape(B, H, n_query, Kg * gs, Dh)
            opened_v = self._gather_kv_tokens(v_cache, pos.clamp_max(v_cache.shape[-2] - 1).reshape(B, H, n_query * Kg * gs)).reshape(B, H, n_query, Kg * gs, Dh)
            opened_visible = group_visible.unsqueeze(-1).expand(B, H, n_query, C, Kg, gs).reshape(B, H, n_query, C, Kg * gs)
            opened_valid = pos < int(total_len)
            opened_scores = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, opened_k) * self.scale
            opened_scores = opened_scores.masked_fill(~(opened_visible & opened_valid.unsqueeze(3)), neg_inf)
        else:
            opened_scores = torch.empty(B, H, n_query, C, 0, device=device, dtype=dtype)
            opened_v = torch.empty(B, H, n_query, 0, Dh, device=device, dtype=dtype)

        cur_chunk_idx = query_abs.view(1, 1, n_query, 1).expand(B, H, n_query, 1)
        cur_group_k = self._gather_kvhead_chunks(group_k_base, cur_chunk_idx).squeeze(3)
        cur_group_ids = ar_m.view(1, 1, 1, M).expand(B, H, n_query, M)
        cur_group_v = self._gather_group_values(
            v_cache,
            cur_chunk_idx.expand(B, H, n_query, M),
            cur_group_ids,
            int(total_len),
        )
        group_end_local = ar_m * gs + (gs - 1)
        current_group_mask = (
            complete_groups[query_abs].view(1, 1, n_query, 1, M)
            & (group_end_local.view(1, 1, 1, 1, M) <= ar_c.view(1, 1, 1, C, 1))
            & query_valid
        )
        current_group_scores = torch.einsum("bhncd,bhnmd->bhncm", q_chunks, cur_group_k) * self.scale
        current_group_scores = current_group_scores.masked_fill(~current_group_mask, neg_inf)

        cur_pos = query_abs.view(1, 1, n_query, 1) * C + ar_c.view(1, 1, 1, C)
        cur_pos = cur_pos.expand(B, H, n_query, C)
        cur_k = self._gather_kv_tokens(k_cache, cur_pos.clamp_max(k_cache.shape[-2] - 1).reshape(B, H, n_query * C)).reshape(B, H, n_query, C, Dh)
        cur_v = self._gather_kv_tokens(v_cache, cur_pos.clamp_max(v_cache.shape[-2] - 1).reshape(B, H, n_query * C)).reshape(B, H, n_query, C, Dh)
        causal_in_chunk = ar_c.view(1, C) <= ar_c.view(C, 1)
        current_token_mask = causal_in_chunk.view(1, 1, 1, C, C) & (cur_pos < int(total_len)).unsqueeze(3) & query_valid
        current_scores = torch.einsum("bhncd,bhned->bhnce", q_chunks, cur_k) * self.scale
        current_scores = current_scores.masked_fill(~current_token_mask, neg_inf)

        scores = torch.cat([group_summary_scores_masked, current_group_scores, current_scores, opened_scores], dim=-1)
        scores = torch.where(query_valid, scores, torch.zeros_like(scores))
        probs = F.dropout(torch.softmax(scores.float(), dim=-1).to(dtype), p=self.dropout_p, training=self.training)
        n_prev = group_summary_scores_masked.shape[-1]
        n_cur_summary = current_group_scores.shape[-1]
        p_prev = probs[..., :n_prev]
        p_cur_summary = probs[..., n_prev:n_prev + n_cur_summary]
        p_cur = probs[..., n_prev + n_cur_summary:n_prev + n_cur_summary + C]
        p_open = probs[..., n_prev + n_cur_summary + C:]

        out = torch.einsum("bhncr,bhnrd->bhncd", p_cur, cur_v)
        if n_prev > 0:
            out = out + torch.einsum("bhncr,bhnrd->bhncd", p_prev, cand_v_flat)
        out = out + torch.einsum("bhncm,bhnmd->bhncd", p_cur_summary, cur_group_v)
        if opened_scores.shape[-1] > 0:
            out = out + torch.einsum("bhncr,bhnrd->bhncd", p_open, opened_v)
        return out.reshape(B, H, s_pad, Dh)[:, :, :L, :]

    def _decode_one_from_state(
        self,
        q_t: torch.Tensor,
        state: _HGAState,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        abs_pos: int,
    ) -> torch.Tensor:
        B, H, Dh = q_t.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = q_t.device, q_t.dtype
        total_len = abs_pos + 1
        n = abs_pos // C
        local = abs_pos - n * C
        max_chunks = state.chunk_k.shape[2]
        ar_c = torch.arange(C, device=device)
        ar_n = torch.arange(max_chunks, device=device)
        ar_m = torch.arange(M, device=device)
        q_chunk = state.q_chunk
        q_valid = ar_c <= local
        neg_inf = -1.0e4

        # Fixed-capacity views.  For StaticCache these shapes never change across
        # compiled decode steps; future chunks/groups are masked out below.
        chunk_k_base = state.chunk_k
        group_k_base = state.group_k
        chunk_k = self._repeat_kv(chunk_k_base)
        group_start_abs = ar_n[:, None] * C + ar_m[None, :] * gs
        valid_groups = group_start_abs < total_len
        complete_groups = (group_start_abs + gs) <= total_len

        Kc = min(self.topk_chunks, max_chunks) if self.topk_chunks > 0 else 0
        if Kc > 0:
            with torch.no_grad():
                scores_roll = torch.einsum("bhcd,bhnd->bhcn", q_chunk, chunk_k) * self.scale
                route_mask = (ar_n < n).view(1, 1, 1, max_chunks) & q_valid.view(1, 1, C, 1)
                scores_for_candidates = scores_roll.masked_fill(~route_mask, neg_inf)
                req_idx, req_scores = self._route_topk_requests(scores_for_candidates, Kc)
                chunk_scores = self._max_route_scores_from_requests(req_idx, req_scores, max_chunks, neg_inf)
                top_chunk_scores, top_chunk_idx = self._topk_scores_indices(chunk_scores, Kc)
                if n > 0:
                    prev_idx = torch.full((B, H, 1), n - 1, device=device, dtype=torch.long)
                    missing_prev = ~(top_chunk_idx == prev_idx).any(dim=-1, keepdim=True)
                    replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
                    top_chunk_idx = top_chunk_idx.scatter(
                        -1,
                        replace_at,
                        torch.where(missing_prev, prev_idx, top_chunk_idx.gather(-1, replace_at)),
                    )
                valid_prev = top_chunk_idx < n
                chunk_requested = self._requests_for_selected_routes_1d(req_idx, top_chunk_idx) & valid_prev.unsqueeze(2)
                chunk_visible = torch.cumsum(chunk_requested.to(torch.int32), dim=2) > 0
                chunk_visible = chunk_visible & valid_prev.unsqueeze(2) & q_valid.view(1, 1, C, 1)
                if n > 0:
                    prev_slot = (top_chunk_idx == (n - 1)).unsqueeze(2)
                    chunk_visible = chunk_visible | (prev_slot & q_valid.view(1, 1, C, 1))
                    chunk_visible = chunk_visible & valid_prev.unsqueeze(2)

            cand_k_groups = self._gather_kvhead_chunks(group_k_base, top_chunk_idx)
            cand_group_valid = valid_groups[top_chunk_idx] & valid_prev.unsqueeze(-1)
            Tgrp = Kc * M
            cand_k_flat = cand_k_groups.reshape(B, H, Tgrp, Dh)
            cand_valid_flat = cand_group_valid.reshape(B, H, Tgrp)
            group_ids = ar_m.view(1, 1, 1, M).expand(B, H, Kc, M)
            cand_v_flat = self._gather_group_values(
                v_cache,
                top_chunk_idx.unsqueeze(-1).expand(B, H, Kc, M),
                group_ids,
                total_len,
            ).reshape(B, H, Tgrp, Dh)

            group_scores_all = torch.einsum("bhcd,bhrd->bhcr", q_chunk, cand_k_flat) * self.scale
            group_mask_all = (
                chunk_visible.unsqueeze(-1).expand(B, H, C, Kc, M).reshape(B, H, C, Tgrp)
                & cand_valid_flat.unsqueeze(2)
                & q_valid.view(1, 1, C, 1)
            )
            group_scores_all = group_scores_all.masked_fill(~group_mask_all, neg_inf)
            group_summary_scores = (torch.einsum("bhd,bhrd->bhr", q_t, cand_k_flat) * self.scale).masked_fill(
                ~group_mask_all[:, :, local, :], neg_inf
            )
        else:
            top_chunk_idx = torch.empty(B, H, 0, device=device, dtype=torch.long)
            Tgrp = 0
            cand_v_flat = torch.empty(B, H, 0, Dh, device=device, dtype=dtype)
            cand_valid_flat = torch.empty(B, H, 0, device=device, dtype=torch.bool)
            group_scores_all = torch.empty(B, H, C, 0, device=device, dtype=dtype)
            group_summary_scores = torch.empty(B, H, 0, device=device, dtype=dtype)

        Kg = min(self.topk_groups, Tgrp) if self.topk_groups > 0 else 0
        Kg_request = min(self.topk_groups // 2, Tgrp) if self.topk_groups > 0 else 0
        if Kg > 0 and Kg_request > 0:
            with torch.no_grad():
                group_req_idx, group_req_scores = self._route_topk_requests(group_scores_all, Kg_request)
                group_scores = self._max_route_scores_from_requests(group_req_idx, group_req_scores, Tgrp, neg_inf)
                _, top_group_idx = self._topk_scores_indices(group_scores, Kg)
                top_group_valid = cand_valid_flat.gather(-1, top_group_idx)
                parent = top_group_idx // M
                source_chunk_idx = top_chunk_idx.gather(-1, parent)
                source_group_idx = top_group_idx - parent * M
                parent_visible = chunk_visible.gather(-1, parent.unsqueeze(2).expand(B, H, C, Kg)) if Kc > 0 else torch.zeros(B, H, C, Kg, device=device, dtype=torch.bool)
                group_requested = self._requests_for_selected_routes_1d(group_req_idx, top_group_idx)
                group_requested = group_requested & top_group_valid.unsqueeze(2) & parent_visible
                group_visible = torch.cumsum(group_requested.to(torch.int32), dim=2) > 0
                group_visible = group_visible & top_group_valid.unsqueeze(2) & parent_visible & q_valid.view(1, 1, C, 1)

            pos = source_chunk_idx * C + source_group_idx * gs
            pos = pos.unsqueeze(-1) + torch.arange(gs, device=device).view(1, 1, 1, gs)
            pos = pos.reshape(B, H, Kg * gs)
            pos_safe = pos.clamp_max(k_cache.shape[-2] - 1)
            opened_k = self._gather_kv_tokens(k_cache, pos_safe)
            opened_v = self._gather_kv_tokens(v_cache, pos_safe)
            opened_visible = group_visible[:, :, local, :].unsqueeze(-1).expand(B, H, Kg, gs).reshape(B, H, Kg * gs)
            opened_valid = pos < total_len
            opened_scores = (torch.einsum("bhd,bhrd->bhr", q_t, opened_k) * self.scale).masked_fill(
                ~(opened_visible & opened_valid), neg_inf
            )
        else:
            opened_scores = torch.empty(B, H, 0, device=device, dtype=dtype)
            opened_v = torch.empty(B, H, 0, Dh, device=device, dtype=dtype)

        cur_chunk_idx = torch.full((B, H, 1), n, device=device, dtype=torch.long)
        cur_group_k = self._gather_kvhead_chunks(group_k_base, cur_chunk_idx).squeeze(2)
        cur_group_ids = ar_m.view(1, 1, M).expand(B, H, M)
        cur_group_v = self._gather_group_values(v_cache, cur_chunk_idx.expand(B, H, M), cur_group_ids, total_len)
        group_end_local = ar_m * gs + (gs - 1)
        current_group_mask = complete_groups[n].view(1, 1, M) & (group_end_local.view(1, 1, M) <= local)
        current_group_scores = (torch.einsum("bhd,bhmd->bhm", q_t, cur_group_k) * self.scale).masked_fill(
            ~current_group_mask, neg_inf
        )

        cur_pos = (n * C + ar_c).view(1, 1, C).expand(B, H, C)
        cur_k = self._gather_kv_tokens(k_cache, cur_pos.clamp_max(k_cache.shape[-2] - 1))
        cur_v = self._gather_kv_tokens(v_cache, cur_pos.clamp_max(v_cache.shape[-2] - 1))
        current_token_mask = (ar_c <= local).view(1, 1, C) & (cur_pos < total_len)
        current_scores = (torch.einsum("bhd,bhcd->bhc", q_t, cur_k) * self.scale).masked_fill(
            ~current_token_mask, neg_inf
        )

        scores = torch.cat([group_summary_scores, current_group_scores, current_scores, opened_scores], dim=-1)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype)
        n_prev = group_summary_scores.shape[-1]
        n_cur_summary = current_group_scores.shape[-1]
        p_prev = probs[..., :n_prev]
        p_cur_summary = probs[..., n_prev:n_prev + n_cur_summary]
        p_cur = probs[..., n_prev + n_cur_summary:n_prev + n_cur_summary + C]
        p_open = probs[..., n_prev + n_cur_summary + C:]

        out = torch.einsum("bhr,bhrd->bhd", p_cur, cur_v)
        if n_prev > 0:
            out = out + torch.einsum("bhr,bhrd->bhd", p_prev, cand_v_flat)
        out = out + torch.einsum("bhm,bhmd->bhd", p_cur_summary, cur_group_v)
        if opened_scores.shape[-1] > 0:
            out = out + torch.einsum("bhr,bhrd->bhd", p_open, opened_v)
        return out

    @staticmethod
    @torch.no_grad()
    def _requests_for_selected_routes_1d(request_idx: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        # request_idx: [B,H,C,Kreq], selected_idx: [B,H,Ksel]
        return (request_idx.unsqueeze(-1) == selected_idx.unsqueeze(-2).unsqueeze(-3)).any(dim=-2)

    def _gather_kvhead_chunks(self, tensor: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
        """Gather tensor[B,KVH,N,*tail] at per-query-head chunk_idx[B,H,*idx]."""
        B, KVH, N = tensor.shape[:3]
        H = chunk_idx.shape[1]
        idx_shape = chunk_idx.shape[2:]
        tail = tensor.shape[3:]
        if math.prod(idx_shape) == 0:
            return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
        b_idx = torch.arange(B, device=tensor.device).view(B, 1, *([1] * len(idx_shape))).expand(B, H, *idx_shape)
        kv_idx = (torch.arange(H, device=tensor.device) // self.num_key_value_groups).view(1, H, *([1] * len(idx_shape))).expand(B, H, *idx_shape)
        return tensor[b_idx, kv_idx, chunk_idx]

    def _gather_kv_tokens(self, kv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Gather compact KV cache kv[B,KVH,S,D] at pos[B,H,R]."""
        B, KVH, S, Dh = kv.shape
        H = pos.shape[1]
        R = pos.shape[2]
        if R == 0:
            return torch.empty(B, H, 0, Dh, device=kv.device, dtype=kv.dtype)
        b_idx = torch.arange(B, device=kv.device).view(B, 1, 1).expand(B, H, R)
        kv_idx = (torch.arange(H, device=kv.device) // self.num_key_value_groups).view(1, H, 1).expand(B, H, R)
        return kv[b_idx, kv_idx, pos]

    def _gather_group_values(
        self,
        v_cache: torch.Tensor,
        chunk_idx: torch.Tensor,
        group_idx: torch.Tensor,
        total_len: int,
    ) -> torch.Tensor:
        # chunk_idx/group_idx: [B,H,*idx], returns [B,H,*idx,D]
        B, H = chunk_idx.shape[:2]
        idx_shape = chunk_idx.shape[2:]
        Dh = v_cache.shape[-1]
        if math.prod(idx_shape) == 0:
            return torch.empty(B, H, *idx_shape, Dh, device=v_cache.device, dtype=v_cache.dtype)
        pos = chunk_idx * self.chunk_size + group_idx * self.group_size
        pos = pos.unsqueeze(-1) + torch.arange(self.group_size, device=v_cache.device).view(*([1] * pos.ndim), self.group_size)
        flat = pos.reshape(B, H, -1)
        vals = self._gather_kv_tokens(v_cache, flat.clamp_max(v_cache.shape[-2] - 1)).reshape(B, H, *idx_shape, self.group_size, Dh)
        mask = (pos < total_len).unsqueeze(-1).to(vals.dtype)
        return (vals * mask).sum(dim=-2) * self.group_kv_scale

    def _rotary_at(self, pos: torch.Tensor, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1).reshape(1, 1, self.head_dim)
        cos = emb.cos()
        sin = emb.sin()
        if batch != 1:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)
        return cos, sin

    def _project_qkv_base(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        q = self.q_proj(hidden_states).reshape(B, S, H, Dh)
        k = self.k_proj(hidden_states).reshape(B, S, KVH, Dh)
        v = self.v_proj(hidden_states).reshape(B, S, KVH, Dh).transpose(1, 2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        return q, k, v

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        B, KVH, S, Dh = hidden_states.shape
        return hidden_states[:, :, None, :, :].expand(
            B, KVH, self.num_key_value_groups, S, Dh
        ).reshape(B, KVH * self.num_key_value_groups, S, Dh)

    def _normalize_position_embeddings(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("Qwen3 passes position_embeddings=(cos, sin); this attention requires it.")
        cos, sin = position_embeddings
        B, S = hidden_states.shape[:2]
        device = hidden_states.device

        def fix(t: torch.Tensor, name: str) -> torch.Tensor:
            t = t.to(device=device, dtype=torch.float32)
            if t.dim() == 3:      # [B, S, D], Qwen3RotaryEmbedding output
                t = t.unsqueeze(1)
            elif t.dim() == 4 and t.shape[1] == 1:
                pass              # [B, 1, S, D]
            else:
                raise ValueError(f"{name} must have shape [B,S,D] or [B,1,S,D]")
            if t.shape[-1] != self.head_dim:
                raise ValueError(f"{name} last dim must be head_dim={self.head_dim}, got {t.shape[-1]}")
            if t.shape[-2] != S:
                raise ValueError(f"{name} sequence dim must match hidden_states S={S}, got {t.shape[-2]}")
            if t.shape[0] == 1 and B != 1:
                t = t.expand(B, -1, -1, -1)
            if t.shape[0] != B:
                raise ValueError(f"{name} batch dim must be 1 or B={B}, got {t.shape[0]}")
            return t

        return fix(cos, "cos"), fix(sin, "sin")

    def _require_layer_idx(self) -> int:
        if self.layer_idx is None:
            raise ValueError("layer_idx must be set when using Qwen3 KV cache, as in Qwen3Attention(config, layer_idx).")
        return int(self.layer_idx)

    @staticmethod
    def _cache_seq_length(cache: Any, layer_idx: int) -> int:
        if cache is None:
            return 0
        try:
            return int(cache.get_seq_length(layer_idx))
        except TypeError:
            return int(cache.get_seq_length())
        except Exception:
            return 0

    @staticmethod
    def _actual_kv_len(k_all: torch.Tensor, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[-1].detach().item()) + 1
        return int(k_all.shape[-2])

    def _build_default_rotary(self, seq_len: int, device: torch.device, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_rotary(seq_len, device, torch.float32)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        if batch != 1:
            cos = cos.expand(batch, -1, -1, -1)
            sin = sin.expand(batch, -1, -1, -1)
        return cos, sin

    @staticmethod
    def _apply_rotary_inverse(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos - torch.cat((-x2, x1), dim=-1) * sin

    @staticmethod
    def _slice_attention_mask(
        attention_mask: Optional[torch.Tensor],
        query_len: int,
        key_len: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.ndim == 4:
            return attention_mask[:, :, -query_len:, :key_len]
        return attention_mask

    # ------------------------------------------------------------------
    # Compatibility/calibration helpers
    # ------------------------------------------------------------------

    def _head_view(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ``p`` is stored as [1, H, 1, 1, 1].  Reshape it so head dim stays at
        # target dimension 1 and every other target dimension is broadcastable.
        shape = [1] * target.ndim
        shape[1] = self.nhead
        return p.reshape(1, self.nhead, 1, 1, 1).reshape(*shape).to(device=target.device, dtype=target.dtype)

    # ------------------------------------------------------------------
    # RoPE summary / router helpers
    # ------------------------------------------------------------------

    def _rope_summary(
        self,
        raw: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
        reduce_dim: int,
        anchor_pos: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raw_sum = (raw * mask).sum(dim=reduce_dim) * scale
        anchor_cos, anchor_sin = self._gather_rotary(cos, sin, anchor_pos)
        endpoint = self._apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
           
        tokenwise = (rope * mask).sum(dim=reduce_dim) * scale
        assert endpoint is not None
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=endpoint)

    def _mix_tokenwise_and_anchor(self, tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        # High-frequency pairs are tokenwise; low-frequency pairs use the anchor.
        mask = self._mixed_tokenwise_mask(tokenwise.device).to(dtype=torch.bool)
        view_shape = [1] * tokenwise.ndim
        view_shape[-1] = self.head_dim
        mask = mask.view(*view_shape)
        return torch.where(mask, tokenwise, anchor)

    def _mixed_cutoff_pair(self) -> int:
        half = self.head_dim // 2
        if self.mixed_rope_cutoff_pair is not None:
            return int(self.mixed_rope_cutoff_pair)
        # Pair i is tokenwise when the phase changes more than threshold radians
        # across one chunk.  RoPE inv_freq is largest for small i, so the first
        # ``cutoff`` pairs are the high-frequency tokenwise part.
        cutoff = 0
        for i in range(half):
            inv_freq = 1.0 / (self.theta ** (i / half))
            if max(1, self.chunk_size - 1) * inv_freq > self.mixed_rope_threshold:
                cutoff = i + 1
        return cutoff

    def _mixed_tokenwise_mask(self, device: torch.device) -> torch.Tensor:
        half = self.head_dim // 2
        cutoff = self._mixed_cutoff_pair()
        pair_mask = torch.arange(half, device=device) < cutoff
        return torch.cat([pair_mask, pair_mask], dim=0)

    # ------------------------------------------------------------------
    # Stateless tensor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    @staticmethod
    def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, H, S, D]
        pad = target_len - x.shape[2]
        if pad <= 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:2], pad, x.shape[-1])], dim=2)

    @staticmethod
    def _causal_rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
        # x: [B, H, S, D]. Returns current token + up to window-1 previous tokens.
        if window <= 1:
            return x
        csum = torch.cumsum(x, dim=2)
        out = csum.clone()
        if x.shape[2] > window:
            out[:, :, window:, :] = out[:, :, window:, :] - csum[:, :, :-window, :]
        return out

    @staticmethod
    def _gather_rotary(cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> CosSin:
        # cos/sin: [B,1,S,D], pos: arbitrary int tensor on same device.
        shape = pos.shape
        flat = pos.reshape(-1)
        cos_g = cos[:, :, flat, :].reshape(cos.shape[0], 1, *shape, cos.shape[-1])
        sin_g = sin[:, :, flat, :].reshape(sin.shape[0], 1, *shape, sin.shape[-1])
        return cos_g, sin_g

    @staticmethod
    def _gather_chunks(tensor: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
        """Gather from ``tensor[B,H,N,*tail]`` using ``chunk_idx[B,H,*idx_shape]``."""
        B, H, Nsrc = tensor.shape[:3]
        idx_shape = chunk_idx.shape[2:]
        tail = tensor.shape[3:]
        flat = math.prod(idx_shape)
        if flat == 0:
            return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
        t = tensor.reshape(B * H, Nsrc, *tail)
        idx = chunk_idx.reshape(B * H, flat)
        bh = torch.arange(B * H, device=tensor.device).view(B * H, 1)
        return t[bh, idx].reshape(B, H, *idx_shape, *tail)

    @staticmethod
    def _gather_groups(tensor: torch.Tensor, chunk_idx: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        """Gather from ``tensor[B,H,N,M,*tail]`` using matching chunk/group indices."""
        B, H, Nsrc, M = tensor.shape[:4]
        idx_shape = chunk_idx.shape[2:]
        tail = tensor.shape[4:]
        flat = math.prod(idx_shape)
        if flat == 0:
            return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
        t = tensor.reshape(B * H, Nsrc, M, *tail)
        cidx = chunk_idx.reshape(B * H, flat)
        gidx = group_idx.reshape(B * H, flat)
        bh = torch.arange(B * H, device=tensor.device).view(B * H, 1)
        return t[bh, cidx, gidx].reshape(B, H, *idx_shape, *tail)

    @staticmethod
    @torch.no_grad()
    def _all_route_indices(prefix_shape: torch.Size, num_routes: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(num_routes, device=device).view(*((1,) * len(prefix_shape)), num_routes)
        return idx.expand(*prefix_shape, num_routes)

    @torch.no_grad()
    def _topk_scores_indices(self, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_routes = scores.shape[-1]
        assert k > 0
        if k >= num_routes:
            return scores, self._all_route_indices(scores.shape[:-1], num_routes, scores.device)
        return torch.topk(scores, k, dim=-1, sorted=False)

    @torch.no_grad()
    def _route_topk_requests(self, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token route requests. No train/eval random route injection."""
        num_routes = scores.shape[-1]
        assert k > 0
        if k >= num_routes:
            return self._all_route_indices(scores.shape[:-1], num_routes, scores.device), scores
        top_scores, top_idx = torch.topk(scores, k, dim=-1)
        return top_idx, top_scores

    @staticmethod
    @torch.no_grad()
    def _max_route_scores_from_requests(
        request_idx: torch.Tensor,
        request_scores: torch.Tensor,
        num_routes: int,
        fill_value: float,
    ) -> torch.Tensor:
        # request_idx/request_scores: [*, token_count, k]
        if request_idx.shape[-1] == num_routes:
            return request_scores.max(dim=-2).values
        prefix_shape = request_idx.shape[:-2]
        flat_prefix = math.prod(prefix_shape)
        route_scores = torch.full(
            (flat_prefix, num_routes),
            fill_value,
            device=request_scores.device,
            dtype=request_scores.dtype,
        )
        route_scores.scatter_reduce_(
            1,
            request_idx.reshape(flat_prefix, -1),
            request_scores.reshape(flat_prefix, -1),
            reduce="amax",
            include_self=True,
        )
        return route_scores.reshape(*prefix_shape, num_routes)

    @staticmethod
    @torch.no_grad()
    def _requests_for_selected_routes(request_idx: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        # request_idx: [B,H,N,C,Kreq], selected_idx: [B,H,N,Ksel]
        return (request_idx.unsqueeze(-1) == selected_idx.unsqueeze(-2).unsqueeze(-3)).any(dim=-2)

    def _get_rotary(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> CosSin:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
