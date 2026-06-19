"""Chunk-routed causal attention with dense-checkpoint compatibility knobs.

This is a drop-in replacement for the previous ``GlobalAttention`` class.  It keeps
q_proj/k_proj/v_proj/o_proj parameter names and shapes unchanged, so checkpoints
trained with the matching DenseAttention SmallLM can be loaded directly.

There is intentionally no KV cache and no separate generation path.  For decoding,
call the model on the whole currently available prefix.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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


RotaryData = Tuple[torch.Tensor, torch.Tensor]

def _safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class HierarchicalGlobalAttention(nn.Module):
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
        router_scale_init: float = 1.0
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

        # Distinct index per decoder layer; set by the HF integration and used as the
        # KV-cache key.  Defaults to 0 so single-layer standalone use still works.
        self.layer_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        rotary_data: Optional[RotaryData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Dispatch between the dense full-sequence path and the incremental KV cache.

        The cache path is only engaged for inference (``not self.training``) of the
        hierarchical (``use_global``) attention with a real cache object.  It mirrors
        the original Qwen3 attention contract: keys/values are stored through
        ``past_key_values.update`` while the hierarchical chunk/group summaries live in
        a side structure attached to the same cache object.
        """
        cache_position = kw.get("cache_position", None)
        if self._cache_active(use_cache, past_key_values) and x.ndim == 3 and x.shape[1] > 0:
            past_len = self._past_len(past_key_values, cache_position, self.layer_idx)
            if past_len > 0:
                if x.shape[1] <= self._decode_max_qlen():
                    return self._decode_forward(x, rotary_data, past_key_values, cache_position, past_len)
                # A long query block with existing context is not a decode step
                # (e.g. chunked-loss eval); fall through to the dense path uncached.
            else:
                out, stats = self._forward_dense(x, rotary_data)
                self._store_prefill_cache(x, rotary_data, past_key_values, cache_position)
                return out, stats
        return self._forward_dense(x, rotary_data)

    def _forward_dense(
        self,
        x: torch.Tensor,
        rotary_data: Optional[RotaryData] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if x.ndim != 3:
            raise AssertionError("expected x with shape [B, S, d_model]")
        if x.shape[-1] != self.d_model:
            raise AssertionError(f"expected hidden dim {self.d_model}, got {x.shape[-1]}")

        B, S, _ = x.shape
        H, Dh = self.nhead, self.head_dim
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = x.device, x.dtype
        stats: Dict[str, Any] = {}

        if S == 0:
            return x.new_empty(B, 0, self.d_model), stats

        if rotary_data is None:
            cos, sin = self._get_rotary(S, x.device, torch.float32)
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos, sin = self._normalize_rotary(rotary_data, x)

        q_raw, k_raw, v = self._project_qkv(x)
        q = self._apply_rotary(q_raw.float(), cos, sin).to(dtype=q_raw.dtype)
        k_rope = self._apply_rotary(k_raw.float(), cos, sin).to(dtype=k_raw.dtype)

        if not self.use_global:
            attn = F.scaled_dot_product_attention(
                q,
                k_rope,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
            out = self.o_proj(attn.transpose(1, 2).contiguous().reshape(B, S, H * Dh))
            return out, stats

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
        chunk_end = (chunk_start + chunk_len.clamp_min(1) - 1).clamp_max(S - 1)
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

        group_start = torch.arange(N, device=device)[:, None] * C + torch.arange(M, device=device)[None, :] * gs
        group_len = valid_chunks.view(N, M, gs).sum(dim=-1)
        group_end = (group_start + group_len.clamp_min(1) - 1).clamp_max(S - 1)
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

        chunk_token_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
        group_token_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)

        # Chunk/group key summaries used by the router.  The tokenwise/mixed modes
        # are designed to be closer to dense RoPE geometry than endpoint rotation.
        chunk_anchor = chunk_middle
        chunk_k = self._rope_summary(
            raw=k_raw_chunks,
            rope=k_chunks,
            mask=chunk_token_mask,
            reduce_dim=3,
            anchor_pos=chunk_anchor,
            cos=cos,
            sin=sin,
            scale=1.0,
        )

        k_raw_groups = k_raw_chunks.reshape(B, H, N, M, gs, Dh)
        k_rope_groups = k_chunks.reshape(B, H, N, M, gs, Dh)
        v_groups_exact = v_chunks.reshape(B, H, N, M, gs, Dh)
        group_anchor = group_middle
        group_k = self._rope_summary(
            raw=k_raw_groups,
            rope=k_rope_groups,
            mask=group_token_mask,
            reduce_dim=4,
            anchor_pos=group_anchor,
            cos=cos,
            sin=sin,
            scale=self.group_kv_scale,
        )
        
        group_v_base = (v_groups_exact * group_token_mask).sum(dim=4) * self.group_kv_scale
        
        token_k = k_chunks.reshape(B, H, N, M, gs, Dh)
        token_v = v_chunks.reshape(B, H, N, M, gs, Dh)

        route_q = q
        route_q_chunks = self._pad_seq(route_q, S_pad).reshape(B, H, N, C, Dh)
        neg_inf = torch.finfo(dtype).min
        
        #if dtype in (torch.float16, torch.bfloat16):
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
        

        # Avoid NaNs in padded query rows.  These rows are removed before return.
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
        if p_prev is not None and group_summary_scores_masked.shape[-1] > 0:
            out_chunks = out_chunks + torch.einsum("bhncr,bhnrd->bhncd", p_prev, cand_v_groups_flat)
        if p_cur_summary is not None:
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

    # ------------------------------------------------------------------
    # KV cache (generation / inference decode)
    # ------------------------------------------------------------------
    #
    # Layout.  The keys/values are stored exactly like the stock Qwen3 attention:
    # rope-applied, kv-head granularity, inside ``past_key_values`` via ``.update``.
    # The hierarchical chunk/group summaries that the router needs are kept in a side
    # structure attached to the same cache object (one entry per layer):
    #
    #   ck : [B, KVH, Nc, Dh]        closed-chunk key summaries
    #   gk : [B, KVH, Nc, M, Dh]     closed-chunk per-group key summaries
    #   gv : [B, KVH, Nc, M, Dh]     closed-chunk per-group value summaries
    #   kraw: [B, KVH, L, Dh]        raw (pre-RoPE) keys of the *current* partial chunk
    #
    # where Nc is the number of completed chunks.  Only the partial active chunk keeps
    # raw keys around (needed for the mixed-RoPE summary); everything older is reduced
    # to summaries, so per-step work and memory stay sub-linear in the sequence length.
    #
    # Routing during decode is per-query causal (each new token selects its own top-k
    # chunks/groups), which is the strictly-causal limit of the chunk-shared selection
    # used at training time.  Beam search (cache reordering) is not supported.

    def _cache_active(self, use_cache: Optional[bool], past_key_values: Any) -> bool:
        return (
            bool(use_cache)
            and past_key_values is not None
            and self.use_global
            and not self.training
        )

    def _decode_max_qlen(self) -> int:
        # Above this query length a call with existing context is treated as a dense
        # block rather than an incremental decode (keeps chunked-loss eval fast).
        return self.chunk_size

    @staticmethod
    def _past_len(past_key_values: Any, cache_position: Optional[torch.Tensor], layer_idx: int) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position.reshape(-1)[0].item())
        try:
            return int(past_key_values.get_seq_length(layer_idx))
        except TypeError:
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                return 0
        except Exception:
            return 0

    def _ha_get_state(self, past_key_values: Any) -> Dict[str, Any]:
        store = getattr(past_key_values, "_ha_state", None)
        if store is None:
            store = {}
            setattr(past_key_values, "_ha_state", store)
        st = store.get(self.layer_idx)
        if st is None:
            st = {"kraw": None, "ck": None, "gk": None, "gv": None}
            store[self.layer_idx] = st
        return st

    @staticmethod
    def _repeat_kv_heads(t: torch.Tensor, rep: int) -> torch.Tensor:
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    @staticmethod
    def _gather_tokens(cache_kv: torch.Tensor, pos: torch.Tensor, rep: int) -> torch.Tensor:
        """Gather per-(batch, head) token vectors from a kv-head cache.

        ``cache_kv``: [B, KVH, S, Dh]; ``pos``: [B, H, T] absolute indices (per q-head);
        returns [B, H, T, Dh].  Uses advanced indexing so only the requested tokens are
        touched (no full repeat of the cache to H heads).
        """
        B, KVH, S, Dh = cache_kv.shape
        H = pos.shape[1]
        device = cache_kv.device
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        h_idx = (torch.arange(H, device=device) // rep).view(1, H, 1)
        return cache_kv[b_idx, h_idx, pos]

    @torch.no_grad()
    def _store_prefill_cache(
        self,
        x: torch.Tensor,
        rotary_data: Optional[RotaryData],
        past_key_values: Any,
        cache_position: Optional[torch.Tensor],
    ) -> None:
        """Populate the KV cache after a prefill forward (positions 0..S-1)."""
        B, S, _ = x.shape
        KVH, Dh = self.kv_heads, self.head_dim
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk

        if rotary_data is None:
            cos, sin = self._get_rotary(S, x.device, torch.float32)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos, sin = self._normalize_rotary(rotary_data, x)

        _, k_kv, v_kv = self._project_qkv_kv(x)
        krope_kv = self._apply_rotary(k_kv.float(), cos, sin).to(dtype=k_kv.dtype)

        # Store rope keys / values exactly like stock Qwen3.
        past_key_values.update(krope_kv, v_kv, self.layer_idx)

        state = self._ha_get_state(past_key_values)
        n_closed = S // C
        if n_closed > 0:
            end = n_closed * C
            krope = krope_kv[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            kraw = k_kv[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            vtok = v_kv[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)

            raw_g = kraw.reshape(B, KVH, n_closed, M, gs, Dh)
            rope_g = krope.reshape(B, KVH, n_closed, M, gs, Dh)
            v_g = vtok.reshape(B, KVH, n_closed, M, gs, Dh)
            ar_n = torch.arange(n_closed, device=x.device)
            ar_m = torch.arange(M, device=x.device)
            g_anchor = ar_n[:, None] * C + ar_m[None, :] * gs + (gs - 1) // 2  # [Nc, M]
            state["gk"] = self._rope_summary_at(raw_g, rope_g, 4, g_anchor, self.group_kv_scale)
            state["gv"] = v_g.sum(dim=4) * self.group_kv_scale

            c_anchor = ar_n * C + (C - 1) // 2  # [Nc]
            state["ck"] = self._rope_summary_at(kraw, krope, 3, c_anchor, 1.0)

        rem = S - n_closed * C
        state["kraw"] = k_kv[:, :, n_closed * C:, :].contiguous() if rem > 0 else None

    @torch.no_grad()
    def _decode_forward(
        self,
        x: torch.Tensor,
        rotary_data: Optional[RotaryData],
        past_key_values: Any,
        cache_position: Optional[torch.Tensor],
        past_len: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Incremental hierarchical attention for new query tokens with cached context."""
        B, L, _ = x.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        rep = H // KVH
        device, dtype = x.device, x.dtype
        scale = self.scale

        if rotary_data is None:
            pos_abs = torch.arange(past_len, past_len + L, device=device)
            cos, sin = self._rotary_for_positions(pos_abs, x.new_zeros(B, KVH, L, Dh))
        else:
            cos, sin = self._normalize_rotary(rotary_data, x)

        q, k_kv, v_kv = self._project_qkv_kv(x)
        q = self._apply_rotary(q.float(), cos, sin).to(dtype=dtype)
        krope_kv = self._apply_rotary(k_kv.float(), cos, sin).to(dtype=dtype)

        krope_full, v_full = past_key_values.update(krope_kv, v_kv, self.layer_idx)

        if cache_position is not None and cache_position.numel() > 0:
            positions = cache_position.reshape(-1).to(device).long()
        else:
            positions = torch.arange(past_len, past_len + L, device=device)

        state = self._ha_get_state(past_key_values)
        kraw_buf = state["kraw"]
        ck, gk, gv = state["ck"], state["gk"], state["gv"]
        ar_m = torch.arange(M, device=device)

        outs: list[torch.Tensor] = []
        for i in range(L):
            p = int(positions[i].item())
            n = p // C
            c = p % C

            ki_raw = k_kv[:, :, i:i + 1, :]  # [B, KVH, 1, Dh]
            kraw_buf = ki_raw if (kraw_buf is None or kraw_buf.shape[2] == 0) else torch.cat([kraw_buf, ki_raw], dim=2)

            qi = q[:, :, i, :]  # [B, H, Dh]
            scores: list[torch.Tensor] = []
            values: list[torch.Tensor] = []

            # ---- Segment C: exact tokens of the current (active) chunk ----
            act_krope = krope_full[:, :, n * C: p + 1, :]  # [B, KVH, c+1, Dh]
            act_v = v_full[:, :, n * C: p + 1, :]
            scores.append(torch.einsum("bhd,bhtd->bht", qi, self._repeat_kv_heads(act_krope, rep)) * scale)
            values.append(self._repeat_kv_heads(act_v, rep))

            # ---- Segment B: completed group summaries of the active chunk ----
            ncomp = (c + 1) // gs
            if ncomp > 0:
                raw_g = kraw_buf[:, :, :ncomp * gs, :].reshape(B, KVH, ncomp, gs, Dh)
                rope_g = act_krope[:, :, :ncomp * gs, :].reshape(B, KVH, ncomp, gs, Dh)
                vg = act_v[:, :, :ncomp * gs, :].reshape(B, KVH, ncomp, gs, Dh)
                anchor = n * C + torch.arange(ncomp, device=device) * gs + (gs - 1) // 2
                gk_act = self._rope_summary_at(raw_g, rope_g, 3, anchor, self.group_kv_scale)  # [B,KVH,ncomp,Dh]
                gv_act = vg.sum(dim=3) * self.group_kv_scale
                scores.append(torch.einsum("bhd,bhtd->bht", qi, self._repeat_kv_heads(gk_act, rep)) * scale)
                values.append(self._repeat_kv_heads(gv_act, rep))

            # ---- Segments A (previous-chunk summaries) + D (opened tokens) ----
            if n > 0 and ck is not None and self.topk_chunks > 0:
                ck_h = self._repeat_kv_heads(ck, rep)  # [B, H, n, Dh]
                sc_chunk = torch.einsum("bhd,bhnd->bhn", qi, ck_h) * scale
                Kc = min(self.topk_chunks, n)
                top_chunk_scores, top_chunk_idx = torch.topk(sc_chunk, Kc, dim=-1, sorted=False)  # [B,H,Kc]

                # Always expose the immediately preceding chunk.
                prev = n - 1
                missing = ~(top_chunk_idx == prev).any(dim=-1, keepdim=True)
                replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
                top_chunk_idx = top_chunk_idx.scatter(
                    -1,
                    replace_at,
                    torch.where(missing, torch.full_like(replace_at, prev), top_chunk_idx.gather(-1, replace_at)),
                )

                gk_h = self._repeat_kv_heads(gk, rep)  # [B, H, n, M, Dh]
                gv_h = self._repeat_kv_heads(gv, rep)
                cand_gk = self._gather_chunks(gk_h, top_chunk_idx).reshape(B, H, Kc * M, Dh)
                cand_gv = self._gather_chunks(gv_h, top_chunk_idx).reshape(B, H, Kc * M, Dh)
                sc_A = torch.einsum("bhd,bhrd->bhr", qi, cand_gk) * scale  # [B, H, Kc*M]
                scores.append(sc_A)
                values.append(cand_gv)

                Tg = Kc * M
                Kg_req = min(self.topk_groups // 2, Tg) if self.topk_groups > 0 else 0
                if Kg_req > 0:
                    _, top_group_idx = torch.topk(sc_A, Kg_req, dim=-1, sorted=False)  # [B, H, Kg_req]
                    parent = top_group_idx // M
                    src_chunk = top_chunk_idx.gather(-1, parent)  # [B, H, Kg_req]
                    src_grp = top_group_idx - parent * M
                    base = src_chunk * C + src_grp * gs  # [B, H, Kg_req]
                    tok_pos = (base.unsqueeze(-1) + torch.arange(gs, device=device)).reshape(B, H, Kg_req * gs)
                    opened_k = self._gather_tokens(krope_full, tok_pos, rep)  # [B, H, Kg_req*gs, Dh]
                    opened_v = self._gather_tokens(v_full, tok_pos, rep)
                    scores.append(torch.einsum("bhd,bhrd->bhr", qi, opened_k) * scale)
                    values.append(opened_v)

            sc = torch.cat(scores, dim=-1)               # [B, H, R]
            val = torch.cat(values, dim=-2)              # [B, H, R, Dh]
            probs = torch.softmax(sc.float(), dim=-1).to(dtype)
            outs.append(torch.einsum("bhr,bhrd->bhd", probs, val))  # [B, H, Dh]

            # ---- Close the chunk when its last token is processed ----
            if c == C - 1:
                full_raw = kraw_buf.reshape(B, KVH, M, gs, Dh)
                full_rope = act_krope.reshape(B, KVH, M, gs, Dh)
                full_v = act_v.reshape(B, KVH, M, gs, Dh)
                g_anchor = n * C + ar_m * gs + (gs - 1) // 2
                gk_n = self._rope_summary_at(full_raw, full_rope, 3, g_anchor, self.group_kv_scale).unsqueeze(2)
                gv_n = (full_v.sum(dim=3) * self.group_kv_scale).unsqueeze(2)
                c_anchor = torch.tensor([n * C + (C - 1) // 2], device=device)
                ck_n = self._rope_summary_at(
                    kraw_buf.reshape(B, KVH, 1, C, Dh),
                    act_krope.reshape(B, KVH, 1, C, Dh),
                    3, c_anchor, 1.0,
                )
                ck = ck_n if ck is None else torch.cat([ck, ck_n], dim=2)
                gk = gk_n if gk is None else torch.cat([gk, gk_n], dim=2)
                gv = gv_n if gv is None else torch.cat([gv, gv_n], dim=2)
                kraw_buf = None

        state["kraw"], state["ck"], state["gk"], state["gv"] = kraw_buf, ck, gk, gv

        out_seq = torch.stack(outs, dim=2)  # [B, H, L, Dh]
        out = self.o_proj(out_seq.transpose(1, 2).contiguous().reshape(B, L, H * Dh))
        return out, {}

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

    def _rope_summary_at(
        self,
        raw: torch.Tensor,
        rope: torch.Tensor,
        reduce_dim: int,
        anchor_pos: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Mixed-RoPE summary for fully-valid chunks/groups (no token mask).

        Equivalent to ``_rope_summary`` with an all-ones mask, but the anchor cos/sin
        are generated from absolute positions via ``self.theta`` instead of being
        indexed out of a host cos/sin table.  This is what the KV cache needs, because
        during decoding the provided cos/sin only cover the new token positions, not
        the (past) anchor positions.
        """
        raw_sum = raw.sum(dim=reduce_dim) * scale
        tokenwise = rope.sum(dim=reduce_dim) * scale
        anchor_cos, anchor_sin = self._rotary_for_positions(anchor_pos, raw_sum)
        endpoint = self._apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=endpoint)

    def _rotary_for_positions(self, pos: torch.Tensor, like: torch.Tensor) -> RotaryData:
        """cos/sin for absolute integer positions ``pos``, broadcastable against ``like``.

        ``like`` has shape ``[B, KVH, *pos.shape, Dh]``; the returned tensors are shaped
        ``[1, 1, *pos.shape, Dh]``.  Matches Qwen RoPE exactly when ``self.theta`` equals
        the model's ``rope_theta``.
        """
        half = self.head_dim // 2
        device = like.device
        inv_freq = 1.0 / (self.theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq  # [*pos, half]
        emb = torch.cat((freqs, freqs), dim=-1)                                       # [*pos, Dh]
        view = (1, 1) + tuple(pos.shape) + (self.head_dim,)
        return emb.cos().reshape(view), emb.sin().reshape(view)

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

    def _project_qkv_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to q (all heads) and k/v at kv-head granularity, pre-RoPE / post-norm.

        Returns q [B, H, S, Dh], k [B, KVH, S, Dh], v [B, KVH, S, Dh].  Keeping k/v at
        kv-head granularity is what the original Qwen3 cache stores, so the KV cache
        stays GQA-compressed.
        """
        B, S, _ = x.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(B, S, H, Dh)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(B, S, KVH, Dh)).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, KVH, Dh).transpose(1, 2)
        return q, k, v

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project_qkv_kv(x)
        rep = self.nhead // self.kv_heads
        return q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

    def _normalize_rotary(self, rotary_data: RotaryData, x: torch.Tensor) -> RotaryData:
        cos, sin = rotary_data
        B, S = x.shape[:2]
        device, dtype = x.device, torch.float32

        def fix(t: torch.Tensor, name: str) -> torch.Tensor:
            t = t.to(device=device, dtype=dtype)
            if t.dim() == 2:      # [S, D]
                t = t.unsqueeze(0).unsqueeze(1)
            elif t.dim() == 3:    # [B, S, D]
                t = t.unsqueeze(1)
            elif t.dim() == 4 and t.shape[1] == 1:
                pass              # [B, 1, S, D]
            else:
                raise ValueError(f"{name} must have shape [S,D], [B,S,D], or [B,1,S,D]")
            if t.shape[-1] != self.head_dim:
                raise ValueError(f"{name} last dim must be head_dim={self.head_dim}, got {t.shape[-1]}")
            if t.shape[-2] < S:
                raise ValueError(f"{name} has only {t.shape[-2]} positions for S={S}")
            if t.shape[-2] != S:
                t = t[:, :, -S:, :]
            if t.shape[0] == 1 and B != 1:
                t = t.expand(B, -1, -1, -1)
            if t.shape[0] != B:
                raise ValueError(f"{name} batch dim must be 1 or B={B}, got {t.shape[0]}")
            return t

        return fix(cos, "cos"), fix(sin, "sin")

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
    def _gather_rotary(cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> RotaryData:
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

    def _get_rotary(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> RotaryData:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
