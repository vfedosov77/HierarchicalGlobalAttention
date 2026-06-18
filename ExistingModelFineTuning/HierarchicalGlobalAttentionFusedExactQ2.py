"""Qwen3-0.6B fused wrapper for hierarchical attention with incremental KV-cache fallback.

The fused path is used only for no-cache CUDA fp32 training/prefill.  Generation
with a Hugging Face Cache delegates to HierarchicalGlobalAttentionExactQ, which
uses compact cached KV plus past_key_value._hga[layer_idx] route state.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch


from ExistingModelFineTuning.HierarchicalGlobalAttentionExactQ2 import GlobalAttention as _RefGlobalAttention


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - allows importing the reference fallback without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_QWEN3_06B_CACHE_IMPL_VERSION = "qwen3-06b-cache-hga-state-fastchunks-2026-06-18-v2"

if _TRITON_AVAILABLE:
    LOG2E = tl.constexpr(1.4426950408889634)
    BIG = tl.constexpr(1 << 20)  # "never visible" threshold sentinel; fits int32
    BIG_INT = 1 << 20            # same value for host-side PyTorch code


    # ---------------------------------------------------------------------------
    # Triton kernels
    # ---------------------------------------------------------------------------

    @triton.jit
    def _dot(a, b, IEEE: tl.constexpr):
        if IEEE:
            return tl.dot(a, b, input_precision="ieee")
        else:
            return tl.dot(a, b)


    @triton.jit
    def _upd(s, vv, m_i, l_i, acc, IEEE: tl.constexpr):
        # one online-softmax accumulation step; s is already masked with -inf
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2((m_i - m_new) * LOG2E)
        p = tl.exp2((s - m_new[:, None]) * LOG2E)
        l_new = l_i * alpha + tl.sum(p, 1)
        acc_new = acc * alpha[:, None] + _dot(p, vv, IEEE)
        return m_new, l_new, acc_new


    @triton.jit
    def _ha_fwd_kernel(
        Q, K, V, GK, GV,
        IDXA, THRA, POSD, THRD,
        OUT, LSE,
        S, N, KC, KG, scale, S_pad, NM,
        C: tl.constexpr, M: tl.constexpr, GS: tl.constexpr, D: tl.constexpr,
        BR: tl.constexpr, IEEE: tl.constexpr,
    ):
        n = tl.program_id(0)
        bh = tl.program_id(1)
        # torch.compile's triton wrapper passes Python floats as fp64; keep fp32 math
        scale = tl.cast(scale, tl.float32)
        offs_c = tl.arange(0, C)
        offs_d = tl.arange(0, D)
        row0 = n * C
        qkv_base = bh * S_pad * D
        g_base = bh * NM * D

        q = tl.load(Q + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        qs = q * scale

        m_i = tl.full([C], -1.0e38, tl.float32)
        l_i = tl.zeros([C], tl.float32)
        acc = tl.zeros([C, D], tl.float32)

        # ---- segment C: current-chunk exact tokens (causal) ----
        kk = tl.load(K + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        vv = tl.load(V + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        s = _dot(qs, tl.trans(kk), IEEE)
        msk = (offs_c[None, :] <= offs_c[:, None]) & ((row0 + offs_c)[None, :] < S)
        s = tl.where(msk, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, IEEE)

        # ---- segment B: current-chunk group summaries ----
        offs_m = tl.arange(0, M)
        gk_own = tl.load(GK + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :])
        gv_own = tl.load(GV + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :])
        s = _dot(qs, tl.trans(gk_own), IEEE)
        complete = (row0 + (offs_m + 1) * GS) <= S
        msk = complete[None, :] & ((offs_m * GS + GS - 1)[None, :] <= offs_c[:, None])
        s = tl.where(msk, s, float("-inf"))
        m_i, l_i, acc = _upd(s, gv_own, m_i, l_i, acc, IEEE)

        # ---- segment A: candidate previous-chunk group summaries ----
        Ra = KC * M
        ia_base = (bh * N + n) * KC
        for t in range(0, tl.cdiv(Ra, BR)):
            r = t * BR + tl.arange(0, BR)
            rm = r < Ra
            kc_i = r // M
            m_loc = r % M
            j = tl.load(IDXA + ia_base + kc_i, mask=rm, other=0)
            thr = tl.load(THRA + ia_base + kc_i, mask=rm, other=BIG)
            grow = j * M + m_loc
            ka = tl.load(GK + g_base + grow[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            va = tl.load(GV + g_base + grow[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            sa = _dot(qs, tl.trans(ka), IEEE)
            mska = rm[None, :] & (offs_c[:, None] >= thr[None, :])
            sa = tl.where(mska, sa, float("-inf"))
            m_i, l_i, acc = _upd(sa, va, m_i, l_i, acc, IEEE)

        # ---- segment D: opened exact tokens ----
        Rd = KG * GS
        id_base = (bh * N + n) * KG
        for t in range(0, tl.cdiv(Rd, BR)):
            r = t * BR + tl.arange(0, BR)
            rm = r < Rd
            kg_i = r // GS
            off = r % GS
            p0 = tl.load(POSD + id_base + kg_i, mask=rm, other=0)
            thr = tl.load(THRD + id_base + kg_i, mask=rm, other=BIG)
            pos = p0 + off
            ko = tl.load(K + qkv_base + pos[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            vo = tl.load(V + qkv_base + pos[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            sd = _dot(qs, tl.trans(ko), IEEE)
            mskd = (rm & (pos < S))[None, :] & (offs_c[:, None] >= thr[None, :])
            sd = tl.where(mskd, sd, float("-inf"))
            m_i, l_i, acc = _upd(sd, vo, m_i, l_i, acc, IEEE)

        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / l_safe[:, None]
        tl.store(OUT + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :], o)
        lse = m_i + tl.log(l_safe)  # padded rows: -1e38, treated as sentinel in bwd
        tl.store(LSE + bh * S_pad + row0 + offs_c, lse)


    @triton.jit
    def _ha_bwd_kernel(
        Q, K, V, GK, GV,
        IDXA, THRA, POSD, THRD,
        DO, LSE, DELTA,
        DQ, DK, DV, DGK, DGV,
        S, N, KC, KG, scale, S_pad, NM,
        C: tl.constexpr, M: tl.constexpr, GS: tl.constexpr, D: tl.constexpr,
        BR: tl.constexpr, IEEE: tl.constexpr,
    ):
        n = tl.program_id(0)
        bh = tl.program_id(1)
        # torch.compile's triton wrapper passes Python floats as fp64; keep fp32 math
        scale = tl.cast(scale, tl.float32)
        offs_c = tl.arange(0, C)
        offs_d = tl.arange(0, D)
        row0 = n * C
        qkv_base = bh * S_pad * D
        g_base = bh * NM * D

        q = tl.load(Q + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        do = tl.load(DO + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        lse_r = tl.load(LSE + bh * S_pad + row0 + offs_c)
        # padded query rows have sentinel lse (-1e38): replace with +1e38 so p == 0
        lse_r = tl.where(lse_r > -1.0e37, lse_r, 1.0e38)
        dlt = tl.load(DELTA + bh * S_pad + row0 + offs_c)
        qs = q * scale
        # one-time transposes; per-tile dK/dV are produced directly in [D, R]
        # layout (dK_tile^T = q^T @ dS) so no per-tile register transposes occur.
        qt = tl.trans(q)
        dot_ = tl.trans(do)
        dq_acc = tl.zeros([C, D], tl.float32)

        # ---- segment C ----
        kk = tl.load(K + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
        kt = tl.load(K + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None])
        vt = tl.load(V + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None])
        s = _dot(qs, kt, IEEE)
        msk = (offs_c[None, :] <= offs_c[:, None]) & ((row0 + offs_c)[None, :] < S)
        s = tl.where(msk, s, float("-inf"))
        p = tl.exp2((s - lse_r[:, None]) * LOG2E)
        dp = _dot(do, vt, IEEE)
        ds = p * (dp - dlt[:, None]) * scale
        dq_acc += _dot(ds, kk, IEEE)
        tl.atomic_add(DK + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None],
                      _dot(qt, ds, IEEE))
        tl.atomic_add(DV + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None],
                      _dot(dot_, p, IEEE))

        # ---- segment B ----
        offs_m = tl.arange(0, M)
        gk_own = tl.load(GK + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :])
        gkt = tl.load(GK + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None])
        gvt = tl.load(GV + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None])
        sb = _dot(qs, gkt, IEEE)
        complete = (row0 + (offs_m + 1) * GS) <= S
        mskb = complete[None, :] & ((offs_m * GS + GS - 1)[None, :] <= offs_c[:, None])
        sb = tl.where(mskb, sb, float("-inf"))
        pb = tl.exp2((sb - lse_r[:, None]) * LOG2E)
        dpb = _dot(do, gvt, IEEE)
        dsb = pb * (dpb - dlt[:, None]) * scale
        dq_acc += _dot(dsb, gk_own, IEEE)
        tl.atomic_add(DGK + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None],
                      _dot(qt, dsb, IEEE))
        tl.atomic_add(DGV + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None],
                      _dot(dot_, pb, IEEE))

        # ---- segment A ----
        Ra = KC * M
        ia_base = (bh * N + n) * KC
        for t in range(0, tl.cdiv(Ra, BR)):
            r = t * BR + tl.arange(0, BR)
            rm = r < Ra
            kc_i = r // M
            m_loc = r % M
            j = tl.load(IDXA + ia_base + kc_i, mask=rm, other=0)
            thr = tl.load(THRA + ia_base + kc_i, mask=rm, other=BIG)
            grow = j * M + m_loc
            ka = tl.load(GK + g_base + grow[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            kat = tl.load(GK + g_base + grow[None, :] * D + offs_d[:, None], mask=rm[None, :], other=0.0)
            vat = tl.load(GV + g_base + grow[None, :] * D + offs_d[:, None], mask=rm[None, :], other=0.0)
            sa = _dot(qs, kat, IEEE)
            mska = rm[None, :] & (offs_c[:, None] >= thr[None, :])
            sa = tl.where(mska, sa, float("-inf"))
            pa = tl.exp2((sa - lse_r[:, None]) * LOG2E)
            dpa = _dot(do, vat, IEEE)
            dsa = pa * (dpa - dlt[:, None]) * scale
            dq_acc += _dot(dsa, ka, IEEE)
            tl.atomic_add(DGK + g_base + grow[None, :] * D + offs_d[:, None],
                          _dot(qt, dsa, IEEE), mask=rm[None, :])
            tl.atomic_add(DGV + g_base + grow[None, :] * D + offs_d[:, None],
                          _dot(dot_, pa, IEEE), mask=rm[None, :])

        # ---- segment D ----
        Rd = KG * GS
        id_base = (bh * N + n) * KG
        for t in range(0, tl.cdiv(Rd, BR)):
            r = t * BR + tl.arange(0, BR)
            rm = r < Rd
            kg_i = r // GS
            off = r % GS
            p0 = tl.load(POSD + id_base + kg_i, mask=rm, other=0)
            thr = tl.load(THRD + id_base + kg_i, mask=rm, other=BIG)
            pos = p0 + off
            ko = tl.load(K + qkv_base + pos[:, None] * D + offs_d[None, :], mask=rm[:, None], other=0.0)
            kot = tl.load(K + qkv_base + pos[None, :] * D + offs_d[:, None], mask=rm[None, :], other=0.0)
            vot = tl.load(V + qkv_base + pos[None, :] * D + offs_d[:, None], mask=rm[None, :], other=0.0)
            sd = _dot(qs, kot, IEEE)
            mskd = (rm & (pos < S))[None, :] & (offs_c[:, None] >= thr[None, :])
            sd = tl.where(mskd, sd, float("-inf"))
            pd = tl.exp2((sd - lse_r[:, None]) * LOG2E)
            dpd = _dot(do, vot, IEEE)
            dsd = pd * (dpd - dlt[:, None]) * scale
            dq_acc += _dot(dsd, ko, IEEE)
            tl.atomic_add(DK + qkv_base + pos[None, :] * D + offs_d[:, None],
                          _dot(qt, dsd, IEEE), mask=rm[None, :])
            tl.atomic_add(DV + qkv_base + pos[None, :] * D + offs_d[:, None],
                          _dot(dot_, pd, IEEE), mask=rm[None, :])

        tl.store(DQ + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :], dq_acc)


    # ---------------------------------------------------------------------------
    # Autograd wrapper
    # ---------------------------------------------------------------------------

    class _HAFusedFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, gk, gv, idxA, thrA, posD, thrD, S, scale, C, M, gs, ieee):
            B, H, S_pad, D = q.shape
            N = S_pad // C
            KC = idxA.shape[-1]
            KG = posD.shape[-1]
            out = torch.empty_like(q)
            lse = torch.empty(B * H * S_pad, device=q.device, dtype=torch.float32)
            grid = (N, B * H)
            _ha_fwd_kernel[grid](
                q, k, v, gk, gv, idxA, thrA, posD, thrD, out, lse,
                S, N, KC, KG, scale, S_pad, N * M,
                C=C, M=M, GS=gs, D=D, BR=64, IEEE=ieee,
                num_warps=4,
            )
            ctx.save_for_backward(q, k, v, gk, gv, idxA, thrA, posD, thrD, out, lse)
            ctx.cfg = (S, scale, C, M, gs, ieee)
            return out

        @staticmethod
        def backward(ctx, dout):
            q, k, v, gk, gv, idxA, thrA, posD, thrD, out, lse = ctx.saved_tensors
            S, scale, C, M, gs, ieee = ctx.cfg
            B, H, S_pad, D = q.shape
            N = S_pad // C
            KC = idxA.shape[-1]
            KG = posD.shape[-1]
            dout = dout.contiguous()
            delta = (out * dout).sum(dim=-1).reshape(-1)
            dq = torch.empty_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            dgk = torch.zeros_like(gk)
            dgv = torch.zeros_like(gv)
            grid = (N, B * H)
            _ha_bwd_kernel[grid](
                q, k, v, gk, gv, idxA, thrA, posD, thrD,
                dout, lse, delta,
                dq, dk, dv, dgk, dgv,
                S, N, KC, KG, scale, S_pad, N * M,
                C=C, M=M, GS=gs, D=D, BR=64, IEEE=ieee,
                num_warps=4, num_stages=1,
            )
            return (dq, dk, dv, dgk, dgv,
                    None, None, None, None, None, None, None, None, None, None)


    # ---------------------------------------------------------------------------
    # Module
    # ---------------------------------------------------------------------------

    class GlobalAttentionFused(_RefGlobalAttention):
        """Drop-in replacement for the reference ``GlobalAttention``.

        Same constructor, same parameters/state_dict, same forward contract.
        """

        fp32_precise: bool = False  # True -> IEEE fp32 tl.dot (for equivalence tests)

        @staticmethod
        @torch.no_grad()
        def _causal_rolling_sum_fast(x: torch.Tensor, window: int) -> torch.Tensor:
            # Same values as the reference _causal_rolling_sum, but the scan runs
            # along the innermost dim (the outer-dim scan kernel is ~10x slower).
            if window <= 1:
                return x
            csum = x.transpose(2, 3).contiguous().cumsum(-1).transpose(2, 3)
            out = csum.clone()
            if x.shape[2] > window:
                out[:, :, window:, :] = out[:, :, window:, :] - csum[:, :, :-window, :]
            return out

        @torch.no_grad()
        def _route_topk_requests_unsorted(self, scores: torch.Tensor, k: int):
            # Same request set as the reference _route_topk_requests; sorting is
            # skipped because every consumer (scatter-amax / scatter-amin) is
            # order-independent.
            num_routes = scores.shape[-1]
            if k >= num_routes:
                return self._all_route_indices(scores.shape[:-1], num_routes, scores.device), scores
            top_scores, top_idx = torch.topk(scores, k, dim=-1, sorted=False)
            return top_idx, top_scores

        def _fused_applicable(self, x: torch.Tensor) -> bool:
            C, M, D = self.chunk_size, self.groups_per_chunk, self.head_dim
            return (
                self.use_global
                and x.is_cuda
                and x.dtype == torch.float32
                and x.ndim == 3
                and x.shape[1] > 0
                and not self.return_router_stats
                and not (self.training and self.dropout_p > 0.0)
                and self.topk_chunks > 0
                and self.topk_groups > 0
                and C >= 16 and (C & (C - 1)) == 0
                and M >= 16 and (M & (M - 1)) == 0
                and D >= 16 and (D & (D - 1)) == 0
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kw: Any,
        ) -> Tuple[torch.Tensor, Optional[Any]]:
            if past_key_value is None:
                past_key_value = kw.get("past_key_values", None)
            if past_key_value is not None or not self._fused_applicable(hidden_states):
                self._last_path = "reference"
                return super().forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kw,
                )
            self._last_path = "fused"

            B, S, _ = hidden_states.shape
            H, Dh = self.nhead, self.head_dim
            C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
            device, dtype = hidden_states.device, hidden_states.dtype

            cos, sin = self._normalize_position_embeddings(position_embeddings, hidden_states)

            q_raw, k_raw_base, v_base = self._project_qkv_base(hidden_states)
            q = self._apply_rotary(q_raw.float(), cos, sin).to(q_raw.dtype)
            k_base = self._apply_rotary(k_raw_base.float(), cos, sin).to(k_raw_base.dtype)
            k_raw = self._repeat_kv(k_raw_base)
            k_rope = self._repeat_kv(k_base)
            v = self._repeat_kv(v_base)

            N = (S + C - 1) // C
            S_pad = N * C

            q_p = self._pad_seq(q, S_pad).contiguous()
            k_p = self._pad_seq(k_rope, S_pad).contiguous()
            v_p = self._pad_seq(v, S_pad).contiguous()
            k_raw_p = self._pad_seq(k_raw, S_pad)

            with torch.no_grad():
                valid_flat = torch.arange(S_pad, device=device) < S
                valid_chunks = valid_flat.view(N, C)                      # [N, C]
                chunk_len = valid_chunks.sum(dim=1)                       # [N]
                group_len = valid_chunks.view(N, M, gs).sum(dim=-1)       # [N, M]
                valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)    # [N, M]
                ar_n = torch.arange(N, device=device)
                ar_m = torch.arange(M, device=device)
                ar_c = torch.arange(C, device=device)
                chunk_start = ar_n * C
                chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)
                group_start = ar_n[:, None] * C + ar_m[None, :] * gs
                group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

            k_raw_chunks = k_raw_p.reshape(B, H, N, C, Dh)
            k_chunks = k_p.reshape(B, H, N, C, Dh)
            v_chunks = v_p.reshape(B, H, N, C, Dh)
            chunk_token_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
            group_token_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)

            k_raw_groups = k_raw_chunks.reshape(B, H, N, M, gs, Dh)
            k_rope_groups = k_chunks.reshape(B, H, N, M, gs, Dh)
            v_groups = v_chunks.reshape(B, H, N, M, gs, Dh)
            group_k = self._rope_summary(
                raw=k_raw_groups,
                rope=k_rope_groups,
                mask=group_token_mask,
                reduce_dim=4,
                anchor_pos=group_middle,
                cos=cos,
                sin=sin,
                scale=self.group_kv_scale,
            ).contiguous()
            group_v = ((v_groups * group_token_mask).sum(dim=4) * self.group_kv_scale).contiguous()

            # ---- routing: identical route set to the reference full path, all no-grad ----
            with torch.no_grad():
                neg_inf = -1.0e4
                query_valid = valid_chunks.view(1, 1, N, C, 1)

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
                route_q_chunks = q_p.reshape(B, H, N, C, Dh)

                # --- stage 1: previous-chunk selection ---
                Kc = min(self.topk_chunks, N)
                scores_roll = torch.einsum("bhncd,bhmd->bhncm", route_q_chunks, chunk_k) * self.scale
                prev_chunk_mask = ar_n[None, :] < ar_n[:, None]
                route_mask = prev_chunk_mask.view(1, 1, N, 1, N) & query_valid
                scores_roll.masked_fill_(~route_mask, neg_inf)
                req_idx, req_scores = self._route_topk_requests_unsorted(scores_roll, Kc)
                chunk_scores = self._max_route_scores_from_requests(req_idx, req_scores, N, neg_inf)
                top_chunk_scores, top_chunk_idx = self._topk_scores_indices(chunk_scores, Kc)
                query_chunk_idx = ar_n.view(1, 1, N, 1)
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

                finite_min = neg_inf / 2
                ar_c32 = ar_c.to(torch.int32)
                c_eff = torch.where(
                    req_scores > finite_min,
                    ar_c32.view(1, 1, 1, C, 1),
                    torch.tensor(BIG_INT, device=device, dtype=torch.int32),
                )
                route_first = torch.full((B * H * N, N), BIG_INT, device=device, dtype=torch.int32)
                route_first.scatter_reduce_(
                    1,
                    req_idx.reshape(B * H * N, -1),
                    c_eff.expand_as(req_idx).reshape(B * H * N, -1),
                    reduce="amin",
                    include_self=True,
                )
                thrA = route_first.view(B, H, N, N).gather(-1, top_chunk_idx).long()
                is_prev_slot = (top_chunk_idx == prev_chunk_idx) & has_prev
                thrA = torch.where(is_prev_slot, torch.zeros_like(thrA), thrA)
                thrA = torch.where(valid_prev, thrA, torch.full_like(thrA, BIG_INT))

                # --- stage 2: opened-group selection ---
                cand_k_flat = self._gather_chunks(group_k, top_chunk_idx).reshape(B, H, N, Kc * M, Dh)
                cand_group_valid = valid_groups[top_chunk_idx] & valid_prev.unsqueeze(-1)
                Tgrp = Kc * M

                gsr = torch.einsum("bhncd,bhnrd->bhncr", route_q_chunks, cand_k_flat) * self.scale
                visA = ar_c.view(1, 1, 1, C, 1) >= thrA.view(B, H, N, 1, Kc)
                gs_mask = (visA.unsqueeze(-1) & cand_group_valid.unsqueeze(3)).reshape(B, H, N, C, Tgrp) & query_valid
                gsr.masked_fill_(~gs_mask, neg_inf)

                Kg = min(self.topk_groups, Tgrp)
                Kg_req = min(self.topk_groups // 2, Tgrp)
                if Kg > 0 and Kg_req > 0:
                    g_req_idx, g_req_scores = self._route_topk_requests_unsorted(gsr, Kg_req)
                    group_scores = self._max_route_scores_from_requests(g_req_idx, g_req_scores, Tgrp, neg_inf)
                    _, top_group_idx = self._topk_scores_indices(group_scores, Kg)
                    c_eff_g = torch.where(
                        g_req_scores > finite_min,
                        ar_c32.view(1, 1, 1, C, 1),
                        torch.tensor(BIG_INT, device=device, dtype=torch.int32),
                    )
                    route_first_g = torch.full((B * H * N, Tgrp), BIG_INT, device=device, dtype=torch.int32)
                    route_first_g.scatter_reduce_(
                        1,
                        g_req_idx.reshape(B * H * N, -1),
                        c_eff_g.expand_as(g_req_idx).reshape(B * H * N, -1),
                        reduce="amin",
                        include_self=True,
                    )
                    thrD = route_first_g.view(B, H, N, Tgrp).gather(-1, top_group_idx).long()
                    parent = top_group_idx // M
                    src_chunk = top_chunk_idx.gather(-1, parent)
                    src_grp = top_group_idx - parent * M
                    posD = src_chunk * C + src_grp * gs
                    posD = torch.where(thrD < BIG_INT, posD, torch.zeros_like(posD))
                else:
                    thrD = torch.zeros(B, H, N, 0, device=device, dtype=torch.long)
                    posD = torch.zeros(B, H, N, 0, device=device, dtype=torch.long)

                idxA32 = top_chunk_idx.to(torch.int32).contiguous()
                thrA32 = thrA.to(torch.int32).contiguous()
                posD32 = posD.to(torch.int32).contiguous()
                thrD32 = thrD.to(torch.int32).contiguous()

            out_p = _HAFusedFn.apply(
                q_p, k_p, v_p, group_k, group_v,
                idxA32, thrA32, posD32, thrD32,
                S, self.scale, C, M, gs, self.fp32_precise,
            )

            out = self.o_proj(
                out_p[:, :, :S, :].transpose(1, 2).contiguous().reshape(B, S, H * Dh)
            )
            return out, None

else:
    class GlobalAttentionFused(_RefGlobalAttention):
        pass


# Convenience alias: swap the import and keep the class name.
GlobalAttention = GlobalAttentionFused
