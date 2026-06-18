"""Triton-fused implementation of the Exact-Q hierarchical attention.

This is the optimized fused variant updated to match
``HierarchicalGlobalAttentionExactQ.GlobalAttention`` for the CUDA/fp32
``use_global=True`` path:

* routing uses the exact RoPE-applied query, not causal rolling sums;
* chunk/group router summaries use the mixed RoPE policy from the ExactQ
  reference: high-frequency pairs are summed tokenwise after RoPE, while
  low-frequency pairs use an anchor-rotated raw sum;
* stage-1 and stage-2 request counts, forced previous-chunk visibility,
  middle anchors, and group value scaling match the ExactQ reference;
* the differentiable joint softmax/value aggregation remains fused in Triton,
  so no route-sized ``[B,H,N,C,routes]`` probability tensor is saved.

Fallback to the ExactQ reference implementation when the fused path does not
apply: ``use_global=False``, CPU tensors, non-fp32 dtype, attention-prob dropout
active in training mode, ``return_router_stats``, unsupported chunk/head
geometry, or a large non-power-of-two number of groups per chunk.  Small group
counts such as ``groups_per_chunk == 4`` stay on the fused path and use exact
manual score/update code instead of padding to fake 16-wide group tiles.  This
version also avoids Triton loop-carried type conflicts in that small-M path.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

import ExistingModelFineTuning.torch_inductor_patch as path
path.apply()
from ExistingModelFineTuning.HierarchicalGlobalAttentionExactQ import GlobalAttention as _RefGlobalAttention

RotaryData = Tuple[torch.Tensor, torch.Tensor]

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
def _upd_one(s, v, m_i, l_i, acc):
    # Online-softmax update for one route column.  This avoids padding tiny
    # current-group blocks (for example M=4) up to tl.dot's preferred width.
    m_new = tl.maximum(m_i, s)
    alpha = tl.exp2((m_i - m_new) * LOG2E)
    p = tl.exp2((s - m_new) * LOG2E)
    l_new = l_i * alpha + p
    acc_new = acc * alpha[:, None] + p[:, None] * v[None, :]
    return m_new, l_new, acc_new


@triton.jit
def _ha_fwd_kernel(
    Q, K, V, GK, GV,
    IDXA, THRA, POSD, THRD,
    OUT, LSE,
    S, N, KC, KG, scale, S_pad, NM,
    C: tl.constexpr, M: tl.constexpr, GS: tl.constexpr, D: tl.constexpr,
    BRA: tl.constexpr, BRD: tl.constexpr, IEEE: tl.constexpr,
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
    if M < 16:
        # Exact small-M path.  Triton matmul tiles prefer at least 16 columns;
        # for M=4 padding would do 4x more current-summary work than needed.
        for m in range(0, M):
            gk_own_m = tl.load(GK + g_base + (n * M + m) * D + offs_d)
            gv_own_m = tl.load(GV + g_base + (n * M + m) * D + offs_d)
            score_m = tl.sum(qs * gk_own_m[None, :], axis=1)
            complete_m = (row0 + (m + 1) * GS) <= S
            mask_m = complete_m & ((m * GS + GS - 1) <= offs_c)
            score_m = tl.where(mask_m, score_m, float("-inf"))
            m_i, l_i, acc = _upd_one(score_m, gv_own_m, m_i, l_i, acc)
    else:
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
    for ta in range(0, tl.cdiv(Ra, BRA)):
        ra = ta * BRA + tl.arange(0, BRA)
        rma = ra < Ra
        kc_i_a = ra // M
        m_loc_a = ra % M
        j_a = tl.load(IDXA + ia_base + kc_i_a, mask=rma, other=0)
        thr_a = tl.load(THRA + ia_base + kc_i_a, mask=rma, other=BIG)
        grow_a = j_a * M + m_loc_a
        ka = tl.load(GK + g_base + grow_a[:, None] * D + offs_d[None, :], mask=rma[:, None], other=0.0)
        va = tl.load(GV + g_base + grow_a[:, None] * D + offs_d[None, :], mask=rma[:, None], other=0.0)
        sa = _dot(qs, tl.trans(ka), IEEE)
        mska = rma[None, :] & (offs_c[:, None] >= thr_a[None, :])
        sa = tl.where(mska, sa, float("-inf"))
        m_i, l_i, acc = _upd(sa, va, m_i, l_i, acc, IEEE)

    # ---- segment D: opened exact tokens ----
    Rd = KG * GS
    id_base = (bh * N + n) * KG
    for td in range(0, tl.cdiv(Rd, BRD)):
        rd = td * BRD + tl.arange(0, BRD)
        rmd = rd < Rd
        kg_i_d = rd // GS
        off_d = rd % GS
        p0_d = tl.load(POSD + id_base + kg_i_d, mask=rmd, other=0)
        thr_d = tl.load(THRD + id_base + kg_i_d, mask=rmd, other=BIG)
        pos_d = p0_d + off_d
        ko = tl.load(K + qkv_base + pos_d[:, None] * D + offs_d[None, :], mask=rmd[:, None], other=0.0)
        vo = tl.load(V + qkv_base + pos_d[:, None] * D + offs_d[None, :], mask=rmd[:, None], other=0.0)
        sd = _dot(qs, tl.trans(ko), IEEE)
        mskd = (rmd & (pos_d < S))[None, :] & (offs_c[:, None] >= thr_d[None, :])
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
    BRA: tl.constexpr, BRD: tl.constexpr, IEEE: tl.constexpr,
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
    if M < 16:
        # Exact small-M backward, paired with the forward's no-padding path.
        for mb in range(0, M):
            gk_own_mb = tl.load(GK + g_base + (n * M + mb) * D + offs_d)
            gv_own_mb = tl.load(GV + g_base + (n * M + mb) * D + offs_d)
            sb_m = tl.sum(qs * gk_own_mb[None, :], axis=1)
            complete_mb = (row0 + (mb + 1) * GS) <= S
            mskb_m = complete_mb & ((mb * GS + GS - 1) <= offs_c)
            sb_m = tl.where(mskb_m, sb_m, float("-inf"))
            pb_m = tl.exp2((sb_m - lse_r) * LOG2E)
            dpb_m = tl.sum(do * gv_own_mb[None, :], axis=1)
            dsb_m = pb_m * (dpb_m - dlt) * scale
            dq_acc += dsb_m[:, None] * gk_own_mb[None, :]
            tl.atomic_add(
                DGK + g_base + (n * M + mb) * D + offs_d,
                tl.sum(q * dsb_m[:, None], axis=0),
            )
            tl.atomic_add(
                DGV + g_base + (n * M + mb) * D + offs_d,
                tl.sum(do * pb_m[:, None], axis=0),
            )
    else:
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
    for ta in range(0, tl.cdiv(Ra, BRA)):
        ra = ta * BRA + tl.arange(0, BRA)
        rma = ra < Ra
        kc_i_a = ra // M
        m_loc_a = ra % M
        j_a = tl.load(IDXA + ia_base + kc_i_a, mask=rma, other=0)
        thr_a = tl.load(THRA + ia_base + kc_i_a, mask=rma, other=BIG)
        grow_a = j_a * M + m_loc_a
        ka = tl.load(GK + g_base + grow_a[:, None] * D + offs_d[None, :], mask=rma[:, None], other=0.0)
        kat = tl.load(GK + g_base + grow_a[None, :] * D + offs_d[:, None], mask=rma[None, :], other=0.0)
        vat = tl.load(GV + g_base + grow_a[None, :] * D + offs_d[:, None], mask=rma[None, :], other=0.0)
        sa = _dot(qs, kat, IEEE)
        mska = rma[None, :] & (offs_c[:, None] >= thr_a[None, :])
        sa = tl.where(mska, sa, float("-inf"))
        pa = tl.exp2((sa - lse_r[:, None]) * LOG2E)
        dpa = _dot(do, vat, IEEE)
        dsa = pa * (dpa - dlt[:, None]) * scale
        dq_acc += _dot(dsa, ka, IEEE)
        tl.atomic_add(DGK + g_base + grow_a[None, :] * D + offs_d[:, None],
                      _dot(qt, dsa, IEEE), mask=rma[None, :])
        tl.atomic_add(DGV + g_base + grow_a[None, :] * D + offs_d[:, None],
                      _dot(dot_, pa, IEEE), mask=rma[None, :])

    # ---- segment D ----
    Rd = KG * GS
    id_base = (bh * N + n) * KG
    for td in range(0, tl.cdiv(Rd, BRD)):
        rd = td * BRD + tl.arange(0, BRD)
        rmd = rd < Rd
        kg_i_d = rd // GS
        off_d = rd % GS
        p0_d = tl.load(POSD + id_base + kg_i_d, mask=rmd, other=0)
        thr_d = tl.load(THRD + id_base + kg_i_d, mask=rmd, other=BIG)
        pos_d = p0_d + off_d
        ko = tl.load(K + qkv_base + pos_d[:, None] * D + offs_d[None, :], mask=rmd[:, None], other=0.0)
        kot = tl.load(K + qkv_base + pos_d[None, :] * D + offs_d[:, None], mask=rmd[None, :], other=0.0)
        vot = tl.load(V + qkv_base + pos_d[None, :] * D + offs_d[:, None], mask=rmd[None, :], other=0.0)
        sd = _dot(qs, kot, IEEE)
        mskd = (rmd & (pos_d < S))[None, :] & (offs_c[:, None] >= thr_d[None, :])
        sd = tl.where(mskd, sd, float("-inf"))
        pd = tl.exp2((sd - lse_r[:, None]) * LOG2E)
        dpd = _dot(do, vot, IEEE)
        dsd = pd * (dpd - dlt[:, None]) * scale
        dq_acc += _dot(dsd, ko, IEEE)
        tl.atomic_add(DK + qkv_base + pos_d[None, :] * D + offs_d[:, None],
                      _dot(qt, dsd, IEEE), mask=rmd[None, :])
        tl.atomic_add(DV + qkv_base + pos_d[None, :] * D + offs_d[:, None],
                      _dot(dot_, pd, IEEE), mask=rmd[None, :])

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
        # For small group counts, previous-summary routes are few (Kc*M), so
        # a 32-wide tile reduces inactive lanes.  Opened-token routes usually
        # dominate, so keep the larger 64-wide tile there for throughput.
        br_a = 32 if M < 16 else 64
        br_d = 64
        _ha_fwd_kernel[grid](
            q, k, v, gk, gv, idxA, thrA, posD, thrD, out, lse,
            S, N, KC, KG, scale, S_pad, N * M,
            C=C, M=M, GS=gs, D=D, BRA=br_a, BRD=br_d, IEEE=ieee,
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
        br_a = 32 if M < 16 else 64
        br_d = 64
        _ha_bwd_kernel[grid](
            q, k, v, gk, gv, idxA, thrA, posD, thrD,
            dout, lse, delta,
            dq, dk, dv, dgk, dgv,
            S, N, KC, KG, scale, S_pad, N * M,
            C=C, M=M, GS=gs, D=D, BRA=br_a, BRD=br_d, IEEE=ieee,
            num_warps=4, num_stages=1,
        )
        return (dq, dk, dv, dgk, dgv,
                None, None, None, None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class GlobalAttentionFused(_RefGlobalAttention):
    """Drop-in optimized variant of the ExactQ ``GlobalAttention``.

    Constructor, parameter names, and fallback forward contract are inherited
    from ``HierarchicalGlobalAttentionExactQ.GlobalAttention``.  The fused path
    is selected only for the CUDA/fp32/no-dropout case supported by the Triton
    kernel; otherwise ``super().forward`` is used.
    """

    fp32_precise: bool = False  # True -> IEEE fp32 tl.dot (for equivalence tests)

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

    def _rope_summary_no_mask(
        self,
        raw: torch.Tensor,
        rope: torch.Tensor,
        reduce_dim: int,
        anchor_pos: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Fast path for fully valid chunks/groups.

        Equivalent to the ExactQ ``_rope_summary(..., mask=ones)`` but avoids
        materializing/broadcasting an all-ones token mask on common aligned
        sequence lengths.
        """
        raw_sum = raw.sum(dim=reduce_dim) * scale
        anchor_cos, anchor_sin = self._gather_rotary(cos, sin, anchor_pos)
        anchor = self._apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
        tokenwise = rope.sum(dim=reduce_dim) * scale
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=anchor)

    def _fused_applicable(self, x: torch.Tensor) -> bool:
        C, M, D = self.chunk_size, self.groups_per_chunk, self.head_dim
        return (
            self.use_global
            and hasattr(self, "_rope_summary")
            and x.is_cuda
            and x.dtype == torch.float32
            and x.ndim == 3
            and x.shape[1] > 0
            and not self.return_router_stats
            and not (self.training and self.dropout_p > 0.0)
            and self.topk_chunks > 0
            and self.topk_groups > 0
            and C >= 16 and (C & (C - 1)) == 0
            # M<16 uses a no-padding scalar-column path for current groups;
            # M>=16 uses tl.dot and therefore remains power-of-two only.
            and M > 0 and (M < 16 or (M & (M - 1)) == 0)
            and D >= 16 and (D & (D - 1)) == 0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = hidden_states
        if not self._fused_applicable(x):
            self._last_path = "reference"
            return super().forward(x, rotary_data=position_embeddings, **kw)
        
        self._last_path = "fused"

        B, S, _ = x.shape
        H, Dh = self.nhead, self.head_dim
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = x.device, x.dtype

        if position_embeddings is None:
            cos, sin = self._get_rotary(S, device, torch.float32)
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos, sin = self._normalize_rotary(position_embeddings, x)

        q_raw, k_raw, v = self._project_qkv(x)
        q = self._apply_rotary(q_raw.float(), cos, sin).to(dtype=q_raw.dtype)
        k_rope = self._apply_rotary(k_raw.float(), cos, sin).to(dtype=k_raw.dtype)

        N = (S + C - 1) // C
        S_pad = N * C

        q_p = self._pad_seq(q, S_pad).contiguous()
        k_p = self._pad_seq(k_rope, S_pad).contiguous()
        v_p = self._pad_seq(v, S_pad).contiguous()
        k_raw_p = self._pad_seq(k_raw, S_pad).contiguous()

        valid_flat = torch.arange(S_pad, device=device) < S
        valid_chunks = valid_flat.view(N, C)                         # [N, C]
        chunk_len = valid_chunks.sum(dim=1)                          # [N]
        group_len = valid_chunks.view(N, M, gs).sum(dim=-1)           # [N, M]
        valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)        # [N, M]

        ar_n = torch.arange(N, device=device)
        ar_m = torch.arange(M, device=device)
        ar_c = torch.arange(C, device=device)
        chunk_start = ar_n * C
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)
        group_start = ar_n[:, None] * C + ar_m[None, :] * gs
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

        q_chunks = q_p.reshape(B, H, N, C, Dh)
        k_chunks = k_p.reshape(B, H, N, C, Dh)
        v_chunks = v_p.reshape(B, H, N, C, Dh)
        k_raw_chunks = k_raw_p.reshape(B, H, N, C, Dh)

        k_raw_groups = k_raw_chunks.reshape(B, H, N, M, gs, Dh)
        k_rope_groups = k_chunks.reshape(B, H, N, M, gs, Dh)
        v_groups = v_chunks.reshape(B, H, N, M, gs, Dh)

        # ---- differentiable group summaries (ExactQ mixed-RoPE math) ----
        # High-frequency RoPE pairs are tokenwise-summed; low-frequency pairs
        # use an anchor-rotated raw sum.  group_v keeps the ExactQ scaling.
        if S_pad == S:
            group_k = self._rope_summary_no_mask(
                raw=k_raw_groups,
                rope=k_rope_groups,
                reduce_dim=4,
                anchor_pos=group_middle,
                cos=cos,
                sin=sin,
                scale=self.group_kv_scale,
            )
            group_v = v_groups.sum(dim=4) * self.group_kv_scale
        else:
            group_token_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)
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
            group_v = (v_groups * group_token_mask).sum(dim=4) * self.group_kv_scale
        group_k = group_k.contiguous()
        group_v = group_v.contiguous()

        # ---- routing: ExactQ reference ops, compressed to per-route thresholds ----
        with torch.no_grad():
            # Match ExactQ router mask sentinel.  The Triton kernel still uses -inf
            # internally for probability masks; exp(-1e4) is numerically zero
            # for all normal valid rows, while this value preserves ExactQ top-k
            # routing behavior.
            neg_inf = -1.0e4
            query_valid = valid_chunks.view(1, 1, N, C, 1)

            # Chunk summaries are router-only, so avoid saving their graph.
            if S_pad == S:
                chunk_k = self._rope_summary_no_mask(
                    raw=k_raw_chunks,
                    rope=k_chunks,
                    reduce_dim=3,
                    anchor_pos=chunk_middle,
                    cos=cos,
                    sin=sin,
                    scale=1.0,
                )
            else:
                chunk_token_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
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

            # --- stage 1: previous-chunk selection ---
            # New ExactQ behavior: route with the exact RoPE-applied query, not
            # a causal rolling sum.  Request count is Kc, then the immediately
            # previous chunk is forced visible from token 0 when needed.
            Kc = min(self.topk_chunks, N)
            scores_q = torch.einsum("bhncd,bhmd->bhncm", q_chunks, chunk_k) * self.scale
            prev_chunk_mask = ar_n[None, :] < ar_n[:, None]
            route_mask = prev_chunk_mask.view(1, 1, N, 1, N) & query_valid
            scores_q.masked_fill_(~route_mask, neg_inf)
            req_idx, req_scores = self._route_topk_requests_unsorted(scores_q, Kc)
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

            # Convert ExactQ's monotone cumsum visibility into one threshold per
            # selected route.  A route is visible from the first token that
            # requested it; the forced previous chunk is visible from token 0.
            # Use the boolean route mask, not a score cutoff, so extreme real
            # logits are never mistaken for masked routes.
            ar_c32 = ar_c.to(torch.int32)
            req_valid = (req_idx < ar_n.view(1, 1, N, 1, 1)) & query_valid
            c_eff = torch.where(
                req_valid,
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

            group_scores_q = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, cand_k_flat) * self.scale
            visA = ar_c.view(1, 1, 1, C, 1) >= thrA.view(B, H, N, 1, Kc)
            group_summary_mask = (
                visA.unsqueeze(-1) & cand_group_valid.unsqueeze(3)
            ).reshape(B, H, N, C, Tgrp) & query_valid
            group_scores_q.masked_fill_(~group_summary_mask, neg_inf)

            Kg = min(self.topk_groups, Tgrp)
            Kg_req = min(self.topk_groups // 2, Tgrp)
            if Kg > 0 and Kg_req > 0:
                group_req_idx, group_req_scores = self._route_topk_requests_unsorted(group_scores_q, Kg_req)
                group_scores = self._max_route_scores_from_requests(group_req_idx, group_req_scores, Tgrp, neg_inf)
                _, top_group_idx = self._topk_scores_indices(group_scores, Kg)

                group_req_valid = group_summary_mask.gather(-1, group_req_idx)
                c_eff_g = torch.where(
                    group_req_valid,
                    ar_c32.view(1, 1, 1, C, 1),
                    torch.tensor(BIG_INT, device=device, dtype=torch.int32),
                )
                route_first_g = torch.full((B * H * N, Tgrp), BIG_INT, device=device, dtype=torch.int32)
                route_first_g.scatter_reduce_(
                    1,
                    group_req_idx.reshape(B * H * N, -1),
                    c_eff_g.expand_as(group_req_idx).reshape(B * H * N, -1),
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
        return out, {}


# Convenience alias: swap the import and keep the class name.
GlobalAttention = GlobalAttentionFused
