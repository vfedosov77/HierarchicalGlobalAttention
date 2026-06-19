"""Qwen3-0.6B fused hierarchical attention with fast Triton prefill/training and KV-cache decode.

This module keeps the existing full-sequence fused training/prefill path and adds an
incremental one-token decode path.  The decode path stores hierarchical route state
on the Hugging Face cache object as ``cache._hga[layer_idx]``; KV stays in the
normal HF cache, while chunk/group summaries and current-chunk route ids stay in
HGA state.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import triton
import triton.language as tl


CosSin = Tuple[torch.Tensor, torch.Tensor]

_QWEN3_06B_CACHE_IMPL_VERSION = "qwen3-06b-hga-fused-train-fastgen-2026-06-18-v3"

LOG2E = 1.4426950408889634
BIG = 1 << 20
BIG_INT = 1 << 20
NEG_INF = -1.0e4


class _HeadRMSNorm(nn.Module):
    """Small Qwen3-compatible per-head RMSNorm used for q_norm/k_norm.

    The parameter name is only ``weight``, so state_dict keys stay compatible with
    Qwen3 attention modules: ``q_norm.weight`` and ``k_norm.weight``.
    """

    def __init__(self, hidden_size: int, eps: float = 1.0e-6, **factory_kwargs: Any) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.ones(int(hidden_size), **factory_kwargs))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (y * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Triton kernels: full chunked forward/backward path
# ---------------------------------------------------------------------------

@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _upd(s, vv, m_i, l_i, acc, IEEE: tl.constexpr):
    # One online-softmax accumulation step; s is already masked with -inf.
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2((m_i - m_new) * LOG2E)
    p = tl.exp2((s - m_new[:, None]) * LOG2E)
    l_new = l_i * alpha + tl.sum(p, 1)
    acc_new = acc * alpha[:, None] + _dot(p, vv, IEEE)
    return m_new, l_new, acc_new


@triton.jit
def _upd1(s, vv, m_i, l_i, acc):
    # One-row online-softmax step for decode. s: [R], vv: [R, D].
    m_new = tl.maximum(m_i, tl.max(s, axis=0))
    alpha = tl.exp2((m_i - m_new) * LOG2E)
    p = tl.exp2((s - m_new) * LOG2E)
    l_new = l_i * alpha + tl.sum(p, axis=0)
    acc_new = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
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

    # C: current-chunk exact tokens.
    kk = tl.load(K + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    vv = tl.load(V + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    s = _dot(qs, tl.trans(kk), IEEE)
    msk = (offs_c[None, :] <= offs_c[:, None]) & ((row0 + offs_c)[None, :] < S)
    s = tl.where(msk, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, IEEE)

    # B: current-chunk group summaries.
    offs_m = tl.arange(0, M)
    gk_own = tl.load(GK + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :])
    gv_own = tl.load(GV + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :])
    s = _dot(qs, tl.trans(gk_own), IEEE)
    complete = (row0 + (offs_m + 1) * GS) <= S
    msk = complete[None, :] & ((offs_m * GS + GS - 1)[None, :] <= offs_c[:, None])
    s = tl.where(msk, s, float("-inf"))
    m_i, l_i, acc = _upd(s, gv_own, m_i, l_i, acc, IEEE)

    # A: selected previous-chunk group summaries.
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

    # D: opened exact groups from previous chunks.
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
    lse = m_i + tl.log(l_safe)
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
    scale = tl.cast(scale, tl.float32)
    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)
    row0 = n * C
    qkv_base = bh * S_pad * D
    g_base = bh * NM * D

    q = tl.load(Q + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    do = tl.load(DO + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    lse_r = tl.load(LSE + bh * S_pad + row0 + offs_c)
    lse_r = tl.where(lse_r > -1.0e37, lse_r, 1.0e38)
    dlt = tl.load(DELTA + bh * S_pad + row0 + offs_c)
    qs = q * scale
    qt = tl.trans(q)
    dot_ = tl.trans(do)
    dq_acc = tl.zeros([C, D], tl.float32)

    # C.
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
    tl.atomic_add(DK + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None], _dot(qt, ds, IEEE))
    tl.atomic_add(DV + qkv_base + (row0 + offs_c)[None, :] * D + offs_d[:, None], _dot(dot_, p, IEEE))

    # B.
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
    tl.atomic_add(DGK + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None], _dot(qt, dsb, IEEE))
    tl.atomic_add(DGV + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None], _dot(dot_, pb, IEEE))

    # A.
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
        tl.atomic_add(DGK + g_base + grow[None, :] * D + offs_d[:, None], _dot(qt, dsa, IEEE), mask=rm[None, :])
        tl.atomic_add(DGV + g_base + grow[None, :] * D + offs_d[:, None], _dot(dot_, pa, IEEE), mask=rm[None, :])

    # D.
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
        tl.atomic_add(DK + qkv_base + pos[None, :] * D + offs_d[:, None], _dot(qt, dsd, IEEE), mask=rm[None, :])
        tl.atomic_add(DV + qkv_base + pos[None, :] * D + offs_d[:, None], _dot(dot_, pd, IEEE), mask=rm[None, :])

    tl.store(DQ + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :], dq_acc)


# ---------------------------------------------------------------------------
# Triton kernel: one-token KV-cache decode path
# ---------------------------------------------------------------------------

@triton.jit
def _ha_decode1_kernel(
    Q, K, V, GK, GV,
    IDXA, THRA, POSD, THRD,
    OUT,
    S_VALID, POS_CUR, S_CACHE, MAX_CHUNKS,
    KC, KG, scale,
    H_TOT: tl.constexpr, KVH: tl.constexpr, KV_GROUPS: tl.constexpr,
    C: tl.constexpr, M: tl.constexpr, GS: tl.constexpr, D: tl.constexpr,
    BR: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H_TOT
    h = bh - b * H_TOT
    kvh = h // KV_GROUPS

    offs_d = tl.arange(0, D)
    q = tl.load(Q + bh * D + offs_d)
    qs = q * tl.cast(scale, tl.float32)

    m_i = tl.full([], -1.0e38, tl.float32)
    l_i = tl.full([], 0.0, tl.float32)
    acc = tl.zeros([D], tl.float32)

    n = POS_CUR // C
    c = POS_CUR - n * C
    row0 = n * C
    k_base = (b * KVH + kvh) * S_CACHE * D
    g_base = (b * KVH + kvh) * MAX_CHUNKS * M * D

    # C: exact tokens in the current chunk, up to the current token.
    offs_c = tl.arange(0, C)
    pos_c = row0 + offs_c
    mask_c = (offs_c <= c) & (pos_c < S_VALID)
    kk = tl.load(K + k_base + pos_c[:, None] * D + offs_d[None, :], mask=mask_c[:, None], other=0.0)
    vv = tl.load(V + k_base + pos_c[:, None] * D + offs_d[None, :], mask=mask_c[:, None], other=0.0)
    s = tl.sum(qs[None, :] * kk, axis=1)
    s = tl.where(mask_c, s, float("-inf"))
    m_i, l_i, acc = _upd1(s, vv, m_i, l_i, acc)

    # B: completed group summaries inside the current chunk.
    offs_m = tl.arange(0, M)
    group_end = offs_m * GS + (GS - 1)
    mask_b = group_end <= c
    gk = tl.load(GK + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :], mask=mask_b[:, None], other=0.0)
    gv = tl.load(GV + g_base + (n * M + offs_m)[:, None] * D + offs_d[None, :], mask=mask_b[:, None], other=0.0)
    s = tl.sum(qs[None, :] * gk, axis=1)
    s = tl.where(mask_b, s, float("-inf"))
    m_i, l_i, acc = _upd1(s, gv, m_i, l_i, acc)

    # A: selected previous-chunk group summaries.
    Ra = KC * M
    idx_base = bh * KC
    for t in range(0, tl.cdiv(Ra, BR)):
        r = t * BR + tl.arange(0, BR)
        rm = r < Ra
        kc_i = r // M
        m_loc = r - kc_i * M
        j = tl.load(IDXA + idx_base + kc_i, mask=rm, other=0)
        thr = tl.load(THRA + idx_base + kc_i, mask=rm, other=BIG)
        valid = rm & (j < n) & (c >= thr) & (thr < BIG)
        grow = j * M + m_loc
        ka = tl.load(GK + g_base + grow[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        va = tl.load(GV + g_base + grow[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        sa = tl.sum(qs[None, :] * ka, axis=1)
        sa = tl.where(valid, sa, float("-inf"))
        m_i, l_i, acc = _upd1(sa, va, m_i, l_i, acc)

    # D: selected exact groups from previous chunks.
    Rd = KG * GS
    gidx_base = bh * KG
    for t in range(0, tl.cdiv(Rd, BR)):
        r = t * BR + tl.arange(0, BR)
        rm = r < Rd
        kg_i = r // GS
        off = r - kg_i * GS
        p0 = tl.load(POSD + gidx_base + kg_i, mask=rm, other=0)
        thr = tl.load(THRD + gidx_base + kg_i, mask=rm, other=BIG)
        pos = p0 + off
        valid = rm & (thr < BIG) & (c >= thr) & (pos < S_VALID)
        ko = tl.load(K + k_base + pos[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        vo = tl.load(V + k_base + pos[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        sd = tl.sum(qs[None, :] * ko, axis=1)
        sd = tl.where(valid, sd, float("-inf"))
        m_i, l_i, acc = _upd1(sd, vo, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe
    tl.store(OUT + bh * D + offs_d, out)


# ---------------------------------------------------------------------------
# Autograd wrapper for the full fused path
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
# One attention class: fused full path + fast generation path
# ---------------------------------------------------------------------------

class GlobalAttentionFused(nn.Module):
    """Single self-contained attention module: fused full path + fast generation.

    There is intentionally no inheritance from the old Python attention module.
    Unsupported inputs fail loudly instead of silently entering a slow path.
    """

    fp32_precise: bool = False  # True -> IEEE fp32 tl.dot for equivalence tests.

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
    ) -> None:
        nn.Module.__init__(self)

        self.config = None
        self.layer_idx = 0
        self.hidden_size = int(d_model)
        self.nhead = int(nhead)
        self.num_heads = self.nhead
        self.kv_heads = int(kv_heads)
        self.num_key_value_heads = self.kv_heads
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.nhead
        if self.nhead % self.kv_heads != 0:
            raise ValueError(f"num_attention_heads={self.nhead} must be divisible by num_key_value_heads={self.kv_heads}")
        self.num_key_value_groups = self.nhead // self.kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.nhead * self.head_dim, bias=use_bias_q)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=use_bias_k)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=use_bias_v)
        self.o_proj = nn.Linear(self.nhead * self.head_dim, self.hidden_size, bias=use_bias_o)

        if q_norm is not None:
            self.q_norm = q_norm
        elif qk_norm:
            self.q_norm = _HeadRMSNorm(self.head_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()

        if k_norm is not None:
            self.k_norm = k_norm
        elif qk_norm:
            self.k_norm = _HeadRMSNorm(self.head_dim, eps=norm_eps)
        else:
            self.k_norm = nn.Identity()

        self.dropout_p = float(dropout)
        self.attention_dropout = self.dropout_p
        self.is_causal = bool(causal)
        self.theta = float(theta)
        self.scale = self.head_dim ** -0.5
        self.scaling = self.scale

        self.use_global = bool(use_global)
        self.chunk_size = int(chunk_size)
        self.group_size = int(group_size)
        self.groups_per_chunk = max(1, self.chunk_size // self.group_size)
        if self.group_size * self.groups_per_chunk != self.chunk_size:
            # Prefer an exact chunk partition for the Triton kernels.
            if self.chunk_size % self.groups_per_chunk == 0:
                self.group_size = self.chunk_size // self.groups_per_chunk
            elif self.chunk_size % self.group_size == 0:
                self.groups_per_chunk = self.chunk_size // self.group_size
            else:
                raise ValueError("chunk_size must be divisible by group_size/groups_per_chunk")

        self.topk_chunks = int(topk_chunks)
        self.topk_groups = int(topk_groups)
        self.group_kv_scale = 1.0 / math.sqrt(max(1, self.group_size))
        self.mixed_rope_threshold = float(mixed_rope_threshold)
        self.mixed_rope_cutoff_pair = mixed_rope_cutoff_pair
        self.return_router_stats = bool(return_router_stats)
        self.router_scale_init = float(router_scale_init)
        self._last_path = "init"

    # ------------------------------- public forward -------------------------------

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
            past_key_value = kw.pop("past_key_values", None)
        else:
            kw.pop("past_key_values", None)

        if not self._fused_applicable(hidden_states):
            self._last_path = "unsupported_no_slow_path"
            raise RuntimeError(
                "GlobalAttentionFused has no external Python attention path. "
                "Use CUDA/Triton, fp32 for training or fp16/bf16/fp32 for decode, "
                "dropout_p=0, return_router_stats=False, and positive top-k settings."
            )

        B, S, _ = hidden_states.shape
        layer_idx = int(getattr(self, "layer_idx", kw.pop("layer_idx", 0)))
        cos, sin = self._normalize_position_embeddings(position_embeddings, hidden_states)

        q_raw, k_raw_base, v_base = self._project_qkv_base(hidden_states)
        q = self._apply_rotary(q_raw.float(), cos, sin).to(q_raw.dtype)
        k_rope_base = self._apply_rotary(k_raw_base.float(), cos, sin).to(k_raw_base.dtype)

        # No cache: use the original full fused autograd path. This is the training path.
        if past_key_value is None:
            self._last_path = "fused_full"
            out = self._forward_full_fused(q, k_raw_base, k_rope_base, v_base, cos, sin, S)
            return out, None

        seen_before = self._incoming_start(cache_position, past_key_value, layer_idx)
        state = self._get_hga_state(past_key_value, layer_idx)
        if state is not None and not self._state_matches(state, B, q.shape[1], k_raw_base.shape[1], q.shape[-1], hidden_states.device):
            state = None

        if state is not None and int(getattr(state, "seen", seen_before)) != int(seen_before):
            # Do not risk using stale route ids.  We can rebuild only at a chunk
            # boundary; inside a partial chunk the missing q_chunk route history is
            # not recoverable from a plain KV cache.
            state = None

        k_cache_base, v_cache_base = self._update_kv_cache(
            past_key_value, k_rope_base, v_base, layer_idx, cache_position
        )
        seen_after = seen_before + S

        if state is None and seen_before > 0:
            if seen_before % self.chunk_size != 0:
                self._last_path = "missing_hga_state_no_slow_path"
                raise RuntimeError(
                    "KV cache has no cache._hga route state for this layer and the "
                    "current chunk is already partially filled. Exact fast decode needs "
                    "q_chunk/top-id history for previous tokens in that chunk. Run the "
                    "prefill through this module, or start from a chunk boundary."
                )
            self._last_path = "rebuild_hga_from_kv_boundary"
            state = self._build_hga_state_from_kv_cache_boundary(
                past_key_value, layer_idx, k_cache_base, v_cache_base, seen_before
            )
            self._set_hga_state(past_key_value, layer_idx, state)

        # Empty-cache prompt/prefill: compute all prompt tokens with the full Triton core,
        # then build compact HGA state once for subsequent one-token decode.
        if S > 1 and seen_before == 0:
            self._last_path = "fused_prefill_build_hga"
            out = self._forward_full_fused(q, k_raw_base, k_rope_base, v_base, cos, sin, S)
            with torch.no_grad():
                state = self._build_hga_state_from_prefill(
                    past_key_value, layer_idx, q, k_raw_base, k_rope_base, v_base, cos, sin, seen_after
                )
                self._set_hga_state(past_key_value, layer_idx, state)
            return out, None

        # One-token decode, or a small continuation with an existing state.
        if state is None:
            state = self._new_hga_state(
                B=B,
                H=q.shape[1],
                KVH=k_raw_base.shape[1],
                D=q.shape[-1],
                dtype=k_rope_base.dtype,
                device=hidden_states.device,
                needed_chunks=max(1, (seen_after + self.chunk_size - 1) // self.chunk_size),
                cache=past_key_value,
            )
            self._set_hga_state(past_key_value, layer_idx, state)

        self._last_path = "fused_decode1" if S == 1 else "fused_decode_loop"
        out_heads = []
        for i in range(S):
            pos = seen_before + i
            cur = self._decode_one_projected(
                q[:, :, i, :].contiguous(),
                k_raw_base[:, :, i, :].contiguous(),
                k_rope_base[:, :, i, :].contiguous(),
                v_base[:, :, i, :].contiguous(),
                k_cache_base,
                v_cache_base,
                state,
                pos,
            )
            out_heads.append(cur)
        state.seen = int(seen_after)
        out_h = torch.stack(out_heads, dim=2)  # [B, H, S, D]
        out = self.o_proj(out_h.transpose(1, 2).contiguous().reshape(B, S, q.shape[1] * q.shape[-1]))
        return out, None

    # ------------------------------- full fused path -------------------------------

    def _forward_full_fused(
        self,
        q: torch.Tensor,
        k_raw_base: torch.Tensor,
        k_rope_base: torch.Tensor,
        v_base: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        S: int,
    ) -> torch.Tensor:
        B, H, _, Dh = q.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = q.device, q.dtype

        k_raw = self._repeat_kv(k_raw_base)
        k_rope = self._repeat_kv(k_rope_base)
        v = self._repeat_kv(v_base)

        N = (S + C - 1) // C
        S_pad = N * C

        q_p = self._pad_seq(q, S_pad).contiguous()
        k_p = self._pad_seq(k_rope, S_pad).contiguous()
        v_p = self._pad_seq(v, S_pad).contiguous()
        k_raw_p = self._pad_seq(k_raw, S_pad)

        with torch.no_grad():
            valid_flat = torch.arange(S_pad, device=device) < S
            valid_chunks = valid_flat.view(N, C)
            chunk_len = valid_chunks.sum(dim=1)
            group_len = valid_chunks.view(N, M, gs).sum(dim=-1)
            valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)
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

        with torch.no_grad():
            neg_inf = NEG_INF
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

            # Stage 1: previous-chunk selection.
            Kc = min(self.topk_chunks, N)
            scores_roll = torch.einsum("bhncd,bhmd->bhncm", route_q_chunks.float(), chunk_k.float()) * self._scale_value()
            scores_roll = scores_roll.to(q.dtype)
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

            # Stage 2: opened-group selection.
            cand_k_flat = self._gather_chunks(group_k, top_chunk_idx).reshape(B, H, N, Kc * M, Dh)
            cand_group_valid = valid_groups[top_chunk_idx] & valid_prev.unsqueeze(-1)
            Tgrp = Kc * M

            gsr = torch.einsum("bhncd,bhnrd->bhncr", route_q_chunks.float(), cand_k_flat.float()) * self._scale_value()
            gsr = gsr.to(q.dtype)
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
            S, self._scale_value(), C, M, gs, self.fp32_precise,
        )
        return self.o_proj(out_p[:, :, :S, :].transpose(1, 2).contiguous().reshape(B, S, H * Dh))

    # ------------------------------- decode path -------------------------------

    @torch.no_grad()
    def _decode_one_projected(
        self,
        q_cur: torch.Tensor,          # [B, H, D]
        k_raw_cur: torch.Tensor,      # [B, KVH, D]
        k_rope_cur: torch.Tensor,     # [B, KVH, D]
        v_cur: torch.Tensor,          # [B, KVH, D]
        k_cache: torch.Tensor,        # [B, KVH, S_cache, D]
        v_cache: torch.Tensor,        # [B, KVH, S_cache, D]
        state: Any,
        pos: int,
    ) -> torch.Tensor:
        B, H, D = q_cur.shape
        KVH = k_raw_cur.shape[1]
        C, M, gs = self.chunk_size, self.groups_per_chunk, self.group_size
        self._ensure_state_capacity(state, (pos // C) + 1)
        self._update_hga_state_one(q_cur, k_raw_cur, k_rope_cur, v_cur, state, pos)

        out_h = torch.empty((B, H, D), device=q_cur.device, dtype=q_cur.dtype)
        if not k_cache.is_contiguous():
            k_cache = k_cache.contiguous()
        if not v_cache.is_contiguous():
            v_cache = v_cache.contiguous()

        grid = (B * H,)
        _ha_decode1_kernel[grid](
            q_cur, k_cache, v_cache, state.group_k, state.group_v,
            state.chunk_top_idx, state.chunk_top_thr,
            state.group_top_pos, state.group_top_thr,
            out_h,
            pos + 1, pos, k_cache.shape[2], state.max_chunks,
            state.chunk_top_idx.shape[-1], state.group_top_pos.shape[-1], self._scale_value(),
            H_TOT=H, KVH=KVH, KV_GROUPS=H // KVH,
            C=C, M=M, GS=gs, D=D, BR=64,
            num_warps=4, num_stages=1,
        )
        return out_h

    @torch.no_grad()
    def _update_hga_state_one(
        self,
        q_cur: torch.Tensor,
        k_raw_cur: torch.Tensor,
        k_rope_cur: torch.Tensor,
        v_cur: torch.Tensor,
        state: Any,
        pos: int,
    ) -> None:
        C, gs = self.chunk_size, self.group_size
        n = pos // C
        c = pos - n * C
        g = c // gs

        if c == 0:
            state.cur_chunk_raw.zero_()
            state.cur_chunk_rope.zero_()
            state.cur_chunk_v.zero_()
            state.q_chunk.zero_()
            state.route_scores.fill_(NEG_INF)
            state.route_first.fill_(BIG_INT)
            state.chunk_top_idx.zero_()
            state.chunk_top_thr.fill_(BIG_INT)
            state.group_top_pos.zero_()
            state.group_top_thr.fill_(BIG_INT)
        if c % gs == 0:
            state.cur_group_raw.zero_()
            state.cur_group_rope.zero_()
            state.cur_group_v.zero_()

        state.cur_chunk_raw.add_(k_raw_cur.float())
        state.cur_chunk_rope.add_(k_rope_cur.float())
        state.cur_chunk_v.add_(v_cur.float())
        state.cur_group_raw.add_(k_raw_cur.float())
        state.cur_group_rope.add_(k_rope_cur.float())
        state.cur_group_v.add_(v_cur.float())
        state.q_chunk[:, :, c, :].copy_(q_cur)

        # Materialize just-completed group/chunk summaries.  These are compact KVH
        # summaries; the decode kernel maps Q heads to their KV head.
        if c % gs == gs - 1:
            anchor = n * C + g * gs + (gs - 1) // 2
            gk = self._summary_from_sums(
                state.cur_group_raw,
                state.cur_group_rope,
                anchor_pos=anchor,
                scale=self.group_kv_scale,
                out_dtype=state.group_k.dtype,
            )
            state.group_k[:, :, n, g, :].copy_(gk)
            state.group_v[:, :, n, g, :].copy_((state.cur_group_v * self.group_kv_scale).to(state.group_v.dtype))

        if c == C - 1:
            anchor = n * C + (C - 1) // 2
            ck = self._summary_from_sums(
                state.cur_chunk_raw,
                state.cur_chunk_rope,
                anchor_pos=anchor,
                scale=1.0,
                out_dtype=state.chunk_k.dtype,
            )
            state.chunk_k[:, :, n, :].copy_(ck)

        self._update_chunk_routes_one(state, q_cur, c, n)
        self._finalize_group_routes(state, length=c + 1, chunk_idx=n)
        state.seen = int(pos + 1)

    @torch.no_grad()
    def _update_chunk_routes_one(self, state: Any, q_cur: torch.Tensor, c: int, chunk_idx: int) -> None:
        if chunk_idx <= 0:
            state.chunk_top_idx.zero_()
            state.chunk_top_thr.fill_(BIG_INT)
            return
        scores = self._score_chunks_compact(q_cur, state.chunk_k[:, :, :chunk_idx, :])
        k_req = min(int(self.topk_chunks), int(chunk_idx))
        req_scores, req_idx = torch.topk(scores, k_req, dim=-1, sorted=False)
        state.route_scores[:, :, :chunk_idx].scatter_reduce_(2, req_idx, req_scores, reduce="amax", include_self=True)
        first = torch.full_like(req_idx, int(c), dtype=torch.int32)
        state.route_first[:, :, :chunk_idx].scatter_reduce_(2, req_idx, first, reduce="amin", include_self=True)
        self._finalize_chunk_routes(state, chunk_idx)

    @torch.no_grad()
    def _rebuild_current_routes(self, state: Any, length: int, chunk_idx: int) -> None:
        state.route_scores.fill_(NEG_INF)
        state.route_first.fill_(BIG_INT)
        state.chunk_top_idx.zero_()
        state.chunk_top_thr.fill_(BIG_INT)
        state.group_top_pos.zero_()
        state.group_top_thr.fill_(BIG_INT)
        if length <= 0 or chunk_idx <= 0:
            return

        q_seen = state.q_chunk[:, :, :length, :]
        B, H, L, D = q_seen.shape
        KVH = state.chunk_k.shape[1]
        G = H // KVH
        qg = q_seen.reshape(B, KVH, G, L, D).float()
        ck = state.chunk_k[:, :, :chunk_idx, :].float()
        scores = torch.einsum("bkgld,bknd->bkgln", qg, ck) * self._scale_value()
        scores = scores.reshape(B, H, L, chunk_idx)
        k_req = min(int(self.topk_chunks), int(chunk_idx))
        req_scores, req_idx = torch.topk(scores, k_req, dim=-1, sorted=False)
        flat_scores = req_scores.reshape(B, H, L * k_req)
        flat_idx = req_idx.reshape(B, H, L * k_req)
        state.route_scores[:, :, :chunk_idx].scatter_reduce_(2, flat_idx, flat_scores, reduce="amax", include_self=True)
        offs = torch.arange(L, device=q_seen.device, dtype=torch.int32).view(1, 1, L, 1).expand_as(req_idx)
        state.route_first[:, :, :chunk_idx].scatter_reduce_(
            2, flat_idx, offs.reshape(B, H, L * k_req), reduce="amin", include_self=True
        )
        self._finalize_chunk_routes(state, chunk_idx)
        self._finalize_group_routes(state, length=length, chunk_idx=chunk_idx)

    @torch.no_grad()
    def _finalize_chunk_routes(self, state: Any, chunk_idx: int) -> None:
        Kc = state.chunk_top_idx.shape[-1]
        B, H = state.route_scores.shape[:2]
        device = state.route_scores.device
        state.chunk_top_idx.zero_()
        state.chunk_top_thr.fill_(BIG_INT)
        if chunk_idx <= 0:
            return

        k = min(Kc, int(chunk_idx))
        scores = state.route_scores[:, :, :chunk_idx]
        top_scores, top_idx = torch.topk(scores, k, dim=-1, sorted=False)
        if k < Kc:
            pad_scores = torch.full((B, H, Kc - k), NEG_INF, device=device, dtype=top_scores.dtype)
            pad_idx = torch.zeros((B, H, Kc - k), device=device, dtype=top_idx.dtype)
            top_scores = torch.cat([top_scores, pad_scores], dim=-1)
            top_idx = torch.cat([top_idx, pad_idx], dim=-1)

        # Force the immediately previous chunk to be visible from the beginning
        # of the current chunk, exactly as the full path does.
        prev = int(chunk_idx - 1)
        prev_t = torch.full((B, H, 1), prev, device=device, dtype=top_idx.dtype)
        missing_prev = ~(top_idx == prev_t).any(dim=-1, keepdim=True)
        replace_at = top_scores.argmin(dim=-1, keepdim=True)
        top_idx = top_idx.scatter(-1, replace_at, torch.where(missing_prev, prev_t, top_idx.gather(-1, replace_at)))

        thr = state.route_first.gather(2, top_idx.clamp_min(0).clamp_max(state.max_chunks - 1).to(torch.long))
        is_prev = top_idx == prev
        valid = (top_idx < chunk_idx) & ((top_scores > (NEG_INF / 2)) | is_prev)
        thr = torch.where(is_prev, torch.zeros_like(thr), thr)
        thr = torch.where(valid, thr, torch.full_like(thr, BIG_INT))
        state.chunk_top_idx.copy_(top_idx.to(torch.int32))
        state.chunk_top_thr.copy_(thr.to(torch.int32))

    @torch.no_grad()
    def _finalize_group_routes(self, state: Any, length: int, chunk_idx: int) -> None:
        Kg = state.group_top_pos.shape[-1]
        Kc = state.chunk_top_idx.shape[-1]
        M, gs = self.groups_per_chunk, self.group_size
        Tgrp = Kc * M
        B, H = state.chunk_top_idx.shape[:2]
        device = state.chunk_top_idx.device
        state.group_top_pos.zero_()
        state.group_top_thr.fill_(BIG_INT)

        kg_req = min(int(self.topk_groups) // 2, Tgrp)
        kg_top = min(int(self.topk_groups), Tgrp)
        if length <= 0 or chunk_idx <= 0 or kg_req <= 0 or kg_top <= 0:
            return

        q_seen = state.q_chunk[:, :, :length, :].float()       # [B, H, L, D]
        cand = self._gather_group_k_for_heads(state, state.chunk_top_idx).reshape(B, H, Tgrp, q_seen.shape[-1]).float()
        scores = torch.einsum("bhld,bhrd->bhlr", q_seen, cand) * self._scale_value()

        l_ar = torch.arange(length, device=device, dtype=torch.int32).view(1, 1, length, 1, 1)
        visible = l_ar >= state.chunk_top_thr.view(B, H, 1, Kc, 1)
        valid_parent = state.chunk_top_thr.view(B, H, 1, Kc, 1) < BIG_INT
        visible = (visible & valid_parent).reshape(B, H, length, Tgrp)
        scores.masked_fill_(~visible, NEG_INF)

        req_scores, req_idx = torch.topk(scores, kg_req, dim=-1, sorted=False)
        group_scores = torch.full((B * H, Tgrp), NEG_INF, device=device, dtype=torch.float32)
        group_scores.scatter_reduce_(
            1,
            req_idx.reshape(B * H, -1),
            req_scores.reshape(B * H, -1),
            reduce="amax",
            include_self=True,
        )
        first = torch.full((B * H, Tgrp), BIG_INT, device=device, dtype=torch.int32)
        offs = torch.arange(length, device=device, dtype=torch.int32).view(1, 1, length, 1).expand_as(req_idx)
        first.scatter_reduce_(
            1,
            req_idx.reshape(B * H, -1),
            offs.reshape(B * H, -1),
            reduce="amin",
            include_self=True,
        )

        top_scores, top_gid = torch.topk(group_scores, kg_top, dim=-1, sorted=False)
        top_first = first.gather(1, top_gid)
        if kg_top < Kg:
            pad = Kg - kg_top
            top_scores = torch.cat([top_scores, torch.full((B * H, pad), NEG_INF, device=device)], dim=-1)
            top_gid = torch.cat([top_gid, torch.zeros((B * H, pad), device=device, dtype=top_gid.dtype)], dim=-1)
            top_first = torch.cat([top_first, torch.full((B * H, pad), BIG_INT, device=device, dtype=torch.int32)], dim=-1)

        top_gid = top_gid.view(B, H, Kg)
        top_first = top_first.view(B, H, Kg)
        top_scores = top_scores.view(B, H, Kg)
        parent = top_gid // M
        src_chunk = state.chunk_top_idx.gather(-1, parent.clamp_max(Kc - 1).to(torch.long))
        src_grp = top_gid - parent * M
        posD = src_chunk * self.chunk_size + src_grp * gs
        valid = (top_scores > (NEG_INF / 2)) & (top_first < BIG_INT)
        posD = torch.where(valid, posD, torch.zeros_like(posD))
        top_first = torch.where(valid, top_first, torch.full_like(top_first, BIG_INT))
        state.group_top_pos.copy_(posD.to(torch.int32))
        state.group_top_thr.copy_(top_first.to(torch.int32))

    # ------------------------------- HGA state -------------------------------

    @torch.no_grad()
    def _build_hga_state_from_prefill(
        self,
        cache: Any,
        layer_idx: int,
        q: torch.Tensor,
        k_raw_base: torch.Tensor,
        k_rope_base: torch.Tensor,
        v_base: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seen: int,
    ) -> Any:
        B, H, S, D = q.shape
        KVH = k_raw_base.shape[1]
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        N = max(1, (seen + C - 1) // C)
        S_pad = N * C
        state = self._new_hga_state(
            B=B, H=H, KVH=KVH, D=D, dtype=k_rope_base.dtype, device=q.device,
            needed_chunks=N + 1, cache=cache,
        )
        state.seen = int(seen)

        k_raw_p = self._pad_seq(k_raw_base, S_pad)
        k_rope_p = self._pad_seq(k_rope_base, S_pad)
        v_p = self._pad_seq(v_base, S_pad)

        valid_flat = torch.arange(S_pad, device=q.device) < seen
        valid_chunks = valid_flat.view(N, C)
        chunk_len = valid_chunks.sum(dim=1)
        group_len = valid_chunks.view(N, M, gs).sum(dim=-1)
        ar_n = torch.arange(N, device=q.device)
        ar_m = torch.arange(M, device=q.device)
        chunk_start = ar_n * C
        chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(max(seen - 1, 0))
        group_start = ar_n[:, None] * C + ar_m[None, :] * gs
        group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(max(seen - 1, 0))

        k_raw_chunks = k_raw_p.reshape(B, KVH, N, C, D)
        k_rope_chunks = k_rope_p.reshape(B, KVH, N, C, D)
        v_chunks = v_p.reshape(B, KVH, N, C, D)
        chunk_mask = valid_chunks.view(1, 1, N, C, 1).to(k_rope_base.dtype)
        group_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(k_rope_base.dtype)

        ck = self._rope_summary(
            raw=k_raw_chunks,
            rope=k_rope_chunks,
            mask=chunk_mask,
            reduce_dim=3,
            anchor_pos=chunk_middle,
            cos=cos,
            sin=sin,
            scale=1.0,
        ).to(state.chunk_k.dtype)
        state.chunk_k[:, :, :N, :].copy_(ck)

        gk = self._rope_summary(
            raw=k_raw_chunks.reshape(B, KVH, N, M, gs, D),
            rope=k_rope_chunks.reshape(B, KVH, N, M, gs, D),
            mask=group_mask,
            reduce_dim=4,
            anchor_pos=group_middle,
            cos=cos,
            sin=sin,
            scale=self.group_kv_scale,
        ).to(state.group_k.dtype)
        gv = (v_chunks.reshape(B, KVH, N, M, gs, D) * group_mask).sum(dim=4) * self.group_kv_scale
        state.group_k[:, :, :N, :, :].copy_(gk)
        state.group_v[:, :, :N, :, :].copy_(gv.to(state.group_v.dtype))

        if seen > 0:
            cur_chunk = (seen - 1) // C
            chunk_off = (seen - 1) % C
            chunk_start_i = cur_chunk * C
            next_chunk_off = seen % C
            if next_chunk_off != 0:
                L = chunk_off + 1
                state.q_chunk[:, :, :L, :].copy_(q[:, :, chunk_start_i:seen, :])
                state.cur_chunk_raw.copy_(k_raw_base[:, :, chunk_start_i:seen, :].float().sum(dim=2))
                state.cur_chunk_rope.copy_(k_rope_base[:, :, chunk_start_i:seen, :].float().sum(dim=2))
                state.cur_chunk_v.copy_(v_base[:, :, chunk_start_i:seen, :].float().sum(dim=2))
                group_start_i = chunk_start_i + (chunk_off // gs) * gs
                if seen % gs != 0:
                    state.cur_group_raw.copy_(k_raw_base[:, :, group_start_i:seen, :].float().sum(dim=2))
                    state.cur_group_rope.copy_(k_rope_base[:, :, group_start_i:seen, :].float().sum(dim=2))
                    state.cur_group_v.copy_(v_base[:, :, group_start_i:seen, :].float().sum(dim=2))
                self._rebuild_current_routes(state, length=L, chunk_idx=cur_chunk)
        return state

    @torch.no_grad()
    def _build_hga_state_from_kv_cache_boundary(
        self,
        cache: Any,
        layer_idx: int,
        k_rope_cache_base: torch.Tensor,
        v_cache_base: torch.Tensor,
        seen: int,
    ) -> Any:
        # Exact rebuild is possible at a chunk boundary because there are no
        # previous tokens in the current chunk, so no q_chunk route history is
        # needed.  Raw K is recovered by applying inverse RoPE to cached RoPE K.
        if seen <= 0:
            raise ValueError("seen must be positive for KV-cache HGA rebuild")
        if seen % self.chunk_size != 0:
            raise RuntimeError("HGA state can be rebuilt from plain KV only at a chunk boundary")

        B, KVH, S_cache, D = k_rope_cache_base.shape
        if seen > S_cache:
            raise RuntimeError(f"KV cache length {S_cache} is shorter than requested seen={seen}")
        H = self.nhead
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        N = seen // C
        device = k_rope_cache_base.device
        dtype = k_rope_cache_base.dtype

        state = self._new_hga_state(
            B=B, H=H, KVH=KVH, D=D, dtype=dtype, device=device,
            needed_chunks=N + 1, cache=cache,
        )
        state.seen = int(seen)
        if N == 0:
            return state

        k_rope = k_rope_cache_base[:, :, :seen, :].contiguous()
        v = v_cache_base[:, :, :seen, :].contiguous()
        pos = torch.arange(seen, device=device, dtype=torch.float32)
        cos, sin = self._rotary_at_positions(pos, device, B)
        k_raw = self._apply_rotary(k_rope.float(), cos, -sin).to(dtype)

        valid_chunks = torch.ones((N, C), device=device, dtype=torch.bool)
        ar_n = torch.arange(N, device=device)
        ar_m = torch.arange(M, device=device)
        chunk_middle = ar_n * C + (C - 1) // 2
        group_middle = ar_n[:, None] * C + ar_m[None, :] * gs + (gs - 1) // 2

        k_raw_chunks = k_raw.reshape(B, KVH, N, C, D)
        k_rope_chunks = k_rope.reshape(B, KVH, N, C, D)
        v_chunks = v.reshape(B, KVH, N, C, D)
        chunk_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
        group_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)

        ck = self._rope_summary(
            raw=k_raw_chunks,
            rope=k_rope_chunks,
            mask=chunk_mask,
            reduce_dim=3,
            anchor_pos=chunk_middle,
            cos=cos,
            sin=sin,
            scale=1.0,
        ).to(state.chunk_k.dtype)
        state.chunk_k[:, :, :N, :].copy_(ck)

        gk = self._rope_summary(
            raw=k_raw_chunks.reshape(B, KVH, N, M, gs, D),
            rope=k_rope_chunks.reshape(B, KVH, N, M, gs, D),
            mask=group_mask,
            reduce_dim=4,
            anchor_pos=group_middle,
            cos=cos,
            sin=sin,
            scale=self.group_kv_scale,
        ).to(state.group_k.dtype)
        gv = (v_chunks.reshape(B, KVH, N, M, gs, D) * group_mask).sum(dim=4) * self.group_kv_scale
        state.group_k[:, :, :N, :, :].copy_(gk)
        state.group_v[:, :, :N, :, :].copy_(gv.to(state.group_v.dtype))
        return state

    def _get_hga_state(self, cache: Any, layer_idx: int) -> Optional[Any]:
        slots = getattr(cache, "_hga", None)
        if isinstance(slots, list) and int(layer_idx) < len(slots):
            return slots[int(layer_idx)]
        if isinstance(slots, dict):
            return slots.get(int(layer_idx))
        return None

    def _set_hga_state(self, cache: Any, layer_idx: int, state: Any) -> None:
        idx = int(layer_idx)
        slots = getattr(cache, "_hga", None)
        if isinstance(slots, dict):
            slots[idx] = state
            return
        if not isinstance(slots, list):
            slots = []
            setattr(cache, "_hga", slots)
        while len(slots) <= idx:
            slots.append(None)
        slots[idx] = state

    def _new_hga_state(
        self,
        *,
        B: int,
        H: int,
        KVH: int,
        D: int,
        dtype: torch.dtype,
        device: torch.device,
        needed_chunks: int,
        cache: Optional[Any] = None,
    ) -> Any:
        C, M = self.chunk_size, self.groups_per_chunk
        max_chunks = self._initial_state_chunks(cache, needed_chunks)
        Kc = max(1, int(self.topk_chunks))
        Kg = max(1, int(self.topk_groups))
        return SimpleNamespace(
            version=_QWEN3_06B_CACHE_IMPL_VERSION,
            seen=0,
            max_chunks=int(max_chunks),
            q_chunk=torch.zeros((B, H, C, D), device=device, dtype=dtype),
            chunk_k=torch.zeros((B, KVH, max_chunks, D), device=device, dtype=dtype),
            group_k=torch.zeros((B, KVH, max_chunks, M, D), device=device, dtype=dtype),
            group_v=torch.zeros((B, KVH, max_chunks, M, D), device=device, dtype=dtype),
            cur_chunk_raw=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            cur_chunk_rope=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            cur_chunk_v=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            cur_group_raw=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            cur_group_rope=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            cur_group_v=torch.zeros((B, KVH, D), device=device, dtype=torch.float32),
            route_scores=torch.full((B, H, max_chunks), NEG_INF, device=device, dtype=torch.float32),
            route_first=torch.full((B, H, max_chunks), BIG_INT, device=device, dtype=torch.int32),
            chunk_top_idx=torch.zeros((B, H, Kc), device=device, dtype=torch.int32),
            chunk_top_thr=torch.full((B, H, Kc), BIG_INT, device=device, dtype=torch.int32),
            group_top_pos=torch.zeros((B, H, Kg), device=device, dtype=torch.int32),
            group_top_thr=torch.full((B, H, Kg), BIG_INT, device=device, dtype=torch.int32),
        )

    def _initial_state_chunks(self, cache: Optional[Any], needed_chunks: int) -> int:
        C = int(self.chunk_size)
        max_len = -1
        if cache is not None:
            fn = getattr(cache, "get_max_cache_shape", None)
            if callable(fn):
                try:
                    v = fn()
                    if torch.is_tensor(v):
                        v = int(v.item())
                    max_len = int(v)
                except Exception:
                    max_len = -1
            if max_len <= 0:
                max_len = int(getattr(cache, "max_cache_len", -1) or -1)
        if max_len > 0:
            return max(int(math.ceil(max_len / C)), int(needed_chunks), 1)
        return max(16, int(needed_chunks), 1)

    @torch.no_grad()
    def _ensure_state_capacity(self, state: Any, needed_chunks: int) -> None:
        if int(needed_chunks) <= int(state.max_chunks):
            return
        old = int(state.max_chunks)
        new = max(int(needed_chunks), old * 2, 16)

        def grow(t: torch.Tensor, fill: float | int = 0) -> torch.Tensor:
            shape = list(t.shape)
            shape[2] = new
            out = torch.empty(shape, device=t.device, dtype=t.dtype)
            if fill == 0:
                out.zero_()
            else:
                out.fill_(fill)
            out[:, :, :old, ...].copy_(t)
            return out

        state.chunk_k = grow(state.chunk_k, 0)
        state.group_k = grow(state.group_k, 0)
        state.group_v = grow(state.group_v, 0)
        state.route_scores = grow(state.route_scores, NEG_INF)
        state.route_first = grow(state.route_first, BIG_INT)
        state.max_chunks = int(new)

    def _state_matches(self, state: Any, B: int, H: int, KVH: int, D: int, device: torch.device) -> bool:
        return (
            getattr(state, "version", None) == _QWEN3_06B_CACHE_IMPL_VERSION
            and getattr(state, "q_chunk", None) is not None
            and state.q_chunk.shape == (B, H, self.chunk_size, D)
            and state.chunk_k.shape[:2] == (B, KVH)
            and state.chunk_k.shape[-1] == D
            and state.q_chunk.device == device
        )

    # ------------------------------- route helpers -------------------------------

    def _score_chunks_compact(self, q: torch.Tensor, chunk_k: torch.Tensor) -> torch.Tensor:
        # q: [B,H,D], chunk_k: [B,KVH,N,D] -> [B,H,N]
        B, H, D = q.shape
        KVH = chunk_k.shape[1]
        N = chunk_k.shape[2]
        G = H // KVH
        qg = q.reshape(B, KVH, G, D).float()
        scores = torch.einsum("bkgd,bknd->bkgn", qg, chunk_k.float()) * self._scale_value()
        return scores.reshape(B, H, N)

    def _gather_group_k_for_heads(self, state: Any, top_chunk_idx: torch.Tensor) -> torch.Tensor:
        # state.group_k: [B,KVH,N,M,D], top_chunk_idx: [B,H,Kc] -> [B,H,Kc,M,D]
        B, H, Kc = top_chunk_idx.shape
        KVH = state.group_k.shape[1]
        M = state.group_k.shape[3]
        G = H // KVH
        device = top_chunk_idx.device
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, Kc, M)
        h_to_kv = (torch.arange(H, device=device) // G).view(1, H, 1, 1).expand(B, H, Kc, M)
        c_idx = top_chunk_idx.to(torch.long).clamp(0, state.max_chunks - 1).view(B, H, Kc, 1).expand(B, H, Kc, M)
        m_idx = torch.arange(M, device=device).view(1, 1, 1, M).expand(B, H, Kc, M)
        return state.group_k[b_idx, h_to_kv, c_idx, m_idx]

    # ------------------------------- cache helpers -------------------------------

    def _incoming_start(self, cache_position: Optional[torch.LongTensor], cache: Any, layer_idx: int) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position.reshape(-1)[0].item())
        state = self._get_hga_state(cache, layer_idx)
        if state is not None:
            return int(getattr(state, "seen", 0))
        fn = getattr(cache, "get_seq_length", None)
        if callable(fn):
            try:
                v = fn(layer_idx)
            except TypeError:
                v = fn()
            if torch.is_tensor(v):
                return int(v.reshape(-1)[0].item())
            return int(v)
        return 0

    def _update_kv_cache(
        self,
        cache: Any,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = {"cache_position": cache_position} if cache_position is not None else None
        if cache_kwargs is not None:
            try:
                return cache.update(k, v, layer_idx, cache_kwargs)
            except TypeError:
                try:
                    return cache.update(k, v, layer_idx, cache_kwargs=cache_kwargs)
                except TypeError:
                    pass
        return cache.update(k, v, layer_idx)

    # ------------------------------- projection / RoPE helpers -------------------------------

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
            if t.dim() == 3:
                t = t.unsqueeze(1)
            elif t.dim() == 4 and t.shape[1] == 1:
                pass
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
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=endpoint)

    def _summary_from_sums(
        self,
        raw_sum: torch.Tensor,
        rope_sum: torch.Tensor,
        *,
        anchor_pos: int,
        scale: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        cos, sin = self._rotary_at_positions(anchor_pos, raw_sum.device, raw_sum.shape[0])
        raw_scaled = raw_sum * float(scale)
        rope_scaled = rope_sum * float(scale)
        endpoint = self._apply_rotary(raw_scaled, cos, sin)
        return self._mix_tokenwise_and_anchor(rope_scaled, endpoint).to(out_dtype)

    def _rotary_at_positions(self, pos: int | torch.Tensor, device: torch.device, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_t = torch.as_tensor(pos, device=device, dtype=torch.float32)
        shape = tuple(pos_t.shape)
        pos_f = pos_t.reshape(-1)
        half = self.head_dim // 2
        theta = float(getattr(self, "theta", getattr(getattr(self, "config", None), "rope_theta", 10000.0)))
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = pos_f[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().reshape(1, 1, *shape, self.head_dim)
        sin = emb.sin().reshape(1, 1, *shape, self.head_dim)
        if batch != 1:
            cos = cos.expand(batch, -1, *([-1] * len(shape)), -1)
            sin = sin.expand(batch, -1, *([-1] * len(shape)), -1)
        return cos, sin

    def _mix_tokenwise_and_anchor(self, tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        mask = self._mixed_tokenwise_mask(tokenwise.device).to(dtype=torch.bool)
        view_shape = [1] * tokenwise.ndim
        view_shape[-1] = self.head_dim
        return torch.where(mask.view(*view_shape), tokenwise, anchor)

    def _mixed_cutoff_pair(self) -> int:
        half = self.head_dim // 2
        cutoff_override = getattr(self, "mixed_rope_cutoff_pair", None)
        if cutoff_override is not None:
            return int(cutoff_override)
        threshold = float(getattr(self, "mixed_rope_threshold", 0.0))
        theta = float(getattr(self, "theta", getattr(getattr(self, "config", None), "rope_theta", 10000.0)))
        cutoff = 0
        for i in range(half):
            inv_freq = 1.0 / (theta ** (i / half))
            if max(1, self.chunk_size - 1) * inv_freq > threshold:
                cutoff = i + 1
        return cutoff

    def _mixed_tokenwise_mask(self, device: torch.device) -> torch.Tensor:
        half = self.head_dim // 2
        cutoff = self._mixed_cutoff_pair()
        pair_mask = torch.arange(half, device=device) < cutoff
        return torch.cat([pair_mask, pair_mask], dim=0)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    @staticmethod
    def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        pad = target_len - x.shape[2]
        if pad <= 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:2], pad, x.shape[-1])], dim=2)

    @staticmethod
    def _gather_rotary(cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> CosSin:
        shape = pos.shape
        flat = pos.reshape(-1)
        cos_g = cos[:, :, flat, :].reshape(cos.shape[0], 1, *shape, cos.shape[-1])
        sin_g = sin[:, :, flat, :].reshape(sin.shape[0], 1, *shape, sin.shape[-1])
        return cos_g, sin_g

    @staticmethod
    def _gather_chunks(tensor: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
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

    # ------------------------------- misc helpers -------------------------------

    def _fused_applicable(self, x: torch.Tensor) -> bool:
        C, M, D = self.chunk_size, self.groups_per_chunk, self.head_dim
        dtype_ok = x.dtype == torch.float32 or (not self.training and x.dtype in (torch.float16, torch.bfloat16))
        return (
            bool(getattr(self, "use_global", True))
            and x.is_cuda
            and dtype_ok
            and x.ndim == 3
            and x.shape[1] > 0
            and not bool(getattr(self, "return_router_stats", False))
            and not (self.training and float(getattr(self, "dropout_p", 0.0)) > 0.0)
            and int(getattr(self, "topk_chunks", 0)) > 0
            and int(getattr(self, "topk_groups", 0)) > 0
            and C >= 16 and (C & (C - 1)) == 0
            and M >= 16 and (M & (M - 1)) == 0
            and D >= 16 and (D & (D - 1)) == 0
        )

    def _scale_value(self) -> float:
        return float(getattr(self, "scale", getattr(self, "scaling", self.head_dim ** -0.5)))

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
    def _route_topk_requests_unsorted(self, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_routes = scores.shape[-1]
        assert k > 0
        if k >= num_routes:
            return self._all_route_indices(scores.shape[:-1], num_routes, scores.device), scores
        top_scores, top_idx = torch.topk(scores, k, dim=-1, sorted=False)
        return top_idx, top_scores

    @staticmethod
    @torch.no_grad()
    def _max_route_scores_from_requests(
        request_idx: torch.Tensor,
        request_scores: torch.Tensor,
        num_routes: int,
        fill_value: float,
    ) -> torch.Tensor:
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


# Convenience alias: swap the import and keep the class name.
GlobalAttention = GlobalAttentionFused
