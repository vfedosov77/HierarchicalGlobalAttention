"""Unified hierarchical global attention for Qwen3-0.6B.

A single self-contained ``GlobalAttention`` class serves every purpose:

* **Training / prefill (no cache).**  When the Triton fast path is applicable
  (CUDA, fp32, power-of-two C/M/head_dim) it runs a fused kernel; otherwise it
  falls back to an equivalent dense torch path.  Both compute the exact
  hierarchical routed attention over the whole sequence.

* **Generation (Hugging Face KV cache).**  Decoding is a *simplified, strictly
  causal, single-token* path.  Closed chunks are never recomputed: their
  chunk/group key summaries live in ``past_key_value._hga[layer_idx]`` and the
  per-token keys/values live in the ordinary Qwen cache.  The routing decisions
  of earlier tokens of the active chunk are remembered as cumulative running-max
  score vectors (the "opened ids" the new token must also see).  A new token
  computes one q/k/v, routes its own top-k chunks/groups, unions that with the
  remembered ids, builds **one** small score row, and reads the corresponding
  token/group values.

The decode path caches *ids/summaries*, never queries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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



import triton
import triton.language as tl

CosSin = Tuple[torch.Tensor, torch.Tensor]

_QWEN3_06B_CACHE_IMPL_VERSION = "qwen3-06b-cache-unified-runningmax-2026-06-18"

_NEG_INF = -1.0e4
_FINITE_MIN = _NEG_INF / 2  # values above this are "real" scores, not the sentinel


# ===========================================================================
# Triton kernels (used only for the no-cache fp32 power-of-two fast path)
# ===========================================================================

LOG2E = tl.constexpr(1.4426950408889634)
BIG = tl.constexpr(1 << 20)
BIG_INT = 1 << 20

@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b)

@triton.jit
def _upd(s, vv, m_i, l_i, acc, IEEE: tl.constexpr):
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

    # segment C: current-chunk exact tokens (causal)
    kk = tl.load(K + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    vv = tl.load(V + qkv_base + (row0 + offs_c)[:, None] * D + offs_d[None, :])
    s = _dot(qs, tl.trans(kk), IEEE)
    msk = (offs_c[None, :] <= offs_c[:, None]) & ((row0 + offs_c)[None, :] < S)
    s = tl.where(msk, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, IEEE)

    # segment B: current-chunk group summaries
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

    # segment A: candidate previous-chunk group summaries
    Ra = KC * M
    ia_base = (bh * N + n) * KC
    for t in range(0, tl.cdiv(Ra, BRA)):
        r = t * BRA + tl.arange(0, BRA)
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

    # segment D: opened exact tokens
    Rd = KG * GS
    id_base = (bh * N + n) * KG
    for t in range(0, tl.cdiv(Rd, BRD)):
        r = t * BRD + tl.arange(0, BRD)
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
    BRA: tl.constexpr, BRD: tl.constexpr, IEEE: tl.constexpr,
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
            tl.atomic_add(DGK + g_base + (n * M + mb) * D + offs_d, tl.sum(q * dsb_m[:, None], axis=0))
            tl.atomic_add(DGV + g_base + (n * M + mb) * D + offs_d, tl.sum(do * pb_m[:, None], axis=0))
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
        tl.atomic_add(DGK + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None], _dot(qt, dsb, IEEE))
        tl.atomic_add(DGV + g_base + (n * M + offs_m)[None, :] * D + offs_d[:, None], _dot(dot_, pb, IEEE))

    Ra = KC * M
    ia_base = (bh * N + n) * KC
    for t in range(0, tl.cdiv(Ra, BRA)):
        r = t * BRA + tl.arange(0, BRA)
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

    Rd = KG * GS
    id_base = (bh * N + n) * KG
    for t in range(0, tl.cdiv(Rd, BRD)):
        r = t * BRD + tl.arange(0, BRD)
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
        # For small group counts, previous-summary routes are few (Kc*M), so a
        # 32-wide tile reduces inactive lanes; opened-token routes usually
        # dominate, so keep the larger 64-wide tile there for throughput.
        br_a = 32 if M < 16 else 64
        br_d = 64
        _ha_fwd_kernel[grid](
            q, k, v, gk, gv, idxA, thrA, posD, thrD, out, lse,
            S, N, KC, KG, scale, S_pad, N * M,
            C=C, M=M, GS=gs, D=D, BRA=br_a, BRD=br_d, IEEE=ieee, num_warps=4,
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
            dout, lse, delta, dq, dk, dv, dgk, dgv,
            S, N, KC, KG, scale, S_pad, N * M,
            C=C, M=M, GS=gs, D=D, BRA=br_a, BRD=br_d, IEEE=ieee, num_warps=4, num_stages=1,
        )
        return (dq, dk, dv, dgk, dgv, None, None, None, None, None, None, None, None, None, None)


# ===========================================================================
# Per-layer routing side-cache stored at ``past_key_value._hga[layer_idx]``
# ===========================================================================

class _HGAState:
    """Compact per-layer decode state.

    ``chunk_k``/``group_k`` are the key summaries of *closed* chunks.  The active
    (partial) chunk contributes through running raw/rope sums.  ``chunk_smax`` and
    ``group_smax`` hold the running max of routing scores across the tokens of the
    active chunk; an entry above ``_FINITE_MIN`` means "some earlier token of this
    chunk opened that chunk/group", i.e. the cumulative set of opened ids.  No
    queries are cached.
    """

    __slots__ = (
        "seen", "max_chunks",
        "chunk_k", "group_k",
        "cur_chunk_raw", "cur_chunk_rope", "cur_group_raw", "cur_group_rope",
        "chunk_smax", "group_smax",
        # Static-decode additions: self-contained KV ring + scratch.
        "kbuf", "vbuf", "max_seq",
    )

    def __init__(self, **kw: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


def _safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# ===========================================================================
# The one class
# ===========================================================================

class GlobalAttention(nn.Module):
    """Hierarchical chunk-routed causal attention (training + generation)."""

    fp32_precise: bool = False  # True -> IEEE fp32 tl.dot for the fused path

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
        decode_route_overflow: str = "topk",
        decode_max_seq: int = 8192,
    ) -> None:
        super().__init__()
        assert causal, "This implementation is causal-only."
        assert nhead % kv_heads == 0
        head_dim = d_model // nhead if head_dim is None else head_dim
        assert head_dim % 2 == 0, "RoPE needs an even head_dim."
        assert chunk_size % group_size == 0
        if decode_route_overflow not in ("topk", "all"):
            raise ValueError("decode_route_overflow must be 'topk' or 'all'")
        if mixed_rope_cutoff_pair is not None and not (0 <= mixed_rope_cutoff_pair <= head_dim // 2):
            raise ValueError(f"mixed_rope_cutoff_pair must be in [0, {head_dim // 2}]")

        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout_p = dropout
        self.causal = causal
        self.chunk_size = chunk_size
        self.group_size = group_size
        self.groups_per_chunk = chunk_size // group_size
        self.topk_chunks = topk_chunks
        self.topk_groups = topk_groups
        self.theta = float(theta)
        self.return_router_stats = return_router_stats
        self.mixed_rope_threshold = float(mixed_rope_threshold)
        self.mixed_rope_cutoff_pair = mixed_rope_cutoff_pair
        self.decode_route_overflow = decode_route_overflow

        self.group_kv_scale = 1.0 / (group_size + math.sqrt(group_size))

        self.q_proj = nn.Linear(d_model, nhead * head_dim, bias=use_bias_q)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_k)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_v)
        self.o_proj = nn.Linear(nhead * head_dim, d_model, bias=use_bias_o)

        make_norm = lambda: RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.q_norm = q_norm if q_norm is not None else make_norm()
        self.k_norm = k_norm if k_norm is not None else make_norm()

        # Qwen3Attention-compatible attributes.
        self.layer_idx = layer_idx
        self.num_key_value_groups = nhead // kv_heads
        self.scaling = self.scale
        self.attention_dropout = dropout
        self.is_causal = True
        self.sliding_window = None
        self._last_path = "init"
        self.decode_max_seq = int(decode_max_seq)

    # ------------------------------------------------------------------
    # Public forward (Qwen3Attention contract)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
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

        q_raw, k_raw_new, v_new = self._project_qkv_base(hidden_states)
        q_new = self._apply_rotary(q_raw.float(), cos, sin).to(dtype=q_raw.dtype)
        k_new = self._apply_rotary(k_raw_new.float(), cos, sin).to(dtype=k_raw_new.dtype)

        # ---- No cache: training / full-sequence prefill ----
        if past_key_value is None:
            return self._forward_no_cache(q_raw, q_new, k_raw_new, k_new, v_new, cos, sin, attention_mask)

        layer_idx = self._require_layer_idx()

        # ---- Fast static decode: single token, state already seeded ----
        # Sync-free and CUDA-graph friendly: positions stay on-GPU, KV lives in
        # the state's own fixed buffers (no DynamicCache append / reallocation).
        if q_len == 1 and cache_position is not None:
            state = self._get_state(past_key_value, layer_idx)
            if state is not None and state.kbuf is not None:
                pos = cache_position.reshape(-1)[0]  # GPU long scalar
                idx = pos.view(1)
                state.kbuf.index_copy_(2, idx, k_new[:, :, 0:1, :])
                state.vbuf.index_copy_(2, idx, v_new[:, :, 0:1, :])
                self._append_static(state, k_raw_new[:, :, 0], k_new[:, :, 0], pos)
                out_t = self._decode_static(q_new[:, :, 0], state, pos)  # [B,H,Dh]
                out = self.o_proj(out_t.reshape(B, 1, self.nhead * self.head_dim))
                return out, None

        past_len = self._input_start_pos(past_key_value, layer_idx, cache_position)
        cache_kwargs = {"sin": sin_qwen, "cos": cos_qwen, "cache_position": cache_position}
        try:
            k_cache, v_cache = past_key_value.update(k_new, v_new, layer_idx, cache_kwargs)
        except TypeError:
            k_cache, v_cache = past_key_value.update(k_new, v_new, layer_idx)
        total_len = self._input_end_pos(k_cache, past_len, q_len, cache_position)
        max_cache_len = self._max_cache_len_from_cache(past_key_value, k_cache, total_len)

        # ---- Prefill (no prior context): compute output, then seed decode state ----
        if past_len == 0:
            out = self._forward_no_cache(q_raw, q_new, k_raw_new, k_new, v_new, cos, sin, attention_mask)[0]
            state = self._build_state_from_prefill(
                q_new, k_raw_new, k_new, v_new, k_cache, v_cache, total_len, max_cache_len
            )
            self._set_state(past_key_value, layer_idx, state)
            return out, None

        state = self._get_state(past_key_value, layer_idx)
        if state is None or int(state.seen) != past_len:
            state = self._rebuild_state_from_cache(k_cache, v_cache, past_len, max_cache_len)
            self._set_state(past_key_value, layer_idx, state)

        # ---- Decode: strictly causal, one token at a time ----
        outs: List[torch.Tensor] = []
        for i in range(q_len):
            abs_pos = self._abs_pos_for_token(past_len, i, cache_position)
            self._append_active(state, k_raw_new[:, :, i], k_new[:, :, i], abs_pos)
            outs.append(self._decode_one(q_new[:, :, i], state, k_cache, v_cache, abs_pos).unsqueeze(2))
        out_seq = outs[0] if q_len == 1 else torch.cat(outs, dim=2)
        out = self.o_proj(out_seq.transpose(1, 2).contiguous().reshape(B, q_len, self.nhead * self.head_dim))
        return out, None

    # ------------------------------------------------------------------
    # No-cache path: fused kernel when applicable, else dense torch
    # ------------------------------------------------------------------

    def _fused_applicable(self) -> bool:
        C, M, D = self.chunk_size, self.groups_per_chunk, self.head_dim
        return (
            not (self.training and self.dropout_p > 0.0)
            and self.topk_chunks > 0 and self.topk_groups > 0
            and C >= 16 and (C & (C - 1)) == 0
            # M<16 uses a no-padding scalar-column path for current groups;
            # M>=16 uses tl.dot and therefore remains power-of-two only.
            and M > 0 and (M < 16 or (M & (M - 1)) == 0)
            and D >= 16 and (D & (D - 1)) == 0
        )

    def _forward_no_cache(
        self,
        q_raw: torch.Tensor,
        q_new: torch.Tensor,
        k_raw_new: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        use_fused = q_new.is_cuda and q_new.dtype == torch.float32 and self._fused_applicable()
        k_all = self._repeat_kv(k_new)
        v_all = self._repeat_kv(v_new)
        k_raw_all = self._repeat_kv(k_raw_new)
        
        assert use_fused
        out = self._forward_fused(q_new, k_all, k_raw_all, v_all, cos, sin)
        return out, None

    def _forward_fused(
        self,
        q: torch.Tensor,
        k_rope: torch.Tensor,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S, Dh = q.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = q.device, q.dtype
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

        group_k = self._rope_summary(
            raw=k_raw_chunks.reshape(B, H, N, M, gs, Dh),
            rope=k_chunks.reshape(B, H, N, M, gs, Dh),
            mask=group_token_mask, reduce_dim=4, anchor_pos=group_middle,
            cos=cos, sin=sin, scale=self.group_kv_scale,
        ).contiguous()
        group_v = ((v_chunks.reshape(B, H, N, M, gs, Dh) * group_token_mask).sum(dim=4) * self.group_kv_scale).contiguous()

        with torch.no_grad():
            query_valid = valid_chunks.view(1, 1, N, C, 1)
            chunk_k = self._rope_summary(
                raw=k_raw_chunks, rope=k_chunks, mask=chunk_token_mask, reduce_dim=3,
                anchor_pos=chunk_middle, cos=cos, sin=sin, scale=1.0,
            )
            route_q_chunks = q_p.reshape(B, H, N, C, Dh)

            Kc = min(self.topk_chunks, N)
            scores_roll = torch.einsum("bhncd,bhmd->bhncm", route_q_chunks, chunk_k) * self.scale
            prev_chunk_mask = ar_n[None, :] < ar_n[:, None]
            route_mask = prev_chunk_mask.view(1, 1, N, 1, N) & query_valid
            scores_roll.masked_fill_(~route_mask, _NEG_INF)
            req_idx, req_scores = self._route_topk_requests(scores_roll, Kc, sorted=False)
            chunk_scores = self._max_route_scores_from_requests(req_idx, req_scores, N, _NEG_INF)
            top_chunk_scores, top_chunk_idx = self._topk_scores_indices(chunk_scores, Kc)
            query_chunk_idx = ar_n.view(1, 1, N, 1)
            prev_chunk_idx = (query_chunk_idx - 1).clamp_min(0).expand(B, H, N, 1)
            has_prev = (query_chunk_idx > 0).expand(B, H, N, 1)
            missing_prev = has_prev & ~(top_chunk_idx == prev_chunk_idx).any(dim=-1, keepdim=True)
            replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
            top_chunk_idx = top_chunk_idx.scatter(
                -1, replace_at,
                torch.where(missing_prev, prev_chunk_idx, top_chunk_idx.gather(-1, replace_at)),
            )
            valid_prev = prev_chunk_mask.expand(B, H, N, N).gather(-1, top_chunk_idx)

            ar_c32 = ar_c.to(torch.int32)
            c_eff = torch.where(
                req_scores > _FINITE_MIN, ar_c32.view(1, 1, 1, C, 1),
                torch.tensor(BIG_INT, device=device, dtype=torch.int32),
            )
            route_first = torch.full((B * H * N, N), BIG_INT, device=device, dtype=torch.int32)
            route_first.scatter_reduce_(1, req_idx.reshape(B * H * N, -1),
                                        c_eff.expand_as(req_idx).reshape(B * H * N, -1),
                                        reduce="amin", include_self=True)
            thrA = route_first.view(B, H, N, N).gather(-1, top_chunk_idx).long()
            is_prev_slot = (top_chunk_idx == prev_chunk_idx) & has_prev
            thrA = torch.where(is_prev_slot, torch.zeros_like(thrA), thrA)
            thrA = torch.where(valid_prev, thrA, torch.full_like(thrA, BIG_INT))

            cand_k_flat = self._gather_chunks(group_k, top_chunk_idx).reshape(B, H, N, Kc * M, Dh)
            cand_group_valid = valid_groups[top_chunk_idx] & valid_prev.unsqueeze(-1)
            Tgrp = Kc * M
            gsr = torch.einsum("bhncd,bhnrd->bhncr", route_q_chunks, cand_k_flat) * self.scale
            visA = ar_c.view(1, 1, 1, C, 1) >= thrA.view(B, H, N, 1, Kc)
            gs_mask = (visA.unsqueeze(-1) & cand_group_valid.unsqueeze(3)).reshape(B, H, N, C, Tgrp) & query_valid
            gsr.masked_fill_(~gs_mask, _NEG_INF)

            Kg = min(self.topk_groups, Tgrp)
            Kg_req = min(self.topk_groups // 2, Tgrp)
            if Kg > 0 and Kg_req > 0:
                g_req_idx, g_req_scores = self._route_topk_requests(gsr, Kg_req, sorted=False)
                group_scores = self._max_route_scores_from_requests(g_req_idx, g_req_scores, Tgrp, _NEG_INF)
                _, top_group_idx = self._topk_scores_indices(group_scores, Kg)
                c_eff_g = torch.where(
                    g_req_scores > _FINITE_MIN, ar_c32.view(1, 1, 1, C, 1),
                    torch.tensor(BIG_INT, device=device, dtype=torch.int32),
                )
                route_first_g = torch.full((B * H * N, Tgrp), BIG_INT, device=device, dtype=torch.int32)
                route_first_g.scatter_reduce_(1, g_req_idx.reshape(B * H * N, -1),
                                              c_eff_g.expand_as(g_req_idx).reshape(B * H * N, -1),
                                              reduce="amin", include_self=True)
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

        out_p = _HAFusedFn.apply(q_p, k_p, v_p, group_k, group_v, idxA32, thrA32, posD32, thrD32,
                                 S, self.scale, C, M, gs, self.fp32_precise)
        return self.o_proj(out_p[:, :, :S, :].transpose(1, 2).contiguous().reshape(B, S, H * Dh))

    # ------------------------------------------------------------------
    # Decode: simplified strictly-causal single-token core
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _append_active(
        self,
        state: _HGAState,
        k_raw_t: torch.Tensor,
        k_rope_t: torch.Tensor,
        abs_pos: int,
    ) -> None:
        """Fold the new token into the active chunk's running key summaries.

        Resets the per-chunk accumulators (sums + routing running-max) at chunk
        boundaries, updates the active group summary, and finalizes the chunk
        summary once the chunk closes.  ``k_raw_t``/``k_rope_t``: [B, KVH, Dh].
        """
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        n = abs_pos // C
        local = abs_pos - n * C
        g = local // gs
        B, KVH, Dh = k_raw_t.shape
        device = k_raw_t.device

        if n >= state.max_chunks:
            self._grow_state(state, n + 1)

        if local == 0:
            state.cur_chunk_raw.zero_()
            state.cur_chunk_rope.zero_()
            state.chunk_smax.fill_(_NEG_INF)
            state.group_smax.fill_(_NEG_INF)
        if local % gs == 0:
            state.cur_group_raw.zero_()
            state.cur_group_rope.zero_()

        state.cur_chunk_raw.add_(k_raw_t)
        state.cur_chunk_rope.add_(k_rope_t)
        state.cur_group_raw.add_(k_raw_t)
        state.cur_group_rope.add_(k_rope_t)

        g_anchor = torch.tensor(n * C + g * gs + (local % gs) // 2, device=device)
        gcos, gsin = self._rotary_at(g_anchor, B, device)
        group_anchor = self._apply_rotary((state.cur_group_raw * self.group_kv_scale).float(), gcos, gsin).to(k_raw_t.dtype)
        state.group_k[:, :, n, g, :] = self._mix_tokenwise_and_anchor(
            state.cur_group_rope * self.group_kv_scale, group_anchor)

        if local == C - 1:
            c_anchor = torch.tensor(n * C + (C - 1) // 2, device=device)
            ccos, csin = self._rotary_at(c_anchor, B, device)
            chunk_anchor = self._apply_rotary(state.cur_chunk_raw.float(), ccos, csin).to(k_raw_t.dtype)
            state.chunk_k[:, :, n, :] = self._mix_tokenwise_and_anchor(state.cur_chunk_rope, chunk_anchor)
        state.seen = int(abs_pos) + 1

    @torch.no_grad()
    def _decode_one(
        self,
        q_t: torch.Tensor,
        state: _HGAState,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        abs_pos: int,
    ) -> torch.Tensor:
        """Attention output for one new token.  ``q_t``: [B, H, Dh] (rope applied)."""
        B, H, Dh = q_t.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        rep = self.num_key_value_groups
        device, dtype = q_t.device, q_t.dtype
        scale = self.scale
        total_len = abs_pos + 1
        n = abs_pos // C
        c = abs_pos - n * C
        ar_m = torch.arange(M, device=device)

        scores: List[torch.Tensor] = []
        values: List[torch.Tensor] = []

        # ---- segments A + D: previously / now opened chunks & groups ----
        cand_v_flat = None
        n_prev = 0
        if n > 0 and self.topk_chunks > 0:
            ck = self._rep_heads(state.chunk_k[:, :, :n, :], rep)          # [B,H,n,Dh]
            sc_ck = torch.einsum("bhd,bhjd->bhj", q_t, ck) * scale          # [B,H,n]
            Kc_own = min(self.topk_chunks, n)
            own_scores, own_idx = torch.topk(sc_ck, Kc_own, dim=-1, sorted=False)
            smax_view = state.chunk_smax[:, :, :n]
            smax_view.scatter_reduce_(-1, own_idx, own_scores, reduce="amax", include_self=True)

            prev = n - 1
            Kc = n if self.decode_route_overflow == "all" else min(self.topk_chunks, n)
            sel_scores, cand_idx = torch.topk(smax_view, Kc, dim=-1, sorted=False)  # [B,H,Kc]
            prev_t = torch.full_like(cand_idx[..., :1], prev)
            missing = ~(cand_idx == prev).any(dim=-1, keepdim=True)
            replace_at = sel_scores.argmin(dim=-1, keepdim=True)
            cand_idx = cand_idx.scatter(-1, replace_at,
                                        torch.where(missing, prev_t, cand_idx.gather(-1, replace_at)))
            cand_valid = (smax_view.gather(-1, cand_idx) > _FINITE_MIN) | (cand_idx == prev)  # [B,H,Kc]

            # segment A: group summaries of candidate chunks
            cand_gk = self._gather_kvhead_chunks(state.group_k, cand_idx).reshape(B, H, Kc * M, Dh)  # [B,H,Kc*M,Dh]
            chunk_grp_idx = cand_idx.unsqueeze(-1).expand(B, H, Kc, M)
            grp_ids = ar_m.view(1, 1, 1, M).expand(B, H, Kc, M)
            cand_v_flat = self._gather_group_values(v_cache, chunk_grp_idx, grp_ids, total_len).reshape(B, H, Kc * M, Dh)
            sc_A = torch.einsum("bhd,bhrd->bhr", q_t, cand_gk) * scale       # [B,H,Kc*M]
            mask_A = cand_valid.unsqueeze(-1).expand(B, H, Kc, M).reshape(B, H, Kc * M)
            sc_A = sc_A.masked_fill(~mask_A, _NEG_INF)
            scores.append(sc_A)
            values.append(cand_v_flat)
            n_prev = Kc * M

            # ---- segment D: open candidate groups to exact tokens ----
            if self.topk_groups > 0:
                Tg = Kc * M
                Kg = Tg if self.decode_route_overflow == "all" else min(self.topk_groups, Tg)
                Kg_req = Tg if self.decode_route_overflow == "all" else min(self.topk_groups // 2, Tg)
                # update group running-max at absolute (chunk, group) positions
                gsmax_view = state.group_smax  # [B,H,Nmax,M]
                own_g_scores, own_g_idx = torch.topk(sc_A, min(Kg_req, Tg), dim=-1, sorted=False)
                own_parent = own_g_idx // M
                own_m = own_g_idx - own_parent * M
                own_abs_chunk = cand_idx.gather(-1, own_parent)
                own_lin = own_abs_chunk * M + own_m                          # into [Nmax*M]
                # only scatter requests whose score is real (visible candidate)
                own_g_scores = torch.where(own_g_scores > _FINITE_MIN, own_g_scores,
                                           torch.full_like(own_g_scores, _NEG_INF))
                gsmax_flat = gsmax_view.reshape(B, H, -1)
                gsmax_flat.scatter_reduce_(-1, own_lin, own_g_scores, reduce="amax", include_self=True)

                # candidate groups = top-Kg of running-max restricted to candidate chunks
                cand_gsmax = self._gather_kvhead_chunks(state.group_smax, cand_idx).reshape(B, H, Tg)  # [B,H,Kc*M]
                cand_gsmax = cand_gsmax.masked_fill(~mask_A, _NEG_INF)
                _, top_g = torch.topk(cand_gsmax, Kg, dim=-1, sorted=False)   # [B,H,Kg] indices into Kc*M
                g_valid = cand_gsmax.gather(-1, top_g) > _FINITE_MIN
                parent = top_g // M
                src_chunk = cand_idx.gather(-1, parent)
                src_grp = top_g - parent * M
                pos = src_chunk * C + src_grp * gs                            # [B,H,Kg]
                pos = pos.unsqueeze(-1) + torch.arange(gs, device=device).view(1, 1, 1, gs)
                pos = pos.reshape(B, H, Kg * gs)
                pos_safe = pos.clamp_max(k_cache.shape[-2] - 1)
                opened_k = self._rep_heads_tokens(self._gather_kv_tokens(k_cache, pos_safe), 1)
                opened_v = self._gather_kv_tokens(v_cache, pos_safe)
                sc_D = torch.einsum("bhd,bhrd->bhr", q_t, opened_k) * scale
                valid_tok = g_valid.unsqueeze(-1).expand(B, H, Kg, gs).reshape(B, H, Kg * gs) & (pos < total_len)
                sc_D = sc_D.masked_fill(~valid_tok, _NEG_INF)
                scores.append(sc_D)
                values.append(opened_v)

        # ---- segment B: completed group summaries of the active chunk ----
        ncomp = (c + 1) // gs
        if ncomp > 0:
            gk_act = self._rep_heads(state.group_k[:, :, n, :ncomp, :], rep)  # [B,H,ncomp,Dh]
            chunk_idx_b = torch.full((B, H, ncomp), n, device=device, dtype=torch.long)
            grp_idx_b = torch.arange(ncomp, device=device).view(1, 1, ncomp).expand(B, H, ncomp)
            gv_act = self._gather_group_values(v_cache, chunk_idx_b, grp_idx_b, total_len)  # [B,H,ncomp,Dh]
            sc_B = torch.einsum("bhd,bhgd->bhg", q_t, gk_act) * scale
            # only complete groups are visible (each group's gs tokens all present)
            scores.append(sc_B)
            values.append(gv_act)
        n_cur_summary = ncomp

        # ---- segment C: exact tokens of the active chunk (causal) ----
        cur_k = self._rep_heads(k_cache[:, :, n * C: abs_pos + 1, :], rep)    # [B,H,c+1,Dh]
        cur_v = self._rep_heads(v_cache[:, :, n * C: abs_pos + 1, :], rep)
        sc_C = torch.einsum("bhd,bhtd->bht", q_t, cur_k) * scale
        scores.append(sc_C)
        values.append(cur_v)

        sc = torch.cat(scores, dim=-1)
        val = torch.cat(values, dim=-2)
        probs = torch.softmax(sc.float(), dim=-1).to(dtype)
        return torch.einsum("bhr,bhrd->bhd", probs, val)

    # ------------------------------------------------------------------
    # Static / branchless decode (CUDA-graph friendly).  ``pos`` is a GPU long
    # scalar tensor; no python-int shapes, no .item() syncs, fixed buffers.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _append_static(self, state: _HGAState, k_raw_t: torch.Tensor,
                       k_rope_t: torch.Tensor, pos: torch.Tensor) -> None:
        """Branchless equivalent of ``_append_active`` (``pos`` is a GPU scalar)."""
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        B, KVH, Dh = k_raw_t.shape
        dtype = k_raw_t.dtype

        n = torch.div(pos, C, rounding_mode="floor")
        local = pos - n * C
        g = torch.div(local, gs, rounding_mode="floor")
        loc_in_g = local - g * gs

        is_chunk_start = (local == 0)
        is_group_start = (loc_in_g == 0)
        is_chunk_end = (local == C - 1)

        keep_chunk = (~is_chunk_start).to(dtype)
        keep_group = (~is_group_start).to(dtype)
        # Reset-then-accumulate the running raw/rope sums.
        state.cur_chunk_raw.mul_(keep_chunk).add_(k_raw_t)
        state.cur_chunk_rope.mul_(keep_chunk).add_(k_rope_t)
        state.cur_group_raw.mul_(keep_group).add_(k_raw_t)
        state.cur_group_rope.mul_(keep_group).add_(k_rope_t)

        # Reset routing running-max at chunk boundaries.
        neg_c = torch.full_like(state.chunk_smax, _NEG_INF)
        state.chunk_smax.copy_(torch.where(is_chunk_start, neg_c, state.chunk_smax))
        neg_g = torch.full_like(state.group_smax, _NEG_INF)
        state.group_smax.copy_(torch.where(is_chunk_start, neg_g, state.group_smax))

        # Update the active group summary (write into slot n*M + g).
        g_anchor = n * C + g * gs + torch.div(loc_in_g, 2, rounding_mode="floor")
        gcos, gsin = self._rotary_at(g_anchor, B, k_raw_t.device)
        group_anchor = self._apply_rotary(
            (state.cur_group_raw * self.group_kv_scale).float(), gcos, gsin).to(dtype)
        group_val = self._mix_tokenwise_and_anchor(
            state.cur_group_rope * self.group_kv_scale, group_anchor)  # [B,KVH,Dh]
        gk_flat = state.group_k.view(B, KVH, state.max_chunks * M, Dh)
        slot_g = (n * M + g).view(1)
        gk_flat.index_copy_(2, slot_g, group_val.unsqueeze(2))

        # Finalize the chunk summary when the chunk closes (slot n), else no-op.
        c_anchor = n * C + (C - 1) // 2
        ccos, csin = self._rotary_at(c_anchor, B, k_raw_t.device)
        chunk_anchor = self._apply_rotary(state.cur_chunk_raw.float(), ccos, csin).to(dtype)
        chunk_val = self._mix_tokenwise_and_anchor(state.cur_chunk_rope, chunk_anchor)
        slot_c = n.view(1)
        cur_chunk_slot = state.chunk_k.index_select(2, slot_c)  # [B,KVH,1,Dh]
        blended = torch.where(is_chunk_end, chunk_val.unsqueeze(2), cur_chunk_slot)
        state.chunk_k.index_copy_(2, slot_c, blended)

    @torch.no_grad()
    def _decode_static(self, q_t: torch.Tensor, state: _HGAState, pos: torch.Tensor) -> torch.Tensor:
        """Branchless equivalent of ``_decode_one``.  ``q_t``: [B,H,Dh] (rope applied)."""
        B, H, Dh = q_t.shape
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        MAXN = state.max_chunks
        rep = self.num_key_value_groups
        device, dtype = q_t.device, q_t.dtype
        scale = self.scale
        k_cache, v_cache = state.kbuf, state.vbuf

        n = torch.div(pos, C, rounding_mode="floor")
        c = pos - n * C
        total_len = pos + 1
        ncomp = torch.div(c + 1, gs, rounding_mode="floor")
        prev = n - 1
        ar_n = torch.arange(MAXN, device=device)
        ar_m = torch.arange(M, device=device)
        ar_c = torch.arange(C, device=device)

        Kc = min(self.topk_chunks, MAXN)
        scores: List[torch.Tensor] = []
        values: List[torch.Tensor] = []

        # ---- segments A + D over closed candidate chunks (idx < n) ----
        valid_closed = ar_n < n  # [MAXN]
        ck = self._rep_heads(state.chunk_k, rep)                      # [B,H,MAXN,Dh]
        sc_ck = torch.einsum("bhd,bhjd->bhj", q_t, ck) * scale        # [B,H,MAXN]
        sc_ck = sc_ck.masked_fill(~valid_closed.view(1, 1, MAXN), _NEG_INF)
        own_scores, own_idx = torch.topk(sc_ck, Kc, dim=-1, sorted=False)
        state.chunk_smax.scatter_reduce_(-1, own_idx, own_scores, reduce="amax", include_self=True)

        sel_scores, cand_idx = torch.topk(state.chunk_smax, Kc, dim=-1, sorted=False)  # [B,H,Kc]
        prev_valid = (prev >= 0)
        prev_clamped = prev.clamp_min(0)
        eq_prev = (cand_idx == prev)
        missing = prev_valid & ~eq_prev.any(dim=-1, keepdim=True)
        replace_at = sel_scores.argmin(dim=-1, keepdim=True)
        prev_fill = torch.where(missing, prev_clamped.expand_as(replace_at),
                                cand_idx.gather(-1, replace_at))
        cand_idx = cand_idx.scatter(-1, replace_at, prev_fill)
        cand_valid = (state.chunk_smax.gather(-1, cand_idx) > _FINITE_MIN) | \
                     (prev_valid & (cand_idx == prev))                # [B,H,Kc]

        cand_gk = self._gather_kvhead_chunks(state.group_k, cand_idx).reshape(B, H, Kc * M, Dh)
        chunk_grp_idx = cand_idx.unsqueeze(-1).expand(B, H, Kc, M)
        grp_ids = ar_m.view(1, 1, 1, M).expand(B, H, Kc, M)
        cand_v_flat = self._gather_group_values(v_cache, chunk_grp_idx, grp_ids, total_len).reshape(B, H, Kc * M, Dh)
        sc_A = torch.einsum("bhd,bhrd->bhr", q_t, cand_gk) * scale
        mask_A = cand_valid.unsqueeze(-1).expand(B, H, Kc, M).reshape(B, H, Kc * M)
        sc_A = sc_A.masked_fill(~mask_A, _NEG_INF)
        scores.append(sc_A)
        values.append(cand_v_flat)

        if self.topk_groups > 0:
            Tg = Kc * M
            Kg = min(self.topk_groups, Tg)
            Kg_req = min(self.topk_groups // 2, Tg)
            own_g_scores, own_g_idx = torch.topk(sc_A, Kg_req, dim=-1, sorted=False)
            own_parent = torch.div(own_g_idx, M, rounding_mode="floor")
            own_m = own_g_idx - own_parent * M
            own_abs_chunk = cand_idx.gather(-1, own_parent)
            own_lin = own_abs_chunk * M + own_m
            own_g_scores = torch.where(own_g_scores > _FINITE_MIN, own_g_scores,
                                       torch.full_like(own_g_scores, _NEG_INF))
            gsmax_flat = state.group_smax.reshape(B, H, -1)
            gsmax_flat.scatter_reduce_(-1, own_lin, own_g_scores, reduce="amax", include_self=True)

            cand_gsmax = self._gather_kvhead_chunks(state.group_smax, cand_idx).reshape(B, H, Tg)
            cand_gsmax = cand_gsmax.masked_fill(~mask_A, _NEG_INF)
            _, top_g = torch.topk(cand_gsmax, Kg, dim=-1, sorted=False)
            g_valid = cand_gsmax.gather(-1, top_g) > _FINITE_MIN
            parent = torch.div(top_g, M, rounding_mode="floor")
            src_chunk = cand_idx.gather(-1, parent)
            src_grp = top_g - parent * M
            pos_d = src_chunk * C + src_grp * gs
            pos_d = pos_d.unsqueeze(-1) + torch.arange(gs, device=device).view(1, 1, 1, gs)
            pos_d = pos_d.reshape(B, H, Kg * gs)
            pos_safe = pos_d.clamp_max(k_cache.shape[-2] - 1)
            opened_k = self._gather_kv_tokens(k_cache, pos_safe)
            opened_v = self._gather_kv_tokens(v_cache, pos_safe)
            sc_D = torch.einsum("bhd,bhrd->bhr", q_t, opened_k) * scale
            valid_tok = g_valid.unsqueeze(-1).expand(B, H, Kg, gs).reshape(B, H, Kg * gs) & (pos_d < total_len)
            sc_D = sc_D.masked_fill(~valid_tok, _NEG_INF)
            scores.append(sc_D)
            values.append(opened_v)

        # ---- segment B: completed group summaries of the active chunk n ----
        slot_n = n.view(1)
        gk_n = state.group_k.index_select(2, slot_n).squeeze(2)       # [B,KVH,M,Dh]
        gk_act = self._rep_heads(gk_n, rep)                          # [B,H,M,Dh]
        chunk_idx_b = n.view(1, 1, 1).expand(B, H, M)
        grp_idx_b = ar_m.view(1, 1, M).expand(B, H, M)
        gv_act = self._gather_group_values(v_cache, chunk_idx_b, grp_idx_b, total_len)  # [B,H,M,Dh]
        sc_B = torch.einsum("bhd,bhgd->bhg", q_t, gk_act) * scale
        mask_B = (ar_m.view(1, 1, M) < ncomp)
        sc_B = sc_B.masked_fill(~mask_B, _NEG_INF)
        scores.append(sc_B)
        values.append(gv_act)

        # ---- segment C: exact tokens of the active chunk (causal), full C window ----
        base = n * C
        idx_c = (base + ar_c).clamp_max(k_cache.shape[-2] - 1)        # [C]
        cur_k = self._rep_heads(k_cache.index_select(2, idx_c), rep)  # [B,H,C,Dh]
        cur_v = self._rep_heads(v_cache.index_select(2, idx_c), rep)
        sc_C = torch.einsum("bhd,bhtd->bht", q_t, cur_k) * scale
        mask_C = (ar_c.view(1, 1, C) <= c)
        sc_C = sc_C.masked_fill(~mask_C, _NEG_INF)
        scores.append(sc_C)
        values.append(cur_v)

        sc = torch.cat(scores, dim=-1)
        val = torch.cat(values, dim=-2)
        probs = torch.softmax(sc.float(), dim=-1).to(dtype)
        return torch.einsum("bhr,bhrd->bhd", probs, val)

    # ------------------------------------------------------------------
    # State build / management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _alloc_state(self, B: int, max_chunks: int, device: torch.device, dtype: torch.dtype) -> _HGAState:
        KVH, Dh, M = self.kv_heads, self.head_dim, self.groups_per_chunk
        C = self.chunk_size
        # Static decode needs fixed-size buffers that never reallocate during the
        # whole generation (so CUDA-graph capture stays valid).  Size to a max.
        max_seq = max(int(self.decode_max_seq), int(max_chunks) * C)
        max_chunks = max(int(max_chunks), (max_seq + C - 1) // C)
        return _HGAState(
            seen=0, max_chunks=int(max_chunks), max_seq=int(max_seq),
            chunk_k=torch.zeros(B, KVH, max_chunks, Dh, device=device, dtype=dtype),
            group_k=torch.zeros(B, KVH, max_chunks, M, Dh, device=device, dtype=dtype),
            cur_chunk_raw=torch.zeros(B, KVH, Dh, device=device, dtype=dtype),
            cur_chunk_rope=torch.zeros(B, KVH, Dh, device=device, dtype=dtype),
            cur_group_raw=torch.zeros(B, KVH, Dh, device=device, dtype=dtype),
            cur_group_rope=torch.zeros(B, KVH, Dh, device=device, dtype=dtype),
            chunk_smax=torch.full((B, self.nhead, max_chunks), _NEG_INF, device=device, dtype=torch.float32),
            group_smax=torch.full((B, self.nhead, max_chunks, M), _NEG_INF, device=device, dtype=torch.float32),
            kbuf=torch.zeros(B, KVH, max_seq, Dh, device=device, dtype=dtype),
            vbuf=torch.zeros(B, KVH, max_seq, Dh, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def _grow_state(self, state: _HGAState, min_chunks: int) -> None:
        old = state.max_chunks
        new = max(int(min_chunks), max(1, old * 2))
        B, KVH, _, Dh = state.chunk_k.shape
        M = self.groups_per_chunk
        H = self.nhead
        def grow4(t, fill):
            n = torch.full((B, t.shape[1], new) + tuple(t.shape[3:]), fill, device=t.device, dtype=t.dtype)
            n[:, :, :old] = t
            return n
        ck = state.chunk_k.new_zeros(B, KVH, new, Dh); ck[:, :, :old] = state.chunk_k
        gk = state.group_k.new_zeros(B, KVH, new, M, Dh); gk[:, :, :old] = state.group_k
        cs = state.chunk_smax.new_full((B, H, new), _NEG_INF); cs[:, :, :old] = state.chunk_smax
        gsm = state.group_smax.new_full((B, H, new, M), _NEG_INF); gsm[:, :, :old] = state.group_smax
        state.chunk_k, state.group_k, state.chunk_smax, state.group_smax = ck, gk, cs, gsm
        state.max_chunks = new

    @torch.no_grad()
    def _build_state_from_prefill(
        self,
        q_new: torch.Tensor,
        k_raw_new: torch.Tensor,
        k_rope_new: torch.Tensor,
        v_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_len: int,
        max_cache_len: int,
    ) -> _HGAState:
        """Seed decode state after a prefill forward over positions 0..total_len-1.

        Closed-chunk summaries are computed vectorized; the partial last chunk is
        replayed token-by-token through the decode path so its running sums and
        opened-id running-max match exactly what decoding produced.
        """
        B = q_new.shape[0]
        KVH, Dh = self.kv_heads, self.head_dim
        C, gs, M = self.chunk_size, self.group_size, self.groups_per_chunk
        device, dtype = k_raw_new.device, k_raw_new.dtype
        n_closed = total_len // C
        max_chunks = max(n_closed + 1, (max(1, int(max_cache_len)) + C - 1) // C)
        state = self._alloc_state(B, max_chunks, device, dtype)

        # Seed the self-contained static KV buffers from the prefill cache.
        seed = min(int(total_len), state.max_seq)
        state.kbuf[:, :, :seed, :] = k_rope_new[:, :, :seed, :]
        state.vbuf[:, :, :seed, :] = v_new[:, :, :seed, :]

        if n_closed > 0:
            end = n_closed * C
            kraw = k_raw_new[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            krope = k_rope_new[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            ar_n = torch.arange(n_closed, device=device)
            ar_m = torch.arange(M, device=device)
            c_anchor = ar_n * C + (C - 1) // 2
            g_anchor = ar_n[:, None] * C + ar_m[None, :] * gs + (gs - 1) // 2
            state.chunk_k[:, :, :n_closed, :] = self._rope_summary_at(kraw, krope, 3, c_anchor, 1.0)
            state.group_k[:, :, :n_closed, :, :] = self._rope_summary_at(
                kraw.reshape(B, KVH, n_closed, M, gs, Dh),
                krope.reshape(B, KVH, n_closed, M, gs, Dh), 4, g_anchor, self.group_kv_scale)

        state.seen = n_closed * C
        # Replay the partial last chunk (<= C-1 tokens) to seed active state.
        for p in range(n_closed * C, total_len):
            self._append_active(state, k_raw_new[:, :, p], k_rope_new[:, :, p], p)
            self._decode_one(q_new[:, :, p], state, k_cache, v_cache, p)
        state.seen = int(total_len)
        return state

    @torch.no_grad()
    def _rebuild_state_from_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_len: int,
        max_cache_len: int,
    ) -> _HGAState:
        """Fallback when state is missing but a KV cache exists (rare).

        Reconstructs raw keys by inverting RoPE, then reuses the prefill builder.
        """
        B = k_cache.shape[0]
        device, dtype = k_cache.device, k_cache.dtype
        if total_len == 0:
            max_chunks = max(1, (max(1, int(max_cache_len)) + self.chunk_size - 1) // self.chunk_size)
            return self._alloc_state(B, max_chunks, device, dtype)
        cos, sin = self._build_default_rotary(total_len, device, B)
        krope = k_cache[:, :, :total_len, :]
        kraw = self._apply_rotary_inverse(krope.float(), cos, sin).to(dtype)
        q_dummy = torch.zeros(B, self.nhead, total_len, self.head_dim, device=device, dtype=dtype)
        return self._build_state_from_prefill(q_dummy, kraw, krope, v_cache[:, :, :total_len, :],
                                              k_cache, v_cache, total_len, max_cache_len)

    @staticmethod
    def _hga_slots(cache: Any, layer_idx: int) -> List[Optional[_HGAState]]:
        slots = getattr(cache, "_hga", None)
        if slots is None or not isinstance(slots, list):
            slots = []
            setattr(cache, "_hga", slots)
        while len(slots) <= int(layer_idx):
            slots.append(None)
        return slots

    def _get_state(self, cache: Any, layer_idx: int) -> Optional[_HGAState]:
        slots = getattr(cache, "_hga", None)
        if isinstance(slots, list) and int(layer_idx) < len(slots):
            return slots[int(layer_idx)]
        return None

    def _set_state(self, cache: Any, layer_idx: int, state: _HGAState) -> None:
        self._hga_slots(cache, layer_idx)[int(layer_idx)] = state

    # ------------------------------------------------------------------
    # Cache position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_cache_len_from_cache(cache: Any, k_cache: torch.Tensor, total_len: int) -> int:
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
    def _input_start_pos(cache: Any, layer_idx: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[0].detach().item())
        return GlobalAttention._cache_seq_length(cache, layer_idx)

    @staticmethod
    def _input_end_pos(k_cache: torch.Tensor, past_len: int, q_len: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[-1].detach().item()) + 1
        return min(int(k_cache.shape[-2]), past_len + q_len)

    @staticmethod
    def _abs_pos_for_token(past_len: int, i: int, cache_position: Optional[torch.Tensor]) -> int:
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position[i].detach().item())
        return past_len + i

    def _require_layer_idx(self) -> int:
        if self.layer_idx is None:
            raise ValueError("layer_idx must be set when using a KV cache.")
        return int(self.layer_idx)

    @staticmethod
    def _cache_seq_length(cache: Any, layer_idx: int) -> int:
        if cache is None:
            return 0
        try:
            return int(cache.get_seq_length(layer_idx))
        except TypeError:
            try:
                return int(cache.get_seq_length())
            except Exception:
                return 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Projection / RoPE / summary helpers
    # ------------------------------------------------------------------

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
            B, KVH, self.num_key_value_groups, S, Dh).reshape(B, KVH * self.num_key_value_groups, S, Dh)

    @staticmethod
    def _rep_heads(t: torch.Tensor, rep: int) -> torch.Tensor:
        # t: [B, KVH, *] -> [B, KVH*rep, *] (repeat_interleave along head dim)
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    @staticmethod
    def _rep_heads_tokens(t: torch.Tensor, rep: int) -> torch.Tensor:
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    def _gather_kvhead_chunks(self, tensor: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
        """Gather tensor[B,KVH,N,*tail] at per-query-head chunk_idx[B,H,*idx]."""
        B, KVH, N = tensor.shape[:3]
        H = chunk_idx.shape[1]
        idx_shape = chunk_idx.shape[2:]
        tail = tensor.shape[3:]
        if math.prod(idx_shape) == 0:
            return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
        b_idx = torch.arange(B, device=tensor.device).view(B, 1, *([1] * len(idx_shape))).expand(B, H, *idx_shape)
        kv_idx = (torch.arange(H, device=tensor.device) // self.num_key_value_groups).view(
            1, H, *([1] * len(idx_shape))).expand(B, H, *idx_shape)
        return tensor[b_idx, kv_idx, chunk_idx]

    def _gather_kv_tokens(self, kv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Gather compact KV cache kv[B,KVH,S,D] at pos[B,H,R] -> [B,H,R,D]."""
        B, KVH, S, Dh = kv.shape
        H, R = pos.shape[1], pos.shape[2]
        if R == 0:
            return torch.empty(B, H, 0, Dh, device=kv.device, dtype=kv.dtype)
        b_idx = torch.arange(B, device=kv.device).view(B, 1, 1).expand(B, H, R)
        kv_idx = (torch.arange(H, device=kv.device) // self.num_key_value_groups).view(1, H, 1).expand(B, H, R)
        return kv[b_idx, kv_idx, pos]

    def _gather_group_values(
        self, v_cache: torch.Tensor, chunk_idx: torch.Tensor, group_idx: torch.Tensor, total_len: int
    ) -> torch.Tensor:
        """Per-group value summaries (sum of group tokens * scale) from the cache."""
        B, H = chunk_idx.shape[:2]
        idx_shape = chunk_idx.shape[2:]
        Dh = v_cache.shape[-1]
        if math.prod(idx_shape) == 0:
            return torch.empty(B, H, *idx_shape, Dh, device=v_cache.device, dtype=v_cache.dtype)
        pos = chunk_idx * self.chunk_size + group_idx * self.group_size
        pos = pos.unsqueeze(-1) + torch.arange(self.group_size, device=v_cache.device).view(
            *([1] * pos.ndim), self.group_size)
        flat = pos.reshape(B, H, -1)
        vals = self._gather_kv_tokens(v_cache, flat.clamp_max(v_cache.shape[-2] - 1)).reshape(
            B, H, *idx_shape, self.group_size, Dh)
        mask = (pos < total_len).unsqueeze(-1).to(vals.dtype)
        return (vals * mask).sum(dim=-2) * self.group_kv_scale

    def _rope_summary(self, raw, rope, mask, reduce_dim, anchor_pos, cos, sin, scale) -> torch.Tensor:
        raw_sum = (raw * mask).sum(dim=reduce_dim) * scale
        anchor_cos, anchor_sin = self._gather_rotary(cos, sin, anchor_pos)
        endpoint = self._apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
        tokenwise = (rope * mask).sum(dim=reduce_dim) * scale
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=endpoint)

    def _rope_summary_at(self, raw, rope, reduce_dim, anchor_pos, scale) -> torch.Tensor:
        raw_sum = raw.sum(dim=reduce_dim) * scale
        tokenwise = rope.sum(dim=reduce_dim) * scale
        anchor_cos, anchor_sin = self._rotary_for_positions(anchor_pos, raw_sum)
        endpoint = self._apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
        return self._mix_tokenwise_and_anchor(tokenwise=tokenwise, anchor=endpoint)

    def _rotary_for_positions(self, pos: torch.Tensor, like: torch.Tensor) -> CosSin:
        half = self.head_dim // 2
        device = like.device
        inv_freq = 1.0 / (self.theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        view = (1, 1) + tuple(pos.shape) + (self.head_dim,)
        return emb.cos().reshape(view), emb.sin().reshape(view)

    def _rotary_at(self, pos: torch.Tensor, batch: int, device: torch.device) -> CosSin:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1).reshape(1, 1, self.head_dim)
        cos, sin = emb.cos(), emb.sin()
        if batch != 1:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)
        return cos, sin

    def _mix_tokenwise_and_anchor(self, tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        mask = self._mixed_tokenwise_mask(tokenwise.device).to(dtype=torch.bool)
        view_shape = [1] * tokenwise.ndim
        view_shape[-1] = self.head_dim
        return torch.where(mask.view(*view_shape), tokenwise, anchor)

    def _mixed_cutoff_pair(self) -> int:
        half = self.head_dim // 2
        if self.mixed_rope_cutoff_pair is not None:
            return int(self.mixed_rope_cutoff_pair)
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

    def _normalize_position_embeddings(self, position_embeddings, hidden_states) -> CosSin:
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

    def _build_default_rotary(self, seq_len: int, device: torch.device, batch: int) -> CosSin:
        cos, sin = self._get_rotary(seq_len, device, torch.float32)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        if batch != 1:
            cos = cos.expand(batch, -1, -1, -1)
            sin = sin.expand(batch, -1, -1, -1)
        return cos, sin

    def _get_rotary(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> CosSin:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    @staticmethod
    def _apply_rotary_inverse(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos - torch.cat((-x2, x1), dim=-1) * sin

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

    @staticmethod
    def _gather_groups(tensor: torch.Tensor, chunk_idx: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
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
    def _slice_attention_mask(attention_mask, query_len, key_len):
        if attention_mask is None:
            return None
        if attention_mask.ndim == 4:
            return attention_mask[:, :, -query_len:, :key_len]
        return attention_mask

    @staticmethod
    @torch.no_grad()
    def _all_route_indices(prefix_shape, num_routes, device):
        idx = torch.arange(num_routes, device=device).view(*((1,) * len(prefix_shape)), num_routes)
        return idx.expand(*prefix_shape, num_routes)

    @torch.no_grad()
    def _topk_scores_indices(self, scores, k):
        num_routes = scores.shape[-1]
        assert k > 0
        if k >= num_routes:
            return scores, self._all_route_indices(scores.shape[:-1], num_routes, scores.device)
        return torch.topk(scores, k, dim=-1, sorted=False)

    @torch.no_grad()
    def _route_topk_requests(self, scores, k, sorted=True):
        num_routes = scores.shape[-1]
        assert k > 0
        if k >= num_routes:
            return self._all_route_indices(scores.shape[:-1], num_routes, scores.device), scores
        top_scores, top_idx = torch.topk(scores, k, dim=-1, sorted=sorted)
        return top_idx, top_scores

    @staticmethod
    @torch.no_grad()
    def _max_route_scores_from_requests(request_idx, request_scores, num_routes, fill_value):
        if request_idx.shape[-1] == num_routes:
            return request_scores.max(dim=-2).values
        prefix_shape = request_idx.shape[:-2]
        flat_prefix = math.prod(prefix_shape)
        route_scores = torch.full((flat_prefix, num_routes), fill_value,
                                  device=request_scores.device, dtype=request_scores.dtype)
        route_scores.scatter_reduce_(1, request_idx.reshape(flat_prefix, -1),
                                     request_scores.reshape(flat_prefix, -1), reduce="amax", include_self=True)
        return route_scores.reshape(*prefix_shape, num_routes)

    @staticmethod
    @torch.no_grad()
    def _requests_for_selected_routes(request_idx, selected_idx):
        return (request_idx.unsqueeze(-1) == selected_idx.unsqueeze(-2).unsqueeze(-3)).any(dim=-2)


# Convenience aliases.
GlobalAttentionFused = GlobalAttention
