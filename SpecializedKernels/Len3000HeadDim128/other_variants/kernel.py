"""Triton kernels: ~3K tokens, head_dim=128, GQA, BHSD layout ``[B, H, S, D]``.

* :func:`_vlm_fwd_kernel` — causal self-attention over the prefill sequence.
* :func:`_diffusion_fwd_kernel` — non-causal queries (diffusion tokens) attending
  routed prompt chunks plus all diffusion keys. No ``record_attention``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = 64
HEAD_DIM = 128
MAX_SEQ = 4096
# Max slots for the smallest one-level block (16 tokens → 256 chunks).
N_BLOCK = MAX_SEQ // 16
LOG2E = tl.constexpr(1.4426950408889634)
ROUTE_BLOCKS = (16, 32, 64, 128)


def n_chunks(seqlen: int, chunk: int = CHUNK) -> int:
    return (seqlen + chunk - 1) // chunk

# Tensor-core dtype for QK / PV dots.
# 0=cast-to-bf16, 1=cast-to-fp8e4, 2=cast-to-fp8e5, 3=native fp8 pointers (no cast).
DOT_BF16 = 0
DOT_FP8_E4 = 1
DOT_FP8_E5 = 2
DOT_FP8_NATIVE = 3


@triton.jit
def _dot(a, b, DOT_KIND: tl.constexpr):
    """tl.dot with a common tensor-core dtype.

    Pointers may mix fp32 Q (FP8 linears / LN upcast) and bf16 K. Cast both
    operands in-register so Triton will compile, then pick the tensor-core type.
    """
    if DOT_KIND == 3:
        return tl.dot(a, tl.trans(b))
    if DOT_KIND == 1:
        return tl.dot(a.to(tl.float8e4nv), tl.trans(b.to(tl.float8e4nv)))
    if DOT_KIND == 2:
        return tl.dot(a.to(tl.float8e5), tl.trans(b.to(tl.float8e5)))
    return tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))


@triton.jit
def _upd(s, vv, m_i, l_i, acc, DOT_KIND: tl.constexpr):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2((m_i - m_new) * LOG2E)
    p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
    l_new = l_i * alpha + tl.sum(p, 1)
    if DOT_KIND == 3 or DOT_KIND == 1:
        acc_new = acc * alpha[:, None] + tl.dot(p.to(tl.float8e4nv), vv.to(tl.float8e4nv))
    elif DOT_KIND == 2:
        acc_new = acc * alpha[:, None] + tl.dot(p.to(tl.float8e5), vv.to(tl.float8e5))
    else:
        acc_new = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv.to(tl.bfloat16))
    return m_new, l_new, acc_new


@triton.jit
def _vlm_fwd_kernel(
    Q, K, V, CK, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_ck_b, stride_ck_h,
    stride_o_b, stride_o_h, stride_o_s,
    S, N, HQ, KVH, SCALE,
    TOPK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_KIND: tl.constexpr,
):
    """One CTA = one query chunk × one query head. Causal, GQA."""
    n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    gqa = HQ // KVH
    h_kv = h // gqa

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_N)
    row = n * BLOCK_C + offs_c
    q_mask = row < S

    q_ptr = Q + b * stride_q_b + h * stride_q_h + row[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < N)
    route = _dot(q, ck, DOT_KIND).to(tl.float32) * SCALE
    cand = (offs_t < n) & (offs_t < N)
    route = tl.where(cand[None, :], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_C], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_C], tl.float32)
    acc = tl.zeros([BLOCK_C, BLOCK_D], tl.float32)

    for _ in tl.static_range(TOPK):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_t == j, float("-inf"), pooled)
        valid = has & (j < n)
        k_row = j * BLOCK_C + offs_c
        k_ok = valid & (k_row < S)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=k_ok[:, None])
        vv = tl.load(v_ptr, mask=k_ok[:, None])
        s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
        s = tl.where(valid, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    k_row = n * BLOCK_C + offs_c
    k_mask = k_row < S
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=k_mask[:, None])
    vv = tl.load(v_ptr, mask=k_mask[:, None])
    s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
    causal = (offs_c[None, :] <= offs_c[:, None]) & k_mask[None, :] & q_mask[:, None]
    s = tl.where(causal, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + row[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _diffusion_fwd_kernel(
    Q, K, V, CK, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_ck_b, stride_ck_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, N_CHUNKS, Q_LEN, K_LEN, HQ, KVH, SCALE,
    TOPK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_KIND: tl.constexpr,
):
    """One CTA = one query head. All Q_LEN queries share one routed prompt set.

    Attends: top-k prompt blocks of ``BLOCK_R`` tokens (selected from block-mean
    keys) + every diffusion key at ``[N_PROMPT, N_PROMPT + Q_LEN)``.
    """
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    gqa = HQ // KVH
    h_kv = h // gqa

    offs_c = tl.arange(0, BLOCK_C)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_N)
    q_mask = offs_c < Q_LEN

    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_c[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < N_CHUNKS)
    route = _dot(q, ck, DOT_KIND).to(tl.float32) * SCALE
    route = tl.where((offs_t < N_CHUNKS)[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_C], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_C], tl.float32)
    acc = tl.zeros([BLOCK_C, BLOCK_D], tl.float32)

    for _ in tl.static_range(TOPK):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_t == j, float("-inf"), pooled)
        valid = has & (j < N_CHUNKS)
        k_row = j * BLOCK_R + offs_r
        k_ok = valid & (k_row < N_PROMPT)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=k_ok[:, None])
        vv = tl.load(v_ptr, mask=k_ok[:, None])
        s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
        s = tl.where(valid, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    # Diffusion keys sit immediately after the prompt (if the cache includes them).
    d_row = N_PROMPT + offs_c
    d_ok = (offs_c < Q_LEN) & (d_row < K_LEN)
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + d_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + d_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=d_ok[:, None])
    vv = tl.load(v_ptr, mask=d_ok[:, None])
    s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
    s = tl.where(d_ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + offs_c[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


# Two-level HGA: coarse blocks, then fine tiles inside the winners.
# VLM query tile is the fine size (local causal context), so 2%/4% budgets
# are possible — a dense current coarse block would already be 128 tokens.
HGA_COARSE = 32
HGA_FINE = 8
HGA_FINE_PER_COARSE = HGA_COARSE // HGA_FINE  # 4
HGA_MAX_COARSE = MAX_SEQ // HGA_COARSE  # 128


@triton.jit
def _vlm_hga32_8_kernel(
    Q, K, V, CK32, CK8, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_c32_b, stride_c32_h,
    stride_c8_b, stride_c8_h,
    stride_o_b, stride_o_h, stride_o_s,
    S, N32, HQ, KVH, SCALE,
    TOPK_C: tl.constexpr,
    TOPK_F: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NFINE: tl.constexpr,
    COARSE: tl.constexpr,
    DENSE_CUR: tl.constexpr,
    DOT_KIND: tl.constexpr,
):
    """VLM two-level. ``DENSE_CUR``: one CTA per coarse (full local block).
    Else one CTA per fine tile (local = current 16, needed for 2%/4%)."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    gqa = HQ // KVH
    h_kv = h // gqa
    if DENSE_CUR:
        coarse_id = pid
        local_f = 0
        row = pid * COARSE + tl.arange(0, BLOCK_C)
    else:
        coarse_id = pid // NFINE
        local_f = pid % NFINE
        row = pid * BLOCK_F + tl.arange(0, BLOCK_C)

    offs_c = tl.arange(0, BLOCK_C)
    offs_f = tl.arange(0, 16)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_N)
    offs_u = tl.arange(0, 16)
    q_mask = (offs_c < BLOCK_C) & (row < S)

    q_ptr = Q + b * stride_q_b + h * stride_q_h + row[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    c32_ptr = CK32 + b * stride_c32_b + h_kv * stride_c32_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    c32 = tl.load(c32_ptr, mask=offs_t[:, None] < N32)
    route = _dot(q, c32, DOT_KIND).to(tl.float32) * SCALE
    if DENSE_CUR:
        cand = (offs_t < coarse_id) & (offs_t < N32)
    else:
        cand = (offs_t <= coarse_id) & (offs_t < N32)
    route = tl.where(cand[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_C], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_C], tl.float32)
    acc = tl.zeros([BLOCK_C, BLOCK_D], tl.float32)

    for _ in tl.range(0, TOPK_C):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_t == j, float("-inf"), pooled)
        if DENSE_CUR:
            valid_c = has & (j < coarse_id) & (j < N32)
            fine_ok = offs_u < NFINE
        else:
            valid_c = has & (j <= coarse_id) & (j < N32)
            fine_ok = (offs_u < NFINE) & ((j < coarse_id) | (offs_u < local_f))
        c8_ptr = (
            CK8 + b * stride_c8_b + h_kv * stride_c8_h
            + (j * NFINE + offs_u)[:, None] * BLOCK_D + offs_d[None, :]
        )
        c8 = tl.load(c8_ptr, mask=(offs_u < NFINE)[:, None] & valid_c)
        rf = _dot(q, c8, DOT_KIND).to(tl.float32) * SCALE
        rf = tl.where(fine_ok[None, :] & valid_c, rf, float("-inf"))
        pooled_f = tl.max(rf, axis=0)
        for _ in tl.range(0, TOPK_F):
            has_f = tl.max(pooled_f) > -1.0e20
            t = tl.argmax(pooled_f, axis=0)
            pooled_f = tl.where(offs_u == t, float("-inf"), pooled_f)
            if DENSE_CUR:
                valid = valid_c & has_f & (t < NFINE)
            else:
                valid = valid_c & has_f & (t < NFINE) & ((j < coarse_id) | (t < local_f))
            k_row = j * COARSE + t * BLOCK_F + offs_f
            k_ok = valid & (offs_f < BLOCK_F) & (k_row < S)
            k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
            v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
            kk = tl.load(k_ptr, mask=k_ok[:, None])
            vv = tl.load(v_ptr, mask=k_ok[:, None])
            s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
            s = tl.where(valid & (offs_f < BLOCK_F)[None, :], s, float("-inf"))
            m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    k_row = row
    k_mask = q_mask
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=k_mask[:, None])
    vv = tl.load(v_ptr, mask=k_mask[:, None])
    s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
    causal = (offs_c[None, :] <= offs_c[:, None]) & k_mask[None, :] & q_mask[:, None]
    s = tl.where(causal, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + row[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _diff_hga32_8_kernel(
    Q, K, V, CK32, CK8, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_c32_b, stride_c32_h,
    stride_c8_b, stride_c8_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, N32, Q_LEN, K_LEN, HQ, KVH, SCALE,
    TOPK_C: tl.constexpr,
    TOPK_F: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NFINE: tl.constexpr,
    COARSE: tl.constexpr,
    DOT_KIND: tl.constexpr,
):
    """64 diffusion queries; two-level prompt routing + all diffusion keys."""
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    gqa = HQ // KVH
    h_kv = h // gqa

    offs_c = tl.arange(0, BLOCK_C)
    offs_f = tl.arange(0, 16)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_N)
    offs_u = tl.arange(0, 16)
    q_mask = offs_c < Q_LEN

    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_c[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    c32_ptr = CK32 + b * stride_c32_b + h_kv * stride_c32_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    c32 = tl.load(c32_ptr, mask=offs_t[:, None] < N32)
    route = _dot(q, c32, DOT_KIND).to(tl.float32) * SCALE
    route = tl.where((offs_t < N32)[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_C], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_C], tl.float32)
    acc = tl.zeros([BLOCK_C, BLOCK_D], tl.float32)

    for _ in tl.range(0, TOPK_C):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_t == j, float("-inf"), pooled)
        valid_c = has & (j < N32)
        c8_ptr = (
            CK8 + b * stride_c8_b + h_kv * stride_c8_h
            + (j * NFINE + offs_u)[:, None] * BLOCK_D + offs_d[None, :]
        )
        c8 = tl.load(c8_ptr, mask=(offs_u < NFINE)[:, None] & valid_c)
        rf = _dot(q, c8, DOT_KIND).to(tl.float32) * SCALE
        rf = tl.where((offs_u < NFINE)[None, :] & valid_c, rf, float("-inf"))
        pooled_f = tl.max(rf, axis=0)
        for _ in tl.range(0, TOPK_F):
            has_f = tl.max(pooled_f) > -1.0e20
            t = tl.argmax(pooled_f, axis=0)
            pooled_f = tl.where(offs_u == t, float("-inf"), pooled_f)
            valid = valid_c & has_f & (t < NFINE)
            k_row = j * COARSE + t * BLOCK_F + offs_f
            k_ok = valid & (offs_f < BLOCK_F) & (k_row < N_PROMPT)
            k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
            v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
            kk = tl.load(k_ptr, mask=k_ok[:, None])
            vv = tl.load(v_ptr, mask=k_ok[:, None])
            s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
            s = tl.where(valid & (offs_f < BLOCK_F)[None, :], s, float("-inf"))
            m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    d_row = N_PROMPT + offs_c
    d_ok = (offs_c < Q_LEN) & (d_row < K_LEN)
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + d_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + d_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=d_ok[:, None])
    vv = tl.load(v_ptr, mask=d_ok[:, None])
    s = _dot(q, kk, DOT_KIND).to(tl.float32) * SCALE
    s = tl.where(d_ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc, DOT_KIND)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + offs_c[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _copy_fp8_rows(
    Src, Dst,
    stride_sb, stride_sh, stride_ss,
    stride_db, stride_dh, stride_ds,
    H, S,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    s_idx = r % S
    tmp = r // S
    h_idx = tmp % H
    b_idx = tmp // H
    offs = tl.arange(0, BLOCK_D)
    x = tl.load(Src + b_idx * stride_sb + h_idx * stride_sh + s_idx * stride_ss + offs)
    tl.store(Dst + b_idx * stride_db + h_idx * stride_dh + s_idx * stride_ds + offs, x.to(tl.float8e4nv))


def copy_to_fp8(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Write ``src`` into preallocated ``dst`` as ``float8_e4m3fn``. No new buffer."""
    if dst.dtype != torch.float8_e4m3fn:
        raise TypeError(f"dst must be float8_e4m3fn, got {dst.dtype}")
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch {tuple(src.shape)} vs {tuple(dst.shape)}")
    if src.shape[-1] != HEAD_DIM:
        raise ValueError(f"last dim must be {HEAD_DIM}")
    b, h, s, _ = src.shape
    _copy_fp8_rows[(b * h * s,)](
        src, dst,
        src.stride(0), src.stride(1), src.stride(2),
        dst.stride(0), dst.stride(1), dst.stride(2),
        h, s,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
    )
    return dst


def fill_chunk_keys(
    k: torch.Tensor,
    chunk_k: torch.Tensor,
    seqlen: int,
    chunk: int = CHUNK,
) -> torch.Tensor:
    """Write per-block mean keys into ``chunk_k`` ``[B, KVH, n_slots, D]``."""
    if k.dtype == getattr(torch, "float8_e4m3fn", None):
        raise TypeError("fill_chunk_keys needs bf16/fp32 K; quantize chunk_k afterwards")
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    b, kvh, _, d = k.shape
    n = n_chunks(seqlen, chunk)
    if chunk_k.shape[2] < n:
        raise ValueError(f"chunk_k has {chunk_k.shape[2]} slots, need {n} for S={seqlen} C={chunk}")
    n_full = seqlen // chunk
    if n_full > 0:
        chunk_k[:, :, :n_full].copy_(
            k[:, :, : n_full * chunk, :].reshape(b, kvh, n_full, chunk, d).mean(dim=3)
        )
    rem = seqlen - n_full * chunk
    if rem > 0:
        chunk_k[:, :, n_full].copy_(k[:, :, n_full * chunk : seqlen, :].sum(dim=2) / float(rem))
    if n < chunk_k.shape[2]:
        chunk_k[:, :, n:].zero_()
    return chunk_k
