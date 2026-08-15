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
N_BLOCK = 64  # max chunks = 4096 tokens
LOG2E = tl.constexpr(1.4426950408889634)


def n_chunks(seqlen: int) -> int:
    return (seqlen + CHUNK - 1) // CHUNK


@triton.jit
def _upd(s, vv, m_i, l_i, acc):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2((m_i - m_new) * LOG2E)
    p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
    l_new = l_i * alpha + tl.sum(p, 1)
    acc_new = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
    return m_new, l_new, acc_new


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["TOPK"],
)
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
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)

    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < N, other=0.0)
    route = tl.dot(q, tl.trans(ck)).to(tl.float32) * SCALE
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
        kk = tl.load(k_ptr, mask=k_ok[:, None], other=0.0)
        vv = tl.load(v_ptr, mask=k_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        s = tl.where(valid, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    k_row = n * BLOCK_C + offs_c
    k_mask = k_row < S
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=k_mask[:, None], other=0.0)
    vv = tl.load(v_ptr, mask=k_mask[:, None], other=0.0)
    s = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
    causal = (offs_c[None, :] <= offs_c[:, None]) & k_mask[None, :] & q_mask[:, None]
    s = tl.where(causal, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(q.dtype)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + row[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["TOPK"],
)
@triton.jit
def _diffusion_fwd_kernel(
    Q, K, V, CK, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_ck_b, stride_ck_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, N_CHUNKS, Q_LEN, HQ, KVH, SCALE,
    TOPK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One CTA = one query head. All Q_LEN queries share one routed prompt set.

    Attends: top-k prompt chunks (selected from chunk-mean keys) + every
    diffusion key at ``[N_PROMPT, N_PROMPT + Q_LEN)``. No causal mask.
    """
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    gqa = HQ // KVH
    h_kv = h // gqa

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_N)
    q_mask = offs_c < Q_LEN

    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_c[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)

    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < N_CHUNKS, other=0.0)
    route = tl.dot(q, tl.trans(ck)).to(tl.float32) * SCALE
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
        k_row = j * BLOCK_C + offs_c
        k_ok = valid & (k_row < N_PROMPT)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=k_ok[:, None], other=0.0)
        vv = tl.load(v_ptr, mask=k_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        s = tl.where(valid, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    # Diffusion keys sit immediately after the prompt.
    d_row = N_PROMPT + offs_c
    d_ok = offs_c < Q_LEN
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + d_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + d_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=d_ok[:, None], other=0.0)
    vv = tl.load(v_ptr, mask=d_ok[:, None], other=0.0)
    s = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
    s = tl.where(d_ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(q.dtype)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + offs_c[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


def fill_chunk_keys(k: torch.Tensor, chunk_k: torch.Tensor, seqlen: int) -> torch.Tensor:
    """Write per-chunk mean keys into ``chunk_k`` ``[B, KVH, N_BLOCK, D]``. No new buffer."""
    b, kvh, _, d = k.shape
    n = n_chunks(seqlen)
    n_full = seqlen // CHUNK
    if n_full > 0:
        chunk_k[:, :, :n_full].copy_(
            k[:, :, : n_full * CHUNK, :].reshape(b, kvh, n_full, CHUNK, d).mean(dim=3)
        )
    rem = seqlen - n_full * CHUNK
    if rem > 0:
        chunk_k[:, :, n_full].copy_(k[:, :, n_full * CHUNK : seqlen, :].sum(dim=2) / float(rem))
    if n < chunk_k.shape[2]:
        chunk_k[:, :, n:].zero_()
    return chunk_k
