"""Stock no-ROS 2L kernel. Do not import from the ROS path.

The live ROS kernel is the parent ``Len3000HeadDim128/kernel.py``. This
copy is the **example** diffusion path: route 11 groups, **pack** them
into a contiguous per-head buffer, then attend 64×(176+64). Reuse the
packed K/V across Euler steps — not just the route indices.

Reusing indices alone still gather-attended scattered 16-token groups
from the 3K cache every call. That is why ``HGA_DIFF_ROUTE_EVERY=3``
barely moved end-to-end time.

Selected budget: 4 previous 128-chunks, then 11 groups (176 tokens)
plus all 64 diffusion keys.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HEAD_DIM = 128
CHUNK = 128
GROUP = 16
GPC = CHUNK // GROUP  # 8
TOPK_C = 4
TOPK_G = 11
MAX_SEQ = 4096
MAX_CHUNKS = MAX_SEQ // CHUNK
MAX_GROUPS = MAX_SEQ // GROUP
N_PACK = TOPK_G * GROUP  # 176 prefix tokens after group routing
N_PACK_PAD = 256         # tl.arange power-of-two; last rows are masked
LOG2E = tl.constexpr(1.4426950408889634)


def n_chunks(seqlen: int, chunk: int = CHUNK) -> int:
    return (seqlen + chunk - 1) // chunk


@triton.jit
def _dot_bf16(a, b):
    return tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))


@triton.jit
def _upd(s, vv, m_i, l_i, acc):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2((m_i - m_new) * LOG2E)
    p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
    l_new = l_i * alpha + tl.sum(p, 1)
    acc_new = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv.to(tl.bfloat16))
    return m_new, l_new, acc_new


@triton.jit
def _vlm_hga2_kernel(
    Q, K, V, GK, Route, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_gk_b, stride_gk_h,
    stride_r_b, stride_r_h,
    stride_o_b, stride_o_h, stride_o_s,
    S, NG, HQ, KVH, SCALE,
    TOPK_G: tl.constexpr,
    TOPK_C: tl.constexpr,
    GPC: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CAND: tl.constexpr,
):
    """One CTA = one 16-token query group × one query head."""
    g = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)
    chunk_id = g // GPC
    local_g = g % GPC

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_CAND)
    offs_k = tl.arange(0, 16)
    row = g * GROUP + offs_q
    q_mask = (offs_q < GROUP) & (row < S)

    q_ptr = Q + b * stride_q_b + h * stride_q_h + row[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    n_real = (1 + TOPK_C) * GPC
    loc = offs_c % GPC
    ch = tl.full([BLOCK_CAND], chunk_id, tl.int32)
    for i in tl.static_range(TOPK_C):
        pid = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + i)
        lo = (i + 1) * GPC
        hi = (i + 2) * GPC
        ch = tl.where((offs_c >= lo) & (offs_c < hi), pid, ch)

    cand_g = ch * GPC + loc
    valid = (offs_c < n_real) & (cand_g < NG) & (
        (ch < chunk_id) | ((ch == chunk_id) & (loc < local_g))
    )
    gk_ptr = GK + b * stride_gk_b + h_kv * stride_gk_h + cand_g[:, None] * BLOCK_D + offs_d[None, :]
    gk = tl.load(gk_ptr, mask=valid[:, None])
    route = _dot_bf16(q, gk).to(tl.float32) * SCALE
    route = tl.where(valid[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_Q], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    for _ in tl.range(0, TOPK_G):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_c == j, float("-inf"), pooled)
        j_in_cur = j < GPC
        j_loc = j % GPC
        j_ch = chunk_id
        for i in tl.static_range(TOPK_C):
            pid = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + i)
            lo = (i + 1) * GPC
            hi = (i + 2) * GPC
            j_ch = tl.where((j >= lo) & (j < hi), pid, j_ch)
        j_ch = tl.where(j_in_cur, chunk_id, j_ch)
        sel_g = j_ch * GPC + j_loc
        valid_j = has & (j < n_real) & (sel_g < NG) & (
            (j_ch < chunk_id) | ((j_ch == chunk_id) & (j_loc < local_g))
        )
        k_row = sel_g * GROUP + offs_k
        k_ok = valid_j & (offs_k < GROUP) & (k_row < S)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=k_ok[:, None])
        vv = tl.load(v_ptr, mask=k_ok[:, None])
        s = _dot_bf16(q, kk).to(tl.float32) * SCALE
        s = tl.where(valid_j & (offs_k < GROUP)[None, :], s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    k_row = g * GROUP + offs_q
    k_mask = (offs_q < GROUP) & (k_row < S)
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=k_mask[:, None])
    vv = tl.load(v_ptr, mask=k_mask[:, None])
    s = _dot_bf16(q, kk).to(tl.float32) * SCALE
    causal = (offs_q[None, :] <= offs_q[:, None]) & k_mask[None, :] & q_mask[:, None]
    s = tl.where(causal, s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + row[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _diff_hga2_kernel(
    Q, K, V, GK, Route, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_gk_b, stride_gk_h,
    stride_r_b, stride_r_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, NG, Q_LEN, K_LEN, HQ, KVH, SCALE,
    TOPK_G: tl.constexpr,
    TOPK_C: tl.constexpr,
    GPC: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CAND: tl.constexpr,
):
    """64 diffusion queries share one 4-chunk route; attend opened groups + all diffusion keys."""
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_CAND)
    offs_k = tl.arange(0, 16)
    q_mask = offs_q < Q_LEN

    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_q[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None])

    n_real = TOPK_C * GPC
    loc = offs_c % GPC
    ch = tl.zeros([BLOCK_CAND], tl.int32)
    for i in tl.static_range(TOPK_C):
        pid = tl.load(Route + b * stride_r_b + h * stride_r_h + i)
        lo = i * GPC
        hi = (i + 1) * GPC
        ch = tl.where((offs_c >= lo) & (offs_c < hi), pid, ch)

    cand_g = ch * GPC + loc
    valid = (offs_c < n_real) & (cand_g < NG)
    gk_ptr = GK + b * stride_gk_b + h_kv * stride_gk_h + cand_g[:, None] * BLOCK_D + offs_d[None, :]
    gk = tl.load(gk_ptr, mask=valid[:, None])
    route = _dot_bf16(q, gk).to(tl.float32) * SCALE
    route = tl.where(valid[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, axis=0)

    m_i = tl.full([BLOCK_Q], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    for _ in tl.range(0, TOPK_G):
        has = tl.max(pooled) > -1.0e20
        j = tl.argmax(pooled, axis=0)
        pooled = tl.where(offs_c == j, float("-inf"), pooled)
        j_loc = j % GPC
        j_ch = tl.zeros([], tl.int32)
        for i in tl.static_range(TOPK_C):
            pid = tl.load(Route + b * stride_r_b + h * stride_r_h + i)
            lo = i * GPC
            hi = (i + 1) * GPC
            j_ch = tl.where((j >= lo) & (j < hi), pid, j_ch)
        sel_g = j_ch * GPC + j_loc
        valid_j = has & (j < n_real) & (sel_g < NG)
        k_row = sel_g * GROUP + offs_k
        k_ok = valid_j & (offs_k < GROUP) & (k_row < N_PROMPT)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=k_ok[:, None])
        vv = tl.load(v_ptr, mask=k_ok[:, None])
        s = _dot_bf16(q, kk).to(tl.float32) * SCALE
        s = tl.where(valid_j & (offs_k < GROUP)[None, :], s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    d_row = N_PROMPT + offs_q
    d_ok = (offs_q < Q_LEN) & (d_row < K_LEN)
    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + d_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + d_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=d_ok[:, None])
    vv = tl.load(v_ptr, mask=d_ok[:, None])
    s = _dot_bf16(q, kk).to(tl.float32) * SCALE
    s = tl.where(d_ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + offs_q[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


# ---------------------------------------------------------------------------
# Specialized 128/16 path. Sizes are constexpr; only scores/indices vary.
# ---------------------------------------------------------------------------

BLOCK_CK = 32          # MAX_CHUNKS, SRAM table of 128-token means
BLOCK_CAND = 64        # 5 chunks × 8 groups, padded
BLOCK_ATT = 256        # 11 routed groups + current, padded (arange needs pow2)


@triton.jit
def _fill_means_kernel(
    K, GK, CK,
    stride_k_b, stride_k_h, stride_k_s,
    stride_gk_b, stride_gk_h,
    stride_ck_b, stride_ck_h,
    S, H,
    GROUP: tl.constexpr,
    GPC: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One CTA = one 128-token chunk × one KV head. Writes 8 group means + 1 chunk mean."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_t = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, BLOCK_D)
    row = c * CHUNK + offs_t
    mask = row < S
    k_ptr = K + b * stride_k_b + h * stride_k_h + row[:, None] * stride_k_s + offs_d[None, :]
    k = tl.load(k_ptr, mask=mask[:, None], other=0.0).to(tl.float32)

    ntok = tl.maximum(tl.sum(mask.to(tl.float32)), 1.0)
    ck = tl.sum(k, 0) / ntok
    ck_ptr = CK + b * stride_ck_b + h * stride_ck_h + c * BLOCK_D + offs_d
    tl.store(ck_ptr, ck.to(CK.dtype.element_ty))

    for g in tl.static_range(8):
        gm = (offs_t >= g * GROUP) & (offs_t < (g + 1) * GROUP) & mask
        acc = tl.where(gm[:, None], k, 0.0)
        n = tl.maximum(tl.sum(gm.to(tl.float32)), 1.0)
        mean = tl.sum(acc, 0) / n
        gid = c * GPC + g
        gk_ptr = GK + b * stride_gk_b + h * stride_gk_h + gid * BLOCK_D + offs_d
        tl.store(gk_ptr, mean.to(GK.dtype.element_ty), mask=tl.max(gm.to(tl.int32)) > 0)


@triton.jit
def _chunk_route_vlm_kernel(
    Q, CK, Route,
    stride_q_b, stride_q_h, stride_q_s,
    stride_ck_b, stride_ck_h,
    stride_r_b, stride_r_h,
    S, NC, HQ, KVH, SCALE,
    CHUNK: tl.constexpr,
    TOPK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CK: tl.constexpr,
):
    """One CTA = one query chunk × one query head. Mean-Q vs chunk-K, exact top-4."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)

    offs_q = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_CK)
    row = c * CHUNK + offs_q
    q_mask = row < S
    q_ptr = Q + b * stride_q_b + h * stride_q_h + row[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0).to(tl.float32)
    ntok = tl.maximum(tl.sum(q_mask.to(tl.float32)), 1.0)
    q_mean = (tl.sum(q, 0) / ntok).to(tl.bfloat16)

    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < NC, other=0.0)
    # Pad mean-Q to 16×D so tl.dot uses the same bf16 tensor cores as PyTorch.
    pad_q = tl.zeros((16, BLOCK_D), tl.bfloat16)
    pad_q = tl.where((tl.arange(0, 16) == 0)[:, None], q_mean[None, :], pad_q)
    sc = tl.dot(pad_q, tl.trans(ck.to(tl.bfloat16))).to(tl.float32)
    scores = tl.sum(tl.where((tl.arange(0, 16) == 0)[:, None], sc, 0.0), 0) * SCALE
    scores = tl.where((offs_t < c) & (offs_t < NC), scores, float("-inf"))

    for i in tl.static_range(TOPK_C):
        j = tl.argmax(scores, 0)
        scores = tl.where(offs_t == j, float("-inf"), scores)
        tl.store(Route + b * stride_r_b + h * stride_r_h + c * TOPK_C + i, j.to(tl.int32))


@triton.jit
def _chunk_route_diff_kernel(
    Q, CK, Route,
    stride_q_b, stride_q_h, stride_q_s,
    stride_ck_b, stride_ck_h,
    stride_r_b, stride_r_h,
    NC, Q_LEN, HQ, KVH, SCALE,
    TOPK_C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CK: tl.constexpr,
):
    """One CTA = one query head. Max-pool 64 queries vs prefix chunk means, top-4."""
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_CK)
    q_mask = offs_q < Q_LEN
    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_q[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)
    ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
    ck = tl.load(ck_ptr, mask=offs_t[:, None] < NC, other=0.0)
    route = _dot_bf16(q, ck).to(tl.float32) * SCALE
    route = tl.where((offs_t < NC)[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, 0)
    for i in tl.static_range(TOPK_C):
        j = tl.argmax(pooled, 0)
        pooled = tl.where(offs_t == j, float("-inf"), pooled)
        tl.store(Route + b * stride_r_b + h * stride_r_h + i, j.to(tl.int32))


@triton.jit
def _vlm_hga2_vec_kernel(
    Q, K, V, GK, Route, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_gk_b, stride_gk_h,
    stride_r_b, stride_r_h,
    stride_o_b, stride_o_h, stride_o_s,
    S, NG, HQ, KVH, SCALE,
    TOPK_G: tl.constexpr,
    TOPK_C: tl.constexpr,
    GPC: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CAND: tl.constexpr,
    BLOCK_ATT: tl.constexpr,
):
    """One CTA = one 16-token query group × one head.

    Vectorized group routing (16×64) then one 16×192 gather-attend.
    Candidate layout: [current 8 groups | winner0 8 | winner1 8 | winner2 8 | winner3 8].
    """
    g = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)
    chunk_id = g // GPC
    local_g = g % GPC

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_CAND)
    row = g * GROUP + offs_q
    q_mask = (offs_q < GROUP) & (row < S)
    q_ptr = Q + b * stride_q_b + h * stride_q_h + row[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)

    w0 = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + 0)
    w1 = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + 1)
    w2 = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + 2)
    w3 = tl.load(Route + b * stride_r_b + h * stride_r_h + chunk_id * TOPK_C + 3)

    loc = offs_c % GPC
    slot = offs_c // GPC
    ch = tl.where(slot == 1, w0, chunk_id)
    ch = tl.where(slot == 2, w1, ch)
    ch = tl.where(slot == 3, w2, ch)
    ch = tl.where(slot == 4, w3, ch)
    cand_g = ch * GPC + loc
    n_real = (1 + TOPK_C) * GPC
    valid = (offs_c < n_real) & (cand_g < NG)
    valid = valid & tl.where(
        slot == 0,
        loc < local_g,
        (ch < chunk_id) & (slot > 0) & (slot <= TOPK_C),
    )

    gk_ptr = GK + b * stride_gk_b + h_kv * stride_gk_h + cand_g[:, None] * BLOCK_D + offs_d[None, :]
    gk = tl.load(gk_ptr, mask=valid[:, None], other=0.0)
    route = _dot_bf16(q, gk).to(tl.float32) * SCALE
    route = tl.where(valid[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, 0)

    sel = tl.zeros([16], tl.int32)
    has = tl.zeros([16], tl.int32)
    ar = tl.arange(0, 16)
    for i in tl.static_range(TOPK_G):
        good = (tl.max(pooled) > -1.0e20).to(tl.int32)
        j = tl.argmax(pooled, 0)
        pooled = tl.where(offs_c == j, float("-inf"), pooled)
        j_loc = j % GPC
        j_slot = j // GPC
        j_ch = tl.where(j_slot == 1, w0, chunk_id)
        j_ch = tl.where(j_slot == 2, w1, j_ch)
        j_ch = tl.where(j_slot == 3, w2, j_ch)
        j_ch = tl.where(j_slot == 4, w3, j_ch)
        gid = j_ch * GPC + j_loc
        sel = tl.where(ar == i, gid, sel)
        has = tl.where(ar == i, good, has)
    sel = tl.where(ar == TOPK_G, g, sel)
    has = tl.where(ar == TOPK_G, 1, has)

    # Three 16×64 tensor-core tiles (4 groups each). Last tile is 3 routed + current.
    m_i = tl.full([BLOCK_Q], -1.0e9, tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)
    offs_n = tl.arange(0, 64)
    for t in tl.static_range(3):
        base = t * 4
        g0 = tl.sum(tl.where(ar == base + 0, sel, 0))
        g1 = tl.sum(tl.where(ar == base + 1, sel, 0))
        g2 = tl.sum(tl.where(ar == base + 2, sel, 0))
        g3 = tl.sum(tl.where(ar == base + 3, sel, 0))
        h0 = tl.sum(tl.where(ar == base + 0, has, 0))
        h1 = tl.sum(tl.where(ar == base + 1, has, 0))
        h2 = tl.sum(tl.where(ar == base + 2, has, 0))
        h3 = tl.sum(tl.where(ar == base + 3, has, 0))
        k_row = tl.where(offs_n < 16, g0 * GROUP + offs_n,
                tl.where(offs_n < 32, g1 * GROUP + (offs_n - 16),
                tl.where(offs_n < 48, g2 * GROUP + (offs_n - 32),
                                      g3 * GROUP + (offs_n - 48))))
        ok = tl.where(offs_n < 16, h0 > 0,
             tl.where(offs_n < 32, h1 > 0,
             tl.where(offs_n < 48, h2 > 0, h3 > 0)))
        ok = ok & (k_row < S)
        k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
        v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
        kk = tl.load(k_ptr, mask=ok[:, None], other=0.0)
        vv = tl.load(v_ptr, mask=ok[:, None], other=0.0)
        s = _dot_bf16(q, kk).to(tl.float32) * SCALE
        # Current group is the last 16 columns of tile 2 — apply causal.
        is_cur = (t == 2) & (offs_n >= 48)
        causal = (~is_cur) | (offs_n[None, :] - 48 <= offs_q[:, None])
        s = tl.where(ok[None, :] & q_mask[:, None] & causal, s, float("-inf"))
        m_i, l_i, acc = _upd(s, vv, m_i, l_i, acc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + row[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _diff_hga2_vec_kernel(
    Q, K, V, GK, CK, Sel, PK, PV, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_gk_b, stride_gk_h,
    stride_ck_b, stride_ck_h,
    stride_sel_b, stride_sel_h,
    stride_pk_b, stride_pk_h,
    stride_pv_b, stride_pv_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, NG, NC, Q_LEN, K_LEN, HQ, KVH, SCALE,
    TOPK_G: tl.constexpr,
    TOPK_C: tl.constexpr,
    GPC: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CAND: tl.constexpr,
    BLOCK_ATT: tl.constexpr,
    BLOCK_CK: tl.constexpr,
    REUSE: tl.constexpr,
    PACK: tl.constexpr,
):
    """64 queries: chunk top-4 + group top-11, or reuse stored group ids."""
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_CAND)
    offs_t = tl.arange(0, BLOCK_CK)
    q_mask = offs_q < Q_LEN
    q_ptr = Q + b * stride_q_b + h * stride_q_h + offs_q[:, None] * stride_q_s + offs_d[None, :]
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)

    ar = tl.arange(0, 16)
    sel = tl.zeros([16], tl.int32)
    has = tl.zeros([16], tl.int32)
    if REUSE:
        raw = tl.load(
            Sel + b * stride_sel_b + h * stride_sel_h + ar,
            mask=ar < TOPK_G,
            other=-1,
        )
        has = (raw >= 0).to(tl.int32)
        sel = tl.where(raw >= 0, raw, 0)
    else:
        ck_ptr = CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :]
        ck = tl.load(ck_ptr, mask=offs_t[:, None] < NC, other=0.0)
        cs = _dot_bf16(q, ck).to(tl.float32) * SCALE
        cs = tl.where((offs_t < NC)[None, :] & q_mask[:, None], cs, float("-inf"))
        cpool = tl.max(cs, 0)
        w0 = tl.argmax(cpool, 0)
        cpool = tl.where(offs_t == w0, float("-inf"), cpool)
        w1 = tl.argmax(cpool, 0)
        cpool = tl.where(offs_t == w1, float("-inf"), cpool)
        w2 = tl.argmax(cpool, 0)
        cpool = tl.where(offs_t == w2, float("-inf"), cpool)
        w3 = tl.argmax(cpool, 0)

        loc = offs_c % GPC
        slot = offs_c // GPC
        ch = tl.where(slot == 0, w0, 0)
        ch = tl.where(slot == 1, w1, ch)
        ch = tl.where(slot == 2, w2, ch)
        ch = tl.where(slot == 3, w3, ch)
        cand_g = ch * GPC + loc
        n_real = TOPK_C * GPC
        valid = (offs_c < n_real) & (cand_g < NG) & (slot < TOPK_C)

        gk_ptr = GK + b * stride_gk_b + h_kv * stride_gk_h + cand_g[:, None] * BLOCK_D + offs_d[None, :]
        gk = tl.load(gk_ptr, mask=valid[:, None], other=0.0)
        route = _dot_bf16(q, gk).to(tl.float32) * SCALE
        route = tl.where(valid[None, :] & q_mask[:, None], route, float("-inf"))
        pooled = tl.max(route, 0)

        for i in tl.static_range(TOPK_G):
            good = (tl.max(pooled) > -1.0e20).to(tl.int32)
            j = tl.argmax(pooled, 0)
            pooled = tl.where(offs_c == j, float("-inf"), pooled)
            j_loc = j % GPC
            j_slot = j // GPC
            j_ch = tl.where(j_slot == 0, w0, 0)
            j_ch = tl.where(j_slot == 1, w1, j_ch)
            j_ch = tl.where(j_slot == 2, w2, j_ch)
            j_ch = tl.where(j_slot == 3, w3, j_ch)
            sel = tl.where(ar == i, j_ch * GPC + j_loc, sel)
            has = tl.where(ar == i, good, has)
        tl.store(
            Sel + b * stride_sel_b + h * stride_sel_h + ar,
            tl.where(has > 0, sel, -1),
            mask=ar < TOPK_G,
        )
        if PACK:
            offs_k = tl.arange(0, 16)
            for i in tl.static_range(TOPK_G):
                gid = tl.sum(tl.where(ar == i, sel, 0))
                ok = (tl.sum(tl.where(ar == i, has, 0)) > 0) & (offs_k < GROUP)
                k_row = gid * GROUP + offs_k
                ok = ok & (k_row < N_PROMPT)
                kk = tl.load(
                    K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :],
                    mask=ok[:, None], other=0.0,
                )
                vv = tl.load(
                    V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :],
                    mask=ok[:, None], other=0.0,
                )
                dest = i * GROUP + offs_k
                tl.store(
                    PK + b * stride_pk_b + h * stride_pk_h + dest[:, None] * BLOCK_D + offs_d[None, :],
                    kk, mask=ok[:, None],
                )
                tl.store(
                    PV + b * stride_pv_b + h * stride_pv_h + dest[:, None] * BLOCK_D + offs_d[None, :],
                    vv, mask=ok[:, None],
                )

    # 176 prefix group tokens + 64 diffusion keys, padded to 256.
    offs_n = tl.arange(0, BLOCK_ATT)
    is_diff = offs_n >= (TOPK_G * GROUP)
    gslot = offs_n // GROUP
    pos = offs_n % GROUP
    gid = tl.sum(tl.where(ar[None, :] == gslot[:, None], sel[None, :], 0), 1)
    ok_g = tl.sum(tl.where(ar[None, :] == gslot[:, None], has[None, :], 0), 1) > 0
    pref_row = gid * GROUP + pos
    d_row = N_PROMPT + (offs_n - TOPK_G * GROUP)
    k_row = tl.where(is_diff, d_row, pref_row)
    ok = tl.where(
        is_diff,
        (offs_n - TOPK_G * GROUP < Q_LEN) & (d_row < K_LEN),
        ok_g & (gslot < TOPK_G) & (pref_row < N_PROMPT),
    )

    k_ptr = K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :]
    v_ptr = V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :]
    kk = tl.load(k_ptr, mask=ok[:, None], other=0.0)
    vv = tl.load(v_ptr, mask=ok[:, None], other=0.0)
    s = _dot_bf16(q, kk).to(tl.float32) * SCALE
    s = tl.where(ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i = tl.max(s, 1)
    p = tl.math.exp2((s - m_i[:, None]) * LOG2E)
    p = tl.where(ok[None, :] & q_mask[:, None], p, 0.0)
    l_i = tl.sum(p, 1)
    acc = tl.dot(p.to(tl.bfloat16), vv.to(tl.bfloat16))
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    o_ptr = Out + b * stride_o_b + h * stride_o_h + offs_q[:, None] * stride_o_s + offs_d[None, :]
    tl.store(o_ptr, o, mask=q_mask[:, None])


@triton.jit
def _diff_route_pack_kernel(
    Q, K, V, GK, CK, Sel, PK, PV,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_gk_b, stride_gk_h,
    stride_ck_b, stride_ck_h,
    stride_sel_b, stride_sel_h,
    stride_pk_b, stride_pk_h,
    stride_pv_b, stride_pv_h,
    N_PROMPT, NG, NC, Q_LEN, HQ, KVH, SCALE,
    TOPK_G: tl.constexpr,
    TOPK_C: tl.constexpr,
    GPC: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CAND: tl.constexpr,
    BLOCK_CK: tl.constexpr,
):
    """One CTA = one query head. Chunk top-4, group top-11, pack 176 K/V."""
    bh = tl.program_id(0)
    b = bh // HQ
    h = bh % HQ
    h_kv = h // (HQ // KVH)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_CAND)
    offs_t = tl.arange(0, BLOCK_CK)
    q_mask = offs_q < Q_LEN
    q = tl.load(
        Q + b * stride_q_b + h * stride_q_h + offs_q[:, None] * stride_q_s + offs_d[None, :],
        mask=q_mask[:, None], other=0.0,
    )
    ck = tl.load(
        CK + b * stride_ck_b + h_kv * stride_ck_h + offs_t[:, None] * BLOCK_D + offs_d[None, :],
        mask=offs_t[:, None] < NC, other=0.0,
    )
    cs = _dot_bf16(q, ck).to(tl.float32) * SCALE
    cs = tl.where((offs_t < NC)[None, :] & q_mask[:, None], cs, float("-inf"))
    cpool = tl.max(cs, 0)
    w0 = tl.argmax(cpool, 0)
    cpool = tl.where(offs_t == w0, float("-inf"), cpool)
    w1 = tl.argmax(cpool, 0)
    cpool = tl.where(offs_t == w1, float("-inf"), cpool)
    w2 = tl.argmax(cpool, 0)
    cpool = tl.where(offs_t == w2, float("-inf"), cpool)
    w3 = tl.argmax(cpool, 0)

    loc = offs_c % GPC
    slot = offs_c // GPC
    ch = tl.where(slot == 0, w0, 0)
    ch = tl.where(slot == 1, w1, ch)
    ch = tl.where(slot == 2, w2, ch)
    ch = tl.where(slot == 3, w3, ch)
    cand_g = ch * GPC + loc
    n_real = TOPK_C * GPC
    valid = (offs_c < n_real) & (cand_g < NG) & (slot < TOPK_C)
    gk = tl.load(
        GK + b * stride_gk_b + h_kv * stride_gk_h + cand_g[:, None] * BLOCK_D + offs_d[None, :],
        mask=valid[:, None], other=0.0,
    )
    route = _dot_bf16(q, gk).to(tl.float32) * SCALE
    route = tl.where(valid[None, :] & q_mask[:, None], route, float("-inf"))
    pooled = tl.max(route, 0)

    ar = tl.arange(0, 16)
    sel = tl.zeros([16], tl.int32)
    has = tl.zeros([16], tl.int32)
    for i in tl.static_range(TOPK_G):
        good = (tl.max(pooled) > -1.0e20).to(tl.int32)
        j = tl.argmax(pooled, 0)
        pooled = tl.where(offs_c == j, float("-inf"), pooled)
        j_loc = j % GPC
        j_slot = j // GPC
        j_ch = tl.where(j_slot == 0, w0, 0)
        j_ch = tl.where(j_slot == 1, w1, j_ch)
        j_ch = tl.where(j_slot == 2, w2, j_ch)
        j_ch = tl.where(j_slot == 3, w3, j_ch)
        sel = tl.where(ar == i, j_ch * GPC + j_loc, sel)
        has = tl.where(ar == i, good, has)
    tl.store(
        Sel + b * stride_sel_b + h * stride_sel_h + ar,
        tl.where(has > 0, sel, -1),
        mask=ar < TOPK_G,
    )

    offs_k = tl.arange(0, 16)
    for i in tl.static_range(TOPK_G):
        gid = tl.sum(tl.where(ar == i, sel, 0))
        ok = (tl.sum(tl.where(ar == i, has, 0)) > 0) & (offs_k < GROUP)
        k_row = gid * GROUP + offs_k
        ok = ok & (k_row < N_PROMPT)
        kk = tl.load(
            K + b * stride_k_b + h_kv * stride_k_h + k_row[:, None] * stride_k_s + offs_d[None, :],
            mask=ok[:, None], other=0.0,
        )
        vv = tl.load(
            V + b * stride_v_b + h_kv * stride_v_h + k_row[:, None] * stride_v_s + offs_d[None, :],
            mask=ok[:, None], other=0.0,
        )
        dest = i * GROUP + offs_k
        tl.store(
            PK + b * stride_pk_b + h * stride_pk_h + dest[:, None] * BLOCK_D + offs_d[None, :],
            kk, mask=ok[:, None],
        )
        tl.store(
            PV + b * stride_pv_b + h * stride_pv_h + dest[:, None] * BLOCK_D + offs_d[None, :],
            vv, mask=ok[:, None],
        )


@triton.jit
def _diff_attend_packed_kernel(
    Q, K, V, PK, PV, Out,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_pk_b, stride_pk_h,
    stride_pv_b, stride_pv_h,
    stride_o_b, stride_o_h, stride_o_s,
    N_PROMPT, Q_LEN, K_LEN, HQ, KVH, SCALE, N_PACK,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PREF: tl.constexpr,
    BLOCK_DIFF: tl.constexpr,
):
    """One CTA = BLOCK_Q queries of one head. Contiguous packed prefix + live diffusion keys."""
    h_tile = tl.program_id(0)
    q_tile = tl.program_id(1)
    b = h_tile // HQ
    h = h_tile % HQ
    h_kv = h // (HQ // KVH)

    offs_q = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < Q_LEN
    q = tl.load(
        Q + b * stride_q_b + h * stride_q_h + offs_q[:, None] * stride_q_s + offs_d[None, :],
        mask=q_mask[:, None], other=0.0,
    )

    offs_p = tl.arange(0, BLOCK_PREF)
    p_ok = offs_p < N_PACK
    pk = tl.load(
        PK + b * stride_pk_b + h * stride_pk_h + offs_p[:, None] * BLOCK_D + offs_d[None, :],
        mask=p_ok[:, None], other=0.0,
    )
    pv = tl.load(
        PV + b * stride_pv_b + h * stride_pv_h + offs_p[:, None] * BLOCK_D + offs_d[None, :],
        mask=p_ok[:, None], other=0.0,
    )
    s = _dot_bf16(q, pk).to(tl.float32) * SCALE
    s = tl.where(p_ok[None, :] & q_mask[:, None], s, float("-inf"))
    m_i = tl.max(s, 1)
    p = tl.math.exp2((s - m_i[:, None]) * LOG2E)
    p = tl.where(p_ok[None, :] & q_mask[:, None], p, 0.0)
    l_i = tl.sum(p, 1)
    acc = tl.dot(p.to(tl.bfloat16), pv.to(tl.bfloat16))

    offs_n = tl.arange(0, BLOCK_DIFF)
    d_row = N_PROMPT + offs_n
    d_ok = (offs_n < Q_LEN) & (d_row < K_LEN)
    kk = tl.load(
        K + b * stride_k_b + h_kv * stride_k_h + d_row[:, None] * stride_k_s + offs_d[None, :],
        mask=d_ok[:, None], other=0.0,
    )
    vv = tl.load(
        V + b * stride_v_b + h_kv * stride_v_h + d_row[:, None] * stride_v_s + offs_d[None, :],
        mask=d_ok[:, None], other=0.0,
    )
    s2 = _dot_bf16(q, kk).to(tl.float32) * SCALE
    s2 = tl.where(d_ok[None, :] & q_mask[:, None], s2, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s2, 1))
    alpha = tl.math.exp2((m_i - m_new) * LOG2E)
    p2 = tl.math.exp2((s2 - m_new[:, None]) * LOG2E)
    p2 = tl.where(d_ok[None, :] & q_mask[:, None], p2, 0.0)
    l_i = l_i * alpha + tl.sum(p2, 1)
    acc = acc * alpha[:, None] + tl.dot(p2.to(tl.bfloat16), vv.to(tl.bfloat16))

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    tl.store(
        Out + b * stride_o_b + h * stride_o_h + offs_q[:, None] * stride_o_s + offs_d[None, :],
        o, mask=q_mask[:, None],
    )


def _k_head_seq_strides(k: torch.Tensor) -> tuple[int, int, int, int, int]:
    """``(B, H, stride_b, stride_h, stride_s)`` for BHSD or BSHD K. No copy."""
    b, d1, d2 = k.shape[0], k.shape[1], k.shape[2]
    if d1 <= 64 and d2 >= d1:
        return b, d1, k.stride(0), k.stride(1), k.stride(2)
    return b, d2, k.stride(0), k.stride(2), k.stride(1)


def fill_hga_means(
    k: torch.Tensor,
    group_k: torch.Tensor,
    chunk_k: torch.Tensor,
    seqlen: int,
) -> None:
    """16-token group means and 128-token chunk means from one K read."""
    b, h, sb, sh, ss = _k_head_seq_strides(k)
    nc = n_chunks(seqlen, CHUNK)
    _fill_means_kernel[(nc, b * h)](
        k, group_k, chunk_k,
        sb, sh, ss,
        group_k.stride(0), group_k.stride(1),
        chunk_k.stride(0), chunk_k.stride(1),
        seqlen, h,
        GROUP=GROUP, GPC=GPC, CHUNK=CHUNK, BLOCK_D=HEAD_DIM,
        num_warps=4, num_stages=2,
    )


def _bhsd_or_bshd_strides(t: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    """``(B, H, S, stride_b, stride_h, stride_s)`` without a copy."""
    b, d1, d2 = t.shape[0], t.shape[1], t.shape[2]
    if d1 <= 64 and d2 >= d1:
        return b, d1, d2, t.stride(0), t.stride(1), t.stride(2)
    return b, d2, d1, t.stride(0), t.stride(2), t.stride(1)


def diff_route_and_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_k: torch.Tensor,
    chunk_k: torch.Tensor,
    sel: torch.Tensor,
    pack_k: torch.Tensor,
    pack_v: torch.Tensor,
    n_prompt: int,
    softmax_scale: float,
) -> None:
    """Write group ids and contiguous packed prefix K/V for every query head."""
    b, hq, q_len, q_b, q_h, q_s = _bhsd_or_bshd_strides(q)
    _, kvh, _, k_b, k_h, k_s = _bhsd_or_bshd_strides(k)
    _, _, _, v_b, v_h, v_s = _bhsd_or_bshd_strides(v)
    nc = n_chunks(n_prompt, CHUNK)
    ng = n_chunks(n_prompt, GROUP)
    _diff_route_pack_kernel[(b * hq,)](
        q, k, v, group_k, chunk_k, sel, pack_k, pack_v,
        q_b, q_h, q_s,
        k_b, k_h, k_s,
        v_b, v_h, v_s,
        group_k.stride(0), group_k.stride(1),
        chunk_k.stride(0), chunk_k.stride(1),
        sel.stride(0), sel.stride(1),
        pack_k.stride(0), pack_k.stride(1),
        pack_v.stride(0), pack_v.stride(1),
        int(n_prompt), ng, nc, q_len, hq, kvh, float(softmax_scale),
        TOPK_G=TOPK_G, TOPK_C=TOPK_C, GPC=GPC, GROUP=GROUP,
        BLOCK_Q=64, BLOCK_D=HEAD_DIM, BLOCK_CAND=32, BLOCK_CK=BLOCK_CK,
        num_warps=4, num_stages=2,
    )


def diff_attend_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pack_k: torch.Tensor,
    pack_v: torch.Tensor,
    out: torch.Tensor,
    n_prompt: int,
    softmax_scale: float,
) -> None:
    """Attend packed 176 prefix tokens + the 64 live diffusion keys."""
    b, hq, q_len, q_b, q_h, q_s = _bhsd_or_bshd_strides(q)
    _, kvh, k_len, k_b, k_h, k_s = _bhsd_or_bshd_strides(k)
    _, _, _, v_b, v_h, v_s = _bhsd_or_bshd_strides(v)
    if tuple(out.shape) == (b, hq, q_len, HEAD_DIM):
        o_b, o_h, o_s = out.stride(0), out.stride(1), out.stride(2)
    else:
        o_b, o_h, o_s = out.stride(0), out.stride(2), out.stride(1)
    n_tiles = (q_len + 15) // 16
    _diff_attend_packed_kernel[(b * hq, n_tiles)](
        q, k, v, pack_k, pack_v, out,
        q_b, q_h, q_s,
        k_b, k_h, k_s,
        v_b, v_h, v_s,
        pack_k.stride(0), pack_k.stride(1),
        pack_v.stride(0), pack_v.stride(1),
        o_b, o_h, o_s,
        int(n_prompt), q_len, k_len, hq, kvh, float(softmax_scale), N_PACK,
        BLOCK_Q=16, BLOCK_D=HEAD_DIM, BLOCK_PREF=N_PACK_PAD, BLOCK_DIFF=64,
        num_warps=4, num_stages=2,
    )


def chunk_route_vlm_fast(
    q: torch.Tensor,
    chunk_k: torch.Tensor,
    out: torch.Tensor,
    seqlen: int,
    q_chunk: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton mean-Q vs chunk-K top-4. ``q_chunk`` is ignored (API compat)."""
    b, d1, d2 = q.shape[0], q.shape[1], q.shape[2]
    if d1 <= 64 and d2 >= d1:
        hq, sb, sh, ss = d1, q.stride(0), q.stride(1), q.stride(2)
    else:
        hq, sb, sh, ss = d2, q.stride(0), q.stride(2), q.stride(1)
    hkv = chunk_k.shape[1]
    nc = n_chunks(seqlen, CHUNK)
    scale = HEAD_DIM ** -0.5
    _chunk_route_vlm_kernel[(nc, b * hq)](
        q, chunk_k, out,
        sb, sh, ss,
        chunk_k.stride(0), chunk_k.stride(1),
        out.stride(0), out.stride(1),
        seqlen, nc, hq, hkv, float(scale),
        CHUNK=CHUNK, TOPK_C=TOPK_C, BLOCK_D=HEAD_DIM, BLOCK_CK=BLOCK_CK,
        num_warps=4, num_stages=2,
    )
    return out


def chunk_route_diff_fast(
    q: torch.Tensor,
    chunk_k: torch.Tensor,
    n_chunks_p: int,
    out: torch.Tensor,
) -> torch.Tensor:
    b, hq, q_len, _ = q.shape
    hkv = chunk_k.shape[1]
    scale = HEAD_DIM ** -0.5
    _chunk_route_diff_kernel[(b * hq,)](
        q, chunk_k, out,
        q.stride(0), q.stride(1), q.stride(2),
        chunk_k.stride(0), chunk_k.stride(1),
        out.stride(0), out.stride(1),
        int(n_chunks_p), q_len, hq, hkv, float(scale),
        TOPK_C=TOPK_C, BLOCK_Q=64, BLOCK_D=HEAD_DIM, BLOCK_CK=BLOCK_CK,
        num_warps=4, num_stages=2,
    )
    return out


def fill_chunk_keys(
    k: torch.Tensor,
    chunk_k: torch.Tensor,
    seqlen: int,
    chunk: int = CHUNK,
) -> torch.Tensor:
    """Write per-block mean keys into ``chunk_k`` ``[B, H, n_slots, D]``."""
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    b, h, _, d = k.shape
    n = n_chunks(seqlen, chunk)
    if chunk_k.shape[2] < n:
        raise ValueError(f"chunk_k has {chunk_k.shape[2]} slots, need {n}")
    n_full = seqlen // chunk
    if n_full > 0:
        chunk_k[:, :, :n_full].copy_(
            k[:, :, : n_full * chunk, :].reshape(b, h, n_full, chunk, d).mean(dim=3)
        )
    rem = seqlen - n_full * chunk
    if rem > 0:
        chunk_k[:, :, n_full].copy_(k[:, :, n_full * chunk : seqlen, :].sum(dim=2) / float(rem))
    if n < chunk_k.shape[2]:
        chunk_k[:, :, n:].zero_()
    return chunk_k


def chunk_route_vlm(q: torch.Tensor, k: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Exact top-``TOPK_C`` previous 128-chunks per query chunk. ``[B, Hq, Nc, 4]``."""
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    nc = n_chunks(s, CHUNK)
    use_s = nc * CHUNK
    if use_s != s:
        q = F.pad(q, (0, 0, 0, use_s - s))
        k = F.pad(k, (0, 0, 0, use_s - s))
    qc = q[:, :, :use_s].reshape(b, hq, nc, CHUNK, d).mean(dim=3)
    kc = k[:, :, :use_s].reshape(b, hkv, nc, CHUNK, d).mean(dim=3)
    gqa = hq // hkv
    scores = torch.matmul(
        qc.reshape(b, hkv, gqa, nc, d),
        kc.unsqueeze(2).transpose(-1, -2),
    ).reshape(b, hq, nc, nc) * (d ** -0.5)
    idx = torch.arange(nc, device=q.device)
    scores = scores.masked_fill(idx[None, None, None, :] >= idx[None, None, :, None], -1.0e9)
    k_eff = min(TOPK_C, max(nc - 1, 1))
    ids = scores.topk(k_eff, dim=-1).indices.to(torch.int32)
    if k_eff < TOPK_C:
        ids = F.pad(ids, (0, TOPK_C - k_eff), value=0)
    if out is None or tuple(out.shape) != tuple(ids.shape):
        return ids.contiguous()
    out.copy_(ids)
    return out


def chunk_route_diff(q: torch.Tensor, chunk_k: torch.Tensor, n_chunks_p: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Shared top-``TOPK_C`` prefix chunks for all diffusion queries. ``[B, Hq, 4]``."""
    b, hq, _q_len, d = q.shape
    hkv = chunk_k.shape[1]
    nc = int(n_chunks_p)
    gqa = hq // hkv
    kc = chunk_k[:, :, :nc]
    qh = q.reshape(b, hkv, gqa, -1, d)
    scores = torch.matmul(qh, kc.unsqueeze(2).transpose(-1, -2)) * (d ** -0.5)
    pooled = scores.amax(dim=3).reshape(b, hq, nc)
    k_eff = min(TOPK_C, max(nc, 1))
    ids = pooled.topk(k_eff, dim=-1).indices.to(torch.int32)
    if k_eff < TOPK_C:
        ids = F.pad(ids, (0, TOPK_C - k_eff), value=0)
    if out is None or tuple(out.shape) != tuple(ids.shape):
        return ids.contiguous()
    out.copy_(ids)
    return out
