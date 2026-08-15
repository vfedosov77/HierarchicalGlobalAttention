"""True two-level HGA: 128-token chunks, then 16-token groups.

Selected budget: each query chunk opens 4 previous chunks; each query
group opens 11 groups inside those chunks plus the current group
(192 tokens).
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
