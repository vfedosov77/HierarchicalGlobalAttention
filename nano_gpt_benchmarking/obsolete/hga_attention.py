# Article: https://arxiv.org/pdf/2606.30709
# WARNING! Only during the training can be used! For validation must be used dense attention because HGA can have small causality leaks!

from __future__ import annotations

from kernels import get_kernel

flash_attn_interface = get_kernel('kernels-community/flash-attn3', version=1).flash_attn_interface

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Small Python helpers


def _qk_norm(x: Tensor) -> Tensor:
    # Match modded-nanogpt's norm(x): F.rms_norm(x, (x.size(-1),)).
    return F.rms_norm(x, (x.size(-1),))


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return triton.cdiv(int(a), int(b))


def _sum_min_prefix(n: int, m: int) -> int:
    # sum_{i=0}^{n-1} min(i, m)
    n = int(n)
    m = int(m)
    if n <= m:
        return n * (n - 1) // 2
    return m * (m - 1) // 2 + (n - m) * m


def _flash_out(x):
    # Some flash_attn_interface builds return out, others return (out, lse).
    return x[0] if isinstance(x, tuple) else x


# -----------------------------------------------------------------------------
# Triton kernels


@triton.jit
def _chunk_sum_qk_kernel(
    q_ptr, k_ptr, q_sum_ptr, k_sum_ptr,
    T, E: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    # q/k are contiguous flattened [T, E]. q_sum/k_sum are [C, E] fp32.
    c = tl.program_id(0)
    eb = tl.program_id(1)

    ts = tl.arange(0, BLOCK_T)
    es = eb * BLOCK_E + tl.arange(0, BLOCK_E)
    toks = c * CHUNK + ts

    mask = (toks[:, None] < T) & (es[None, :] < E)
    offs = toks[:, None] * E + es[None, :]

    qv = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    kv = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    qs = tl.sum(qv, axis=0)
    ks = tl.sum(kv, axis=0)

    c_offs = c * E + es
    tl.store(q_sum_ptr + c_offs, qs, mask=es < E)
    tl.store(k_sum_ptr + c_offs, ks, mask=es < E)


@triton.jit
def _score_matmul_kernel(
    q_sum_ptr, k_sum_ptr, scores_ptr,
    C, CMAX: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr,
):
    # Tiled matmul: scores = q_sum @ k_sum.T.
    # q_sum/k_sum are fixed-capacity [CMAX,E]; scores is [CMAX,CMAX].
    # C is runtime actual chunk count, so changing sequence length does not
    # change the compiled route matmul variant.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    ms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    es = tl.arange(0, BLOCK_E)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for e0 in tl.static_range(0, E, BLOCK_E):
        e = e0 + es
        qv = tl.load(q_sum_ptr + ms[:, None] * E + e[None, :],
                     mask=(ms[:, None] < C) & (e[None, :] < E), other=0.0)
        kv = tl.load(k_sum_ptr + ns[None, :] * E + e[:, None],
                     mask=(ns[None, :] < C) & (e[:, None] < E), other=0.0)
        acc += tl.dot(qv, kv, input_precision="tf32")

    tl.store(scores_ptr + ms[:, None] * CMAX + ns[None, :], acc,
             mask=(ms[:, None] < C) & (ns[None, :] < C))


@triton.jit
def _route_select_from_scores_kernel(
    scores_ptr, selected_ptr,
    C, CMAX: tl.constexpr, ROUTE_TOPK: tl.constexpr, LOCAL_PREV: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_SLOTS: tl.constexpr,
):
    # One program per q chunk. BLOCK_C is fixed route_max_chunks, not next_power_of_2(C),
    # so selection cost/compiled variant does not jump at C=33/65.
    q = tl.program_id(0)
    kc = tl.arange(0, BLOCK_C)
    valid_k = kc < q
    scores = tl.load(scores_ptr + q * CMAX + kc, mask=kc < C, other=-3.402823e38).to(tl.float32)
    scores = tl.where(valid_k, scores, -3.402823e38)

    slots = tl.arange(0, BLOCK_SLOTS)
    sel = tl.full((BLOCK_SLOTS,), -1, tl.int32)
    count = tl.full((), 0, tl.int32)
    target = tl.minimum(q, MAX_PAST).to(tl.int32)

    for _ in tl.static_range(0, ROUTE_TOPK):
        mx = tl.max(scores, axis=0)
        # deterministic tie-break: highest index among tied scores
        idx = tl.max(tl.where(scores == mx, kc, 0), axis=0).to(tl.int32)
        add = (count < target) & (mx > -3.0e38)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    for off in tl.static_range(1, LOCAL_PREV + 1):
        idx = (q - off).to(tl.int32)
        exists = tl.sum(tl.where((slots < MAX_PAST) & (sel == idx), 1, 0), axis=0) > 0
        add = (idx >= 0) & (count < target) & (~exists)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    for _ in tl.static_range(0, MAX_PAST):
        mx = tl.max(scores, axis=0)
        idx = tl.max(tl.where(scores == mx, kc, 0), axis=0).to(tl.int32)
        exists = tl.sum(tl.where((slots < MAX_PAST) & (sel == idx), 1, 0), axis=0) > 0
        add = (count < target) & (~exists) & (mx > -3.0e38)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    tl.store(selected_ptr + q * MAX_PAST + slots, sel, mask=slots < MAX_PAST)


@triton.jit
def _route_select_kernel(
    q_sum_ptr, k_sum_ptr, selected_ptr,
    C: tl.constexpr, E: tl.constexpr,
    ROUTE_TOPK: tl.constexpr, LOCAL_PREV: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_SLOTS: tl.constexpr, E_PAD: tl.constexpr, BLOCK_E_STEP: tl.constexpr,
):
    # One program per q chunk. Produces selected[q, 0:target_count].
    # target_count is exactly min(q_chunk, MAX_PAST), so KV packing lengths are static.
    q = tl.program_id(0)

    kc = tl.arange(0, BLOCK_C)
    valid_k = kc < q  # strictly past chunks only; current chunk appended separately
    scores = tl.zeros((BLOCK_C,), tl.float32)

    for e0 in tl.static_range(0, E_PAD, BLOCK_E_STEP):
        es = e0 + tl.arange(0, BLOCK_E_STEP)
        qv = tl.load(q_sum_ptr + q * E + es, mask=es < E, other=0.0).to(tl.float32)
        kvals = tl.load(
            k_sum_ptr + kc[:, None] * E + es[None, :],
            mask=(kc[:, None] < C) & (es[None, :] < E),
            other=0.0,
        ).to(tl.float32)
        part = tl.sum(kvals * qv[None, :], axis=1)
        scores += tl.where(valid_k, part, 0.0)

    scores = tl.where(valid_k, scores, -3.402823e38)

    slots = tl.arange(0, BLOCK_SLOTS)
    sel = tl.full((BLOCK_SLOTS,), -1, tl.int32)
    count = tl.full((), 0, tl.int32)
    target = tl.minimum(q, MAX_PAST).to(tl.int32)

    # 1) Highest-scoring routed chunks.
    for _ in tl.static_range(0, ROUTE_TOPK):
        mx = tl.max(scores, axis=0)
        idx = tl.max(tl.where(scores == mx, kc, 0), axis=0).to(tl.int32)
        add = (count < target) & (mx > -3.0e38)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    # 2) Mandatory local previous chunks.
    for off in tl.static_range(1, LOCAL_PREV + 1):
        idx = (q - off).to(tl.int32)
        exists = tl.sum(tl.where((slots < MAX_PAST) & (sel == idx), 1, 0), axis=0) > 0
        add = (idx >= 0) & (count < target) & (~exists)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    # 3) Fill the remaining fixed budget with next-best past chunks. This avoids
    # variable allocation / CPU sync and usually improves FA arithmetic intensity.
    for _ in tl.static_range(0, MAX_PAST):
        mx = tl.max(scores, axis=0)
        idx = tl.max(tl.where(scores == mx, kc, 0), axis=0).to(tl.int32)
        exists = tl.sum(tl.where((slots < MAX_PAST) & (sel == idx), 1, 0), axis=0) > 0
        add = (count < target) & (~exists) & (mx > -3.0e38)
        sel = tl.where((slots == count) & add, idx, sel)
        count += tl.where(add, 1, 0)
        scores = tl.where(kc == idx, -3.402823e38, scores)

    tl.store(selected_ptr + q * MAX_PAST + slots, sel, mask=slots < MAX_PAST)


@triton.jit
def _fixed_cu_seqlens_kernel(
    cu_q_ptr, cu_k_ptr,
    T, C, CHUNK: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_CP1: tl.constexpr,
):
    # Computes cu_q[i] = min(i*CHUNK, T)
    # and cu_k[i] = sum_{j<i} (min(j,MAX_PAST)*CHUNK + q_len[j]).
    i = tl.arange(0, BLOCK_CP1)
    mask = i <= C

    q_prefix = tl.minimum(i * CHUNK, T)
    tl.store(cu_q_ptr + i, q_prefix.to(tl.int32), mask=mask)

    p = tl.minimum(i, MAX_PAST)
    # sum_{j=0}^{i-1} min(j, MAX_PAST)
    past_blocks = (p * (p - 1)) // 2 + (i - p) * MAX_PAST
    k_prefix = past_blocks * CHUNK + q_prefix
    tl.store(cu_k_ptr + i, k_prefix.to(tl.int32), mask=mask)


@triton.jit
def _pack_kv_forward_kernel(
    k_ptr, v_ptr, kp_ptr, vp_ptr, selected_ptr, cu_k_ptr,
    T: tl.constexpr, E: tl.constexpr, C: tl.constexpr,
    CHUNK: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    # k/v are contiguous flattened [T, E]. kp/vp are flattened [total_k, E].
    pid_cs = tl.program_id(0)
    tb = tl.program_id(1)
    eb = tl.program_id(2)

    slot = pid_cs % (MAX_PAST + 1)
    c = pid_cs // (MAX_PAST + 1)
    count = tl.minimum(c, MAX_PAST)

    is_past = slot < count
    is_cur = slot == count
    valid_slot = is_past | is_cur

    src_chunk = tl.load(selected_ptr + c * MAX_PAST + slot, mask=is_past, other=c).to(tl.int32)
    src_chunk = tl.where(is_cur, c, src_chunk)

    toks = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    es = eb * BLOCK_E + tl.arange(0, BLOCK_E)

    src_t = src_chunk * CHUNK + toks
    dst_base = tl.load(cu_k_ptr + c) + slot * CHUNK
    dst_t = dst_base + toks

    mask = valid_slot & (src_t[:, None] < T) & (toks[:, None] < CHUNK) & (es[None, :] < E)
    src_offs = src_t[:, None] * E + es[None, :]
    dst_offs = dst_t[:, None] * E + es[None, :]

    kval = tl.load(k_ptr + src_offs, mask=mask, other=0.0)
    vval = tl.load(v_ptr + src_offs, mask=mask, other=0.0)
    tl.store(kp_ptr + dst_offs, kval, mask=mask)
    tl.store(vp_ptr + dst_offs, vval, mask=mask)


@triton.jit
def _pack_kv_backward_kernel(
    gkp_ptr, gvp_ptr, gk_ptr, gv_ptr, selected_ptr, cu_k_ptr,
    T: tl.constexpr, E: tl.constexpr, C: tl.constexpr,
    CHUNK: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    pid_cs = tl.program_id(0)
    tb = tl.program_id(1)
    eb = tl.program_id(2)

    slot = pid_cs % (MAX_PAST + 1)
    c = pid_cs // (MAX_PAST + 1)
    count = tl.minimum(c, MAX_PAST)

    is_past = slot < count
    is_cur = slot == count
    valid_slot = is_past | is_cur

    src_chunk = tl.load(selected_ptr + c * MAX_PAST + slot, mask=is_past, other=c).to(tl.int32)
    src_chunk = tl.where(is_cur, c, src_chunk)

    toks = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    es = eb * BLOCK_E + tl.arange(0, BLOCK_E)

    orig_t = src_chunk * CHUNK + toks
    pack_base = tl.load(cu_k_ptr + c) + slot * CHUNK
    pack_t = pack_base + toks

    mask = valid_slot & (orig_t[:, None] < T) & (toks[:, None] < CHUNK) & (es[None, :] < E)
    orig_offs = orig_t[:, None] * E + es[None, :]
    pack_offs = pack_t[:, None] * E + es[None, :]

    gk = tl.load(gkp_ptr + pack_offs, mask=mask, other=0.0)
    gv = tl.load(gvp_ptr + pack_offs, mask=mask, other=0.0)
    tl.atomic_add(gk_ptr + orig_offs, gk, sem="relaxed", mask=mask)
    tl.atomic_add(gv_ptr + orig_offs, gv, sem="relaxed", mask=mask)


@triton.jit
def _pack_kv_fixed_forward_kernel(
    k_ptr, v_ptr, kp_ptr, vp_ptr, selected_ptr,
    T, N_MAIN, E: tl.constexpr, START_C: tl.constexpr,
    CHUNK: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    # Pack steady-state chunks into [N_MAIN, (MAX_PAST+1)*CHUNK, E].
    # For each q chunk c=START_C+n, slots 0..MAX_PAST-1 are selected past chunks,
    # slot MAX_PAST is the current chunk.
    pid_ns = tl.program_id(0)
    tb = tl.program_id(1)
    eb = tl.program_id(2)

    slot = pid_ns % (MAX_PAST + 1)
    n = pid_ns // (MAX_PAST + 1)
    c = START_C + n

    is_cur = slot == MAX_PAST
    src_chunk = tl.load(selected_ptr + c * MAX_PAST + slot, mask=slot < MAX_PAST, other=c).to(tl.int32)
    src_chunk = tl.where(is_cur, c, src_chunk)

    toks = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    es = eb * BLOCK_E + tl.arange(0, BLOCK_E)

    src_t = src_chunk * CHUNK + toks
    dst_t = n * ((MAX_PAST + 1) * CHUNK) + slot * CHUNK + toks

    dst_mask = (n < N_MAIN) & (toks[:, None] < CHUNK) & (es[None, :] < E)
    src_mask = dst_mask & (src_t[:, None] < T)
    src_offs = src_t[:, None] * E + es[None, :]
    dst_offs = dst_t[:, None] * E + es[None, :]

    kval = tl.load(k_ptr + src_offs, mask=src_mask, other=0.0)
    vval = tl.load(v_ptr + src_offs, mask=src_mask, other=0.0)
    # Store zeros for padded tail tokens in the final chunk. This lets the
    # fast fixed-shape batched FlashAttention path handle T % CHUNK != 0.
    tl.store(kp_ptr + dst_offs, kval, mask=dst_mask)
    tl.store(vp_ptr + dst_offs, vval, mask=dst_mask)


@triton.jit
def _pack_kv_fixed_backward_kernel(
    gkp_ptr, gvp_ptr, gk_ptr, gv_ptr, selected_ptr,
    T, N_MAIN, E: tl.constexpr, START_C: tl.constexpr,
    CHUNK: tl.constexpr, MAX_PAST: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    pid_ns = tl.program_id(0)
    tb = tl.program_id(1)
    eb = tl.program_id(2)

    slot = pid_ns % (MAX_PAST + 1)
    n = pid_ns // (MAX_PAST + 1)
    c = START_C + n

    is_cur = slot == MAX_PAST
    src_chunk = tl.load(selected_ptr + c * MAX_PAST + slot, mask=slot < MAX_PAST, other=c).to(tl.int32)
    src_chunk = tl.where(is_cur, c, src_chunk)

    toks = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    es = eb * BLOCK_E + tl.arange(0, BLOCK_E)

    orig_t = src_chunk * CHUNK + toks
    pack_t = n * ((MAX_PAST + 1) * CHUNK) + slot * CHUNK + toks

    mask = (n < N_MAIN) & (orig_t[:, None] < T) & (toks[:, None] < CHUNK) & (es[None, :] < E)
    orig_offs = orig_t[:, None] * E + es[None, :]
    pack_offs = pack_t[:, None] * E + es[None, :]

    gk = tl.load(gkp_ptr + pack_offs, mask=mask, other=0.0)
    gv = tl.load(gvp_ptr + pack_offs, mask=mask, other=0.0)
    tl.atomic_add(gk_ptr + orig_offs, gk, sem="relaxed", mask=mask)
    tl.atomic_add(gv_ptr + orig_offs, gv, sem="relaxed", mask=mask)


# -----------------------------------------------------------------------------
# Autograd wrapper for Triton K/V repacking


class _PackKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k: Tensor,
        v: Tensor,
        selected: Tensor,
        cu_k: Tensor,
        T: int,
        H: int,
        D: int,
        C: int,
        CHUNK: int,
        MAX_PAST: int,
        PACK_BLOCK_T: int,
        PACK_BLOCK_E: int,
    ):
        E = H * D
        total_k = _sum_min_prefix(C, MAX_PAST) * CHUNK + T
        kp = torch.empty((total_k, H, D), device=k.device, dtype=k.dtype)
        vp = torch.empty((total_k, H, D), device=v.device, dtype=v.dtype)

        grid = (C * (MAX_PAST + 1), _cdiv(CHUNK, PACK_BLOCK_T), _cdiv(E, PACK_BLOCK_E))
        _pack_kv_forward_kernel[grid](
            k, v, kp, vp, selected, cu_k,
            T, E, C, CHUNK, MAX_PAST,
            BLOCK_T=PACK_BLOCK_T, BLOCK_E=PACK_BLOCK_E,
        )

        ctx.save_for_backward(selected, cu_k)
        ctx.meta = (T, H, D, C, CHUNK, MAX_PAST, PACK_BLOCK_T, PACK_BLOCK_E)
        return kp, vp

    @staticmethod
    def backward(ctx, grad_kp: Tensor, grad_vp: Tensor):
        selected, cu_k = ctx.saved_tensors
        T, H, D, C, CHUNK, MAX_PAST, PACK_BLOCK_T, PACK_BLOCK_E = ctx.meta
        E = H * D

        grad_k = torch.zeros((T, H, D), device=grad_kp.device, dtype=grad_kp.dtype)
        grad_v = torch.zeros((T, H, D), device=grad_vp.device, dtype=grad_vp.dtype)

        grid = (C * (MAX_PAST + 1), _cdiv(CHUNK, PACK_BLOCK_T), _cdiv(E, PACK_BLOCK_E))
        _pack_kv_backward_kernel[grid](
            grad_kp.contiguous(), grad_vp.contiguous(), grad_k, grad_v, selected, cu_k,
            T, E, C, CHUNK, MAX_PAST,
            BLOCK_T=PACK_BLOCK_T, BLOCK_E=PACK_BLOCK_E,
        )

        return grad_k, grad_v, None, None, None, None, None, None, None, None, None, None


class _PackKVFixed(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k: Tensor,
        v: Tensor,
        selected: Tensor,
        T: int,
        H: int,
        D: int,
        START_C: int,
        N_MAIN: int,
        CHUNK: int,
        MAX_PAST: int,
        PACK_BLOCK_T: int,
        PACK_BLOCK_E: int,
    ):
        E = H * D
        KLEN = (MAX_PAST + 1) * CHUNK
        kp = torch.empty((N_MAIN, KLEN, H, D), device=k.device, dtype=k.dtype)
        vp = torch.empty((N_MAIN, KLEN, H, D), device=v.device, dtype=v.dtype)

        grid = (N_MAIN * (MAX_PAST + 1), _cdiv(CHUNK, PACK_BLOCK_T), _cdiv(E, PACK_BLOCK_E))
        _pack_kv_fixed_forward_kernel[grid](
            k, v, kp, vp, selected,
            T, N_MAIN, E, START_C, CHUNK, MAX_PAST,
            BLOCK_T=PACK_BLOCK_T, BLOCK_E=PACK_BLOCK_E,
        )

        ctx.save_for_backward(selected)
        ctx.meta = (T, H, D, START_C, N_MAIN, CHUNK, MAX_PAST, PACK_BLOCK_T, PACK_BLOCK_E)
        return kp, vp

    @staticmethod
    def backward(ctx, grad_kp: Tensor, grad_vp: Tensor):
        (selected,) = ctx.saved_tensors
        T, H, D, START_C, N_MAIN, CHUNK, MAX_PAST, PACK_BLOCK_T, PACK_BLOCK_E = ctx.meta
        E = H * D

        grad_k = torch.zeros((T, H, D), device=grad_kp.device, dtype=grad_kp.dtype)
        grad_v = torch.zeros((T, H, D), device=grad_vp.device, dtype=grad_vp.dtype)

        grid = (N_MAIN * (MAX_PAST + 1), _cdiv(CHUNK, PACK_BLOCK_T), _cdiv(E, PACK_BLOCK_E))
        _pack_kv_fixed_backward_kernel[grid](
            grad_kp.contiguous(), grad_vp.contiguous(), grad_k, grad_v, selected,
            T, N_MAIN, E, START_C, CHUNK, MAX_PAST,
            BLOCK_T=PACK_BLOCK_T, BLOCK_E=PACK_BLOCK_E,
        )

        return grad_k, grad_v, None, None, None, None, None, None, None, None, None, None


# -----------------------------------------------------------------------------
# Drop-in attention module


class CausalSelfAttentionHGA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        paired: bool = False,
        *,
        route_chunk_size: int = 64,
        route_topk: int = 4,
        route_local_prev: int = 1,
        route_max_chunks: int = 64,
        route_sum_block_e: int = 32,
        route_score_block_e: int = 64,
        pack_block_t: int = 16,
        pack_block_e: int = 64,
        route_matmul_block_m: int = 16,
        route_matmul_block_n: int = 16,
        route_matmul_block_e: int = 64,
        use_batched_flash: bool = True,
        route_respect_doc_boundaries: bool = False,
    ):
        super().__init__()
        assert not paired, "This routed FlashAttention replacement is only for the non-paired final layer."
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = False
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"

        assert route_chunk_size > 0 and (route_chunk_size & (route_chunk_size - 1)) == 0
        assert route_topk >= 0
        assert route_local_prev >= 0
        assert route_topk + route_local_prev > 0, "need at least one routed/local past block for this optimized path"
        assert route_max_chunks > 0 and (route_max_chunks & (route_max_chunks - 1)) == 0
        assert route_sum_block_e > 0 and (route_sum_block_e & (route_sum_block_e - 1)) == 0
        assert route_score_block_e > 0 and (route_score_block_e & (route_score_block_e - 1)) == 0
        assert pack_block_t > 0 and (pack_block_t & (pack_block_t - 1)) == 0
        assert pack_block_e > 0 and (pack_block_e & (pack_block_e - 1)) == 0
        assert route_matmul_block_m > 0 and (route_matmul_block_m & (route_matmul_block_m - 1)) == 0
        assert route_matmul_block_n > 0 and (route_matmul_block_n & (route_matmul_block_n - 1)) == 0
        assert route_matmul_block_e > 0 and (route_matmul_block_e & (route_matmul_block_e - 1)) == 0
        if route_respect_doc_boundaries:
            raise NotImplementedError(
                "This fastest Triton/Flash path intentionally omits doc-boundary routing. "
                "Use route_respect_doc_boundaries=False, align packed docs to chunk_size, "
                "or use the older doc-aware PyTorch routing version."
            )

        self.route_chunk_size = int(route_chunk_size)
        self.route_topk = int(route_topk)
        self.route_local_prev = int(route_local_prev)
        self.max_past = int(route_topk + route_local_prev)
        self.route_max_chunks = int(route_max_chunks)
        self.route_sum_block_e = int(route_sum_block_e)
        self.route_score_block_e = int(route_score_block_e)
        self.pack_block_t = int(pack_block_t)
        self.pack_block_e = int(pack_block_e)
        self.route_matmul_block_m = int(route_matmul_block_m)
        self.route_matmul_block_n = int(route_matmul_block_n)
        self.route_matmul_block_e = int(route_matmul_block_e)
        if not use_batched_flash:
            raise NotImplementedError("v5 has no slow T%CHUNK fallback for main chunks; keep use_batched_flash=True")
        self.use_batched_flash = True
        self.route_respect_doc_boundaries = False

    @torch.no_grad()
    def _build_routing_metadata(self, q0: Tensor, k0: Tensor, T: int) -> tuple[Tensor, Tensor, Tensor, int, int]:
        # q0/k0 are contiguous [T,H,D]. Routing is detached/non-differentiable.
        H, D = self.num_heads, self.head_dim
        E = H * D
        CHUNK = self.route_chunk_size
        C = _cdiv(T, CHUNK)
        CMAX = self.route_max_chunks
        MAX_PAST = self.max_past
        assert C <= CMAX, f"sequence has {C} chunks but route_max_chunks={CMAX}; increase route_max_chunks"

        # Fixed-capacity scratch avoids power-of-two jumps and repeated Triton specialization.
        q_sum = torch.empty((CMAX, E), device=q0.device, dtype=torch.float32)
        k_sum = torch.empty((CMAX, E), device=k0.device, dtype=torch.float32)

        grid_sum = (C, _cdiv(E, self.route_sum_block_e))
        _chunk_sum_qk_kernel[grid_sum](
            q0, k0, q_sum, k_sum,
            T, E, CHUNK,
            BLOCK_T=CHUNK, BLOCK_E=self.route_sum_block_e,
        )

        scores = torch.empty((CMAX, CMAX), device=q0.device, dtype=torch.float32)
        grid_score = (_cdiv(C, self.route_matmul_block_m), _cdiv(C, self.route_matmul_block_n))
        _score_matmul_kernel[grid_score](
            q_sum, k_sum, scores,
            C, CMAX, E,
            BLOCK_M=self.route_matmul_block_m,
            BLOCK_N=self.route_matmul_block_n,
            BLOCK_E=self.route_matmul_block_e,
        )

        selected = torch.empty((CMAX, MAX_PAST), device=q0.device, dtype=torch.int32)
        _route_select_from_scores_kernel[(C,)](
            scores, selected,
            C, CMAX,
            ROUTE_TOPK=self.route_topk,
            LOCAL_PREV=self.route_local_prev,
            MAX_PAST=MAX_PAST,
            BLOCK_C=CMAX,
            BLOCK_SLOTS=_next_power_of_2(MAX_PAST),
        )

        cu_q = torch.empty((C + 1,), device=q0.device, dtype=torch.int32)
        cu_k = torch.empty((C + 1,), device=q0.device, dtype=torch.int32)
        _fixed_cu_seqlens_kernel[(1,)](
            cu_q, cu_k,
            T, C, CHUNK, MAX_PAST,
            BLOCK_CP1=_next_power_of_2(C + 1),
        )

        max_q = CHUNK
        max_k = (MAX_PAST + 1) * CHUNK
        return selected, cu_q, cu_k, max_q, max_k

    def _routed_flash_attn(self, q: Tensor, k: Tensor, v: Tensor, yarn) -> Tensor:
        B, T, H, D = q.shape
        assert B == 1
        C = _cdiv(T, self.route_chunk_size)
        CHUNK = self.route_chunk_size
        MAX_PAST = self.max_past
        TP = C * CHUNK
        pad = TP - T

        # Make contiguous once. q0 participates in autograd directly. k0/v0 get
        # gradients through custom Triton pack/unpack autograd functions.
        q0_true = q[0].contiguous()
        k0 = k[0].contiguous()
        v0 = v[0].contiguous()

        # Pad Q to full chunks so regular batched flash_attn_func handles the final
        # partial chunk. The output is sliced back to true T below. K/V padded tail is
        # zero-filled inside the pack kernel, so no uninitialized final-chunk reads.
        if pad:
            q0 = F.pad(q0_true, (0, 0, 0, 0, 0, pad))
        else:
            q0 = q0_true

        selected, cu_q, cu_k, max_q, max_k = self._build_routing_metadata(q0_true.detach(), k0.detach(), T)

        # Prefix: the first MAX_PAST chunks have shorter K lengths. Do not use
        # flash_attn_varlen_func here; use a few tiny regular flash_attn_func calls.
        # This keeps the whole implementation on the fixed/batched FlashAttention API.
        y_parts = []
        start_c = min(MAX_PAST, C)
        if start_c > 0:
            T_early = min(T, start_c * CHUNK)
            C_early = start_c
            cu_q_e = torch.empty((C_early + 1,), device=q0.device, dtype=torch.int32)
            cu_k_e = torch.empty((C_early + 1,), device=q0.device, dtype=torch.int32)
            _fixed_cu_seqlens_kernel[(1,)](
                cu_q_e, cu_k_e,
                T_early, C_early, CHUNK, MAX_PAST,
                BLOCK_CP1=_next_power_of_2(C_early + 1),
            )
            k_pre, v_pre = _PackKV.apply(
                k0[:T_early], v0[:T_early], selected[:C_early], cu_k_e,
                T_early, H, D, C_early, CHUNK, MAX_PAST,
                self.pack_block_t, self.pack_block_e,
            )
            early_out = []
            for c in range(C_early):
                qs = c * CHUNK
                qe = min((c + 1) * CHUNK, T_early)
                q_len = qe - qs
                ks = _sum_min_prefix(c, MAX_PAST) * CHUNK + min(c * CHUNK, T_early)
                ke = _sum_min_prefix(c + 1, MAX_PAST) * CHUNK + min((c + 1) * CHUNK, T_early)
                k_len = ke - ks
                y_c = _flash_out(flash_attn_interface.flash_attn_func(
                    q0_true[qs:qe].view(1, q_len, H, D).contiguous(),
                    k_pre[ks:ke].view(1, k_len, H, D).contiguous(),
                    v_pre[ks:ke].view(1, k_len, H, D).contiguous(),
                    causal=True,
                    softmax_scale=yarn.attn_scale,
                    window_size=(-1, -1),
                ))
                early_out.append(y_c.reshape(q_len, H, D))
            y_parts.append(torch.cat(early_out, dim=0))

        # Main path: always padded fixed-shape batched FlashAttention, even when
        # original T is not CHUNK-aligned. No T%CHUNK varlen branch.
        n_main = C - start_c
        if n_main > 0:
            k_fix, v_fix = _PackKVFixed.apply(
                k0, v0, selected,
                T, H, D, start_c, n_main, CHUNK, MAX_PAST,
                self.pack_block_t, self.pack_block_e,
            )
            q_main = q0[start_c * CHUNK : TP].view(n_main, CHUNK, H, D).contiguous()
            y_main = _flash_out(flash_attn_interface.flash_attn_func(
                q_main,
                k_fix,
                v_fix,
                causal=True,
                softmax_scale=yarn.attn_scale,
                window_size=(-1, -1),
            ))
            y_parts.append(y_main.reshape(n_main * CHUNK, H, D))

        y0 = torch.cat(y_parts, dim=0)[:T]
        return y0.view(B, T, H, D)

    def forward(self, x: Tensor, attn_args, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0

        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        yarn = attn_args.yarn

        q, k, v = F.linear(
            x,
            sa_lambdas[0] * qkvo_w[: self.dim * 3].type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        q, k = _qk_norm(q), _qk_norm(k)
        q, k = yarn.rotary(q), yarn.rotary(k)

        if key_offset:
            k[:, 1:, :, self.head_dim // 2 :] = k[:, :-1, :, self.head_dim // 2 :]

        if aux_v is not None:
            v = v + aux_v.view_as(v)

        y = self._routed_flash_attn(q, k, v, yarn)

        # Gated XSA, same as original. Non-paired only.
        if attn_args.xsa_alpha is not None:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3 :].type_as(y))
        return y