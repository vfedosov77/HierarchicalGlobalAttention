"""Fused HGA decode attention.

The reference path (``RoutedKV.attend``) materialises ``[B,H,L,R]`` scores, fills
a boolean mask, runs ``softmax`` and a second einsum.  For decode (small ``L``,
moderate ``R``) that is launch-bound and round-trips the scores through HBM
several times.

:func:`fused_decode_attention` does it in **one** flash-style pass: each program
owns one ``(b, h, l)`` query, streams the key/value axis in blocks, keeps a
running max / denominator / accumulator (online softmax), and never materialises
the full score row.  A boolean visibility mask ``[B,H,L,R]`` (HGA's routed
visibility) is applied per key.  Math is done in fp32 to track an SDPA baseline
across many layers, exactly like the reference.

A pure-PyTorch fallback (:func:`_torch_decode_attention`) is used when Triton is
unavailable or for tiny shapes; it is numerically identical to the reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

_NEG = -1.0e4  # finite mask fill (matches the reference for fp16/bf16 safety)


def _torch_decode_attention(
    q: torch.Tensor,        # [B, H, L, Dh]
    k: torch.Tensor,        # [B, H, R, Dh]
    v: torch.Tensor,        # [B, H, R, Dh]
    mask: torch.Tensor,     # [B, H, L, R] bool (True == visible)
    scale: float,
) -> torch.Tensor:
    out_dtype = v.dtype
    scores = torch.einsum("bhld,bhrd->bhlr", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask, _NEG)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhlr,bhrd->bhld", probs, v.float()).to(out_dtype)


if HAS_TRITON:

    @triton.jit
    def _decode_attn_kernel(
        q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr,
        scale,
        B, H, L, R,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kr, stride_kd,
        stride_vb, stride_vh, stride_vr, stride_vd,
        stride_mb, stride_mh, stride_ml, stride_mr,
        stride_ob, stride_oh, stride_ol, stride_od,
        BLOCK_R: tl.constexpr,
        DHEAD: tl.constexpr,
    ):
        pid = tl.program_id(0)          # b*H*L + h*L + l
        l = pid % L
        bh = pid // L
        h = bh % H
        b = bh // H

        d = tl.arange(0, DHEAD)
        q = tl.load(
            q_ptr + b * stride_qb + h * stride_qh + l * stride_ql + d * stride_qd
        ).to(tl.float32)

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.zeros((), tl.float32)
        acc = tl.zeros((DHEAD,), tl.float32)

        for r0 in range(0, R, BLOCK_R):
            r = r0 + tl.arange(0, BLOCK_R)
            r_valid = r < R
            # k block [BLOCK_R, DHEAD]
            k_blk = tl.load(
                k_ptr + b * stride_kb + h * stride_kh
                + r[:, None] * stride_kr + d[None, :] * stride_kd,
                mask=r_valid[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k_blk, axis=1) * scale  # [BLOCK_R]
            vis = tl.load(
                mask_ptr + b * stride_mb + h * stride_mh + l * stride_ml + r * stride_mr,
                mask=r_valid, other=0,
            )
            keep = r_valid & (vis != 0)
            scores = tl.where(keep, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(keep, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            v_blk = tl.load(
                v_ptr + b * stride_vb + h * stride_vh
                + r[:, None] * stride_vr + d[None, :] * stride_vd,
                mask=r_valid[:, None], other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v_blk, axis=0)
            m_i = m_new

        # if every key was masked, l_i==0 -> emit zeros (matches softmax of all -inf -> nan
        # in reference; HGA guarantees >=1 visible key for any real query so this is a guard).
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out = acc / l_safe
        tl.store(
            out_ptr + b * stride_ob + h * stride_oh + l * stride_ol + d * stride_od,
            out.to(out_ptr.dtype.element_ty),
        )


def fused_decode_attention(
    q: torch.Tensor,        # [B, H, L, Dh]
    k: torch.Tensor,        # [B, H, R, Dh]
    v: torch.Tensor,        # [B, H, R, Dh]
    mask: torch.Tensor,     # [B, H, L, R] bool
    scale: float,
    block_r: int = 128,
) -> torch.Tensor:
    """Fused online-softmax attention over assembled routed KV.

    Equivalent to ``softmax((q@k^T)*scale + maskfill) @ v`` in fp32, returned in
    ``v.dtype``.  Uses Triton when available + on CUDA + ``head_dim`` a power of
    two; otherwise falls back to the exact PyTorch reference.
    """
    B, H, L, Dh = q.shape
    R = k.shape[2]
    pow2 = (Dh & (Dh - 1)) == 0
    if not (HAS_TRITON and q.is_cuda and pow2 and R > 0):
        return _torch_decode_attention(q, k, v, mask, scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mask = mask.contiguous()
    out = torch.empty((B, H, L, Dh), device=q.device, dtype=v.dtype)
    grid = (B * H * L,)
    mask_i = mask.to(torch.int8)
    _decode_attn_kernel[grid](
        q, k, v, mask_i, out,
        scale,
        B, H, L, R,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mask_i.stride(),
        *out.stride(),
        BLOCK_R=block_r,
        DHEAD=Dh,
        num_warps=4,
    )
    return out
