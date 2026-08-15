"""Alpamayo-native attention: BHSD ``[B, H, S, D]``, no Flash layout transpose.

Two kernels
-----------
``vlm_prefill_attention(q, k, v)``
    Causal self-attention for VLM **prefill** (visual tokens + any extra prompt
    tokens; no CoT decode). GQA-safe. ``q/k/v`` already projected and RoPE'd.

``diffusion_cross_attention(q, k, v)``
    Diffusion expert queries (typically 64 waypoints) attend a **routed** slice
    of the VLM prefix plus all diffusion keys. Replaces dense SDPA **and**
    ``record_attention`` (the full 64×3K softmax used only to pick tokens).

Both write ``[B, H, S_q, D]``. Optional ``out`` / ``chunk_k`` reuse caller
buffers so a CUDA-graphed loop allocates nothing on the hot path.

HuggingFace
-----------
``attach_to_alpamayo(vlm, expert)`` registers the two backends on
``ALL_ATTENTION_FUNCTIONS`` and points the modules at them. HF still does one
output ``transpose(1, 2)`` (its API); inputs are not transposed.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .kernel import (
    CHUNK,
    DOT_BF16,
    DOT_FP8_E4,
    DOT_FP8_E5,
    DOT_FP8_NATIVE,
    HEAD_DIM,
    N_BLOCK,
    ROUTE_BLOCKS,
    _diff_hga32_8_kernel,
    _diffusion_fwd_kernel,
    _vlm_fwd_kernel,
    _vlm_hga32_8_kernel,
    copy_to_fp8,
    fill_chunk_keys,
    n_chunks,
)

_DOT_NAME = {
    "bf16": DOT_BF16,
    "fp16": DOT_BF16,  # tensor cores still bf16; fp16 inputs cast the same way
    "fp8": DOT_FP8_NATIVE,
    "fp8e4": DOT_FP8_E4,
    "fp8e5": DOT_FP8_E5,
    "fp8native": DOT_FP8_NATIVE,
}


def resolve_dot_kind(name: Optional[str] = None) -> int:
    """0=bf16, 1=fp8 e4m3, 2=fp8 e5m2. Default ``HGA_DOT_DTYPE`` or bf16."""
    if name is None:
        name = os.environ.get("HGA_DOT_DTYPE", "bf16")
    key = str(name).strip().lower()
    if key not in _DOT_NAME:
        raise ValueError(f"unknown dot dtype {name!r}; expected one of {sorted(_DOT_NAME)}")
    return _DOT_NAME[key]

DEFAULT_TOPK = 3
VLM_ATTN_NAME = "alpamayo_vlm_routed"
DIFFUSION_ATTN_NAME = "alpamayo_diffusion_routed"


def prompt_chunk_keys(
    k: torch.Tensor,
    n_prompt: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    chunk: int = CHUNK,
) -> torch.Tensor:
    """Mean key per routing block. Call **once** when the VLM prefix is loaded.

    ``k`` is ``[B, H_kv, S, 128]``. Reuse ``[B, H_kv, N_BLOCK, 128]`` for every
    diffusion Euler step. ``chunk`` is the one-level block size (16/32/64/128).
    """
    if n_prompt is None:
        n_prompt = k.shape[2]
    b, kvh = k.shape[0], k.shape[1]
    if out is None or out.shape[0] != b or out.shape[1] != kvh or out.shape[-1] != HEAD_DIM:
        out = torch.empty(b, kvh, N_BLOCK, HEAD_DIM, device=k.device, dtype=k.dtype)
    elif out.shape[2] < n_chunks(n_prompt, chunk):
        out = torch.empty(b, kvh, N_BLOCK, HEAD_DIM, device=k.device, dtype=k.dtype)
    fill_chunk_keys(k, out, n_prompt, chunk=chunk)
    return out


def _check_heads(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    hq, kvh = q.shape[1], k.shape[1]
    if hq % kvh != 0:
        raise ValueError(f"nheads={hq} must be divisible by kv_heads={kvh}")
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    return hq, kvh


def _output_strides(
    out: Optional[torch.Tensor], q: torch.Tensor, b: int, hq: int, s: int
) -> Optional[Tuple[int, int, int]]:
    """``(stride_b, stride_h, stride_s)`` for BHSD ``[B,H,S,D]`` or HF BSHD ``[B,S,H,D]``."""
    if out is None:
        return None
    d = q.shape[-1]
    if tuple(out.shape) == (b, hq, s, d):
        return out.stride(0), out.stride(1), out.stride(2)
    if tuple(out.shape) == (b, s, hq, d):
        return out.stride(0), out.stride(2), out.stride(1)
    raise ValueError(
        f"out shape {tuple(out.shape)} is neither BHSD nor BSHD for B={b} H={hq} S={s} D={d}"
    )


def _use_native_fp8(module: Any) -> bool:
    kind = getattr(module, "_alpamayo_dot_kind", None)
    if kind is None:
        kind = resolve_dot_kind()
    return int(kind) == DOT_FP8_NATIVE and hasattr(torch, "float8_e4m3fn")


def _fp8_view(buf: Optional[torch.Tensor], b: int, h: int, s: int) -> Optional[torch.Tensor]:
    if buf is None or buf.dtype != torch.float8_e4m3fn:
        return None
    if buf.shape[0] < b or buf.shape[1] < h or buf.shape[2] < s or buf.shape[3] != HEAD_DIM:
        return None
    return buf[:b, :h, :s]


def _pack_fp8(src: torch.Tensor, buf: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    dest = _fp8_view(buf, src.shape[0], src.shape[1], src.shape[2])
    if dest is None:
        return None
    copy_to_fp8(src, dest)
    return dest


def _need_chunk_k(
    k: torch.Tensor,
    seqlen: int,
    chunk_k: Optional[torch.Tensor],
    *,
    fill: bool = True,
    chunk: int = CHUNK,
) -> torch.Tensor:
    b, kvh = k.shape[0], k.shape[1]
    need = n_chunks(seqlen, chunk)
    if (
        chunk_k is None
        or chunk_k.shape[0] != b
        or chunk_k.shape[1] != kvh
        or chunk_k.shape[-1] != HEAD_DIM
        or chunk_k.shape[2] < need
    ):
        chunk_k = torch.empty(b, kvh, N_BLOCK, HEAD_DIM, device=k.device, dtype=k.dtype)
        fill = True
    if fill:
        if k.dtype == getattr(torch, "float8_e4m3fn", None):
            raise TypeError("fill_chunk_keys needs bf16/fp32 K; pass precomputed chunk_k")
        fill_chunk_keys(k, chunk_k, seqlen, chunk=chunk)
    return chunk_k


# Fine-mean table can be larger than N_BLOCK (e.g. 4-token tiles → S/4 slots).
_HGA2_ROUTE_N = 128  # power-of-two coarse slots; covers coarse ≥ 32 at S ≤ 4096
_HGA2_MAX_FINE_PER_COARSE = 16  # kernel pads fine routing / attend to 16


def parse_hierarchy(hierarchy: Tuple[int, int]) -> Tuple[int, int, int]:
    """Return ``(coarse, fine, fine_per_coarse)``. Fine tiles per coarse must be 1..16."""
    if len(hierarchy) != 2:
        raise ValueError(f"hierarchy must be (coarse, fine), got {hierarchy!r}")
    coarse, fine = int(hierarchy[0]), int(hierarchy[1])
    if coarse not in ROUTE_BLOCKS:
        raise ValueError(f"hierarchy coarse must be one of {ROUTE_BLOCKS}, got {coarse}")
    if fine < 1 or coarse % fine != 0:
        raise ValueError(f"fine={fine} must divide coarse={coarse}")
    nfine = coarse // fine
    if nfine > _HGA2_MAX_FINE_PER_COARSE:
        raise ValueError(
            f"{coarse}/{fine} has {nfine} fine tiles; kernel max is {_HGA2_MAX_FINE_PER_COARSE}"
        )
    return coarse, fine, nfine


def _need_fine_means(
    k: torch.Tensor,
    seqlen: int,
    fine: int,
    buf: Optional[torch.Tensor],
    fill: Optional[bool] = None,
) -> torch.Tensor:
    b, kvh = k.shape[0], k.shape[1]
    need = n_chunks(seqlen, fine)
    if fill is None:
        fill = buf is None
    have = (
        buf is not None
        and buf.shape[0] == b
        and buf.shape[1] == kvh
        and buf.shape[-1] == HEAD_DIM
        and buf.shape[2] >= need
    )
    # Means may stay bf16 while attend K is e4m3 — do not realloc on dtype mismatch.
    if not have:
        if k.dtype == getattr(torch, "float8_e4m3fn", None):
            raise TypeError("fill_chunk_keys needs bf16/fp32 K; pass precomputed chunk_k_fine")
        buf = torch.empty(b, kvh, need, HEAD_DIM, device=k.device, dtype=k.dtype)
        fill = True
    if fill:
        if k.dtype == getattr(torch, "float8_e4m3fn", None):
            raise TypeError("fill_chunk_keys needs bf16/fp32 K; pass precomputed chunk_k_fine")
        fill_chunk_keys(k, buf, seqlen, chunk=fine)
    return buf


# VLM tokens = (dense current coarse | current fine) + topk_c * topk_f * fine.
_HGA2_SPARSITY = {
    (128, 16): {
        8: (2, 4, True),   # 128 + 128 = 256
        4: (7, 1, False),  # 16 + 112 = 128
        2: (1, 3, False),  # 16 + 48  = 64
    },
    (64, 8): {
        # 4%: keep local 64 and route eight 8-token tiles (better local than 128/16 @ 4%).
        4: (2, 4, True),   # 64 + 64 = 128
    },
}


def resolve_hga_from_env() -> Tuple[Optional[Tuple[int, int]], int, int, bool]:
    """``(hierarchy or None, topk_c, topk_f, dense_current)``."""
    raw = os.environ.get("HGA_HIERARCHY", "").strip()
    hier: Optional[Tuple[int, int]] = None
    if raw:
        parts = raw.replace(",", "/").replace("x", "/").replace("-", "/").split("/")
        if len(parts) != 2:
            raise ValueError(f"HGA_HIERARCHY={raw!r}; expected coarse/fine e.g. 128/16")
        hier = (int(parts[0]), int(parts[1]))
        parse_hierarchy(hier)
    topk_c = int(os.environ.get("HGA_TOPK", os.environ.get("HGA_TOPK_C", str(DEFAULT_TOPK))))
    topk_f = int(os.environ.get("HGA_TOPK_FINE", "2"))
    dense = os.environ.get("HGA_DENSE_CURRENT", "").strip().lower() in ("1", "true", "yes")
    sp = os.environ.get("HGA_SPARSITY", "").strip()
    if sp:
        pct = int(str(sp).replace("%", ""))
        if hier is None:
            raise ValueError("HGA_SPARSITY requires HGA_HIERARCHY")
        table = _HGA2_SPARSITY.get(hier)
        if table is None or pct not in table:
            raise ValueError(f"HGA_SPARSITY={pct} not mapped for hierarchy {hier}")
        topk_c, topk_f, dense = table[pct]
    elif hier == (128, 16) and not os.environ.get("HGA_DENSE_CURRENT"):
        # Default 128/16 without HGA_SPARSITY: 8% dense-current.
        dense = True
    return hier, topk_c, topk_f, dense


def hga2_vlm_tokens(coarse: int, fine: int, topk_c: int, topk_f: int, dense_current: bool) -> int:
    local = int(coarse) if dense_current else int(fine)
    return local + int(topk_c) * int(topk_f) * int(fine)


def hga2_diff_tokens(fine: int, topk_c: int, topk_f: int) -> int:
    return int(topk_c) * int(topk_f) * int(fine)


def _vlm_hga2(
    q, k, v, hq, kvh, b, s, *, softmax_scale, topk, topk_fine, out, dot_kind,
    hierarchy, chunk_k, chunk_k_fine, dense_current: bool = False,
):
    coarse, fine, nfine = parse_hierarchy(hierarchy)
    if int(topk_fine) > nfine:
        raise ValueError(f"topk_fine={topk_fine} exceeds {nfine} tiles in a {coarse}/{fine} block")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, s)
    n_c = n_chunks(s, coarse)
    if n_c > _HGA2_ROUTE_N:
        raise ValueError(f"S={s} coarse={coarse} needs {n_c} blocks; max {_HGA2_ROUTE_N}")
    # Always refill VLM means from this layer's K (buffer is only to avoid malloc).
    fp8 = getattr(torch, "float8_e4m3fn", None)
    fill = k.dtype != fp8
    ck_c = _need_chunk_k(k, s, chunk_k, fill=fill, chunk=coarse)
    ck_f = _need_fine_means(k, s, fine, chunk_k_fine, fill=fill)
    kind = int(resolve_dot_kind() if dot_kind is None else dot_kind)
    if dense_current:
        grid_n = n_c
        q_tile = int(coarse)
        warps = 8
    else:
        grid_n = n_chunks(s, fine)
        q_tile = 16
        warps = 4
    _vlm_hga32_8_kernel[(grid_n, b * hq)](
        q, k, v, ck_c, ck_f, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck_c.stride(0), ck_c.stride(1),
        ck_f.stride(0), ck_f.stride(1),
        o_b, o_h, o_s,
        s, n_c, hq, kvh, float(softmax_scale),
        TOPK_C=int(topk), TOPK_F=int(topk_fine),
        BLOCK_C=q_tile, BLOCK_F=int(fine), BLOCK_D=HEAD_DIM,
        BLOCK_N=_HGA2_ROUTE_N, NFINE=int(nfine), COARSE=int(coarse),
        DENSE_CUR=int(bool(dense_current)),
        DOT_KIND=kind,
        num_warps=warps, num_stages=2,
    )
    return out


def _diff_hga2(
    q, k, v, hq, kvh, b, q_len, k_len, n_prompt, *, softmax_scale, topk, topk_fine, out, dot_kind,
    hierarchy, chunk_k, chunk_k_fine,
):
    coarse, fine, nfine = parse_hierarchy(hierarchy)
    if int(topk_fine) > nfine:
        raise ValueError(f"topk_fine={topk_fine} exceeds {nfine} tiles in a {coarse}/{fine} block")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, q_len)
    n_c = n_chunks(n_prompt, coarse)
    if n_c > _HGA2_ROUTE_N:
        raise ValueError(f"prompt={n_prompt} coarse={coarse} needs {n_c} blocks; max {_HGA2_ROUTE_N}")
    fp8 = getattr(torch, "float8_e4m3fn", None)
    # Prefix means are filled once/frame in bf16. Attend K may already be e4m3.
    fill_c = chunk_k is None and k.dtype != fp8
    ck_c = _need_chunk_k(k, n_prompt, chunk_k, fill=fill_c, chunk=coarse)
    ck_f = _need_fine_means(
        k, n_prompt, fine, chunk_k_fine,
        fill=False if chunk_k_fine is not None else None,
    )
    # Fine/coarse means stay bf16; expert K/V may be e4m3. Cast in-register.
    kind = DOT_BF16
    _diff_hga32_8_kernel[(b * hq,)](
        q, k, v, ck_c, ck_f, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck_c.stride(0), ck_c.stride(1),
        ck_f.stride(0), ck_f.stride(1),
        o_b, o_h, o_s,
        n_prompt, n_c, q_len, k_len, hq, kvh, float(softmax_scale),
        TOPK_C=int(topk), TOPK_F=int(topk_fine),
        BLOCK_C=CHUNK, BLOCK_F=int(fine), BLOCK_D=HEAD_DIM,
        BLOCK_N=_HGA2_ROUTE_N, NFINE=int(nfine), COARSE=int(coarse),
        DOT_KIND=kind,
        num_warps=8, num_stages=2,
    )
    return out


def vlm_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk: int = DEFAULT_TOPK,
    out: Optional[torch.Tensor] = None,
    chunk_k: Optional[torch.Tensor] = None,
    dot_kind: Optional[int] = None,
    route_block: int = CHUNK,
    hierarchy: Optional[Tuple[int, int]] = None,
    topk_fine: int = 2,
    chunk_k_fine: Optional[torch.Tensor] = None,
    dense_current: bool = False,
) -> torch.Tensor:
    """Causal routed self-attention for VLM prefill.

    Parameters
    ----------
    q, k, v
        ``[B, H_q, S, 128]`` and ``[B, H_kv, S, 128]`` (GQA: H_q=32, H_kv=8
        on Qwen3-VL-8B). Same sequence length. Already RoPE'd.
    softmax_scale
        Defaults to ``head_dim ** -0.5``.
    topk
        Previous chunks opened besides the current chunk (3 × 64 → ~8.5% at S=3000).
    hierarchy
        If set, two-level routing ``(coarse, fine)`` (e.g. ``(64, 8)``). ``topk``
        is coarse winners; ``topk_fine`` is fine tiles inside each winner.
    out, chunk_k
        Optional preallocated ``[B, H_q, S, 128]`` and coarse-mean keys.
    chunk_k_fine
        Optional fine-mean keys when ``hierarchy`` is set.

    Returns
    -------
    Tensor
        ``[B, H_q, S, 128]`` (same layout as ``q``, no transpose).
    """
    if q.shape[2] != k.shape[2]:
        raise ValueError("vlm_prefill_attention expects q_len == k_len (prefill self-attn)")
    hq, kvh = _check_heads(q, k)
    b, _, s, _ = q.shape
    if hierarchy is not None:
        return _vlm_hga2(
            q, k, v, hq, kvh, b, s,
            softmax_scale=softmax_scale, topk=topk, topk_fine=topk_fine,
            out=out, dot_kind=dot_kind, hierarchy=hierarchy,
            chunk_k=chunk_k, chunk_k_fine=chunk_k_fine,
            dense_current=dense_current,
        )
    if route_block not in ROUTE_BLOCKS:
        raise ValueError(f"route_block must be one of {ROUTE_BLOCKS}, got {route_block}")
    if n_chunks(s, route_block) > N_BLOCK:
        raise ValueError(f"S={s} C={route_block} exceeds max {N_BLOCK} blocks")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, s)
    ck = _need_chunk_k(
        k, s, chunk_k,
        fill=chunk_k is None or k.dtype != chunk_k.dtype,
        chunk=route_block,
    )
    n = n_chunks(s, route_block)
    _vlm_fwd_kernel[(n, b * hq)](
        q, k, v, ck, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck.stride(0), ck.stride(1),
        o_b, o_h, o_s,
        s, n, hq, kvh, float(softmax_scale),
        TOPK=int(topk), BLOCK_C=int(route_block), BLOCK_D=HEAD_DIM, BLOCK_N=N_BLOCK,
        DOT_KIND=int(resolve_dot_kind() if dot_kind is None else dot_kind),
        num_warps=8, num_stages=2,
    )
    return out


def diffusion_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk: int = DEFAULT_TOPK,
    n_prompt: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    chunk_k: Optional[torch.Tensor] = None,
    dot_kind: Optional[int] = None,
    route_block: int = CHUNK,
    hierarchy: Optional[Tuple[int, int]] = None,
    topk_fine: int = 2,
    chunk_k_fine: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Diffusion queries attend routed VLM-prefix chunks + all diffusion keys.

    This **does not** run ``record_attention``. Routing is the same cheap
    chunk-mean top-k as the VLM kernel (SRAM table of ~48 keys), not a full
    softmax over every prefix token.

    Parameters
    ----------
    q
        ``[B, H_q, Q, 128]`` — typically Q=64 waypoints.
    k, v
        ``[B, H_kv, n_prompt + Q, 128]`` from the expert cache (prefix then
        diffusion slots). If ``n_prompt`` is omitted it is ``k_len - q_len``.
    topk
        Prompt chunks to open (3 × 64 = 192 prefix tokens, plus all Q diffusion keys).

    Returns
    -------
    Tensor
        ``[B, H_q, Q, 128]``.
    """
    hq, kvh = _check_heads(q, k)
    b, _, q_len, _ = q.shape
    k_len = k.shape[2]
    if hierarchy is not None:
        n_p = n_prompt
        if n_p is None:
            n_p = k_len - q_len if k_len > q_len else k_len
        if n_p < 0:
            n_p = 0
        if n_p > k_len:
            n_p = k_len
        if n_p + q_len > k_len:
            n_p = max(0, k_len - q_len)
        return _diff_hga2(
            q, k, v, hq, kvh, b, q_len, k_len, n_p,
            softmax_scale=softmax_scale, topk=topk, topk_fine=topk_fine,
            out=out, dot_kind=dot_kind, hierarchy=hierarchy,
            chunk_k=chunk_k, chunk_k_fine=chunk_k_fine,
        )
    if n_prompt is None:
        n_prompt = k_len - q_len if k_len > q_len else k_len
    if n_prompt < 0:
        n_prompt = 0
    if n_prompt > k_len:
        n_prompt = k_len
    # Eager expert cache may still be prefix-only (no diffusion tail yet).
    if n_prompt + q_len > k_len:
        n_prompt = max(0, k_len - q_len)
    if route_block not in ROUTE_BLOCKS:
        raise ValueError(f"route_block must be one of {ROUTE_BLOCKS}, got {route_block}")
    if q_len > CHUNK:
        raise ValueError(f"diffusion kernel expects q_len≤{CHUNK}, got {q_len}")
    if n_chunks(n_prompt, route_block) > N_BLOCK:
        raise ValueError(f"prompt length {n_prompt} C={route_block} exceeds max {N_BLOCK} blocks")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, q_len)
    # Prefix K is fixed for the whole Euler loop — skip the mean if the caller
    # already built chunk_k (see prompt_chunk_keys).
    ck = _need_chunk_k(k, n_prompt, chunk_k, fill=chunk_k is None, chunk=route_block)
    n = n_chunks(n_prompt, route_block)
    if not getattr(diffusion_cross_attention, "_logged", False):
        kind = int(resolve_dot_kind() if dot_kind is None else dot_kind)
        print(
            f"[routed] diffusion q={q.dtype} k={k.dtype} ck={ck.dtype} "
            f"q_len={q_len} k_len={k_len} n_prompt={n_prompt} n_chunks={n} "
            f"route_block={route_block} dot_kind={kind}",
            flush=True,
        )
        diffusion_cross_attention._logged = True
    _diffusion_fwd_kernel[(b * hq,)](
        q, k, v, ck, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck.stride(0), ck.stride(1),
        o_b, o_h, o_s,
        n_prompt, n, q_len, k_len, hq, kvh, float(softmax_scale),
        TOPK=int(topk), BLOCK_C=CHUNK, BLOCK_R=int(route_block),
        BLOCK_D=HEAD_DIM, BLOCK_N=N_BLOCK,
        DOT_KIND=int(resolve_dot_kind() if dot_kind is None else dot_kind),
        num_warps=8, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# HuggingFace ALL_ATTENTION_FUNCTIONS adapters
# ---------------------------------------------------------------------------

def hf_vlm_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """HF signature. ``query`` is ``[B, H, S, D]``; returns ``([B, S, H, D], None)``."""
    if dropout not in (0, 0.0):
        raise NotImplementedError("routed VLM attention does not support dropout")
    # Decode / cache-extend steps have q_len != k_len. The specialized kernel
    # is prefill-only; SDPA is faster for q_len=1 anyway.
    if query.shape[2] != key.shape[2]:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=scaling,
            is_causal=True,
            enable_gqa=query.shape[1] != key.shape[1],
        ).transpose(1, 2).contiguous(), None
    topk = int(getattr(module, "_alpamayo_topk", DEFAULT_TOPK))
    route_block = int(getattr(module, "_alpamayo_route_block", CHUNK))
    hierarchy = getattr(module, "_alpamayo_hierarchy", None)
    topk_fine = int(getattr(module, "_alpamayo_topk_fine", 2))
    dense_current = bool(getattr(module, "_alpamayo_dense_current", False))
    s = query.shape[2]
    hq = query.shape[1]
    out_bshd = getattr(module, "_alpamayo_out_bshd", None)
    if (
        out_bshd is not None
        and out_bshd.shape[1] >= s
        and out_bshd.shape[2] == hq
        and out_bshd.dtype != torch.float8_e4m3fn
    ):
        dest = out_bshd[:, :s]
    else:
        dest = None
    ck = getattr(module, "_alpamayo_chunk_k", None)
    ckf = getattr(module, "_alpamayo_chunk_k_fine", None)
    q_in, k_in, v_in, ck_in = query, key, value, ck
    kind = getattr(module, "_alpamayo_dot_kind", None)
    if _use_native_fp8(module):
        q8 = _pack_fp8(query, getattr(module, "_alpamayo_q_fp8", None))
        k8 = _pack_fp8(key, getattr(module, "_alpamayo_k_fp8", None))
        v8 = _pack_fp8(value, getattr(module, "_alpamayo_v_fp8", None))
        if q8 is not None and k8 is not None and v8 is not None:
            q_in, k_in, v_in = q8, k8, v8
            kind = DOT_FP8_NATIVE
            if ck is not None and ck.dtype != torch.float8_e4m3fn:
                fill_chunk_keys(key, ck, s, chunk=route_block)
                ck8 = getattr(module, "_alpamayo_ck_fp8", None)
                packed = _pack_fp8(ck, ck8) if ck8 is not None else None
                ck_in = packed
            elif ck is not None and ck.dtype == torch.float8_e4m3fn:
                ck_in = ck
    result = vlm_prefill_attention(
        q_in, k_in, v_in, softmax_scale=scaling, topk=topk, out=dest, chunk_k=ck_in,
        dot_kind=kind, route_block=route_block,
        hierarchy=hierarchy, topk_fine=topk_fine, chunk_k_fine=ckf,
        dense_current=dense_current,
    )
    if dest is not None:
        return dest, None
    return result.transpose(1, 2).contiguous(), None


def hf_diffusion_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """HF signature for the expert. Does **not** call ``record_attention``."""
    if dropout not in (0, 0.0):
        raise NotImplementedError("routed diffusion attention does not support dropout")
    topk = int(getattr(module, "_alpamayo_topk", DEFAULT_TOPK))
    route_block = int(getattr(module, "_alpamayo_route_block", CHUNK))
    hierarchy = getattr(module, "_alpamayo_hierarchy", None)
    topk_fine = int(getattr(module, "_alpamayo_topk_fine", 2))
    n_prompt = getattr(module, "_alpamayo_n_prompt", None)
    chunk_k = getattr(module, "_alpamayo_chunk_k", None)
    if chunk_k is not None and not getattr(module, "_alpamayo_chunk_k_ready", False):
        chunk_k = None  # buffer exists but averages not written this frame yet
    chunk_k_fine = getattr(module, "_alpamayo_chunk_k_fine", None)
    if chunk_k_fine is not None and not getattr(module, "_alpamayo_chunk_k_fine_ready", False):
        chunk_k_fine = None
    q_len, hq = query.shape[2], query.shape[1]
    out_bshd = getattr(module, "_alpamayo_out_bshd", None)
    dest = None
    if (
        out_bshd is not None
        and out_bshd.shape[1] >= q_len
        and out_bshd.shape[2] == hq
        and out_bshd.dtype != torch.float8_e4m3fn
    ):
        dest = out_bshd[:, :q_len]
    q_in, k_in, v_in, ck_in = query, key, value, chunk_k
    kind = getattr(module, "_alpamayo_dot_kind", None)
    if _use_native_fp8(module):
        q8 = _pack_fp8(query, getattr(module, "_alpamayo_q_fp8", None))
        k8_buf = getattr(module, "_alpamayo_k_fp8", None)
        v8_buf = getattr(module, "_alpamayo_v_fp8", None)
        ck8 = getattr(module, "_alpamayo_ck_fp8", None)
        k_len = key.shape[2]
        k8 = _fp8_view(k8_buf, key.shape[0], key.shape[1], k_len)
        v8 = _fp8_view(v8_buf, value.shape[0], value.shape[1], k_len)
        prefix_n = int(getattr(module, "_alpamayo_kv_fp8_n", 0) or 0)
        if q8 is not None and k8 is not None and v8 is not None:
            if prefix_n > 0 and k_len > prefix_n:
                copy_to_fp8(key[:, :, prefix_n:k_len], k8[:, :, prefix_n:k_len])
                copy_to_fp8(value[:, :, prefix_n:k_len], v8[:, :, prefix_n:k_len])
            elif prefix_n < k_len:
                copy_to_fp8(key, k8)
                copy_to_fp8(value, v8)
            q_in, k_in, v_in = q8, k8, v8
            kind = DOT_FP8_NATIVE
            if chunk_k is not None and ck8 is not None and chunk_k.dtype != torch.float8_e4m3fn:
                ck_in = _pack_fp8(chunk_k, ck8)
            elif ck8 is not None:
                ck_in = ck8
            elif chunk_k is not None and chunk_k.dtype == torch.float8_e4m3fn:
                ck_in = chunk_k
    result = diffusion_cross_attention(
        q_in, k_in, v_in,
        softmax_scale=scaling, topk=topk, n_prompt=n_prompt, chunk_k=ck_in, out=dest,
        dot_kind=kind, route_block=route_block,
        hierarchy=hierarchy, topk_fine=topk_fine, chunk_k_fine=chunk_k_fine,
    )
    if dest is not None:
        return dest, None
    return result.transpose(1, 2).contiguous(), None


def _stamp_module(mod: Any, name: str, topk: int, n_prompt: Optional[int] = None) -> None:
    cfg = getattr(mod, "config", None)
    if cfg is not None:
        cfg._attn_implementation = name
        if hasattr(cfg, "is_causal") and name == DIFFUSION_ATTN_NAME:
            cfg.is_causal = False
    if hasattr(mod, "_attn_implementation"):
        mod._attn_implementation = name
    for child in mod.modules():
        if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
            child._alpamayo_topk = topk
            if n_prompt is not None:
                child._alpamayo_n_prompt = int(n_prompt)
            ccfg = getattr(child, "config", None)
            if ccfg is not None:
                ccfg._attn_implementation = name


def attach_to_alpamayo(
    vlm: Optional[Any] = None,
    expert: Optional[Any] = None,
    *,
    topk: int = DEFAULT_TOPK,
    n_prompt: Optional[int] = None,
) -> None:
    """Point Alpamayo's VLM and/or diffusion expert at the routed kernels.

    Call once after the model is loaded, before CUDA-graph capture::

        from SpecializedKernels.Len3000HeadDim128 import attach_to_alpamayo
        attach_to_alpamayo(model.vlm, model.expert, topk=3)

    Turn **off** ``kv_top_fraction < 1`` / ``record_attention`` on the expert
    (use a static prompt cache). Routing is inside ``diffusion_cross_attention``.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if VLM_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(VLM_ATTN_NAME, hf_vlm_attention)
    if DIFFUSION_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(DIFFUSION_ATTN_NAME, hf_diffusion_attention)

    if vlm is not None:
        # Qwen3-VL: language-model layers only (vision encoder is head_dim=72).
        text = getattr(getattr(vlm, "model", vlm), "language_model", vlm)
        _stamp_module(text, VLM_ATTN_NAME, topk)
        root_cfg = getattr(vlm, "config", None)
        if root_cfg is not None:
            text_cfg = getattr(root_cfg, "text_config", root_cfg)
            if text_cfg is not None:
                text_cfg._attn_implementation = VLM_ATTN_NAME

    if expert is not None:
        _stamp_module(expert, DIFFUSION_ATTN_NAME, topk, n_prompt=n_prompt)


def set_diffusion_prompt(
    expert: Any,
    k_prefix: torch.Tensor,
    n_prompt: Optional[int] = None,
) -> torch.Tensor:
    """Call once per frame after the VLM prefix is copied into the expert cache.

    Stores chunk-mean keys on every expert attention layer so the 10 Euler
    steps do not re-mean the prefix.
    """
    ck = prompt_chunk_keys(k_prefix, n_prompt=n_prompt)
    n = int(n_prompt if n_prompt is not None else k_prefix.shape[2])
    for child in expert.modules():
        if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
            child._alpamayo_chunk_k = ck
            child._alpamayo_n_prompt = n
            child._alpamayo_chunk_k_ready = True
    return ck
