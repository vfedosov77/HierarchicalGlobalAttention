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

from typing import Any, Optional, Tuple

import torch

from .kernel import (
    CHUNK,
    HEAD_DIM,
    N_BLOCK,
    _diffusion_fwd_kernel,
    _vlm_fwd_kernel,
    fill_chunk_keys,
    n_chunks,
)

DEFAULT_TOPK = 3
VLM_ATTN_NAME = "alpamayo_vlm_routed"
DIFFUSION_ATTN_NAME = "alpamayo_diffusion_routed"


def prompt_chunk_keys(
    k: torch.Tensor,
    n_prompt: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean key per 64-token prompt chunk. Call **once** when the VLM prefix is loaded.

    ``k`` is ``[B, H_kv, S, 128]`` (full cache) or just the prefix.
    Reuse the returned ``[B, H_kv, 64, 128]`` for every diffusion Euler step.
    """
    if n_prompt is None:
        n_prompt = k.shape[2]
    b, kvh = k.shape[0], k.shape[1]
    if out is None or out.shape != (b, kvh, N_BLOCK, HEAD_DIM):
        out = torch.empty(b, kvh, N_BLOCK, HEAD_DIM, device=k.device, dtype=k.dtype)
    fill_chunk_keys(k, out, n_prompt)
    return out


def _check_heads(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    hq, kvh = q.shape[1], k.shape[1]
    if hq % kvh != 0:
        raise ValueError(f"nheads={hq} must be divisible by kv_heads={kvh}")
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    return hq, kvh


def _need_chunk_k(
    k: torch.Tensor,
    seqlen: int,
    chunk_k: Optional[torch.Tensor],
    *,
    fill: bool = True,
) -> torch.Tensor:
    b, kvh = k.shape[0], k.shape[1]
    if chunk_k is None or chunk_k.shape != (b, kvh, N_BLOCK, HEAD_DIM):
        chunk_k = torch.empty(b, kvh, N_BLOCK, HEAD_DIM, device=k.device, dtype=k.dtype)
        fill = True
    if fill:
        fill_chunk_keys(k, chunk_k, seqlen)
    return chunk_k


def vlm_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk: int = DEFAULT_TOPK,
    out: Optional[torch.Tensor] = None,
    chunk_k: Optional[torch.Tensor] = None,
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
        Previous 64-token chunks opened besides the current chunk (3 → ~8.5% at S=3000).
    out, chunk_k
        Optional preallocated ``[B, H_q, S, 128]`` and ``[B, H_kv, 64, 128]``.

    Returns
    -------
    Tensor
        ``[B, H_q, S, 128]`` (same layout as ``q``, no transpose).
    """
    if q.shape[2] != k.shape[2]:
        raise ValueError("vlm_prefill_attention expects q_len == k_len (prefill self-attn)")
    hq, kvh = _check_heads(q, k)
    b, _, s, _ = q.shape
    if n_chunks(s) > N_BLOCK:
        raise ValueError(f"S={s} exceeds max {N_BLOCK * CHUNK}")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None or out.shape != q.shape:
        out = torch.empty_like(q)
    ck = _need_chunk_k(k, s, chunk_k)
    n = n_chunks(s)
    _vlm_fwd_kernel[(n, b * hq)](
        q, k, v, ck, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck.stride(0), ck.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        s, n, hq, kvh, float(softmax_scale),
        TOPK=int(topk), BLOCK_C=CHUNK, BLOCK_D=HEAD_DIM, BLOCK_N=N_BLOCK,
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
    if n_prompt is None:
        n_prompt = k_len - q_len
    if n_prompt < 0 or n_prompt + q_len > k_len:
        raise ValueError(f"bad n_prompt={n_prompt} for q_len={q_len} k_len={k_len}")
    if q_len > CHUNK:
        raise ValueError(f"diffusion kernel expects q_len≤{CHUNK}, got {q_len}")
    if n_chunks(n_prompt) > N_BLOCK:
        raise ValueError(f"prompt length {n_prompt} exceeds max {N_BLOCK * CHUNK}")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None or out.shape != q.shape:
        out = torch.empty_like(q)
    # Prefix K is fixed for the whole Euler loop — skip the mean if the caller
    # already built chunk_k (see prompt_chunk_keys).
    ck = _need_chunk_k(k, n_prompt, chunk_k, fill=chunk_k is None)
    n = n_chunks(n_prompt)
    _diffusion_fwd_kernel[(b * hq,)](
        q, k, v, ck, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        ck.stride(0), ck.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        n_prompt, n, q_len, hq, kvh, float(softmax_scale),
        TOPK=int(topk), BLOCK_C=CHUNK, BLOCK_D=HEAD_DIM, BLOCK_N=N_BLOCK,
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
    topk = int(getattr(module, "_alpamayo_topk", DEFAULT_TOPK))
    out = vlm_prefill_attention(query, key, value, softmax_scale=scaling, topk=topk)
    return out.transpose(1, 2).contiguous(), None


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
    n_prompt = getattr(module, "_alpamayo_n_prompt", None)
    chunk_k = getattr(module, "_alpamayo_chunk_k", None)
    out = diffusion_cross_attention(
        query, key, value,
        softmax_scale=scaling, topk=topk, n_prompt=n_prompt, chunk_k=chunk_k,
    )
    return out.transpose(1, 2).contiguous(), None


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
    return ck
