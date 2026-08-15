"""True two-level routed attention for Alpamayo (BHSD ``[B, H, S, 128]``).

128-token chunks, then 16-token groups. Default budget: 4 previous chunks,
11 groups + current group = 192 tokens.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .kernel import (
    BLOCK_CAND,
    BLOCK_CK,
    CHUNK,
    GPC,
    GROUP,
    HEAD_DIM,
    MAX_CHUNKS,
    MAX_GROUPS,
    TOPK_C,
    TOPK_G,
    _diff_hga2_vec_kernel,
    _vlm_hga2_kernel,
    chunk_route_vlm_fast,
    fill_chunk_keys,
    fill_hga_means,
    n_chunks,
)

DEFAULT_TOPK = TOPK_C
VLM_ATTN_NAME = "alpamayo_vlm_routed"
DIFFUSION_ATTN_NAME = "alpamayo_diffusion_routed"


def prompt_chunk_keys(
    k: torch.Tensor,
    n_prompt: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    chunk: int = CHUNK,
) -> torch.Tensor:
    if n_prompt is None:
        n_prompt = k.shape[2]
    b, h = k.shape[0], k.shape[1]
    need = n_chunks(n_prompt, chunk)
    if out is None or out.shape[0] != b or out.shape[1] != h or out.shape[2] < need:
        slots = max(need, MAX_CHUNKS if chunk == CHUNK else MAX_GROUPS)
        out = torch.empty(b, h, slots, HEAD_DIM, device=k.device, dtype=k.dtype)
    fill_chunk_keys(k, out, n_prompt, chunk=chunk)
    return out


def _check_heads(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    hq, kvh = q.shape[1], k.shape[1]
    if hq % kvh != 0:
        raise ValueError(f"nheads={hq} must be divisible by kv_heads={kvh}")
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    return hq, kvh


def _output_strides(out: Optional[torch.Tensor], q: torch.Tensor, b: int, hq: int, s: int):
    if out is None:
        return None
    d = q.shape[-1]
    if tuple(out.shape) == (b, hq, s, d):
        return out.stride(0), out.stride(1), out.stride(2)
    if tuple(out.shape) == (b, s, hq, d):
        return out.stride(0), out.stride(2), out.stride(1)
    raise ValueError(f"out shape {tuple(out.shape)} is neither BHSD nor BSHD")


def _need_means(k: torch.Tensor, seqlen: int, chunk: int, buf: Optional[torch.Tensor]) -> torch.Tensor:
    b, h = k.shape[0], k.shape[1]
    need = n_chunks(seqlen, chunk)
    if (
        buf is None
        or buf.shape[0] != b
        or buf.shape[1] != h
        or buf.shape[2] < need
        or buf.dtype != k.dtype
    ):
        slots = max(need, MAX_GROUPS if chunk == GROUP else MAX_CHUNKS)
        buf = torch.empty(b, h, slots, HEAD_DIM, device=k.device, dtype=k.dtype)
    fill_chunk_keys(k, buf, seqlen, chunk=chunk)
    return buf


def _need_hga_tables(
    k: torch.Tensor,
    seqlen: int,
    group_k: Optional[torch.Tensor],
    chunk_k: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, h = k.shape[0], k.shape[1]
    ng = n_chunks(seqlen, GROUP)
    nc = n_chunks(seqlen, CHUNK)
    if (
        group_k is None
        or group_k.shape[0] != b
        or group_k.shape[1] != h
        or group_k.shape[2] < ng
        or group_k.dtype != k.dtype
    ):
        group_k = torch.empty(b, h, max(ng, MAX_GROUPS), HEAD_DIM, device=k.device, dtype=k.dtype)
    if (
        chunk_k is None
        or chunk_k.shape[0] != b
        or chunk_k.shape[1] != h
        or chunk_k.shape[2] < nc
        or chunk_k.dtype != k.dtype
    ):
        chunk_k = torch.empty(b, h, max(nc, MAX_CHUNKS), HEAD_DIM, device=k.device, dtype=k.dtype)
    fill_hga_means(k, group_k, chunk_k, seqlen)
    return group_k, chunk_k


def _need_route(q: torch.Tensor, nc: int, buf: Optional[torch.Tensor]) -> torch.Tensor:
    b, hq = q.shape[0], q.shape[1]
    if buf is None or buf.shape[0] != b or buf.shape[1] != hq or buf.shape[2] < nc:
        buf = torch.empty(b, hq, max(nc, MAX_CHUNKS), TOPK_C, device=q.device, dtype=torch.int32)
    return buf


def vlm_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    chunk_k: Optional[torch.Tensor] = None,
    group_k: Optional[torch.Tensor] = None,
    route: Optional[torch.Tensor] = None,
    q_chunk: Optional[torch.Tensor] = None,
    **_ignored: Any,
) -> torch.Tensor:
    """Causal two-level self-attention. Extra kwargs ignored (old one-level API)."""
    if q.shape[2] != k.shape[2]:
        raise ValueError("vlm_prefill_attention expects q_len == k_len")
    hq, kvh = _check_heads(q, k)
    b, _, s, _ = q.shape
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, s)
    gk, ck = _need_hga_tables(k, s, group_k, chunk_k)
    nc = n_chunks(s, CHUNK)
    rt = _need_route(q, nc, route)
    chunk_route_vlm_fast(q, ck, rt, s, q_chunk=q_chunk)
    ng = n_chunks(s, GROUP)
    # Serial 16×16 attend is faster here than a 16×256 gather (coalesced K/V).
    _vlm_hga2_kernel[(ng, b * hq)](
        q, k, v, gk, rt,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        gk.stride(0), gk.stride(1),
        rt.stride(0), rt.stride(1),
        o_b, o_h, o_s,
        s, ng, hq, kvh, float(softmax_scale),
        TOPK_G=TOPK_G, TOPK_C=TOPK_C, GPC=GPC, GROUP=GROUP,
        BLOCK_Q=16, BLOCK_D=HEAD_DIM, BLOCK_CAND=BLOCK_CAND,
        num_warps=4, num_stages=2,
    )
    return out


def diffusion_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    n_prompt: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    chunk_k: Optional[torch.Tensor] = None,
    group_k: Optional[torch.Tensor] = None,
    route: Optional[torch.Tensor] = None,
    **_ignored: Any,
) -> torch.Tensor:
    """Diffusion queries attend 11 routed 16-groups of the prefix + all diffusion keys."""
    hq, kvh = _check_heads(q, k)
    b, _, q_len, _ = q.shape
    k_len = k.shape[2]
    if n_prompt is None:
        n_prompt = k_len - q_len if k_len > q_len else k_len
    n_prompt = max(0, min(int(n_prompt), k_len))
    if n_prompt + q_len > k_len:
        n_prompt = max(0, k_len - q_len)
    if q_len > 64:
        raise ValueError(f"diffusion kernel expects q_len≤64, got {q_len}")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    if out is None:
        out = torch.empty_like(q)
    o_b, o_h, o_s = _output_strides(out, q, b, hq, q_len)
    nc = n_chunks(n_prompt, CHUNK)
    have_ck = chunk_k is not None and chunk_k.shape[2] >= nc and chunk_k.dtype == k.dtype
    have_gk = group_k is not None and group_k.shape[2] >= n_chunks(n_prompt, GROUP) and group_k.dtype == k.dtype
    if have_ck and have_gk:
        ck, gk = chunk_k, group_k
    else:
        gk, ck = _need_hga_tables(k, n_prompt, group_k, chunk_k)
    if not getattr(diffusion_cross_attention, "_logged", False):
        print(
            f"[routed] 2L diffusion q={q.dtype} k={k.dtype} "
            f"q_len={q_len} n_prompt={n_prompt} chunks={nc} "
            f"topk_c={TOPK_C} topk_g={TOPK_G}",
            flush=True,
        )
        diffusion_cross_attention._logged = True
    _diff_hga2_vec_kernel[(b * hq,)](
        q, k, v, gk, ck,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        gk.stride(0), gk.stride(1),
        ck.stride(0), ck.stride(1),
        o_b, o_h, o_s,
        n_prompt, n_chunks(n_prompt, GROUP), nc, q_len, k_len, hq, kvh, float(softmax_scale),
        TOPK_G=TOPK_G, TOPK_C=TOPK_C, GPC=GPC, GROUP=GROUP,
        BLOCK_Q=64, BLOCK_D=HEAD_DIM, BLOCK_CAND=32, BLOCK_ATT=256, BLOCK_CK=BLOCK_CK,
        num_warps=4, num_stages=2,
    )
    return out


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
    if dropout not in (0, 0.0):
        raise NotImplementedError("routed VLM attention does not support dropout")
    if query.shape[2] != key.shape[2]:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, scale=scaling,
            is_causal=True, enable_gqa=query.shape[1] != key.shape[1],
        ).transpose(1, 2).contiguous(), None
    s, hq = query.shape[2], query.shape[1]
    out_bshd = getattr(module, "_alpamayo_out_bshd", None)
    dest = None
    if out_bshd is not None and out_bshd.shape[1] >= s and out_bshd.shape[2] == hq:
        dest = out_bshd[:, :s]
    result = vlm_prefill_attention(
        query, key, value, softmax_scale=scaling, out=dest,
        group_k=getattr(module, "_alpamayo_group_k", None),
        chunk_k=getattr(module, "_alpamayo_chunk_k", None),
        route=getattr(module, "_alpamayo_route", None),
        q_chunk=getattr(module, "_alpamayo_q_chunk", None),
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
    if dropout not in (0, 0.0):
        raise NotImplementedError("routed diffusion attention does not support dropout")
    n_prompt = getattr(module, "_alpamayo_n_prompt", None)
    chunk_k = getattr(module, "_alpamayo_chunk_k", None)
    if chunk_k is not None and not getattr(module, "_alpamayo_chunk_k_ready", False):
        chunk_k = None
    group_k = getattr(module, "_alpamayo_group_k", None)
    if group_k is not None and not getattr(module, "_alpamayo_group_k_ready", False):
        group_k = None
    q_len, hq = query.shape[2], query.shape[1]
    out_bshd = getattr(module, "_alpamayo_out_bshd", None)
    dest = None
    if out_bshd is not None and out_bshd.shape[1] >= q_len and out_bshd.shape[2] == hq:
        dest = out_bshd[:, :q_len]
    result = diffusion_cross_attention(
        query, key, value,
        softmax_scale=scaling, n_prompt=n_prompt, out=dest,
        chunk_k=chunk_k, group_k=group_k,
        route=getattr(module, "_alpamayo_route_diff", None),
    )
    if dest is not None:
        return dest, None
    return result.transpose(1, 2).contiguous(), None


def _stamp_module(mod: Any, name: str, n_prompt: Optional[int] = None) -> None:
    cfg = getattr(mod, "config", None)
    if cfg is not None:
        cfg._attn_implementation = name
        if hasattr(cfg, "is_causal") and name == DIFFUSION_ATTN_NAME:
            cfg.is_causal = False
    if hasattr(mod, "_attn_implementation"):
        mod._attn_implementation = name
    for child in mod.modules():
        if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
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
    **_ignored: Any,
) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if VLM_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(VLM_ATTN_NAME, hf_vlm_attention)
    if DIFFUSION_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(DIFFUSION_ATTN_NAME, hf_diffusion_attention)

    if vlm is not None:
        text = getattr(getattr(vlm, "model", vlm), "language_model", vlm)
        _stamp_module(text, VLM_ATTN_NAME)
        root_cfg = getattr(vlm, "config", None)
        if root_cfg is not None:
            text_cfg = getattr(root_cfg, "text_config", root_cfg)
            if text_cfg is not None:
                text_cfg._attn_implementation = VLM_ATTN_NAME

    if expert is not None:
        _stamp_module(expert, DIFFUSION_ATTN_NAME, n_prompt=n_prompt)


def set_diffusion_prompt(
    expert: Any,
    k_prefix: torch.Tensor,
    n_prompt: Optional[int] = None,
) -> torch.Tensor:
    n = int(n_prompt if n_prompt is not None else k_prefix.shape[2])
    gk, ck = _need_hga_tables(k_prefix, n, None, None)
    for child in expert.modules():
        if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
            child._alpamayo_chunk_k = ck
            child._alpamayo_group_k = gk
            child._alpamayo_n_prompt = n
            child._alpamayo_chunk_k_ready = True
            child._alpamayo_group_k_ready = True
    return ck
