"""Optional FlashAttention-shaped wrapper (``[B, S, H, D]``).

Prefer :func:`vlm_prefill_attention` inside Alpamayo (already BHSD).
This exists so existing ``from flash_attn import flash_attn_func`` swaps still work.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .attention import vlm_prefill_attention
from .kernel import HEAD_DIM


def _expand_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hq, hk = q.shape[2], k.shape[2]
    if hk == hq:
        return k, v
    if hq % hk != 0:
        raise ValueError(f"GQA requires nheads={hq} divisible by kv_heads={hk}")
    rep = hq // hk
    return k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    topk: int = 3,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if not causal:
        raise NotImplementedError("this flash_attn_func wrapper is causal-only")
    if dropout_p != 0.0 or return_attn_probs or window_size != (-1, -1):
        raise NotImplementedError("dropout / probs / sliding window not supported")
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    k, v = _expand_kv(q, k, v)
    out = vlm_prefill_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        softmax_scale=softmax_scale,
        topk=topk,
    )
    return out.transpose(1, 2).contiguous()
