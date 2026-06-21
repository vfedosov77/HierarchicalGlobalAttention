"""Chunk-routed causal attention for the 40M SmallLM benchmark.

Same approach as ``QwenRoutedAttention`` in ``chat_qwen30b_fp8.py``: routing only
*selects* which previous chunks/groups each query attends to; attention is then
computed over the **real token K/V** of the selected items — no summary vector
ever enters the attention math, so a model that was never trained on the chunk
summaries (here: dense weights) is not corrupted.

Architecture (this is the whole point — keep it thin):

* The :class:`ChunkRouter` does *selection only*.  It scores queries against the
  resident chunk-/group-**key** summaries to pick the top chunks/groups, pulls
  their real token K/V into VRAM, and hands them back as a :class:`RoutedKV`.
  Group **value** summaries are a cheap by-product that we never attend to.
* The *attention* (``RoutedKV.attend(..., use_summaries=False)``) computes every
  q·k score, the softmax and the output — over real tokens only.
* ``route_query_block`` is a single entry point that auto-selects the fast
  chunk-parallel path for a fresh multi-chunk prefill and the incremental path
  for single-token decode, and seeds the KV-cache store so decode continues
  seamlessly.  No bespoke prefill loop, no ``ExactChunkRouter`` needed.

Training / cache-less eval use plain causal SDPA (stable for fine-tuning from a
dense checkpoint).  The ``__init__`` signature is a superset of the old
``HierarchicalGlobalAttention`` so ``build_ha_model`` needs no changes.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from KvRouter import ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore
    from KvRouter.cache_store import ChunkPlacementPolicy
except ModuleNotFoundError:
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore,
    )
    from ExistingModelFineTuning.KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore


try:
    from utilities import RMSNorm  # type: ignore
except Exception:
    class RMSNorm(nn.Module):  # type: ignore[no-redef]
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = x.float()
            out = self.weight.float() * x_fp32 * torch.rsqrt(
                x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return out.to(dtype=x.dtype)


class HierarchicalGlobalAttentionRouted(nn.Module):
    """Chunk-routed causal self-attention backed by :class:`ChunkRouter`."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 16,
        kv_heads: int = 8,
        dropout: float = 0.0,
        use_bias_q: bool = False,
        use_bias_k: bool = False,
        use_bias_v: bool = False,
        use_bias_o: bool = False,
        causal: bool = True,
        chunk_size: int = 64,
        group_size: int = 16,
        topk_chunks: int = 20,
        topk_groups: int = 32,
        keep_first: int = 1,
        keep_last: int = 2,
        head_dim: Optional[int] = None,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
        theta: float = 1_000_000.0,
        layer_idx: int = 0,
        cache_location: str = "vram",
        num_layers: int = 0,
        **kwargs: Any,          # absorb unused HA kwargs (topk_groups, return_router_stats, …)
    ) -> None:
        super().__init__()
        assert nhead % kv_heads == 0
        head_dim = d_model // nhead if head_dim is None else head_dim

        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.theta = float(theta)
        self.dropout_p = dropout
        self.layer_idx = layer_idx
        self.cache_location = cache_location
        self.num_layers = num_layers
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, nhead * head_dim, bias=use_bias_q)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_k)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_v)
        self.o_proj = nn.Linear(nhead * head_dim, d_model, bias=use_bias_o)

        if qk_norm:
            self.q_norm: nn.Module = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm: nn.Module = RMSNorm(head_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self._cfg = RouterConfig(
            nhead=nhead, kv_heads=kv_heads, head_dim=head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups,
            theta=theta,
        )
        self._policy = ChunkPlacementPolicy(
            keep_first=keep_first, keep_last=keep_last, first_token_level=True,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Any = None,
        cache_position: Optional[torch.Tensor] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, dict]:
        x = hidden_states
        B, S, _ = x.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        rep = H // KVH

        q_raw = self.q_norm(self.q_proj(x).view(B, S, H, Dh)).transpose(1, 2)    # [B, H,   S, Dh]
        k_raw = self.k_norm(self.k_proj(x).view(B, S, KVH, Dh)).transpose(1, 2)  # [B, KVH, S, Dh]
        v     = self.v_proj(x).view(B, S, KVH, Dh).transpose(1, 2)               # [B, KVH, S, Dh]

        cos, sin = self._get_rotary_for(position_embeddings, cache_position, B, S, x.device)
        q_rope = self._apply_rotary(q_raw.float(), cos, sin).to(q_raw.dtype)
        k_rope = self._apply_rotary(k_raw.float(), cos, sin).to(k_raw.dtype)

        # Training or eval without cache → plain causal SDPA.
        if self.training or past_key_value is None:
            k_exp = k_rope.repeat_interleave(rep, dim=1)
            v_exp = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(
                q_rope, k_exp, v_exp,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            out = self.o_proj(out.transpose(1, 2).contiguous().view(B, S, H * Dh))
            return out, {}

        # Inference with KV cache → route + attend over real tokens.
        start_pos = 0
        if cache_position is not None and cache_position.numel() > 0:
            start_pos = int(cache_position.reshape(-1)[0].item())

        router = self._get_router(past_key_value, B, x.dtype, x.device)
        if self.layer_idx == 0 and start_pos == 0:
            router.reset()

        cos_r, sin_r = cos[:1].to(x.dtype), sin[:1].to(x.dtype)   # [1, 1, S, Dh] for the vectorized path
        segments = router.route_query_block(
            self.layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            out_heads[:, :, lo:hi] = routed.attend(q_rope[:, :, lo:hi], use_summaries=False)

        out = self.o_proj(out_heads.transpose(1, 2).contiguous().view(B, S, H * Dh))
        return out, {}

    # ------------------------------------------------------------------
    def _get_rotary_for(
        self,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.Tensor],
        B: int,
        S: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.to(device=device, dtype=torch.float32)
            sin = sin.to(device=device, dtype=torch.float32)
            if cos.dim() == 2:          # [S, D]
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.dim() == 3:        # [B, S, D]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            if cos.shape[-2] != S:      # [B, 1, S, D] — slice to last S positions if longer
                cos = cos[:, :, -S:, :]
                sin = sin[:, :, -S:, :]
            return cos, sin

        pos_start = int(cache_position[0].item()) if cache_position is not None else 0
        cos_full, sin_full = self._get_rotary(pos_start + S, device)
        cos = cos_full[pos_start:].unsqueeze(0).unsqueeze(0)   # [1, 1, S, D]
        sin = sin_full[pos_start:].unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _get_rotary(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (
            torch.arange(0, half, device=device, dtype=torch.float32) / half
        ))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    # ------------------------------------------------------------------
    def _make_store(self, B: int, dtype: torch.dtype, device: torch.device):
        kwargs = dict(
            compute_device=device,
            policy=self._policy,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            chunk_size=self.chunk_size,
            groups_per_chunk=self._cfg.groups_per_chunk,
            batch_size=B,
            dtype=dtype,
        )
        if self.cache_location == "vram":
            return VramKVCacheStore(**kwargs)
        return RamKVCacheStore(pin_memory=False, num_layers=self.num_layers, **kwargs)

    def _get_router(
        self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device
    ) -> ChunkRouter:
        holder = pkv if pkv is not None else self
        router = getattr(holder, "_kv_router", None)
        if router is None:
            store = self._make_store(B, dtype, device)
            router = ChunkRouter(self._cfg, store)
            setattr(holder, "_kv_router", router)
        return router
