"""Hierarchical sparse attention for Qwen3 with pluggable long-context strategies.

Extends the routed-attention design (:mod:`qwen_routed_attention`) with:

* automatic cache sizing for contexts well beyond 32K (up to the model's native 262K on
  Qwen3-30B-A3B-Instruct-2507-FP8);
* a :class:`~long_context_strategy.LongContextStrategy` hook so RoPE / attend behaviour
  can switch between native model RoPE and training-free DCA extrapolation without touching
  the KvRouter.

For Qwen3-30B-Instruct-2507 the default strategy is **native** (standard 262K RoPE applied
by ``Qwen3MoeRotaryEmbedding`` in the decoder — no DCA required).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_EFT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EFT)
for _p in (_ROOT, _EFT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from KvRouter import ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore  # type: ignore
    from KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore,
    )
    from ExistingModelFineTuning.KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore

from ExistingModelFineTuning.Qwen3LongContext.long_context_strategy import (
    LongContextSettings,
    LongContextStrategy,
    make_strategy,
    resolve_long_context_settings,
)
from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    QwenRoutedAttention,
    _iter_attention_layers,
)


class QwenHierarchicalAttention(nn.Module):
    """Routed Qwen3 attention with long-context strategy support.

    Identical routing / KV-store behaviour to :class:`QwenRoutedAttention`; differs only
    in how Q/K are RoPE-encoded and how the final attend step is executed.
    """

    def __init__(
        self,
        orig: nn.Module,
        config: Any,
        *,
        chunk_size: int = 64,
        group_size: int = 16,
        keep_first: int = 2,
        keep_last: int = 2,
        topk_chunks: int = 8,
        topk_groups: int = 32,
        cache_location: str = "vram",
        vram_cache_chunks: int = 256,
        vram_summary_chunks: Optional[int] = None,
        vram_cache_reserve_gb: float = 1.5,
        long_context: Optional[LongContextSettings] = None,
        force_dca: bool = False,
        target_context: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.orig = orig
        self.layer_idx = int(getattr(orig, "layer_idx", 0))

        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.num_heads))
        self.num_layers = int(getattr(config, "num_hidden_layers", 0))
        self.chunk_size = chunk_size
        self.cache_location = cache_location
        self.vram_cache_chunks = vram_cache_chunks
        self.vram_cache_reserve_gb = vram_cache_reserve_gb

        lc = long_context or resolve_long_context_settings(
            config, chunk_size=chunk_size, target_context=target_context, force_dca=force_dca,
            vram_summary_chunks=vram_summary_chunks,
        )
        self.long_context = lc
        self.vram_summary_chunks = lc.vram_summary_chunks
        self._strategy: LongContextStrategy = make_strategy(lc, self.head_dim)

        self._cfg = RouterConfig(
            nhead=self.num_heads, kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups,
            theta=float(getattr(config, "rope_theta", 1_000_000.0)),
        )
        self._policy = ChunkPlacementPolicy(
            keep_last=keep_last, keep_first=keep_first, first_token_level=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        cache_position: Optional[torch.Tensor] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        o = self.orig
        B, S, _ = hidden_states.shape
        H, KVH, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        q_raw = o.q_norm(o.q_proj(hidden_states).view(B, S, H, Dh)).transpose(1, 2)
        k_raw = o.k_norm(o.k_proj(hidden_states).view(B, S, KVH, Dh)).transpose(1, 2)
        v = o.v_proj(hidden_states).view(B, S, KVH, Dh).transpose(1, 2)

        cos, sin = position_embeddings

        start_pos = 0
        if cache_position is not None and cache_position.numel() > 0:
            start_pos = int(cache_position.reshape(-1)[0].item())
        else:
            pos_ids = kw.get("position_ids", None)
            if pos_ids is not None and pos_ids.numel() > 0:
                start_pos = int(pos_ids.reshape(-1)[0].item())

        abs_positions = torch.arange(start_pos, start_pos + S, device=hidden_states.device)
        abs_positions = abs_positions.view(1, S).expand(B, S)

        q_rope, k_rope = self._strategy.prepare_qk(q_raw, k_raw, abs_positions, cos, sin)

        router = self._get_router(past_key_values, B, hidden_states.dtype, hidden_states.device)
        if self.layer_idx == 0 and start_pos == 0:
            router.reset()

        cos_r = cos.reshape(1, 1, S, Dh).to(hidden_states.dtype)
        sin_r = sin.reshape(1, 1, S, Dh).to(hidden_states.dtype)
        segments = router.route_query_block(
            self.layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            out_heads[:, :, lo:hi] = self._strategy.attend(
                q_rope[:, :, lo:hi], routed,
                query_abs_start=start_pos + lo,
                model_cos=cos, model_sin=sin,
                router=router, layer_idx=self.layer_idx,
                use_summaries=False,
            )

        out = o.o_proj(out_heads.transpose(1, 2).reshape(B, S, H * Dh))
        return out, None

    def _make_store(self, B: int, dtype: torch.dtype, device: torch.device):
        kwargs = dict(
            compute_device=device, policy=self._policy, kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, chunk_size=self.chunk_size,
            groups_per_chunk=self._cfg.groups_per_chunk, batch_size=B, dtype=dtype,
        )
        if self.cache_location == "vram":
            return VramKVCacheStore(**kwargs)
        return RamKVCacheStore(
            pin_memory=False, vram_cache_chunks=self.vram_cache_chunks,
            vram_summary_chunks=self.vram_summary_chunks,
            num_layers=self.num_layers, vram_cache_reserve_gb=self.vram_cache_reserve_gb, **kwargs,
        )

    def _get_router(self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        holder = pkv if pkv is not None else self
        router = getattr(holder, "_kv_router", None)
        if router is None:
            store = self._make_store(B, dtype, device)
            router = ChunkRouter(self._cfg, store)
            setattr(holder, "_kv_router", router)
        return router


def _is_replacement_attn(a: nn.Module) -> bool:
    return isinstance(a, (QwenRoutedAttention, QwenHierarchicalAttention))


def restore_original_attention(model: nn.Module) -> int:
    n = 0
    for layer in _iter_attention_layers(model):
        a = layer.self_attn
        if _is_replacement_attn(a):
            layer.self_attn = a.orig
            n += 1
    return n


def replace_qwen_attention_with_hierarchical(
    model: nn.Module,
    *,
    keep_first: int = 2,
    keep_last: int = 2,
    topk_chunks: int = 8,
    topk_groups: int = 32,
    chunk_size: int = 64,
    group_size: int = 16,
    cache_location: str = "vram",
    vram_cache_chunks: int = 256,
    vram_summary_chunks: Optional[int] = None,
    vram_cache_reserve_gb: float = 1.5,
    force_dca: bool = False,
    target_context: Optional[int] = None,
) -> int:
    """Replace every ``self_attn`` with :class:`QwenHierarchicalAttention` (idempotent)."""
    restore_original_attention(model)
    config = model.config
    lc = resolve_long_context_settings(
        config, chunk_size=chunk_size, target_context=target_context, force_dca=force_dca,
        vram_summary_chunks=vram_summary_chunks,
    )
    count = 0
    for layer in _iter_attention_layers(model):
        orig = layer.self_attn
        layer.self_attn = QwenHierarchicalAttention(
            orig, config, chunk_size=chunk_size, group_size=group_size,
            keep_first=keep_first, keep_last=keep_last, topk_chunks=topk_chunks,
            topk_groups=topk_groups, cache_location=cache_location,
            vram_cache_chunks=vram_cache_chunks, long_context=lc,
            vram_cache_reserve_gb=vram_cache_reserve_gb,
        )
        count += 1
    return count