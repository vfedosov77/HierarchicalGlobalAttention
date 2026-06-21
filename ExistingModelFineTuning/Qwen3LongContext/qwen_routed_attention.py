"""Router-backed sparse attention for pretrained Qwen3 (MoE) models.

This swaps each ``Qwen3MoeAttention`` for a drop-in module that keeps the *original* (here
FP8-quantized) Q/K/V/O projections and q/k RMSNorms **by reference** and only changes *what
each query attends to*.  All KV-cache bookkeeping is delegated to the ``KvRouter`` package
(``ChunkRouter`` + ``KVCacheStore``); the cache can live in VRAM (``VramKVCacheStore``) or in
host RAM (``RamKVCacheStore``, only routed chunks pulled to VRAM) without touching this file.

The attention is exactly the 40M ``HierarchicalGlobalAttentionRouted`` design: routing only
*selects* which previous chunks/groups each query attends to (scoring queries against the
resident chunk-/group-**key** summaries), then attention is computed over the **real token
K/V** of the selected items via ``RoutedKV.attend(use_summaries=False)``.  Group **value**
summaries are never attended, so a pretrained model that never learned the summaries is not
corrupted.  ``ChunkRouter.route_query_block`` auto-selects the fast chunk-parallel
``vectorized`` path for a fresh multi-chunk prefill and the incremental path for single-token
decode, seeding the store so decode continues seamlessly — no bespoke prefill loop and no
``ExactChunkRouter`` needed.

Two routing granularities, both selecting whole chunks at the first level and then exposing
*real tokens* (never summaries) at the second:

* **group-level routing** (``group_size`` < ``chunk_size``): the routed chunks are opened at
  group granularity — the top-``topk_groups`` groups of the selected chunks become exact token
  KV.  Finer, cheaper recall.
* **whole-chunk routing** (``group_size == chunk_size`` ⇒ one group per chunk): opening a
  "group" exposes the whole selected chunk's tokens.  This reproduces the old exact router's
  pattern (full token KV of every selected chunk) with no special code path.

MInference's static **A-shape** pattern (first ``n_init`` sink tokens + last ``n_local`` local
tokens) maps onto ``keep_first`` / ``keep_last`` chunks (chunk_size 64): n_init=128 → keep_first=2,
n_local=128 → keep_last=2.  The top-k routed middle is an additional recall path on top of A-shape.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

# --- KvRouter package import (works from repo root or ExistingModelFineTuning/) -------------
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

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


# =================================================================================================
# Drop-in attention module
# =================================================================================================
class QwenRoutedAttention(nn.Module):
    """Replacement for ``Qwen3MoeAttention`` that routes through a shared ``ChunkRouter``.

    Adds no parameters: it reuses ``orig``'s projections and norms by reference (FP8-safe).
    Routing selects chunks/groups; attention runs over their real tokens (``use_summaries=False``).
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
        vram_cache_reserve_gb: float = 1.5,
    ) -> None:
        super().__init__()
        self.orig = orig            # keeps original projections/norms as a child (shared weights)
        self.layer_idx = int(getattr(orig, "layer_idx", 0))

        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.num_heads))
        self.num_layers = int(getattr(config, "num_hidden_layers", 0))
        self.chunk_size = chunk_size
        self.cache_location = cache_location
        self.vram_cache_chunks = vram_cache_chunks
        self.vram_cache_reserve_gb = vram_cache_reserve_gb

        self._cfg = RouterConfig(
            nhead=self.num_heads, kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups,
            theta=float(getattr(config, "rope_theta", 1_000_000.0)),
        )
        # Sinks resident at token granularity (the routed attention reads real tokens only).
        self._policy = ChunkPlacementPolicy(
            keep_last=keep_last, keep_first=keep_first, first_token_level=True,
        )

    # ------------------------------------------------------------------
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

        q = o.q_norm(o.q_proj(hidden_states).view(B, S, H, Dh)).transpose(1, 2)    # [B,H,S,Dh]
        k_raw = o.k_norm(o.k_proj(hidden_states).view(B, S, KVH, Dh)).transpose(1, 2)  # pre-rope
        v = o.v_proj(hidden_states).view(B, S, KVH, Dh).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k_raw, cos, sin)   # HF rotate_half convention

        # The decoder forwards ``position_ids`` (not ``cache_position``) to attention; both
        # encode the absolute start of this block, which is what the streaming router needs.
        start_pos = 0
        if cache_position is not None and cache_position.numel() > 0:
            start_pos = int(cache_position.reshape(-1)[0].item())
        else:
            pos_ids = kw.get("position_ids", None)
            if pos_ids is not None and pos_ids.numel() > 0:
                start_pos = int(pos_ids.reshape(-1)[0].item())

        router = self._get_router(past_key_values, B, hidden_states.dtype, hidden_states.device)
        if self.layer_idx == 0 and start_pos == 0:
            router.reset()

        # cos/sin in [1, 1, S, Dh] for the vectorized chunk-parallel prefill path.
        cos_r = cos.reshape(1, 1, S, Dh).to(hidden_states.dtype)
        sin_r = sin.reshape(1, 1, S, Dh).to(hidden_states.dtype)
        segments = router.route_query_block(
            self.layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            # use_summaries=False: score & attend real tokens only (group V summaries unused).
            out_heads[:, :, lo:hi] = routed.attend(q_rope[:, :, lo:hi], use_summaries=False)

        out = o.o_proj(out_heads.transpose(1, 2).reshape(B, S, H * Dh))
        return out, None

    # ------------------------------------------------------------------
    def _make_store(self, B: int, dtype: torch.dtype, device: torch.device):
        kwargs = dict(
            compute_device=device, policy=self._policy, kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, chunk_size=self.chunk_size,
            groups_per_chunk=self._cfg.groups_per_chunk, batch_size=B, dtype=dtype,
        )
        if self.cache_location == "vram":
            return VramKVCacheStore(**kwargs)
        # RAM tier: cold KV record in host memory, only routed chunks pulled to VRAM each step.
        # pin_memory=False keeps the (multi-GB at 32K) record off the limited pinned pool and
        # makes H2D/D2H copies synchronous (no async-lifetime hazard on the streaming path).
        # A bounded LRU VRAM cache keeps recurring chunks resident so consecutive decode steps
        # only copy newly-required chunks.  It is auto-sized to the free VRAM (num_layers +
        # reserve) so a long-context prefill cannot OOM the bank on a memory-tight card.
        return RamKVCacheStore(
            pin_memory=False, vram_cache_chunks=self.vram_cache_chunks,
            num_layers=self.num_layers, vram_cache_reserve_gb=self.vram_cache_reserve_gb, **kwargs,
        )

    def _get_router(self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        """One router/store shared by all layers, attached to the ``past_key_values`` object."""
        holder = pkv if pkv is not None else self  # fall back to per-module if no cache passed
        router = getattr(holder, "_kv_router", None)
        if router is None:
            store = self._make_store(B, dtype, device)
            router = ChunkRouter(self._cfg, store)
            setattr(holder, "_kv_router", router)
        return router


# =================================================================================================
# Model surgery
# =================================================================================================
def _iter_attention_layers(model: nn.Module):
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("model.model.layers not found")
    for layer in layers:
        if hasattr(layer, "self_attn"):
            yield layer


def restore_original_attention(model: nn.Module) -> int:
    """Undo a previous replacement, putting the original ``self_attn`` modules back."""
    n = 0
    for layer in _iter_attention_layers(model):
        a = layer.self_attn
        if isinstance(a, QwenRoutedAttention):
            layer.self_attn = a.orig
            n += 1
    return n


def replace_qwen_attention_with_router(
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
    vram_cache_reserve_gb: float = 1.5,
) -> int:
    """Replace every ``self_attn`` with a ``QwenRoutedAttention`` (idempotent: unwraps first).

    ``group_size`` selects the routing granularity: ``< chunk_size`` for group-level routing,
    ``== chunk_size`` for whole-chunk routing (one group per chunk).
    """
    restore_original_attention(model)
    config = model.config
    count = 0
    for layer in _iter_attention_layers(model):
        orig = layer.self_attn
        layer.self_attn = QwenRoutedAttention(
            orig, config, chunk_size=chunk_size, group_size=group_size,
            keep_first=keep_first, keep_last=keep_last, topk_chunks=topk_chunks,
            topk_groups=topk_groups, cache_location=cache_location,
            vram_cache_chunks=vram_cache_chunks, vram_cache_reserve_gb=vram_cache_reserve_gb,
        )
        count += 1
    return count
