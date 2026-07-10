"""Router-backed sparse attention for pretrained **Ornith-1.0-9B** (Qwen3.5-hybrid).

Ornith is a Qwen3.5 hybrid: 3 of every 4 layers are ``linear_attention``
(``Qwen3_5GatedDeltaNet`` — constant recurrent state, already O(1) in context) and every
4th is ``full_attention`` (``Qwen3_5Attention`` — the only O(S) KV-cache cost).  HGA is a
drop-in exact-token router; it applies **only** to the ``full_attention`` layers (indices
3, 7, …, 31 for the 32-layer text stack).  Linear layers are left untouched — routing is
technically undefined for a recurrent delta-net and they carry no context-growing KV.

This mirrors :class:`~ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention.QwenRoutedAttention`
but reproduces three Qwen3.5-specific details of ``Qwen3_5Attention`` **1:1** (see
[docs/ORNITH_HGA.md](../../docs/ORNITH_HGA.md) §4):

1. **output gate** — ``q_proj`` emits ``num_heads * head_dim * 2``; the second half is a
   per-element gate applied as ``attn_output * sigmoid(gate)`` *before* ``o_proj``;
2. **partial RoPE** — only the first ``rotary_dim = round(head_dim * partial_rotary_factor)``
   dims are rotated (64 of 256), tail passthrough.  The exact-token path stays numerically
   exact regardless (it reuses the model's own ``apply_rotary_pos_emb`` by reference); the
   partial factor is threaded into ``RouterConfig(rotary_dim=…)`` only so the summary
   *selection* geometry matches the queries;
3. **multimodal wrapper** — attention params live in ``config.get_text_config()`` and the
   text layers hang under ``model.model.language_model.layers``.

As with the Qwen3 path this adds **no parameters**: it keeps ``orig``'s projections and
q/k norms by reference.  Routing selects chunks/groups; attention runs over their real
tokens (``use_summaries=False``).
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

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
    from KvRouter import ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore, FsKVCacheStore  # type: ignore
    from KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore, FsKVCacheStore,
    )
    from ExistingModelFineTuning.KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore

# The model's *own* partial-RoPE application — reused by reference so the exact-token path
# is bit-identical to stock (rotates the first ``cos.shape[-1]`` dims, passes the tail).
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


# =================================================================================================
# config helpers
# =================================================================================================
def _rope_param(config: Any, name: str, default: float) -> float:
    """Read a RoPE parameter from a Qwen3.5 text config (attribute or ``rope_parameters`` dict)."""
    val = getattr(config, name, None)
    if val is None:
        params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(params, dict):
            val = params.get(name)
    return default if val is None else float(val)


class _RouterHolder:
    """Stable owner of the shared ``_kv_router`` (one per surgery, survives all forward calls)."""

    _kv_router = None


# =================================================================================================
# Drop-in attention module
# =================================================================================================
class OrnithRoutedAttention(nn.Module):
    """Replacement for ``Qwen3_5Attention`` (full-attention layers only) routed via a shared store.

    Adds no parameters: reuses ``orig``'s q/k/v/o projections and q/k norms by reference.
    """

    def __init__(
        self,
        orig: nn.Module,
        config: Any,                 # the *text* config (config.get_text_config())
        *,
        first_attn_layer_idx: int,
        num_wrapped_layers: int,
        shared_router_holder: Any = None,
        chunk_size: int = 64,
        group_size: int = 16,
        keep_first: int = 2,
        keep_last: int = 8,
        topk_chunks: int = 16,
        topk_groups: int = 32,
        cache_location: str = "ram",
        vram_cache_chunks: int = 256,
        vram_summary_chunks: int = 4096,
        vram_cache_reserve_gb: float = 1.5,
        ram_budget_gb: float = 12.0,
        fs_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.orig = orig            # keeps original projections/norms as a child (shared weights)
        self.layer_idx = int(getattr(orig, "layer_idx", 0))
        self.first_attn_layer_idx = int(first_attn_layer_idx)
        # One store shared by ALL wrapped layers *and* across all prefill blocks / decode steps.
        # Binding to a holder owned by the surgery (not to ``past_key_values``, which is ``None``
        # on a manually-driven first block) avoids splitting the KV history between blocks.
        # NB: never store ``self`` here — nn.Module would register it as its own child (recursion).
        self._holder = shared_router_holder if shared_router_holder is not None else _RouterHolder()

        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.num_heads))
        # Number of *wrapped* (full-attention) layers — the store sizes its VRAM budget by this,
        # not by the full 32-layer stack (24 of which are linear and never wrapped).
        self.num_layers = int(num_wrapped_layers)
        self.chunk_size = chunk_size
        self.cache_location = cache_location
        self.vram_cache_chunks = vram_cache_chunks
        self.vram_summary_chunks = vram_summary_chunks
        self.vram_cache_reserve_gb = vram_cache_reserve_gb
        self.ram_budget_gb = ram_budget_gb
        self.fs_cache_dir = fs_cache_dir

        # Partial RoPE: rotate only the first ``rotary_dim`` dims (Qwen3.5 partial_rotary_factor).
        partial = _rope_param(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(round(self.head_dim * partial))
        theta = _rope_param(config, "rope_theta", 1_000_000.0)
        self._cfg = RouterConfig(
            nhead=self.num_heads, kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups, theta=theta,
            rotary_dim=None if rotary_dim >= self.head_dim else rotary_dim,
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

        # (1) q_proj emits query+gate interleaved as [.., H, 2*Dh]; split, q_norm the query.
        qg = o.q_proj(hidden_states).view(B, S, H, Dh * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)           # each [B,S,H,Dh]
        gate = gate.reshape(B, S, H * Dh)
        q = o.q_norm(query).transpose(1, 2)                # [B,H,S,Dh]
        k_raw = o.k_norm(o.k_proj(hidden_states).view(B, S, KVH, Dh)).transpose(1, 2)  # pre-rope
        v = o.v_proj(hidden_states).view(B, S, KVH, Dh).transpose(1, 2)

        # (2) partial RoPE via the model's own kernel (rotates first cos.shape[-1] dims).
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k_raw, cos, sin)

        # Absolute block start for the streaming router (position_ids or cache_position).
        start_pos = 0
        if cache_position is not None and cache_position.numel() > 0:
            start_pos = int(cache_position.reshape(-1)[0].item())
        else:
            pos_ids = kw.get("position_ids", None)
            if pos_ids is not None and pos_ids.numel() > 0:
                start_pos = int(pos_ids.reshape(-1)[0].item())

        router = self._get_router(past_key_values, B, hidden_states.dtype, hidden_states.device)
        # (3) Reset on the *first wrapped* layer (idx 3 for Ornith) — layer 0 is linear/unwrapped
        # so ``layer_idx == 0`` would never fire.
        if self.layer_idx == self.first_attn_layer_idx and start_pos == 0:
            router.reset()

        # cos/sin in [1, 1, S, rotary_dim] for the vectorized chunk-parallel prefill path.
        # RoPE positions are identical across the batch (uniform prefill / B==1 long-context),
        # so row 0 represents the whole block; the router broadcasts it over B.
        rd = cos.shape[-1]
        cos_r = cos.reshape(-1, S, rd)[:1].reshape(1, 1, S, rd).to(hidden_states.dtype)
        sin_r = sin.reshape(-1, S, rd)[:1].reshape(1, 1, S, rd).to(hidden_states.dtype)
        segments = router.route_query_block(
            self.layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            # use_summaries=False: score & attend real tokens only (group V summaries unused).
            out_heads[:, :, lo:hi] = routed.attend(q_rope[:, :, lo:hi], use_summaries=False)

        # (1 cont.) apply the output gate before o_proj.
        out = out_heads.transpose(1, 2).reshape(B, S, H * Dh)
        out = out * torch.sigmoid(gate)
        out = o.o_proj(out)
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
        if self.cache_location == "fs":
            return FsKVCacheStore(
                vram_cache_chunks=self.vram_cache_chunks,
                vram_summary_chunks=self.vram_summary_chunks,
                num_layers=self.num_layers, vram_cache_reserve_gb=self.vram_cache_reserve_gb,
                ram_budget_gb=self.ram_budget_gb, fs_cache_dir=self.fs_cache_dir, **kwargs,
            )
        # RAM tier: cold KV record in host memory, only routed chunks pulled to VRAM each step.
        return RamKVCacheStore(
            pin_memory=False, vram_cache_chunks=self.vram_cache_chunks,
            vram_summary_chunks=self.vram_summary_chunks,
            num_layers=self.num_layers, vram_cache_reserve_gb=self.vram_cache_reserve_gb, **kwargs,
        )

    def _get_router(self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        """One router/store shared by all wrapped layers, held by the surgery's shared holder."""
        holder = self._holder
        router = getattr(holder, "_kv_router", None)
        if router is None:
            store = self._make_store(B, dtype, device)
            router = ChunkRouter(self._cfg, store)
            setattr(holder, "_kv_router", router)
        return router


# =================================================================================================
# Model surgery
# =================================================================================================
def _text_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the text decoder layers of an Ornith / Qwen3.5 model.

    ``Qwen3_5ForConditionalGeneration.model = Qwen3_5Model`` whose ``.language_model`` is the
    ``Qwen3_5TextModel`` holding ``.layers``.  Falls back to plain ``model.model.layers`` /
    ``model.layers`` so a bare text model or a synthetic test model also works.
    """
    for path in (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    ):
        node: Any = model
        ok = True
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                ok = False
                break
        if ok:
            return node
    raise RuntimeError("could not locate text decoder layers (…language_model.layers)")


def _is_full_attention(layer: nn.Module) -> bool:
    lt = getattr(layer, "layer_type", None)
    if lt is not None:
        return lt == "full_attention"
    return hasattr(layer, "self_attn")   # full layers carry self_attn; linear carry linear_attn


def _iter_ornith_attention_layers(model: nn.Module):
    for layer in _text_layers(model):
        if _is_full_attention(layer):
            yield layer


def restore_ornith_attention(model: nn.Module) -> int:
    """Undo a previous replacement, putting the original ``self_attn`` modules back."""
    n = 0
    for layer in _iter_ornith_attention_layers(model):
        a = layer.self_attn
        if isinstance(a, OrnithRoutedAttention):
            layer.self_attn = a.orig
            n += 1
    return n


def replace_ornith_attention_with_router(
    model: nn.Module,
    *,
    keep_first: int = 2,
    keep_last: int = 8,
    topk_chunks: int = 16,
    topk_groups: int = 32,
    chunk_size: int = 64,
    group_size: int = 16,
    cache_location: str = "ram",
    vram_cache_chunks: int = 256,
    vram_summary_chunks: int = 4096,
    vram_cache_reserve_gb: float = 1.5,
    ram_budget_gb: float = 12.0,
    fs_cache_dir: Optional[str] = None,
) -> int:
    """Replace every full-attention ``self_attn`` with an ``OrnithRoutedAttention`` (idempotent).

    Only ``full_attention`` layers are wrapped; ``linear_attention`` (``Qwen3_5GatedDeltaNet``)
    layers are left untouched.  ``group_size < chunk_size`` ⇒ group-level routing;
    ``group_size == chunk_size`` ⇒ whole-chunk routing.  ``cache_location`` selects the cold-KV
    tier: ``"vram"`` / ``"ram"`` (default; routed chunks pulled to VRAM) / ``"fs"`` (NVMe spill).
    """
    restore_ornith_attention(model)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config

    wrapped = list(_iter_ornith_attention_layers(model))
    if not wrapped:
        return 0
    idxs = [int(getattr(l.self_attn, "layer_idx", 0)) for l in wrapped]
    first_attn_layer_idx = min(idxs)
    num_wrapped = len(wrapped)
    holder = _RouterHolder()

    for layer in wrapped:
        orig = layer.self_attn
        layer.self_attn = OrnithRoutedAttention(
            orig, config, first_attn_layer_idx=first_attn_layer_idx,
            num_wrapped_layers=num_wrapped, shared_router_holder=holder,
            chunk_size=chunk_size, group_size=group_size,
            keep_first=keep_first, keep_last=keep_last, topk_chunks=topk_chunks,
            topk_groups=topk_groups, cache_location=cache_location,
            vram_cache_chunks=vram_cache_chunks, vram_summary_chunks=vram_summary_chunks,
            vram_cache_reserve_gb=vram_cache_reserve_gb,
            ram_budget_gb=ram_budget_gb, fs_cache_dir=fs_cache_dir,
        )
    return num_wrapped
