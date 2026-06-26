"""Training-time HGA routed attention for Qwen3 (drop-in, no SDPA fallback).

Design goals (from the big-model PoC brief)
-------------------------------------------
* **Drop-in, close to normal attention.**  Each ``Qwen3Attention`` is replaced by a module
  that reuses the *original* projections and q/k RMSNorms **by reference** and only changes
  *what each query attends to* (the routing).  No new attention parameters; LoRA is added
  separately on the original Linear layers.
* **Router always used.**  There is deliberately **no dense-SDPA fallback** — every forward
  goes through ``ChunkRouter.route_query_block`` so the path that is fine-tuned is the routed
  one (the whole point of the experiment).
* **Gradients via the live block.**  A fresh-sequence prefill attends over the *live*
  (grad-carrying) block K/V assembled by the router, so gradients flow straight back into the
  Q/K/V/O projections (hence into LoRA).  Which chunks keep gradients vs. live on host RAM is
  decided by the *store* (Stage 2's hybrid store); this module is store-agnostic.

The router/store lifecycle is owned by a :class:`RouterController`: one router+store per
forward, shared by all layers (keyed by ``layer_idx``), created right before the model forward.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

# --- KvRouter import (Finetune/ -> Qwen3LongContext/ -> ExistingModelFineTuning/ -> repo root) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_QLC = os.path.dirname(_HERE)
_EFT = os.path.dirname(_QLC)
_ROOT = os.path.dirname(_EFT)
for _p in (_EFT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, ChunkPlacementPolicy,
        VramGradKVCacheStore, HybridGradKVCacheStore,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, ChunkPlacementPolicy,
        VramGradKVCacheStore, HybridGradKVCacheStore,
    )

# Qwen3's own RoPE application (rotate_half convention) — reuse the model's definition.
try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ModuleNotFoundError:  # pragma: no cover
    from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


# =================================================================================================
# Store factory type: (batch, dtype, device, policy, cfg) -> KVCacheStore
# =================================================================================================
StoreFactory = Callable[[int, torch.dtype, torch.device, ChunkPlacementPolicy, RouterConfig], Any]


def vram_grad_store_factory(
    B: int, dtype: torch.dtype, device: torch.device,
    policy: ChunkPlacementPolicy, cfg: RouterConfig,
) -> VramGradKVCacheStore:
    """Default Stage-1 store: everything resident & grad-carrying (no RAM cold tier yet)."""
    return VramGradKVCacheStore(
        compute_device=device, policy=policy, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim,
        chunk_size=cfg.chunk_size, groups_per_chunk=cfg.groups_per_chunk,
        batch_size=B, dtype=dtype,
    )


def hybrid_grad_store_factory(
    B: int, dtype: torch.dtype, device: torch.device,
    policy: ChunkPlacementPolicy, cfg: RouterConfig,
) -> HybridGradKVCacheStore:
    """Stage-2 store: grad hot window (last ``keep_last`` chunks) in VRAM, cold KV on host RAM.

    This is what makes 32K-token LoRA training fit in 16GB: only the model + the hot window are
    grad-resident on the GPU; every older chunk's K/V is detached on CPU and pulled back only for
    the few routed chunks.  A bounded LRU VRAM cache (default 1000 chunks) keeps the *most-useful*
    (most-recently-routed) cold chunks resident so they are not re-copied from RAM every gather.
    Drop-in for :func:`vram_grad_store_factory` (attention unchanged).
    """
    return HybridGradKVCacheStore(
        compute_device=device, policy=policy, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim,
        chunk_size=cfg.chunk_size, groups_per_chunk=cfg.groups_per_chunk,
        batch_size=B, dtype=dtype, ram_device=torch.device("cpu"),
    )


def vram_hybrid_store_factory(
    B: int, dtype: torch.dtype, device: torch.device,
    policy: ChunkPlacementPolicy, cfg: RouterConfig,
) -> HybridGradKVCacheStore:
    """Stage-2 store with a **VRAM-resident cold tier** (no host-RAM offload).

    Same grad hot window as :func:`hybrid_grad_store_factory`, but the detached cold chunks
    (and group summaries) are kept on the *compute device* instead of host RAM — i.e. exactly
    the way the **inference** VRAM cache works (chunks resident in VRAM, no gradients), with the
    last ``keep_last`` chunks additionally carrying gradients for training.  Use this when VRAM is
    plentiful or to avoid PCIe D2H/H2D traffic on cold gathers; use the RAM variant when the full
    KV cache would not fit in VRAM.
    """
    return HybridGradKVCacheStore(
        compute_device=device, policy=policy, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim,
        chunk_size=cfg.chunk_size, groups_per_chunk=cfg.groups_per_chunk,
        batch_size=B, dtype=dtype, ram_device=device,
    )


def make_hybrid_store_factory(*, cold: str = "ram", vram_cache_chunks: int = 1000) -> StoreFactory:
    """Build a hybrid-store factory with a configurable cold tier and VRAM working-set cache.

    * ``cold="ram"`` keeps only the grad hot window + a bounded LRU VRAM cache of the
      ``vram_cache_chunks`` *most-recently-routed* cold chunks on the GPU; all other chunks' K/V
      live on host RAM (pulled transiently only for chunks not already cached).
    * ``cold="vram"`` keeps the whole detached cold tier resident in VRAM (inference-style); the
      LRU cache is then a no-op (chunks are already on the compute device).
    """
    ram_to_device = cold == "vram"

    def _factory(
        B: int, dtype: torch.dtype, device: torch.device,
        policy: ChunkPlacementPolicy, cfg: RouterConfig,
    ) -> HybridGradKVCacheStore:
        return HybridGradKVCacheStore(
            compute_device=device, policy=policy, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim,
            chunk_size=cfg.chunk_size, groups_per_chunk=cfg.groups_per_chunk,
            batch_size=B, dtype=dtype,
            ram_device=device if ram_to_device else torch.device("cpu"),
            vram_cache_chunks=vram_cache_chunks,
        )

    return _factory


# =================================================================================================
# Controller: owns the per-forward router/store, shared by every routed attention layer
# =================================================================================================
class RouterController:
    """Owns the router/store for one forward and hands it to every routed attention module.

    Call :meth:`begin` once per forward (per training step) *before* running the HF model, so the
    layers route over a fresh store.  ``start_pos`` is the absolute position of the block's first
    token (0 for a fresh full-sequence prefill).
    """

    def __init__(self, cfg: RouterConfig, policy: ChunkPlacementPolicy,
                 store_factory: StoreFactory = vram_grad_store_factory) -> None:
        self.cfg = cfg
        self.policy = policy
        self.store_factory = store_factory
        self.router: Optional[ChunkRouter] = None
        self.start_pos: int = 0
        self.store: Any = None

    def begin(self, batch_size: int, dtype: torch.dtype, device: torch.device,
              start_pos: int = 0) -> ChunkRouter:
        self.store = self.store_factory(batch_size, dtype, device, self.policy, self.cfg)
        self.router = ChunkRouter(self.cfg, self.store)
        self.start_pos = int(start_pos)
        return self.router

    def detach_cache(self) -> None:
        """Truncated-BPTT boundary: detach all live cache tensors so the next block's backward
        starts from detached leaves (the cached K/V values persist, their graph history is cut)."""
        if self.router is not None:
            self.router.detach_graph()


# =================================================================================================
# Drop-in training attention
# =================================================================================================
class Qwen3RoutedTrainAttention(nn.Module):
    """Replacement for ``Qwen3Attention`` that always routes (no SDPA), grad-preserving.

    Reuses ``orig``'s ``q_proj/k_proj/v_proj/o_proj`` and ``q_norm/k_norm`` by reference (so LoRA
    on those Linear layers is picked up automatically). ``controller`` provides the shared router.
    """

    def __init__(self, orig: nn.Module, controller: RouterController, config: Any) -> None:
        super().__init__()
        self.orig = orig
        self.controller = controller
        self.layer_idx = int(getattr(orig, "layer_idx", 0))
        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.num_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        cache_position: Optional[torch.Tensor] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, None]:
        o = self.orig
        B, S, _ = hidden_states.shape
        H, KVH, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        # Qwen3 projection + per-head q/k RMSNorm (identical math to the dense model).
        q = o.q_norm(o.q_proj(hidden_states).view(B, S, H, Dh)).transpose(1, 2)        # [B,H,S,Dh]
        k_raw = o.k_norm(o.k_proj(hidden_states).view(B, S, KVH, Dh)).transpose(1, 2)  # pre-rope
        v = o.v_proj(hidden_states).view(B, S, KVH, Dh).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = kw["position_embeddings"]
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k_raw, cos, sin)

        router = self.controller.router
        if router is None:
            raise RuntimeError("RouterController.begin() must be called before the model forward")
        start_pos = self.controller.start_pos

        # cos/sin in [1,1,S,Dh] for the vectorized chunk-parallel prefill path (positions are
        # identical across the batch, so a single row suffices).
        cos_r = cos[:1].unsqueeze(1).to(hidden_states.dtype)
        sin_r = sin[:1].unsqueeze(1).to(hidden_states.dtype)
        segments = router.route_query_block(
            self.layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        # Training: score/attend in the compute dtype (bf16) so the [B,H,L,R] score & prob tensors
        # are NOT retained in fp32 (half the activation, faster matmul).  Inference keeps the fp32
        # path (score_dtype=None) for SDPA-exact equivalence.
        sdtype = q_rope.dtype if self.training else None
        for routed, lo, hi in segments:
            # use_summaries=False: score & attend the REAL token K/V only (the pretrained model
            # never learned group-value summaries), so this stays close to ordinary attention.
            out_heads[:, :, lo:hi] = routed.attend(q_rope[:, :, lo:hi], use_summaries=False,
                                                   score_dtype=sdtype)

        out = o.o_proj(out_heads.transpose(1, 2).reshape(B, S, H * Dh))
        return out, None


# =================================================================================================
# Model surgery
# =================================================================================================
def _iter_decoder_layers(model: nn.Module):
    core = getattr(model, "model", None)
    if core is None:  # PeftModel etc.
        core = getattr(getattr(model, "base_model", model), "model", None)
    core = getattr(core, "model", core)  # unwrap one extra level if needed
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    for layer in layers:
        if hasattr(layer, "self_attn"):
            yield layer


def unpatch_qwen3(model: nn.Module) -> int:
    """Restore the original ``self_attn`` modules."""
    n = 0
    for layer in _iter_decoder_layers(model):
        a = layer.self_attn
        if isinstance(a, Qwen3RoutedTrainAttention):
            layer.self_attn = a.orig
            n += 1
    return n


def patch_qwen3_with_router(
    model: nn.Module,
    *,
    chunk_size: int = 64,
    group_size: int = 16,
    keep_first: int = 2,
    keep_last: int = 8,
    topk_chunks: int = 32,
    topk_groups: int = 64,
    store_factory: StoreFactory = vram_grad_store_factory,
) -> RouterController:
    """Replace every ``self_attn`` with :class:`Qwen3RoutedTrainAttention` (idempotent).

    Returns the shared :class:`RouterController`; call ``controller.begin(...)`` before each
    forward.  ``keep_last`` is the local-context window kept token-level (the hot window that
    will carry gradients once the Stage-2 hybrid store lands); ``keep_first`` are attention
    sinks; ``topk_chunks``/``topk_groups`` are the routed recall budget.
    """
    unpatch_qwen3(model)
    config = model.config
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    cfg = RouterConfig(
        nhead=int(config.num_attention_heads),
        kv_heads=int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        head_dim=head_dim,
        chunk_size=chunk_size, group_size=group_size,
        topk_chunks=topk_chunks, topk_groups=topk_groups,
        theta=float(getattr(config, "rope_theta", 1_000_000.0)),
        current_group_summaries=False,
    )
    policy = ChunkPlacementPolicy(keep_last=keep_last, keep_first=keep_first, first_token_level=True)
    controller = RouterController(cfg, policy, store_factory)

    for layer in _iter_decoder_layers(model):
        layer.self_attn = Qwen3RoutedTrainAttention(layer.self_attn, controller, config)
    return controller
