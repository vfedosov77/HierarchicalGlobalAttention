"""Long-context LoRA fine-tuning of a Qwen3 model with HGA routed attention.

Stage 1 (this commit): a *training* drop-in that replaces every Qwen3 ``self_attn`` with a
router-backed sparse attention (NO SDPA fallback — the router is always used, so the routed
path is the one that gets fine-tuned) and exposes a tiny controller that owns the per-forward
router/store lifecycle.  LoRA (attn + mlp) is the only trainable part.
"""

from .routed_attention import (
    RouterController,
    Qwen3RoutedTrainAttention,
    patch_qwen3_with_router,
    unpatch_qwen3,
    vram_grad_store_factory,
    hybrid_grad_store_factory,
    vram_hybrid_store_factory,
    make_hybrid_store_factory,
)
from .streaming import streaming_forward

__all__ = [
    "RouterController",
    "Qwen3RoutedTrainAttention",
    "patch_qwen3_with_router",
    "unpatch_qwen3",
    "vram_grad_store_factory",
    "hybrid_grad_store_factory",
    "vram_hybrid_store_factory",
    "make_hybrid_store_factory",
    "streaming_forward",
]
