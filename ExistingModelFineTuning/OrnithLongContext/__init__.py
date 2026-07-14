"""Router-backed sparse attention surgery for Ornith-1.0-9B (Qwen3.5-hybrid)."""

from .ornith_routed_attention import (
    OrnithRoutedAttention,
    replace_ornith_attention_with_router,
    restore_ornith_attention,
)

__all__ = [
    "OrnithRoutedAttention",
    "replace_ornith_attention_with_router",
    "restore_ornith_attention",
]
