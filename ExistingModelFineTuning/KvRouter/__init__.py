"""Tiered KV-cache router for hierarchical chunk-routed attention.

``ChunkRouter`` does the chunk/group selection and attention assembly; ``KVCacheStore``
(``RamKVCacheStore`` for now, NVMe later) owns where the KV lives and how it moves between
VRAM / RAM / disk.  See the module docstrings and ``gen_opt/OFFLOAD_ANALYSIS.md``.
"""

from .cache_store import (
    ChunkPlacementPolicy,
    FsKVCacheStore,
    KVCacheStore,
    RamKVCacheStore,
    VramKVCacheStore,
)
from .chunk_router import ChunkRouter, RouterConfig, RoutedKV

__all__ = [
    "ChunkPlacementPolicy",
    "KVCacheStore",
    "RamKVCacheStore",
    "VramKVCacheStore",
    "FsKVCacheStore",
    "ChunkRouter",
    "RouterConfig",
    "RoutedKV",
]
