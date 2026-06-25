"""Tiered KV-cache router for hierarchical chunk-routed attention.

``ChunkRouter`` does the chunk/group selection and attention assembly; ``KVCacheStore``
(``RamKVCacheStore`` for now, NVMe later) owns where the KV lives and how it moves between
VRAM / RAM / disk.  See the module docstrings and ``gen_opt/OFFLOAD_ANALYSIS.md``.

Cache-store classes live in dedicated modules:

* :mod:`.chunk_placement_policy` — :class:`ChunkPlacementPolicy`
* :mod:`.kv_cache_store`         — :class:`KVCacheStore` (ABC)
* :mod:`.ram_kv_cache_store`     — :class:`RamKVCacheStore`
* :mod:`.vram_kv_cache_store`    — :class:`VramKVCacheStore`
* :mod:`.vram_grad_kv_cache_store` — :class:`VramGradKVCacheStore` (gradient-preserving, for training)
* :mod:`.fs_disk_manager`        — :class:`_FsDiskManager` + I/O helpers
* :mod:`.fs_kv_cache_store`      — :class:`FsKVCacheStore`

Tests live under :mod:`.Tests`.
"""

from .chunk_placement_policy import ChunkPlacementPolicy
from .kv_cache_store import KVCacheStore
from .ram_kv_cache_store import RamKVCacheStore
from .vram_kv_cache_store import VramKVCacheStore
from .vram_grad_kv_cache_store import VramGradKVCacheStore
from .fs_kv_cache_store import FsKVCacheStore
from .chunk_router import ChunkRouter, RouterConfig, RoutedKV

__all__ = [
    "ChunkPlacementPolicy",
    "KVCacheStore",
    "RamKVCacheStore",
    "VramKVCacheStore",
    "VramGradKVCacheStore",
    "FsKVCacheStore",
    "ChunkRouter",
    "RouterConfig",
    "RoutedKV",
]
