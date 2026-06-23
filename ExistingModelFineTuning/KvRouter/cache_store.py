"""Backward-compatibility re-export shim for the tiered KV-cache store.

All classes have been split into separate modules:

* :mod:`.chunk_placement_policy` — :class:`ChunkPlacementPolicy`
* :mod:`.layer_store`            — :class:`_LayerStore` (internal)
* :mod:`.kv_cache_store`         — :class:`KVCacheStore` (ABC)
* :mod:`.ram_kv_cache_store`     — :class:`RamKVCacheStore`
* :mod:`.vram_kv_cache_store`    — :class:`VramKVCacheStore`
* :mod:`.fs_disk_manager`        — :class:`_FsDiskManager` + I/O helpers
* :mod:`.fs_kv_cache_store`      — :class:`FsKVCacheStore`, :class:`_FsLayerRecord`

This module re-exports everything so that existing ``from .cache_store import X`` imports
continue to work without modification.
"""

from .chunk_placement_policy import ChunkPlacementPolicy
from .kv_cache_store import KVCacheStore
from .layer_store import _LayerStore
from .ram_kv_cache_store import RamKVCacheStore
from .vram_kv_cache_store import VramKVCacheStore
from .fs_kv_cache_store import FsKVCacheStore

__all__ = [
    "ChunkPlacementPolicy",
    "KVCacheStore",
    "_LayerStore",
    "RamKVCacheStore",
    "VramKVCacheStore",
    "FsKVCacheStore",
]
