"""All-VRAM KV-cache store: the cold group/token record lives on the compute device too.

This is the cache used for **training** (and for ``cache_location="vram"`` generation):
nothing is offloaded to host RAM, so gathers are a same-device index with no PCIe
copy — the fastest option when the whole context fits in VRAM.
"""
from __future__ import annotations

import torch

from .chunk_placement_policy import ChunkPlacementPolicy
from .ram_kv_cache_store import RamKVCacheStore


class VramKVCacheStore(RamKVCacheStore):
    """All-VRAM tier: the cold group/token record lives on the compute device too.

    This is the cache used for **training** (and for ``cache_location="vram"`` generation):
    nothing is offloaded to host RAM, so gathers are a same-device index with no PCIe
    copy — the fastest option when the whole context fits in VRAM.  It is a thin
    specialisation of :class:`RamKVCacheStore` with ``storage_device == compute_device``
    and no pinning.
    """

    def __init__(
        self,
        *,
        compute_device: torch.device,
        policy: ChunkPlacementPolicy,
        kv_heads: int,
        head_dim: int,
        chunk_size: int,
        groups_per_chunk: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        initial_capacity: int = 64,
    ) -> None:
        super().__init__(
            compute_device=compute_device,
            policy=policy,
            kv_heads=kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            groups_per_chunk=groups_per_chunk,
            batch_size=batch_size,
            dtype=dtype,
            pin_memory=False,
            initial_capacity=initial_capacity,
            storage_device=compute_device,
        )
