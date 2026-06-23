"""Abstract base class for all tiered KV-cache store implementations.

Storage model (per attention layer, GQA / kv-head granularity)
--------------------------------------------------------------
For every *closed* chunk ``i`` we keep three kinds of artefact, at three temperatures:

================  ==========================  ============  ===================================
kind              shape (per chunk)           tier          why
================  ==========================  ============  ===================================
``chunk_k``       ``[B, KVH, Dh]``            HOT (VRAM)    routing scan table; scanned every
                                                            step, tiny (1 vec / 64 tokens)
``group_k/v``     ``[B, KVH, M, Dh]``         WARM (RAM)    fetched only for routed chunks
``token_k/v``     ``[B, KVH, C, Dh]``         COLD (RAM)    fetched only for *opened* groups
================  ==========================  ============  ===================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from .chunk_placement_policy import ChunkPlacementPolicy


class KVCacheStore(ABC):
    """Abstract tiered store.  One instance serves all layers of a model."""

    def __init__(self, *, compute_device: torch.device, policy: ChunkPlacementPolicy) -> None:
        self.compute_device = compute_device
        self.policy = policy

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def num_closed_chunks(self, layer: int) -> int: ...

    # -- routing table (HOT) ----------------------------------------------
    @abstractmethod
    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        """``[B, KVH, n_closed, Dh]`` on the compute device (detached), or None."""

    # -- ingest ------------------------------------------------------------
    @abstractmethod
    def append_closed_chunk(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, Dh]
        group_k: torch.Tensor,   # [B, KVH, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, C, Dh]
    ) -> None:
        """Store a freshly-closed chunk.  Tensors may carry grad."""

    # -- fetch into VRAM ---------------------------------------------------
    @abstractmethod
    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group K/V summaries for ``chunk_idx[B, H, *]`` → ``[B, H, *, M, Dh]`` on device."""

    @abstractmethod
    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token K/V for matching ``(chunk_idx, group_idx)[B, H, *]`` → ``[B, H, *, gs, Dh]``."""

    # -- always-resident windows (grad-preserving) ------------------------
    @abstractmethod
    def hot_group_summaries(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Live group K/V for chunks ``[lo, hi)`` → ``[B, KVH, hi-lo, M, Dh]`` (grad kept)."""

    @abstractmethod
    def hot_tokens(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Live token K/V for chunks ``[lo, hi)`` → ``[B, KVH, hi-lo, C, Dh]`` (grad kept)."""

    # -- IO hint (overridden by NVMe backends) ----------------------------
    def prefetch(self, layer: int, chunk_idx: torch.Tensor) -> None:  # noqa: B027 - optional hook
        """Best-effort async hint that ``chunk_idx`` will be gathered soon.  No-op for RAM."""
