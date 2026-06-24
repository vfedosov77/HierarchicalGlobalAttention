"""Internal per-layer backing store dataclass for the tiered KV-cache."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class _LayerStore:
    """Backing tensors for one layer.  ``None`` until the first chunk is appended."""

    # HOT — resident routing table, detached. [B, KVH, cap, Dh], filled to n_closed.
    chunk_k: Optional[torch.Tensor] = None
    n_closed: int = 0

    # COLD/WARM — complete CPU record of every closed chunk (detached, pinned).
    cpu_group_k: Optional[torch.Tensor] = None   # [B, KVH, cap, M, Dh]
    cpu_group_v: Optional[torch.Tensor] = None
    cpu_token_k: Optional[torch.Tensor] = None   # [B, KVH, cap, C, Dh]
    cpu_token_v: Optional[torch.Tensor] = None

    # Live (grad-carrying) hot windows, keyed by absolute chunk id.
    live_group_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_group_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_token_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_token_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]

    # Segment-TBPTT boundary snapshot (see RamKVCacheStore.commit / rewind).  ``n_committed`` is
    # the closed-chunk count at the last segment boundary; the ``committed_live_*`` dicts hold the
    # detached live-window tensors as of that boundary so a gradient-checkpointing recompute can be
    # rewound to a byte-identical prefix.  Default 0 / empty ⇒ no segment boundary taken yet.
    n_committed: int = 0
    committed_live_group_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    committed_live_group_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    committed_live_token_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    committed_live_token_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.live_group_k = {}
        self.live_group_v = {}
        self.live_token_k = {}
        self.live_token_v = {}
        self.committed_live_group_k = {}
        self.committed_live_group_v = {}
        self.committed_live_token_k = {}
        self.committed_live_token_v = {}
