"""Chunk placement policy for the tiered KV-cache store.

``ChunkPlacementPolicy`` specifies which chunks are kept live on the compute device and at
what granularity, controlling the always-resident windows (attention sinks + recent context).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ChunkPlacementPolicy:
    """Which chunks are kept live on the compute device, and at what granularity.

    ``keep_last``        number of most-recently-closed chunks always resident (live, grad).
    ``keep_first``       number of leading chunks always resident (attention sinks).
    ``first_token_level``if True the first window keeps full token KV resident; otherwise
                         only its group summaries are resident (cheaper).
    The last window is always token-level (it is the local context).
    """

    keep_last: int = 1
    keep_first: int = 0
    first_token_level: bool = False

    def hot_first_range(self, n_closed: int) -> Tuple[int, int]:
        """``[lo, hi)`` chunk ids kept resident as the *first* window."""
        hi = min(self.keep_first, n_closed)
        return 0, hi

    def hot_last_range(self, n_closed: int) -> Tuple[int, int]:
        """``[lo, hi)`` chunk ids kept resident as the *last* window (no overlap w/ first)."""
        if self.keep_last <= 0:
            return n_closed, n_closed
        # Clamp into [0, n_closed]; never overlap the first window, never invert.
        lo = min(max(self.keep_first, n_closed - self.keep_last), n_closed)
        return lo, n_closed
