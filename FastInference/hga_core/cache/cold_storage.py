"""Cold storage adapter (L3) — **stub for v0**.

Per the plan, the filesystem / NVMe tier is implemented only *after* the
RAM+VRAM fast path meets its decode targets.  This adapter defines the contract
the manager will use (async spill / prefetch of cold or inactive sessions) and
defaults to a no-op so the RAM+VRAM path has zero disk involvement.

The mature FS implementation in ``ExistingModelFineTuning/KvRouter`` (paged
per-layer files, writer pool, ``posix_fadvise(DONTNEED)``) is the template to
port here once L1/L2 are fast.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..config import HgaConfig


class HgaColdStorageAdapter:
    """No-op L3 by default.  Enabled only when ``cfg.enable_fs_spill`` is True."""

    def __init__(self, cfg: HgaConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enable_fs_spill)
        self.bytes_spilled = 0
        self.bytes_loaded = 0
        if self.enabled:
            raise NotImplementedError(
                "FS spill is intentionally not implemented in v0. Bring up the "
                "RAM+VRAM fast path first (see FastInference/README.md roadmap)."
            )

    def spill(self, layer: int, chunk_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        return None

    def load(self, layer: int, chunk_id: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return None

    def reset(self) -> None:
        self.bytes_spilled = 0
        self.bytes_loaded = 0
