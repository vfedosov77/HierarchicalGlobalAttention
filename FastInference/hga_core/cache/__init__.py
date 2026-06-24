"""Tiered HGA cache manager.

Tiers (hot -> cold):

* :class:`HgaGpuSummaryStore` — chunk summaries (always GPU) + group summaries
  (GPU while context fits ``gpu_summary_chunks``).  Tiny vs token KV.
* :class:`HgaGpuTokenBank`    — small LRU bank of token K/V chunks on GPU; the
  routed working set is staged here.  Protects sink/local/current selections.
* :class:`HgaPinnedHostKV`    — full closed token K/V in pinned host RAM slabs.
* :class:`HgaColdStorageAdapter` — optional L3 (NVMe/FS) spill; **stub in v0**.
* :class:`HgaCacheManager`    — composes the tiers and exposes the gather API the
  attention backend calls.
"""

from .gpu_summary_store import HgaGpuSummaryStore
from .gpu_token_bank import HgaGpuTokenBank
from .pinned_host_kv import HgaPinnedHostKV
from .cold_storage import HgaColdStorageAdapter
from .manager import HgaCacheManager

__all__ = [
    "HgaGpuSummaryStore",
    "HgaGpuTokenBank",
    "HgaPinnedHostKV",
    "HgaColdStorageAdapter",
    "HgaCacheManager",
]
