"""GPU token bank.

A small, fixed-capacity LRU pool of **token-level** K/V chunks on the GPU.  The
routed working set (opened groups + windows + active chunk) is staged here from
pinned host RAM.  Capacity is ``gpu_token_chunks`` chunks per layer.

Eviction rules (the LRU *protects* the deterministic / current selections):

* sink (``keep_first``) and local (``keep_last``) chunks are pinned (never evicted);
* the active chunk is pinned;
* the rest follow LRU, so temporally-stable routed selections stay resident.

v0 implements residency + LRU + protection.  Async prefetch on a side stream is
wired through :meth:`prefetch` (the manager issues next-token prefetch there).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Set

import torch

from ..config import HgaConfig
from .pinned_host_kv import HgaPinnedHostKV


class HgaGpuTokenBank:
    def __init__(
        self,
        cfg: HgaConfig,
        host: HgaPinnedHostKV,
        device: torch.device,
        dtype: torch.dtype,
        capacity_chunks: Optional[int] = None,
    ):
        self.cfg = cfg
        self.host = host
        self.device = device
        self.dtype = dtype
        self.capacity = capacity_chunks or cfg.gpu_token_chunks
        C, KVH, Dh = cfg.chunk_size, cfg.num_kv_heads, cfg.head_dim
        # contiguous bank staging [KVH, capacity, C, Dh] for K and V, per layer
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}
        self._slot_of: Dict[int, "OrderedDict[int, int]"] = {}   # layer -> {chunk_id: slot} (LRU order)
        self._free: Dict[int, list] = {}
        self._pinned: Dict[int, Set[int]] = {}                   # protected chunk ids
        self.hits = 0
        self.misses = 0
        self.stream = torch.cuda.Stream(device) if (cfg.prefetch_stream and device.type == "cuda") else None

    def reset(self) -> None:
        self._k.clear(); self._v.clear(); self._slot_of.clear()
        self._free.clear(); self._pinned.clear()
        self.hits = self.misses = 0

    def _ensure_layer(self, layer: int) -> None:
        if layer in self._k:
            return
        C, KVH, Dh = self.cfg.chunk_size, self.cfg.num_kv_heads, self.cfg.head_dim
        self._k[layer] = torch.empty((KVH, self.capacity, C, Dh), device=self.device, dtype=self.dtype)
        self._v[layer] = torch.empty((KVH, self.capacity, C, Dh), device=self.device, dtype=self.dtype)
        self._slot_of[layer] = OrderedDict()
        self._free[layer] = list(range(self.capacity))
        self._pinned[layer] = set()

    def set_protected(self, layer: int, chunk_ids: Set[int]) -> None:
        self._ensure_layer(layer)
        self._pinned[layer] = set(chunk_ids)

    def _evict_one(self, layer: int) -> int:
        slot_of = self._slot_of[layer]
        for cid in list(slot_of.keys()):
            if cid not in self._pinned[layer]:
                slot = slot_of.pop(cid)
                return slot
        # everything pinned -> grow not allowed in v0; reuse oldest anyway
        cid, slot = slot_of.popitem(last=False)
        return slot

    @torch.no_grad()
    def ensure_resident(self, layer: int, chunk_ids: torch.Tensor) -> Dict[int, int]:
        """Make ``chunk_ids`` (1-D unique LongTensor) resident; return ``{chunk_id: slot}``."""
        self._ensure_layer(layer)
        slot_of = self._slot_of[layer]
        want = [int(c) for c in chunk_ids.tolist()]
        missing = []
        for cid in want:
            if cid in slot_of:
                slot_of.move_to_end(cid)         # LRU touch
                self.hits += 1
            else:
                missing.append(cid)
                self.misses += 1
        if missing:
            hk, hv = self.host.fetch_chunks(
                layer, torch.tensor(missing, dtype=torch.long), self.device, self.stream,
            )
            if self.stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
            for i, cid in enumerate(missing):
                if self._free[layer]:
                    slot = self._free[layer].pop()
                else:
                    slot = self._evict_one(layer)
                self._k[layer][:, slot] = hk[:, i]
                self._v[layer][:, slot] = hv[:, i]
                slot_of[cid] = slot
                slot_of.move_to_end(cid)
        return {cid: slot_of[cid] for cid in want}

    def slots_tensor(self, layer: int, chunk_ids: torch.Tensor) -> torch.Tensor:
        slot_of = self._slot_of[layer]
        return torch.tensor([slot_of[int(c)] for c in chunk_ids.tolist()],
                            dtype=torch.long, device=self.device)

    def slot_map(self, layer: int, num_chunks: int) -> torch.Tensor:
        """Dense ``[num_chunks]`` LongTensor mapping chunk id -> bank slot.

        Non-resident ids map to 0 (callers must ``ensure_resident`` the ids they
        will index first).  Built once per layer-forward for vectorized gather.
        """
        slot_of = self._slot_of[layer]
        sm = torch.zeros(num_chunks, dtype=torch.long, device=self.device)
        if slot_of:
            ids = torch.tensor(list(slot_of.keys()), dtype=torch.long, device=self.device)
            slots = torch.tensor(list(slot_of.values()), dtype=torch.long, device=self.device)
            sm[ids] = slots
        return sm

    def bank(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._k[layer], self._v[layer]

    @torch.no_grad()
    def prefetch(self, layer: int, chunk_ids: torch.Tensor) -> None:
        """Warm ``chunk_ids`` for the next token on the side stream (best-effort)."""
        try:
            self.ensure_resident(layer, torch.unique(chunk_ids))
        except Exception:
            pass
