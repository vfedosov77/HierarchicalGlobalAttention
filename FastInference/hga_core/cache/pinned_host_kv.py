"""Pinned host KV store.

The full closed-token K/V history lives in **pinned** host RAM so that staging
into the GPU token bank is an async ``cudaMemcpyAsync`` on a side stream.  Memory
is organised as fixed-size per-chunk slabs (``chunk_size`` tokens) to match the
HGA page layout and keep transfers regular.

v0 keeps one contiguous pinned tensor per layer and grows it in chunk units.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..config import HgaConfig


class HgaPinnedHostKV:
    def __init__(self, cfg: HgaConfig, dtype: torch.dtype):
        self.cfg = cfg
        self.dtype = dtype
        self._k: Dict[int, torch.Tensor] = {}   # [KVH, n, C, Dh] pinned host
        self._v: Dict[int, torch.Tensor] = {}
        self._n: Dict[int, int] = {}

    def reset(self) -> None:
        self._k.clear()
        self._v.clear()
        self._n.clear()

    def num_chunks(self, layer: int) -> int:
        return self._n.get(layer, 0)

    @torch.no_grad()
    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append closed-chunk token K/V ``[KVH, n_new, C, Dh]`` to pinned host RAM."""
        k = self._to_pinned(k)
        v = self._to_pinned(v)
        if layer not in self._k:
            self._k[layer] = k
            self._v[layer] = v
        else:
            self._k[layer] = torch.cat([self._k[layer], k], dim=1)
            self._v[layer] = torch.cat([self._v[layer], v], dim=1)
        self._n[layer] = self._k[layer].shape[1]

    def _to_pinned(self, t: torch.Tensor) -> torch.Tensor:
        t = t.detach().to("cpu", self.dtype).contiguous()
        if not t.is_pinned():
            pinned = torch.empty_like(t, pin_memory=True)
            pinned.copy_(t)
            return pinned
        return t

    @torch.no_grad()
    def fetch_chunks(
        self, layer: int, chunk_ids: torch.Tensor, device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the requested chunks ``[KVH, n, C, Dh]`` to ``device`` (async if pinned).

        ``chunk_ids`` is a 1-D LongTensor of unique chunk ids on the host.
        """
        hk, hv = self._k[layer], self._v[layer]
        ids = chunk_ids.to("cpu")
        sel_k = hk.index_select(1, ids)
        sel_v = hv.index_select(1, ids)
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream(device))
        with ctx:
            dk = sel_k.to(device, non_blocking=True)
            dv = sel_v.to(device, non_blocking=True)
        return dk, dv
