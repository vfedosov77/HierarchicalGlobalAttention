"""GPU-resident summary store.

Routing summaries are tiny: for 32K context there are only ``32768 / 64 = 512``
chunks, and a chunk/group summary is ``M``/``C`` times smaller than the token
K/V of the same chunk.  Production rule:

* ``chunk_k``           — **always** GPU-resident.
* ``group_k`` / ``group_v`` — GPU-resident while ``n_chunks <= gpu_summary_chunks``.

This store grows per-layer summary tables in place and serves
``gather_group_summaries(layer, chunk_ids)`` for the group-routing stage without
ever touching the (much larger) token bank.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..config import HgaConfig


class HgaGpuSummaryStore:
    def __init__(self, cfg: HgaConfig, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        L, KVH, M, Dh = cfg.num_layers, cfg.num_kv_heads, cfg.groups_per_chunk, cfg.head_dim
        # batch=1 first (v0); a batch dim can be added when batch>1 lands.
        self._chunk_k: Dict[int, torch.Tensor] = {}   # [KVH, n, Dh]
        self._group_k: Dict[int, torch.Tensor] = {}   # [KVH, n, M, Dh]
        self._group_v: Dict[int, torch.Tensor] = {}
        self._n: Dict[int, int] = {}

    def reset(self) -> None:
        self._chunk_k.clear()
        self._group_k.clear()
        self._group_v.clear()
        self._n.clear()

    def num_closed_chunks(self, layer: int) -> int:
        return self._n.get(layer, 0)

    @torch.no_grad()
    def append(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [KVH, n_new, Dh]
        group_k: torch.Tensor,   # [KVH, n_new, M, Dh]
        group_v: torch.Tensor,   # [KVH, n_new, M, Dh]
    ) -> None:
        chunk_k = chunk_k.to(self.device, self.dtype)
        group_k = group_k.to(self.device, self.dtype)
        group_v = group_v.to(self.device, self.dtype)
        if layer not in self._chunk_k:
            self._chunk_k[layer] = chunk_k
            self._group_k[layer] = group_k
            self._group_v[layer] = group_v
        else:
            self._chunk_k[layer] = torch.cat([self._chunk_k[layer], chunk_k], dim=1)
            self._group_k[layer] = torch.cat([self._group_k[layer], group_k], dim=1)
            self._group_v[layer] = torch.cat([self._group_v[layer], group_v], dim=1)
        self._n[layer] = self._chunk_k[layer].shape[1]

    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        """``[1, KVH, n, Dh]`` chunk-K table (batch dim added for backend convenience)."""
        t = self._chunk_k.get(layer)
        return None if t is None else t.unsqueeze(0)

    @torch.no_grad()
    def gather_group_summaries(
        self, layer: int, chunk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather group summaries for ``chunk_ids`` ``[B, H, Kc]`` -> ``[B, H, Kc, M, Dh]``.

        ``H`` is the query-head count (GQA-expanded); the underlying tables are
        ``KVH``-headed and expanded on the fly.
        """
        gk = self._group_k[layer]   # [KVH, n, M, Dh]
        gv = self._group_v[layer]
        B, H, Kc = chunk_ids.shape
        rep = self.cfg.gqa_rep
        # map query head -> kv head
        kvh = torch.div(torch.arange(H, device=chunk_ids.device), rep, rounding_mode="floor")
        # index: [KVH, n, M, Dh] -> per (h, Kc)
        idx = chunk_ids.clamp_min(0)
        out_k = gk[kvh][None].expand(B, H, -1, -1, -1)          # [B,H,n,M,Dh]
        out_v = gv[kvh][None].expand(B, H, -1, -1, -1)
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, Kc, gk.shape[-2], gk.shape[-1])
        return out_k.gather(2, gather_idx), out_v.gather(2, gather_idx)
