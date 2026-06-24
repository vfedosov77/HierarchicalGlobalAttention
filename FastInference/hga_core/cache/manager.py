"""HGA tiered cache manager.

Composes the GPU summary store, GPU token bank, pinned host KV, and (stub) cold
storage, and exposes the two operations the attention backend needs:

* :meth:`append_closed_chunk` — at prefill / chunk-close time, build mixed-RoPE
  summaries from the chunk's token K/V, store the summaries on GPU and the token
  K/V in pinned host RAM (with a GPU token-bank copy of recent chunks).
* :meth:`assemble_routed_kv` — given a :class:`RouteMetadata` and the active
  chunk's live K/V, gather every visible key/value into one ``[B,H,R,Dh]`` tensor
  pair plus a boolean visibility mask, ready for
  :func:`~FastInference.hga_core.kernels.decode_attention.fused_decode_attention`.

Segment order (key axis), matching the reference router:
``[first-window tokens] [last-window tokens] [routed-middle group summaries]
  [opened-group tokens] [active-chunk tokens]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..config import HgaConfig
from ..route_metadata import RouteMetadata
from ..summaries import build_chunk_summaries, build_group_summaries
from .gpu_summary_store import HgaGpuSummaryStore
from .gpu_token_bank import HgaGpuTokenBank
from .pinned_host_kv import HgaPinnedHostKV
from .cold_storage import HgaColdStorageAdapter


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp8_e4m3": torch.bfloat16}[name]


class HgaCacheManager:
    def __init__(self, cfg: HgaConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        kv_dtype = _dtype(cfg.kv_dtype)
        self.summaries = HgaGpuSummaryStore(cfg, device, kv_dtype)
        self.host = HgaPinnedHostKV(cfg, kv_dtype)
        self.token_bank = HgaGpuTokenBank(cfg, self.host, device, kv_dtype)
        self.cold = HgaColdStorageAdapter(cfg)

    def reset(self) -> None:
        self.summaries.reset()
        self.host.reset()
        self.token_bank.reset()
        self.cold.reset()

    def num_closed_chunks(self, layer: int) -> int:
        return self.summaries.num_closed_chunks(layer)

    # ------------------------------------------------------------------
    # prefill / chunk close
    # ------------------------------------------------------------------
    @torch.no_grad()
    def append_closed_chunk(
        self,
        layer: int,
        k_raw: torch.Tensor,    # [KVH, n_new, C, Dh] pre-rope
        k_rope: torch.Tensor,   # [KVH, n_new, C, Dh] rope-applied
        v: torch.Tensor,        # [KVH, n_new, C, Dh]
        start_chunk: int,
    ) -> None:
        cfg = self.cfg
        KVH, n_new, C, Dh = k_raw.shape
        M, gs = cfg.groups_per_chunk, cfg.group_size
        # reshape into groups: [KVH, n_new, M, gs, Dh]
        kr = k_raw.view(KVH, n_new, M, gs, Dh).unsqueeze(0)     # add B=1
        kp = k_rope.view(KVH, n_new, M, gs, Dh).unsqueeze(0)
        vv = v.view(KVH, n_new, M, gs, Dh).unsqueeze(0)
        chunk_start = (torch.arange(n_new, device=k_raw.device) + start_chunk) * C
        group_k, group_v = build_group_summaries(cfg, kr, kp, vv, chunk_start)  # [1,KVH,n_new,M,Dh]
        chunk_k = build_chunk_summaries(cfg, group_k)                            # [1,KVH,n_new,Dh]
        self.summaries.append(layer, chunk_k[0], group_k[0], group_v[0])
        # token K/V (rope-applied keys) to pinned host + recent into bank
        self.host.append(layer, k_rope, v)

    # ------------------------------------------------------------------
    # assembly
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _gather_token_chunks(
        self, layer: int, chunk_ids: torch.Tensor, H: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather token K/V for ``chunk_ids`` (1-D unique) -> ``[H, n, C, Dh]`` (GQA-expanded)."""
        uniq = torch.unique(chunk_ids)
        self.token_bank.ensure_resident(layer, uniq)
        bank_k, bank_v = self.token_bank.bank(layer)         # [KVH, cap, C, Dh]
        slots = self.token_bank.slots_tensor(layer, chunk_ids)
        kvh = bank_k[:, slots]                               # [KVH, n, C, Dh]
        vvh = bank_v[:, slots]
        rep = self.cfg.gqa_rep
        if rep > 1:
            kvh = kvh.repeat_interleave(rep, dim=0)
            vvh = vvh.repeat_interleave(rep, dim=0)
        return kvh, vvh

    @torch.no_grad()
    def assemble_routed_kv(
        self,
        layer: int,
        route: RouteMetadata,
        active_k: torch.Tensor,     # [H, L_act, Dh]  active-chunk keys (rope), GQA-expanded or KVH
        active_v: torch.Tensor,     # [H, L_act, Dh]
        active_causal_offset: int,  # within-chunk position of the FIRST query token
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble ``(k, v, mask)`` for the fused decode kernel.

        Returns ``k,v`` ``[B,H,R,Dh]`` and ``mask`` ``[B,H,L,R]`` (bool).  Batch=1
        in v0.  ``L`` is the number of query tokens (== ``active_k.shape[1]``).
        """
        cfg = self.cfg
        device = self.device
        B = 1
        H = route.num_q_heads
        L = active_k.shape[-2]
        Dh = cfg.head_dim
        seg_k, seg_v, seg_mask = [], [], []

        # ---- deterministic windows (token-level, fully visible) ----
        for lo, hi in ((route.first_lo, route.first_hi), (route.last_lo, route.last_hi)):
            n = hi - lo
            if n <= 0:
                continue
            ids = torch.arange(lo, hi, device=device)
            kk, vv = self._gather_token_chunks(layer, ids, H)     # [H, n, C, Dh]
            kk = kk.reshape(H, n * cfg.chunk_size, Dh)
            vv = vv.reshape(H, n * cfg.chunk_size, Dh)
            seg_k.append(kk); seg_v.append(vv)
            seg_mask.append(torch.ones(B, H, L, n * cfg.chunk_size, dtype=torch.bool, device=device))

        # ---- routed-middle group summaries (optional) ----
        if route.use_summaries and route.num_routed_chunks > 0:
            gk, gv = self.summaries.gather_group_summaries(layer, route.mid_chunk_idx)  # [B,H,Kc,M,Dh]
            Kc, M = gk.shape[2], gk.shape[3]
            gk = gk.reshape(B, H, Kc * M, Dh)
            gv = gv.reshape(B, H, Kc * M, Dh)
            # visibility: expand chunk vis [B,H,L,Kc] over M groups
            vis = route.mid_vis.unsqueeze(-1).expand(B, H, L, Kc, M).reshape(B, H, L, Kc * M)
            seg_k.append(gk[0]); seg_v.append(gv[0]); seg_mask.append(vis)

        # ---- opened-group tokens (head-specific routing) ----
        if route.num_opened_groups > 0:
            parent = route.open_chunk_idx       # [B,H,Kg]
            grp = route.open_group_idx          # [B,H,Kg]
            ok, ov, ovis = self._gather_opened_groups(layer, parent, grp, route.open_vis, H, L)
            seg_k.append(ok[0]); seg_v.append(ov[0]); seg_mask.append(ovis)

        # ---- active-chunk tokens (causal) ----
        if L > 0:
            ak = active_k if active_k.dim() == 3 else active_k
            if ak.shape[0] != H:  # KVH -> H expand
                rep = cfg.gqa_rep
                ak = ak.repeat_interleave(rep, dim=0)
                av = active_v.repeat_interleave(rep, dim=0)
            else:
                av = active_v
            L_act = ak.shape[-2]
            seg_k.append(ak)
            seg_v.append(av)
            # causal: query i (within-chunk pos = offset+i) sees active token j
            # (within-chunk pos = j) iff j <= offset + i.
            qpos = torch.arange(L, device=device) + active_causal_offset
            kpos = torch.arange(L_act, device=device)
            cm = (kpos.view(1, -1) <= qpos.view(-1, 1))
            seg_mask.append(cm.view(1, 1, L, L_act).expand(B, H, L, L_act))

        k = self._cat_segments(seg_k, B, H, Dh)
        v = self._cat_segments(seg_v, B, H, Dh)
        mask = torch.cat(seg_mask, dim=-1) if seg_mask else torch.ones(B, H, L, 0, dtype=torch.bool, device=device)
        return k, v, mask

    @staticmethod
    def _cat_segments(segs, B, H, Dh):
        norm = []
        for s in segs:
            if s.dim() == 3:        # [H, R, Dh]
                s = s.unsqueeze(0)  # [1, H, R, Dh]
            norm.append(s)
        return torch.cat(norm, dim=-2)

    @torch.no_grad()
    def _gather_opened_groups(
        self, layer: int, parent: torch.Tensor, grp: torch.Tensor,
        open_vis: torch.Tensor, H: int, L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather opened groups' token K/V ``[B,H,Kg*gs,Dh]`` + visibility ``[B,H,L,Kg*gs]``.

        Routing is head-specific, so each query head selects its own groups.
        """
        cfg = self.cfg
        device = self.device
        B, _, Kg = parent.shape
        gs, C = cfg.group_size, cfg.chunk_size
        Dh = cfg.head_dim
        rep = cfg.gqa_rep
        # residency for all parent chunks across heads
        uniq = torch.unique(parent)
        self.token_bank.ensure_resident(layer, uniq)
        bank_k, bank_v = self.token_bank.bank(layer)     # [KVH, cap, C, Dh]
        n_chunks = self.num_closed_chunks(layer)
        slot_map = self.token_bank.slot_map(layer, n_chunks)        # [n_chunks] chunk->slot

        # vectorized gather (NO python head/group loop):
        #   out[b,h,kg,t] = bank[h//rep, slot_map[parent], grp*gs + t]
        kvh = (torch.arange(H, device=device) // rep)              # [H]
        slot = slot_map[parent.clamp_min(0)]                        # [B,H,Kg]
        t = torch.arange(gs, device=device)                        # [gs]
        tok = grp.unsqueeze(-1) * gs + t                           # [B,H,Kg,gs]
        kvh_e = kvh.view(1, H, 1, 1).expand(B, H, Kg, gs)
        slot_e = slot.unsqueeze(-1).expand(B, H, Kg, gs)
        out_k = bank_k[kvh_e, slot_e, tok]                         # [B,H,Kg,gs,Dh]
        out_v = bank_v[kvh_e, slot_e, tok]
        out_k = out_k.reshape(B, H, Kg * gs, Dh)
        out_v = out_v.reshape(B, H, Kg * gs, Dh)
        vis = open_vis.unsqueeze(-1).expand(B, H, L, Kg, gs).reshape(B, H, L, Kg * gs)
        return out_k, out_v, vis
