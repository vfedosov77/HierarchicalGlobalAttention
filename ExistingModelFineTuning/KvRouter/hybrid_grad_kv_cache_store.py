"""Hybrid gradient store: grad hot window in VRAM, detached cold KV in host RAM.

This is the store that makes **long-context LoRA training cheap**: of the whole sequence's
KV, only the last ``keep_last`` *closed* chunks (the "hot window") are kept resident on the
compute device as **grad-carrying** token K/V; every older chunk is **detached** and parked in
host RAM, pulled back to VRAM (detached) only for the few chunks the router actually routes to.
The active (partial) chunk is held live by the router itself, so gradients reach exactly:

    current chunk  +  the last ``keep_last`` closed chunks   (everything else is stop-grad).

That is precisely the "only the last N chunks have gradients, the rest live on RAM" regime: the
query side (and therefore ``q_proj`` / LoRA) still gets gradient at *every* position, while the
key/value side only updates from the recent window — a truncated-BPTT-through-the-KV-cache.

Tiering (per layer)
-------------------
================  =====================  =========================  ======================
artefact          where                  grad?                      why
================  =====================  =========================  ======================
``chunk_k``       VRAM (compute device)  detached                   routing scan table; tiny
                                                                    (1 vec / chunk), scanned
                                                                    every step
``group_k/v``     host RAM               detached                   only routed chunks gathered
``token_k/v``     host RAM (cold) +      cold detached / hot grad    cold: opened-group recall;
                  VRAM (last keep_last)                              hot: local context (trained)
================  =====================  =========================  ======================

Only ``chunk_k`` (a single key vector per chunk) and the hot window stay *permanently* on the
GPU, so peak VRAM is bounded by the model + the hot window regardless of context length.

Bounded VRAM working-set cache (``vram_cache_chunks``)
-----------------------------------------------------
The cold token K/V live on host RAM, but copying them across PCIe on *every* routed gather is
wasteful: the router keeps selecting the same "useful" chunks step after step.  So a bounded
**LRU VRAM cache** (default 1000 chunks) holds the most-recently-routed cold chunks resident on
the compute device — exactly like the inference :class:`RamKVCacheStore` token bank.  Only
*newly required* chunks are copied from RAM each gather; recurring chunks stay in VRAM.  The bank
grows lazily toward ``vram_cache_chunks`` and evicts least-recently-used chunks once full; if a
single gather needs more distinct chunks than the cache holds it transparently streams straight
from RAM (bounded, never OOM).  ``vram_cache_chunks=0`` disables it (always stream from RAM); for
the VRAM cold tier (``ram_device == compute_device``) it is a no-op (chunks already resident).
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional, Tuple

import torch

from .chunk_placement_policy import ChunkPlacementPolicy
from .kv_cache_store import KVCacheStore


class _Grow:
    """Amortized-doubling append buffer along dim=2 for ``[B, KVH, n, *tail]`` tensors.

    Allocates on the same device/dtype as the first appended tensor; appends are O(1) amortized
    (capacity doubles), and :attr:`data` returns the used ``[B, KVH, n, *tail]`` view.
    """

    def __init__(self) -> None:
        self.buf: Optional[torch.Tensor] = None
        self.n: int = 0

    def append(self, x: torch.Tensor) -> None:  # x: [B, KVH, k, *tail]
        k = x.shape[2]
        if self.buf is None:
            cap = max(8, k)
            self.buf = x.new_zeros(x.shape[0], x.shape[1], cap, *x.shape[3:])
        while self.n + k > self.buf.shape[2]:
            nb = self.buf.new_zeros(self.buf.shape[0], self.buf.shape[1],
                                    self.buf.shape[2] * 2, *self.buf.shape[3:])
            nb[:, :, : self.n] = self.buf[:, :, : self.n]
            self.buf = nb
        self.buf[:, :, self.n : self.n + k] = x
        self.n += k

    @property
    def data(self) -> Optional[torch.Tensor]:
        return None if self.buf is None else self.buf[:, :, : self.n]


class _HybridLayer:
    """Per-layer state (kept in a plain object so it can be stashed/restored wholesale)."""

    def __init__(self) -> None:
        self.chunk_k = _Grow()                       # VRAM, detached  [B,KVH,n,Dh]
        self.group_k = _Grow()                       # RAM,  detached  [B,KVH,n,M,Dh]
        self.group_v = _Grow()                       # RAM,  detached
        self.cold_tk = _Grow()                       # RAM,  detached  [B,KVH,n_evicted,C,Dh]
        self.cold_tv = _Grow()                       # RAM,  detached
        self.hot: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()  # VRAM, GRAD
        self.n_closed = 0


class HybridGradKVCacheStore(KVCacheStore):
    """Grad hot window (last ``keep_last`` chunks, VRAM) + detached cold KV (host RAM).

    Same constructor signature as the other stores (extra kwargs ignored). ``ram_device`` is where
    the cold record lives (default CPU). The hot-window size is ``policy.keep_last``.
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
        ram_device: torch.device = torch.device("cpu"),
        vram_cache_chunks: int = 1000,
        **_ignored,
    ) -> None:
        super().__init__(compute_device=compute_device, policy=policy)
        self.kvh = kv_heads
        self.dh = head_dim
        self.C = chunk_size
        self.M = groups_per_chunk
        self.B = batch_size
        self.dtype = dtype
        self.ram_device = ram_device
        self.keep_last = int(policy.keep_last)
        self._layers: Dict[int, _HybridLayer] = {}

        # -- bounded LRU VRAM cache for the cold token K/V (the "most-useful chunks in VRAM") -----
        # Holds up to ``vram_cache_chunks`` cold chunks resident on the compute device so routed
        # gathers copy only newly-required chunks from RAM.  Disabled (0) or no-op when the cold
        # record already lives on the compute device (VRAM cold tier).
        self.vram_cache_chunks = int(vram_cache_chunks)
        self._bank_init = 64                          # initial bank capacity (chunks); grows to cap
        self._bank_k: Dict[int, torch.Tensor] = {}    # [B,KVH,cap,C,Dh] on compute device
        self._bank_v: Dict[int, torch.Tensor] = {}
        self._id2slot: Dict[int, torch.Tensor] = {}   # cold chunk id -> bank slot (-1 = absent)
        self._lru: Dict[int, "OrderedDict[int, int]"] = {}
        self._cached: Dict[int, set] = {}
        self._free: Dict[int, List[int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # -- internal ----------------------------------------------------------
    def _layer(self, layer: int) -> _HybridLayer:
        st = self._layers.get(layer)
        if st is None:
            st = _HybridLayer()
            self._layers[layer] = st
        return st

    @staticmethod
    def _gather(src: torch.Tensor, chunk_idx: torch.Tensor, rep: int) -> torch.Tensor:
        """Gather ``src[B, KVH, n, *tail]`` by ``chunk_idx[B, H, *]`` → ``[B, H, *, tail]`` (same device)."""
        B = src.shape[0]
        H = chunk_idx.shape[1]
        extra = chunk_idx.ndim - 2
        dev = src.device
        b = torch.arange(B, device=dev).view(B, 1, *([1] * extra))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * extra))
        return src[b, kv, chunk_idx]

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._layers.clear()
        self._bank_k.clear()
        self._bank_v.clear()
        self._id2slot.clear()
        self._lru.clear()
        self._cached.clear()
        self._free.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    # -- bounded LRU VRAM cache for cold token chunks ----------------------
    def _ensure_bank(self, layer: int, n_ev: int) -> None:
        """Create the layer's bank (lazily, small) and grow its id->slot map to ``n_ev`` chunks."""
        dev = self.compute_device
        if layer not in self._bank_k:
            cap0 = min(self.vram_cache_chunks, self._bank_init)
            self._bank_k[layer] = torch.empty(self.B, self.kvh, cap0, self.C, self.dh,
                                              device=dev, dtype=self.dtype)
            self._bank_v[layer] = torch.empty(self.B, self.kvh, cap0, self.C, self.dh,
                                              device=dev, dtype=self.dtype)
            self._id2slot[layer] = torch.full((n_ev,), -1, dtype=torch.long, device=dev)
            self._lru[layer] = OrderedDict()
            self._cached[layer] = set()
            self._free[layer] = list(range(cap0))
        elif self._id2slot[layer].shape[0] < n_ev:
            old = self._id2slot[layer]                # cold record grew -> extend id->slot map
            grown = torch.full((n_ev,), -1, dtype=torch.long, device=dev)
            grown[: old.shape[0]] = old
            self._id2slot[layer] = grown

    def _grow_bank(self, layer: int) -> None:
        """Double the bank capacity (capped at ``vram_cache_chunks``); add the new slots to free."""
        bk, bv = self._bank_k[layer], self._bank_v[layer]
        cur = bk.shape[2]
        new = min(self.vram_cache_chunks, max(cur * 2, cur + 1))
        if new <= cur:
            return
        nk = bk.new_empty(self.B, self.kvh, new, self.C, self.dh)
        nv = bv.new_empty(self.B, self.kvh, new, self.C, self.dh)
        nk[:, :, :cur] = bk
        nv[:, :, :cur] = bv
        self._bank_k[layer] = nk
        self._bank_v[layer] = nv
        self._free[layer].extend(range(cur, new))

    def _load_missing(self, layer: int, st: "_HybridLayer", unique: List[int]) -> None:
        """Make every cold chunk id in ``unique`` resident in the layer's VRAM bank (LRU evict)."""
        i2s = self._id2slot[layer]
        lru = self._lru[layer]
        cached = self._cached[layer]
        free = self._free[layer]
        dev = self.compute_device
        src_k, src_v = st.cold_tk.data, st.cold_tv.data
        # Touch hits first so they are never evicted to make room for this step's misses.
        for cid in unique:
            if cid in cached:
                lru.move_to_end(cid)
        for cid in unique:
            if cid in cached:
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if not free:
                if self._bank_k[layer].shape[2] < self.vram_cache_chunks:
                    self._grow_bank(layer)          # still room to grow toward the cap
                else:
                    old_id, slot = lru.popitem(last=False)  # evict oldest (not in this request)
                    cached.discard(old_id)
                    i2s[old_id] = -1
                    free.append(slot)
            slot = free.pop()
            self._bank_k[layer][:, :, slot] = src_k[:, :, cid].to(dev)
            self._bank_v[layer][:, :, slot] = src_v[:, :, cid].to(dev)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    def _bank_slots(self, layer: int, st: "_HybridLayer", chunk_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Ensure every chunk in ``chunk_idx`` is resident; return its bank slots (same shape) on
        the compute device, or ``None`` to signal the caller to stream straight from RAM."""
        if self.vram_cache_chunks <= 0 or self.ram_device == self.compute_device:
            return None                              # disabled, or cold already on device
        n_ev = st.cold_tk.n
        if n_ev == 0:
            return None
        unique = torch.unique(chunk_idx).tolist()
        if len(unique) > self.vram_cache_chunks:
            return None                              # working set bigger than cache -> stream
        self._ensure_bank(layer, n_ev)
        self._load_missing(layer, st, unique)
        return self._id2slot[layer][chunk_idx.detach().to(self.compute_device)]

    def detach_hot(self) -> None:
        """Cut the autograd history of the grad hot window (truncated BPTT between blocks).

        The cached token K/V *values* are kept, but their ``grad_fn`` is dropped, so the next
        block's ``backward`` does not try to traverse a previous block's already-freed graph.
        """
        for st in self._layers.values():
            if st.hot:
                st.hot = deque((k.detach(), v.detach()) for k, v in st.hot)

    def num_closed_chunks(self, layer: int) -> int:
        return self._layer(layer).n_closed

    # -- routing table (HOT, on device) -----------------------------------
    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        st = self._layer(layer)
        if st.chunk_k.data is None or st.n_closed == 0:
            return None
        return st.chunk_k.data[:, :, : st.n_closed]

    # -- ingest ------------------------------------------------------------
    def append_closed_chunk(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, Dh]
        group_k: torch.Tensor,   # [B, KVH, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, C, Dh]   (live, grad)
        token_v: torch.Tensor,   # [B, KVH, C, Dh]   (live, grad)
    ) -> None:
        st = self._layer(layer)
        # routing table -> device, detached; summaries -> RAM, detached.
        st.chunk_k.append(chunk_k.detach().unsqueeze(2).to(self.compute_device))
        st.group_k.append(group_k.detach().unsqueeze(2).to(self.ram_device))
        st.group_v.append(group_v.detach().unsqueeze(2).to(self.ram_device))
        # token K/V stay GRAD on the device as the hot window.
        st.hot.append((token_k, token_v))
        if len(st.hot) > self.keep_last:
            etk, etv = st.hot.popleft()              # evict oldest -> detach to RAM (stop-grad)
            st.cold_tk.append(etk.detach().unsqueeze(2).to(self.ram_device))
            st.cold_tv.append(etv.detach().unsqueeze(2).to(self.ram_device))
        st.n_closed += 1

    def seed_closed_chunks(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, N, Dh]
        group_k: torch.Tensor,   # [B, KVH, N, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, N, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, N, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, N, C, Dh]
    ) -> None:
        """Bulk-ingest ``N`` closed chunks (used by the vectorized prefill seed; detached path)."""
        N = chunk_k.shape[2]
        for i in range(N):
            self.append_closed_chunk(
                layer, chunk_k[:, :, i], group_k[:, :, i], group_v[:, :, i],
                token_k[:, :, i], token_v[:, :, i],
            )

    # -- fetch into VRAM (cold = detached RAM->device; hot = grad) ---------
    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        ci = chunk_idx.to(self.ram_device)
        gk = self._gather(st.group_k.data, ci, rep).to(self.compute_device)
        gv = self._gather(st.group_v.data, ci, rep).to(self.compute_device)
        return gk, gv

    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opened-group token K/V for cold chunks (routed middle) → device, detached.

        Uses the bounded LRU VRAM cache when enabled (only newly-required chunks cross PCIe);
        otherwise streams straight from the RAM record.
        """
        st = self._layer(layer)
        H = chunk_idx.shape[1]
        rep = H // self.kvh
        gs = self.C // self.M
        dev = self.compute_device
        slots = self._bank_slots(layer, st, chunk_idx)
        if slots is not None:
            # Index the resident VRAM bank by slot instead of the RAM record by chunk id.
            gi = group_idx.to(dev)
            tok = gi.unsqueeze(-1) * gs + torch.arange(gs, device=dev)   # [B,H,Kg,gs]
            sl = slots.unsqueeze(-1).expand_as(tok)                      # [B,H,Kg,gs]
            b = torch.arange(self.B, device=dev).view(self.B, 1, 1, 1)
            kv = (torch.arange(H, device=dev) // rep).view(1, H, 1, 1)
            k = self._bank_k[layer][b, kv, sl, tok]
            v = self._bank_v[layer][b, kv, sl, tok]
            return k, v
        # Bypass: stream straight from the RAM record (cache off / won't fit / VRAM cold tier).
        src_k, src_v = st.cold_tk.data, st.cold_tv.data
        rd = self.ram_device
        ci = chunk_idx.to(rd)
        gi = group_idx.to(rd)
        tok = gi.unsqueeze(-1) * gs + torch.arange(gs, device=rd)   # [B,H,Kg,gs]
        cidx = ci.unsqueeze(-1).expand_as(tok)                      # [B,H,Kg,gs]
        b = torch.arange(self.B, device=rd).view(self.B, 1, 1, 1)
        kv = (torch.arange(H, device=rd) // rep).view(1, H, 1, 1)
        k = src_k[b, kv, cidx, tok].to(self.compute_device)
        v = src_v[b, kv, cidx, tok].to(self.compute_device)
        return k, v

    def _full_token_record(self, st: _HybridLayer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reassemble the whole ``[B,KVH,n_closed,C,Dh]`` token record on device (rare path)."""
        parts_k, parts_v = [], []
        if st.cold_tk.data is not None:
            parts_k.append(st.cold_tk.data.to(self.compute_device))
            parts_v.append(st.cold_tv.data.to(self.compute_device))
        if st.hot:
            parts_k.append(torch.stack([k for k, _ in st.hot], dim=2))
            parts_v.append(torch.stack([v for _, v in st.hot], dim=2))
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def gather_chunk_tokens(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        tk, tv = self._full_token_record(st)
        return self._gather(tk, chunk_idx, rep), self._gather(tv, chunk_idx, rep)

    # -- always-resident windows ------------------------------------------
    def hot_group_summaries(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        gk = st.group_k.data[:, :, lo:hi].to(self.compute_device)
        gv = st.group_v.data[:, :, lo:hi].to(self.compute_device)
        return gk, gv

    def hot_tokens(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token K/V for chunks ``[lo, hi)``: cold chunks (detached, from RAM) + hot chunks (grad)."""
        st = self._layer(layer)
        n_ev = st.cold_tk.n
        parts_k, parts_v = [], []
        # cold portion [lo, min(hi, n_ev)) -> detached, RAM -> device
        c_hi = min(hi, n_ev)
        if c_hi > lo:
            parts_k.append(st.cold_tk.data[:, :, lo:c_hi].to(self.compute_device))
            parts_v.append(st.cold_tv.data[:, :, lo:c_hi].to(self.compute_device))
        # hot portion [max(lo, n_ev), hi) -> grad, already on device
        h_lo = max(lo, n_ev)
        if hi > h_lo:
            ks = [st.hot[j][0].unsqueeze(2) for j in range(h_lo - n_ev, hi - n_ev)]
            vs = [st.hot[j][1].unsqueeze(2) for j in range(h_lo - n_ev, hi - n_ev)]
            parts_k.append(torch.cat(ks, dim=2))
            parts_v.append(torch.cat(vs, dim=2))
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)
