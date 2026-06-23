"""NVMe/filesystem KV-cache store — RAM-bounded tier with disk spillover.

The RAM tier (:class:`~.ram_kv_cache_store.RamKVCacheStore`) holds the *complete* cold record in
host memory.  For very long contexts that record outgrows RAM, so :class:`FsKVCacheStore` adds a
third tier *below* RAM: host RAM becomes a **bounded LRU page cache** (12 GB by default) over a
disk-backed full record, exactly mirroring how the VRAM banks are a bounded cache over the RAM
record.  The tiering is therefore recursive:

    VRAM banks   (bounded)  ── cache of ──▶  RAM staging
    RAM staging  (bounded)  ── cache of ──▶  disk files   (the full record)
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from .chunk_placement_policy import ChunkPlacementPolicy
from .fs_disk_manager import (
    _FsDiskManager,
    _fadvise_dontneed,
    _pread_all,
    _pwrite_all,
    _raw_bytes,
)
from .layer_store import _LayerStore
from .ram_kv_cache_store import RamKVCacheStore


class _FsLayerRecord:
    """The full cold record of one layer, tiered as a bounded RAM page cache over disk files.

    Chunks are immutable once closed, so a chunk that has been written to disk needs no rewrite:
    evicting it from RAM later is free.  RAM holds up to ``ram_cap`` chunks in contiguous staging
    tensors (so gathers stay a single fancy-index op); the rest live in four per-layer files
    (``k``/``v`` × token/group), each laid out chunk-major at ``offset = chunk_id * bytes``.
    """

    def __init__(self, store: "FsKVCacheStore", layer: int) -> None:
        self.s = store
        self.layer = layer
        self.B, self.KVH, self.C, self.M, self.Dh = store.B, store.kvh, store.C, store.M, store.dh
        self.dtype = store.dtype
        itemsize = store._dtype_bytes
        self.tok_bytes = self.B * self.KVH * self.C * self.Dh * itemsize
        self.grp_bytes = self.B * self.KVH * self.M * self.Dh * itemsize
        self.ram_cap = max(1, int(store._fs_ram_cap))

        self.n_closed = 0
        self.cap = 0  # current staging capacity (chunks), grows by doubling up to ram_cap
        self.ram_gk: Optional[torch.Tensor] = None
        self.ram_gv: Optional[torch.Tensor] = None
        self.ram_tk: Optional[torch.Tensor] = None
        self.ram_tv: Optional[torch.Tensor] = None

        self.id2slot: Dict[int, int] = {}
        self.lru: "OrderedDict[int, int]" = OrderedDict()
        self.free: List[int] = []
        self._slotmap: Optional[torch.Tensor] = None  # [cap_ids] long, chunk_id → slot, -1 = absent

        # Disk side (lazy: files are only created the first time a chunk actually spills).
        self.lock = threading.Lock()          # guards ``on_disk`` / ``pending`` vs. the writer thread
        self.on_disk: set = set()
        self.pending: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._fds: Optional[Tuple[int, int, int, int]] = None  # (tk, tv, gk, gv)
        self._bounce_tk: Optional[torch.Tensor] = None
        self._bounce_tv: Optional[torch.Tensor] = None
        self._bounce_gk: Optional[torch.Tensor] = None
        self._bounce_gv: Optional[torch.Tensor] = None

    # -- capacity bookkeeping ---------------------------------------------
    @property
    def capacity_ids(self) -> int:
        """Size of the id→slot space (≥ ``n_closed``); used to size the VRAM banks' id maps."""
        return 0 if self._slotmap is None else int(self._slotmap.shape[0])

    def _ensure_slotmap(self, need: int) -> None:
        cur = 0 if self._slotmap is None else self._slotmap.shape[0]
        if need <= cur:
            return
        cap = max(64, cur * 2, need)
        new = torch.full((cap,), -1, dtype=torch.long)
        if self._slotmap is not None:
            new[:cur] = self._slotmap
        self._slotmap = new

    def _set_slot(self, cid: int, slot: int) -> None:
        self._slotmap[cid] = slot

    def _grow_staging(self, new_cap: int) -> None:
        B, KVH, C, M, Dh = self.B, self.KVH, self.C, self.M, self.Dh
        gk = torch.empty(B, KVH, new_cap, M, Dh, dtype=self.dtype)
        gv = torch.empty(B, KVH, new_cap, M, Dh, dtype=self.dtype)
        tk = torch.empty(B, KVH, new_cap, C, Dh, dtype=self.dtype)
        tv = torch.empty(B, KVH, new_cap, C, Dh, dtype=self.dtype)
        if self.ram_gk is not None:
            old = self.cap
            gk[:, :, :old] = self.ram_gk
            gv[:, :, :old] = self.ram_gv
            tk[:, :, :old] = self.ram_tk
            tv[:, :, :old] = self.ram_tv
        self.free.extend(range(self.cap, new_cap))
        self.ram_gk, self.ram_gv, self.ram_tk, self.ram_tv = gk, gv, tk, tv
        self.cap = new_cap

    def _alloc_slot(self, protect: Optional[set] = None) -> int:
        if self.free:
            return self.free.pop()
        if self.cap < self.ram_cap:
            self._grow_staging(min(self.ram_cap, max(64, self.cap * 2, self.cap + 1)))
            return self.free.pop()
        return self._evict_one(protect)

    def _evict_one(self, protect: Optional[set]) -> int:
        # Oldest first; skip any chunk that must stay resident for the in-flight request.
        for cid in list(self.lru.keys()):
            if protect and cid in protect:
                continue
            slot = self.lru.pop(cid)
            del self.id2slot[cid]
            self._set_slot(cid, -1)
            with self.lock:
                already = cid in self.on_disk or cid in self.pending
            if not already:
                self._spill(cid, slot)
            return slot
        raise RuntimeError("FS record: no evictable slot (ram_cap too small for the request)")

    # -- disk files --------------------------------------------------------
    def _open_files(self) -> None:
        if self._fds is not None:
            return
        d = self.s.disk
        self._fds = (
            d.open_file(f"L{self.layer}.tk"),
            d.open_file(f"L{self.layer}.tv"),
            d.open_file(f"L{self.layer}.gk"),
            d.open_file(f"L{self.layer}.gv"),
        )
        B, KVH, C, M, Dh = self.B, self.KVH, self.C, self.M, self.Dh
        self._bounce_tk = torch.empty(B, KVH, C, Dh, dtype=self.dtype)
        self._bounce_tv = torch.empty(B, KVH, C, Dh, dtype=self.dtype)
        self._bounce_gk = torch.empty(B, KVH, M, Dh, dtype=self.dtype)
        self._bounce_gv = torch.empty(B, KVH, M, Dh, dtype=self.dtype)

    def _spill(self, cid: int, slot: int) -> None:
        """Hand chunk ``cid`` (currently in RAM ``slot``) to the writer thread, then free the slot.

        Only a cheap RAM→RAM clone happens on the caller; the ``pwrite`` runs off-thread, so the
        compute path is never blocked on disk.  Until the write lands, the clone serves any re-read.
        """
        self._open_files()
        tk = self.ram_tk[:, :, slot].clone()
        tv = self.ram_tv[:, :, slot].clone()
        gk = self.ram_gk[:, :, slot].clone()
        gv = self.ram_gv[:, :, slot].clone()
        with self.lock:
            self.pending[cid] = (tk, tv, gk, gv)
        fd_tk, fd_tv, fd_gk, fd_gv = self._fds  # type: ignore[misc]
        tok_off = cid * self.tok_bytes
        grp_off = cid * self.grp_bytes

        def _do() -> None:
            _pwrite_all(fd_tk, _raw_bytes(tk), tok_off)
            _pwrite_all(fd_tv, _raw_bytes(tv), tok_off)
            _pwrite_all(fd_gk, _raw_bytes(gk), grp_off)
            _pwrite_all(fd_gv, _raw_bytes(gv), grp_off)
            _fadvise_dontneed(fd_tk, tok_off, self.tok_bytes)
            _fadvise_dontneed(fd_tv, tok_off, self.tok_bytes)
            _fadvise_dontneed(fd_gk, grp_off, self.grp_bytes)
            _fadvise_dontneed(fd_gv, grp_off, self.grp_bytes)
            with self.lock:
                self.pending.pop(cid, None)
                self.on_disk.add(cid)

        self.s.disk.submit(_do)

    def _fill_slot(self, cid: int, slot: int) -> None:
        """Materialise chunk ``cid`` into RAM ``slot`` from the pending-write buffer or from disk."""
        with self.lock:
            pend = self.pending.get(cid)
        if pend is not None:
            tk, tv, gk, gv = pend
            self.ram_tk[:, :, slot] = tk
            self.ram_tv[:, :, slot] = tv
            self.ram_gk[:, :, slot] = gk
            self.ram_gv[:, :, slot] = gv
            return
        fd_tk, fd_tv, fd_gk, fd_gv = self._fds  # type: ignore[misc]
        tok_off = cid * self.tok_bytes
        grp_off = cid * self.grp_bytes
        _pread_all(fd_tk, _raw_bytes(self._bounce_tk), tok_off)
        _pread_all(fd_tv, _raw_bytes(self._bounce_tv), tok_off)
        _pread_all(fd_gk, _raw_bytes(self._bounce_gk), grp_off)
        _pread_all(fd_gv, _raw_bytes(self._bounce_gv), grp_off)
        _fadvise_dontneed(fd_tk, tok_off, self.tok_bytes)
        _fadvise_dontneed(fd_tv, tok_off, self.tok_bytes)
        _fadvise_dontneed(fd_gk, grp_off, self.grp_bytes)
        _fadvise_dontneed(fd_gv, grp_off, self.grp_bytes)
        self.ram_tk[:, :, slot] = self._bounce_tk
        self.ram_tv[:, :, slot] = self._bounce_tv
        self.ram_gk[:, :, slot] = self._bounce_gk
        self.ram_gv[:, :, slot] = self._bounce_gv

    # -- public surface used by the store ---------------------------------
    def ensure_resident(self, unique_ids: List[int]) -> None:
        """Make every chunk in ``unique_ids`` RAM-resident (loading from disk under LRU eviction)."""
        protect: set = set()
        for cid in unique_ids:
            if cid in self.id2slot:
                self.lru.move_to_end(cid)
                protect.add(cid)
        for cid in unique_ids:
            if cid in self.id2slot:
                continue
            slot = self._alloc_slot(protect)
            self._fill_slot(cid, slot)
            self.id2slot[cid] = slot
            self.lru[cid] = slot
            self._set_slot(cid, slot)
            protect.add(cid)

    def slots_for(self, chunk_idx: torch.Tensor) -> torch.Tensor:
        """Resident slot indices for ``chunk_idx`` (ensures residency first)."""
        unique = torch.unique(chunk_idx).tolist()
        self.ensure_resident(unique)
        return self._slotmap[chunk_idx.detach().to("cpu")]

    def append(self, idx: int, gk: torch.Tensor, gv: torch.Tensor,
               tk: torch.Tensor, tv: torch.Tensor) -> None:
        self.n_closed = idx + 1
        self._ensure_slotmap(idx + 1)
        slot = self._alloc_slot()
        self.ram_gk[:, :, slot] = gk
        self.ram_gv[:, :, slot] = gv
        self.ram_tk[:, :, slot] = tk
        self.ram_tv[:, :, slot] = tv
        self.id2slot[idx] = slot
        self.lru[idx] = slot
        self._set_slot(idx, slot)
        with self.lock:
            self.on_disk.discard(idx)

    def seed(self, gk: torch.Tensor, gv: torch.Tensor,
             tk: torch.Tensor, tv: torch.Tensor) -> None:
        """Bulk-ingest ``N`` chunks: the last ``ram_cap`` stay in RAM, the rest spill to disk once."""
        N = gk.shape[2]
        self.n_closed = N
        self._ensure_slotmap(N)
        keep = min(N, self.ram_cap)
        disk_n = N - keep
        if disk_n > 0:
            self._open_files()
            fd_tk, fd_tv, fd_gk, fd_gv = self._fds  # type: ignore[misc]
            # File layout is chunk-major ([cid][B,KVH,...]); permute the [B,KVH,N,...] input so the
            # spilled prefix is one contiguous write per file (offset 0 = chunk 0).
            tkb = tk[:, :, :disk_n].permute(2, 0, 1, 3, 4).contiguous()
            tvb = tv[:, :, :disk_n].permute(2, 0, 1, 3, 4).contiguous()
            gkb = gk[:, :, :disk_n].permute(2, 0, 1, 3, 4).contiguous()
            gvb = gv[:, :, :disk_n].permute(2, 0, 1, 3, 4).contiguous()
            _pwrite_all(fd_tk, _raw_bytes(tkb), 0)
            _pwrite_all(fd_tv, _raw_bytes(tvb), 0)
            _pwrite_all(fd_gk, _raw_bytes(gkb), 0)
            _pwrite_all(fd_gv, _raw_bytes(gvb), 0)
            _fadvise_dontneed(fd_tk, 0, disk_n * self.tok_bytes)
            _fadvise_dontneed(fd_tv, 0, disk_n * self.tok_bytes)
            _fadvise_dontneed(fd_gk, 0, disk_n * self.grp_bytes)
            _fadvise_dontneed(fd_gv, 0, disk_n * self.grp_bytes)
            with self.lock:
                self.on_disk.update(range(disk_n))
        # Resident tail: chunks [disk_n, N).
        if self.cap < keep:
            self._grow_staging(min(self.ram_cap, max(self.cap, keep)))
        for j, cid in enumerate(range(disk_n, N)):
            slot = self._alloc_slot()
            self.ram_gk[:, :, slot] = gk[:, :, cid]
            self.ram_gv[:, :, slot] = gv[:, :, cid]
            self.ram_tk[:, :, slot] = tk[:, :, cid]
            self.ram_tv[:, :, slot] = tv[:, :, cid]
            self.id2slot[cid] = slot
            self.lru[cid] = slot
            self._set_slot(cid, slot)

    def reset(self) -> None:
        """Drop all RAM state and delete this layer's spill files."""
        if self._fds is not None:
            for fd in self._fds:
                try:
                    os.ftruncate(fd, 0)
                except OSError:
                    pass
        self.n_closed = 0
        self.cap = 0
        self.ram_gk = self.ram_gv = self.ram_tk = self.ram_tv = None
        self.id2slot.clear()
        self.lru.clear()
        self.free.clear()
        self._slotmap = None
        with self.lock:
            self.on_disk.clear()
            self.pending.clear()


class FsKVCacheStore(RamKVCacheStore):
    """RAM-bounded tier with NVMe/disk spillover for contexts that exceed host RAM.

    Identical to :class:`RamKVCacheStore` except the cold record is held in a bounded RAM page
    cache (``ram_budget_gb``, 12 GB by default) backed by per-layer disk files: when RAM fills, the
    least-recently-used chunks are spilled to disk asynchronously and pulled back on demand.  VRAM
    behaviour (``chunk_k`` routing table, hot windows, token/summary banks) is unchanged, so this is
    a drop-in replacement for the RAM store at any context length.
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
        initial_capacity: int = 64,
        vram_cache_chunks: int = 0,
        vram_summary_chunks: int = 0,
        num_layers: int = 0,
        vram_cache_reserve_gb: float = 1.5,
        ram_budget_gb: float = 12.0,
        fs_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            compute_device=compute_device,
            policy=policy,
            kv_heads=kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            groups_per_chunk=groups_per_chunk,
            batch_size=batch_size,
            dtype=dtype,
            pin_memory=False,
            initial_capacity=initial_capacity,
            storage_device=torch.device("cpu"),
            vram_cache_chunks=vram_cache_chunks,
            vram_summary_chunks=vram_summary_chunks,
            num_layers=num_layers,
            vram_cache_reserve_gb=vram_cache_reserve_gb,
        )
        if num_layers <= 0:
            raise ValueError("FsKVCacheStore requires num_layers > 0 to budget the RAM page cache")
        self.ram_budget_gb = float(ram_budget_gb)
        # Per-layer RAM capacity (chunks): the whole budget covers token K/V + group K/V across all
        # layers, so the total resident host memory is bounded by ``ram_budget_gb`` regardless of
        # how long the context grows.
        per_chunk_all_layers = (
            self.num_layers * self.B * self.kvh * (self.C + self.M) * self.dh * self._dtype_bytes * 2
        )
        budget = int(self.ram_budget_gb * 1024**3)
        self._fs_ram_cap = max(1, budget // per_chunk_all_layers) if per_chunk_all_layers > 0 else 1
        self.disk = _FsDiskManager(root=fs_cache_dir)
        self._rec: Dict[int, _FsLayerRecord] = {}
        if os.environ.get("KVR_DEBUG"):
            import sys as _sys
            print(f"[kvr] FsKVCacheStore: ram_budget={self.ram_budget_gb}GB → ram_cap={self._fs_ram_cap} "
                  f"chunks/layer ({self._fs_ram_cap * self.C} tok/layer); spill dir {self.disk.dir}",
                  file=_sys.stderr, flush=True)

    # -- record access -----------------------------------------------------
    def _rec_for(self, layer: int) -> _FsLayerRecord:
        rec = self._rec.get(layer)
        if rec is None:
            rec = _FsLayerRecord(self, layer)
            self._rec[layer] = rec
        return rec

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        for rec in self._rec.values():
            rec.reset()
        self._rec.clear()

    def close(self) -> None:
        """Release the writer thread and delete all spill files (idempotent)."""
        self.disk.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            self.disk.close()
        except Exception:
            pass

    # -- capacity (chunk_k VRAM table only; the cold record self-manages) ---
    def _ensure_capacity(self, st: _LayerStore, need: int) -> None:
        cur = 0 if st.chunk_k is None else st.chunk_k.shape[2]
        if need <= cur:
            return
        cap = max(self._init_cap, cur * 2, need)
        ck = torch.zeros(self.B, self.kvh, cap, self.dh, device=self.compute_device, dtype=self.dtype)
        if st.chunk_k is not None:
            ck[:, :, : st.n_closed] = st.chunk_k[:, :, : st.n_closed]
        st.chunk_k = ck

    # -- ingest ------------------------------------------------------------
    def append_closed_chunk(
        self,
        layer: int,
        chunk_k: torch.Tensor,
        group_k: torch.Tensor,
        group_v: torch.Tensor,
        token_k: torch.Tensor,
        token_v: torch.Tensor,
    ) -> None:
        st = self._layer(layer)
        idx = st.n_closed
        self._ensure_capacity(st, idx + 1)
        st.chunk_k[:, :, idx] = chunk_k.detach()
        # Cold record (RAM-bounded, disk-backed): detached host copies.
        sd = self.storage_device
        self._rec_for(layer).append(
            idx,
            group_k.detach().to(sd),
            group_v.detach().to(sd),
            token_k.detach().to(sd),
            token_v.detach().to(sd),
        )
        # Live grad-carrying copies for the hot windows.
        st.live_group_k[idx] = group_k
        st.live_group_v[idx] = group_v
        st.live_token_k[idx] = token_k
        st.live_token_v[idx] = token_v
        st.n_closed = idx + 1
        self._evict_outside_windows(st)

    def seed_closed_chunks(
        self,
        layer: int,
        chunk_k: torch.Tensor,
        group_k: torch.Tensor,
        group_v: torch.Tensor,
        token_k: torch.Tensor,
        token_v: torch.Tensor,
    ) -> None:
        st = self._layer(layer)
        N = chunk_k.shape[2]
        if N == 0:
            return
        self._ensure_capacity(st, N)
        st.chunk_k[:, :, :N] = chunk_k.detach()
        sd = self.storage_device
        self._rec_for(layer).seed(
            group_k.detach().to(sd),
            group_v.detach().to(sd),
            token_k.detach().to(sd),
            token_v.detach().to(sd),
        )
        st.n_closed = N
        st.live_group_k.clear(); st.live_group_v.clear()
        st.live_token_k.clear(); st.live_token_v.clear()
        f_lo, f_hi = self.policy.hot_first_range(N)
        l_lo, l_hi = self.policy.hot_last_range(N)
        for cid in sorted(set(range(f_lo, f_hi)) | set(range(l_lo, l_hi))):
            st.live_group_k[cid] = group_k[:, :, cid]
            st.live_group_v[cid] = group_v[:, :, cid]
        token_windows = list(range(l_lo, l_hi))
        if self.policy.first_token_level:
            token_windows += list(range(f_lo, f_hi))
        for cid in token_windows:
            st.live_token_k[cid] = token_k[:, :, cid]
            st.live_token_v[cid] = token_v[:, :, cid]

    # -- VRAM bank id-map sizing (record capacity instead of a contiguous tensor) --
    def _ensure_bank(self, layer: int, st: _LayerStore) -> None:
        cap = self._effective_cap()
        rec_cap = self._rec_for(layer).capacity_ids
        dev = self.compute_device
        if self._bank_k.get(layer) is None:
            self._bank_k[layer] = torch.empty(self.B, self.kvh, cap, self.C, self.dh,
                                              device=dev, dtype=self.dtype)
            self._bank_v[layer] = torch.empty(self.B, self.kvh, cap, self.C, self.dh,
                                              device=dev, dtype=self.dtype)
            self._id2slot[layer] = torch.full((rec_cap,), -1, dtype=torch.long, device=dev)
            self._lru[layer] = OrderedDict()
            self._cached[layer] = set()
            self._free[layer] = list(range(cap))
            self._i2s_cap[layer] = rec_cap
        elif self._i2s_cap[layer] < rec_cap:
            old = self._id2slot[layer]
            grown = torch.full((rec_cap,), -1, dtype=torch.long, device=dev)
            grown[: old.shape[0]] = old
            self._id2slot[layer] = grown
            self._i2s_cap[layer] = rec_cap

    def _ensure_summary_bank(self, layer: int, st: _LayerStore) -> None:
        cap = self._effective_summary_cap()
        rec_cap = self._rec_for(layer).capacity_ids
        dev = self.compute_device
        if self._sbank_gk.get(layer) is None:
            self._sbank_gk[layer] = torch.empty(self.B, self.kvh, cap, self.M, self.dh,
                                                device=dev, dtype=self.dtype)
            self._sbank_gv[layer] = torch.empty(self.B, self.kvh, cap, self.M, self.dh,
                                                device=dev, dtype=self.dtype)
            self._sid2slot[layer] = torch.full((rec_cap,), -1, dtype=torch.long, device=dev)
            self._slru[layer] = OrderedDict()
            self._scached[layer] = set()
            self._sfree[layer] = list(range(cap))
            self._si2s_cap[layer] = rec_cap
        elif self._si2s_cap[layer] < rec_cap:
            old = self._sid2slot[layer]
            grown = torch.full((rec_cap,), -1, dtype=torch.long, device=dev)
            grown[: old.shape[0]] = old
            self._sid2slot[layer] = grown
            self._si2s_cap[layer] = rec_cap

    # -- VRAM bank feeders (pull from the resident RAM slot, not a contiguous record) --
    def _load_missing(self, layer: int, st: _LayerStore, unique: List[int]) -> None:
        rec = self._rec_for(layer)
        rec.ensure_resident(unique)
        i2s = self._id2slot[layer]
        lru = self._lru[layer]
        cached = self._cached[layer]
        free = self._free[layer]
        bank_k, bank_v = self._bank_k[layer], self._bank_v[layer]
        dev = self.compute_device
        for cid in unique:
            if cid in cached:
                lru.move_to_end(cid)
        for cid in unique:
            if cid in cached:
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if free:
                slot = free.pop()
            else:
                old_id, slot = lru.popitem(last=False)
                cached.discard(old_id)
                i2s[old_id] = -1
            rslot = rec.id2slot[cid]
            bank_k[:, :, slot] = rec.ram_tk[:, :, rslot].to(dev)
            bank_v[:, :, slot] = rec.ram_tv[:, :, rslot].to(dev)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    def _load_missing_summaries(self, layer: int, st: _LayerStore, unique: List[int]) -> None:
        rec = self._rec_for(layer)
        rec.ensure_resident(unique)
        i2s = self._sid2slot[layer]
        lru = self._slru[layer]
        cached = self._scached[layer]
        free = self._sfree[layer]
        bank_gk, bank_gv = self._sbank_gk[layer], self._sbank_gv[layer]
        dev = self.compute_device
        for cid in unique:
            if cid in cached:
                lru.move_to_end(cid)
        for cid in unique:
            if cid in cached:
                self.summary_hits += 1
                continue
            self.summary_misses += 1
            if free:
                slot = free.pop()
            else:
                old_id, slot = lru.popitem(last=False)
                cached.discard(old_id)
                i2s[old_id] = -1
            rslot = rec.id2slot[cid]
            bank_gk[:, :, slot] = rec.ram_gk[:, :, rslot].to(dev)
            bank_gv[:, :, slot] = rec.ram_gv[:, :, rslot].to(dev)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    # -- direct (no-bank) gathers stream from the resident RAM staging ------
    def _gather_stage(self, stage: torch.Tensor, slots: torch.Tensor, rep: int) -> torch.Tensor:
        """Gather ``stage[B,KVH,cap,*tail]`` by resident ``slots[B,H,*]`` → ``[B,H,*,tail]`` on device."""
        B, KVH = stage.shape[0], stage.shape[1]
        H = slots.shape[1]
        b = torch.arange(B).view(B, 1, *([1] * (slots.ndim - 2)))
        kv = (torch.arange(H) // rep).view(1, H, *([1] * (slots.ndim - 2)))
        gathered = stage[b, kv, slots]
        return gathered.to(self.compute_device)

    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        slots = self._summary_slots(layer, st, chunk_idx)
        if slots is None:  # summary cache off / won't fit → stream from resident RAM staging
            rec = self._rec_for(layer)
            rslots = rec.slots_for(chunk_idx)
            return (self._gather_stage(rec.ram_gk, rslots, rep),
                    self._gather_stage(rec.ram_gv, rslots, rep))
        dev = self.compute_device
        H, nd = chunk_idx.shape[1], chunk_idx.ndim
        b = torch.arange(self.B, device=dev).view(self.B, *([1] * (nd - 1)))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * (nd - 2)))
        return self._sbank_gk[layer][b, kv, slots], self._sbank_gv[layer][b, kv, slots]

    def gather_chunk_tokens(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self._rec_for(layer)
        rep = chunk_idx.shape[1] // self.kvh
        rslots = rec.slots_for(chunk_idx)
        return (self._gather_stage(rec.ram_tk, rslots, rep),
                self._gather_stage(rec.ram_tv, rslots, rep))

    def gather_chunk_tokens_kvh(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        B, KVH, K = chunk_idx.shape
        dev = self.compute_device
        rec = self._rec_for(layer)
        unique = torch.unique(chunk_idx).tolist()
        cap = self._effective_cap()
        if cap <= 0 or len(unique) > cap:
            # Bank off / won't fit → stream straight from RAM staging.
            rslots = rec.slots_for(chunk_idx)
            b = torch.arange(B).view(B, 1, 1)
            g = torch.arange(KVH).view(1, KVH, 1)
            k = rec.ram_tk[b, g, rslots].to(dev)
            v = rec.ram_tv[b, g, rslots].to(dev)
            return k, v
        self._ensure_bank(layer, st)
        self._load_missing(layer, st, unique)
        slots = self._id2slot[layer][chunk_idx]
        bank_k, bank_v = self._bank_k[layer], self._bank_v[layer]
        b = torch.arange(B, device=dev).view(B, 1, 1)
        g = torch.arange(KVH, device=dev).view(1, KVH, 1)
        return bank_k[b, g, slots], bank_v[b, g, slots]

    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        H = chunk_idx.shape[1]
        rep = H // self.kvh
        gs = self.C // self.M
        dev = self.compute_device
        slots = self._bank_slots(layer, st, chunk_idx)
        if slots is not None:  # bank tier: slice the gs opened tokens out of the resident chunk
            tok = group_idx.detach().to(dev).unsqueeze(-1) * gs + torch.arange(gs, device=dev)
            slot_exp = slots.unsqueeze(-1).expand_as(tok)
            b = torch.arange(self.B, device=dev).view(self.B, 1, 1, 1)
            kv = (torch.arange(H, device=dev) // rep).view(1, H, 1, 1)
            return self._bank_k[layer][b, kv, slot_exp, tok], self._bank_v[layer][b, kv, slot_exp, tok]
        # Bank off / won't fit → slice from resident RAM staging.
        rec = self._rec_for(layer)
        rslots = rec.slots_for(chunk_idx)  # [B,H,Kg]
        tok = group_idx.detach().to("cpu").unsqueeze(-1) * gs + torch.arange(gs)
        slot_exp = rslots.unsqueeze(-1).expand_as(tok)
        b = torch.arange(self.B).view(self.B, 1, 1, 1)
        kv = (torch.arange(H) // rep).view(1, H, 1, 1)
        k = rec.ram_tk[b, kv, slot_exp, tok]
        v = rec.ram_tv[b, kv, slot_exp, tok]
        return k.to(dev), v.to(dev)

    # -- IO hint -----------------------------------------------------------
    def prefetch(self, layer: int, chunk_idx: torch.Tensor) -> None:
        """Warm RAM residency for ``chunk_idx`` ahead of the gather (loads any spilled chunks)."""
        rec = self._rec_for(layer)
        rec.ensure_resident(torch.unique(chunk_idx).tolist())
