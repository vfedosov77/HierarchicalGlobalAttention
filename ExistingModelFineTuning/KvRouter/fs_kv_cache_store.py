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
    _fallocate,
    _pread_all,
    _pwrite_all,
    _raw_bytes,
)
from .layer_store import _LayerStore
from .ram_kv_cache_store import RamKVCacheStore


def _available_host_gb() -> Optional[float]:
    """Currently-available host RAM in GB (Linux ``MemAvailable``), or ``None`` if unknown.

    ``MemAvailable`` already accounts for reclaimable page cache, so it is the right figure to size
    a memory budget against without double-counting the kernel's own caches.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2  # kB -> GB
    except (OSError, ValueError, IndexError):
        pass
    return None


class _PagedCache:
    """Bounded RAM page cache for **one** ``(K, V)`` tensor pair of a layer, disk-backed.

    Holds up to ``ram_cap`` chunks in contiguous staging tensors ``sk``/``sv`` of shape
    ``[B, KVH, cap, *tail]`` (so gathers stay a single fancy-index op); chunks beyond that spill to
    **one** per-layer file (``L{layer}.{prefix}``) laid out chunk-major with ``K`` and ``V`` adjacent
    (``offset = chunk_id * 2 * bytes``; ``K`` first, then ``V``), so each spill/reload is a single
    ``pwrite``/``pread`` over one fd instead of two.  Chunks are immutable once closed, so a chunk
    already on disk never needs rewriting — evicting it from RAM later is free.

    Group summaries (``M·Dh`` per chunk) and token K/V (``C·Dh`` per chunk) get **separate**
    instances of this cache with independent ``ram_cap`` budgets, so scoring a routed chunk's tiny
    group summary never drags its ~``C/M``× larger token K/V across PCIe / off disk.
    """

    def __init__(self, store: "FsKVCacheStore", layer: int, prefix: str,
                 tail: Tuple[int, ...], ram_cap: int, prealloc_chunks: int = 0) -> None:
        self.s = store
        self.layer = layer
        self.prefix = prefix                 # "t" (token) or "g" (group) → file name L{layer}.{prefix}
        self.B, self.KVH = store.B, store.kvh
        self.tail = tuple(tail)              # (C, Dh) for tokens, (M, Dh) for groups
        self.dtype = store.dtype
        ntail = 1
        for d in self.tail:
            ntail *= d
        self.bytes = self.B * self.KVH * ntail * store._dtype_bytes  # per-chunk K (or V) payload bytes
        self.cbytes = 2 * self.bytes         # combined K+V per-chunk stride on disk
        self.ram_cap = max(1, int(ram_cap))
        self.prealloc_chunks = max(0, int(prealloc_chunks))

        self.cap = 0  # current staging capacity (chunks), grows by doubling up to ram_cap
        self.sk: Optional[torch.Tensor] = None
        self.sv: Optional[torch.Tensor] = None

        self.id2slot: Dict[int, int] = {}
        self.lru: "OrderedDict[int, int]" = OrderedDict()
        self.free: List[int] = []
        self._slotmap: Optional[torch.Tensor] = None  # [cap_ids] long, chunk_id → slot, -1 = absent

        # Disk side (lazy: the file is only created the first time a chunk actually spills — unless
        # preallocation is requested, in which case it is created and reserved up front).
        self.lock = threading.Lock()          # guards ``on_disk`` / ``pending`` vs. the writer thread
        self.on_disk: set = set()
        self.pending: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._fd: Optional[int] = None        # single combined K+V file
        self._bkv: Optional[torch.Tensor] = None  # [2, B, KVH, *tail] read bounce buffer
        if self.prealloc_chunks > 0:
            # Pre-grow the resident staging to its known bound and reserve the disk file, avoiding
            # repeated staging reallocation/copies and file-extension metadata churn at runtime.
            self._grow_staging(min(self.ram_cap, self.prealloc_chunks))
            self._open_files()

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
        sk = torch.empty(self.B, self.KVH, new_cap, *self.tail, dtype=self.dtype)
        sv = torch.empty(self.B, self.KVH, new_cap, *self.tail, dtype=self.dtype)
        if self.sk is not None:
            old = self.cap
            sk[:, :, :old] = self.sk
            sv[:, :, :old] = self.sv
        self.free.extend(range(self.cap, new_cap))
        self.sk, self.sv = sk, sv
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
        # Every resident chunk is protected by the in-flight request (its working set exceeds the
        # current staging).  Grow rather than fail — one gather's resident set is bounded by the
        # routing config and must fit; ``ram_cap`` only bounds retention between gathers.
        self._grow_staging(self.cap + max(64, self.cap // 4))
        return self.free.pop()

    # -- disk files --------------------------------------------------------
    def _open_files(self) -> None:
        if self._fd is not None:
            return
        self._fd = self.s.disk.open_file(f"L{self.layer}.{self.prefix}")
        if self.prealloc_chunks > 0:
            _fallocate(self._fd, self.prealloc_chunks * self.cbytes)
        self._bkv = torch.empty(2, self.B, self.KVH, *self.tail, dtype=self.dtype)

    def _spill(self, cid: int, slot: int) -> None:
        """Hand chunk ``cid`` (currently in RAM ``slot``) to the writer pool, then free the slot.

        Only a cheap RAM→RAM clone happens on the caller; the ``pwrite`` runs off-thread, so the
        compute path is never blocked on disk.  Until the write lands, the clone serves any re-read.
        K and V are packed into one contiguous buffer and written with a single ``pwrite``.
        """
        self._open_files()
        kv = torch.stack((self.sk[:, :, slot], self.sv[:, :, slot]), dim=0).contiguous()
        with self.lock:
            self.pending[cid] = (kv[0], kv[1])
        fd = self._fd  # type: ignore[misc]
        off = cid * self.cbytes

        def _do() -> None:
            _pwrite_all(fd, _raw_bytes(kv), off)
            _fadvise_dontneed(fd, off, self.cbytes)
            with self.lock:
                self.pending.pop(cid, None)
                self.on_disk.add(cid)

        self.s.disk.submit(_do)

    def _fill_slot(self, cid: int, slot: int) -> None:
        """Materialise chunk ``cid`` into RAM ``slot`` from the pending-write buffer or from disk."""
        with self.lock:
            pend = self.pending.get(cid)
        if pend is not None:
            k, v = pend
            self.sk[:, :, slot] = k
            self.sv[:, :, slot] = v
            return
        fd = self._fd  # type: ignore[misc]
        off = cid * self.cbytes
        _pread_all(fd, _raw_bytes(self._bkv), off)
        _fadvise_dontneed(fd, off, self.cbytes)
        self.s.disk_reads += 1
        self.s.disk_read_bytes += self.cbytes
        self.sk[:, :, slot] = self._bkv[0]
        self.sv[:, :, slot] = self._bkv[1]

    # -- public surface used by the record --------------------------------
    def ensure_resident(self, unique_ids: List[int]) -> None:
        """Make every chunk in ``unique_ids`` RAM-resident (loading from disk under LRU eviction)."""
        # A single gather needs ALL its distinct chunks resident *simultaneously*.  The per-step
        # working set can exceed ``ram_cap`` — e.g. the opened-token gather unions the chunks the
        # query heads route to, and divergent heads can request hundreds of distinct chunks (bounded
        # by the routing config: ~ topk_chunks * heads).  ``ram_cap`` is a *retention* target between
        # gathers, NOT a hard cap on one gather's resident set, so grow the staging to fit the request
        # rather than failing with "no evictable slot".  Keeping ``ram_cap`` small therefore makes the
        # staging grow only to the actual working set instead of pre-allocating a large fixed cap.
        need = len(unique_ids)
        if need > self.cap:
            self._grow_staging(need)
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

    def append(self, idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._ensure_slotmap(idx + 1)
        slot = self._alloc_slot()
        self.sk[:, :, slot] = k
        self.sv[:, :, slot] = v
        self.id2slot[idx] = slot
        self.lru[idx] = slot
        self._set_slot(idx, slot)
        with self.lock:
            self.on_disk.discard(idx)

    def seed(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Bulk-ingest ``N`` chunks: the last ``ram_cap`` stay in RAM, the rest spill to disk once."""
        N = k.shape[2]
        self._ensure_slotmap(N)
        keep = min(N, self.ram_cap)
        disk_n = N - keep
        if disk_n > 0:
            self._open_files()
            fd = self._fd  # type: ignore[misc]
            # File layout is chunk-major with K and V adjacent ([cid] -> [K][V]); permute the input to
            # [cid, B, KVH, ...] and stack K,V on a new axis so the spilled prefix is one contiguous
            # write (offset 0 = chunk 0, K then V).
            perm = (2, 0, 1) + tuple(range(3, 3 + len(self.tail)))
            kb = k[:, :, :disk_n].permute(*perm)
            vb = v[:, :, :disk_n].permute(*perm)
            kv = torch.stack((kb, vb), dim=1).contiguous()  # [disk_n, 2, B, KVH, *tail]
            _pwrite_all(fd, _raw_bytes(kv), 0)
            _fadvise_dontneed(fd, 0, disk_n * self.cbytes)
            with self.lock:
                self.on_disk.update(range(disk_n))
        # Resident tail: chunks [disk_n, N).
        if self.cap < keep:
            self._grow_staging(min(self.ram_cap, max(self.cap, keep)))
        for cid in range(disk_n, N):
            slot = self._alloc_slot()
            self.sk[:, :, slot] = k[:, :, cid]
            self.sv[:, :, slot] = v[:, :, cid]
            self.id2slot[cid] = slot
            self.lru[cid] = slot
            self._set_slot(cid, slot)

    def reset(self) -> None:
        """Drop all RAM state and truncate this cache's spill file.

        The caller (the store) must drain the shared disk writer (``disk.flush()``) *before* this,
        so no in-flight spill closure can land in a file region after it is truncated.
        """
        if self._fd is not None:
            try:
                os.ftruncate(self._fd, 0)
            except OSError:
                pass
            if self.prealloc_chunks > 0:
                _fallocate(self._fd, self.prealloc_chunks * self.cbytes)
        self.cap = 0
        self.sk = self.sv = None
        self.id2slot.clear()
        self.lru.clear()
        self.free.clear()
        self._slotmap = None
        with self.lock:
            self.on_disk.clear()
            self.pending.clear()
        if self.prealloc_chunks > 0:
            # Restore the preallocated resident staging dropped above.
            self._grow_staging(min(self.ram_cap, self.prealloc_chunks))


class _FsLayerRecord:
    """The full cold record of one layer as **two independent** disk-backed RAM page caches:

    * a *group-summary* cache (``M·Dh`` per chunk) sized generously (``_fs_group_ram_cap``) so the
      per-step routing decision — which only reads summaries — stays RAM-resident even for long
      contexts; and
    * a *token* cache (``C·Dh`` per chunk) on a bounded budget (``_fs_token_ram_cap``) that spills
      to disk, since full token K/V is only needed for the handful of chunks whose groups are
      actually opened and attended.

    Splitting them is the whole point: ``prefetch`` + ``gather_group_summaries`` warm and read only
    the *group* cache, so scoring ``TOPK_CHUNKS`` routed chunks never loads their (much larger)
    token K/V from disk.
    """

    def __init__(self, store: "FsKVCacheStore", layer: int) -> None:
        self.s = store
        self.layer = layer
        C, M, Dh = store.C, store.M, store.dh
        prealloc = store._fs_prealloc_chunks
        self.groups = _PagedCache(store, layer, "g", (M, Dh), store._fs_group_ram_cap, prealloc)
        self.tokens = _PagedCache(store, layer, "t", (C, Dh), store._fs_token_ram_cap, prealloc)

    # -- id→slot capacity (max of the two caches; both grow with n_closed) -
    @property
    def capacity_ids(self) -> int:
        return max(self.groups.capacity_ids, self.tokens.capacity_ids)

    # -- staging views (kept for the store's VRAM-bank feeders & direct gathers) --
    @property
    def ram_gk(self) -> Optional[torch.Tensor]:
        return self.groups.sk

    @property
    def ram_gv(self) -> Optional[torch.Tensor]:
        return self.groups.sv

    @property
    def ram_tk(self) -> Optional[torch.Tensor]:
        return self.tokens.sk

    @property
    def ram_tv(self) -> Optional[torch.Tensor]:
        return self.tokens.sv

    # -- ingest ------------------------------------------------------------
    def append(self, idx: int, gk: torch.Tensor, gv: torch.Tensor,
               tk: torch.Tensor, tv: torch.Tensor) -> None:
        self.groups.append(idx, gk, gv)
        self.tokens.append(idx, tk, tv)

    def seed(self, gk: torch.Tensor, gv: torch.Tensor,
             tk: torch.Tensor, tv: torch.Tensor) -> None:
        self.groups.seed(gk, gv)
        self.tokens.seed(tk, tv)

    # -- residency (independent per tier) ---------------------------------
    def ensure_group_resident(self, unique_ids: List[int]) -> None:
        self.groups.ensure_resident(unique_ids)

    def ensure_token_resident(self, unique_ids: List[int]) -> None:
        self.tokens.ensure_resident(unique_ids)

    def group_slots_for(self, chunk_idx: torch.Tensor) -> torch.Tensor:
        return self.groups.slots_for(chunk_idx)

    def token_slots_for(self, chunk_idx: torch.Tensor) -> torch.Tensor:
        return self.tokens.slots_for(chunk_idx)

    def reset(self) -> None:
        self.groups.reset()
        self.tokens.reset()


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
        fs_group_ram_frac: float = 0.25,
        fs_cache_dir: Optional[str] = None,
        fs_writer_threads: int = 24,
        max_context_tokens: Optional[int] = None,
        host_safety_frac: float = 0.5,
        host_reserve_gb: float = 1.5,
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
        # --- Adaptive host-RAM safety clamp ------------------------------------------------------
        # The whole point of the fs tier is to keep host RAM *bounded* while the cold KV spills to
        # disk — but a configured ``ram_budget_gb`` larger than the machine can spare still drives the
        # box into the OOM killer (the cold record + the model weights' host overhead + the CUDA
        # context + everything else must coexist in physical RAM).  On a memory-tight host that is
        # exactly the "process killed at long context" failure.  So clamp the budget to a safe slice
        # of the RAM that is *actually available right now* (after the model has loaded): keep at most
        # ``host_safety_frac`` of available RAM, and never less than ``host_reserve_gb`` free on top.
        # On a roomy box the clamp is a no-op (available >> budget); on a small box it shrinks the
        # budget so more KV spills to disk (slower) instead of OOM-killing the process.  Override the
        # policy with ``KVR_RAM_BUDGET_GB`` (hard value) or disable via ``host_safety_frac<=0``.
        env_budget = os.environ.get("KVR_RAM_BUDGET_GB")
        if env_budget:
            self.ram_budget_gb = max(0.1, float(env_budget))
        elif host_safety_frac > 0:
            avail = _available_host_gb()
            if avail is not None:
                safe = max(0.25, avail * float(host_safety_frac) - float(host_reserve_gb))
                if self.ram_budget_gb > safe:
                    import sys as _sys
                    print(f"[kvr] FsKVCacheStore: clamping ram_budget {self.ram_budget_gb:.1f}GB -> "
                          f"{safe:.1f}GB to fit available host RAM ({avail:.1f}GB free); the excess "
                          f"KV will spill to disk. Override with KVR_RAM_BUDGET_GB.",
                          file=_sys.stderr, flush=True)
                    self.ram_budget_gb = safe
        # The RAM page cache is split into two *independent* tiers (group summaries vs. token K/V),
        # so the per-step routing decision (which only reads the tiny ``M·Dh`` group summaries) stays
        # RAM-resident even when the bulky ``C·Dh`` token K/V has to spill to disk.  ``ram_budget_gb``
        # is divided between them: ``fs_group_ram_frac`` to summaries, the rest to tokens.  Because a
        # summary is ~``C/M``× smaller than a token chunk, even a modest fraction holds far more
        # summary chunks than the same bytes of tokens — enough to keep *all* summaries resident for
        # realistic contexts while host memory stays bounded by ``ram_budget_gb``.
        gfrac = min(max(float(fs_group_ram_frac), 0.0), 0.9)
        budget = int(self.ram_budget_gb * 1024**3)
        group_pc = self.num_layers * self.B * self.kvh * self.M * self.dh * self._dtype_bytes * 2
        token_pc = self.num_layers * self.B * self.kvh * self.C * self.dh * self._dtype_bytes * 2
        group_budget = int(budget * gfrac)
        token_budget = budget - group_budget
        self._fs_group_ram_cap = max(1, group_budget // group_pc) if group_pc > 0 else 1
        self._fs_token_ram_cap = max(1, token_budget // token_pc) if token_pc > 0 else 1
        # Optional preallocation: when the target max context is known, reserve the per-layer spill
        # files (and pre-grow the resident staging) for that many chunks up front, so runtime spills
        # never pay file-extension / staging-realloc overhead.  ``0`` (default) = grow lazily.
        if max_context_tokens and max_context_tokens > 0:
            self._fs_prealloc_chunks = (int(max_context_tokens) + self.C - 1) // self.C
        else:
            self._fs_prealloc_chunks = 0
        self.disk = _FsDiskManager(root=fs_cache_dir, num_workers=fs_writer_threads)
        self._rec: Dict[int, _FsLayerRecord] = {}
        # Disk-read instrumentation (counts actual pread-from-disk fills, i.e. cold misses that
        # were not served from the in-RAM pending-write buffer).  Reset alongside the cache counters.
        self.disk_reads = 0
        self.disk_read_bytes = 0
        if os.environ.get("KVR_DEBUG"):
            import sys as _sys
            print(f"[kvr] FsKVCacheStore: ram_budget={self.ram_budget_gb}GB (group_frac={gfrac}) → "
                  f"group_cap={self._fs_group_ram_cap} chunks/layer, "
                  f"token_cap={self._fs_token_ram_cap} chunks/layer "
                  f"({self._fs_token_ram_cap * self.C} tok/layer); "
                  f"writers={self.disk._num_workers}, prealloc={self._fs_prealloc_chunks} chunks/layer; "
                  f"spill dir {self.disk.dir}",
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
        self.disk_reads = 0
        self.disk_read_bytes = 0
        # Drain the shared writer first so no in-flight spill closure can land in a file region
        # that a per-cache reset() is about to truncate (stale-write / corruption guard).
        self.disk.flush()
        for rec in self._rec.values():
            rec.reset()
        self._rec.clear()

    def close(self) -> None:
        """Release the writer thread(s), free the host staging, and delete all spill files (idempotent)."""
        # Drain any queued spills first so no in-flight write races the fd close below.
        self.disk.flush()
        self.disk.close()
        # Drop the per-layer host-RAM staging tensors and VRAM banks so the (multi-GB) cold record is
        # released immediately, not whenever GC happens to run — important when a session store is
        # replaced by a new one (prefix-miss / reset) on a memory-tight host.
        for rec in self._rec.values():
            rec.groups.sk = rec.groups.sv = None
            rec.tokens.sk = rec.tokens.sv = None
        self._rec.clear()
        self._bank_k.clear(); self._bank_v.clear()
        self._sbank_gk.clear(); self._sbank_gv.clear()
        self._layers.clear()

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
        rec.ensure_token_resident(unique)
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
            rslot = rec.tokens.id2slot[cid]
            bank_k[:, :, slot] = rec.ram_tk[:, :, rslot].to(dev)
            bank_v[:, :, slot] = rec.ram_tv[:, :, rslot].to(dev)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    def _load_missing_summaries(self, layer: int, st: _LayerStore, unique: List[int]) -> None:
        rec = self._rec_for(layer)
        rec.ensure_group_resident(unique)
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
            rslot = rec.groups.id2slot[cid]
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
            rslots = rec.group_slots_for(chunk_idx)
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
        rslots = rec.token_slots_for(chunk_idx)
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
            rslots = rec.token_slots_for(chunk_idx)
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
        rslots = rec.token_slots_for(chunk_idx)  # [B,H,Kg]
        tok = group_idx.detach().to("cpu").unsqueeze(-1) * gs + torch.arange(gs)
        slot_exp = rslots.unsqueeze(-1).expand_as(tok)
        b = torch.arange(self.B).view(self.B, 1, 1, 1)
        kv = (torch.arange(H) // rep).view(1, H, 1, 1)
        k = rec.ram_tk[b, kv, slot_exp, tok]
        v = rec.ram_tv[b, kv, slot_exp, tok]
        return k.to(dev), v.to(dev)

    # -- IO hint -----------------------------------------------------------
    def prefetch(self, layer: int, chunk_idx: torch.Tensor) -> None:
        """Warm *group-summary* residency for ``chunk_idx`` ahead of the routing gather.

        Routing scores only the group summaries of the candidate (routed-middle) chunks, so this
        loads just their tiny ``M·Dh`` summaries from disk — never their token K/V.  Token pages
        are pulled lazily (and only for the chunks whose groups are actually opened) by
        :meth:`gather_tokens` / :meth:`gather_chunk_tokens`.
        """
        rec = self._rec_for(layer)
        rec.ensure_group_resident(torch.unique(chunk_idx).tolist())
