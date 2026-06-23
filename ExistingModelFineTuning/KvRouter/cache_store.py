"""Tiered KV-cache storage for hierarchical chunk-routed attention.

This module isolates *where the KV lives* (VRAM / RAM / — later — NVMe) and *how it
moves between tiers* from the routing logic (see ``chunk_router.ChunkRouter``).  The
router only ever asks the store for tensors on the compute device; the store decides
which tier serves them and pays the transfer cost.

Storage model (per attention layer, GQA / kv-head granularity)
--------------------------------------------------------------
For every *closed* chunk ``i`` we keep three kinds of artefact, at three temperatures:

================  ==========================  ============  ===================================
kind              shape (per chunk)           tier          why
================  ==========================  ============  ===================================
``chunk_k``       ``[B, KVH, Dh]``            HOT (VRAM)    routing scan table; scanned every
                                                            step, tiny (1 vec / 64 tokens)
``group_k/v``     ``[B, KVH, M, Dh]``         WARM (RAM)    fetched only for routed chunks
``token_k/v``     ``[B, KVH, C, Dh]``         COLD (RAM)    fetched only for *opened* groups
================  ==========================  ============  ===================================

``chunk_k`` is held resident on the compute device as one growing buffer, because the
per-step chunk-routing scan reads *all* of it.  ``group_*``/``token_*`` are held in a
complete CPU record (pinned, contiguous, chunk-major) and only the routed/opened slices
are pulled to VRAM.  This is exactly the "bounded working set" that makes 100K–1M
context viable on a memory-limited GPU (see ``gen_opt/OFFLOAD_ANALYSIS.md``).

Two independent bounded LRU VRAM caches accelerate the cold tier so consecutive steps copy
only *newly required* slices across PCIe:

* a **group-summary cache** (``vram_summary_chunks``) for ``group_*`` — group-level routing only
  reads summaries (``M·Dh`` per chunk), so this cache is small per entry and sized to span the
  whole context, keeping the per-step routing decision GPU-resident; and
* a **token bank** (``vram_cache_chunks``) for ``token_*`` — only chunks whose groups are actually
  *opened* (and the always-resident windows) ever enter it.

The two are slot-independent: a chunk can have its summary resident for routing without paying
for its full token K/V until it is opened.

Always-resident windows
-----------------------
Two contiguous spans of chunks are additionally kept *live on the compute device* by the
``ChunkPlacementPolicy``:

* the last ``keep_last`` closed chunks (the recently-closed context every query sees), and
* the first ``keep_first`` chunks (attention sinks),

with ``first_token_level`` choosing whether the first window is resident at token
granularity (full KV) or only as group summaries.

Gradient contract
-----------------
The store **never detaches on the hot path**.  Freshly-closed chunks handed to
``append_closed_chunk`` are kept as the *live, grad-carrying* tensors the router just
computed (in the ``hot_first`` / ``hot_last`` live buffers).  A detached copy is also
written to the CPU record so the data survives once the chunk leaves the hot window.
Detaching therefore happens at exactly one place — when a chunk is evicted from the hot
window in ``_offload`` — which is the *moment it becomes cache-only*.  Until then the
router can backprop into it.  (Routing scores are computed under ``no_grad`` in the
router, so ``chunk_k`` never needs grad and is stored detached.)

Extending to NVMe
-----------------
The NVMe/disk tier is implemented by :class:`FsKVCacheStore` (a subclass of
``RamKVCacheStore``): host RAM becomes a *bounded* LRU page cache (12 GB by default) over a
disk-backed full record, so the cold ``group_*``/``token_*`` record can exceed RAM.  ``chunk_k``
and the hot windows stay identical (VRAM/RAM).  Least-recently-used chunks spill to per-layer
files asynchronously (off the compute path) and are pulled back on demand; explicit
``pread``/``pwrite`` + ``posix_fadvise(DONTNEED)`` keep the kernel page cache small so it never
behaves like swap, and the spill files are removed on exit / signal / reset.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import threading
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import torch


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


@dataclass
class _LayerStore:
    """Backing tensors for one layer.  ``None`` until the first chunk is appended."""

    # HOT — resident routing table, detached. [B, KVH, cap, Dh], filled to n_closed.
    chunk_k: Optional[torch.Tensor] = None
    n_closed: int = 0

    # COLD/WARM — complete CPU record of every closed chunk (detached, pinned).
    cpu_group_k: Optional[torch.Tensor] = None   # [B, KVH, cap, M, Dh]
    cpu_group_v: Optional[torch.Tensor] = None
    cpu_token_k: Optional[torch.Tensor] = None   # [B, KVH, cap, C, Dh]
    cpu_token_v: Optional[torch.Tensor] = None

    # Live (grad-carrying) hot windows, keyed by absolute chunk id.
    live_group_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_group_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_token_k: Dict[int, torch.Tensor] = None  # type: ignore[assignment]
    live_token_v: Dict[int, torch.Tensor] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.live_group_k = {}
        self.live_group_v = {}
        self.live_token_k = {}
        self.live_token_v = {}


class KVCacheStore(ABC):
    """Abstract tiered store.  One instance serves all layers of a model."""

    def __init__(self, *, compute_device: torch.device, policy: ChunkPlacementPolicy) -> None:
        self.compute_device = compute_device
        self.policy = policy

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def num_closed_chunks(self, layer: int) -> int: ...

    # -- routing table (HOT) ----------------------------------------------
    @abstractmethod
    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        """``[B, KVH, n_closed, Dh]`` on the compute device (detached), or None."""

    # -- ingest ------------------------------------------------------------
    @abstractmethod
    def append_closed_chunk(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, Dh]
        group_k: torch.Tensor,   # [B, KVH, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, C, Dh]
    ) -> None:
        """Store a freshly-closed chunk.  Tensors may carry grad; see module docstring."""

    # -- fetch into VRAM ---------------------------------------------------
    @abstractmethod
    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group K/V summaries for ``chunk_idx[B, H, *]`` → ``[B, H, *, M, Dh]`` on device."""

    @abstractmethod
    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token K/V for matching ``(chunk_idx, group_idx)[B, H, *]`` → ``[B, H, *, gs, Dh]``."""

    # -- always-resident windows (grad-preserving) ------------------------
    @abstractmethod
    def hot_group_summaries(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Live group K/V for chunks ``[lo, hi)`` → ``[B, KVH, hi-lo, M, Dh]`` (grad kept)."""

    @abstractmethod
    def hot_tokens(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Live token K/V for chunks ``[lo, hi)`` → ``[B, KVH, hi-lo, C, Dh]`` (grad kept)."""

    # -- IO hint (overridden by NVMe backends) ----------------------------
    def prefetch(self, layer: int, chunk_idx: torch.Tensor) -> None:  # noqa: B027 - optional hook
        """Best-effort async hint that ``chunk_idx`` will be gathered soon.  No-op for RAM."""


class RamKVCacheStore(KVCacheStore):
    """Naive RAM tier: ``chunk_k`` + hot windows on GPU, the rest in pinned CPU memory.

    The growing buffers double in capacity to keep amortised append cost O(1).  This is
    the reference implementation; an NVMe variant only needs to replace the
    ``cpu_token_*`` record (the bulk) with a paged/mmap backend.
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
        pin_memory: bool = True,
        initial_capacity: int = 64,
        storage_device: Optional[torch.device] = None,
        vram_cache_chunks: int = 0,
        vram_summary_chunks: int = 0,
        num_layers: int = 0,
        vram_cache_reserve_gb: float = 1.5,
    ) -> None:
        super().__init__(compute_device=compute_device, policy=policy)
        self.kvh = kv_heads
        self.dh = head_dim
        self.C = chunk_size
        self.M = groups_per_chunk
        self.B = batch_size
        self.dtype = dtype
        # Bounded LRU VRAM cache of token-level chunk K/V (per layer), so that the cold-tier
        # gather only copies *newly required* chunks across PCIe each step; chunks selected by
        # consecutive tokens (always the sink/local windows, and the slowly-changing routed
        # middle) stay resident.  0 disables it.  No effect for the VRAM tier (record already
        # on the compute device).
        #
        # ``vram_cache_chunks`` is an *upper bound* per layer; the actual bank is auto-sized to
        # fit the free VRAM at the moment routing first kicks in (see ``_effective_cap``).  This
        # is what keeps a long-context prefill from OOMing on a memory-tight card: the bank is a
        # speed optimisation, so if it would not fit we shrink it (or disable it entirely and
        # fall back to gathering straight from the RAM record) rather than crash.
        self.vram_cache_chunks = int(vram_cache_chunks)
        self.num_layers = int(num_layers)  # used to budget the per-layer bank across all layers
        self.vram_cache_reserve_gb = float(vram_cache_reserve_gb)  # VRAM left free for activations
        self._dtype_bytes = torch.empty((), dtype=dtype).element_size()
        self._eff_cap: Optional[int] = None  # resolved per-layer bank capacity (lazy, from free VRAM)
        self._bank_k: Dict[int, torch.Tensor] = {}
        self._bank_v: Dict[int, torch.Tensor] = {}
        self._id2slot: Dict[int, torch.Tensor] = {}
        self._lru: Dict[int, "OrderedDict[int, int]"] = {}
        self._cached: Dict[int, set] = {}
        self._free: Dict[int, List[int]] = {}
        self._i2s_cap: Dict[int, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        # ---- Independent group-summary VRAM cache (the routing accelerator) -------------------
        # Group routing (the per-step group-level decision) only needs the *group summaries* of
        # the routed-middle chunks (M·Dh per chunk), NOT their full token K/V (C·Dh, here 16×
        # bigger).  Serving summaries from the token bank therefore forced a whole-chunk PCIe copy
        # for every routed-middle chunk just to read its tiny summary — the dominant cold-tier
        # traffic, since routed-middle chunks churn every step while only a few are actually opened.
        #
        # This separate, slot-independent LRU cache holds *only* group summaries, so at the same
        # VRAM budget it keeps ~C/M× more chunks resident than the token bank.  Sized large enough
        # (``vram_summary_chunks``) it covers the whole context → group routing becomes effectively
        # GPU-resident (≈0 misses), like the chunk-level scan, and the token bank then only pays for
        # the handful of chunks whose groups are actually *opened* and attended.
        self.vram_summary_chunks = int(vram_summary_chunks)
        self._eff_scap: Optional[int] = None
        self._sbank_gk: Dict[int, torch.Tensor] = {}
        self._sbank_gv: Dict[int, torch.Tensor] = {}
        self._sid2slot: Dict[int, torch.Tensor] = {}
        self._slru: Dict[int, "OrderedDict[int, int]"] = {}
        self._scached: Dict[int, set] = {}
        self._sfree: Dict[int, List[int]] = {}
        self._si2s_cap: Dict[int, int] = {}
        self.summary_hits = 0
        self.summary_misses = 0
        # Where the cold (group/token) record lives.  RAM tier => CPU; VRAM tier =>
        # the compute device (used by the training cache).  ``chunk_k`` and the hot
        # windows are always on the compute device regardless.
        self.storage_device = torch.device(storage_device) if storage_device is not None else torch.device("cpu")
        # Pinning + non-blocking H2D only make sense for a CPU record feeding a CUDA GPU.
        self.pin_memory = (
            pin_memory and self.storage_device.type == "cpu" and compute_device.type == "cuda"
        )
        self._init_cap = initial_capacity
        self._layers: Dict[int, _LayerStore] = {}

    # -- helpers -----------------------------------------------------------
    def _layer(self, layer: int) -> _LayerStore:
        st = self._layers.get(layer)
        if st is None:
            st = _LayerStore()
            self._layers[layer] = st
        return st

    def _new_cpu(self, *shape: int) -> torch.Tensor:
        # Allocates a slab of the cold record on the configured storage tier.  Named
        # ``_new_cpu`` for history; the device is ``self.storage_device`` (CPU or VRAM).
        t = torch.empty(*shape, dtype=self.dtype, device=self.storage_device)
        if self.pin_memory:
            t = t.pin_memory()
        return t

    def _ensure_capacity(self, st: _LayerStore, need: int) -> None:
        cur = 0 if st.chunk_k is None else st.chunk_k.shape[2]
        if need <= cur:
            return
        cap = max(self._init_cap, cur * 2, need)
        B, KVH, Dh, C, M = self.B, self.kvh, self.dh, self.C, self.M

        ck = torch.zeros(B, KVH, cap, Dh, device=self.compute_device, dtype=self.dtype)
        gk = self._new_cpu(B, KVH, cap, M, Dh)
        gv = self._new_cpu(B, KVH, cap, M, Dh)
        tk = self._new_cpu(B, KVH, cap, C, Dh)
        tv = self._new_cpu(B, KVH, cap, C, Dh)
        if st.chunk_k is not None:
            n = st.n_closed
            ck[:, :, :n] = st.chunk_k[:, :, :n]
            gk[:, :, :n] = st.cpu_group_k[:, :, :n]
            gv[:, :, :n] = st.cpu_group_v[:, :, :n]
            tk[:, :, :n] = st.cpu_token_k[:, :, :n]
            tv[:, :, :n] = st.cpu_token_v[:, :, :n]
        st.chunk_k, st.cpu_group_k, st.cpu_group_v, st.cpu_token_k, st.cpu_token_v = ck, gk, gv, tk, tv

    def _offload(self, st: _LayerStore, chunk_id: int) -> None:
        """Drop a chunk's live (grad) copy; it survives only as the detached CPU record.

        This is the single detach point.  Called when a chunk leaves the hot window.
        """
        st.live_group_k.pop(chunk_id, None)
        st.live_group_v.pop(chunk_id, None)
        st.live_token_k.pop(chunk_id, None)
        st.live_token_v.pop(chunk_id, None)

    def _evict_outside_windows(self, st: _LayerStore) -> None:
        n = st.n_closed
        f_lo, f_hi = self.policy.hot_first_range(n)
        l_lo, l_hi = self.policy.hot_last_range(n)
        keep = set(range(f_lo, f_hi)) | set(range(l_lo, l_hi))
        for cid in list(st.live_token_k.keys() | st.live_group_k.keys()):
            if cid not in keep:
                self._offload(st, cid)
        # First window beyond token-level granularity keeps only summaries live.
        if not self.policy.first_token_level:
            for cid in range(f_lo, f_hi):
                st.live_token_k.pop(cid, None)
                st.live_token_v.pop(cid, None)

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._layers.clear()
        self._bank_k.clear(); self._bank_v.clear(); self._id2slot.clear()
        self._lru.clear(); self._cached.clear(); self._free.clear(); self._i2s_cap.clear()
        self._eff_cap = None
        self.cache_hits = 0
        self.cache_misses = 0
        # Independent group-summary cache.
        self._sbank_gk.clear(); self._sbank_gv.clear(); self._sid2slot.clear()
        self._slru.clear(); self._scached.clear(); self._sfree.clear(); self._si2s_cap.clear()
        self._eff_scap = None
        self.summary_hits = 0
        self.summary_misses = 0

    # -- VRAM-aware bank sizing --------------------------------------------
    def _effective_cap(self) -> int:
        """Per-layer VRAM-bank capacity (chunks), resolved once from free VRAM.

        The bank costs ``num_layers * B * KVH * cap * C * Dh * dtype_bytes * 2`` (K+V) of VRAM.
        We size ``cap`` so that the whole bank fits in the currently-free VRAM minus a reserve
        kept for activations.  If even one chunk per layer will not fit, returns 0 — the gather
        then streams straight from the RAM record (slower, but bounded and never OOM).
        """
        if self._eff_cap is not None:
            return self._eff_cap
        cap = self.vram_cache_chunks
        if cap > 0 and self.num_layers > 0 and self.compute_device.type == "cuda":
            # Size off the TRUE driver-free memory (mem_get_info), NOT total-minus-allocated:
            # the latter ignores the CUDA context / non-torch overhead and, under inference_mode,
            # the freed-but-uncounted activation, so it overestimates the headroom and grows the
            # bank until a later activation peak OOMs.  Driver-free reflects the caching allocator's
            # reserved high-water (≈ weights + peak activation so far), so on a memory-tight card it
            # correctly shrinks the bank toward 0 rather than crowding out activations.
            free, _ = torch.cuda.mem_get_info(self.compute_device)
            budget = free - int(self.vram_cache_reserve_gb * 1024**3)
            # Per resident chunk: token K+V (2·C·Dh), all layers.  Group summaries live in the
            # separate summary cache now, so the token bank only pays for the chunks it opens.
            per_chunk_all_layers = (
                self.num_layers * self.B * self.kvh * self.C * self.dh * self._dtype_bytes * 2
            )
            fit = int(budget // per_chunk_all_layers) if per_chunk_all_layers > 0 else 0
            cap = max(0, min(cap, fit))
            import os as _os
            if _os.environ.get("KVR_DEBUG"):
                import sys as _sys
                print(f"[kvr] _effective_cap: free={free/1e9:.2f}GB reserve={self.vram_cache_reserve_gb}GB "
                      f"budget={budget/1e9:.2f}GB per_chunk={per_chunk_all_layers/1e6:.2f}MB "
                      f"fit={fit} -> cap={cap}", file=_sys.stderr, flush=True)
        self._eff_cap = cap
        return cap

    def _effective_summary_cap(self) -> int:
        """Per-layer capacity (chunks) of the independent group-summary VRAM cache.

        Resolved once from free VRAM, like :meth:`_effective_cap`, but with the *summary* per-chunk
        cost (``M·Dh`` instead of ``C·Dh``) — so for the same VRAM it holds ``C/M`` × more chunks.
        Set ``vram_summary_chunks`` large enough to span the whole context and group routing sees
        ≈0 cold-tier misses.  Returns 0 to disable (then summaries stream straight from the record).
        """
        if self._eff_scap is not None:
            return self._eff_scap
        cap = self.vram_summary_chunks
        if cap > 0 and self.num_layers > 0 and self.compute_device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.compute_device)
            budget = free - int(self.vram_cache_reserve_gb * 1024**3)
            # Per resident chunk: group-summary K+V (2·M·Dh), all layers — ~C/M× cheaper than a
            # token chunk, so a few thousand summaries cost only a few hundred MB.
            per_chunk_all_layers = (
                self.num_layers * self.B * self.kvh * self.M * self.dh * self._dtype_bytes * 2
            )
            fit = int(budget // per_chunk_all_layers) if per_chunk_all_layers > 0 else 0
            cap = max(0, min(cap, fit))
            import os as _os
            if _os.environ.get("KVR_DEBUG"):
                import sys as _sys
                print(f"[kvr] _effective_summary_cap: free={free/1e9:.2f}GB budget={budget/1e9:.2f}GB "
                      f"per_chunk={per_chunk_all_layers/1e6:.3f}MB fit={fit} -> scap={cap}",
                      file=_sys.stderr, flush=True)
        self._eff_scap = cap
        return cap

    # -- bounded LRU VRAM chunk cache (cold-tier gather accelerator) --------
    def _ensure_bank(self, layer: int, st: _LayerStore) -> None:
        cap = self._effective_cap()
        rec_cap = st.cpu_token_k.shape[2]
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
            # Record grew (capacity doubled): extend the id→slot map in place, preserving the
            # resident bank and its LRU state — no need to re-copy chunks across PCIe.
            old = self._id2slot[layer]
            grown = torch.full((rec_cap,), -1, dtype=torch.long, device=dev)
            grown[: old.shape[0]] = old
            self._id2slot[layer] = grown
            self._i2s_cap[layer] = rec_cap

    def _load_missing(self, layer: int, st: _LayerStore, unique: List[int]) -> None:
        """Make every chunk id in ``unique`` resident in the layer's VRAM bank (LRU eviction)."""
        i2s = self._id2slot[layer]
        lru = self._lru[layer]
        cached = self._cached[layer]
        free = self._free[layer]
        bank_k, bank_v = self._bank_k[layer], self._bank_v[layer]
        dev = self.compute_device
        # Touch hits first so they are never evicted to make room for this step's misses.
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
                old_id, slot = lru.popitem(last=False)  # oldest, guaranteed not in this request
                cached.discard(old_id)
                i2s[old_id] = -1
            # One PCIe copy per newly-required chunk brings its token K/V resident at the slot.
            bank_k[:, :, slot] = st.cpu_token_k[:, :, cid].to(dev, non_blocking=self.pin_memory)
            bank_v[:, :, slot] = st.cpu_token_v[:, :, cid].to(dev, non_blocking=self.pin_memory)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    # -- bounded LRU VRAM group-summary cache (routing accelerator) ---------
    def _ensure_summary_bank(self, layer: int, st: _LayerStore) -> None:
        cap = self._effective_summary_cap()
        rec_cap = st.cpu_group_k.shape[2]
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

    def _load_missing_summaries(self, layer: int, st: _LayerStore, unique: List[int]) -> None:
        """Make every chunk id in ``unique`` resident in the group-summary cache (LRU eviction).

        Copies only the chunk's group K/V (``M·Dh``) per miss — never its token K/V — so routing
        churn costs ~C/M× less PCIe than the token bank.
        """
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
            bank_gk[:, :, slot] = st.cpu_group_k[:, :, cid].to(dev, non_blocking=self.pin_memory)
            bank_gv[:, :, slot] = st.cpu_group_v[:, :, cid].to(dev, non_blocking=self.pin_memory)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

    def _summary_slots(self, layer: int, st: _LayerStore, chunk_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Make every chunk in ``chunk_idx`` resident in the summary cache; return its slots (same
        shape) on the compute device, or ``None`` to stream straight from the RAM record (cache off
        / won't fit / this step needs more distinct chunks than the cache holds)."""
        cap = self._effective_summary_cap()
        unique = torch.unique(chunk_idx).tolist()
        if cap <= 0 or len(unique) > cap:
            return None
        self._ensure_summary_bank(layer, st)
        self._load_missing_summaries(layer, st, unique)
        return self._sid2slot[layer][chunk_idx.detach().to(self.compute_device)]

    def _bank_slots(self, layer: int, st: _LayerStore, chunk_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Make every chunk in ``chunk_idx`` resident; return its slots (same shape) on the compute
        device, or ``None`` to signal the caller to stream straight from the RAM record (cache off /
        won't fit / this step needs more distinct chunks than the bank holds)."""
        cap = self._effective_cap()
        unique = torch.unique(chunk_idx).tolist()
        if cap <= 0 or len(unique) > cap:
            return None
        self._ensure_bank(layer, st)
        self._load_missing(layer, st, unique)
        return self._id2slot[layer][chunk_idx.detach().to(self.compute_device)]

    def gather_chunk_tokens_kvh(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """KV-head-granular full-chunk token K/V for ``chunk_idx[B, KVH, K]`` → ``[B, KVH, K, C, Dh]``.

        Routed at KV-head granularity (the ``rep`` query heads of a group share KV), so the
        per-step working set is small and stable — ideal for the LRU VRAM cache, which keeps
        recurring chunks resident and copies only newly-required ones from the cold tier.
        """
        st = self._layer(layer)
        B, KVH, K = chunk_idx.shape
        dev = self.compute_device

        def _direct(idx_dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
            idx = chunk_idx.to(idx_dev)
            b = torch.arange(B, device=idx_dev).view(B, 1, 1)
            g = torch.arange(KVH, device=idx_dev).view(1, KVH, 1)
            k = st.cpu_token_k[b, g, idx]
            v = st.cpu_token_v[b, g, idx]
            if idx_dev != dev:
                k = k.to(dev, non_blocking=self.pin_memory)
                v = v.to(dev, non_blocking=self.pin_memory)
            return k, v

        # VRAM tier (record already on compute device) → plain same-device index, no cache.
        if self.storage_device == self.compute_device:
            return _direct(dev)

        unique = torch.unique(chunk_idx).tolist()
        # Cache disabled / won't fit VRAM, or this step needs more distinct chunks than the bank
        # holds → bypass and stream straight from the RAM record (bounded, never OOM).
        cap = self._effective_cap()
        if cap <= 0 or len(unique) > cap:
            return _direct(self.storage_device)

        self._ensure_bank(layer, st)
        self._load_missing(layer, st, unique)
        slots = self._id2slot[layer][chunk_idx]  # [B,KVH,K] on compute device
        bank_k, bank_v = self._bank_k[layer], self._bank_v[layer]
        b = torch.arange(B, device=dev).view(B, 1, 1)
        g = torch.arange(KVH, device=dev).view(1, KVH, 1)
        return bank_k[b, g, slots], bank_v[b, g, slots]

    def num_closed_chunks(self, layer: int) -> int:
        return self._layer(layer).n_closed

    # -- HOT routing table -------------------------------------------------
    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        st = self._layer(layer)
        if st.chunk_k is None or st.n_closed == 0:
            return None
        return st.chunk_k[:, :, : st.n_closed]

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

        # HOT routing table: detached copy (routing is non-differentiable).
        st.chunk_k[:, :, idx] = chunk_k.detach()

        # Complete CPU record: detached copy (survives eviction).  non_blocking is safe
        # only into pinned memory; values are needed lazily so correctness holds either way.
        st.cpu_group_k[:, :, idx].copy_(group_k.detach(), non_blocking=self.pin_memory)
        st.cpu_group_v[:, :, idx].copy_(group_v.detach(), non_blocking=self.pin_memory)
        st.cpu_token_k[:, :, idx].copy_(token_k.detach(), non_blocking=self.pin_memory)
        st.cpu_token_v[:, :, idx].copy_(token_v.detach(), non_blocking=self.pin_memory)

        # Live grad-carrying copies for the hot windows (router keeps gradients here).
        st.live_group_k[idx] = group_k
        st.live_group_v[idx] = group_v
        st.live_token_k[idx] = token_k
        st.live_token_v[idx] = token_v

        st.n_closed = idx + 1
        self._evict_outside_windows(st)

    def seed_closed_chunks(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, N, Dh]
        group_k: torch.Tensor,   # [B, KVH, N, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, N, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, N, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, N, C, Dh]
    ) -> None:
        """Bulk-ingest ``N`` freshly-closed chunks in one shot (the vectorized prefill seed).

        Equivalent to ``N`` back-to-back :meth:`append_closed_chunk` calls but without the
        per-chunk Python loop / per-chunk eviction — one slab copy into the record plus a
        single hot-window setup, so prefill stays as fast as the dense reference.
        """
        st = self._layer(layer)
        N = chunk_k.shape[2]
        if N == 0:
            return
        self._ensure_capacity(st, N)

        st.chunk_k[:, :, :N] = chunk_k.detach()
        st.cpu_group_k[:, :, :N].copy_(group_k.detach(), non_blocking=self.pin_memory)
        st.cpu_group_v[:, :, :N].copy_(group_v.detach(), non_blocking=self.pin_memory)
        st.cpu_token_k[:, :, :N].copy_(token_k.detach(), non_blocking=self.pin_memory)
        st.cpu_token_v[:, :, :N].copy_(token_v.detach(), non_blocking=self.pin_memory)
        st.n_closed = N

        # Rebuild the live (hot-window) buffers directly, mirroring _evict_outside_windows:
        # both windows keep group summaries; the last window (and, if configured, the first)
        # additionally keep token-level KV.
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

    # -- fetch into VRAM ---------------------------------------------------
    def _gather_record(
        self, cpu: torch.Tensor, chunk_idx: torch.Tensor, rep: int
    ) -> torch.Tensor:
        """Gather ``cpu[B, KVH, cap, *tail]`` by ``chunk_idx[B, H, *]`` → ``[B, H, *, tail]`` on device.

        Indexing happens on the CPU record so only the routed slices cross PCIe (one
        async copy).  ``rep = H // KVH`` maps each query head to its kv-head.
        """
        B, KVH = cpu.shape[0], cpu.shape[1]
        tail = cpu.shape[3:]
        H = chunk_idx.shape[1]
        dev = self.storage_device
        idx_cpu = chunk_idx.detach().to(dev)
        b = torch.arange(B, device=dev).view(B, 1, *([1] * (chunk_idx.ndim - 2)))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * (chunk_idx.ndim - 2)))
        gathered = cpu[b, kv, idx_cpu]  # [B, H, *, tail]
        return gathered.to(self.compute_device, non_blocking=self.pin_memory)

    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        slots = None if self.storage_device == self.compute_device else self._summary_slots(layer, st, chunk_idx)
        if slots is None:  # VRAM tier or cache bypass → direct record index (M·Dh per chunk, no cache)
            return (self._gather_record(st.cpu_group_k, chunk_idx, rep),
                    self._gather_record(st.cpu_group_v, chunk_idx, rep))
        dev = self.compute_device
        H, nd = chunk_idx.shape[1], chunk_idx.ndim
        b = torch.arange(self.B, device=dev).view(self.B, *([1] * (nd - 1)))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * (nd - 2)))
        return self._sbank_gk[layer][b, kv, slots], self._sbank_gv[layer][b, kv, slots]

    def gather_chunk_tokens(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-chunk token K/V for ``chunk_idx[B, H, *]`` → ``[B, H, *, C, Dh]`` on device.

        Like :meth:`gather_tokens` but pulls the *whole* chunk (all ``C`` tokens) rather than a
        single group — used by the exact (non-summary) routed attention, which attends over the
        actual token KV of every selected chunk.  ``token_k`` holds the RoPE-applied keys, so the
        gathered K already carries each token's absolute-position rotation.
        """
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        k = self._gather_record(st.cpu_token_k, chunk_idx, rep)
        v = self._gather_record(st.cpu_token_v, chunk_idx, rep)
        return k, v

    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opened-group token K/V for ``(chunk_idx, group_idx)[B, H, Kg]`` → ``[B, H, Kg, gs, Dh]``."""
        st = self._layer(layer)
        H = chunk_idx.shape[1]
        rep = H // self.kvh
        gs = self.C // self.M
        dev = self.compute_device
        slots = None if self.storage_device == self.compute_device else self._bank_slots(layer, st, chunk_idx)
        if slots is not None:  # bank tier: slice the gs opened tokens out of the resident chunk
            tok = group_idx.detach().to(dev).unsqueeze(-1) * gs + torch.arange(gs, device=dev)  # [B,H,Kg,gs]
            slot_exp = slots.unsqueeze(-1).expand_as(tok)
            b = torch.arange(self.B, device=dev).view(self.B, 1, 1, 1)
            kv = (torch.arange(H, device=dev) // rep).view(1, H, 1, 1)
            return self._bank_k[layer][b, kv, slot_exp, tok], self._bank_v[layer][b, kv, slot_exp, tok]
        # VRAM tier or bank bypass → direct record index (one PCIe copy of just the opened slices).
        sd = self.storage_device
        B = st.cpu_token_k.shape[0]
        tok = (group_idx.unsqueeze(-1) * gs + torch.arange(gs, device=group_idx.device))
        tok_cpu = tok.detach().to(sd)
        cidx_cpu = chunk_idx.detach().to(sd).unsqueeze(-1).expand_as(tok_cpu)
        b = torch.arange(B, device=sd).view(B, 1, *([1] * (tok_cpu.ndim - 2)))
        kv = (torch.arange(H, device=sd) // rep).view(1, H, *([1] * (tok_cpu.ndim - 2)))
        k = st.cpu_token_k[b, kv, cidx_cpu, tok_cpu]  # [B, H, Kg, gs, Dh]
        v = st.cpu_token_v[b, kv, cidx_cpu, tok_cpu]
        return (
            k.to(self.compute_device, non_blocking=self.pin_memory),
            v.to(self.compute_device, non_blocking=self.pin_memory),
        )

    # -- always-resident windows ------------------------------------------
    def _stack_live(self, live: Dict[int, torch.Tensor], lo: int, hi: int) -> torch.Tensor:
        return torch.stack([live[c] for c in range(lo, hi)], dim=2)

    def hot_group_summaries(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        return self._stack_live(st.live_group_k, lo, hi), self._stack_live(st.live_group_v, lo, hi)

    def hot_tokens(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        return self._stack_live(st.live_token_k, lo, hi), self._stack_live(st.live_token_v, lo, hi)


class VramKVCacheStore(RamKVCacheStore):
    """All-VRAM tier: the cold group/token record lives on the compute device too.

    This is the cache used for **training** (and for ``cache_location="vram"`` generation):
    nothing is offloaded to host RAM, so gathers are a same-device index with no PCIe
    copy — the fastest option when the whole context fits in VRAM.  It is a thin
    specialisation of :class:`RamKVCacheStore` with ``storage_device == compute_device``
    and no pinning.
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
            storage_device=compute_device,
        )


# =====================================================================================
# NVMe / filesystem tier
# =====================================================================================
#
# The RAM tier (:class:`RamKVCacheStore`) holds the *complete* cold record in host memory.
# For very long contexts that record outgrows RAM, so :class:`FsKVCacheStore` adds a third
# tier *below* RAM: host RAM becomes a **bounded LRU page cache** (12 GB by default) over a
# disk-backed full record, exactly mirroring how the VRAM banks are a bounded cache over the
# RAM record.  The tiering is therefore recursive:
#
#     VRAM banks   (bounded)  ── cache of ──▶  RAM staging
#     RAM staging  (bounded)  ── cache of ──▶  disk files   (the full record)
#
# Design goals, and how they are met:
#
# * **Never behave like a swap file.**  We do *not* mmap a giant file and let the kernel page it
#   in/out (that creates global memory pressure and makes the whole machine hang).  Instead we use
#   explicit ``pread``/``pwrite`` on our own files, keep our RAM use hard-bounded, and call
#   ``posix_fadvise(POSIX_FADV_DONTNEED)`` after every read/write so our file data never lingers in
#   the kernel page cache.  The OS therefore sees no memory pressure from us and stays responsive
#   for other applications.
# * **Non-blocking eviction.**  Spilling an evicted chunk to disk happens on a background writer
#   thread (the heavy ``pwrite`` runs there, off the compute path).  The main thread only pays a
#   cheap RAM→RAM clone for the hand-off, so a long context never stalls the GPU on disk writes.
# * **No leftover files.**  All files live in a private temp directory that is removed on normal
#   exit (``atexit``), on a fatal signal (SIGINT/SIGTERM/SIGHUP — chained to the previous handler),
#   and when the cache is reset/closed.  Resetting the store deletes every spilled file.


def _raw_bytes(t: torch.Tensor) -> memoryview:
    """Read-only ``memoryview`` of a *contiguous* CPU tensor's raw bytes (dtype-agnostic).

    ``numpy()`` rejects bf16, so we reinterpret as ``uint8`` first; the view shares storage, so
    no copy is made.
    """
    return memoryview(t.contiguous().view(torch.uint8).numpy())


def _pwrite_all(fd: int, mv: memoryview, offset: int) -> None:
    n, total = 0, len(mv)
    while n < total:
        n += os.pwrite(fd, mv[n:], offset + n)


def _pread_all(fd: int, mv: memoryview, offset: int) -> None:
    n, total = 0, len(mv)
    while n < total:
        r = os.preadv(fd, [mv[n:]], offset + n)
        if r == 0:
            raise EOFError(f"short read at offset {offset + n} (wanted {total - n} more bytes)")
        n += r


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    """Tell the kernel we will not reuse this file region soon → drop it from the page cache.

    This is the key to *not* behaving like swap: without it, every byte we write/read would sit in
    the kernel page cache as reclaimable/dirty memory and, at long contexts, balloon into the very
    system-wide memory pressure that makes a machine unresponsive.  Best-effort: silently ignored on
    platforms / filesystems that do not support it.
    """
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


class _FsDiskManager:
    """Shared owner of the on-disk spill area: a private temp dir, fds, and the writer thread.

    One instance is shared by all layers of a store.  It guarantees the spill files are removed on
    interpreter exit and on fatal signals, so the application never leaves temporary files behind.
    """

    _registry: "weakref.WeakSet[_FsDiskManager]" = weakref.WeakSet()
    _handlers_installed = False
    _orig_handlers: Dict[int, object] = {}
    _reg_lock = threading.Lock()

    def __init__(self, *, root: Optional[str] = None, max_pending: int = 64) -> None:
        # NOTE: the spill dir must be on *real disk* (NVMe/SSD), never on a tmpfs like ``/tmp`` on
        # many distros — spilling onto tmpfs would put the "disk" tier back in RAM and reintroduce
        # exactly the swap-style memory pressure we are avoiding.  Default: a hidden dir in the cwd.
        if root is None:
            root = os.environ.get("KVR_FS_CACHE_DIR") or os.path.join(os.getcwd(), ".kvr_fscache")
        os.makedirs(root, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="kvr_fscache_", dir=root)
        self._fds: List[int] = []
        self._fd_lock = threading.Lock()
        self._queue: "Queue[Optional[Callable[[], None]]]" = Queue(maxsize=max_pending)
        self._closed = False
        self._thread = threading.Thread(target=self._writer_loop, name="kvr-fs-writer", daemon=True)
        self._thread.start()
        self._raise_fd_limit()
        with _FsDiskManager._reg_lock:
            _FsDiskManager._registry.add(self)
            self._install_handlers()

    # -- file handles ------------------------------------------------------
    def open_file(self, name: str) -> int:
        path = os.path.join(self.dir, name)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        with self._fd_lock:
            self._fds.append(fd)
        return fd

    @staticmethod
    def _raise_fd_limit() -> None:
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            want = min(hard, max(soft, 8192))
            if want > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except Exception:
            pass

    # -- async writes ------------------------------------------------------
    def submit(self, fn: Callable[[], None]) -> None:
        """Queue a spill closure for the writer thread (blocks only if the queue is full = backpressure)."""
        if self._closed:
            fn()  # drain synchronously during shutdown
            return
        self._queue.put(fn)

    def flush(self) -> None:
        self._queue.join()

    def _writer_loop(self) -> None:
        while True:
            fn = self._queue.get()
            try:
                if fn is None:
                    return
                fn()
            except Exception:  # pragma: no cover - a spill failure must not kill the writer
                import traceback
                import sys

                traceback.print_exc(file=sys.stderr)
            finally:
                self._queue.task_done()

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        with self._fd_lock:
            if self._closed:
                return
            self._closed = True
            fds = list(self._fds)
            self._fds.clear()
        try:
            self._queue.put(None)
            self._thread.join(timeout=5)
        except Exception:
            pass
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)
        _FsDiskManager._registry.discard(self)

    # -- exit / signal cleanup --------------------------------------------
    @classmethod
    def _install_handlers(cls) -> None:
        if cls._handlers_installed:
            return
        cls._handlers_installed = True
        atexit.register(cls._cleanup_all)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                cls._orig_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, cls._signal_handler)
            except (ValueError, OSError):
                # Not on the main thread, or signal unsupported on this platform → skip; atexit
                # still covers the normal-exit case.
                pass

    @classmethod
    def _cleanup_all(cls) -> None:
        for mgr in list(cls._registry):
            mgr.close()

    @classmethod
    def _signal_handler(cls, signum, frame):  # noqa: ANN001
        cls._cleanup_all()
        prev = cls._orig_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            # Restore the default action and re-raise so the process terminates as expected.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # prev == SIG_IGN → swallow.


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
