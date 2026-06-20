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
Subclass ``KVCacheStore`` (or swap the CPU buffers in ``RamKVCacheStore`` for a
memory-mapped / paged backend).  The router depends only on the abstract interface:
``chunk_summaries``, ``gather_group_summaries``, ``gather_tokens``, the ``hot_*``
accessors, ``append_closed_chunk`` and ``prefetch``.  An NVMe store keeps ``chunk_k`` and
the hot windows identical (VRAM/RAM) and backs the cold ``token_*`` record with an
``mmap``'d, chunk-/group-contiguous file so an opened group is one sequential read; it
overrides ``prefetch`` to issue the async read and ``gather_tokens`` to consume it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
            free, _ = torch.cuda.mem_get_info(self.compute_device)
            budget = free - int(self.vram_cache_reserve_gb * 1024**3)
            per_chunk_all_layers = self.num_layers * self.B * self.kvh * self.C * self.dh * self._dtype_bytes * 2
            fit = int(budget // per_chunk_all_layers) if per_chunk_all_layers > 0 else 0
            cap = max(0, min(cap, fit))
        self._eff_cap = cap
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
            bank_k[:, :, slot] = st.cpu_token_k[:, :, cid].to(dev, non_blocking=self.pin_memory)
            bank_v[:, :, slot] = st.cpu_token_v[:, :, cid].to(dev, non_blocking=self.pin_memory)
            i2s[cid] = slot
            cached.add(cid)
            lru[cid] = slot

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
        gk = self._gather_record(st.cpu_group_k, chunk_idx, rep)
        gv = self._gather_record(st.cpu_group_v, chunk_idx, rep)
        return gk, gv

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
        st = self._layer(layer)
        B, KVH = st.cpu_token_k.shape[0], st.cpu_token_k.shape[1]
        H = chunk_idx.shape[1]
        rep = H // self.kvh
        gs = self.C // self.M
        dev = self.storage_device
        # token slice = group_idx * gs + [0..gs)
        tok = (group_idx.unsqueeze(-1) * gs + torch.arange(gs, device=group_idx.device))
        tok_cpu = tok.detach().to(dev)
        cidx_cpu = chunk_idx.detach().to(dev).unsqueeze(-1).expand_as(tok_cpu)
        b = torch.arange(B, device=dev).view(B, 1, *([1] * (tok_cpu.ndim - 2)))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * (tok_cpu.ndim - 2)))
        k = st.cpu_token_k[b, kv, cidx_cpu, tok_cpu]  # [B, H, *, gs, Dh]
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
