"""RAM-tier KV-cache store: ``chunk_k`` + hot windows on GPU, the rest in pinned CPU memory.

Two independent bounded LRU VRAM caches accelerate the cold tier:
* a **group-summary cache** (``vram_summary_chunks``) for ``group_*`` — group-level routing only
  reads summaries (``M·Dh`` per chunk), so this cache is small per entry and sized to span the
  whole context, keeping the per-step routing decision GPU-resident; and
* a **token bank** (``vram_cache_chunks``) for ``token_*`` — only chunks whose groups are actually
  *opened* (and the always-resident windows) ever enter it.

Gradient contract
-----------------
The store **never detaches on the hot path**.  Freshly-closed chunks handed to
``append_closed_chunk`` are kept as the *live, grad-carrying* tensors the router just
computed (in the ``hot_first`` / ``hot_last`` live buffers).  Detaching happens at exactly
one place — when a chunk is evicted from the hot window in ``_offload``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from .chunk_placement_policy import ChunkPlacementPolicy
from .kv_cache_store import KVCacheStore
from .layer_store import _LayerStore


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
