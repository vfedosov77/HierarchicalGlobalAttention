"""Hierarchical chunk/group router for offloaded KV-cache attention.

``ChunkRouter`` is the selection + assembly engine factored out of
``HierarchicalGlobalAttention._decode_forward``.  It owns *no* storage — all KV lives in
a :class:`~kv_router.cache_store.KVCacheStore` — and it owns *no* projections — the caller
hands it post-projection Q/K/V.  Its single responsibility is:

    given the Q (and freshly projected K/V) of new tokens, decide *what each token may
    attend to*, pull exactly that KV from the store into VRAM, and return it together with
    the attention mask — keeping gradients on the live (not-yet-evicted) KV.

Two-level routing (mirrors the training/decode reference)
---------------------------------------------------------
1. **Chunk level.** Score the new tokens against the resident ``chunk_k`` table and pick
   the top-``topk_chunks`` previous chunks.  The first ``keep_first`` and last
   ``keep_last`` chunks are *always* exposed (configurable) and are excluded from the
   top-k candidate pool so they are never double-counted.
2. **Group level.** Score the routed chunks' group summaries and open the top-``topk_groups``
   groups to exact token KV.

Output segments (concatenated along the key axis, each with its own visibility mask)
------------------------------------------------------------------------------------
* first-window  : token KV  *or* group summaries (``first_token_level``)
* last-window   : token KV (always token-level — the local context)
* routed-middle : group summaries of top-k chunks
* opened        : exact token KV of the top-k opened groups (from routed-middle)
* current-grp   : completed group summaries of the active (partial) chunk
* current-tok   : exact tokens of the active chunk, causally masked

The first/last windows are read from the store's *live* (grad-carrying) buffers; routed
and opened KV are read from the cold record (detached) — old routed context is a
cache-only approximation, exactly as in incremental decode.

Streaming contract
------------------
``decode_block`` processes new tokens that all fall inside the *current* (last, possibly
partial) chunk; it closes that chunk when its last token arrives.  Feed longer inputs one
chunk-worth at a time (see :meth:`prefill`).  Routing is pooled across the block (one
fetch per block), which is the chunk-shared-routing optimisation the offload analysis
relies on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .cache_store import KVCacheStore
from .vectorized import vectorized_routed_attention

_NEG = -1.0e4  # finite mask fill (matches the reference for fp16/bf16 safety)


@dataclass
class RouterConfig:
    nhead: int
    kv_heads: int
    head_dim: int
    chunk_size: int
    group_size: int
    topk_chunks: int = 20
    topk_groups: int = 32
    # Expose completed-group summaries of the *active* chunk (additive, on top of the
    # exact in-chunk tokens) — matches the reference architecture.  Disable to recover
    # plain causal attention when every chunk is kept token-level.
    current_group_summaries: bool = True
    theta: float = 1_000_000.0
    mixed_rope_threshold: float = 0.5
    mixed_rope_cutoff_pair: Optional[int] = None

    @property
    def groups_per_chunk(self) -> int:
        return self.chunk_size // self.group_size

    @property
    def rep(self) -> int:
        return self.nhead // self.kv_heads

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def group_kv_scale(self) -> float:
        # Original heuristic, kept for checkpoint/behaviour continuity.
        return 1.0 / (self.group_size + math.sqrt(self.group_size))


@dataclass
class RoutedKV:
    """Everything the attention needs: assembled keys/values + per-token visibility mask.

    ``k``/``v``: ``[B, H, R, Dh]`` (head-expanded).  ``mask``: ``[B, H, L, R]`` bool, True =
    token may attend.  ``scale``: softmax scale.  Call :meth:`attend` to get the output.
    """

    k: torch.Tensor
    v: torch.Tensor
    mask: torch.Tensor
    scale: float

    def attend(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, H, L, Dh] -> out: [B, H, L, Dh]
        scores = torch.einsum("bhld,bhrd->bhlr", q, self.k) * self.scale
        scores = scores.masked_fill(~self.mask, _NEG)
        probs = torch.softmax(scores.float(), dim=-1).to(self.v.dtype)
        return torch.einsum("bhlr,bhrd->bhld", probs, self.v)


class ChunkRouter:
    """Stateless-per-layer router driving a tiered :class:`KVCacheStore`."""

    def __init__(self, cfg: RouterConfig, store: KVCacheStore) -> None:
        self.cfg = cfg
        self.store = store
        # Active (partial) chunk accumulators, per layer: pre-rope k, rope k, v.
        self._active_kraw: Dict[int, Optional[torch.Tensor]] = {}
        self._active_krope: Dict[int, Optional[torch.Tensor]] = {}
        self._active_v: Dict[int, Optional[torch.Tensor]] = {}
        self._active_start: Dict[int, int] = {}  # absolute position of the active chunk's first token

    # =====================================================================
    # Public API
    # =====================================================================
    def reset(self) -> None:
        self.store.reset()
        self._active_kraw.clear()
        self._active_krope.clear()
        self._active_v.clear()
        self._active_start.clear()

    @torch.no_grad()
    def _route_decision(self, q: torch.Tensor, layer: int, n_closed: int):
        """Pick routed-middle chunks & opened groups (non-differentiable, like the reference).

        Returns ``(mid_idx[B,H,Sc], gk_mid, gv_mid, open_chunk[B,H,Kg], open_grp[B,H,Kg])``.
        ``gk_mid``/``gv_mid`` are the routed chunks' group summaries already fetched into VRAM
        (``[B,H,Sc,M,Dh]``) — returned so ``decode_block`` reuses them instead of re-gathering.
        The routed-middle KV is intentionally detached (cold cache-only approximation).
        """
        cfg = self.cfg
        B, H, L, Dh = q.shape
        device = q.device
        f_lo, f_hi = self.store.policy.hot_first_range(n_closed)
        l_lo, _ = self.store.policy.hot_last_range(n_closed)
        mid_lo, mid_hi = f_hi, l_lo  # candidate pool = chunks strictly between the windows
        n_mid = max(0, mid_hi - mid_lo)
        M = cfg.groups_per_chunk

        if n_mid == 0:
            empty = torch.empty(B, H, 0, dtype=torch.long, device=device)
            none_kv = (None, None)
            return empty, none_kv[0], none_kv[1], empty.clone(), empty.clone()

        ck = self.store.chunk_summaries(layer)[:, :, mid_lo:mid_hi]  # [B,KVH,n_mid,Dh]
        ck = self._rep_heads(ck)                                     # [B,H,n_mid,Dh]
        sc = torch.einsum("bhld,bhnd->bhln", q, ck) * cfg.scale       # [B,H,L,n_mid]
        pooled = sc.max(dim=2).values                                # [B,H,n_mid]

        Kc = min(cfg.topk_chunks, n_mid)
        _, mid_rel = torch.topk(pooled, Kc, dim=-1, sorted=False)     # [B,H,Kc] (relative)
        mid_idx = mid_rel + mid_lo                                    # absolute chunk ids

        # Group level: open top-k groups among the routed chunks' M*Kc groups.  Fetch the
        # routed group summaries once here and reuse them for the output segment.
        self.store.prefetch(layer, mid_idx)
        gk_mid, gv_mid = self.store.gather_group_summaries(layer, mid_idx)  # [B,H,Kc,M,Dh]
        gk_flat = gk_mid.reshape(B, H, Kc * M, Dh)
        sc_g = torch.einsum("bhld,bhrd->bhlr", q, gk_flat) * cfg.scale
        pooled_g = sc_g.max(dim=2).values                            # [B,H,Kc*M]
        Kg = min(cfg.topk_groups, Kc * M)
        _, top_g = torch.topk(pooled_g, Kg, dim=-1, sorted=False)    # [B,H,Kg] in [0,Kc*M)
        parent = top_g // M
        open_chunk = mid_idx.gather(-1, parent)                      # [B,H,Kg]
        open_grp = top_g - parent * M                                # [B,H,Kg]
        return mid_idx, gk_mid, gv_mid, open_chunk, open_grp

    def decode_block(
        self,
        layer: int,
        q: torch.Tensor,        # [B, H, L, Dh]   rope-applied
        k_rope: torch.Tensor,   # [B, KVH, L, Dh] rope-applied
        k_raw: torch.Tensor,    # [B, KVH, L, Dh] pre-rope (for mixed-rope summaries)
        v: torch.Tensor,        # [B, KVH, L, Dh]
        start_pos: int,
    ) -> RoutedKV:
        """Route ``L`` new tokens (all inside the current active chunk) and assemble their KV.

        ``start_pos`` is the absolute position of the first new token; the block must not
        cross a chunk boundary except by completing the active chunk.
        """
        cfg = self.cfg
        C, M, gs = cfg.chunk_size, cfg.groups_per_chunk, cfg.group_size
        B, H, L, Dh = q.shape
        device, dtype = q.device, q.dtype
        n = start_pos // C
        c0 = start_pos % C
        assert c0 + L <= C, "decode_block must stay within one chunk; feed chunk-by-chunk."
        n_closed = self.store.num_closed_chunks(layer)
        assert n_closed == n, f"active chunk {n} != closed {n_closed}; out-of-order block"

        # -- accumulate the active chunk's KV (kept live → grad preserved) --
        self._append_active(layer, k_rope, k_raw, v, start_pos)
        act_krope = self._active_krope[layer]   # [B,KVH,c0+L,Dh]
        act_kraw = self._active_kraw[layer]
        act_v = self._active_v[layer]
        cur_len = act_krope.shape[2]            # == c0 + L

        seg_k: list[torch.Tensor] = []
        seg_v: list[torch.Tensor] = []
        seg_mask: list[torch.Tensor] = []       # each [B,H,L,r]
        # local query positions within the active chunk
        q_local = torch.arange(c0, c0 + L, device=device)  # [L]

        # -- routing decision (chunk + group) --------------------------------
        mid_idx, gk_mid, gv_mid, open_chunk, open_grp = self._route_decision(q, layer, n_closed)

        # =================================================================
        # Segment: FIRST window (attention sinks) — live, grad-carrying
        # =================================================================
        f_lo, f_hi = self.store.policy.hot_first_range(n_closed)
        if f_hi > f_lo:
            if self.store.policy.first_token_level:
                k_f, v_f = self.store.hot_tokens(layer, f_lo, f_hi)   # [B,KVH,nf,C,Dh]
                self._add_block(seg_k, seg_v, seg_mask, k_f, v_f, L, gs=C)
            else:
                gk_f, gv_f = self.store.hot_group_summaries(layer, f_lo, f_hi)  # [B,KVH,nf,M,Dh]
                self._add_block(seg_k, seg_v, seg_mask, gk_f, gv_f, L, gs=M)

        # =================================================================
        # Segment: LAST window (recent local context) — live, grad-carrying
        # =================================================================
        l_lo, l_hi = self.store.policy.hot_last_range(n_closed)
        if l_hi > l_lo:
            k_l, v_l = self.store.hot_tokens(layer, l_lo, l_hi)        # [B,KVH,nl,C,Dh]
            self._add_block(seg_k, seg_v, seg_mask, k_l, v_l, L, gs=C)

        # =================================================================
        # Segment: routed-middle group summaries (detached cold record)
        # =================================================================
        Sc = mid_idx.shape[2]
        if Sc > 0:
            gk_m = gk_mid.reshape(B, H, Sc * M, Dh)  # reused from _route_decision
            gv_m = gv_mid.reshape(B, H, Sc * M, Dh)
            # visible to every block token (all middle chunks are < active chunk)
            mask = torch.ones(B, H, L, Sc * M, dtype=torch.bool, device=device)
            seg_k.append(gk_m); seg_v.append(gv_m); seg_mask.append(mask)

        # =================================================================
        # Segment: opened token KV (detached cold record)
        # =================================================================
        Kg = open_chunk.shape[2]
        if Kg > 0:
            k_o, v_o = self.store.gather_tokens(layer, open_chunk, open_grp)  # [B,H,Kg,gs,Dh]
            k_o = k_o.reshape(B, H, Kg * gs, Dh)
            v_o = v_o.reshape(B, H, Kg * gs, Dh)
            mask = torch.ones(B, H, L, Kg * gs, dtype=torch.bool, device=device)
            seg_k.append(k_o); seg_v.append(v_o); seg_mask.append(mask)

        # =================================================================
        # Segment: current chunk completed group summaries
        # =================================================================
        ncomp_max = cur_len // gs
        if ncomp_max > 0 and cfg.current_group_summaries:
            gk_c, gv_c = self._active_group_summaries(layer, ncomp_max, n)  # [B,H,ncomp,Dh]
            # group g (covering tokens [g*gs, g*gs+gs)) is visible once token >= g*gs+gs-1
            g_end = torch.arange(ncomp_max, device=device) * gs + (gs - 1)   # [ncomp]
            vis = (g_end.view(1, 1, 1, ncomp_max) <= q_local.view(1, 1, L, 1)).expand(B, H, L, ncomp_max)
            seg_k.append(gk_c); seg_v.append(gv_c); seg_mask.append(vis)

        # =================================================================
        # Segment: current chunk exact tokens (causal within chunk)
        # =================================================================
        k_cur = self._rep_heads(act_krope)   # [B,H,cur_len,Dh]
        v_cur = self._rep_heads(act_v)
        tok_pos = torch.arange(cur_len, device=device)               # [cur_len]
        causal = (tok_pos.view(1, 1, 1, cur_len) <= q_local.view(1, 1, L, 1)).expand(B, H, L, cur_len)
        seg_k.append(k_cur); seg_v.append(v_cur); seg_mask.append(causal)

        routed = RoutedKV(
            k=torch.cat(seg_k, dim=2),
            v=torch.cat(seg_v, dim=2),
            mask=torch.cat(seg_mask, dim=3),
            scale=cfg.scale,
        )

        # -- close the chunk if this block filled it ------------------------
        if cur_len == C:
            self._close_active_chunk(layer, n)

        return routed

    @torch.no_grad()
    def prefill(
        self,
        layer: int,
        q: torch.Tensor,        # [B, H, S, Dh]
        k_rope: torch.Tensor,   # [B, KVH, S, Dh]
        k_raw: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Convenience: stream a long prefix chunk-by-chunk, return attention output ``[B,H,S,Dh]``.

        Inference-only (``no_grad``); for differentiable prefill use the model's dense
        training path and seed the store from its closed-chunk summaries instead.
        """
        C = self.cfg.chunk_size
        S = q.shape[2]
        outs = []
        p = start_pos
        done = 0
        while done < S:
            take = min(C - (p % C), S - done)
            routed = self.decode_block(
                layer,
                q[:, :, done:done + take],
                k_rope[:, :, done:done + take],
                k_raw[:, :, done:done + take],
                v[:, :, done:done + take],
                p,
            )
            outs.append(routed.attend(q[:, :, done:done + take]))
            p += take
            done += take
        return torch.cat(outs, dim=2)

    # =====================================================================
    # Unified entry point — single-chunk vs vectorized multi-chunk
    # =====================================================================
    def expected_position(self, layer: int) -> int:
        """Absolute position of the next token the store/active-chunk expects to ingest."""
        n_closed = self.store.num_closed_chunks(layer)
        act = self._active_krope.get(layer)
        active_len = 0 if act is None else act.shape[2]
        return n_closed * self.cfg.chunk_size + active_len

    def is_empty(self, layer: int) -> bool:
        return self.expected_position(layer) == 0

    def rotary_table(self, start_pos: int, length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``[1, 1, length, Dh]`` for absolute positions ``[start_pos, start_pos+length)``."""
        pos = torch.arange(start_pos, start_pos + length, device=device)
        cos, sin = self._rotary_for_positions(pos, torch.empty(1, 1, length, self.cfg.head_dim, device=device))
        return cos, sin

    def process_query_block(
        self,
        layer: int,
        q: torch.Tensor,        # [B, H, S, Dh]   rope-applied (head-expanded)
        k_rope: torch.Tensor,   # [B, KVH, S, Dh] rope-applied
        k_raw: torch.Tensor,    # [B, KVH, S, Dh] pre-rope
        v: torch.Tensor,        # [B, KVH, S, Dh]
        start_pos: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        populate_store: bool = True,
        max_chunks_at_once: Optional[int] = None,
    ) -> torch.Tensor:
        """Route + attend a block of new queries, returning the output ``[B, H, S, Dh]``.

        Two branches, picked from the tokens' chunk span:

        * **single active chunk** (decode / sub-chunk feeds): the incremental
          :meth:`decode_block` path — minimal per-step work.
        * **multiple chunks** (prefill / teacher-forced training): the fast
          :func:`vectorized_routed_attention` chunk-parallel path, tiled into windows of
          ``max_chunks_at_once`` chunks to bound peak VRAM.  When ``populate_store`` is set
          the closed chunks are seeded into the store so a later decode continues seamlessly.
        """
        C = self.cfg.chunk_size
        S = q.shape[2]
        first_chunk = start_pos // C
        last_chunk = (start_pos + S - 1) // C

        if first_chunk == last_chunk:
            routed = self.decode_block(layer, q, k_rope, k_raw, v, start_pos)
            return routed.attend(q)

        if start_pos == 0:
            return self._vectorized_block(
                layer, q, k_rope, k_raw, v, cos, sin, populate_store, max_chunks_at_once
            )

        # Rare: a multi-chunk block continuing on top of resident context (e.g. chunked
        # eval).  Stream it chunk-by-chunk through the incremental path, which correctly
        # routes against the store.  (Not the hot path; correctness over peak throughput.)
        outs = []
        p = start_pos
        done = 0
        while done < S:
            take = min(C - (p % C), S - done)
            routed = self.decode_block(
                layer, q[:, :, done:done + take], k_rope[:, :, done:done + take],
                k_raw[:, :, done:done + take], v[:, :, done:done + take], p,
            )
            outs.append(routed.attend(q[:, :, done:done + take]))
            p += take
            done += take
        return torch.cat(outs, dim=2)

    def _vectorized_block(
        self,
        layer: int,
        q: torch.Tensor,
        k_rope: torch.Tensor,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        populate_store: bool,
        max_chunks_at_once: Optional[int],
    ) -> torch.Tensor:
        cfg = self.cfg
        C = cfg.chunk_size
        B, H, S, Dh = q.shape
        device, dtype = q.device, q.dtype
        N = (S + C - 1) // C

        if cos is None or sin is None:
            cos, sin = self.rotary_table(0, S, device)
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        k_rope_h = self._rep_heads(k_rope)
        k_raw_h = self._rep_heads(k_raw)
        v_h = self._rep_heads(v)

        W = max_chunks_at_once if (max_chunks_at_once and max_chunks_at_once > 0) else N
        if N <= W:
            out = vectorized_routed_attention(cfg, q, k_rope_h, k_raw_h, v_h, cos, sin)
        else:
            # Tile the query-chunk axis; each window sees the full causal prefix of keys so
            # routing stays exact, and we keep only the window's own outputs.
            outs = []
            w0 = 0
            while w0 < N:
                w1 = min(N, w0 + W)
                s_end = min(S, w1 * C)
                s_keep0 = w0 * C
                o = vectorized_routed_attention(
                    cfg, q[:, :, :s_end], k_rope_h[:, :, :s_end], k_raw_h[:, :, :s_end],
                    v_h[:, :, :s_end], cos[:, :, :s_end], sin[:, :, :s_end],
                )
                outs.append(o[:, :, s_keep0:s_end])
                w0 = w1
            out = torch.cat(outs, dim=2)

        if populate_store:
            self._seed_store_from_block(layer, k_rope, k_raw, v)
        return out

    @torch.no_grad()
    def _seed_store_from_block(
        self, layer: int, k_rope: torch.Tensor, k_raw: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Populate the store with the block's closed chunks and stash the partial remainder.

        Computes every closed chunk's summaries *vectorized over the chunk axis* and hands
        them to the store in a single :meth:`KVCacheStore.seed_closed_chunks` call, so the
        summaries match :meth:`_close_active_chunk` while keeping prefill as fast as the dense
        reference.  Assumes the block starts at position 0 (fresh sequence).
        """
        cfg = self.cfg
        C, gs, M = cfg.chunk_size, cfg.group_size, cfg.groups_per_chunk
        B, KVH, S, Dh = k_rope.shape
        n_closed = S // C
        device = k_rope.device

        if n_closed > 0:
            end = n_closed * C
            krope = k_rope[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            kraw = k_raw[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)
            vtok = v[:, :, :end, :].reshape(B, KVH, n_closed, C, Dh)

            raw_g = kraw.reshape(B, KVH, n_closed, M, gs, Dh)
            rope_g = krope.reshape(B, KVH, n_closed, M, gs, Dh)
            v_g = vtok.reshape(B, KVH, n_closed, M, gs, Dh)
            ar_n = torch.arange(n_closed, device=device)
            ar_m = torch.arange(M, device=device)
            g_anchor = ar_n[:, None] * C + ar_m[None, :] * gs + (gs - 1) // 2   # [N, M]
            gk = self._rope_summary_at(raw_g, rope_g, 4, g_anchor, cfg.group_kv_scale)  # [B,KVH,N,M,Dh]
            gv = v_g.sum(dim=4) * cfg.group_kv_scale

            c_anchor = ar_n * C + (C - 1) // 2                                  # [N]
            ck = self._rope_summary_at(kraw, krope, 3, c_anchor, 1.0)           # [B,KVH,N,Dh]

            self.store.seed_closed_chunks(layer, ck, gk, gv, krope, vtok)

        rem = S - n_closed * C
        if rem > 0:
            self._active_krope[layer] = k_rope[:, :, n_closed * C:, :]
            self._active_kraw[layer] = k_raw[:, :, n_closed * C:, :]
            self._active_v[layer] = v[:, :, n_closed * C:, :]
            self._active_start[layer] = n_closed * C
        else:
            self._active_krope[layer] = None
            self._active_kraw[layer] = None
            self._active_v[layer] = None

    # =====================================================================
    # Active-chunk handling
    # =====================================================================
    def _append_active(
        self, layer: int, k_rope: torch.Tensor, k_raw: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> None:
        if self._active_krope.get(layer) is None:
            self._active_krope[layer] = k_rope
            self._active_kraw[layer] = k_raw
            self._active_v[layer] = v
            self._active_start[layer] = start_pos - (start_pos % self.cfg.chunk_size)
        else:
            self._active_krope[layer] = torch.cat([self._active_krope[layer], k_rope], dim=2)
            self._active_kraw[layer] = torch.cat([self._active_kraw[layer], k_raw], dim=2)
            self._active_v[layer] = torch.cat([self._active_v[layer], v], dim=2)

    def _active_group_summaries(
        self, layer: int, ncomp: int, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixed-rope group K + scaled-sum group V summaries for the ncomp completed groups."""
        cfg = self.cfg
        B, KVH, _, Dh = self._active_krope[layer].shape
        gs, M = cfg.group_size, cfg.groups_per_chunk
        raw = self._active_kraw[layer][:, :, : ncomp * gs].reshape(B, KVH, ncomp, gs, Dh)
        rope = self._active_krope[layer][:, :, : ncomp * gs].reshape(B, KVH, ncomp, gs, Dh)
        vg = self._active_v[layer][:, :, : ncomp * gs].reshape(B, KVH, ncomp, gs, Dh)
        anchor = n * cfg.chunk_size + torch.arange(ncomp, device=raw.device) * gs + (gs - 1) // 2
        gk = self._rope_summary_at(raw, rope, 3, anchor, cfg.group_kv_scale)  # [B,KVH,ncomp,Dh]
        gv = vg.sum(dim=3) * cfg.group_kv_scale
        return self._rep_heads(gk), self._rep_heads(gv)

    def _close_active_chunk(self, layer: int, n: int) -> None:
        """Compute the chunk's summaries and hand the closed chunk to the store (grad kept)."""
        cfg = self.cfg
        C, gs, M = cfg.chunk_size, cfg.group_size, cfg.groups_per_chunk
        krope = self._active_krope[layer]
        kraw = self._active_kraw[layer]
        v = self._active_v[layer]
        B, KVH, _, Dh = krope.shape

        # group summaries
        raw_g = kraw.reshape(B, KVH, M, gs, Dh)
        rope_g = krope.reshape(B, KVH, M, gs, Dh)
        v_g = v.reshape(B, KVH, M, gs, Dh)
        g_anchor = n * C + torch.arange(M, device=krope.device) * gs + (gs - 1) // 2
        gk = self._rope_summary_at(raw_g, rope_g, 3, g_anchor, cfg.group_kv_scale)   # [B,KVH,M,Dh]
        gv = v_g.sum(dim=3) * cfg.group_kv_scale                                     # [B,KVH,M,Dh]

        # chunk summary
        c_anchor = torch.tensor([n * C + (C - 1) // 2], device=krope.device)
        ck = self._rope_summary_at(
            kraw.reshape(B, KVH, 1, C, Dh), krope.reshape(B, KVH, 1, C, Dh), 3, c_anchor, 1.0
        ).squeeze(2)  # [B,KVH,Dh]

        self.store.append_closed_chunk(layer, ck, gk, gv, krope, v)
        self._active_krope[layer] = None
        self._active_kraw[layer] = None
        self._active_v[layer] = None

    # =====================================================================
    # Small tensor helpers
    # =====================================================================
    def _rep_heads(self, t: torch.Tensor) -> torch.Tensor:
        """KVH -> H along dim 1 (GQA expand)."""
        rep = self.cfg.rep
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    def _add_block(
        self,
        seg_k: list,
        seg_v: list,
        seg_mask: list,
        k: torch.Tensor,
        v: torch.Tensor,
        L: int,
        gs: int,
    ) -> None:
        """Append a fully-visible (causally-earlier) block of ``[B,KVH,n,gs,Dh]`` summaries/tokens."""
        B, KVH, ncnk = k.shape[0], k.shape[1], k.shape[2]
        Dh = k.shape[-1]
        k = self._rep_heads(k).reshape(B, self.cfg.nhead, ncnk * gs, Dh)
        v = self._rep_heads(v).reshape(B, self.cfg.nhead, ncnk * gs, Dh)
        mask = torch.ones(B, self.cfg.nhead, L, ncnk * gs, dtype=torch.bool, device=k.device)
        seg_k.append(k); seg_v.append(v); seg_mask.append(mask)

    # ---- mixed-RoPE summary math (mirrors HierarchicalGlobalAttention) ----
    def _rope_summary_at(
        self, raw: torch.Tensor, rope: torch.Tensor, reduce_dim: int, anchor_pos: torch.Tensor, scale: float
    ) -> torch.Tensor:
        raw_sum = raw.sum(dim=reduce_dim) * scale
        tokenwise = rope.sum(dim=reduce_dim) * scale
        a_cos, a_sin = self._rotary_for_positions(anchor_pos, raw_sum)
        endpoint = self._apply_rotary(raw_sum.float(), a_cos, a_sin).to(dtype=raw_sum.dtype)
        return self._mix_tokenwise_and_anchor(tokenwise, endpoint)

    def _rotary_for_positions(self, pos: torch.Tensor, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.cfg.head_dim // 2
        device = like.device
        inv_freq = 1.0 / (self.cfg.theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        freqs = pos.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        view = (1, 1) + tuple(pos.shape) + (self.cfg.head_dim,)
        return emb.cos().reshape(view), emb.sin().reshape(view)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def _mixed_cutoff_pair(self) -> int:
        half = self.cfg.head_dim // 2
        if self.cfg.mixed_rope_cutoff_pair is not None:
            return int(self.cfg.mixed_rope_cutoff_pair)
        cutoff = 0
        for i in range(half):
            inv_freq = 1.0 / (self.cfg.theta ** (i / half))
            if max(1, self.cfg.chunk_size - 1) * inv_freq > self.cfg.mixed_rope_threshold:
                cutoff = i + 1
        return cutoff

    def _mix_tokenwise_and_anchor(self, tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        half = self.cfg.head_dim // 2
        cutoff = self._mixed_cutoff_pair()
        pair_mask = torch.arange(half, device=tokenwise.device) < cutoff
        mask = torch.cat([pair_mask, pair_mask], dim=0)
        view_shape = [1] * tokenwise.ndim
        view_shape[-1] = self.cfg.head_dim
        return torch.where(mask.view(*view_shape), tokenwise, anchor)
