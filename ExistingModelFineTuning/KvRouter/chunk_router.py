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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .cache_store import KVCacheStore
from .vectorized import assemble_routed_kv

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
    """The routed context the attention should attend over — **data only, no scores**.

    The router's job ends here: it has *selected* what each query may see and pulled that KV
    into VRAM.  Computing q·k, the softmax and the output is the *attention's* job (call
    :meth:`attend`, or read the fields and do it however the variant wants).

    Two kinds of routed KV are exposed **separately** so a variant can pick:

    * ``token_*`` — exact token-level K/V (current chunk, opened groups, any token-level
      windows).  Always present.
    * ``summary_*`` — group-**summary** K/V (a learned-from-scratch approximation of routed
      chunks).  ``None`` when the variant does not produce summaries (e.g. the exact router),
      and skippable at attend time via ``use_summaries=False`` for models never trained on them.

    Tensors are head-expanded ``[B, H, R, Dh]`` with mask ``[B, H, L, R]`` (flat layout), or
    chunk-parallel ``[B, H, N, R, Dh]`` with mask ``[B, H, N, C, R]`` when ``chunked`` (the
    vectorized prefill path).  ``scale`` is the softmax scale.
    """

    token_k: torch.Tensor
    token_v: torch.Tensor
    token_mask: torch.Tensor
    scale: float
    summary_k: Optional[torch.Tensor] = None
    summary_v: Optional[torch.Tensor] = None
    summary_mask: Optional[torch.Tensor] = None
    chunked: bool = False
    chunk_size: int = 0

    def _segments(self, use_summaries: bool):
        # Token-only (e.g. the exact router, or use_summaries=False): hand back the tensors
        # directly — no concat (cheaper in the launch-bound decode regime).
        if not (use_summaries and self.summary_k is not None):
            return self.token_k, self.token_v, self.token_mask
        return (
            torch.cat([self.token_k, self.summary_k], dim=-2),
            torch.cat([self.token_v, self.summary_v], dim=-2),
            torch.cat([self.token_mask, self.summary_mask], dim=-1),
        )

    def attend(self, q: torch.Tensor, use_summaries: bool = True) -> torch.Tensor:
        """Default attention over the routed KV.  ``q``: ``[B, H, L, Dh]`` → ``[B, H, L, Dh]``.

        Accumulates scores and the probs·V product in fp32 (as torch SDPA does internally);
        the upcast is a no-op for fp32 callers and is required for bf16 to track an SDPA
        baseline across many layers.  This is just the *default* — a consumer may read
        ``token_*``/``summary_*`` and compute the scores itself instead.
        """
        k, v, mask = self._segments(use_summaries)
        out_dtype = v.dtype
        if self.chunked:
            B, H, S, Dh = q.shape
            C, N = self.chunk_size, k.shape[2]
            pad = N * C - S
            qc = (q if pad == 0 else torch.cat([q, q.new_zeros(B, H, pad, Dh)], dim=2)).reshape(B, H, N, C, Dh)
            scores = torch.einsum("bhncd,bhnrd->bhncr", qc.float(), k.float()) * self.scale
            scores = scores.masked_fill(~mask, _NEG)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhncr,bhnrd->bhncd", probs, v.float())
            return out.reshape(B, H, N * C, Dh)[:, :, :S].to(out_dtype)
        scores = torch.einsum("bhld,bhrd->bhlr", q.float(), k.float()) * self.scale
        scores = scores.masked_fill(~mask, _NEG)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("bhlr,bhrd->bhld", probs, v.float()).to(out_dtype)


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
        """Select routed-middle chunks + opened groups (non-differentiable, like the reference).

        Candidate pool = chunks strictly between the first/last hot windows.  The
        immediately-preceding chunk is **force-included** as a routed summary *when it lies
        inside that pool* (i.e. it is not already exposed token-level by a ``keep_last`` window)
        — this is the reference ``_decode_forward`` behaviour.

        Returns ``(mid_idx[B,H,Kc], gk_mid, gv_mid, mid_vis[B,H,L,Kc],
        open_chunk[B,H,Kg], open_grp[B,H,Kg], open_vis[B,H,L,Kg])``.  ``gk/gv_mid`` are the
        routed chunks' group summaries (``[B,H,Kc,M,Dh]``) already fetched into VRAM.

        The materialized chunk/group *sets* are pooled over the ``L`` query positions (so a
        single VRAM fetch serves the whole block), but ``mid_vis`` / ``open_vis`` give each
        query position a **causal** view: position ``t`` only sees a routed chunk/group that
        itself or an earlier position in the block actually requested (cumulative-OR over the
        block, mirroring the vectorized prefill ``cumsum`` mask).  This stops an early token
        from attending to a chunk that only a *later* token in the same block selected.  The
        force-included previous chunk is deterministic (a past chunk) and stays visible to all
        positions.  For ``L == 1`` every position-mask is all-true, so this reduces exactly to
        the reference per-token routing.
        """
        cfg = self.cfg
        B, H, L, Dh = q.shape
        device = q.device
        f_lo, f_hi = self.store.policy.hot_first_range(n_closed)
        l_lo, _ = self.store.policy.hot_last_range(n_closed)
        mid_lo, mid_hi = f_hi, l_lo
        n_mid = max(0, mid_hi - mid_lo)
        M = cfg.groups_per_chunk

        if n_mid == 0 or cfg.topk_chunks <= 0:
            empty = torch.empty(B, H, 0, dtype=torch.long, device=device)
            empty_vis = torch.empty(B, H, L, 0, dtype=torch.bool, device=device)
            return empty, None, None, empty_vis, empty.clone(), empty.clone(), empty_vis.clone()

        ck = self._rep_heads(self.store.chunk_summaries(layer)[:, :, mid_lo:mid_hi])  # [B,H,n_mid,Dh]
        sc = torch.einsum("bhld,bhnd->bhln", q, ck) * cfg.scale       # [B,H,L,n_mid]
        Kc = min(cfg.topk_chunks, n_mid)
        pooled = sc.max(dim=2).values                                # [B,H,n_mid]
        top_scores, mid_rel = torch.topk(pooled, Kc, dim=-1, sorted=False)  # [B,H,Kc] (relative)
        # Per-query requests (each position's own top-Kc), for causal visibility.
        _, req_rel = torch.topk(sc, Kc, dim=-1, sorted=False)        # [B,H,L,Kc]

        prev = n_closed - 1
        prev_in_mid = mid_lo <= prev < mid_hi
        if prev_in_mid:                                              # prev not already a window
            prev_rel = prev - mid_lo
            missing = ~(mid_rel == prev_rel).any(dim=-1, keepdim=True)
            replace_at = top_scores.argmin(dim=-1, keepdim=True)
            mid_rel = mid_rel.scatter(
                -1, replace_at,
                torch.where(missing, torch.full_like(replace_at, prev_rel), mid_rel.gather(-1, replace_at)),
            )
        mid_idx = mid_rel + mid_lo                                    # absolute chunk ids [B,H,Kc]

        # Causal visibility: chunk slot k is visible to position t iff some position <= t
        # requested mid_rel[k]; cumulative-OR over the block dim.
        mid_requested = (req_rel.unsqueeze(-2) == mid_rel.unsqueeze(2).unsqueeze(-1)).any(dim=-1)  # [B,H,L,Kc]
        mid_vis = torch.cumsum(mid_requested.to(torch.int32), dim=2) > 0                           # [B,H,L,Kc]
        if prev_in_mid:                                              # force-included prev: visible to all
            mid_vis = mid_vis | (mid_rel == prev_rel).unsqueeze(2)

        self.store.prefetch(layer, mid_idx)
        gk_mid, gv_mid = self.store.gather_group_summaries(layer, mid_idx)  # [B,H,Kc,M,Dh]
        # Materialize ``topk_groups`` opened groups (matching the vectorized prefill/training
        # budget); the per-query *request* visibility uses ``topk_groups // 2`` (also matching
        # vectorized's ``Kg_request``) so the two paths expose the same routed token content.
        Kg = min(cfg.topk_groups, Kc * M) if cfg.topk_groups > 0 else 0
        Kg_request = min(cfg.topk_groups // 2, Kc * M) if cfg.topk_groups > 0 else 0
        if Kg <= 0:
            empty = torch.empty(B, H, 0, dtype=torch.long, device=device)
            empty_vis = torch.empty(B, H, L, 0, dtype=torch.bool, device=device)
            return mid_idx, gk_mid, gv_mid, mid_vis, empty, empty.clone(), empty_vis
        gk_flat = gk_mid.reshape(B, H, Kc * M, Dh)
        sc_g = torch.einsum("bhld,bhrd->bhlr", q, gk_flat) * cfg.scale   # [B,H,L,Kc*M]
        pooled_g = sc_g.max(dim=2).values                            # [B,H,Kc*M]
        _, top_g = torch.topk(pooled_g, Kg, dim=-1, sorted=False)    # [B,H,Kg]
        _, req_g = torch.topk(sc_g, Kg_request, dim=-1, sorted=False)  # [B,H,L,Kg_request] per-query
        parent = top_g // M
        open_chunk = mid_idx.gather(-1, parent)                      # [B,H,Kg]
        open_grp = top_g - parent * M

        # Causal visibility for opened groups: requested by some position <= t, AND the parent
        # routed chunk is visible to t.
        grp_requested = (req_g.unsqueeze(-2) == top_g.unsqueeze(2).unsqueeze(-1)).any(dim=-1)  # [B,H,L,Kg]
        open_vis = torch.cumsum(grp_requested.to(torch.int32), dim=2) > 0
        parent_vis = torch.gather(mid_vis, -1, parent.unsqueeze(2).expand(B, H, L, Kg))        # [B,H,L,Kg]
        open_vis = open_vis & parent_vis
        return mid_idx, gk_mid, gv_mid, mid_vis, open_chunk, open_grp, open_vis

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

        Returns a :class:`RoutedKV` with **separate** token-level and group-summary segments
        (data only — the caller computes the scores).  Segments:

        * first/last hot windows (``keep_first`` sinks / ``keep_last`` recent chunks): token KV
          for the last window (and the first when ``first_token_level``), else group summaries;
        * routed-middle chunks (``topk_chunks`` selected, prev force-included): group summaries;
        * opened groups (``topk_groups // 2`` of the routed groups): exact token KV;
        * active-chunk completed-group summaries (causal) and the active chunk's exact tokens.

        With ``keep_first == keep_last == 0`` this reduces exactly to the reference
        ``HierarchicalGlobalAttention._decode_forward``, so summary-mode decode matches the
        vectorized prefill/training path.  The routed-middle / opened-group *sets* are pooled
        across the block (one VRAM fetch for the ``L`` tokens) but each query position gets a
        **causal** view of them (see :meth:`_route_decision`): position ``t`` only attends to a
        routed chunk/group that itself or an earlier position requested.  For ``L == 1`` this
        equals the reference's per-token routing.
        """
        cfg = self.cfg
        C, M, gs = cfg.chunk_size, cfg.groups_per_chunk, cfg.group_size
        B, H, L, Dh = q.shape
        device = q.device
        n = start_pos // C
        c0 = start_pos % C
        assert c0 + L <= C, "decode_block must stay within one chunk; feed chunk-by-chunk."
        n_closed = self.store.num_closed_chunks(layer)
        assert n_closed == n, f"active chunk {n} != closed {n_closed}; out-of-order block"

        # -- accumulate the active chunk's KV (kept live → grad preserved) --
        self._append_active(layer, k_rope, k_raw, v, start_pos)
        act_krope = self._active_krope[layer]   # [B,KVH,cur_len,Dh]
        act_v = self._active_v[layer]
        cur_len = act_krope.shape[2]            # == c0 + L
        q_local = torch.arange(c0, c0 + L, device=device)  # [L]

        tok_k: list[torch.Tensor] = []
        tok_v: list[torch.Tensor] = []
        tok_mask: list[torch.Tensor] = []
        sum_k: list[torch.Tensor] = []
        sum_v: list[torch.Tensor] = []
        sum_mask: list[torch.Tensor] = []

        mid_idx, gk_mid, gv_mid, mid_vis, open_chunk, open_grp, open_vis = self._route_decision(q, layer, n_closed)

        # ---- first window (attention sinks): token KV or group summaries ----
        f_lo, f_hi = self.store.policy.hot_first_range(n_closed)
        if f_hi > f_lo:
            if self.store.policy.first_token_level:
                k_f, v_f = self.store.hot_tokens(layer, f_lo, f_hi)
                self._add_block(tok_k, tok_v, tok_mask, k_f, v_f, L, gs=C)
            else:
                gk_f, gv_f = self.store.hot_group_summaries(layer, f_lo, f_hi)
                self._add_block(sum_k, sum_v, sum_mask, gk_f, gv_f, L, gs=M)

        # ---- last window (recent local context): token KV ----
        l_lo, l_hi = self.store.policy.hot_last_range(n_closed)
        if l_hi > l_lo:
            k_l, v_l = self.store.hot_tokens(layer, l_lo, l_hi)
            self._add_block(tok_k, tok_v, tok_mask, k_l, v_l, L, gs=C)

        # ---- routed-middle chunks: group summaries (causal per-query visibility) ----
        Sc = mid_idx.shape[2]
        if Sc > 0:
            sum_k.append(gk_mid.reshape(B, H, Sc * M, Dh))
            sum_v.append(gv_mid.reshape(B, H, Sc * M, Dh))
            # each chunk's M group summaries inherit that chunk's per-query visibility
            sum_mask.append(mid_vis.unsqueeze(-1).expand(B, H, L, Sc, M).reshape(B, H, L, Sc * M))

        # ---- opened groups: exact token KV (causal per-query visibility) ----
        Kg = open_chunk.shape[2]
        if Kg > 0:
            k_o, v_o = self.store.gather_tokens(layer, open_chunk, open_grp)  # [B,H,Kg,gs,Dh]
            tok_k.append(k_o.reshape(B, H, Kg * gs, Dh))
            tok_v.append(v_o.reshape(B, H, Kg * gs, Dh))
            tok_mask.append(open_vis.unsqueeze(-1).expand(B, H, L, Kg, gs).reshape(B, H, L, Kg * gs))

        # ---- active chunk completed group summaries (causal visibility) ----
        ncomp = cur_len // gs
        if ncomp > 0 and cfg.current_group_summaries:
            gk_c, gv_c = self._active_group_summaries(layer, ncomp, n)    # [B,H,ncomp,Dh]
            g_end = torch.arange(ncomp, device=device) * gs + (gs - 1)
            vis = (g_end.view(1, 1, 1, ncomp) <= q_local.view(1, 1, L, 1)).expand(B, H, L, ncomp)
            sum_k.append(gk_c); sum_v.append(gv_c); sum_mask.append(vis)

        # ---- active chunk exact tokens (causal within the chunk) ----
        tok_pos = torch.arange(cur_len, device=device)
        causal = (tok_pos.view(1, 1, 1, cur_len) <= q_local.view(1, 1, L, 1)).expand(B, H, L, cur_len)
        tok_k.append(self._rep_heads(act_krope)); tok_v.append(self._rep_heads(act_v)); tok_mask.append(causal)

        routed = RoutedKV(
            token_k=torch.cat(tok_k, dim=2),
            token_v=torch.cat(tok_v, dim=2),
            token_mask=torch.cat(tok_mask, dim=3),
            scale=cfg.scale,
            summary_k=torch.cat(sum_k, dim=2) if sum_k else None,
            summary_v=torch.cat(sum_v, dim=2) if sum_v else None,
            summary_mask=torch.cat(sum_mask, dim=3) if sum_mask else None,
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

    def route_query_block(
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
    ) -> List[Tuple[RoutedKV, int, int]]:
        """Route a block of new queries and return its assembled KV — **the attention attends**.

        The router only decides *what* each query sees and pulls that KV into VRAM; it does
        not compute scores.  Returns a list of ``(routed, lo, hi)`` segments — the caller does
        ``out[:, :, lo:hi] = routed.attend(q[:, :, lo:hi], use_summaries=...)`` for each.

        * **single active chunk** (decode / sub-chunk feeds) → one flat :meth:`decode_block`
          segment.
        * **multiple chunks at a fresh sequence** (prefill / teacher-forced training) → one
          chunk-parallel segment via :func:`assemble_routed_kv` (fast); the closed chunks are
          seeded into the store when ``populate_store`` so a later decode continues seamlessly.
        * **multiple chunks over resident context** (rare, e.g. chunked eval) → one flat
          segment per chunk through the incremental path.
        """
        C = self.cfg.chunk_size
        S = q.shape[2]
        first_chunk = start_pos // C
        last_chunk = (start_pos + S - 1) // C

        if first_chunk == last_chunk:
            return [(self.decode_block(layer, q, k_rope, k_raw, v, start_pos), 0, S)]

        if start_pos == 0:
            routed = self._assemble_vectorized(layer, q, k_rope, k_raw, v, cos, sin, populate_store)
            return [(routed, 0, S)]

        segs: List[Tuple[RoutedKV, int, int]] = []
        p, done = start_pos, 0
        while done < S:
            take = min(C - (p % C), S - done)
            routed = self.decode_block(
                layer, q[:, :, done:done + take], k_rope[:, :, done:done + take],
                k_raw[:, :, done:done + take], v[:, :, done:done + take], p,
            )
            segs.append((routed, done, done + take))
            p += take
            done += take
        return segs

    def _assemble_vectorized(
        self,
        layer: int,
        q: torch.Tensor,
        k_rope: torch.Tensor,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        populate_store: bool,
    ) -> RoutedKV:
        """Chunk-parallel routing + KV assembly over a fresh-sequence block → chunked RoutedKV."""
        cfg = self.cfg
        C = cfg.chunk_size
        S = q.shape[2]
        device, dtype = q.device, q.dtype

        if cos is None or sin is None:
            cos, sin = self.rotary_table(0, S, device)
            cos = cos.to(dtype); sin = sin.to(dtype)

        tk, tv, tm, sk, sv, sm = assemble_routed_kv(
            cfg, q, self._rep_heads(k_rope), self._rep_heads(k_raw), self._rep_heads(v), cos, sin,
            keep_first=self.store.policy.keep_first,
            keep_last=self.store.policy.keep_last,
            first_token_level=self.store.policy.first_token_level,
        )
        if populate_store:
            self._seed_store_from_block(layer, k_rope, k_raw, v)
        return RoutedKV(
            token_k=tk, token_v=tv, token_mask=tm, scale=cfg.scale,
            summary_k=sk, summary_v=sv, summary_mask=sm, chunked=True, chunk_size=C,
        )

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
