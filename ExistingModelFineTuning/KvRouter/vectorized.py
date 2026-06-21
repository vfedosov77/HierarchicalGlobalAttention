"""Vectorized multi-chunk routed attention (the fast prefill / training path).

This is the chunk-parallel routed attention used by :meth:`ChunkRouter.process_query_block`
whenever the new-query block spans more than one chunk (prefill, teacher-forced training).
It is a faithful port of the validated ``HierarchicalGlobalAttention._forward_dense``
algorithm — every query chunk routes, opens groups and attends *in parallel* across the
chunk dimension, so the whole block is a handful of big batched matmuls rather than a
per-chunk Python loop.

Contract
--------
Inputs are already projected, head-expanded (``H`` heads) and RoPE-applied:

* ``q``       ``[B, H, S, Dh]``  RoPE-applied queries
* ``k_rope``  ``[B, H, S, Dh]``  RoPE-applied keys
* ``k_raw``   ``[B, H, S, Dh]``  pre-RoPE keys (for the mixed-RoPE summaries)
* ``v``       ``[B, H, S, Dh]``
* ``cos``/``sin`` ``[1, 1, S, Dh]`` rotary tables covering the block's absolute positions

The block is assumed to start at a chunk boundary (``start_pos == 0`` in the offloaded
generation flow, or a full teacher-forced sequence in training); there is no resident
prior context to route against — exactly the regime this vectorized path targets.  Returns
the attention output ``[B, H, S, Dh]`` (pre ``o_proj``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from .chunk_router import RouterConfig

RotaryData = Tuple[torch.Tensor, torch.Tensor]
_NEG = -1.0e4  # finite mask fill (fp16/bf16-safe, matches the reference)


# ---------------------------------------------------------------------------
# stateless tensor helpers (ported verbatim from the reference _forward_dense)
# ---------------------------------------------------------------------------
def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
    pad = target_len - x.shape[2]
    if pad <= 0:
        return x
    return torch.cat([x, x.new_zeros(*x.shape[:2], pad, x.shape[-1])], dim=2)


def _gather_rotary(cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> RotaryData:
    shape = pos.shape
    flat = pos.reshape(-1)
    cos_g = cos[:, :, flat, :].reshape(cos.shape[0], 1, *shape, cos.shape[-1])
    sin_g = sin[:, :, flat, :].reshape(sin.shape[0], 1, *shape, sin.shape[-1])
    return cos_g, sin_g


def _gather_chunks(tensor: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
    B, H, Nsrc = tensor.shape[:3]
    idx_shape = chunk_idx.shape[2:]
    tail = tensor.shape[3:]
    flat = math.prod(idx_shape)
    if flat == 0:
        return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
    t = tensor.reshape(B * H, Nsrc, *tail)
    idx = chunk_idx.reshape(B * H, flat)
    bh = torch.arange(B * H, device=tensor.device).view(B * H, 1)
    return t[bh, idx].reshape(B, H, *idx_shape, *tail)


def _gather_groups(tensor: torch.Tensor, chunk_idx: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    B, H, Nsrc, M = tensor.shape[:4]
    idx_shape = chunk_idx.shape[2:]
    tail = tensor.shape[4:]
    flat = math.prod(idx_shape)
    if flat == 0:
        return torch.empty(B, H, *idx_shape, *tail, device=tensor.device, dtype=tensor.dtype)
    t = tensor.reshape(B * H, Nsrc, M, *tail)
    cidx = chunk_idx.reshape(B * H, flat)
    gidx = group_idx.reshape(B * H, flat)
    bh = torch.arange(B * H, device=tensor.device).view(B * H, 1)
    return t[bh, cidx, gidx].reshape(B, H, *idx_shape, *tail)


@torch.no_grad()
def _all_route_indices(prefix_shape: torch.Size, num_routes: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(num_routes, device=device).view(*((1,) * len(prefix_shape)), num_routes)
    return idx.expand(*prefix_shape, num_routes)


@torch.no_grad()
def _topk_scores_indices(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_routes = scores.shape[-1]
    assert k > 0
    if k >= num_routes:
        return scores, _all_route_indices(scores.shape[:-1], num_routes, scores.device)
    return torch.topk(scores, k, dim=-1, sorted=False)


@torch.no_grad()
def _route_topk_requests(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_routes = scores.shape[-1]
    assert k > 0
    if k >= num_routes:
        return _all_route_indices(scores.shape[:-1], num_routes, scores.device), scores
    top_scores, top_idx = torch.topk(scores, k, dim=-1)
    return top_idx, top_scores


@torch.no_grad()
def _max_route_scores_from_requests(
    request_idx: torch.Tensor, request_scores: torch.Tensor, num_routes: int, fill_value: float
) -> torch.Tensor:
    if request_idx.shape[-1] == num_routes:
        return request_scores.max(dim=-2).values
    prefix_shape = request_idx.shape[:-2]
    flat_prefix = math.prod(prefix_shape)
    route_scores = torch.full(
        (flat_prefix, num_routes), fill_value, device=request_scores.device, dtype=request_scores.dtype
    )
    route_scores.scatter_reduce_(
        1,
        request_idx.reshape(flat_prefix, -1),
        request_scores.reshape(flat_prefix, -1),
        reduce="amax",
        include_self=True,
    )
    return route_scores.reshape(*prefix_shape, num_routes)


@torch.no_grad()
def _requests_for_selected_routes(request_idx: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
    return (request_idx.unsqueeze(-1) == selected_idx.unsqueeze(-2).unsqueeze(-3)).any(dim=-2)


# ---------------------------------------------------------------------------
# mixed-RoPE summary math (cfg-parametrised; mirrors the reference)
# ---------------------------------------------------------------------------
def _mixed_cutoff_pair(cfg: "RouterConfig") -> int:
    half = cfg.head_dim // 2
    if cfg.mixed_rope_cutoff_pair is not None:
        return int(cfg.mixed_rope_cutoff_pair)
    cutoff = 0
    for i in range(half):
        inv_freq = 1.0 / (cfg.theta ** (i / half))
        if max(1, cfg.chunk_size - 1) * inv_freq > cfg.mixed_rope_threshold:
            cutoff = i + 1
    return cutoff


def _mix_tokenwise_and_anchor(cfg: "RouterConfig", tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    half = cfg.head_dim // 2
    cutoff = _mixed_cutoff_pair(cfg)
    pair_mask = torch.arange(half, device=tokenwise.device) < cutoff
    mask = torch.cat([pair_mask, pair_mask], dim=0)
    view_shape = [1] * tokenwise.ndim
    view_shape[-1] = cfg.head_dim
    return torch.where(mask.view(*view_shape), tokenwise, anchor)


def _rope_summary(
    cfg: "RouterConfig",
    raw: torch.Tensor,
    rope: torch.Tensor,
    mask: torch.Tensor,
    reduce_dim: int,
    anchor_pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    raw_sum = (raw * mask).sum(dim=reduce_dim) * scale
    anchor_cos, anchor_sin = _gather_rotary(cos, sin, anchor_pos)
    endpoint = _apply_rotary(raw_sum.float(), anchor_cos, anchor_sin).to(dtype=raw_sum.dtype)
    tokenwise = (rope * mask).sum(dim=reduce_dim) * scale
    return _mix_tokenwise_and_anchor(cfg, tokenwise=tokenwise, anchor=endpoint)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def assemble_routed_kv(
    cfg: "RouterConfig",
    q: torch.Tensor,        # [B, H, S, Dh] rope-applied
    k_rope: torch.Tensor,   # [B, H, S, Dh] rope-applied
    k_raw: torch.Tensor,    # [B, H, S, Dh] pre-rope
    v: torch.Tensor,        # [B, H, S, Dh]
    cos: torch.Tensor,      # [1, 1, S, Dh]
    sin: torch.Tensor,      # [1, 1, S, Dh]
    *,
    keep_first: int = 0,
    keep_last: int = 0,
    first_token_level: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-parallel routing + KV assembly over a whole block (the fast prefill/training path).

    Selects, per query chunk, the routed previous chunks (group summaries, prev-chunk
    force-included) and the opened groups (exact tokens), and assembles them — together with the
    active chunk's completed-group summaries and exact tokens — into **separate** token and
    summary segments.  Computes *no* attention scores: it returns the routed KV + visibility
    masks for the caller's attention to score (matching the :class:`RoutedKV` contract).

    ``keep_first`` / ``keep_last`` mirror :class:`~kv_router.cache_store.ChunkPlacementPolicy`:
    for each query chunk the first ``keep_first`` chunks (attention sinks) and the last
    ``keep_last`` closed chunks before it are exposed **in full** — token-level (or, for the
    first window when ``first_token_level`` is False, as group summaries) — *and* excluded from
    the routed candidate pool so they are never double-counted.  With
    ``keep_first == keep_last == 0`` this reduces exactly to the original behaviour (the routing
    pool is every previous chunk and the immediately-preceding chunk is force-included).

    Returns ``(token_k, token_v, token_mask, summary_k, summary_v, summary_mask)`` in the
    chunk-parallel layout: K/V ``[B, H, N, R, Dh]`` and mask ``[B, H, N, C, R]``.
    """
    B, H, S, _ = q.shape
    Dh = cfg.head_dim
    C, gs, M = cfg.chunk_size, cfg.group_size, cfg.groups_per_chunk
    device, dtype = q.device, q.dtype
    scale = cfg.scale
    group_kv_scale = cfg.group_kv_scale

    if S == 0:
        ek = q.new_empty(B, H, 0, 0, Dh)
        em = torch.empty(B, H, 0, 0, 0, dtype=torch.bool, device=q.device)
        return ek, ek.clone(), em, ek.clone(), ek.clone(), em.clone()

    # Pad so chunk tensors are rectangular.
    N = (S + C - 1) // C
    S_pad = N * C
    valid_flat = torch.arange(S_pad, device=device) < S
    valid_chunks = valid_flat.view(N, C)                          # [N, C]
    chunk_len = valid_chunks.sum(dim=1)                           # [N]
    valid_groups = valid_chunks.view(N, M, gs).any(dim=-1)        # [N, M]
    complete_groups = valid_chunks.view(N, M, gs).all(dim=-1)     # [N, M]

    # keep_first / keep_last windows, per query chunk n (its previous count == n):
    #   first window  chunks [0, f_hi)          f_hi = min(keep_first, n)
    #   last  window  chunks [l_lo, n)          l_lo = min(max(keep_first, n - keep_last), n)
    #   routed pool   chunks [f_hi, l_lo)       (everything between the two windows)
    keep_first = max(0, int(keep_first))
    keep_last = max(0, int(keep_last))
    qchunk = torch.arange(N, device=device)
    f_hi = torch.clamp(qchunk, max=keep_first) if keep_first > 0 else torch.zeros_like(qchunk)
    if keep_last > 0:
        l_lo = torch.minimum(torch.clamp(qchunk - keep_last, min=keep_first), qchunk)
    else:
        l_lo = qchunk

    q_p = _pad_seq(q, S_pad)
    k_p = _pad_seq(k_rope, S_pad)
    v_p = _pad_seq(v, S_pad)
    k_raw_p = _pad_seq(k_raw, S_pad)

    q_chunks = q_p.reshape(B, H, N, C, Dh)
    k_chunks = k_p.reshape(B, H, N, C, Dh)
    v_chunks = v_p.reshape(B, H, N, C, Dh)
    k_raw_chunks = k_raw_p.reshape(B, H, N, C, Dh)

    chunk_start = torch.arange(N, device=device) * C
    chunk_middle = (chunk_start + (chunk_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

    group_start = torch.arange(N, device=device)[:, None] * C + torch.arange(M, device=device)[None, :] * gs
    group_len = valid_chunks.view(N, M, gs).sum(dim=-1)
    group_middle = (group_start + (group_len.clamp_min(1) - 1) // 2).clamp_max(S - 1)

    chunk_token_mask = valid_chunks.view(1, 1, N, C, 1).to(dtype)
    group_token_mask = valid_chunks.view(1, 1, N, M, gs, 1).to(dtype)

    chunk_k = _rope_summary(
        cfg, raw=k_raw_chunks, rope=k_chunks, mask=chunk_token_mask, reduce_dim=3,
        anchor_pos=chunk_middle, cos=cos, sin=sin, scale=1.0,
    )

    k_raw_groups = k_raw_chunks.reshape(B, H, N, M, gs, Dh)
    k_rope_groups = k_chunks.reshape(B, H, N, M, gs, Dh)
    v_groups_exact = v_chunks.reshape(B, H, N, M, gs, Dh)
    group_k = _rope_summary(
        cfg, raw=k_raw_groups, rope=k_rope_groups, mask=group_token_mask, reduce_dim=4,
        anchor_pos=group_middle, cos=cos, sin=sin, scale=group_kv_scale,
    )
    group_v_base = (v_groups_exact * group_token_mask).sum(dim=4) * group_kv_scale

    token_k = k_chunks.reshape(B, H, N, M, gs, Dh)
    token_v = v_chunks.reshape(B, H, N, M, gs, Dh)

    route_q_chunks = q_chunks
    neg_inf = _NEG
    query_valid = valid_chunks.view(1, 1, N, C, 1)

    # Routed candidate pool per query chunk: strictly-previous chunks that are *not* already
    # exposed by a keep_first / keep_last window, i.e. c in [f_hi(n), l_lo(n)).
    cand = torch.arange(N, device=device)
    prev_strict = cand.view(1, N) < qchunk.view(N, 1)
    in_mid = (cand.view(1, N) >= f_hi.view(N, 1)) & (cand.view(1, N) < l_lo.view(N, 1))
    route_pool = prev_strict & in_mid                                # [N(query), N(cand)]

    # 1) Choose previous chunks; expose their group summaries.
    Kc = min(cfg.topk_chunks, N) if cfg.topk_chunks > 0 else 0
    if Kc > 0:
        with torch.no_grad():
            scores_roll = torch.einsum("bhncd,bhmd->bhncm", route_q_chunks, chunk_k) * scale
            route_mask = route_pool.view(1, 1, N, 1, N) & query_valid
            scores_for_candidates = scores_roll.masked_fill(~route_mask, neg_inf)
            req_idx, req_scores = _route_topk_requests(scores_for_candidates, Kc)
            chunk_scores = _max_route_scores_from_requests(req_idx, req_scores, N, neg_inf)

        top_chunk_scores, top_chunk_idx = _topk_scores_indices(chunk_scores, Kc)
        query_chunk_idx = torch.arange(N, device=device).view(1, 1, N, 1)
        prev_chunk_idx = (query_chunk_idx - 1).clamp_min(0).expand(B, H, N, 1)
        has_prev = (query_chunk_idx > 0).expand(B, H, N, 1)
        # Force-include the immediately-preceding chunk only when it falls inside the routed
        # pool (i.e. it is not already exposed token-level by a keep_last window).
        prev_in_mid = (prev_chunk_idx >= f_hi.view(1, 1, N, 1)) & (prev_chunk_idx < l_lo.view(1, 1, N, 1))
        missing_prev = (
            has_prev & prev_in_mid
            & ~(top_chunk_idx == prev_chunk_idx).any(dim=-1, keepdim=True)
        )
        replace_at = top_chunk_scores.argmin(dim=-1, keepdim=True)
        top_chunk_idx = top_chunk_idx.scatter(
            -1, replace_at,
            torch.where(missing_prev, prev_chunk_idx, top_chunk_idx.gather(-1, replace_at)),
        )

        valid_prev = route_pool.expand(B, H, N, N).gather(-1, top_chunk_idx)
        chunk_requested = _requests_for_selected_routes(req_idx, top_chunk_idx) & valid_prev.unsqueeze(3)
        chunk_visible = torch.cumsum(chunk_requested.to(torch.int32), dim=3) > 0
        chunk_visible = chunk_visible & valid_prev.unsqueeze(3) & query_valid

        prev_slot = (top_chunk_idx == prev_chunk_idx).unsqueeze(3)
        chunk_visible = chunk_visible | (prev_slot & has_prev.unsqueeze(3) & query_valid).expand(B, H, N, C, Kc)
        chunk_visible = chunk_visible & valid_prev.unsqueeze(3)

        cand_k_groups = _gather_chunks(group_k, top_chunk_idx)       # [B,H,N,Kc,M,D]
        cand_v_groups = _gather_chunks(group_v_base, top_chunk_idx)
        cand_group_valid = (
            _gather_chunks(valid_groups.view(1, 1, N, M).expand(B, H, -1, -1), top_chunk_idx)
            & valid_prev.unsqueeze(-1)
        )
        Tgrp = Kc * M
        cand_k_groups_flat = cand_k_groups.reshape(B, H, N, Tgrp, Dh)
        cand_v_groups_flat = cand_v_groups.reshape(B, H, N, Tgrp, Dh)
        cand_group_valid_flat = cand_group_valid.reshape(B, H, N, Tgrp)

        group_summary_scores = torch.einsum("bhncd,bhnrd->bhncr", q_chunks, cand_k_groups_flat) * scale
        group_summary_visible = chunk_visible.unsqueeze(-1).expand(B, H, N, C, Kc, M).reshape(B, H, N, C, Tgrp)
        group_summary_mask = group_summary_visible & cand_group_valid_flat.unsqueeze(3) & query_valid
        group_summary_scores_masked = group_summary_scores.masked_fill(~group_summary_mask, neg_inf)
    else:
        top_chunk_idx = torch.empty(B, H, N, 0, device=device, dtype=torch.long)
        chunk_visible = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)
        Tgrp = 0
        cand_k_groups_flat = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
        cand_v_groups_flat = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
        cand_group_valid_flat = torch.empty(B, H, N, 0, device=device, dtype=torch.bool)
        group_summary_scores_masked = torch.empty(B, H, N, C, 0, device=device, dtype=dtype)
        group_summary_mask = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)

    # 2) Open selected previous groups to exact tokens.
    Kg = min(cfg.topk_groups, Tgrp) if cfg.topk_groups > 0 else 0
    Kg_request = min(cfg.topk_groups // 2, Tgrp) if cfg.topk_groups > 0 else 0
    if Kg > 0 and Kg_request > 0:
        with torch.no_grad():
            group_scores_roll = torch.einsum("bhncd,bhnrd->bhncr", route_q_chunks, cand_k_groups_flat) * scale
            group_scores_roll = group_scores_roll.masked_fill(~group_summary_mask, neg_inf)
            group_req_idx, group_req_scores = _route_topk_requests(group_scores_roll, Kg_request)
            group_scores = _max_route_scores_from_requests(group_req_idx, group_req_scores, Tgrp, neg_inf)

        _, top_group_idx = _topk_scores_indices(group_scores, Kg)
        top_group_valid = cand_group_valid_flat.gather(-1, top_group_idx)
        parent_candidate = top_group_idx // M
        parent_visible = chunk_visible.gather(-1, parent_candidate.unsqueeze(3).expand(B, H, N, C, Kg))
        group_requested = (
            _requests_for_selected_routes(group_req_idx, top_group_idx)
            & top_group_valid.unsqueeze(3)
            & parent_visible
        )
        group_visible = torch.cumsum(group_requested.to(torch.int32), dim=3) > 0
        group_visible = group_visible & top_group_valid.unsqueeze(3) & parent_visible & query_valid

        source_chunk_idx = top_chunk_idx.gather(-1, parent_candidate)
        source_group_idx = top_group_idx - parent_candidate * M
        opened_k = _gather_groups(token_k, source_chunk_idx, source_group_idx).reshape(B, H, N, Kg * gs, Dh)
        opened_v = _gather_groups(token_v, source_chunk_idx, source_group_idx).reshape(B, H, N, Kg * gs, Dh)
        opened_token_valid = _gather_groups(
            valid_chunks.view(N, M, gs).view(1, 1, N, M, gs).expand(B, H, -1, -1, -1),
            source_chunk_idx, source_group_idx,
        ).reshape(B, H, N, Kg * gs)

        opened_mask = group_visible.unsqueeze(-1).expand(B, H, N, C, Kg, gs).reshape(B, H, N, C, Kg * gs)
        opened_mask = opened_mask & opened_token_valid.unsqueeze(3) & query_valid
    else:
        opened_k = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
        opened_v = torch.empty(B, H, N, 0, Dh, device=device, dtype=dtype)
        opened_mask = torch.empty(B, H, N, C, 0, device=device, dtype=torch.bool)

    # 3) Visibility masks for the current chunk's completed-group summaries and exact tokens.
    #    (No q·k scores here — the attention computes them from the returned K/V.)
    local_t = torch.arange(C, device=device)
    group_end_local = torch.arange(M, device=device) * gs + (gs - 1)
    current_group_mask = (
        complete_groups.view(1, 1, N, 1, M)
        & (group_end_local.view(1, 1, 1, 1, M) <= local_t.view(1, 1, 1, C, 1))
        & query_valid
    ).expand(B, H, N, C, M)
    causal_in_chunk = local_t.view(1, C) <= local_t.view(C, 1)
    current_token_mask = (
        causal_in_chunk.view(1, 1, 1, C, C)
        & valid_chunks.view(1, 1, N, 1, C)
        & valid_chunks.view(1, 1, N, C, 1)
    ).expand(B, H, N, C, C)

    # Assemble the routed KV as separate token / summary segments (data only).  Order within
    # each kind is arbitrary — the attention softmax over their union is permutation-invariant.
    summary_k = torch.cat([cand_k_groups_flat, group_k], dim=3)                  # [B,H,N,Tgrp+M,Dh]
    summary_v = torch.cat([cand_v_groups_flat, group_v_base], dim=3)
    summary_mask = torch.cat([group_summary_mask, current_group_mask], dim=-1)   # [B,H,N,C,Tgrp+M]
    out_token_k = torch.cat([k_chunks, opened_k], dim=3)                         # [B,H,N,C+Kg*gs,Dh]
    out_token_v = torch.cat([v_chunks, opened_v], dim=3)
    out_token_mask = torch.cat([current_token_mask, opened_mask], dim=-1)        # [B,H,N,C,C+Kg*gs]

    # 4) Always-resident keep_first / keep_last windows (fully visible; window chunks are
    #    strictly before the query chunk, so every query token sees all their tokens).
    if keep_first > 0 or keep_last > 0:
        vc_bh = valid_chunks.view(1, 1, N, C).expand(B, H, N, C)
        vg_bh = valid_groups.view(1, 1, N, M).expand(B, H, N, M)
        query_valid6 = query_valid.view(1, 1, N, C, 1, 1)

        win_tok_k, win_tok_v, win_tok_mask = [], [], []
        win_sum_k, win_sum_v, win_sum_mask = [], [], []

        # A window spans at most ``N`` chunks regardless of how large keep_first/keep_last are,
        # so cap the materialised slot count at ``N`` — otherwise a "keep everything" config
        # (e.g. keep_last=10000) would allocate keep_last·C columns per query chunk and OOM.
        kl = min(keep_last, N)
        kf = min(keep_first, N)

        if kl > 0:                                           # last window: token-level
            j = torch.arange(kl, device=device)
            cid = l_lo.view(N, 1) + j.view(1, kl)                        # [N, kl]
            slot_valid = cid < qchunk.view(N, 1)                        # [N, kl]
            cidx = torch.where(slot_valid, cid, torch.zeros_like(cid))
            cidx = cidx.view(1, 1, N, kl).expand(B, H, N, kl)
            win_tok_k.append(_gather_chunks(k_chunks, cidx).reshape(B, H, N, kl * C, Dh))
            win_tok_v.append(_gather_chunks(v_chunks, cidx).reshape(B, H, N, kl * C, Dh))
            src_valid = _gather_chunks(vc_bh, cidx)                      # [B,H,N,kl,C]
            win_tok_mask.append((
                slot_valid.view(1, 1, N, 1, kl, 1)
                & src_valid.view(B, H, N, 1, kl, C)
                & query_valid6
            ).reshape(B, H, N, C, kl * C))

        if kf > 0:                                           # first window: token or summary
            w = torch.arange(kf, device=device)
            slot_valid = w.view(1, kf) < f_hi.view(N, 1)                # [N, kf]
            cid = torch.where(slot_valid, w.view(1, kf).expand(N, kf),
                              torch.zeros(N, kf, device=device, dtype=torch.long))  # clamp OOB slots
            cidx = cid.reshape(1, 1, N, kf).expand(B, H, N, kf)
            if first_token_level:
                win_tok_k.append(_gather_chunks(k_chunks, cidx).reshape(B, H, N, kf * C, Dh))
                win_tok_v.append(_gather_chunks(v_chunks, cidx).reshape(B, H, N, kf * C, Dh))
                src_valid = _gather_chunks(vc_bh, cidx)                  # [B,H,N,kf,C]
                win_tok_mask.append((
                    slot_valid.view(1, 1, N, 1, kf, 1)
                    & src_valid.view(B, H, N, 1, kf, C)
                    & query_valid6
                ).reshape(B, H, N, C, kf * C))
            else:
                win_sum_k.append(_gather_chunks(group_k, cidx).reshape(B, H, N, kf * M, Dh))
                win_sum_v.append(_gather_chunks(group_v_base, cidx).reshape(B, H, N, kf * M, Dh))
                src_gvalid = _gather_chunks(vg_bh, cidx)                 # [B,H,N,kf,M]
                win_sum_mask.append((
                    slot_valid.view(1, 1, N, 1, kf, 1)
                    & src_gvalid.view(B, H, N, 1, kf, M)
                    & query_valid6
                ).reshape(B, H, N, C, kf * M))

        if win_tok_k:
            out_token_k = torch.cat([out_token_k, *win_tok_k], dim=3)
            out_token_v = torch.cat([out_token_v, *win_tok_v], dim=3)
            out_token_mask = torch.cat([out_token_mask, *win_tok_mask], dim=-1)
        if win_sum_k:
            summary_k = torch.cat([summary_k, *win_sum_k], dim=3)
            summary_v = torch.cat([summary_v, *win_sum_v], dim=3)
            summary_mask = torch.cat([summary_mask, *win_sum_mask], dim=-1)

    return out_token_k, out_token_v, out_token_mask, summary_k, summary_v, summary_mask
