"""Fused routing top-k for HGA.

Two cheap selection stages run before the expensive token attention:

* :func:`chunk_topk` — score the query block against compact ``chunk_summary_k``
  and pick the strongest ``topk_chunks`` candidate chunks.
* :func:`group_topk` — score the selected chunks' ``group_summary_k`` and open
  the strongest ``topk_groups`` groups to exact token attention.

Both summaries are tiny (for 32K context there are only 512 chunks), so the
score matmul + ``torch.topk`` is already a small, GPU-resident, fused op.  These
helpers centralise the pooling + causal-visibility logic so the SGLang and vLLM
backends share one implementation and produce identical selections to the
reference router.

The heavy decode-speed lever is :func:`~FastInference.hga_core.kernels.decode_attention.fused_decode_attention`;
these routing helpers are intentionally kept simple and correct.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def chunk_topk(
    q: torch.Tensor,            # [B, H, L, Dh]  (query, head-expanded)
    chunk_k: torch.Tensor,      # [B, H, n_mid, Dh]  candidate chunk summaries
    topk: int,
    scale: float,
    force_prev_rel: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the top-``topk`` candidate chunks for the query block.

    Returns ``(mid_rel[B,H,Kc], mid_vis[B,H,L,Kc], scores[B,H,L,n_mid])``.

    * ``mid_rel`` — relative chunk ids (offsets into the candidate pool), pooled
      across the ``L`` query positions (one fetch per block).
    * ``mid_vis`` — causal visibility: chunk slot ``k`` is visible to position
      ``t`` iff some position ``<= t`` requested it (cumulative-OR over the block),
      mirroring the vectorized-prefill cumsum mask.
    * ``scores`` — raw ``q·chunk_k`` scores (returned for the group stage / debug).
    """
    B, H, L, _ = q.shape
    n_mid = chunk_k.shape[2]
    Kc = min(topk, n_mid)
    sc = torch.einsum("bhld,bhnd->bhln", q, chunk_k) * scale          # [B,H,L,n_mid]
    pooled = sc.max(dim=2).values                                    # [B,H,n_mid]
    top_scores, mid_rel = torch.topk(pooled, Kc, dim=-1, sorted=False)
    _, req_rel = torch.topk(sc, Kc, dim=-1, sorted=False)            # [B,H,L,Kc]

    if force_prev_rel is not None and 0 <= force_prev_rel < n_mid:
        missing = ~(mid_rel == force_prev_rel).any(dim=-1, keepdim=True)
        replace_at = top_scores.argmin(dim=-1, keepdim=True)
        mid_rel = mid_rel.scatter(
            -1, replace_at,
            torch.where(missing, torch.full_like(replace_at, force_prev_rel),
                        mid_rel.gather(-1, replace_at)),
        )

    requested = (req_rel.unsqueeze(-2) == mid_rel.unsqueeze(2).unsqueeze(-1)).any(dim=-1)
    mid_vis = torch.cumsum(requested.to(torch.int32), dim=2) > 0     # [B,H,L,Kc]
    if force_prev_rel is not None and 0 <= force_prev_rel < n_mid:
        mid_vis = mid_vis | (mid_rel == force_prev_rel).unsqueeze(2)
    return mid_rel, mid_vis, sc


def group_topk(
    q: torch.Tensor,            # [B, H, L, Dh]
    group_k: torch.Tensor,      # [B, H, Kc, M, Dh]  group summaries of routed chunks
    mid_idx: torch.Tensor,      # [B, H, Kc]  absolute chunk ids of the routed chunks
    mid_vis: torch.Tensor,      # [B, H, L, Kc]  chunk visibility (parent gating)
    topk: int,
    topk_request: int,
    scale: float,
    groups_per_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Open the top-``topk`` groups across the routed chunks.

    Returns ``(open_chunk[B,H,Kg], open_grp[B,H,Kg], open_vis[B,H,L,Kg])``.
    ``open_chunk`` is the absolute parent chunk id; ``open_grp`` the group index
    within that chunk.  Visibility is causal *and* gated by parent-chunk
    visibility, exactly like the reference ``_route_decision``.
    """
    B, H, Kc, M, Dh = group_k.shape
    device = q.device
    L = q.shape[2]
    Kg = min(topk, Kc * M) if topk > 0 else 0
    if Kg <= 0:
        empty = torch.empty(B, H, 0, dtype=torch.long, device=device)
        empty_vis = torch.empty(B, H, L, 0, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_vis
    Kg_req = min(topk_request, Kc * M)
    gk_flat = group_k.reshape(B, H, Kc * M, Dh)
    sc_g = torch.einsum("bhld,bhrd->bhlr", q, gk_flat) * scale       # [B,H,L,Kc*M]
    pooled_g = sc_g.max(dim=2).values
    _, top_g = torch.topk(pooled_g, Kg, dim=-1, sorted=False)        # [B,H,Kg]
    _, req_g = torch.topk(sc_g, Kg_req, dim=-1, sorted=False)        # [B,H,L,Kg_req]
    parent = top_g // M
    open_chunk = mid_idx.gather(-1, parent)
    open_grp = top_g - parent * M

    grp_req = (req_g.unsqueeze(-2) == top_g.unsqueeze(2).unsqueeze(-1)).any(dim=-1)
    open_vis = torch.cumsum(grp_req.to(torch.int32), dim=2) > 0
    parent_vis = torch.gather(mid_vis, -1, parent.unsqueeze(2).expand(B, H, L, Kg))
    open_vis = open_vis & parent_vis
    return open_chunk, open_grp, open_vis
