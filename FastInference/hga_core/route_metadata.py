"""Engine-neutral routing metadata.

``RouteMetadata`` is the contract between an engine's scheduler/model-runner and
the HGA attention backend.  It is deliberately framework-agnostic: it carries
only plain tensors and ints, with **no** dependency on ``DynamicCache``,
``ForwardBatch`` or any vLLM/SGLang type.  Both the SGLang backend and the
(future) vLLM backend populate it from their own batch objects, and both the
fused kernels and the cache manager consume it.

A ``RouteMetadata`` describes, *for one attention layer and one forward step*,
which token-level keys each query may attend to.  It is produced by the routing
stage (chunk top-k then group top-k) and is stable enough to be reused across
the small block of tokens that share a routing decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RouteMetadata:
    """Routing selection for a single (layer, forward) — engine-neutral.

    Index tensors address *chunks* and *groups* in the per-request cold KV
    record; the cache manager turns them into concrete GPU token banks.
    """

    # batch / shape
    batch_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    # absolute number of CLOSED chunks in context (per request); scalar if batch=1
    n_closed: int

    # ---- chunk-level selection ----
    # routed-middle chunk ids [B, H, Kc] (absolute chunk index in the record)
    mid_chunk_idx: torch.Tensor
    # causal visibility of each routed chunk to each query position [B, H, L, Kc]
    mid_vis: torch.Tensor

    # ---- group-level selection ----
    # parent chunk id of each opened group [B, H, Kg]
    open_chunk_idx: torch.Tensor
    # group index within its parent chunk [B, H, Kg]
    open_group_idx: torch.Tensor
    # causal visibility of each opened group to each query position [B, H, L, Kg]
    open_vis: torch.Tensor

    # ---- deterministic windows (chunk-id ranges, half-open) ----
    first_lo: int
    first_hi: int
    last_lo: int
    last_hi: int

    # active (partial) chunk: absolute start position of its first token
    active_start: int

    # optional: whether group summaries are attended in addition to opened tokens
    use_summaries: bool = True

    # optional: page-table view (filled by the cache manager) — token-bank slots
    # for the opened/window chunks, used by the fused gather+attention kernel.
    page_table: Optional[torch.Tensor] = None

    @property
    def num_routed_chunks(self) -> int:
        return int(self.mid_chunk_idx.shape[-1])

    @property
    def num_opened_groups(self) -> int:
        return int(self.open_chunk_idx.shape[-1])

    def to(self, device: torch.device) -> "RouteMetadata":
        def mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t

        return RouteMetadata(
            batch_size=self.batch_size,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            n_closed=self.n_closed,
            mid_chunk_idx=mv(self.mid_chunk_idx),
            mid_vis=mv(self.mid_vis),
            open_chunk_idx=mv(self.open_chunk_idx),
            open_group_idx=mv(self.open_group_idx),
            open_vis=mv(self.open_vis),
            first_lo=self.first_lo,
            first_hi=self.first_hi,
            last_lo=self.last_lo,
            last_hi=self.last_hi,
            active_start=self.active_start,
            use_summaries=self.use_summaries,
            page_table=mv(self.page_table),
        )
