"""Segment-wise streaming forward for long-context routed training.

The drop-in attention (:mod:`.routed_attention`) routes whatever block the HF model hands it, via
``ChunkRouter.route_query_block``.  For long sequences we do **not** want the one-shot vectorized
prefill (which would materialize the whole block's K/V live in VRAM); we want the **incremental**
path, where each chunk is closed into the store and only the hot window stays grad-resident.  This
module drives the HF model in chunk-aligned **segments**, persisting one router/store across the
whole sequence, so:

* the **first** segment is a single chunk (``first_chunk == last_chunk`` → incremental decode path,
  never the ``start_pos == 0`` vectorized path), and
* every later segment is ``block_chunks`` chunks fed at ``start_pos > 0`` (the router's incremental
  while-loop closes them one chunk at a time).

With the Stage-2 hybrid store this keeps peak VRAM at *model + hot window* regardless of sequence
length: older chunks are detached on host RAM.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .routed_attention import RouterController
except ImportError:  # pragma: no cover - run as a top-level script
    from routed_attention import RouterController


def _segment_plan(n_chunks: int, block_chunks: int) -> List[int]:
    """Chunk counts per segment: first segment = 1 chunk (force incremental), rest = block_chunks."""
    if n_chunks <= 0:
        return []
    plan = [1]
    remaining = n_chunks - 1
    while remaining > 0:
        take = min(block_chunks, remaining)
        plan.append(take)
        remaining -= take
    return plan


def streaming_forward(
    model: Any,
    controller: RouterController,
    input_ids: torch.Tensor,          # [B, T]
    *,
    chunk_size: int,
    block_chunks: int = 8,
    device: torch.device,
    dtype: torch.dtype,
    labels: Optional[torch.Tensor] = None,   # [B, T]  (-100 ignored)
    begin: bool = True,
    start_pos: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run ``model`` over ``input_ids`` in chunk-aligned segments through the incremental router.

    Returns ``(logits[B, T, V], loss_or_None)``.  ``loss`` is next-token CE over the whole sequence
    (computed once on the concatenated logits — fine for the smoke; real 32K training should switch
    to per-segment backward / truncated BPTT to bound activation memory).
    """
    B, T = input_ids.shape
    if T % chunk_size != 0:
        raise ValueError(f"sequence length {T} must be a multiple of chunk_size {chunk_size}")
    n_chunks = T // chunk_size

    if begin:
        controller.begin(B, dtype, device, start_pos=start_pos)

    logits_segs: List[torch.Tensor] = []
    pos = start_pos
    tok = 0
    for seg_chunks in _segment_plan(n_chunks, block_chunks):
        seg_len = seg_chunks * chunk_size
        controller.start_pos = pos
        seg_ids = input_ids[:, tok : tok + seg_len].to(device)
        position_ids = torch.arange(pos, pos + seg_len, device=device).unsqueeze(0).expand(B, -1)
        out = model(input_ids=seg_ids, position_ids=position_ids, use_cache=False)
        logits_segs.append(out.logits)
        pos += seg_len
        tok += seg_len

    logits = torch.cat(logits_segs, dim=1)  # [B, T, V]

    loss: Optional[torch.Tensor] = None
    if labels is not None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().to(device)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    return logits, loss
