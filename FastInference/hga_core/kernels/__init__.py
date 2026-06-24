"""Fused kernels for HGA inference.

* :func:`fused_decode_attention` — single-pass flash-style online softmax over
  the already-assembled routed K/V (sink + local + opened groups + active chunk,
  plus optional group summaries).  Replaces the reference
  ``einsum -> mask -> softmax -> einsum`` in ``RoutedKV.attend``.
* :func:`chunk_topk` / :func:`group_topk` — routing top-k over compact summaries.
"""

from .decode_attention import fused_decode_attention, HAS_TRITON
from .route_topk import chunk_topk, group_topk

__all__ = ["fused_decode_attention", "chunk_topk", "group_topk", "HAS_TRITON"]
