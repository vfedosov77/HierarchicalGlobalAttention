"""Vendored DCA helpers from HKUNLP/ChunkLlama (``chunkqwen_attn_replace.py``).

Source: https://github.com/HKUNLP/ChunkLlama
Paper: Training-Free Long-Context Scaling of Large Language Models (An et al., 2024)

Only the geometry constants and log-sum-exp path merge are reused here.  Flash-attn
forward / dense KV-cache paths stay in upstream ChunkLlama; this repo adapts the
three-path merge for sparse KvRouter segments in ``dca_rope.dca_attend``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def dca_geometry(
    pretraining_length: int,
    local_window_size: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Return ``(chunk_size, local_window, chunk_len)`` per ``replace_with_chunkqwen``."""
    chunk_size = pretraining_length * 3 // 4
    local_window = local_window_size if local_window_size else pretraining_length // 16
    return chunk_size, local_window, chunk_size - local_window


def merge_attn_path_outputs(
    paths: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Stable LSE merge for parallel DCA paths (intra / successive / inter).

    Extracted from ChunkLlama ``merge_attn_outputs`` (per-chunk branch).
    Each path is ``(attn_output, softmax_lse)`` with shapes
    ``[B, H, L, D]`` and ``[B, H, L]``.
    """
    attn_outputs = torch.stack([p[0] for p in paths])
    logits = torch.stack([p[1] for p in paths])
    max_logits = torch.max(logits, dim=0).values
    stable_logits = logits - max_logits.unsqueeze(0)
    lse_s = torch.exp(stable_logits).detach()
    lse_sum = torch.sum(lse_s, dim=0)
    lse_s /= lse_sum.clamp_min(1e-9)
    return (attn_outputs * lse_s.unsqueeze(-1)).sum(dim=0).to(out_dtype)