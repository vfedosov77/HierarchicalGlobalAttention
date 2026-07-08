# Pure-torch chunk router (fallback for hga_spda_attention).
#
# This is the routing implementation used whenever FlashAttention-3 is NOT available
# (non-Hopper GPUs, dev boxes, tests). It is kept in a separate module so the main
# attention file can stay lean on Hopper, where the FA3 routing path is used and this
# code is never imported.

from __future__ import annotations

import torch
from torch import Tensor


def route_torch(q_sum: Tensor, k_sum: Tensor, doc_of_chunk: Tensor, topk: int):
    """Per-document, strictly-causal chunk routing (pure torch).

    q_sum/k_sum: [C, H, D] (float, detached). Returns:
      sel        [C, S]  selected key-chunk indices (S = topk_eff + 1, last slot = own chunk)
      slot_valid [C, S]  True where the slot is a genuinely selected past chunk / own chunk
    """
    C = q_sum.size(0)
    dev = q_sum.device
    scores = torch.einsum("ihd,jhd->ij", q_sum, k_sum)          # [C, C]

    ii = torch.arange(C, device=dev)
    same_doc = doc_of_chunk[:, None] == doc_of_chunk[None, :]   # [C, C]
    past = ii[None, :] < ii[:, None]                            # strictly past
    valid_key = same_doc & past
    scores = scores.masked_fill(~valid_key, float("-inf"))

    topk_eff = min(int(topk), C - 1) if C > 1 else 0
    if topk_eff > 0:
        topv, topi = scores.topk(topk_eff, dim=-1)             # [C, topk_eff]
        valid_past = topv > -1e30
    else:
        topi = torch.empty(C, 0, dtype=torch.long, device=dev)
        valid_past = torch.empty(C, 0, dtype=torch.bool, device=dev)

    own = ii.reshape(C, 1)                                      # own chunk index == i
    sel = torch.cat([topi, own], dim=-1)                       # [C, S]
    slot_valid = torch.cat([valid_past, torch.ones(C, 1, dtype=torch.bool, device=dev)], dim=-1)
    return sel, slot_valid
