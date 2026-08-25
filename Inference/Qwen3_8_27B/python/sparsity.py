"""Chunk/group budgets so 1-level and 2-level share the same routed chunk count."""

from __future__ import annotations


def topk_chunks(n_closed: int, chunk_size: int = 64, keep_first: int = 2,
                keep_last: int = 8, frac_l1: float = 0.08) -> int:
    n_kv = n_closed * chunk_size
    n_win = min(keep_first + keep_last, n_closed)
    n_mid = n_closed - n_win
    if n_mid <= 0:
        return 0
    want = int(round(frac_l1 * n_kv / chunk_size) - n_win)
    return max(0, min(max(want, 1 if n_mid else 0), n_mid))


def topk_groups(n_closed: int, topk: int, levels: int, chunk_size: int = 64,
                group_size: int = 16, keep_first: int = 2, keep_last: int = 8,
                frac_l2: float = 0.04) -> int:
    gpc = chunk_size // group_size
    if topk <= 0:
        return 0
    if levels != 2:
        return topk * gpc
    n_kv = n_closed * chunk_size
    n_win_g = min(keep_first + keep_last, n_closed) * gpc
    want = int(round(frac_l2 * n_kv / group_size) - n_win_g)
    floor_g = topk  # at least one group per routed chunk
    return max(1, min(max(want, floor_g), topk * gpc))
