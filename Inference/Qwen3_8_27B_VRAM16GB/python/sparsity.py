"""Chunk/group budgets so 1-level and 2-level share the same routed chunk count."""

from __future__ import annotations

MIN_ROUTED_CHUNKS = 3
MIN_OPEN_GROUPS = 6


def routing_base_chunks(n_closed: int, n_q: int = 1, chunk_size: int = 64,
                        keep_first: int = 2, keep_last: int = 7) -> int:
    sink = min(max(0, keep_first), n_closed)
    query_chunks = min(keep_last + 1, (max(1, n_q) + chunk_size - 1) // chunk_size)
    extra_local = max(0, keep_last - max(0, query_chunks - 1))
    return max(0, n_closed - sink - min(extra_local, n_closed - sink))


def topk_chunks(n_closed: int, chunk_size: int = 64, keep_first: int = 2,
                keep_last: int = 7, frac_l1: float = 0.08,
                n_q: int = 1) -> int:
    n_win = min(keep_first + keep_last, n_closed)
    n_mid = n_closed - n_win
    if n_mid <= 0:
        return 0
    want = int(round(frac_l1 * routing_base_chunks(
        n_closed, n_q, chunk_size, keep_first, keep_last)))
    return min(max(want, MIN_ROUTED_CHUNKS), n_mid)


def topk_groups(n_closed: int, topk: int, levels: int, chunk_size: int = 64,
                group_size: int = 16, keep_first: int = 2,
                keep_last: int = 7, frac_l2: float = 0.04,
                n_q: int = 1) -> int:
    gpc = chunk_size // group_size
    if topk <= 0:
        return 0
    if levels != 2:
        return topk * gpc
    n_base = routing_base_chunks(n_closed, n_q, chunk_size, keep_first, keep_last)
    want = int(round(frac_l2 * n_base * chunk_size / group_size))
    floor_g = max(topk, MIN_OPEN_GROUPS)
    return min(max(want, floor_g), topk * gpc)
