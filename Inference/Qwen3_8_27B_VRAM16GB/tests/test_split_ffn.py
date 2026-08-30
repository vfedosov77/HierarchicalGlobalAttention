#!/usr/bin/env python3
"""Unit tests for split-FFN tiling, packing, budget, and EDF order."""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def dense_swiglu(x, up, gate, down):
    """x [T,H], up/gate [H,I], down [I,H] -> [T,H] float64."""
    t, h = len(x), len(x[0])
    inter = len(up[0])
    out = [[0.0] * h for _ in range(t)]
    for ti in range(t):
        up_p = [0.0] * inter
        gate_p = [0.0] * inter
        for j in range(inter):
            u = g = 0.0
            for i in range(h):
                u += x[ti][i] * up[i][j]
                g += x[ti][i] * gate[i][j]
            up_p[j] = u
            gate_p[j] = silu(g)
        gated = [up_p[j] * gate_p[j] for j in range(inter)]
        for i in range(h):
            s = 0.0
            for j in range(inter):
                s += gated[j] * down[j][i]
            out[ti][i] = s
    return out


def tiled_swiglu(x, up, gate, down, tile_ch: int):
    t, h = len(x), len(x[0])
    inter = len(up[0])
    acc = [[0.0] * h for _ in range(t)]
    for begin in range(0, inter, tile_ch):
        end = min(begin + tile_ch, inter)
        width = end - begin
        for ti in range(t):
            up_p = [0.0] * width
            gate_p = [0.0] * width
            for j, jj in enumerate(range(begin, end)):
                u = g = 0.0
                for i in range(h):
                    u += x[ti][i] * up[i][jj]
                    g += x[ti][i] * gate[i][jj]
                up_p[j] = u
                gate_p[j] = silu(g)
            gated = [up_p[j] * gate_p[j] for j in range(width)]
            for i in range(h):
                s = 0.0
                for j, jj in enumerate(range(begin, end)):
                    s += gated[j] * down[jj][i]
                acc[ti][i] += s
    return acc


def can_slice(blck: int, n_embd: int, n_ff: int, tile_ch: int) -> bool:
    if blck <= 0 or tile_ch <= 0 or n_embd <= 0 or n_ff <= 0:
        return False
    if n_ff % tile_ch != 0:
        return False
    return n_embd % blck == 0 and n_ff % blck == 0 and tile_ch % blck == 0


def plan_slots(budget: int, core: int, tile: int, n_tiles: int, min_slots: int) -> int:
    if tile == 0 or budget <= core:
        return 0
    n = min(n_tiles, (budget - core) // tile)
    return n if n >= min_slots else 0


def edf_cmp(prio_a, dl_a, tile_a, prio_b, dl_b, tile_b) -> int:
    if prio_a != prio_b:
        return prio_a - prio_b
    if dl_a != dl_b:
        return -1 if dl_a < dl_b else 1
    if tile_a != tile_b:
        return tile_a - tile_b
    return 0


def cyclic_target(layer: int, which: int) -> int:
    targets = [16, 24, 48, 56]
    ranked = []
    for t in targets:
        dist = (t - layer + 64) % 64
        if dist == 0:
            continue
        ranked.append((dist, t))
    ranked.sort()
    if which < 0 or which >= len(ranked):
        return -1
    return ranked[which][1]


def pack_down(src: bytes, n_rows: int, src_row: int, dst_row: int, col_off: int) -> bytes:
    out = bytearray()
    for r in range(n_rows):
        start = r * src_row + col_off
        out.extend(src[start : start + dst_row])
    return bytes(out)


class SplitFfnMathTest(unittest.TestCase):
    def test_tiled_swiglu_matches_dense(self) -> None:
        h, inter, tile, t = 8, 16, 4, 3
        rng = 1
        def rnd():
            nonlocal rng
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            return (rng / 0x7FFFFFFF) * 2.0 - 1.0

        x = [[rnd() for _ in range(h)] for _ in range(t)]
        up = [[rnd() for _ in range(inter)] for _ in range(h)]
        gate = [[rnd() for _ in range(inter)] for _ in range(h)]
        down = [[rnd() for _ in range(h)] for _ in range(inter)]
        ref = dense_swiglu(x, up, gate, down)
        got = tiled_swiglu(x, up, gate, down, tile)
        for ti in range(t):
            for i in range(h):
                self.assertAlmostEqual(ref[ti][i], got[ti][i], places=9)

    def test_qwen38_tile_divides_exactly(self) -> None:
        self.assertEqual(17408 / 1024, 17)
        self.assertTrue(can_slice(256, 5120, 17408, 1024))
        self.assertTrue(can_slice(32, 5120, 17408, 1024))
        self.assertFalse(can_slice(256, 5120, 17408, 1000))
        self.assertFalse(can_slice(256, 5119, 17408, 1024))

    def test_ffn_down_repack_strided(self) -> None:
        n_rows, n_ff, tile, blck = 4, 16, 4, 4
        src_row = n_ff  # 1 byte per element
        src = bytes(range(n_rows * src_row))
        tile_index = 2
        got = pack_down(src, n_rows, src_row, tile, tile_index * tile)
        expected = bytearray()
        for r in range(n_rows):
            expected.extend(src[r * src_row + 8 : r * src_row + 12])
        self.assertEqual(got, bytes(expected))
        self.assertEqual(len(got), n_rows * tile)

    def test_budget_requires_two_slots_and_caps_at_17(self) -> None:
        self.assertEqual(plan_slots(100, 80, 10, 17, 2), 2)
        self.assertEqual(plan_slots(100, 80, 15, 17, 2), 0)
        self.assertEqual(plan_slots(300, 50, 10, 17, 2), 17)
        self.assertEqual(plan_slots(90, 90, 10, 17, 2), 0)

    def test_dynamic_bank_uses_all_slots_before_permanent_tiles(self) -> None:
        n = plan_slots(200, 40, 10, 17, 2)
        self.assertEqual(n, 16)
        self.assertLess(n, 17)

    def test_edf_current_missing_outranks_prefix(self) -> None:
        self.assertLess(edf_cmp(0, 100, 5, 1, 16, 0), 0)
        self.assertLess(edf_cmp(1, 24, 0, 2, 48, 0), 0)
        self.assertLess(edf_cmp(1, 48, 0, 1, 56, 0), 0)
        self.assertLess(edf_cmp(1, 48, 3, 1, 48, 4), 0)

    def test_cyclic_order_16_24_48_56(self) -> None:
        self.assertEqual(cyclic_target(0, 0), 16)
        self.assertEqual(cyclic_target(0, 1), 24)
        self.assertEqual(cyclic_target(17, 0), 24)
        self.assertEqual(cyclic_target(17, 1), 48)
        self.assertEqual(cyclic_target(25, 0), 48)
        self.assertEqual(cyclic_target(25, 1), 56)
        self.assertEqual(cyclic_target(49, 0), 56)
        self.assertEqual(cyclic_target(49, 1), 16)
        self.assertEqual(cyclic_target(57, 0), 16)
        self.assertEqual(cyclic_target(57, 1), 24)

    def test_slot_wraparound_n10(self) -> None:
        n_slots, n_tiles = 10, 17
        schedule = []
        for tile in range(n_tiles):
            slot = tile % n_slots
            schedule.append((slot, tile))
        refill = []
        for tile in range(n_tiles):
            slot = tile % n_slots
            if tile + n_slots >= n_tiles:
                refill.append((slot, "next", slot))
        self.assertEqual(schedule[0], (0, 0))
        self.assertEqual(schedule[10], (0, 10))
        self.assertEqual(schedule[16], (6, 16))
        # After last use, every slot refills the counterpart prefix at the same index.
        self.assertEqual({s for s, kind, t in refill if kind == "next"}, set(range(n_slots)))
        self.assertIn((7, "next", 7), refill)
        self.assertIn((0, "next", 0), refill)

    def test_phase_reset_is_idempotent(self) -> None:
        state = {"pass": 3, "jobs": [1, 2], "occ": {(0, 0): (16, 3)}}
        state = {"pass": 0, "jobs": [], "occ": {}}
        self.assertEqual(state["pass"], 0)
        self.assertEqual(state["jobs"], [])
        self.assertEqual(state["occ"], {})


if __name__ == "__main__":
    unittest.main()
