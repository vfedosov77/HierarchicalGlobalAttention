#!/usr/bin/env python3
"""Print the 1-level vs 2-level token budget, then run the C++ microbench if built."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparsity import topk_chunks, topk_groups

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print(f"{'seq':>8} {'chunks':>8} {'Kc':>6} {'Kg_1L':>8} {'tok_1L':>8} {'frac1':>7} "
          f"{'Kg_2L':>8} {'tok_2L':>8} {'frac2':>7}")
    keep_first, keep_last, C, gs = 2, 8, 64, 16
    for seq in (4096, 8192, 16384, 32768, 65536, 131072):
        n_closed = seq // C
        kc = topk_chunks(n_closed)
        g1 = topk_groups(n_closed, kc, 1)
        g2 = topk_groups(n_closed, kc, 2)
        win = min(keep_first + keep_last, n_closed) * C
        t1 = win + g1 * gs
        t2 = win + g2 * gs
        print(f"{seq:8d} {n_closed:8d} {kc:6d} {g1:8d} {t1:8d} {t1/seq:7.3f} "
              f"{g2:8d} {t2:8d} {t2/seq:7.3f}")

    if "--run-cpp" in sys.argv:
        bench = ROOT / "build" / "hga-bench"
        if not bench.is_file():
            print(f"C++ bench not built ({bench}). Run scripts/setup.sh")
            return 1
        args = [a for a in sys.argv[1:] if a != "--run-cpp"]
        print("\n== C++ microbench ==")
        return subprocess.call([str(bench), *args])
    print("\n(pass --run-cpp to also launch build/hga-bench)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
