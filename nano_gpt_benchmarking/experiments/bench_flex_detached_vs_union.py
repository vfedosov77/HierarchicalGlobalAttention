"""Benchmark: _flex_attend (single union call) vs _flex_attend_detached (two calls,
routed K/V detached) on a VERY LONG packed sequence (~300K tokens) with many documents.

Both variants use the production held ``_RoutedFlexMaskState`` (persistent BlockMask
buffers, only the routed band updated in place) and are compiled the way production
compiles them: ``torch.compile(dynamic=False, fullgraph=True)``. We route once with
``_route_static`` and time the two attend variants on the SAME selection. Reports
forward-only and forward+backward wall time and peak memory, and which is faster.

It also verifies that changing ONLY the document count (with a fixed-length, padded
``cu_seqlens`` -- the training loader's layout) does NOT trigger a recompilation of the
full compiled ``chunk_routed_flex_attention`` entry (which includes routing). A recompile
there would mean per-doc-count graph thrash; the script reports an ERROR if it happens.

Run:  python bench_flex_detached_vs_union.py
"""
import os
import sys

import torch
import torch._dynamo as dyn
from torch._dynamo.utils import counters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import M, make_packed_batch  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16
H, D = 6, 128
CH = 64
TOPK = 3
WINDOW = 22                         # routing window in CHUNKS (bounds the near-diagonal band)
SCALE = 1.0 / (D ** 0.5)

# ~300K tokens (multiple of CH): 307200 = 64 * 4800 -> C = 4800 chunks.
T = 307200
# Document counts to sweep (the "50-150 docs" regime).
DOC_COUNTS = [50, 100, 150]
# Fixed-length padded cu_seqlens (training-loader layout): doc count varies as VALUES
# only, never as a shape -> a well-behaved graph must not recompile across doc counts.
MAX_NUM_DOCS = 256
ITERS = 6
WARMUP = 3


def _bench(fn, q, k, v, sel, sv, win, state, backward):
    """Return (ms_per_iter, peak_MB) for fn, or (None, None) on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(WARMUP):                                 # warmup also triggers compile
            if backward:
                q.grad = k.grad = v.grad = None
                y = fn(q, k, v, sel, sv, CH, SCALE, win, state)
                y.float().pow(2).mean().backward()
            else:
                with torch.no_grad():
                    fn(q, k, v, sel, sv, CH, SCALE, win, state)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            if backward:
                q.grad = k.grad = v.grad = None
                y = fn(q, k, v, sel, sv, CH, SCALE, win, state)
                y.float().pow(2).mean().backward()
            else:
                with torch.no_grad():
                    fn(q, k, v, sel, sv, CH, SCALE, win, state)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return ms, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def check_recompile_on_doc_count(detach_routed):
    """Compile the full chunk_routed_flex_attention entry once, then sweep doc counts
    (fixed-length cu_seqlens). unique_graphs must stay constant -> no recompilation.
    Returns (ok: bool, per_doc_graph_counts: list[(ndoc, graphs)])."""
    C = T // CH
    dyn.reset()
    counters.clear()
    fn = torch.compile(M.chunk_routed_flex_attention, dynamic=False, fullgraph=True)
    state = M._RoutedFlexMaskState(C, T, CH, DEVICE)            # held across all doc counts

    per_doc = []
    baseline = None
    ok = True
    for nd in DOC_COUNTS:
        q, k, v, cu = make_packed_batch(
            T, H, D, n_docs=nd, max_num_docs=MAX_NUM_DOCS, device=DEVICE, dtype=DTYPE, seed=nd,
        )
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        y = fn(qg, kg, vg, cu, chunk_size=CH, topk=TOPK, softmax_scale=SCALE,
               window=WINDOW, detach_routed=detach_routed, mask_state=state)
        y.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        graphs = counters["stats"].get("unique_graphs", 0)
        if baseline is None:
            baseline = graphs                                   # after the first (compiling) run
        elif graphs > baseline:
            ok = False                                          # a new graph == a recompile
        per_doc.append((nd, graphs))
        del q, k, v, qg, kg, vg, y
        torch.cuda.empty_cache()
    return ok, per_doc


def main():
    C = T // CH
    print(f"device={torch.cuda.get_device_name()}  T={T} C={C} H={H} D={D} chunk={CH} "
          f"topk={TOPK} window={WINDOW}  docs={DOC_COUNTS}")

    # ------------------------------------------------------------------ speed comparison
    union = torch.compile(M._flex_attend, dynamic=False, fullgraph=True)
    detached = torch.compile(M._flex_attend_detached, dynamic=False, fullgraph=True)
    win = min(WINDOW, C - 1)
    state = M._RoutedFlexMaskState(C, T, CH, DEVICE)            # shared held mask state

    for nd in DOC_COUNTS:
        q, k, v, cu = make_packed_batch(
            T, H, D, n_docs=nd, max_num_docs=MAX_NUM_DOCS, device=DEVICE, dtype=DTYPE, seed=nd,
        )
        with torch.no_grad():
            sel, sv = M._route_static(q, k, cu, CH, TOPK, SCALE, win)

        header = f"T={T} C={C} docs={nd:4d}"
        for backward in (False, True):
            tag = "fwd+bwd" if backward else "fwd    "
            qg = q.clone().requires_grad_(backward)
            kg = k.clone().requires_grad_(backward)
            vg = v.clone().requires_grad_(backward)
            u_ms, u_pk = _bench(union, qg, kg, vg, sel, sv, win, state, backward)
            d_ms, d_pk = _bench(detached, qg, kg, vg, sel, sv, win, state, backward)

            def fmt(ms, pk):
                return "OOM" if ms is None else f"{ms:8.2f}ms {pk:7.0f}MB"

            if u_ms and d_ms:
                faster = "detached" if d_ms < u_ms else "union"
                ratio = max(u_ms, d_ms) / min(u_ms, d_ms)
                verdict = f"  -> {faster} faster ({ratio:.2f}x)"
            else:
                verdict = ""
            print(f"{header} {tag}  union={fmt(u_ms, u_pk)}  "
                  f"detached={fmt(d_ms, d_pk)}{verdict}")
            del qg, kg, vg
            torch.cuda.empty_cache()
        del q, k, v, sel, sv
        torch.cuda.empty_cache()

    # ------------------------------------------------------ recompilation-on-doc-count check
    print("\n[recompile check] varying doc count with fixed-length cu_seqlens "
          f"(max_num_docs={MAX_NUM_DOCS}):")
    all_ok = True
    for detach_routed in (False, True):
        ok, per_doc = check_recompile_on_doc_count(detach_routed)
        counts = ", ".join(f"docs={nd}:graphs={g}" for nd, g in per_doc)
        mode = "detached" if detach_routed else "union   "
        status = "OK (no recompile)" if ok else "ERROR: RECOMPILATION TRIGGERED BY DOC COUNT"
        print(f"  {mode}  {counts}  -> {status}")
        all_ok = all_ok and ok
        torch.cuda.empty_cache()

    if not all_ok:
        print("\nERROR: document-count changes forced a recompilation (see above).")
        sys.exit(1)
    print("\nOK: document-count changes do NOT force a recompilation.")


if __name__ == "__main__":
    main()
