"""Benchmark: FA3 chunk routing (``_route_flash``) vs static-shape routing (``_route_static``).

Both live in ``hga_spda_attention`` and produce the SAME output -- a per-query-chunk top-k
selection ``(sel, slot_valid)`` with the own chunk in the last slot -- but by very different
means:

  * ``_route_static``: a fully-traceable, static-shape per-head windowed causal chunk-vs-chunk
    score over fp32 chunk mean-summaries, head-averaged, masked to the same document, then a
    plain ``topk``. This is what the compiled FlexAttention path uses (no custom ops).
  * ``_route_flash``: FlashAttention-3 (Hopper only) over the chunk summaries with a one-hot
    positional V, whose attention OUTPUT is decoded back into the routing weights. Driven here
    through ``_route_chunks`` (the production entry) so the summary pooling + ``_doc_chunk_layout``
    build are included in its cost, exactly as a real step pays them.

Conditions match ``bench_old_vs_simplified.py``:

    ~300K-token fully-packed sequence (T = 307200 -> C = 4800 chunks), 50/100/150 docs with a
    fixed-length padded cu_seqlens (training-loader layout), routing window = 22 chunks,
    topk = 3, H = 6, D = 128, chunk = 64, bf16.

Routing runs under ``no_grad`` (it is a pure selection), so this is forward-only: per-router
wall time + peak memory, plus an AGREEMENT metric between the two selections (exact-match and
Jaccard overlap of the routed past-chunk sets per query chunk). ``_route_static`` is timed both
eager and ``torch.compile``d (its production mode); ``_route_flash`` is timed eager (the FA3
kernel cannot be captured by ``fullgraph=True``).

NOTE: real FA3 requires a Hopper GPU with the ``kernels-community/flash-attn3`` kernel. When it
is unavailable (``M.fa3 is None``, e.g. an Ampere dev box) the "flash" row instead runs
``_route_onehot_sim`` -- the torch simulation that reproduces the EXACT FA3 one-hot-V routing
output -- so the script still runs and the agreement metric stays meaningful; a warning is
printed and the timing then reflects the torch sim, not the FA3 kernel.

Run:  python bench_route_flash_vs_static.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # parent dir holds the module

from _common import make_packed_batch  # noqa: E402

import hga_spda_attention as M  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16
H, D = 6, 128
CH = 64
TOPK = 3
WINDOW = 22                         # routing window in CHUNKS (bounds the near-diagonal band)
SCALE = 1.0 / (D ** 0.5)

T = 307200                          # 64 * 4800 -> C = 4800 chunks (~300K tokens)
DOC_COUNTS = [50, 100, 150]
MAX_NUM_DOCS = 256                  # fixed-length padded cu_seqlens (loader layout)
ITERS = 20
WARMUP = 5


def route_static(q, k, cu):
    """Static-shape router (production FlexAttention path). Returns (sel, slot_valid)."""
    with torch.no_grad():
        return M._route_static(q, k, cu, CH, TOPK, SCALE, WINDOW)


_FA3 = M.fa3 is not None                                     # real FA3 kernel present?


def route_flash(q, k, cu):
    """FA3 router via the production entry (summary pooling + layout build + FA3 decode).

    When the FA3 kernel is unavailable (``M.fa3 is None``) this uses ``_route_onehot_sim`` --
    the torch simulation of the exact FA3 one-hot-V routing output -- instead, so the routing
    DECISIONS are identical to what FA3 would produce (only the timing then reflects the sim).
    Returns (sel, slot_valid).
    """
    with torch.no_grad():
        layout = M._doc_chunk_layout(cu, T, CH, DEVICE)
        return M._route_chunks(
            q, k, layout, CH, SCALE, TOPK,
            use_flash=_FA3, use_onehot_sim=not _FA3, window=WINDOW,
        )


def _bench(fn, q, k, cu):
    """Return (ms_per_iter, peak_MB) for a routing callable, or (None, None) on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(WARMUP):                                 # warmup also triggers compile
            fn(q, k, cu)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn(q, k, cu)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return ms, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def _agreement(sel_a, sv_a, sel_b, sv_b, topk):
    """Compare two routers' selections: fraction of query chunks whose routed PAST-chunk set
    matches exactly, and the mean Jaccard overlap of those sets. The own chunk (last slot) is
    excluded (always selected and identical). Returns (exact_frac, mean_jaccard, n_compared)."""
    C = sel_a.shape[0]
    a_idx = sel_a[:, :topk].tolist()
    a_val = sv_a[:, :topk].tolist()
    b_idx = sel_b[:, :topk].tolist()
    b_val = sv_b[:, :topk].tolist()
    exact = 0
    jac = 0.0
    n = 0
    for c in range(C):
        A = {a_idx[c][j] for j in range(topk) if a_val[c][j]}
        B = {b_idx[c][j] for j in range(topk) if b_val[c][j]}
        if not A and not B:
            continue
        n += 1
        union = len(A | B)
        jac += (len(A & B) / union) if union else 1.0
        exact += (A == B)
    if n == 0:
        return 1.0, 1.0, 0
    return exact / n, jac / n, n


def main():
    C = T // CH
    print(f"device={torch.cuda.get_device_name()}  T={T} C={C} H={H} D={D} chunk={CH} "
          f"topk={TOPK} window={WINDOW}  docs={DOC_COUNTS}")
    if not _FA3:
        print("WARNING: FA3 kernel unavailable (M.fa3 is None) -> the 'flash' row runs the torch\n"
              "         one-hot SIM (identical routing decisions to FA3, but sim timing).")
    else:
        print("FA3 kernel: available")
    print("static: fp32-summary windowed causal top-k (compiled = FlexAttention-path mode)")
    print("flash:  FA3 one-hot-V routing via _route_chunks (summary pool + layout + FA3 decode)\n")

    static_c = torch.compile(route_static, dynamic=False, fullgraph=True)

    for nd in DOC_COUNTS:
        q, k, _v, cu = make_packed_batch(
            T, H, D, n_docs=nd, max_num_docs=MAX_NUM_DOCS, device=DEVICE, dtype=DTYPE, seed=nd,
        )
        header = f"T={T} C={C} docs={nd:4d}"

        def fmt(ms, pk):
            return "OOM" if ms is None else f"{ms:8.3f}ms {pk:7.0f}MB"

        se_ms, se_pk = _bench(route_static, q, k, cu)
        sc_ms, sc_pk = _bench(static_c, q, k, cu)
        fl_ms, fl_pk = _bench(route_flash, q, k, cu)
        print(f"{header}  static/eager    {fmt(se_ms, se_pk)}")
        print(f"{header}  static/compiled {fmt(sc_ms, sc_pk)}")
        print(f"{header}  flash /eager    {fmt(fl_ms, fl_pk)}")

        # Agreement between the two selections (own chunk excluded).
        sel_s, sv_s = route_static(q, k, cu)
        sel_f, sv_f = route_flash(q, k, cu)
        ex, jac, n = _agreement(
            sel_s.cpu(), sv_s.cpu(), sel_f.cpu(), sv_f.cpu(), TOPK,
        )
        print(f"{header}  agree           exact={ex:6.2%}  jaccard={jac:6.2%}  "
              f"(over {n} routed query chunks)\n")

        del q, k, _v, cu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
