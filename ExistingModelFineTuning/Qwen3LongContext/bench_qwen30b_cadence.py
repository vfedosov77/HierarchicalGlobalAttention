#!/usr/bin/env python3
"""Routing-cadence decode-latency benchmark.

Measures how much routing amortization (``HGA_ROUTE_CADENCE``) speeds up decode, reporting the
only two figures that matter to a caller:

    wall   = host wall-time per decode step (``perf_counter``, CUDA-synced),
    tok/s  = decode throughput = 1000 / wall.

Isolation: each (context, cadence) point is a **fresh subprocess**.  The router reads
``HGA_ROUTE_CADENCE`` once, in ``ChunkRouter.__init__``, so a new process guarantees a clean
router/allocator/LRU state per point (stronger than rebuilding the cache in one process).  The
*outer* driver launches those runs and tabulates their wall-time; the hidden *inner*
(``--inner``) process builds the model, prefills the context, and times the decode window.

The outer driver sweeps every context in ``--bench-ctxs`` x every cadence in ``--bench-cadences``
and writes an HTML results table to ``--html`` (default ``docs/cadence_bench.html``).

Run (from the repo root, venv active; small model on a weak GPU):

    HGA_MODEL=Qwen/Qwen3-0.6B \
      python -m ExistingModelFineTuning.Qwen3LongContext.bench_qwen30b_cadence \
        --cache ram --bench-ctxs 8192,32768,65536,130816 --bench-cadences 1,64
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json
import os
import subprocess
import sys
import tempfile
import time

# Repo root (two levels up: Qwen3LongContext -> ExistingModelFineTuning -> repo root); the inner
# run is launched as a package module from here so ``from ExistingModelFineTuning...`` imports work.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODULE = "ExistingModelFineTuning.Qwen3LongContext.bench_qwen30b_cadence"


# ---------------------------------------------------------------------------
# Small pure helpers (covered by --selftest)
# ---------------------------------------------------------------------------

def _fmt_ctx(n: int) -> str:
    """Human context label, e.g. 130816 -> '128K', 8192 -> '8K'."""
    return f"{n / 1024:.0f}K"


def _tok_per_s(wall_ms: float) -> float:
    return 1000.0 / wall_ms if wall_ms > 0 else 0.0


# ---------------------------------------------------------------------------
# Shared: real long context
# ---------------------------------------------------------------------------

def _bench_context_ids(tok, n: int):
    """A real long context of ``n`` tokens, tiled from the bundled training text."""
    import torch  # local: only the inner (GPU) process needs torch

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "TrainData",
                     "The-Master-and-Margarita.txt")
    )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ids = tok(f.read(), return_tensors="pt").input_ids
    if ids.shape[1] < n:
        ids = ids.repeat(1, n // ids.shape[1] + 1)
    return ids[:, :n].contiguous()


# ---------------------------------------------------------------------------
# Inner run: build model, prefill, time the decode window
# ---------------------------------------------------------------------------

def _run_inner(args) -> None:
    import torch
    from transformers import DynamicCache

    from ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8 import (
        PREFILL_BLOCK, _dispose_cache, build_model,
    )

    # The router reads this in ChunkRouter.__init__; the outer already sets it in the child env,
    # we re-assert it here so a standalone ``--inner`` invocation is self-contained.
    os.environ["HGA_ROUTE_CADENCE"] = str(args.cadence)
    device = "cuda"

    model, tok = build_model(args.cache, verbose=False)
    ids_full = _bench_context_ids(tok, args.bench_ctx).to(device)

    cache = DynamicCache()
    pos = args.bench_ctx
    with torch.inference_mode():
        last = None
        for s in range(0, args.bench_ctx, PREFILL_BLOCK):
            e = min(s + PREFILL_BLOCK, args.bench_ctx)
            cp = torch.arange(s, e, device=device)
            out = model(input_ids=ids_full[:, s:e], past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            last = out.logits[:, -1]
        torch.cuda.synchronize()
        nxt = int(last.argmax(-1))

        def step(nxt: int, pos: int):
            cp = torch.tensor([pos], device=device)
            out = model(input_ids=torch.tensor([[nxt]], device=device),
                        past_key_values=cache, cache_position=cp,
                        position_ids=cp.unsqueeze(0), use_cache=True)
            return int(out.logits[:, -1].argmax(-1)), pos + 1

        for _ in range(args.bench_warmup):
            nxt, pos = step(nxt, pos)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.bench_steps):
            nxt, pos = step(nxt, pos)
        torch.cuda.synchronize()
        wall_ms_total = (time.perf_counter() - t0) * 1000.0

    _dispose_cache(cache)
    with open(args.wall_json, "w") as f:
        json.dump({"cadence": args.cadence, "ctx": args.bench_ctx,
                   "steps": args.bench_steps, "wall_ms_total": wall_ms_total}, f)
    print(f"[inner] cadence={args.cadence} ctx={args.bench_ctx} "
          f"wall={wall_ms_total / args.bench_steps:.2f} ms/tok", flush=True)


# ---------------------------------------------------------------------------
# Outer driver: one subprocess per (context, cadence) point
# ---------------------------------------------------------------------------

def _run_point(cad: int, ctx: int, args, workdir: str) -> dict:
    base = os.path.join(workdir, f"ctx{ctx}_cad{cad}")
    wall_json = base + ".wall.json"
    inner_cmd = [
        sys.executable, "-m", _MODULE, "--inner",
        "--cadence", str(cad),
        "--cache", args.cache,
        "--bench-ctx", str(ctx),
        "--bench-steps", str(args.bench_steps),
        "--bench-warmup", str(args.bench_warmup),
        "--wall-json", wall_json,
    ]
    env = dict(os.environ)
    env["HGA_ROUTE_CADENCE"] = str(cad)
    proc = subprocess.run(inner_cmd, cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise RuntimeError(f"inner run failed for ctx={ctx} cadence={cad} "
                           f"(exit {proc.returncode})")

    with open(wall_json) as f:
        wj = json.load(f)
    wall = wj["wall_ms_total"] / wj["steps"]
    return {"ctx": ctx, "cad": cad, "wall": wall, "toks": _tok_per_s(wall)}


def _build_html(rows: list[dict], meta: dict) -> str:
    """Results table: one row per (context, cadence) with wall ms/tok and tok/s."""
    head = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Routing-cadence decode benchmark</title>"
        "<style>"
        "body{font:14px system-ui,sans-serif;margin:2rem;color:#111}"
        "table{border-collapse:collapse;margin-top:.5rem}"
        "th,td{border:1px solid #ccc;padding:.35rem .7rem;text-align:right}"
        "th{background:#f2f2f2}td:first-child,th:first-child{text-align:left}"
        "caption{text-align:left;font-weight:600;padding-bottom:.4rem}"
        "</style>"
    )
    m = " &middot; ".join(
        f"{_html.escape(k)}: {_html.escape(str(v))}" for k, v in meta.items()
    )
    body = [head,
            "<h2>Routing-cadence decode benchmark</h2>",
            f"<p>{m}</p>",
            "<table>",
            "<caption>wall ms/tok &amp; tok/s per context and cadence</caption>",
            "<tr><th>Context</th><th>Cadence</th><th>wall ms/tok</th><th>tok/s</th></tr>"]
    for r in rows:
        body.append(
            f"<tr><td>{_fmt_ctx(r['ctx'])}</td><td>{r['cad']}</td>"
            f"<td>{r['wall']:.2f}</td><td>{r['toks']:.1f}</td></tr>"
        )
    body.append("</table>")
    return "".join(body)


def _run_outer(args) -> None:
    ctxs = [int(c) for c in args.bench_ctxs.split(",") if c.strip()]
    cadences = [int(c) for c in args.bench_cadences.split(",") if c.strip()]
    print(
        f"\nRouting-cadence decode benchmark  |  decode={args.bench_steps} tok "
        f"(warmup {args.bench_warmup})  cache={args.cache}\n"
        f"cadence=1 = per-token routing (reference); higher = chunk-shared routing.\n"
        f"{'ctx':>8} {'cadence':>8} {'wall ms/tok':>12} {'tok/s':>8}",
        flush=True,
    )
    print("-" * 40, flush=True)

    workdir = tempfile.mkdtemp(prefix="cadence_bench_")
    rows: list[dict] = []
    try:
        for ctx in ctxs:
            for cad in cadences:
                r = _run_point(cad, ctx, args, workdir)
                rows.append(r)
                print(f"{_fmt_ctx(ctx):>8} {cad:>8} {r['wall']:>12.2f} {r['toks']:>8.1f}",
                      flush=True)
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    meta = {
        "model": os.environ.get("HGA_MODEL", "Qwen/Qwen3-0.6B"),
        "tier": args.cache,
        "decode steps": args.bench_steps,
        "warmup": args.bench_warmup,
        "date": _dt.date.today().isoformat(),
    }
    out = os.path.abspath(args.html)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(_build_html(rows, meta))
    print("-" * 40, flush=True)
    print(f"HTML results table -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Self-check (no GPU): the pure formatting/throughput helpers and HTML builder
# ---------------------------------------------------------------------------

def _selftest() -> None:
    assert _fmt_ctx(8192) == "8K"
    assert _fmt_ctx(130816) == "128K"       # capped 128K point rounds to the label
    assert abs(_tok_per_s(100.0) - 10.0) < 1e-9
    assert _tok_per_s(0.0) == 0.0
    doc = _build_html(
        [{"ctx": 8192, "cad": 1, "wall": 90.4, "toks": 11.1}],
        {"tier": "ram"},
    )
    assert "<table>" in doc and "8K" in doc and "11.1" in doc and "tier: ram" in doc
    print("[selftest] helpers + HTML OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="routing-cadence decode-latency benchmark")
    ap.add_argument("--cache", choices=("ram", "fs", "vram"), default="ram",
                    help="Cold-KV tier for the router (default: ram)")
    ap.add_argument("--bench-ctxs", default="8192,32768,65536,130816",
                    help="Comma-separated prefill context lengths in tokens "
                         "(default: 8192,32768,65536,130816 = 8K/32K/64K/128K)")
    ap.add_argument("--bench-steps", type=int, default=64,
                    help="Decode steps measured per point (default: 64)")
    ap.add_argument("--bench-warmup", type=int, default=8,
                    help="Warmup decode steps before timing (default: 8)")
    ap.add_argument("--bench-cadences", default="1,64",
                    help="Comma-separated HGA_ROUTE_CADENCE values; first is the per-token "
                         "reference (default: 1,64)")
    ap.add_argument("--html", default=os.path.join(REPO_ROOT, "docs", "cadence_bench.html"),
                    help="Output HTML results table (default: docs/cadence_bench.html)")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the offline helper/HTML self-check and exit (no GPU)")
    # Hidden: the inner GPU worker launched by the outer driver.
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cadence", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--bench-ctx", type=int, default=8192, help=argparse.SUPPRESS)
    ap.add_argument("--wall-json", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
    elif args.inner:
        _run_inner(args)
    else:
        _run_outer(args)


if __name__ == "__main__":
    main()
