#!/usr/bin/env python3
"""CUDA-graph decode probe (Level A of docs/CUDA_GRAPHS.md).

The real-model sweep (docs/OVERLAP.md §7) showed the decode step is *host/launch-bound*, not
H2D-bound: GPU util sits at 28-34% and CPU-routing/H2D overlap gives no benefit.  The indicated
lever is to remove per-step kernel-launch overhead with **CUDA graphs**.

This is the cheap **measuring probe** the plan asks for *before* touching the router: it A/Bs a
plain **dense** Qwen3-0.6B decode step (no HGA router) with and without CUDA graphs, to answer one
question — *how much can removing launch overhead buy on this card?*

  * eager   : StaticCache decode, one ``model(...)`` per token.
  * graphed : the same, but the decode model is ``torch.compile(model, mode="reduce-overhead")`` —
              torch's CUDA-graph fast path (graph capture + a little kernel fusion).

Prefill is always run eagerly into the StaticCache; only the single-token **decode** step is
graphed (matching the plan — we never graph the variable-length prefill).

Decision fork (see the plan): graphed ≥ ~20% faster ⇒ pursue Level B (graph the HGA decode via a
static padded RoutedKV buffer).  < ~5% ⇒ this card's decode is not launch-bound; close the CUDA
-graph direction and look elsewhere (larger kernels / different hardware).

Usage:
    source .venv/bin/activate
    cd ~/HierarchicalGlobalAttention

    # A/B probe (dense 0.6B), 8K context, 80 timed decode steps:
    python -m ExistingModelFineTuning.Qwen3LongContext.bench_cudagraph --ctx 8192 --steps 80

    # Correctness self-check only (no timing, no model download beyond the 0.6B weights):
    python -m ExistingModelFineTuning.Qwen3LongContext.bench_cudagraph --selftest

ponytail: the graphed path is measured via ``torch.compile(reduce-overhead)`` rather than a
hand-rolled ``torch.cuda.CUDAGraph`` capture.  reduce-overhead *is* CUDA graphs under the hood and
is the version-robust platform feature (torch 2.12 / transformers 5.12); a manual capture of the
HF forward is brittle across releases.  Ceiling caveat: reduce-overhead also fuses a few kernels,
so it is a slight *over*-estimate of pure launch-overhead removal — which is fine for a go/no-go
probe (if even this over-estimate is small, hand-rolled graphs won't beat it).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as base
from ExistingModelFineTuning.Qwen3LongContext import chat_qwen0_6b_fp8 as small


def _make_cache(model, max_len: int) -> StaticCache:
    """A fixed-length StaticCache (graph-safe: in-place index writes, no per-step reallocation)."""
    return StaticCache(config=model.config, max_cache_len=max_len)


def _prefill(model, cache: StaticCache, ids: torch.Tensor, ctx: int, block: int) -> int:
    """Blocked eager prefill of `ctx` tokens into `cache`; return the first token to decode."""
    device = ids.device
    out = None
    for s in range(0, ctx, block):
        e = min(s + block, ctx)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
    return int(out.logits[:, -1].argmax(-1))


def _decode_step(decode_model, cache: StaticCache, pos: int, token: int, device):
    """One greedy decode step through `decode_model`; return (next_token_id, last_logits)."""
    cp = torch.tensor([pos], device=device)
    o = decode_model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                     cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
    logits = o.logits[:, -1]
    return int(logits.argmax(-1)), logits


def _decode_pass(model, decode_model, ids: torch.Tensor, *, ctx: int, steps: int, block: int,
                 warmup: int, label: str):
    """Prefill eagerly, warm up `decode_model`, then time `steps` single-token decode steps."""
    device = ids.device
    cache = _make_cache(model, ctx + warmup + steps + 8)
    with torch.inference_mode():
        nxt = _prefill(model, cache, ids, ctx, block)
        p = ctx
        for _ in range(warmup):  # compiles + captures the CUDA graph for the graphed path
            nxt, _ = _decode_step(decode_model, cache, p, nxt, device)
            p += 1
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        with small._GpuSampler(device) as sampler:
            t0 = time.perf_counter()
            for _ in range(steps):
                nxt, _ = _decode_step(decode_model, cache, p, nxt, device)
                p += 1
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

    ms = 1e3 * dt / steps
    util = sampler.mean
    peak_gb = torch.cuda.max_memory_reserved(device) / 2**30
    base._dispose_cache(cache)  # no-op for StaticCache (no router); frees VRAM banks if present
    del cache
    torch.cuda.empty_cache()
    print(f"[{label}] {ms:.3f} ms/token   NVML-util {util:.1f}% (coarse; use --idle for true)   "
          f"peak {peak_gb:.2f} GB", flush=True)
    return ms, util, peak_gb


def _compiled_decode(model):
    """A reduce-overhead (CUDA-graph) compiled view of `model` for the graphed decode path."""
    return torch.compile(model, mode="reduce-overhead", fullgraph=False)


def _busy_idle_pass(model, decode_model, ids: torch.Tensor, *, ctx: int, steps: int, block: int,
                    warmup: int, label: str):
    """Measure ABSOLUTE GPU-busy vs idle per token via torch.profiler (device kernel time).

    NVML ``utilization`` is a coarse *time-fraction* metric and is misleading when the step gets
    much shorter (a fixed per-step host cost becomes a bigger fraction).  The honest question the
    owner asks — *where do we idle?* — needs absolute numbers: sum of CUDA kernel durations
    (GPU-busy) vs wall-clock; idle = wall - busy; true_util = busy / wall.
    """
    from torch.profiler import ProfilerActivity, profile

    device = ids.device
    cache = _make_cache(model, ctx + warmup + steps + 8)
    with torch.inference_mode():
        nxt = _prefill(model, cache, ids, ctx, block)
        p = ctx
        for _ in range(warmup):
            nxt, _ = _decode_step(decode_model, cache, p, nxt, device)
            p += 1
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(steps):
                nxt, _ = _decode_step(decode_model, cache, p, nxt, device)
                p += 1
            torch.cuda.synchronize()
            wall_ms = 1e3 * (time.perf_counter() - t0)

    gpu_us = sum(e.self_device_time_total for e in prof.key_averages())  # µs, all kernels, all steps
    busy_ms = gpu_us / 1e3
    wall_pt = wall_ms / steps
    busy_pt = busy_ms / steps
    idle_pt = wall_pt - busy_pt
    true_util = 100.0 * busy_ms / wall_ms if wall_ms else float("nan")
    base._dispose_cache(cache)
    del cache
    torch.cuda.empty_cache()
    print(f"[{label}] wall {wall_pt:.3f}  GPU-busy {busy_pt:.3f}  idle {idle_pt:.3f} ms/token  "
          f"| true GPU-util {true_util:.1f}%  ({1e3 / wall_pt:.1f} tok/s)", flush=True)
    return wall_pt, busy_pt, idle_pt, true_util


def _idle(model, tok, *, ctx: int, steps: int, block: int) -> None:
    """A/B the ABSOLUTE busy/idle breakdown (torch.profiler) — the real 'where do we idle?' view."""
    device = torch.device("cuda")
    ids = small._synth_ids(tok, ctx, device)
    warmup = max(8, min(16, steps))
    print(f"[idle] dense decode  ctx={ctx} steps={steps} block={block}", flush=True)
    ew, eb, ei, eu = _busy_idle_pass(
        model, model, ids, ctx=ctx, steps=steps, block=block, warmup=warmup, label="eager  ")
    torch._dynamo.reset()
    cmodel = _compiled_decode(model)
    gw, gb_, gi, gu = _busy_idle_pass(
        model, cmodel, ids, ctx=ctx, steps=steps, block=block, warmup=warmup, label="graphed")
    print(f"[idle Δ] wall {gw - ew:+.3f}  GPU-busy {gb_ - eb:+.3f}  idle {gi - ei:+.3f} ms/token  "
          f"| true-util {gu - eu:+.1f} pts", flush=True)
    print("[idle note] NVML util% can fall while the step gets faster: a fixed per-step HOST cost "
          "(argmax D2H sync + input prep) is a bigger fraction of a shorter step. The absolute "
          "idle-ms/token above is the honest metric — that is what must shrink toward 0.", flush=True)


def _profile(model, tok, *, ctx: int, steps: int, block: int) -> None:
    device = torch.device("cuda")
    ids = small._synth_ids(tok, ctx, device)
    warmup = max(8, min(16, steps))  # reduce-overhead needs a few iters to capture the graph

    print(f"[probe] dense decode  ctx={ctx} steps={steps} block={block} warmup={warmup}", flush=True)
    eager_ms, eager_util, eager_gb = _decode_pass(
        model, model, ids, ctx=ctx, steps=steps, block=block, warmup=warmup, label="eager  ")

    torch._dynamo.reset()
    cmodel = _compiled_decode(model)
    graph_ms, graph_util, graph_gb = _decode_pass(
        model, cmodel, ids, ctx=ctx, steps=steps, block=block, warmup=warmup, label="graphed")

    speedup = 100.0 * (eager_ms - graph_ms) / eager_ms if eager_ms else 0.0
    verdict = ("PURSUE Level B (>=20%)" if speedup >= 20.0
               else "CLOSE graphs (<5%)" if speedup < 5.0
               else "MARGINAL (profile deeper)")
    print(f"[delta] {eager_ms - graph_ms:+.3f} ms/token ({speedup:+.1f}% faster)   "
          f"GPU util {graph_util - eager_util:+.1f} pts   peak {graph_gb - eager_gb:+.2f} GB",
          flush=True)
    print(f"[verdict] {verdict}", flush=True)


def _greedy_ids(model, decode_model, ids: torch.Tensor, ctx: int, n: int, block: int):
    """Return (list of `n` greedy token ids, step-0 logits) decoding through `decode_model`."""
    device = ids.device
    cache = _make_cache(model, ctx + n + 8)
    got: list[int] = []
    first_logits = None
    with torch.inference_mode():
        nxt = _prefill(model, cache, ids, ctx, block)
        p = ctx
        for i in range(n):
            got.append(nxt)
            nxt, logits = _decode_step(decode_model, cache, p, nxt, device)
            if i == 0:
                first_logits = logits.detach().float().clone()
            p += 1
    base._dispose_cache(cache)
    del cache
    torch.cuda.empty_cache()
    return got, first_logits


def _selftest(model, tok, *, ctx: int, block: int) -> None:
    """Assert the graphed decode matches eager token-for-token (correctness gate before timing)."""
    device = torch.device("cuda")
    ids = small._synth_ids(tok, ctx, device)
    n = 16

    eager_ids, eager_logits = _greedy_ids(model, model, ids, ctx, n, block)
    torch._dynamo.reset()
    cmodel = _compiled_decode(model)
    graph_ids, graph_logits = _greedy_ids(model, cmodel, ids, ctx, n, block)

    max_abs = (eager_logits - graph_logits).abs().max().item()
    print(f"[selftest] eager ids  : {eager_ids}", flush=True)
    print(f"[selftest] graphed ids: {graph_ids}", flush=True)
    print(f"[selftest] step-0 logits max|Δ| = {max_abs:.3e}", flush=True)
    # Token-id equality is the decisive gate; logits closeness is secondary and loose because
    # reduce-overhead fuses kernels → fp16/bf16 rounding differs (fp32 stays tight).
    tol = 1e-3 if model.dtype == torch.float32 else 1e-1
    assert graph_ids == eager_ids, "graphed decode diverged from eager (token mismatch)"
    assert max_abs < tol, f"graphed logits drifted too far from eager ({max_abs:.3e} >= {tol})"
    print("[selftest] PASSED — graphed decode is numerically faithful to eager", flush=True)


def _load(model_id: str, dtype: str = "auto"):
    print(f"Loading {model_id} (dtype={dtype}) ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    td = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=td, device_map="cuda", attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s "
          f"({base.gb(torch.cuda.memory_allocated()):.1f}GB VRAM); dense (no router)", flush=True)
    # Turing (SM<8.0) has no bf16 tensor cores; torch.inductor SILENTLY skips bf16 codegen, so a
    # bf16 graphed run measures ~0% speedup — a false negative.  is_bf16_supported() is unreliable
    # here (it reports emulated support), so gate on compute capability instead.
    if model.dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print("[WARN] bf16 model on a pre-Ampere card: inductor will skip bf16 compilation and the "
              "CUDA-graph benefit will be MASKED (~0%). Re-run with --dtype float16 for a valid "
              "probe.", flush=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA-graph decode probe for dense Qwen3-0.6B")
    ap.add_argument("--model", default=small.MODEL, help=f"HF model id (default: {small.MODEL})")
    ap.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto",
                    help="Model dtype; float16 avoids the bf16 compile-skip on Turing (SM75) cards")
    ap.add_argument("--ctx", type=int, default=8192, help="prefill context length")
    ap.add_argument("--steps", type=int, default=80, help="timed decode steps")
    ap.add_argument("--selftest", action="store_true",
                    help="Only run the eager-vs-graphed correctness check and exit")
    ap.add_argument("--idle", action="store_true",
                    help="A/B the absolute GPU-busy vs idle ms/token (torch.profiler) and exit")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    model, tok = _load(args.model, args.dtype)

    if args.selftest:
        _selftest(model, tok, ctx=min(args.ctx, 2048), block=base.PREFILL_BLOCK)
    elif args.idle:
        _idle(model, tok, ctx=args.ctx, steps=args.steps, block=base.PREFILL_BLOCK)
    else:
        _profile(model, tok, ctx=args.ctx, steps=args.steps, block=base.PREFILL_BLOCK)


if __name__ == "__main__":
    main()
