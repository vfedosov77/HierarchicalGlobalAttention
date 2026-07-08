#!/usr/bin/env python3
"""Chat + overlap profiler for **Qwen3-0.6B** on a router-backed KV cache.

A thin replica of :mod:`chat_qwen30b_fp8`: it reuses that module's generation loop, terminal
and browser UI, cache-planning and disposal wholesale (imported as ``base``) and only swaps the
model to the small dense Qwen3-0.6B plus adds an **honest overlap profiler** that runs on the
*real* model — with real weights, projections and the production cache tier — instead of a
synthetic micro-benchmark.  The 0.6B fits comfortably, so it is the right vehicle to measure the
CPU-routing / H2D overlap where it actually lives (in a full transformer step), not in isolation.

Usage:
    source .venv/bin/activate
    cd ~/HierarchicalGlobalAttention

    # Terminal chat:
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen0_6b_fp8

    # Browser UI:
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen0_6b_fp8 --ui

    # Overlap A/B profile (real model, RAM tier), 32K context, 200 decode steps:
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen0_6b_fp8 \
        --profile-overlap --cache ram --ctx 32768 --steps 200

Note: the filename keeps the ``_fp8`` suffix of the 30B sibling for symmetry, but the default
model is the plain (bf16) ``Qwen/Qwen3-0.6B``.  Point ``--model`` at an FP8 build if you have one.
"""

from __future__ import annotations

import os

# Mirror the 30B chat: set before CUDA initialises (expandable segments reduce fragmentation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as base
from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    _iter_attention_layers,
    replace_qwen_attention_with_router,
)


MODEL = "Qwen/Qwen3-0.6B"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


# =================================================================================================
# Overlap profiler (runs on the real model)
# =================================================================================================
class _GpuSampler:
    """Poll NVML GPU utilization (%) from a side thread while a timed loop runs.

    ``torch.cuda.utilization`` is the nvidia-smi metric (percent of the sample period with >=1
    kernel running); ``mem_get_info`` gives device-wide used bytes from the same source.  The
    overlap ON-vs-OFF *delta* is the signal, not the absolute value.

    ponytail: NVML samples coarsely (~tens of ms) and device-wide (naive polling); a torch.profiler
    device-time sum would be finer.  Copy of the sampler in KvRouter/Tests/bench_overlap.py — kept
    local so this chat has no dependency on a test module.
    """

    def __init__(self, device, period_s: float = 0.004):
        self._dev = device
        self._period = period_s
        self._stop = threading.Event()
        self._util: list[int] = []
        self._mem: list[int] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._util.append(torch.cuda.utilization(self._dev))
                free, total = torch.cuda.mem_get_info(self._dev)
                self._mem.append(total - free)
            except Exception:
                pass
            time.sleep(self._period)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def mean(self) -> float:
        return sum(self._util) / len(self._util) if self._util else float("nan")

    @property
    def peak_mem_gb(self) -> float:
        return max(self._mem) / 2**30 if self._mem else float("nan")


def _synth_ids(tok, ctx: int, device) -> torch.Tensor:
    """A realistic [1, ctx] prompt: tokenized training text, tiled to length if it is too short.

    ponytail: latency/util/memory don't depend on prompt *content*, only length — tiling a real
    corpus keeps token ids in-distribution without needing a ctx-long unique document.
    """
    path = os.path.join(_ROOT, "TrainData", "The-Master-and-Margarita.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            ids = tok(fh.read(), return_tensors="pt").input_ids[0]
    except OSError:
        ids = torch.arange(1000, 1000 + ctx)  # fallback: dummy in-vocab ids
    if ids.numel() < ctx:
        ids = ids.repeat((ctx // max(1, ids.numel())) + 1)
    return ids[:ctx].unsqueeze(0).to(device)


def _one_pass(model, ids: torch.Tensor, ctx: int, steps: int, block: int, warmup: int,
              decode_model=None):
    """Blocked prefill of `ctx` tokens, warmup, then time `steps` single-token decode steps.

    A fresh ``DynamicCache`` is created here so the store (and its overlap copy-stream) is built
    now — ``HGA_OVERLAP`` must already be set in the environment by the caller.  Prefill always
    runs eagerly through ``model``; the timed **decode** step runs through ``decode_model`` (a
    ``torch.compile(mode="reduce-overhead")`` view for the graphed A/B), defaulting to ``model``.
    """
    device = ids.device
    dm = decode_model if decode_model is not None else model
    cache = DynamicCache()
    with torch.inference_mode():
        out = None
        for s in range(0, ctx, block):
            e = min(s + block, ctx)
            cp = torch.arange(s, e, device=device)
            out = model(input_ids=ids[:, s:e], past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        nxt = int(out.logits[:, -1].argmax(-1))
        p = ctx

        def _step(pos: int, token: int) -> int:
            cp = torch.tensor([pos], device=device)
            o = dm(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                   cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            return int(o.logits[:, -1].argmax(-1))

        for _ in range(warmup):
            nxt = _step(p, nxt)
            p += 1
        torch.cuda.synchronize()

        from torch.profiler import ProfilerActivity, profile

        torch.cuda.reset_peak_memory_stats(device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof, _GpuSampler(device) as sampler:
            t0 = time.perf_counter()
            for _ in range(steps):
                nxt = _step(p, nxt)
                p += 1
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

    # Refined metric: ABSOLUTE GPU-busy vs idle per token (sum of CUDA kernel durations vs wall).
    # NVML util% is a coarse time-fraction and misleads when the step shrinks; the honest question
    # -- where do we idle? -- needs absolute ms: idle = wall - busy, true_util = busy / wall.
    gpu_us = sum(e.self_device_time_total for e in prof.key_averages())  # µs, all kernels+steps
    ms = 1e3 * dt / steps
    busy_ms = (gpu_us / 1e3) / steps
    idle_ms = ms - busy_ms
    true_util = 100.0 * busy_ms / ms if ms else float("nan")
    util = sampler.mean
    peak_gb = torch.cuda.max_memory_reserved(device) / 2**30
    base._dispose_cache(cache)
    return ms, busy_ms, idle_ms, true_util, util, peak_gb


def _profile_overlap(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                     cache_tier: str) -> None:
    """A/B the CPU-routing + H2D overlap on the real model: overlap OFF then ON, same prompt."""
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)

    print(f"[profile] model={model_id} ctx={ctx} steps={steps} block={block} tier={cache_tier}",
          flush=True)
    prev = os.environ.get("HGA_OVERLAP")
    try:
        os.environ["HGA_OVERLAP"] = "0"
        off_ms, off_busy, off_idle, off_tu, off_util, off_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
        os.environ["HGA_OVERLAP"] = "1"
        on_ms, on_busy, on_idle, on_tu, on_util, on_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
    finally:
        if prev is None:
            os.environ.pop("HGA_OVERLAP", None)
        else:
            os.environ["HGA_OVERLAP"] = prev

    speedup = 100.0 * (off_ms - on_ms) / off_ms if off_ms else 0.0
    print(f"[overlap OFF] wall {off_ms:.3f}  GPU-busy {off_busy:.3f}  idle {off_idle:.3f} ms/token "
          f"| true-util {off_tu:.1f}%  (NVML {off_util:.1f}%)  peak {off_gb:.2f} GB", flush=True)
    print(f"[overlap ON ] wall {on_ms:.3f}  GPU-busy {on_busy:.3f}  idle {on_idle:.3f} ms/token "
          f"| true-util {on_tu:.1f}%  (NVML {on_util:.1f}%)  peak {on_gb:.2f} GB", flush=True)
    print(f"[delta] wall {off_ms - on_ms:+.3f}  GPU-busy {on_busy - off_busy:+.3f}  "
          f"idle {on_idle - off_idle:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"true-util {on_tu - off_tu:+.1f} pts   peak {on_gb - off_gb:+.2f} GB", flush=True)
    print("[note] idle-ms/token is the honest 'where do we idle?' metric; NVML util% is coarse and "
          "falls on shorter steps even when absolute idle is unchanged.", flush=True)


def _profile_graph(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                   cache_tier: str) -> None:
    """A/B the routed decode step eager vs CUDA-graphed (Level B, docs/CUDA_GRAPHS.md §8).

    The router is put in **static-decode** mode (``HGA_STATIC_DECODE=1`` → each decode block pads
    its token KV to a fixed ``Kfix`` width), which gives the attention compute a constant shape so
    ``torch.compile(mode="reduce-overhead")`` — torch's CUDA-graph fast path — can capture/replay it.
    Prefill stays eager; only the single-token decode step is graphed.  Same busy/idle metric as the
    overlap probe: the honest signal is absolute GPU-idle ms/token (and true-util), not NVML util%.
    """
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)  # the graphed path compiles + captures during warmup

    print(f"[profile-graph] model={model_id} ctx={ctx} steps={steps} block={block} tier={cache_tier}",
          flush=True)
    prev_ov = os.environ.get("HGA_OVERLAP")
    prev_sd = os.environ.get("HGA_STATIC_DECODE")
    try:
        os.environ["HGA_OVERLAP"] = "0"           # isolate the graph effect from overlap
        os.environ["HGA_STATIC_DECODE"] = "1"     # static shapes so reduce-overhead can graph
        eg_ms, eg_busy, eg_idle, eg_tu, eg_util, eg_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
        graph_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        try:
            gr = _one_pass(model, ids, ctx, steps, block, warmup, decode_model=graph_model)
        except Exception as exc:  # reduce-overhead vs the stateful router — the expected §2 wall
            print(f"[eager  ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
                  f"| true-util {eg_tu:.1f}%  peak {eg_gb:.2f} GB", flush=True)
            print("[graphed] FAILED — whole-model reduce-overhead is incompatible with the stateful "
                  "router.", flush=True)
            print(f"[why] {type(exc).__name__}: {str(exc).splitlines()[0][:160]}", flush=True)
            print("[finding] The router retains tensors computed *inside* captured subgraphs (active-"
                  "chunk KV accumulators, live windows) across decode steps; CUDA graphs recycle that "
                  "memory pool every replay -> overwrite. Confirms docs/CUDA_GRAPHS.md §2: you cannot "
                  "wrap model() in a graph. Level B needs split-capture of a STATELESS compute "
                  "submodule (router stays eager); full-step graph is Level C (GPU-only, sync-free "
                  "router). See §8.", flush=True)
            return
        gr_ms, gr_busy, gr_idle, gr_tu, gr_util, gr_gb = gr
    finally:
        for key, prev in (("HGA_OVERLAP", prev_ov), ("HGA_STATIC_DECODE", prev_sd)):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

    speedup = 100.0 * (eg_ms - gr_ms) / eg_ms if eg_ms else 0.0
    print(f"[eager  ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
          f"| true-util {eg_tu:.1f}%  (NVML {eg_util:.1f}%)  peak {eg_gb:.2f} GB", flush=True)
    print(f"[graphed] wall {gr_ms:.3f}  GPU-busy {gr_busy:.3f}  idle {gr_idle:.3f} ms/token "
          f"| true-util {gr_tu:.1f}%  (NVML {gr_util:.1f}%)  peak {gr_gb:.2f} GB", flush=True)
    print(f"[delta] wall {eg_ms - gr_ms:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"idle {gr_idle - eg_idle:+.3f}  true-util {gr_tu - eg_tu:+.1f} pts", flush=True)
    print("[note] idle-ms/token / true-util are the honest metrics; a big idle drop = graph removed "
          "the per-step launch floor. NVML util% is coarse and misleads on shorter steps.", flush=True)


def _profile_mlp_graph(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                       cache_tier: str) -> None:
    """Cheap probe (Level B step-(c), docs/CUDA_GRAPHS.md §8): graph only each layer's **MLP**.

    The decoder-layer MLP (SwiGLU: gate/up/down projections) is fully **stateless** — it retains no
    KV, no cross-step tensors — so it is graph-safe where the whole model is not.  We wrap every
    ``layer.mlp`` in ``torch.compile(mode="reduce-overhead")`` in place and A/B the same busy/idle
    metric.  A partial idle drop here is a lower-bound signal for how much of the per-step launch
    floor a full split-capture (option (a)) could reclaim; it does not touch the router.
    """
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)  # the compiled MLPs capture their graphs during warmup

    print(f"[profile-mlp-graph] model={model_id} ctx={ctx} steps={steps} block={block} "
          f"tier={cache_tier}", flush=True)
    prev_ov = os.environ.get("HGA_OVERLAP")
    layers = list(_iter_attention_layers(model))
    assert all(hasattr(ly, "mlp") for ly in layers), "decoder layers expose no .mlp to graph"
    try:
        os.environ["HGA_OVERLAP"] = "0"          # isolate the MLP-graph effect from overlap
        eg_ms, eg_busy, eg_idle, eg_tu, eg_util, eg_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)

        originals = [ly.mlp for ly in layers]
        for ly in layers:
            ly.mlp = torch.compile(ly.mlp, mode="reduce-overhead", fullgraph=False)
        try:
            gr_ms, gr_busy, gr_idle, gr_tu, gr_util, gr_gb = _one_pass(
                model, ids, ctx, steps, block, warmup)
        finally:
            for ly, orig in zip(layers, originals):
                ly.mlp = orig
    finally:
        if prev_ov is None:
            os.environ.pop("HGA_OVERLAP", None)
        else:
            os.environ["HGA_OVERLAP"] = prev_ov

    speedup = 100.0 * (eg_ms - gr_ms) / eg_ms if eg_ms else 0.0
    print(f"[eager    ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
          f"| true-util {eg_tu:.1f}%  (NVML {eg_util:.1f}%)  peak {eg_gb:.2f} GB", flush=True)
    print(f"[mlp-graph] wall {gr_ms:.3f}  GPU-busy {gr_busy:.3f}  idle {gr_idle:.3f} ms/token "
          f"| true-util {gr_tu:.1f}%  (NVML {gr_util:.1f}%)  peak {gr_gb:.2f} GB", flush=True)
    print(f"[delta] wall {eg_ms - gr_ms:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"idle {gr_idle - eg_idle:+.3f}  true-util {gr_tu - eg_tu:+.1f} pts", flush=True)
    print("[note] MLP is only part of the per-layer launch cost (attention + router dominate); a "
          "small idle drop here is the floor, split-capture of the full compute (a) reclaims more.",
          flush=True)


def _greedy_tokens(model, ids: torch.Tensor, ctx: int, block: int, n: int) -> list[int]:
    """Prefill ``ctx`` tokens then greedily decode ``n`` more, returning the generated token ids.

    Used to check numerical equivalence of the eager vs split-capture-graphed decode compute:
    identical compute ⇒ identical greedy token sequence.  Honors whatever ``HGA_*`` env is set.
    """
    device = ids.device
    cache = DynamicCache()
    out_tokens: list[int] = []
    with torch.inference_mode():
        out = None
        for s in range(0, ctx, block):
            e = min(s + block, ctx)
            cp = torch.arange(s, e, device=device)
            out = model(input_ids=ids[:, s:e], past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        nxt = int(out.logits[:, -1].argmax(-1))
        p = ctx
        for _ in range(n):
            out_tokens.append(nxt)
            cp = torch.tensor([p], device=device)
            o = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                      cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            nxt = int(o.logits[:, -1].argmax(-1))
            p += 1
    base._dispose_cache(cache)
    return out_tokens


def _selfcheck_split_graph(model, tok, *, ctx: int, n: int = 24) -> bool:
    """One runnable check for option (a): graphed compute must match eager token-for-token.

    Runs the same greedy decode twice — once eager (``HGA_GRAPH_COMPUTE=0``), once with the split-
    capture graph (``=1``) — under static-decode, and asserts the token sequences are identical.
    """
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    prev = {k: os.environ.get(k) for k in ("HGA_OVERLAP", "HGA_STATIC_DECODE", "HGA_GRAPH_COMPUTE")}
    try:
        os.environ["HGA_OVERLAP"] = "0"
        os.environ["HGA_STATIC_DECODE"] = "1"
        os.environ["HGA_GRAPH_COMPUTE"] = "0"
        eager = _greedy_tokens(model, ids, ctx, base.PREFILL_BLOCK, n)
        os.environ["HGA_GRAPH_COMPUTE"] = "1"
        graphed = _greedy_tokens(model, ids, ctx, base.PREFILL_BLOCK, n)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    match = eager == graphed
    if match:
        print(f"[selfcheck] graphed==eager token-for-token over {n} greedy steps (OK)", flush=True)
    else:
        diff = next((i for i, (a, b) in enumerate(zip(eager, graphed)) if a != b), n)
        print(f"[selfcheck] MISMATCH at step {diff}: eager={eager[:diff+1]} graphed={graphed[:diff+1]}",
              flush=True)
    return match


def _profile_split_graph(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                         cache_tier: str) -> None:
    """Option (a) (docs/CUDA_GRAPHS.md §8): CUDA-graph the STATELESS decode-compute submodule.

    The router decision stays eager; only ``attend`` (over the static-decode padded ``RoutedKV``) +
    ``o_proj`` are wrapped in ``torch.compile(mode="reduce-overhead")`` per layer (see
    ``_RoutedDecodeCompute`` / ``HGA_GRAPH_COMPUTE`` in ``qwen_routed_attention.py``).  This is the
    graph-safe split that whole-model graphing (``--profile-graph``) could not achieve.  First a
    token-for-token self-check, then the same eager-vs-graphed busy/idle A/B.
    """
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)  # the compiled compute captures its graph during warmup

    print(f"[profile-split-graph] model={model_id} ctx={ctx} steps={steps} block={block} "
          f"tier={cache_tier}", flush=True)
    if not _selfcheck_split_graph(model, tok, ctx=min(ctx, 2048)):
        print("[abort] graphed compute diverges from eager — not benchmarking a wrong kernel.",
              flush=True)
        return

    prev = {k: os.environ.get(k) for k in ("HGA_OVERLAP", "HGA_STATIC_DECODE", "HGA_GRAPH_COMPUTE")}
    try:
        os.environ["HGA_OVERLAP"] = "0"
        os.environ["HGA_STATIC_DECODE"] = "1"
        os.environ["HGA_GRAPH_COMPUTE"] = "0"
        eg_ms, eg_busy, eg_idle, eg_tu, eg_util, eg_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
        os.environ["HGA_GRAPH_COMPUTE"] = "1"
        gr_ms, gr_busy, gr_idle, gr_tu, gr_util, gr_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    speedup = 100.0 * (eg_ms - gr_ms) / eg_ms if eg_ms else 0.0
    print(f"[eager      ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
          f"| true-util {eg_tu:.1f}%  (NVML {eg_util:.1f}%)  peak {eg_gb:.2f} GB", flush=True)
    print(f"[split-graph] wall {gr_ms:.3f}  GPU-busy {gr_busy:.3f}  idle {gr_idle:.3f} ms/token "
          f"| true-util {gr_tu:.1f}%  (NVML {gr_util:.1f}%)  peak {gr_gb:.2f} GB", flush=True)
    print(f"[delta] wall {eg_ms - gr_ms:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"idle {gr_idle - eg_idle:+.3f}  true-util {gr_tu - eg_tu:+.1f} pts", flush=True)
    print("[note] only the compute region is graphed; the eager router-decision (top-k/gather/"
          ".tolist()) still bounds each step — its host tail caps this to the Level-C ceiling.",
          flush=True)


def _profile_fullstep_graph(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                            cache_tier: str) -> None:
    """Level C3 CEILING probe (docs/CUDA_GRAPHS.md §9): manually CUDA-graph the WHOLE decode step.

    ``--profile-graph`` showed that whole-model ``torch.compile(reduce-overhead)`` *fragments* the
    step at the router's remaining host work (each fragment a separate graph) and so reclaims none
    of the ~97 ms idle.  A single **manual** ``torch.cuda.CUDAGraph`` capture of the entire forward
    is the only thing that removes the 28-layer Python launch chain (as Level A did for the dense
    model).  On the VRAM tier the routed decode step has just **one** device→host sync (the layer-0
    ``start_pos`` read), which we bypass during capture via ``router._capture_start_pos``.

    This is a *measuring probe, not a correct decoder*: a single captured graph freezes the router's
    per-step geometry (active-chunk fill count, chunk-close events) — that progress lives in Python
    state a graph replay cannot advance — so the tokens are only valid for the captured step.  What
    it measures is the **idle floor a fully replay-safe one-graph step would reach**, i.e. whether
    the large router rewrite (GPU-parameterised geometry) to make the whole step graph-safe would
    pay off.  Requires ``--cache vram`` (RAM/FS tiers do host-indexed H2D copies that cannot be
    captured — docs §9 tier fact).
    """
    if cache_tier != "vram":
        print("[fullstep-graph] requires --cache vram (RAM/FS cold-copy needs host indices that "
              "cannot be captured; docs/CUDA_GRAPHS.md §9 tier fact).", flush=True)
        return

    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)

    print(f"[profile-fullstep-graph] model={model_id} ctx={ctx} steps={steps} block={block} "
          f"tier={cache_tier}", flush=True)

    from torch.profiler import ProfilerActivity, profile

    def _time(run_one, n: int):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(n):
                run_one()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        gpu_us = sum(e.self_device_time_total for e in prof.key_averages())
        ms = 1e3 * dt / n
        busy = (gpu_us / 1e3) / n
        return ms, busy, ms - busy, (100.0 * busy / ms if ms else float("nan"))

    prev = {k: os.environ.get(k) for k in ("HGA_OVERLAP", "HGA_STATIC_DECODE", "HGA_GRAPH_COMPUTE")}
    os.environ["HGA_OVERLAP"] = "0"
    os.environ["HGA_STATIC_DECODE"] = "1"
    os.environ["HGA_GRAPH_COMPUTE"] = "0"
    cache = DynamicCache()
    try:
        with torch.inference_mode():
            out = None
            for s in range(0, ctx, block):
                e = min(s + block, ctx)
                cp = torch.arange(s, e, device=device)
                out = model(input_ids=ids[:, s:e], past_key_values=cache,
                            cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            nxt = int(out.logits[:, -1].argmax(-1))
            p = ctx

            def _eager_step(pos: int, token: int) -> int:
                cp = torch.tensor([pos], device=device)
                o = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                          cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
                return int(o.logits[:, -1].argmax(-1))

            for _ in range(warmup):
                nxt = _eager_step(p, nxt); p += 1
            torch.cuda.synchronize()

            # -- eager baseline (real decode, state advances) --
            st = {"p": p, "nxt": nxt}
            def _eager_one():
                st["nxt"] = _eager_step(st["p"], st["nxt"]); st["p"] += 1
            eg_ms, eg_busy, eg_idle, eg_tu = _time(_eager_one, steps)

            # -- manual whole-step capture on the frozen mid-chunk geometry --
            router = cache._kv_router
            cap_pos = st["p"]
            router._capture_start_pos = cap_pos          # sync-free start_pos during capture/replay
            static_tok = torch.tensor([[st["nxt"]]], device=device)
            static_cp = torch.tensor([cap_pos], device=device)
            static_pid = static_cp.unsqueeze(0)

            def _fwd():
                return model(input_ids=static_tok, past_key_values=cache,
                             cache_position=static_cp, position_ids=static_pid, use_cache=True)

            # Warm on a side stream (initialises cuBLAS workspaces etc. so capture is clean).
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = _fwd()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                graph_out = _fwd()

            def _replay_one():
                g.replay()
            gr_ms, gr_busy, gr_idle, gr_tu = _time(_replay_one, steps)
            g.reset()
            del g, graph_out
    except Exception as exc:
        print(f"[fullstep-graph] CAPTURE FAILED — {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:180]}", flush=True)
        print("[finding] The whole VRAM decode step could not be captured as one graph. See the "
              "traceback for the offending op (a residual sync or a non-capturable allocation).",
              flush=True)
        return
    finally:
        router = getattr(cache, "_kv_router", None)
        if router is not None:
            router._capture_start_pos = None
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        base._dispose_cache(cache)

    speedup = 100.0 * (eg_ms - gr_ms) / eg_ms if eg_ms else 0.0
    print(f"[eager       ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
          f"| true-util {eg_tu:.1f}%", flush=True)
    print(f"[graph-replay] wall {gr_ms:.3f}  GPU-busy {gr_busy:.3f}  idle {gr_idle:.3f} ms/token "
          f"| true-util {gr_tu:.1f}%", flush=True)
    print(f"[delta] wall {eg_ms - gr_ms:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"idle {gr_idle - eg_idle:+.3f}  true-util {gr_tu - eg_tu:+.1f} pts", flush=True)
    print("[note] CEILING probe: replay reuses the captured step's frozen geometry (Python router "
          "state does not advance) so tokens are NOT valid past the captured step — it measures the "
          "idle floor a replay-safe one-graph step (GPU-parameterised router) would reach.",
          flush=True)


@torch.no_grad()
def _hybrid_graph_decode(model, ids: torch.Tensor, ctx: int, block: int, n_steps: int, *,
                         record: bool = False, profile_it: bool = False):
    """Level C3 **hybrid** graph decode (docs/CUDA_GRAPHS.md §9): a *correct* one-graph-per-chunk

    decoder on the VRAM tier.  Within a chunk the intra-chunk geometry is GPU-parameterised
    (``router._graph_pos_gpu`` → offset / active write-slot / causal mask on-device), so ONE captured
    ``torch.cuda.CUDAGraph`` replays across all ``C`` steady-state steps; the token feedback stays on
    GPU (``argmax`` → static input buffer, no D2H).  The chunk-close seam (summaries + hand-off to the
    store) runs **eagerly** once per ``C`` tokens, after which the graph is re-captured for the next
    chunk (the closed-chunk count — and thus the routing shapes — changed).  Capture warmup is
    idempotent (the active write is an ``index_copy`` to a fixed slot), so no snapshot/restore is
    needed.  Requires ``ctx % chunk_size == 0`` (decode starts on a fresh chunk).
    """
    device = ids.device
    C = base.CHUNK_SIZE
    assert ctx % C == 0, "hybrid graph decode needs ctx to be a multiple of chunk_size"
    num_layers = int(model.config.num_hidden_layers)
    KVH = int(model.config.num_key_value_heads)
    Dh = int(getattr(model.config, "head_dim",
                     model.config.hidden_size // model.config.num_attention_heads))
    dt = next(model.parameters()).dtype

    cache = DynamicCache()
    # -- eager blocked prefill --
    out = None
    for s in range(0, ctx, block):
        e = min(s + block, ctx)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
    router = cache._kv_router

    # Pre-allocate the empty active-chunk buffers so no allocation happens inside a graph capture.
    for L in range(num_layers):
        if router._active_krope.get(L) is None:
            router._active_krope[L] = torch.zeros(1, KVH, C, Dh, device=device, dtype=dt)
            router._active_kraw[L] = torch.zeros(1, KVH, C, Dh, device=device, dtype=dt)
            router._active_v[L] = torch.zeros(1, KVH, C, Dh, device=device, dtype=dt)
            router._active_len[L] = 0

    static_token = out.logits[:, -1].argmax(-1).view(1, 1).clone()   # [1,1] GPU input buffer
    static_cp = torch.tensor([ctx], device=device)                   # [1] position (advances)
    static_pid = static_cp.view(1, 1)

    def _fwd():
        return model(input_ids=static_token, past_key_values=cache, cache_position=static_cp,
                     position_ids=static_pid, use_cache=True)

    holder = {"g": None, "out": None}

    def _capture(chunk_start: int) -> None:
        if holder["g"] is not None:
            holder["g"].reset()
        router._capture_start_pos = chunk_start          # host-frozen n; GPU offset drives geometry
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                            # idempotent warmup (index_copy to fixed slot)
                _fwd()
        torch.cuda.current_stream().wait_stream(s)
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            o = _fwd()
        holder["g"], holder["out"] = gg, o

    rec: list[torch.Tensor] = []
    metrics = None
    try:
        p = ctx
        static_cp.fill_(p)
        _capture((p // C) * C)

        from torch.profiler import ProfilerActivity, profile
        prof_ctx = profile(activities=[ProfilerActivity.CUDA]) if profile_it else None
        t0 = 0.0
        if profile_it:
            torch.cuda.synchronize()
            prof_ctx.__enter__()
            t0 = time.perf_counter()

        for _ in range(n_steps):
            static_cp.fill_(p)
            if record:
                rec.append(static_token.reshape(-1).clone())   # the token FED this step (matches eager)
            holder["g"].replay()
            nxt = holder["out"].logits[:, -1].argmax(-1)   # GPU [1], no D2H
            static_token.copy_(nxt.view(1, 1))
            p += 1
            if p % C == 0:                                  # chunk filled -> eager close + re-capture
                n_done = (p - 1) // C
                for L in range(num_layers):
                    router._close_active_chunk(L, n_done)
                static_cp.fill_(p)
                _capture(p)

        if profile_it:
            torch.cuda.synchronize()
            dt_s = time.perf_counter() - t0
            prof_ctx.__exit__(None, None, None)
            gpu_us = sum(e.self_device_time_total for e in prof_ctx.key_averages())
            ms = 1e3 * dt_s / n_steps
            busy = (gpu_us / 1e3) / n_steps
            metrics = (ms, busy, ms - busy, (100.0 * busy / ms if ms else float("nan")))
    finally:
        router._capture_start_pos = None
        router._graph_pos_gpu = None
        if holder["g"] is not None:
            holder["g"].reset()
        base._dispose_cache(cache)

    tokens = [int(t.item()) for t in rec] if record else []
    return tokens, metrics


def _selfcheck_hybrid_graph(model, tok, *, ctx: int, n: int = 160) -> bool:
    """The one runnable check for the hybrid decoder: graphed tokens == eager tokens, spanning at
    least two chunk closes (n > 2*chunk_size) so the eager close-seam + re-capture are exercised."""
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    prev = {k: os.environ.get(k) for k in ("HGA_OVERLAP", "HGA_STATIC_DECODE", "HGA_GRAPH_COMPUTE")}
    try:
        os.environ["HGA_OVERLAP"] = "0"
        os.environ["HGA_STATIC_DECODE"] = "1"
        os.environ["HGA_GRAPH_COMPUTE"] = "0"
        eager = _greedy_tokens(model, ids, ctx, base.PREFILL_BLOCK, n)
        graphed, _ = _hybrid_graph_decode(model, ids, ctx, base.PREFILL_BLOCK, n, record=True)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    match = eager == graphed
    if match:
        print(f"[selfcheck] hybrid graphed==eager token-for-token over {n} greedy steps (OK)",
              flush=True)
    else:
        diff = next((i for i, (a, b) in enumerate(zip(eager, graphed)) if a != b),
                    min(len(eager), len(graphed)))
        print(f"[selfcheck] MISMATCH at step {diff}: eager={eager[max(0, diff-2):diff+1]} "
              f"graphed={graphed[max(0, diff-2):diff+1]}", flush=True)
    return match


def _profile_hybrid_graph(model, tok, *, model_id: str, ctx: int, steps: int, block: int,
                          cache_tier: str) -> None:
    """Option (2) (docs/CUDA_GRAPHS.md §9): benchmark the *correct* hybrid one-graph-per-chunk decode
    vs eager on the VRAM tier.  First a token-for-token self-check (aborts on divergence), then an
    eager-vs-hybrid busy/idle A/B — the honest signal is absolute GPU-idle ms/token (and true-util).
    """
    if cache_tier != "vram":
        print("[hybrid-graph] requires --cache vram (RAM/FS cold-copy needs host indices that "
              "cannot be captured; docs/CUDA_GRAPHS.md §9 tier fact).", flush=True)
        return
    device = torch.device("cuda")
    ids = _synth_ids(tok, ctx, device)
    warmup = min(16, steps)

    print(f"[profile-hybrid-graph] model={model_id} ctx={ctx} steps={steps} block={block} "
          f"tier={cache_tier}", flush=True)
    if not _selfcheck_hybrid_graph(model, tok, ctx=min(ctx, 2048), n=3 * base.CHUNK_SIZE):
        print("[abort] hybrid graphed decode diverges from eager — not benchmarking a wrong kernel.",
              flush=True)
        return

    prev = {k: os.environ.get(k) for k in ("HGA_OVERLAP", "HGA_STATIC_DECODE", "HGA_GRAPH_COMPUTE")}
    try:
        os.environ["HGA_OVERLAP"] = "0"
        os.environ["HGA_STATIC_DECODE"] = "1"
        os.environ["HGA_GRAPH_COMPUTE"] = "0"
        eg_ms, eg_busy, eg_idle, eg_tu, eg_util, eg_gb = _one_pass(
            model, ids, ctx, steps, block, warmup)
        _, hy = _hybrid_graph_decode(model, ids, ctx, block, steps, profile_it=True)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    hy_ms, hy_busy, hy_idle, hy_tu = hy
    speedup = 100.0 * (eg_ms - hy_ms) / eg_ms if eg_ms else 0.0
    print(f"[eager       ] wall {eg_ms:.3f}  GPU-busy {eg_busy:.3f}  idle {eg_idle:.3f} ms/token "
          f"| true-util {eg_tu:.1f}%", flush=True)
    print(f"[hybrid-graph] wall {hy_ms:.3f}  GPU-busy {hy_busy:.3f}  idle {hy_idle:.3f} ms/token "
          f"| true-util {hy_tu:.1f}%", flush=True)
    print(f"[delta] wall {eg_ms - hy_ms:+.3f} ms/token ({speedup:+.1f}% latency)  "
          f"idle {hy_idle - eg_idle:+.3f}  true-util {hy_tu - eg_tu:+.1f} pts", flush=True)
    print("[note] correct decoder: one CUDA-graph per chunk replays the C intra-chunk steps "
          "(argmax on-GPU); the chunk-close seam (1/C tokens) runs eager, then re-captures.",
          flush=True)


# =================================================================================================
# Entry point
# =================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Chat / overlap-profile Qwen3-0.6B (router KV cache)")
    ap.add_argument("--model", default=MODEL, help=f"HF model id (default: {MODEL})")
    ap.add_argument("--ui", action="store_true", help="Open browser UI instead of terminal chat")
    ap.add_argument("--host", default="0.0.0.0", help="UI server host")
    ap.add_argument("--port", type=int, default=7860, help="UI server port")
    ap.add_argument("--cache", choices=("ram", "fs", "vram"), default=base.CACHE_LOCATION,
                    help="Cold-KV tier: ram (host RAM), fs (RAM-bounded + NVMe spillover), vram")
    ap.add_argument("--ram-budget-gb", type=float, default=base.RAM_BUDGET_GB,
                    help="Host-RAM ceiling for the fs tier before chunks spill to disk")
    ap.add_argument("--fs-cache-dir", default=base.FS_CACHE_DIR,
                    help="Directory for fs-tier spill files (must be a real disk, not tmpfs)")
    ap.add_argument("--profile-overlap", action="store_true",
                    help="Run an overlap OFF/ON A/B profile on the real model and exit")
    ap.add_argument("--profile-graph", action="store_true",
                    help="Run an eager-vs-CUDA-graphed decode A/B (Level B, static-decode router) and exit")
    ap.add_argument("--profile-mlp-graph", action="store_true",
                    help="Cheap probe: graph only each layer's stateless MLP and A/B busy/idle, then exit")
    ap.add_argument("--profile-split-graph", action="store_true",
                    help="Option (a): CUDA-graph the stateless decode-compute (attend+o_proj), self-check + A/B, exit")
    ap.add_argument("--profile-fullstep-graph", action="store_true",
                    help="Level C3 ceiling probe: manually CUDA-graph the WHOLE decode step (VRAM tier) and A/B replays vs eager, exit")
    ap.add_argument("--profile-hybrid-graph", action="store_true",
                    help="Level C3 hybrid: correct one-graph-per-chunk decode (VRAM tier), self-check + eager A/B, exit")
    ap.add_argument("--ctx", type=int, nargs="+", default=[32768],
                    help="[profile] prefill context length(s); multiple values sweep in one load")
    ap.add_argument("--steps", type=int, default=200, help="[profile] timed decode steps")
    ap.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto",
                    help="Compute dtype; 'auto' picks fp16 on pre-Ampere (no bf16 tensor cores)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"

    # Turing/Volta (SM < 8.0) have NO bf16 tensor cores: bf16 matmuls run emulated and inflate
    # GPU-busy, skewing the overlap idle/busy breakdown. Qwen3's 'auto' is bf16 -> force fp16 here.
    if args.dtype == "auto":
        pre_ampere = torch.cuda.get_device_capability()[0] < 8
        dtype = torch.float16 if pre_ampere else "auto"
        if pre_ampere:
            print("[dtype] pre-Ampere GPU (no bf16 tensor cores) -> loading in float16", flush=True)
    else:
        dtype = getattr(torch, args.dtype)

    model_id = args.model
    print(f"Loading {model_id} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa",
    )
    model.eval()

    n = replace_qwen_attention_with_router(
        model, cache_location=args.cache,
        keep_first=base.KEEP_FIRST, keep_last=base.KEEP_LAST, topk_chunks=base.TOPK_CHUNKS,
        topk_groups=base.TOPK_GROUPS, chunk_size=base.CHUNK_SIZE, group_size=base.GROUP_SIZE,
        vram_cache_chunks=base.VRAM_CACHE_CHUNKS, vram_summary_chunks=base.VRAM_SUMMARY_CHUNKS,
        vram_cache_reserve_gb=base.VRAM_CACHE_RESERVE_GB,
        ram_budget_gb=args.ram_budget_gb, fs_cache_dir=args.fs_cache_dir,
        dca_chunk=base.DCA_CHUNK, dca_local=base.DCA_LOCAL,
    )
    torch.cuda.synchronize()
    print(
        f"Loaded in {time.perf_counter() - t0:.1f}s  "
        f"({base.gb(torch.cuda.memory_allocated()):.1f}GB / "
        f"{base.gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB VRAM); router on {n} layers; "
        f"tier={args.cache}",
        flush=True,
    )

    if args.profile_overlap:
        for ctx in args.ctx:
            _profile_overlap(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                             block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.profile_graph:
        for ctx in args.ctx:
            _profile_graph(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                           block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.profile_mlp_graph:
        for ctx in args.ctx:
            _profile_mlp_graph(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                               block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.profile_split_graph:
        for ctx in args.ctx:
            _profile_split_graph(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                                 block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.profile_fullstep_graph:
        for ctx in args.ctx:
            _profile_fullstep_graph(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                                    block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.profile_hybrid_graph:
        for ctx in args.ctx:
            _profile_hybrid_graph(model, tok, model_id=model_id, ctx=ctx, steps=args.steps,
                                  block=base.PREFILL_BLOCK, cache_tier=args.cache)
    elif args.ui:
        base._run_ui(model, tok, args.host, args.port)
    else:
        base._terminal_chat(model, tok)


if __name__ == "__main__":
    main()
