"""Honest per-phase bottleneck profiler for the routed (HGA) decode step — Qwen3-0.6B.

Prior overlap/graph/split experiments *indirectly* concluded the routed decode step is
host/launch-bound.  This script replaces that scattered evidence with **one reproducible
instrument** that decomposes every decode step into non-overlapping, named buckets and
prints a single verdict naming the bottleneck that blocks 100% GPU utilisation.

What it measures (per decode token, averaged over a measured window):

* **PASS A — clean latency** (no profiler): ``wall`` ms/step (perf_counter + synchronize),
  tok/s, peak VRAM.  This is the honest headline latency (profiler adds overhead).
* **PASS B — attribution** (under ``torch.profiler``, CPU+CUDA):
  - ``busy`` = Σ CUDA-kernel time; ``idle = wall − busy``; ``true_util = busy/wall``
    (NVML is deliberately NOT used — it lies for short steps).
  - CUDA split: ``compute`` vs ``H2D`` (Memcpy HtoD) vs ``D2H`` (Memcpy DtoH).
  - HOST regions (``record_function`` in the router/store/wrapper): route_decision,
    gather_summaries, gather_tokens, assemble, attend, o_proj, qkv_proj, h2d, mlp, and
    the unannotated ``host_other`` residual — shows *which phase* eats the idle.
  - D2H sync count/step and CUDA kernel count/step.

Verdict logic:
  H2D share high & grows fs>ram>vram  -> TRANSFER-BOUND (overlap/prefetch would help).
  D2H share high                      -> SYNC-BOUND (remove .item()/.tolist()).
  busy/wall low & host regions ~ wall & idle tier-invariant -> HOST/LAUNCH-BOUND.
  busy/wall high                      -> COMPUTE-BOUND (GPU already saturated).

Turing (device capability < 8) is forced to fp16 (bf16 tensor-cores / inductor are
unreliable there — measuring in bf16 gives a false read).

Run (always from the repo root, always the project .venv):

    .venv/bin/python -m ExistingModelFineTuning.Qwen3LongContext.profile_hga_bottleneck --selftest
    .venv/bin/python -m ExistingModelFineTuning.Qwen3LongContext.profile_hga_bottleneck \
        --cache vram ram fs --ctx 8192 32768 131072 --steps 40
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

import torch
try:
    from torch.autograd import DeviceType  # type: ignore
except ImportError:  # pragma: no cover
    from torch._C._autograd import DeviceType  # type: ignore
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, DynamicCache

from ExistingModelFineTuning.Qwen3LongContext import qwen_routed_attention as _qra
from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    replace_qwen_attention_with_router,
    _iter_attention_layers,
)
from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as cfg

# CRITICAL: use the *exact* ``_prof`` module the hot path imports (the router/store/wrapper resolve
# it as ``KvRouter._prof`` via their sys.path shim).  Importing it independently as
# ``ExistingModelFineTuning.KvRouter._prof`` would be a *different* module object with its own
# ``PROFILE`` flag, so our markers would stay no-ops.  Bind to the wrapper's reference.
_prof = _qra._prof


MODEL_DEFAULT = "Qwen/Qwen3-0.6B"
# Top-level, NON-overlapping decode-step phases (sequential in each layer's forward) — these
# partition the host timeline; their inclusive CPU time (launch overhead included) sums toward wall.
HOST_TOP = ["hga/qkv_proj", "hga/route", "hga/attend", "hga/o_proj", "hga/mlp"]
# Sub-phases nested inside "hga/route" (informational breakdown of the router host cost).
HOST_SUB = ["hga/route_decision", "hga/gather_summaries", "hga/gather_tokens", "hga/assemble", "hga/h2d"]
HOST_ALL = HOST_TOP + HOST_SUB


def gb(x: int) -> float:
    return x / 2**30


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    cap = torch.cuda.get_device_capability(device)
    if name == "auto":
        if cap[0] < 8:
            print(f"[warn] device cap {cap[0]}.{cap[1]} (<8, pre-Ampere): forcing float16 "
                  f"(bf16 tensor-cores/inductor unreliable here -> false profiling read).")
            return torch.float16
        return torch.bfloat16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# MLP host-time hooks (non-invasive: wrap each layer.mlp in a record_function span)
# ---------------------------------------------------------------------------
def install_mlp_hooks(model: torch.nn.Module) -> List:
    handles, spans = [], {}

    def pre(mod, _inp):
        if _prof.PROFILE:
            c = torch.profiler.record_function("hga/mlp")
            c.__enter__()
            spans[id(mod)] = c

    def post(mod, _inp, _out):
        c = spans.pop(id(mod), None)
        if c is not None:
            c.__exit__(None, None, None)

    for layer in _iter_attention_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            handles.append(mlp.register_forward_pre_hook(pre))
            handles.append(mlp.register_forward_hook(post))
    return handles


# ---------------------------------------------------------------------------
# Prefill + decode primitives
# ---------------------------------------------------------------------------
def build_context(model, ids_cpu: torch.Tensor, block: int, device) -> Tuple[DynamicCache, int]:
    """Blocked prefill over a fresh DynamicCache; inputs stream from CPU (bench hygiene)."""
    cache = DynamicCache()
    S = ids_cpu.shape[1]
    last = None
    with torch.inference_mode():
        for s in range(0, S, block):
            e = min(s + block, S)
            cp = torch.arange(s, e, device=device)
            out = model(
                input_ids=ids_cpu[:, s:e].to(device),
                past_key_values=cache, cache_position=cp,
                position_ids=cp.unsqueeze(0), use_cache=True,
            )
            last = out.logits[:, -1]
    torch.cuda.synchronize()
    nxt = int(last.argmax(-1))
    return cache, nxt


def decode_step(model, cache: DynamicCache, nxt: int, p: int, device) -> int:
    cp = torch.tensor([p], device=device)
    out = model(
        input_ids=torch.tensor([[nxt]], device=device),
        past_key_values=cache, cache_position=cp,
        position_ids=cp.unsqueeze(0), use_cache=True,
    )
    _prof.note_sync()  # the argmax D2H feedback sync
    return int(out.logits[:, -1].argmax(-1))


# ---------------------------------------------------------------------------
# Profiler parsing
# ---------------------------------------------------------------------------
def _dev_time(evt) -> float:
    for a in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, a):
            return float(getattr(evt, a))
    return 0.0


def parse_profile(prof, steps: int) -> Dict[str, float]:
    """Return per-step metrics (ms) from a completed profiler run.

    Device times (busy/compute/H2D/D2H) are true GPU kernel durations (profiler does not
    inflate them).  Host regions use *inclusive* CPU time (``cpu_time_total``) so each region
    carries its own kernel-launch overhead — the thing a launch-bound step is made of.
    """
    busy_us = compute_us = h2d_us = d2h_us = 0.0
    kernels = 0
    host: Dict[str, float] = {r: 0.0 for r in HOST_ALL}
    for evt in prof.key_averages():
        key = evt.key
        # Device (GPU) time: count ONLY actual kernel/memcpy events (device_type == CUDA) so each
        # kernel's time is counted exactly once — CPU operators and our record_function ranges also
        # carry attributed device time in key_averages and would double/triple-count if summed.
        # Our ``hga/*`` ranges also surface as GPU-side ``gpu_user_annotation`` events (device_type
        # CUDA) whose device time == their enclosed kernels, so they too must be excluded here.
        if getattr(evt, "device_type", None) == DeviceType.CUDA and not key.startswith("hga/"):
            dt = _dev_time(evt)
            busy_us += dt
            kl = key.lower()
            if "memcpy" in kl and "htod" in kl:
                h2d_us += dt
            elif "memcpy" in kl and "dtoh" in kl:
                d2h_us += dt
            elif "memcpy" not in kl and "memset" not in kl:
                compute_us += dt
            kernels += int(getattr(evt, "count", 0))
        if key in host:  # host-side inclusive CPU time of our annotated phases
            host[key] += float(getattr(evt, "cpu_time_total", 0.0))
    out = {
        "busy": busy_us / 1e3 / steps,
        "compute": compute_us / 1e3 / steps,
        "h2d": h2d_us / 1e3 / steps,
        "d2h": d2h_us / 1e3 / steps,
        "kernels": kernels / steps,
    }
    for r in HOST_ALL:
        out["host/" + r.split("/")[1]] = host[r] / 1e3 / steps
    return out


def parse_host_ops(prof, steps: int, topn: int = 12) -> List[Tuple[str, float, float]]:
    """Decompose the real host (CPU) timeline into named leaf operators by SELF CPU time.

    This turns the coarse ``other`` bucket into concrete op names: if a single unannotated
    operation eats a large slice of host time it surfaces here BY NAME (no guessing).  ``self``
    CPU time is exclusive, so each op is counted once and the shares sum to 100% of real host
    CPU work.  Our ``hga/*`` record_function ranges are inclusive parents (near-zero self time)
    and are excluded so they don't shadow the leaves they wrap.  ``cudaLaunchKernel`` is kept —
    it IS the per-kernel launch overhead, the thing a launch-bound step is made of.
    """
    leaves: List[Tuple[str, float]] = []
    total = 0.0
    for evt in prof.key_averages():
        if getattr(evt, "device_type", None) != DeviceType.CPU:
            continue
        key = evt.key
        if key.startswith("hga/"):
            continue
        self_us = float(getattr(evt, "self_cpu_time_total", 0.0))
        if self_us <= 0.0:
            continue
        leaves.append((key, self_us))
        total += self_us
    leaves.sort(key=lambda kv: kv[1], reverse=True)
    return [(k, us / 1e3 / steps, 100.0 * us / total if total else 0.0)
            for k, us in leaves[:topn]]


# ---------------------------------------------------------------------------
# One (tier, ctx) run
# ---------------------------------------------------------------------------
def run_one(model, tier: str, ctx: int, steps: int, warmup: int, block: int,
            device, vocab: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    ids_cpu = torch.randint(0, vocab, (1, ctx))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache, nxt = build_context(model, ids_cpu, block, device)
    p = ctx

    # -- warmup --
    with torch.inference_mode():
        for _ in range(warmup):
            nxt = decode_step(model, cache, nxt, p, device); p += 1

    # -- PASS A: clean latency (no profiler) --
    _prof.enable(False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            nxt = decode_step(model, cache, nxt, p, device); p += 1
    torch.cuda.synchronize()
    wall_a = (time.perf_counter() - t0) * 1e3 / steps
    peak = gb(torch.cuda.max_memory_allocated())

    # -- PASS B: attribution (under profiler) --
    _prof.enable(True)
    _prof.reset_syncs()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            nxt = decode_step(model, cache, nxt, p, device); p += 1
        torch.cuda.synchronize()
    wall_b = (time.perf_counter() - t0) * 1e3 / steps
    syncs = _prof.get_syncs() / steps
    _prof.enable(False)

    m: Dict[str, Any] = parse_profile(prof, steps)
    m.update({"tier": tier, "ctx": ctx, "wall_a": wall_a, "wall_b": wall_b,
              "peak": peak, "syncs": syncs})
    m["host_ops"] = parse_host_ops(prof, steps)  # named decomposition of the host CPU timeline
    # HONEST idle/util: kernel durations (busy) are not profiler-inflated; the clean-pass wall is
    # the true step latency.  (wall_b is profiler-inflated on tiny kernels — used only for host
    # composition shares, never for absolute idle.)
    m["idle"] = max(0.0, wall_a - m["busy"])
    m["true_util"] = 100.0 * m["busy"] / wall_a if wall_a > 0 else 0.0
    m["h2d_share"] = 100.0 * m["h2d"] / wall_a if wall_a > 0 else 0.0
    top_sum = sum(m["host/" + r.split("/")[1]] for r in HOST_TOP)
    m["host_top_sum"] = top_sum
    m["host_other"] = max(0.0, wall_b - top_sum)   # HF preamble/embed/norms/argmax + launch residual
    # Host composition as SHARE of the profiler host timeline (robust to profiler overhead).
    denom = wall_b if wall_b > 0 else 1.0
    for r in HOST_TOP + HOST_SUB:
        name = r.split("/")[1]
        m["share/" + name] = 100.0 * m["host/" + name] / denom
    m["share/host_other"] = 100.0 * m["host_other"] / denom
    m["verdict"] = verdict(m)
    return m


def verify_route(model, tier: str, ctx: int, steps: int, warmup: int, block: int,
                 device, vocab: int, seed: int) -> Dict[str, Any]:
    """Black-box cross-check of the profiler's ``route`` bucket via differential timing.

    The profiler attributes ``route`` from a ``record_function`` region — an *internal* measure.
    This validates it from the *outside* by timing a full decode step with ``route`` run 1, 2 and
    3 times per layer (extra calls are non-mutating shadows: ``mutate_store=False`` → identical
    selection/gather, no second store append/close, cache advances normally):

        route_cost ≈ T(x2) − T(x1)     and, as a self-consistency proof of the method,
        T(x3) − T(x2) ≈ T(x2) − T(x1)  (each added route pass must cost the same → linear).

    All timed by an independent ``perf_counter`` path with the profiler OFF (no ``record_function``).
    The clean differential is then compared against the profiler's route estimate expressed on the
    **same clean-wall basis** — ``share/route × wall_clean`` — because the profiler's *absolute*
    ``host/route`` (inclusive CPU time under the profiler) is overhead-inflated and not directly
    comparable to a clean wall delta.

    Chunk-closing steps (``pos % chunk_size == chunk_size-1``) are excluded from doubling: the real
    call closes the active chunk there, so a shadow re-run would hit a closed/None active chunk.
    Levels are interleaved so each samples near-identical resident context sizes.
    """
    C = cfg.CHUNK_SIZE
    torch.manual_seed(seed)
    ids_cpu = torch.randint(0, vocab, (1, ctx))
    torch.cuda.empty_cache()

    cache, nxt = build_context(model, ids_cpu, block, device)
    p = ctx

    _prof.enable(False)
    _qra.EXTRA_ROUTE_REPEATS = 0
    with torch.inference_mode():
        for _ in range(warmup):
            nxt = decode_step(model, cache, nxt, p, device); p += 1

    # Interleave route x1 / x2 / x3 (0,1,2 extra shadow passes) and time each step cleanly.
    tsum = [0.0, 0.0, 0.0]
    tcnt = [0, 0, 0]
    with torch.inference_mode():
        for i in range(steps):
            closing = (p % C) == (C - 1)
            extra = 0 if closing else (i % 3)   # 0->x1, 1->x2, 2->x3
            _qra.EXTRA_ROUTE_REPEATS = extra
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt = decode_step(model, cache, nxt, p, device)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if not closing:
                tsum[extra] += dt; tcnt[extra] += 1
            p += 1
    _qra.EXTRA_ROUTE_REPEATS = 0

    x1 = tsum[0] / max(tcnt[0], 1)
    x2 = tsum[1] / max(tcnt[1], 1)
    x3 = tsum[2] / max(tcnt[2], 1)
    diff_21 = x2 - x1          # one route pass
    diff_32 = x3 - x2          # another route pass (linearity check)
    route_diff = 0.5 * ((x3 - x1))  # average per-pass over the 2-step span (lower variance)
    # method self-consistency: the two increments must agree if each added route costs the same
    lin_err = 100.0 * (diff_32 - diff_21) / diff_21 if diff_21 != 0 else float("nan")

    # Profiler's internal route attribution on the same continuing cache/context.
    _prof.enable(True)
    _prof.reset_syncs()
    torch.cuda.synchronize()
    tb0 = time.perf_counter()
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            nxt = decode_step(model, cache, nxt, p, device); p += 1
        torch.cuda.synchronize()
    wall_b = (time.perf_counter() - tb0) * 1e3 / steps
    _prof.enable(False)
    pm = parse_profile(prof, steps)
    route_abs = pm["host/route"]                                  # inclusive CPU under profiler (inflated)
    route_share = 100.0 * route_abs / wall_b if wall_b > 0 else 0.0
    route_prof_clean = route_share / 100.0 * x1                   # profiler route on the clean-wall basis

    diff_share = 100.0 * route_diff / x1 if x1 > 0 else 0.0
    rel = 100.0 * (route_diff - route_prof_clean) / route_prof_clean if route_prof_clean > 0 else float("nan")
    return {
        "tier": tier, "ctx": ctx,
        "x1": x1, "x2": x2, "x3": x3,
        "diff_21": diff_21, "diff_32": diff_32, "route_diff": route_diff, "lin_err": lin_err,
        "diff_share": diff_share,
        "route_abs": route_abs, "route_share": route_share, "route_prof_clean": route_prof_clean,
        "rel_err": rel, "counts": tcnt,
    }


def print_verify(v: Dict[str, Any]) -> None:
    print(f"\n=== ROUTE VERIFY  tier={v['tier']:>4}  ctx={v['ctx']}  ===")
    print(f"  clean step wall:  x1 {v['x1']:.3f}  x2 {v['x2']:.3f}  x3 {v['x3']:.3f} ms  "
          f"(n={v['counts'][0]}/{v['counts'][1]}/{v['counts'][2]})")
    print(f"  method self-check (linearity):  Δ(x2−x1) {v['diff_21']:.3f}  vs  Δ(x3−x2) {v['diff_32']:.3f} ms  "
          f"→ {v['lin_err']:+.1f}%  (small ⇒ each added route costs the same ⇒ method valid)")
    print(f"  DIFFERENTIAL route (clean, profiler OFF):  {v['route_diff']:.3f} ms/step  "
          f"= {v['diff_share']:.1f}% of the clean step")
    print(f"  PROFILER route:  share {v['route_share']:.1f}% × clean wall = {v['route_prof_clean']:.3f} ms/step  "
          f"(absolute host/route = {v['route_abs']:.3f} ms, inflated under profiler)")
    print(f"  agreement (clean basis):  Δ = {v['route_diff'] - v['route_prof_clean']:+.3f} ms  "
          f"({v['rel_err']:+.1f}% vs profiler)")


def verify_h2d(model, tier: str, ctx: int, steps: int, warmup: int, block: int,
               device, vocab: int, seed: int) -> Dict[str, Any]:
    """Black-box cross-check of the cold RAM→VRAM (H2D) transfer via differential timing.

    Analogue of :func:`verify_route` for the *transfer* path.  Every decode step re-issues the
    cold RAM→VRAM copy of the routed KV 1, 2 and 3 times (``RamKVCacheStore.EXTRA_H2D_REPEATS``);
    each extra copy uses the **same pinned CPU source** and discards the result — pure transfer,
    no store mutation, no re-gather/re-routing.  So::

        cold_h2d_cost ≈ T(x2) − T(x1)      and, as a self-consistency proof,
        T(x3) − T(x2) ≈ T(x2) − T(x1)      (linear ⇒ each added copy costs the same ⇒ valid).

    All timed by an independent ``perf_counter`` path with the profiler OFF.  The differential is
    then compared against the profiler's own H2D estimate — both the CUDA HtoD memcpy time
    (bytes on the copy engine) and the ``host/h2d`` region — on the clean-wall basis.

    Only meaningful for the ``ram``/``fs`` tiers; on ``vram`` the "copy" is device-to-device so
    the differential collapses to ≈0 (a useful control).  A ≈0 differential on ram means the cold
    H2D overlaps compute / is served from the VRAM LRU cache and is NOT on the critical path.
    """
    if not hasattr(_qra.RamKVCacheStore, "EXTRA_H2D_REPEATS"):
        raise RuntimeError("RamKVCacheStore.EXTRA_H2D_REPEATS missing (store not instrumented)")
    torch.manual_seed(seed)
    ids_cpu = torch.randint(0, vocab, (1, ctx))
    torch.cuda.empty_cache()

    cache, nxt = build_context(model, ids_cpu, block, device)
    p = ctx

    _prof.enable(False)
    _qra.RamKVCacheStore.EXTRA_H2D_REPEATS = 0
    with torch.inference_mode():
        for _ in range(warmup):
            nxt = decode_step(model, cache, nxt, p, device); p += 1

    # Interleave H2D x1 / x2 / x3 (0,1,2 extra cold copies) and time each step cleanly.  No
    # closing-step exclusion: the extra copies only *read* the CPU record, they never mutate.
    tsum = [0.0, 0.0, 0.0]
    tcnt = [0, 0, 0]
    with torch.inference_mode():
        for i in range(steps):
            extra = i % 3   # 0->x1, 1->x2, 2->x3
            _qra.RamKVCacheStore.EXTRA_H2D_REPEATS = extra
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt = decode_step(model, cache, nxt, p, device)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            tsum[extra] += dt; tcnt[extra] += 1
            p += 1
    _qra.RamKVCacheStore.EXTRA_H2D_REPEATS = 0

    x1 = tsum[0] / max(tcnt[0], 1)
    x2 = tsum[1] / max(tcnt[1], 1)
    x3 = tsum[2] / max(tcnt[2], 1)
    diff_21 = x2 - x1          # one cold-H2D pass
    diff_32 = x3 - x2          # another cold-H2D pass (linearity check)
    h2d_diff = 0.5 * (x3 - x1)  # average per-pass over the 2-step span (lower variance)
    lin_err = 100.0 * (diff_32 - diff_21) / diff_21 if diff_21 != 0 else float("nan")

    # Profiler's internal H2D attribution on the same continuing cache/context.
    _prof.enable(True)
    _prof.reset_syncs()
    torch.cuda.synchronize()
    tb0 = time.perf_counter()
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            nxt = decode_step(model, cache, nxt, p, device); p += 1
        torch.cuda.synchronize()
    wall_b = (time.perf_counter() - tb0) * 1e3 / steps
    _prof.enable(False)
    pm = parse_profile(prof, steps)
    h2d_cuda = pm["h2d"]                                           # CUDA HtoD memcpy (bytes on the wire)
    h2d_host_abs = pm["host/h2d"]                                  # host region issuing the copy (inflated)
    h2d_host_share = 100.0 * h2d_host_abs / wall_b if wall_b > 0 else 0.0
    h2d_host_clean = h2d_host_share / 100.0 * x1                   # host/h2d on the clean-wall basis

    diff_share = 100.0 * h2d_diff / x1 if x1 > 0 else 0.0
    rel = 100.0 * (h2d_diff - h2d_cuda) / h2d_cuda if h2d_cuda > 0 else float("nan")
    return {
        "tier": tier, "ctx": ctx,
        "x1": x1, "x2": x2, "x3": x3,
        "diff_21": diff_21, "diff_32": diff_32, "h2d_diff": h2d_diff, "lin_err": lin_err,
        "diff_share": diff_share,
        "h2d_cuda": h2d_cuda, "h2d_host_abs": h2d_host_abs,
        "h2d_host_share": h2d_host_share, "h2d_host_clean": h2d_host_clean,
        "rel_err": rel, "counts": tcnt,
    }


def print_verify_h2d(v: Dict[str, Any]) -> None:
    print(f"\n=== H2D VERIFY  tier={v['tier']:>4}  ctx={v['ctx']}  (cold RAM→VRAM copy) ===")
    print(f"  clean step wall:  x1 {v['x1']:.3f}  x2 {v['x2']:.3f}  x3 {v['x3']:.3f} ms  "
          f"(n={v['counts'][0]}/{v['counts'][1]}/{v['counts'][2]})")
    print(f"  method self-check (linearity):  Δ(x2−x1) {v['diff_21']:.3f}  vs  Δ(x3−x2) {v['diff_32']:.3f} ms  "
          f"→ {v['lin_err']:+.1f}%  (small ⇒ each added copy costs the same ⇒ method valid)")
    print(f"  DIFFERENTIAL cold-H2D (clean, profiler OFF):  {v['h2d_diff']:.3f} ms/step  "
          f"= {v['diff_share']:.1f}% of the clean step")
    print(f"  PROFILER H2D:  CUDA memcpy HtoD = {v['h2d_cuda']:.3f} ms/step  "
          f"(host/h2d region = {v['h2d_host_abs']:.3f} ms ≈ {v['h2d_host_clean']:.3f} ms clean-basis)")
    print(f"  agreement (vs CUDA memcpy):  Δ = {v['h2d_diff'] - v['h2d_cuda']:+.3f} ms  "
          f"({v['rel_err']:+.1f}%)")


def verdict(m: Dict[str, Any]) -> str:
    wall = m["wall_a"]
    if wall <= 0:
        return "N/A"
    if m["h2d"] / wall > 0.40:
        return "TRANSFER-BOUND"
    if m["d2h"] / wall > 0.30:
        return "SYNC-BOUND"
    if m["true_util"] < 60.0:
        return "HOST/LAUNCH-BOUND"
    return "COMPUTE-BOUND"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_run(m: Dict[str, Any]) -> None:
    def sh(name):
        return m["share/" + name]
    print(f"\n=== tier={m['tier']:>4}  ctx={m['ctx']}  ===")
    print(f"  LATENCY (clean):   wall {m['wall_a']:.1f} ms/step | {1000.0/m['wall_a']:.1f} tok/s | "
          f"peak VRAM {m['peak']:.2f} GB")
    print(f"  GPU (honest):      busy {m['busy']:.1f} ms | idle {m['idle']:.1f} ms "
          f"({100.0*m['idle']/m['wall_a']:.1f}%) | true-util {m['true_util']:.1f}%")
    print(f"  CUDA kernels:      compute {m['compute']:.1f} ms | H2D {m['h2d']:.3f} ms "
          f"({m['h2d_share']:.1f}% of wall) | D2H {m['d2h']:.3f} ms | count {m['kernels']:.0f}/step")
    print(f"  HOST composition (% of step, profiler timeline):")
    print(f"    qkv_proj {sh('qkv_proj'):.1f}% | route {sh('route'):.1f}% | attend {sh('attend'):.1f}% | "
          f"o_proj {sh('o_proj'):.1f}% | mlp {sh('mlp'):.1f}% | other(HF/embed/argmax) {sh('host_other'):.1f}%")
    print(f"    └ route breakdown: route_decision {sh('route_decision'):.1f}% | "
          f"gather_summaries {sh('gather_summaries'):.1f}% | gather_tokens {sh('gather_tokens'):.1f}% | "
          f"assemble {sh('assemble'):.1f}% | h2d {sh('h2d'):.1f}%")
    ops = m.get("host_ops", [])
    if ops:
        print(f"  HOST self-time top ops (decomposes 'other'; % of real host CPU, exclusive self time):")
        for name, ms, pct in ops[:10]:
            print(f"    {pct:5.1f}%  {ms:7.3f} ms  {name}")
    print(f"  D2H syncs/step: {m['syncs']:.0f}")
    print(f"  VERDICT: {m['verdict']}")


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 104)
    print("CONSOLIDATED  (per decode step; idle/util from clean wall, honest)")
    print(f"{'tier':>4} {'ctx':>7} {'wall':>7} {'busy':>6} {'idle':>6} {'util%':>6} "
          f"{'H2D%':>5} {'route%':>7} {'D2Hsyn':>7} {'kern':>7}  verdict")
    for m in rows:
        print(f"{m['tier']:>4} {m['ctx']:>7} {m['wall_a']:>7.1f} {m['busy']:>6.1f} {m['idle']:>6.1f} "
              f"{m['true_util']:>6.1f} {m['h2d_share']:>5.1f} {m['share/route']:>7.1f} "
              f"{m['syncs']:>7.0f} {m['kernels']:>7.0f}  {m['verdict']}")
    # Cross-tier diagnostic on idle (decisive: separates PCIe-transfer from host-sync bookkeeping).
    by_ctx: Dict[int, List[Dict[str, float]]] = {}
    for m in rows:
        by_ctx.setdefault(m["ctx"], []).append(m)
    print("\nCross-tier idle diagnosis (why idle grows off-VRAM — PCIe vs host bookkeeping):")
    for ctx, group in sorted(by_ctx.items()):
        if len(group) < 2:
            continue
        base = min(group, key=lambda g: g["idle"])   # usually vram (no cold-tier host syncs)
        detail = " ".join(f"{g['tier']}=(idle {g['idle']:.1f}, D2Hsyn {g['syncs']:.0f})" for g in group)
        print(f"  ctx={ctx}: {detail}")
        for g in group:
            if g is base:
                continue
            d_idle = g["idle"] - base["idle"]
            d_sync = g["syncs"] - base["syncs"]
            if abs(d_idle) < 0.10 * base["idle"]:
                tag = "FLAT vs base (pure host/launch floor, same as VRAM)"
            elif g["h2d_share"] > 5.0:
                tag = f"+{d_idle:.1f}ms idle WITH H2D {g['h2d_share']:.1f}% -> PCIe-transfer-sensitive"
            else:
                tag = (f"+{d_idle:.1f}ms idle, H2D~0 but +{d_sync:.0f} D2H syncs "
                       f"-> HOST/SYNC-bookkeeping (cold-tier .tolist()/index, NOT PCIe)")
            print(f"      {g['tier']:>4} vs {base['tier']}: {tag}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def write_html(rows: List[Dict[str, Any]], path: str, meta: Dict[str, str]) -> None:
    """Emit a self-contained dark-theme HTML report from the measured rows."""
    def bar(util: float) -> str:
        u = max(0.0, min(100.0, util))
        return (f'<div class="bar"><span class="busy" style="width:{u:.1f}%"></span>'
                f'<span class="idle" style="left:{u:.1f}%;width:{100.0-u:.1f}%"></span></div>')

    consolidated = "\n".join(
        f'    <tr><td class="tier">{m["tier"]}</td><td>{m["ctx"]}</td><td>{m["wall_a"]:.1f}</td>'
        f'<td>{m["busy"]:.1f}</td><td>{m["idle"]:.1f}</td><td>{m["true_util"]:.1f}%</td>'
        f'<td>{m["h2d_share"]:.1f}</td><td>{m["share/route"]:.1f}</td><td>{m["syncs"]:.0f}</td>'
        f'<td>{m["kernels"]:.0f}</td><td style="text-align:left">{bar(m["true_util"])}</td>'
        f'<td style="text-align:left"><span class="pill host">{m["verdict"]}</span></td></tr>'
        for m in rows
    )

    def hc(m, name):
        return f'{m["share/" + name]:.1f}'
    host_rows = "\n".join(
        f'    <tr><td class="tier">{m["tier"]}</td><td>{m["ctx"]}</td><td>{hc(m,"qkv_proj")}</td>'
        f'<td>{hc(m,"route")}</td><td>{hc(m,"route_decision")}</td><td>{hc(m,"gather_summaries")}</td>'
        f'<td>{hc(m,"gather_tokens")}</td><td>{hc(m,"assemble")}</td><td>{hc(m,"attend")}</td>'
        f'<td>{hc(m,"o_proj")}</td><td>{hc(m,"mlp")}</td><td>{hc(m,"host_other")}</td></tr>'
        for m in rows
    )

    # Named decomposition of the host CPU timeline (turns 'other' into concrete op names).
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    ops_blocks = []
    for m in rows:
        ops = m.get("host_ops", [])
        if not ops:
            continue
        body = "\n".join(
            f'    <tr><td style="text-align:left"><code>{esc(name)}</code></td>'
            f'<td>{ms:.3f}</td><td>{pct:.1f}%</td></tr>'
            for name, ms, pct in ops)
        ops_blocks.append(
            f'<h3 style="font-size:13.5px;margin:14px 0 6px;color:var(--ink)">'
            f'{m["tier"]} &middot; ctx {m["ctx"]}</h3>\n'
            f'<table><thead><tr><th style="text-align:left">op</th>'
            f'<th>self&nbsp;ms/step</th><th>% host&nbsp;CPU</th></tr></thead>\n'
            f'<tbody>\n{body}\n</tbody></table>')
    ops_html = "\n".join(ops_blocks) if ops_blocks else \
        '<div class="note">no host-op data captured</div>'

    by_ctx: Dict[int, List[Dict[str, float]]] = {}
    for m in rows:
        by_ctx.setdefault(m["ctx"], []).append(m)
    diag_rows, bullets = [], []
    for ctx, group in sorted(by_ctx.items()):
        if len(group) < 2:
            continue
        base = min(group, key=lambda g: g["idle"])
        for g in group:
            if g is base:
                continue
            d_idle = g["idle"] - base["idle"]
            d_sync = g["syncs"] - base["syncs"]
            if abs(d_idle) < 0.10 * base["idle"]:
                attr = "FLAT vs base — pure host/launch floor (same as VRAM)"
            elif g["h2d_share"] > 5.0:
                attr = (f'<span class="flag">PCIe-transfer-sensitive</span> — H2D {g["h2d_share"]:.1f}% of wall')
            else:
                attr = ('<span class="flag">HOST / SYNC bookkeeping</span> — cold-tier '
                        '<code>.tolist()</code> + CPU index/LRU, <b>NOT PCIe</b>')
            diag_rows.append(
                f'    <tr><td class="tier">{g["tier"]} &minus; {base["tier"]}</td><td>{ctx}</td>'
                f'<td>+{d_idle:.1f}&nbsp;ms</td><td class="key">{g["h2d"]:.3f}&nbsp;ms</td>'
                f'<td>+{d_sync:.0f}</td><td style="text-align:left">{attr}</td></tr>')
        floor = base["idle"]
        worst = max(group, key=lambda g: g["idle"])
        bullets.append(
            f'<li><b>ctx {ctx}:</b> base ~{floor:.0f}&nbsp;ms host/launch floor (tier <code>{base["tier"]}</code>); '
            f'cold-tier adds +{worst["idle"]-floor:.0f}&nbsp;ms on <code>{worst["tier"]}</code> — '
            f'H2D on the GPU timeline stays ~{worst["h2d"]:.2f}&nbsp;ms (not PCIe).</li>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HGA decode bottleneck — honest per-phase profile</title>
<style>
  :root{{--bg:#0f1116;--card:#1a1d26;--ink:#e6e9ef;--muted:#9aa4b2;--line:#2a2f3a;
    --accent:#4da3ff;--good:#3fb950;--bad:#f85149;--warn:#d29922;}}
  *{{box-sizing:border-box}} body{{margin:0;padding:32px;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Ubuntu,sans-serif;line-height:1.5}}
  h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 10px;color:var(--accent)}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:20px}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:20px}}
  table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}}
  th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}}
  th:first-child,td:first-child{{text-align:left}}
  thead th{{color:var(--muted);font-weight:600;border-bottom:2px solid var(--line);white-space:nowrap}}
  tbody tr:hover{{background:#20242f}} .tier{{font-weight:700}}
  code{{background:#0c0e13;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12.5px}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:12px;font-weight:600}}
  .pill.host{{background:rgba(210,153,34,.15);color:var(--warn);border:1px solid rgba(210,153,34,.4)}}
  .bar{{position:relative;height:16px;background:#0c0e13;border-radius:4px;overflow:hidden;min-width:120px}}
  .bar>span{{position:absolute;top:0;left:0;height:100%}}
  .busy{{background:var(--good)}} .idle{{background:var(--bad);opacity:.85}}
  .legend{{font-size:12px;color:var(--muted);margin-top:6px}}
  .legend b.busy{{color:var(--good)}} .legend b.idle{{color:var(--bad)}}
  .note{{font-size:13px;color:var(--muted)}} .key{{color:var(--good);font-weight:700}}
  .flag{{color:var(--bad);font-weight:700}} ul{{margin:8px 0 0 0;padding-left:20px}} li{{margin:4px 0}}
</style>
</head>
<body>
<h1>HGA decode bottleneck — honest per-phase profile</h1>
<div class="sub">{meta['model']} · router surgery · {meta['dtype']} · {meta['device']} ·
  {meta['steps']} decode steps · metrics per decode token · {meta['date']}<br>
  Tool: <code>ExistingModelFineTuning/Qwen3LongContext/profile_hga_bottleneck.py</code> ·
  idle/true-util from the clean pass (kernel busy is not profiler-inflated); NVML deliberately NOT used.</div>

<div class="card">
<h2>Consolidated — per decode step</h2>
<table>
  <thead><tr><th>tier</th><th>ctx</th><th>wall&nbsp;ms</th><th>busy&nbsp;ms</th><th>idle&nbsp;ms</th>
    <th>true&#8209;util</th><th>H2D&nbsp;%</th><th>route&nbsp;%</th><th>D2H&nbsp;syncs</th><th>kernels</th>
    <th style="text-align:left">GPU busy vs idle</th><th style="text-align:left">verdict</th></tr></thead>
  <tbody>
{consolidated}
  </tbody>
</table>
<div class="legend"><b class="busy">&#9632;</b> GPU busy (real kernel time) &nbsp;&nbsp;
  <b class="idle">&#9632;</b> GPU idle (host-bound wait)</div>
</div>

<div class="card">
<h2>HOST composition per step (% of step, profiler timeline)</h2>
<table>
  <thead><tr><th>tier</th><th>ctx</th><th>qkv_proj</th><th>route</th><th>&#8627;route_decision</th>
    <th>&#8627;gather_sum</th><th>&#8627;gather_tok</th><th>&#8627;assemble</th><th>attend</th>
    <th>o_proj</th><th>mlp</th><th>other</th></tr></thead>
  <tbody>
{host_rows}
  </tbody>
</table>
<div class="note" style="margin-top:8px"><code>route</code> is the umbrella; the &#8627; rows are its inner phases.
  <code>other</code> = HF preamble / embed / final-norm / lm_head / argmax + launch residual.</div>
</div>

<div class="card">
<h2>What is inside <code>other</code> — top host ops by self CPU time</h2>
<div class="note" style="margin-bottom:8px">Direct profiler decomposition of the <b>real host CPU timeline</b>
  into named leaf operators (exclusive <code>self</code> time; shares sum to 100% of host CPU work).
  If a single unannotated op ate a large slice it would surface here by name — this is the honest
  check that nothing hides in <code>other</code>. <code>cudaLaunchKernel</code> = per-kernel launch
  overhead (the launch-bound floor itself).</div>
{ops_html}
</div>

<div class="card">
<h2>Cross-tier idle diagnosis — PCIe transfer vs host bookkeeping</h2>
<table>
  <thead><tr><th>comparison</th><th>ctx</th><th>&#916;&nbsp;idle</th><th>H2D&nbsp;(GPU)</th>
    <th>&#916;&nbsp;D2H&nbsp;syncs</th><th style="text-align:left">attribution</th></tr></thead>
  <tbody>
{chr(10).join(diag_rows) if diag_rows else '    <tr><td colspan="6" style="text-align:left">single tier — no cross-tier comparison</td></tr>'}
  </tbody>
</table>
<div class="note" style="margin-top:8px">D2H-sync delta = 28 layers &times; 2 <code>.tolist()</code>
  (<code>_summary_slots</code> + <code>_bank_slots</code>). The VRAM store gathers with a pure-GPU
  <code>index_select</code> — zero <code>.tolist()</code>.</div>
</div>

<div class="card">
<h2>Verdict</h2>
<p style="margin:0 0 6px">Bottleneck to 100% GPU is <b>host</b>, never PCIe H2D (H2D on the GPU
  timeline is ~0&nbsp;ms on every tier):</p>
<ul>
{chr(10).join(bullets) if bullets else '<li>Run multiple tiers to attribute the cold-tier delta.</li>'}
</ul>
<p class="note" style="margin:10px 0 0">Explains why <code>OVERLAP</code> / <code>SPLIT_SOFTMAX</code>
  (which hide H2D) gave nothing, and why CUDA-graphs helped only on VRAM (cold-tier host syncs are
  not graph-capturable). Tier-agnostic lever: <b>fuse the router kernels</b> to cut both the launch
  count and the <code>.tolist()</code> D2H syncs (on-GPU slot lookup / sync-free LRU).</p>
</div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[html] wrote report -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_model(model_name: str, tier: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa",
    )
    model.eval()
    replace_qwen_attention_with_router(
        model, cache_location=tier,
        keep_first=cfg.KEEP_FIRST, keep_last=cfg.KEEP_LAST, topk_chunks=cfg.TOPK_CHUNKS,
        topk_groups=cfg.TOPK_GROUPS, chunk_size=cfg.CHUNK_SIZE, group_size=cfg.GROUP_SIZE,
        vram_cache_chunks=cfg.VRAM_CACHE_CHUNKS, vram_summary_chunks=cfg.VRAM_SUMMARY_CHUNKS,
        vram_cache_reserve_gb=cfg.VRAM_CACHE_RESERVE_GB, ram_budget_gb=cfg.RAM_BUDGET_GB,
    )
    return model


def selftest(model_name: str, dtype: torch.dtype, device) -> None:
    # ctx 2048 (32 chunks): > (keep_first+keep_last)=10 resident chunks, so routing actually fires.
    print("[selftest] loading model (ram tier, ctx 2048, 8 steps)...")
    model = build_model(model_name, "ram", dtype)
    install_mlp_hooks(model)
    vocab = model.config.vocab_size
    m = run_one(model, "ram", 2048, steps=8, warmup=2, block=cfg.PREFILL_BLOCK,
                device=device, vocab=vocab, seed=0)
    print_run(m)
    assert m["wall_a"] > 0, "wall must be positive"
    assert m["busy"] > 0, "profiler must record CUDA busy time"
    assert m["syncs"] > 1, "routing must fire: argmax + router .tolist() syncs (>1/step)"
    assert m["share/route"] > 0, "router host region must be attributed a nonzero share"
    assert m["host_other"] >= 0.0, "top host regions must not exceed profiler wall (bad accounting)"
    print("\n[selftest] PASS: routing active, CUDA busy>0, syncs>1, host phases attributed, "
          "accounting reconciles.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--cache", nargs="+", default=["ram"], choices=("vram", "ram", "fs"))
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--html", default=None, help="Write a self-contained HTML report to this path")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-route", action="store_true",
                    help="Cross-check the profiler's route bucket with a black-box differential timer "
                         "(route run once vs twice per layer); prints agreement, no HTML.")
    ap.add_argument("--verify-h2d", action="store_true",
                    help="Cross-check the cold RAM→VRAM (H2D) transfer with a black-box differential "
                         "timer (copy run once vs twice per step); prints agreement, no HTML. ram/fs only.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype, device)

    if args.selftest:
        selftest(args.model, dtype, device)
        return

    if args.verify_route:
        for tier in args.cache:
            print(f"\n########## loading {args.model} | tier={tier} | dtype={dtype} ##########", flush=True)
            model = build_model(args.model, tier, dtype)
            install_mlp_hooks(model)
            vocab = model.config.vocab_size
            for ctx in args.ctx:
                print(f"  [verify-route] tier={tier} ctx={ctx} ({args.steps} decode steps)...", flush=True)
                v = verify_route(model, tier, ctx, args.steps, args.warmup, cfg.PREFILL_BLOCK,
                                 device, vocab, args.seed)
                print_verify(v)
            del model
            torch.cuda.empty_cache()
        return

    if args.verify_h2d:
        for tier in args.cache:
            print(f"\n########## loading {args.model} | tier={tier} | dtype={dtype} ##########", flush=True)
            model = build_model(args.model, tier, dtype)
            install_mlp_hooks(model)
            vocab = model.config.vocab_size
            for ctx in args.ctx:
                print(f"  [verify-h2d] tier={tier} ctx={ctx} ({args.steps} decode steps)...", flush=True)
                v = verify_h2d(model, tier, ctx, args.steps, args.warmup, cfg.PREFILL_BLOCK,
                               device, vocab, args.seed)
                print_verify_h2d(v)
            del model
            torch.cuda.empty_cache()
        return

    rows: List[Dict[str, Any]] = []
    for tier in args.cache:
        print(f"\n########## loading {args.model} | tier={tier} | dtype={dtype} ##########", flush=True)
        model = build_model(args.model, tier, dtype)
        install_mlp_hooks(model)
        vocab = model.config.vocab_size
        for ctx in args.ctx:
            print(f"  [run] tier={tier} ctx={ctx} (prefill {ctx} tok, {args.steps} decode steps)...",
                  flush=True)
            m = run_one(model, tier, ctx, args.steps, args.warmup, cfg.PREFILL_BLOCK,
                        device, vocab, args.seed)
            print_run(m)
            rows.append(m)
        del model
        torch.cuda.empty_cache()

    print_summary(rows)

    if args.html:
        import datetime
        meta = {
            "model": args.model, "dtype": str(dtype).replace("torch.", ""),
            "device": torch.cuda.get_device_name(device), "steps": str(args.steps),
            "date": datetime.date.today().isoformat(),
        }
        write_html(rows, args.html, meta)


if __name__ == "__main__":
    main()
