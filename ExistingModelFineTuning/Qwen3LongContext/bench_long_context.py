#!/usr/bin/env python3
"""Long-context benchmark harness for the RAM/FS/VRAM-cached KvRouter chat model.

Measures **where the time and bytes go** for `chat_qwen30b_fp8.py`'s routed attention *before*
any architecture change, so the real bottleneck (Python/launch overhead, CPU↔GPU / NVMe transfer,
attention math, DCA, or the MoE MLP) is visible rather than guessed.

Per matrix cell (DCA on/off × cache ram|fs|vram × context × output length) it reports:

* **Throughput / latency** — `prefill_tok_s`, `decode_tok_s`, `TTFT` (prefill + first token),
  and decode inter-token latency `ITL` p50 / p99.
* **Per-stage GPU time** (from a short, separate profiling pass) — qkv, RoPE, chunk top-k,
  group top-k, KV gather, attention, MLP, plus `attn_total` and an `other` remainder; and the
  **GPU-busy vs wall** gap that exposes Python / launch-bound steps.
* **Transfer / cache counters** — H2D bytes/token (derived from token-bank + summary-cache
  misses), disk reads & bytes/token (fs tier), token-bank hit rate, summary-cache hit rate.

The model is loaded once and reconfigured per cell.  Random token ids are used (the goal is the
*performance* profile, which is content-independent); pass `--prompt-file` to use real text.

This box's GPU may be too small to load the FP8 weights — run on a capable machine:

    source ~/my_env/bin/activate
    cd ~/HierarchicalGlobalAttention

    # Full requested matrix (warning: 128K prefill + 8K decode cells are expensive):
    python -m ExistingModelFineTuning.Qwen3LongContext.bench_long_context \
        --ctx 32768,131072 --out 256,2048,8192 --cache ram,fs,vram --dca 0,1 \
        --json bench_results.json

    # Quick smoke (one small cell):
    python -m ExistingModelFineTuning.Qwen3LongContext.bench_long_context \
        --ctx 4096 --out 128 --cache ram --dca 0 --warmup 4 --profile-steps 8
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import statistics
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    replace_qwen_attention_with_router,
    restore_original_attention,
)

try:
    from KvRouter.bench_profiler import prof  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter.bench_profiler import prof  # type: ignore


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# Router config — mirrors chat_qwen30b_fp8.py so the numbers describe the shipped chat.
CHUNK_SIZE = 64
GROUP_SIZE = 16
KEEP_FIRST = 2
KEEP_LAST = 8
TOPK_CHUNKS = 20
TOPK_GROUPS = 32
VRAM_CACHE_CHUNKS = 400
VRAM_SUMMARY_CHUNKS = 8192
VRAM_CACHE_RESERVE_GB = 1.5
# Official Qwen3-30B-A3B-Instruct-2507 DCA (HF config_1m.json): chunk 128K, local 4K.
DCA_CHUNK = 131072
DCA_LOCAL = 4096

# Per-stage labels in display order; `other` is attn_total minus the named attention sub-stages.
_ATTN_STAGES = ["qkv", "rope", "chunk_topk", "group_topk", "kv_gather", "attention"]


def gb(x: float) -> float:
    return x / 1024**3


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading {model_name} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda", attn_implementation="sdpa",
    ).eval()
    torch.cuda.synchronize()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s "
          f"({gb(torch.cuda.memory_allocated()):.1f}GB allocated)", flush=True)
    return tok, model


def instrument_mlp(model) -> None:
    """Wrap every decoder layer's MLP forward so its GPU time is attributed to the `mlp` stage.

    The wrapper is permanent but costs nothing when profiling is off (`prof.section` returns a
    shared no-op context manager), so it never affects the throughput/ITL passes.
    """
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "_bench_wrapped", False):
            continue
        orig_forward = mlp.forward

        def make(of):
            def fwd(*a, **k):
                with prof.section("mlp"):
                    return of(*a, **k)
            return fwd

        mlp.forward = make(orig_forward)
        mlp._bench_wrapped = True


def build_ids(tok, n_tokens: int, device, seed: int, prompt_file: Optional[str]) -> torch.Tensor:
    """A length-`n_tokens` context.  Random in-vocab ids by default (perf is content-independent);
    if `prompt_file` is given its text is tiled/truncated to the requested length."""
    if prompt_file:
        text = open(prompt_file, "r", encoding="utf-8", errors="ignore").read()
        ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        if not ids:
            ids = [tok.eos_token_id or 0]
        while len(ids) < n_tokens:
            ids = ids + ids
        ids = torch.tensor(ids[:n_tokens], dtype=torch.long)
        return ids.unsqueeze(0).to(device)
    vocab = 0
    for attr in ("vocab_size",):
        vocab = int(getattr(tok, attr, 0) or 0)
        if vocab:
            break
    if not vocab:
        try:
            vocab = len(tok)
        except TypeError:
            vocab = 151000
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, max(2, vocab - 1), (1, n_tokens), generator=g, dtype=torch.long)
    return ids.to(device)


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def _store_of(cache):
    router = getattr(cache, "_kv_router", None)
    return getattr(router, "store", None) if router is not None else None


def _dispose(cache) -> None:
    store = _store_of(cache)
    if store is not None:
        try:
            disk = getattr(store, "disk", None)
            if disk is not None and hasattr(disk, "flush"):
                disk.flush()
            if hasattr(store, "close"):
                store.close()
            elif hasattr(store, "reset"):
                store.reset()
        except Exception:
            pass
    try:
        if cache is not None and hasattr(cache, "_kv_router"):
            delattr(cache, "_kv_router")
    except AttributeError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _h2d_bytes(store, cache_misses: int, summary_misses: int) -> int:
    """H2D bytes brought across PCIe, derived from cold-tier misses (the cached-path traffic).

    Each token-bank miss copies one chunk's K+V (`B·KVH·C·Dh`·2); each summary-cache miss copies
    one chunk's group K+V (`B·KVH·M·Dh`·2).  This is exact for the cached path the matrix uses
    (caches are sized to fit); a fully-uncached fallback path would not be counted."""
    if store is None:
        return 0
    b = store._dtype_bytes
    tok_pc = 2 * store.B * store.kvh * store.C * store.dh * b
    sum_pc = 2 * store.B * store.kvh * store.M * store.dh * b
    return cache_misses * tok_pc + summary_misses * sum_pc


def run_cell(model, ids, out_len: int, prefill_block: int, warmup: int,
             profile_steps: int, device) -> Dict:
    """Prefill `ids`, decode `out_len` tokens (timing TTFT/ITL/throughput + counters), then a short
    profiling pass for the per-stage GPU breakdown.  Returns a metrics dict."""
    S = ids.shape[1]
    torch.cuda.reset_peak_memory_stats()
    prof.disable()
    prof.reset()

    cache = DynamicCache()
    try:
        return _run_cell_inner(model, ids, out_len, prefill_block, warmup, profile_steps, device, cache, S)
    finally:
        _dispose(cache)


def _run_cell_inner(model, ids, out_len, prefill_block, warmup, profile_steps, device, cache, S) -> Dict:
    with torch.inference_mode():
        # ---- prefill (blocked, like the chat) ----
        torch.cuda.synchronize()
        t_pf = time.perf_counter()
        out = None
        for s in range(0, S, prefill_block):
            e = min(s + prefill_block, S)
            cp = torch.arange(s, e, device=device)
            out = model(input_ids=ids[:, s:e], past_key_values=cache, cache_position=cp,
                        position_ids=cp.unsqueeze(0), use_cache=True)
        torch.cuda.synchronize()
        prefill_t = time.perf_counter() - t_pf

        nxt = int(out.logits[:, -1].argmax(-1))
        p = S

        # ---- first decode token → TTFT ----
        torch.cuda.synchronize()
        t_first = time.perf_counter()
        cp = torch.tensor([p], device=device)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        torch.cuda.synchronize()
        first_tok_t = time.perf_counter() - t_first
        nxt = int(out.logits[:, -1].argmax(-1)); p += 1
        ttft = prefill_t + first_tok_t

        # ---- steady-state decode, per-token ITL ----
        # The first `warmup` decode steps are run untimed (lazy VRAM-bank allocation, kernel
        # autotune, cache fill) so ITL/throughput reflect steady state.  Total generated == out_len.
        itls: List[float] = []
        n_decode = max(0, out_len - 1)
        n_warm = min(max(0, warmup), n_decode)
        store = _store_of(cache)
        for _ in range(n_warm):
            cp = torch.tensor([p], device=device)
            out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            nxt = int(out.logits[:, -1].argmax(-1)); p += 1
        torch.cuda.synchronize()
        # snapshot counters after warmup so the rate reflects steady decode
        c0 = _counter_snapshot(store)
        for _ in range(n_decode - n_warm):
            torch.cuda.synchronize()
            t = time.perf_counter()
            cp = torch.tensor([p], device=device)
            out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            torch.cuda.synchronize()
            itls.append(time.perf_counter() - t)
            nxt = int(out.logits[:, -1].argmax(-1)); p += 1
        c1 = _counter_snapshot(store)

        # ---- per-stage profiling pass (bounded number of extra decode steps) ----
        stage_ms, gpu_busy_ms, wall_ms = _profile_pass(
            model, cache, nxt, p, profile_steps, device)

    # ---- assemble metrics ----
    decode_steps = len(itls)
    decode_t = sum(itls)
    metrics: Dict = {
        "ctx": S, "out": out_len,
        "ttft_s": ttft, "prefill_t_s": prefill_t,
        "prefill_tok_s": S / prefill_t if prefill_t > 0 else 0.0,
        "decode_tok_s": decode_steps / decode_t if decode_t > 0 else 0.0,
        "itl_p50_ms": (statistics.median(itls) * 1e3) if itls else 0.0,
        "itl_p99_ms": (_pct(itls, 99) * 1e3) if itls else 0.0,
        "peak_gb": gb(torch.cuda.max_memory_allocated()),
    }

    # transfer / cache counters over the steady-decode window
    dh = max(1, decode_steps)
    d_hits = c1["cache_hits"] - c0["cache_hits"]
    d_miss = c1["cache_misses"] - c0["cache_misses"]
    d_shits = c1["summary_hits"] - c0["summary_hits"]
    d_smiss = c1["summary_misses"] - c0["summary_misses"]
    d_dreads = c1["disk_reads"] - c0["disk_reads"]
    d_dbytes = c1["disk_read_bytes"] - c0["disk_read_bytes"]
    h2d = _h2d_bytes(store, d_miss, d_smiss)
    metrics.update({
        "tokbank_hit_rate": 100.0 * d_hits / max(1, d_hits + d_miss),
        "summary_hit_rate": 100.0 * d_shits / max(1, d_shits + d_smiss),
        "h2d_mb_per_tok": (h2d / 1e6) / dh,
        "disk_reads_per_tok": d_dreads / dh,
        "disk_mb_per_tok": (d_dbytes / 1e6) / dh,
    })

    # per-stage GPU time per decode step (summed over all layers)
    steps = max(1, profile_steps)
    per_step = {k: v / steps for k, v in stage_ms.items()}
    attn_total = per_step.get("attn_total", 0.0)
    named = sum(per_step.get(s, 0.0) for s in _ATTN_STAGES)
    per_step["other_attn"] = max(0.0, attn_total - named)
    metrics["stage_ms_per_step"] = per_step
    metrics["gpu_busy_ms_per_step"] = gpu_busy_ms / steps
    metrics["wall_ms_per_step"] = wall_ms / steps
    metrics["python_overhead_ms_per_step"] = max(0.0, (wall_ms - gpu_busy_ms) / steps)
    return metrics


def _counter_snapshot(store) -> Dict[str, int]:
    if store is None:
        return {k: 0 for k in ("cache_hits", "cache_misses", "summary_hits",
                               "summary_misses", "disk_reads", "disk_read_bytes")}
    return {
        "cache_hits": getattr(store, "cache_hits", 0),
        "cache_misses": getattr(store, "cache_misses", 0),
        "summary_hits": getattr(store, "summary_hits", 0),
        "summary_misses": getattr(store, "summary_misses", 0),
        "disk_reads": getattr(store, "disk_reads", 0),
        "disk_read_bytes": getattr(store, "disk_read_bytes", 0),
    }


def _profile_pass(model, cache, nxt: int, p: int, profile_steps: int, device):
    """Run `profile_steps` extra decode steps with the profiler on; return
    (stage_ms_totals, gpu_busy_ms, wall_ms).  gpu_busy ≈ attn_total + mlp summed over the pass."""
    if profile_steps <= 0:
        return {}, 0.0, 0.0
    prof.reset()
    prof.enable()
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode():
        for _ in range(profile_steps):
            cp = torch.tensor([p], device=device)
            out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                        cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            nxt = int(out.logits[:, -1].argmax(-1)); p += 1
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t) * 1e3
    prof.disable()
    totals = dict(prof.totals_ms())
    gpu_busy = totals.get("attn_total", 0.0) + totals.get("mlp", 0.0)
    prof.reset()
    return totals, gpu_busy, wall_ms


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cell(label: str, m: Dict) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(f"  prefill {m['prefill_tok_s']:8.1f} tok/s   decode {m['decode_tok_s']:7.2f} tok/s   "
          f"TTFT {m['ttft_s']:7.2f}s   ITL p50 {m['itl_p50_ms']:7.1f}ms  p99 {m['itl_p99_ms']:7.1f}ms   "
          f"peak {m['peak_gb']:.1f}GB", flush=True)
    print(f"  H2D {m['h2d_mb_per_tok']:7.2f} MB/tok   disk {m['disk_reads_per_tok']:6.1f} reads/tok "
          f"({m['disk_mb_per_tok']:6.2f} MB/tok)   tok-bank hit {m['tokbank_hit_rate']:5.1f}%   "
          f"summary hit {m['summary_hit_rate']:5.1f}%", flush=True)
    ps = m["stage_ms_per_step"]
    order = ["qkv", "rope", "chunk_topk", "group_topk", "kv_gather", "attention",
             "other_attn", "attn_total", "mlp"]
    parts = "  ".join(f"{k}={ps.get(k, 0.0):.2f}" for k in order)
    print(f"  per-step GPU ms (all layers): {parts}", flush=True)
    print(f"  GPU-busy {m['gpu_busy_ms_per_step']:.2f}ms / wall {m['wall_ms_per_step']:.2f}ms  "
          f"→ Python/launch overhead {m['python_overhead_ms_per_step']:.2f}ms/step "
          f"({100.0 * m['python_overhead_ms_per_step'] / max(1e-9, m['wall_ms_per_step']):.0f}%)", flush=True)
    print(f"  bottleneck: {diagnose(m)}", flush=True)


def diagnose(m: Dict) -> str:
    """One-line heuristic: where the dominant decode-step cost is."""
    wall = m["wall_ms_per_step"]
    py = m["python_overhead_ms_per_step"]
    ps = m["stage_ms_per_step"]
    flags = []
    if wall > 0 and py / wall > 0.5:
        flags.append(f"Python/launch-bound ({100*py/wall:.0f}% of step not on GPU)")
    if m["disk_reads_per_tok"] > 0.5:
        flags.append(f"NVMe-bound ({m['disk_mb_per_tok']:.1f} MB/tok read)")
    if m["h2d_mb_per_tok"] > 8.0:
        flags.append(f"PCIe H2D-heavy ({m['h2d_mb_per_tok']:.1f} MB/tok)")
    # dominant GPU stage
    gpu_stages = {k: ps.get(k, 0.0) for k in _ATTN_STAGES + ["mlp"]}
    if any(v > 0 for v in gpu_stages.values()):
        top = max(gpu_stages, key=gpu_stages.get)
        flags.append(f"top GPU stage={top} ({gpu_stages[top]:.2f}ms/step)")
    return "; ".join(flags) if flags else "n/a"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ctx", default="32768,131072", help="comma list of context lengths")
    ap.add_argument("--out", default="256,2048,8192", help="comma list of output (decode) lengths")
    ap.add_argument("--cache", default="ram,fs,vram", help="comma list of tiers: ram|fs|vram")
    ap.add_argument("--dca", default="0,1", help="comma list of DCA flags: 0=off 1=on")
    ap.add_argument("--prefill-block", type=int, default=CHUNK_SIZE)
    ap.add_argument("--warmup", type=int, default=8, help="(reserved) warmup decode steps")
    ap.add_argument("--profile-steps", type=int, default=16,
                    help="extra decode steps timed per-stage for the GPU breakdown")
    ap.add_argument("--ram-budget-gb", type=float, default=12.0)
    ap.add_argument("--fs-cache-dir", default=None)
    ap.add_argument("--vram-cache", type=int, default=VRAM_CACHE_CHUNKS)
    ap.add_argument("--vram-summary", type=int, default=VRAM_SUMMARY_CHUNKS)
    ap.add_argument("--dca-chunk", type=int, default=DCA_CHUNK)
    ap.add_argument("--dca-local", type=int, default=DCA_LOCAL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-file", default=None, help="use this text instead of random ids")
    ap.add_argument("--json", default=None, help="write all cell metrics to this JSON file")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    ctx_list, out_list = _ints(args.ctx), _ints(args.out)
    cache_list = [c.strip() for c in args.cache.split(",") if c.strip()]
    dca_list = _ints(args.dca)

    tok, model = load_model(args.model)
    instrument_mlp(model)
    device = "cuda"

    n_cells = len(dca_list) * len(cache_list) * len(ctx_list) * len(out_list)
    print(f"\nMatrix: {len(dca_list)} DCA × {len(cache_list)} cache × {len(ctx_list)} ctx × "
          f"{len(out_list)} out = {n_cells} cells\n"
          f"Router: chunk={CHUNK_SIZE} group={GROUP_SIZE} keep_first={KEEP_FIRST} keep_last={KEEP_LAST} "
          f"topk_chunks={TOPK_CHUNKS} topk_groups={TOPK_GROUPS}\n", flush=True)

    results: List[Dict] = []
    for dca in dca_list:
        for cache_loc in cache_list:
            for ctx in ctx_list:
                ids = build_ids(tok, ctx, device, args.seed, args.prompt_file)
                for out_len in out_list:
                    label = f"dca={'on' if dca else 'off'} cache={cache_loc} ctx={ctx} out={out_len}"
                    print(f"\n[running] {label}", flush=True)
                    try:
                        replace_qwen_attention_with_router(
                            model, cache_location=cache_loc,
                            keep_first=KEEP_FIRST, keep_last=KEEP_LAST, topk_chunks=TOPK_CHUNKS,
                            topk_groups=TOPK_GROUPS, chunk_size=CHUNK_SIZE, group_size=GROUP_SIZE,
                            vram_cache_chunks=args.vram_cache, vram_summary_chunks=args.vram_summary,
                            vram_cache_reserve_gb=VRAM_CACHE_RESERVE_GB,
                            ram_budget_gb=args.ram_budget_gb, fs_cache_dir=args.fs_cache_dir,
                            dca_chunk=(args.dca_chunk if dca else 0),
                            dca_local=(args.dca_local if dca else 0),
                        )
                        m = run_cell(model, ids, out_len, args.prefill_block,
                                     args.warmup, args.profile_steps, device)
                        m["label"] = label
                        m["dca"] = bool(dca)
                        m["cache"] = cache_loc
                        results.append(m)
                        print_cell(label, m)
                    except Exception as exc:  # OOM or other per-cell failure → record and continue
                        torch.cuda.synchronize()
                        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                        results.append({"label": label, "dca": bool(dca), "cache": cache_loc,
                                        "ctx": ctx, "out": out_len, "error": str(exc)})
                    finally:
                        restore_original_attention(model)
                        gc.collect()
                        torch.cuda.empty_cache()
                del ids
                gc.collect()
                torch.cuda.empty_cache()

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 100, flush=True)
    _print_summary(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} cells to {args.json}", flush=True)


def _print_summary(results: List[Dict]) -> None:
    hdr = (f"{'cell':<48} {'prefill':>9} {'decode':>8} {'TTFT':>8} {'p50':>8} {'p99':>8} "
           f"{'H2D/tok':>9} {'disk/tok':>9}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for m in results:
        if "error" in m:
            print(f"{m['label']:<48} {'ERROR: ' + m['error'][:40]:>60}", flush=True)
            continue
        print(f"{m['label']:<48} {m['prefill_tok_s']:8.0f}/s {m['decode_tok_s']:7.2f}/s "
              f"{m['ttft_s']:7.1f}s {m['itl_p50_ms']:6.0f}ms {m['itl_p99_ms']:6.0f}ms "
              f"{m['h2d_mb_per_tok']:7.2f}MB {m['disk_mb_per_tok']:7.2f}MB", flush=True)


if __name__ == "__main__":
    main()
