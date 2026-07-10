#!/usr/bin/env python3
"""Quality test: KvRouter HGA vs. the original Ornith-1.0-9B (Qwen3.5 hybrid).

The Ornith analogue of
:mod:`ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed`.  Two stages:

1. **Offline selftests** (fast, no model download) — proves the surgery is exact: the
   ``OrnithRoutedAttention`` wrapper (gate + partial-RoPE + GQA) == stock ``Qwen3_5Attention``
   at full routing coverage, only the 8 full-attention layers are wrapped, and the reference
   Qwen3 dense-equivalence is not regressed.  These live in
   :mod:`.Tests.test_ornith_routed` and are simply invoked here.

2. **compare_on_ornith** — loads the 9B once (4-bit NF4 + fp16 compute for a 16GB Turing card,
   like :mod:`.try_ornith`) and runs a teacher-forced forward over an N-token context under
   attention implementations that share the *same* projections, RoPE and router config (all
   routed variants use ``attend(use_summaries=False)`` — real token KV, group value summaries
   never attended):
       * baseline : original Ornith dense full-attention (the reference)
       * group    : ChunkRouter group-level routing (open top groups of selected chunks)
       * chunk    : ChunkRouter whole-chunk routing (group_size == chunk_size; full chunks)
   Reports greedy next-token agreement vs. baseline (overall + by position) and per-token
   perplexity, picks the lower-loss variant, plus peak VRAM.

   ``--ram`` runs the long-context needle-in-haystack retrieval check instead (irrelevant
   prefix in host RAM, only routed chunks pulled to VRAM; VRAM stays flat as context grows).

Run (always from repo root, always the project .venv):

    python -m ExistingModelFineTuning.OrnithLongContext.test_ornith_routed --selftest-only
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -m ExistingModelFineTuning.OrnithLongContext.test_ornith_routed --tokens 4096
    python -m ExistingModelFineTuning.OrnithLongContext.test_ornith_routed --ram --ctx-sizes 2048 32768
"""

from __future__ import annotations

import argparse
import os
import time

# Antifragmentation for the tight 16GB card — must be set before the first CUDA alloc.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from .ornith_routed_attention import (
    replace_ornith_attention_with_router,
    restore_ornith_attention,
)
from .try_ornith import MODEL, _blocked_generate, _load_model, _needle_prompt, gb

# Model-agnostic helpers reused verbatim from the Qwen3 quality test (no model load at import).
from ..Qwen3LongContext.test_qwen30b_routed import (
    Metric,
    build_ids,
    variant_kwargs,
)


# -------------------------------------------------------------------------------------------------
# Stage 1: offline selftests (no model)
# -------------------------------------------------------------------------------------------------
def run_selftests() -> None:
    """Invoke the offline unit-equivalence selftests (wrapper == stock, hybrid navigation)."""
    from .Tests.test_ornith_routed import (
        test_full_model_all_full,
        test_hybrid_structure,
        test_partial_rotary_router_shapes,
        test_wrapper_matches_stock_attention,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[selftest] device = {dev}", flush=True)
    test_wrapper_matches_stock_attention(dev)
    test_full_model_all_full(dev)
    test_hybrid_structure(dev)
    test_partial_rotary_router_shapes(dev)
    print("[selftest] ALL PASSED\n", flush=True)


# -------------------------------------------------------------------------------------------------
# Stage 2: teacher-forced quality comparison on the real 9B
# -------------------------------------------------------------------------------------------------
@torch.inference_mode()
def streamed_predictions(model, ids: torch.Tensor, block: int):
    """Feed the sequence in cache-backed blocks; yield ``(start, logits[1,blk,V])`` per block.

    Positions are driven **explicitly** (``position_ids`` / ``cache_position``): HGA does not grow
    the model's KV cache for the full-attention layers (their KV lives in the router's store), so
    the model's own position bookkeeping would stall — the manual positions keep RoPE correct.
    The hybrid cache is created by the model on the first (``pkv=None``) forward and threaded on,
    so the linear (GatedDeltaNet) layers keep their recurrent state across blocks.  Blocked +
    cached forward is mathematically identical to a single full forward for both layer types.
    """
    S = ids.shape[1]
    device = ids.device
    pkv = None
    for s in range(0, S, block):
        e = min(s + block, S)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=pkv, cache_position=cp,
                    position_ids=cp.unsqueeze(0), use_cache=True)
        pkv = out.past_key_values
        yield s, out.logits


def compare_on_ornith(args) -> None:
    """Teacher-forced greedy-match + perplexity: dense baseline vs group vs chunk routing."""
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"

    t0 = time.perf_counter()
    model, tok = _load_model(args.model, args.load, args.attn)
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s  "
          f"({gb(torch.cuda.memory_allocated()):.1f}GB allocated)", flush=True)

    ids = build_ids(tok, args.tokens, device)
    S = ids.shape[1]
    C = args.chunk_size
    tail_start = min((args.keep_first + args.keep_last) * C, S - 2)
    print(f"[data] context = {S} tokens, block = {args.block}; routing active for positions > {tail_start}\n"
          f"[cfg ] keep_first={args.keep_first} ({args.keep_first*C} sink tok), "
          f"keep_last={args.keep_last} ({args.keep_last*C} local tok), topk_chunks={args.topk} "
          f"=> {(args.keep_first+args.keep_last+args.topk)} active chunks / {S//C} total; "
          f"cache={args.cache_location}\n", flush=True)

    def run(name: str, ref_pred) -> Metric:
        m = Metric(S, tail_start, ref_pred)
        t = time.perf_counter()
        for s, logits in streamed_predictions(model, ids, args.block):
            m.add(s, logits, ids)
        torch.cuda.synchronize()
        print(f"[run ] {name} {time.perf_counter()-t:.1f}s", flush=True)
        torch.cuda.empty_cache()
        return m

    # --- baseline (reference): original dense attention, no patch ---
    restore_ornith_attention(model)
    base = run("baseline", None)
    base_pred = base.pred  # CPU long [S-1]

    # --- routed variants: group-level vs whole-chunk routing (both use_summaries=False) ---
    results = {}
    for name in ("group", "chunk"):
        kw = variant_kwargs(name, keep_first=args.keep_first, keep_last=args.keep_last,
                            topk=args.topk, chunk_size=C)
        n = replace_ornith_attention_with_router(model, cache_location=args.cache_location, **kw)
        results[name] = run(f"{name} ({n} full-attn layers)", base_pred)
        restore_ornith_attention(model)

    print("\nResults (greedy-match + perplexity/loss measured against baseline):")
    print(base.line("baseline"))
    for name in ("group", "chunk"):
        print(results[name].line(name))
    best = min(("group", "chunk"), key=lambda nm: results[nm].ppl)
    print(f"\n[best by loss @ {S} tok] {best}  "
          f"(ppl group={results['group'].ppl:.3f}, chunk={results['chunk'].ppl:.3f}; "
          f"baseline={base.ppl:.3f})")
    print(f"[mem ] peak allocated = {gb(torch.cuda.max_memory_allocated()):.1f}GB / "
          f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB")


# -------------------------------------------------------------------------------------------------
# Stage 2b: long-context needle-in-haystack (RAM cache; VRAM flat with context)
# -------------------------------------------------------------------------------------------------
def compare_ram(args) -> None:
    """Needle-in-haystack retrieval with a growing irrelevant prefix held in host RAM.

    Only routed chunks are pulled to VRAM, so peak VRAM should stay bounded as the context grows.
    The buried 'magic code' must appear in the routed answer (HIT); a large ``topk`` may be needed
    because the summary scoring ranks the mid-context needle conservatively (docs/ORNITH_HGA.md §10).
    """
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    t0 = time.perf_counter()
    model, tok = _load_model(args.model, args.load, args.attn)
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s\n"
          f"[cfg ] variant={args.variant} cache={args.cache_location}  keep_first={args.keep_first} "
          f"keep_last={args.keep_last} topk_chunks={args.topk}  prefill-block={args.block}\n",
          flush=True)

    for ctx in args.ctx_sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        prompt = _needle_prompt(tok, ctx, args.magic)
        kw = variant_kwargs(args.variant, keep_first=args.keep_first, keep_last=args.keep_last,
                            topk=args.topk, chunk_size=args.chunk_size)
        n = replace_ornith_attention_with_router(model, cache_location=args.cache_location, **kw)
        t = time.perf_counter()
        ans = _blocked_generate(model, tok, prompt, args.max_new, args.block, thinking=args.thinking)
        dt = time.perf_counter() - t
        peak = gb(torch.cuda.max_memory_allocated())
        restore_ornith_attention(model)
        hit = args.magic.replace(" ", "") in ans.replace(" ", "")
        print(f"[routed {args.variant} ~{ctx} tok]  ({n} layers, {dt:.0f}s, peak {peak:.2f}GB)\n"
              f"  -> {ans!r}\n  [{'HIT' if hit else 'MISS'}] magic={args.magic}\n", flush=True)


# -------------------------------------------------------------------------------------------------
# Stage 2c: decode speed — base (dense) vs patched (HGA)
# -------------------------------------------------------------------------------------------------
@torch.inference_mode()
def _prefill_then_decode(model, ids: torch.Tensor, block: int, max_new: int):
    """Blocked prefill (explicit positions) then greedy decode; returns (prefill_s, decode_tok_s).

    Identical driver for base and patched runs — the only difference is whether HGA is installed,
    so the decode tok/s numbers isolate the routing overhead.  Positions are driven explicitly
    because HGA freezes the model's KV-cache length on the full-attention layers.
    """
    device = ids.device
    S = ids.shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pkv = None
    out = None
    for s in range(0, S, block):
        e = min(s + block, S)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=pkv, cache_position=cp,
                    position_ids=cp.unsqueeze(0), use_cache=True)
        pkv = out.past_key_values
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    td = time.perf_counter()
    for i in range(max_new):
        p = S + i
        out = model(input_ids=cur, past_key_values=pkv, use_cache=True,
                    position_ids=torch.tensor([[p]], device=device),
                    cache_position=torch.tensor([p], device=device))
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    decode_tok_s = max_new / (time.perf_counter() - td)
    return prefill_s, decode_tok_s


def compare_speed(args) -> None:
    """Decode tok/s: base Ornith (dense) vs patched Ornith (HGA routed) at the same context.

    Sweeps ``--ctx-sizes`` (loading the model once) so the base-vs-patched crossover and the
    point where dense KV pressure shows up are visible; a base OOM at a long context is caught
    and reported (that is where HGA's flat-VRAM invariant becomes the only option).
    """
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    t0 = time.perf_counter()
    model, tok = _load_model(args.model, args.load, args.attn)
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s\n"
          f"[bench] prefill block={args.block}, decode {args.max_new} tokens; "
          f"variant={args.variant} cache={args.cache_location} topk_chunks={args.topk}\n", flush=True)

    def run(ids, patched: bool):
        restore_ornith_attention(model)
        if patched:
            kw = variant_kwargs(args.variant, keep_first=args.keep_first, keep_last=args.keep_last,
                                topk=args.topk, chunk_size=args.chunk_size)
            replace_ornith_attention_with_router(model, cache_location=args.cache_location, **kw)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            pf, dec = _prefill_then_decode(model, ids, args.block, args.max_new)
            peak = gb(torch.cuda.max_memory_allocated())
            res = (pf, dec, peak)
        except torch.cuda.OutOfMemoryError:
            res = None
        restore_ornith_attention(model)
        torch.cuda.empty_cache()
        return res

    for ctx in args.ctx_sizes:
        ids = build_ids(tok, ctx, device)
        S = ids.shape[1]
        print(f"[ctx {S} tok]", flush=True)
        base = run(ids, False)
        if base is None:
            print("  base  (dense)   OOM", flush=True)
        else:
            b_pf, b_dec, b_mem = base
            print(f"  base  (dense)   prefill {b_pf:6.1f}s   decode {b_dec:5.2f} tok/s   peak {b_mem:.2f}GB", flush=True)
        pat = run(ids, True)
        if pat is None:
            print("  patched (HGA)   OOM", flush=True)
        else:
            p_pf, p_dec, p_mem = pat
            ratio = f"{p_dec / base[1]:.2f}x" if base is not None else "n/a"
            print(f"  patched (HGA)   prefill {p_pf:6.1f}s   decode {p_dec:5.2f} tok/s   peak {p_mem:.2f}GB"
                  f"   (ratio {ratio})", flush=True)
        print(flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--load", default="4bit", choices=["4bit", "fp16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--selftest-only", action="store_true", help="offline selftests only, no model")
    ap.add_argument("--tokens", type=int, default=4096, help="teacher-forced context length")
    ap.add_argument("--block", type=int, default=256, help="blocked prefill slice (multiple of chunk)")
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--topk", type=int, default=16, help="topk_chunks for the routed variants")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--cache-location", default="ram", choices=["ram", "fs", "vram"])
    ap.add_argument("--variant", choices=["group", "chunk"], default="group",
                    help="routing granularity for --ram")
    ap.add_argument("--ram", action="store_true", help="long-context needle retrieval test")
    ap.add_argument("--bench", action="store_true", help="decode tok/s: base (dense) vs patched (HGA)")
    ap.add_argument("--ctx-sizes", type=int, nargs="+", default=[2048, 32768],
                    help="irrelevant-prefix context sizes for --ram")
    ap.add_argument("--max-new", type=int, default=128, help="decode tokens for the --ram answer")
    ap.add_argument("--magic", default="ZQ-7731-XK")
    ap.add_argument("--thinking", action="store_true", help="allow the model's chain-of-thought")
    args = ap.parse_args()

    run_selftests()
    if args.selftest_only:
        return
    if args.ram:
        compare_ram(args)
    elif args.bench:
        compare_speed(args)
    else:
        compare_on_ornith(args)


if __name__ == "__main__":
    main()
