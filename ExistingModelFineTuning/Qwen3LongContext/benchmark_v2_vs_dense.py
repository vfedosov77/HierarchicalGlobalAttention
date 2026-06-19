"""Benchmark v2_static HierarchicalGlobalAttention vs Qwen native dense attention.

Loads Qwen3-0.6B, times the *native dense* attention (prefill forward + greedy
decode) in compiled mode, then replaces every attention module with the
hierarchical ``HierarchicalGlobalAttention`` implementation and times the same workloads.

The hierarchical class is loaded by file path (default: the v2_static variant)
and monkey-patched onto the existing replacement machinery in
``replace_qwen_attention_finetune`` so that all the Qwen-specific glue
(weight copying, q/k RMSNorm, doubled head_dim, bias handling) is reused.

Run from the repository root:

    python -m ExistingModelFineTuning.Qwen3LongContext.benchmark_v2_vs_dense \
        --context-lens 2048,8192 --compile true --compile-mode default
"""

from __future__ import annotations

import argparse
import importlib.util
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

base = "/home/vladimir/torch_compile_tmp"

import os
os.environ["TMPDIR"] = f"{base}/tmp"
os.environ["TMP"] = f"{base}/tmp"
os.environ["TEMP"] = f"{base}/tmp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{base}/inductor"
os.environ["TRITON_CACHE_DIR"] = f"{base}/triton"
os.environ["CUDA_CACHE_PATH"] = f"{base}/cuda"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

for p in [
    os.environ["TMPDIR"],
    os.environ["TORCHINDUCTOR_CACHE_DIR"],
    os.environ["TRITON_CACHE_DIR"],
    os.environ["CUDA_CACHE_PATH"],
]:
    os.makedirs(p, exist_ok=True)

print("TMPDIR =", os.environ["TMPDIR"])
print("TORCHINDUCTOR_CACHE_DIR =", os.environ["TORCHINDUCTOR_CACHE_DIR"])
print("TRITON_CACHE_DIR =", os.environ["TRITON_CACHE_DIR"])

from ExistingModelFineTuning.Qwen3LongContext import replace_qwen_attention_finetune as R
import ExistingModelFineTuning.torch_inductor_patch as path
path.apply()

def log(msg: str) -> None:
    print(msg, flush=True)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_attention_class(path: str):
    """Import a HierarchicalGlobalAttention class from an arbitrary .py file by path."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"_ha_variant_{p.stem}", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cls_name = "HierarchicalGlobalAttention" if hasattr(mod, "HierarchicalGlobalAttention") else "GlobalAttention"
    if not hasattr(mod, cls_name):
        raise AttributeError(f"{p} has neither HierarchicalGlobalAttention nor GlobalAttention class")
    return getattr(mod, cls_name)


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def time_prefill(model: nn.Module, input_ids: torch.Tensor, warmup: int, repeats: int) -> Dict[str, float]:
    """Time a single full-sequence forward (use_cache=False) — the prefill cost."""
    for _ in range(warmup):
        _ = R.call_model_for_logits(model, input_ids, use_cache=False)
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = R.call_model_for_logits(model, input_ids, use_cache=False)
        sync()
        times.append(time.perf_counter() - t0)
        del out
    times.sort()
    seq = input_ids.shape[-1] * input_ids.shape[0]
    median = times[len(times) // 2]
    rep = {
        "prefill_median_s": median,
        "prefill_min_s": times[0],
        "prefill_tokens_per_s": seq / median,
    }
    if torch.cuda.is_available():
        rep["prefill_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return rep


@torch.no_grad()
def time_decode(model: nn.Module, tokenizer: Any, device: torch.device, prompt: str,
                new_tokens: int, warmup: int, repeats: int) -> Dict[str, float]:
    """Time greedy generation (prefill + per-token decode) via model.generate."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    gen_kwargs = dict(max_new_tokens=new_tokens, do_sample=False, use_cache=True,
                      pad_token_id=tokenizer.eos_token_id)
    for _ in range(warmup):
        _ = model.generate(**inputs, **gen_kwargs)
    sync()
    times: List[float] = []
    produced = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        sync()
        times.append(time.perf_counter() - t0)
        produced = int(out.shape[-1] - inputs["input_ids"].shape[-1])
        del out
    total = sum(times)
    return {
        "decode_seconds_mean": total / len(times),
        "decode_new_tokens": produced,
        "decode_tokens_per_s": (produced * len(times)) / max(total, 1e-9),
    }


@torch.no_grad()
def time_decode_long(model: nn.Module, device: torch.device, context_ids: torch.Tensor,
                     new_tokens: int, warmup: int, repeats: int,
                     tokenizer: Any) -> Dict[str, float]:
    """Time decode starting from a long pre-filled context.

    This is the regime where hierarchical attention should win: dense attention
    pays O(context) per generated token, while HA routes to a bounded number of
    chunks/groups.  We separately report the per-token decode speed by subtracting
    a measured prefill cost (single forward over the context) from the total.
    """
    gen_kwargs = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                      do_sample=False, use_cache=True,
                      pad_token_id=getattr(tokenizer, "eos_token_id", 0))
    inputs = {"input_ids": context_ids}
    for _ in range(warmup):
        _ = model.generate(**inputs, **gen_kwargs)
    sync()

    # Measure prefill cost alone (1 forward over the context) so we can isolate
    # the per-token decode rate from the one-off prefill.
    prefill_t: List[float] = []
    for _ in range(max(1, warmup)):
        _ = R.call_model_for_logits(model, context_ids, use_cache=False)
    sync()
    for _ in range(repeats):
        t0 = time.perf_counter()
        o = R.call_model_for_logits(model, context_ids, use_cache=False)
        sync()
        prefill_t.append(time.perf_counter() - t0)
        del o
    prefill_t.sort()
    prefill_med = prefill_t[len(prefill_t) // 2]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    times: List[float] = []
    produced = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        sync()
        times.append(time.perf_counter() - t0)
        produced = int(out.shape[-1] - context_ids.shape[-1])
        del out
    times.sort()
    total_med = times[len(times) // 2]
    decode_only = max(total_med - prefill_med, 1e-9)
    rep = {
        "ctx_len": int(context_ids.shape[-1]),
        "gen_total_median_s": total_med,
        "prefill_median_s": prefill_med,
        "decode_only_median_s": decode_only,
        "decode_new_tokens": produced,
        "decode_tokens_per_s": produced / decode_only,
        "gen_tokens_per_s": produced / total_med,
    }
    if torch.cuda.is_available():
        rep["decode_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return rep


@torch.no_grad()
def teacher_forcing_loss(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Mean next-token CE loss for one full-sequence (use_cache=False) forward."""
    out = R.call_model_for_logits(model, input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean")
    return float(loss.detach().cpu())


def maybe_compile(model: nn.Module, args: argparse.Namespace, tag: str):
    if not args.compile:
        return None
    log(f"[compile] {tag}: torch.compile(mode={args.compile_mode})")
    orig = model.forward
    model.forward = torch.compile(orig, mode=args.compile_mode,
                                  fullgraph=args.compile_fullgraph, dynamic=args.compile_dynamic)
    return orig


def uncompile(model: nn.Module, orig) -> None:
    if orig is not None:
        model.forward = orig
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--torch-dtype", default="float32", choices=["float32", "bf16", "bfloat16", "fp16", "float16"],
                   help="bf16 is needed to fit big models (e.g. 8B) at 8K on a 32GB GPU")
    p.add_argument("--attn-impl", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ha-file", default=str(Path(__file__).resolve().parents[1]
                                            / "TestModel40M/ha_variants/v2_static.py"))
    p.add_argument("--context-lens", default="4,8000",
                   help="comma-separated prefill context lengths to benchmark")
    p.add_argument("--prefill-warmup", type=int, default=2)
    p.add_argument("--prefill-repeats", type=int, default=5)
    p.add_argument("--compile", type=str2bool, default=False)
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--compile-fullgraph", action="store_true")
    p.add_argument("--compile-dynamic", type=str2bool, default=False)
    p.add_argument("--dynamo-cache-size-limit", type=int, default=256)
    # decode benchmark
    p.add_argument("--decode", type=str2bool, default=True)
    p.add_argument("--decode-prompt", default="Give me a concise explanation of rotary position embeddings.")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--decode-warmup", type=int, default=1)
    p.add_argument("--decode-repeats", type=int, default=3)
    p.add_argument("--decode-context-len", type=int, default=8192,
                   help="If >0, time decode from a random prompt of this length (the regime "
                        "where HA beats dense). 0 falls back to the short text prompt.")
    # loss-equality check (optimized HA full-forward loss vs reference HA math)
    p.add_argument("--check-loss", type=str2bool, default=True)
    p.add_argument("--loss-len", type=int, default=2048,
                   help="sequence length for the teacher-forcing loss-equality check")
    p.add_argument("--ref-ha-file", default=str(Path(__file__).resolve().parents[1]
                                                 / "TestModel40M/ha_variants/v2_static.py"),
                   help="reference HA implementation whose full-forward loss the "
                        "optimized variant must reproduce (same architecture).")
    # HierarchicalGlobalAttention config knobs forwarded to the constructor via the existing glue
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--topk-chunks", type=int, default=20)
    p.add_argument("--topk-groups", type=int, default=32)
    p.add_argument("--decode-max-seq", type=int, default=8192)
    return p.parse_args()


def build_replace_args(args: argparse.Namespace) -> argparse.Namespace:
    """Construct the argparse.Namespace that replace_attention_modules expects."""
    import json
    extra = {
        "chunk_size": args.chunk_size,
        "group_size": args.group_size,
        "topk_chunks": args.topk_chunks,
        "topk_groups": args.topk_groups,
        "qk_norm": True,
        "decode_max_seq": args.decode_max_seq,
    }
    return SimpleNamespace(
        bias_mode="requested",
        use_bias_q=True, use_bias_k=True, use_bias_v=True, use_bias_o=False,
        dropout=0.0,
        global_attention_extra_kwargs=json.dumps(extra),
        strict_forward_signature=False,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16}[args.torch_dtype]
    log(f"[env] torch_dtype={dtype}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        try:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, args.dynamo_cache_size_limit)
        except Exception:
            pass

    context_lens = [int(x) for x in args.context_lens.split(",") if x.strip()]
    log(f"[env] device={dev} dtype={dtype} model={args.model_name} ha_file={args.ha_file}")
    log(f"[env] compile={args.compile} mode={args.compile_mode} context_lens={context_lens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = int(getattr(tokenizer, "vocab_size", 151936))

    # Deterministic random token ids for prefill timing (no dataset download needed).
    gen = torch.Generator().manual_seed(args.seed)
    prefill_inputs = {
        S: torch.randint(0, vocab, (1, S), generator=gen).to(dev) for S in context_lens
    }
    # Long decode context + loss-check input (shared across dense/HA for a fair compare).
    decode_ctx = None
    if args.decode and args.decode_context_len > 0:
        decode_ctx = torch.randint(0, vocab, (1, args.decode_context_len), generator=gen).to(dev)
    loss_ids = torch.randint(0, vocab, (1, args.loss_len), generator=gen).to(dev) if args.check_loss else None

    results: Dict[str, Any] = {"dense": {}, "ha": {}}

    # ---------------- Dense (native) ----------------
    log("\n========== NATIVE DENSE ATTENTION ==========")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, attn_implementation=args.attn_impl, trust_remote_code=True
    ).to(dev)
    model.config.use_cache = True
    model.eval()
    orig = None# maybe_compile(model, args, "dense")
    for S in context_lens:
        rep = time_prefill(model, prefill_inputs[S], args.prefill_warmup, args.prefill_repeats)
        results["dense"][f"prefill_{S}"] = rep
        log(f"[dense] ctx={S}: {rep}")
    if args.decode:
        if decode_ctx is not None:
            rep = time_decode_long(model, dev, decode_ctx, args.decode_tokens,
                                   args.decode_warmup, args.decode_repeats, tokenizer)
        else:
            rep = time_decode(model, tokenizer, dev, args.decode_prompt,
                              args.decode_tokens, args.decode_warmup, args.decode_repeats)
        results["dense"]["decode"] = rep
        log(f"[dense] decode: {rep}")
    if loss_ids is not None:
        results["dense"]["loss"] = teacher_forcing_loss(model, loss_ids)
        log(f"[dense] teacher-forcing loss (len={args.loss_len}): {results['dense']['loss']:.6f}")
    uncompile(model, orig)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- HierarchicalGlobalAttention ----------------
    log("\n========== HIERARCHICAL GLOBAL ATTENTION (v2_static) ==========")
    R.HierarchicalGlobalAttention = load_attention_class(args.ha_file)  # monkeypatch the glue
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, attn_implementation=args.attn_impl, trust_remote_code=True
    ).to(dev)
    model.config.use_cache = True
    n = R.replace_attention_modules(model, build_replace_args(args))
    log(f"[replace] replaced {n} attention modules with HierarchicalGlobalAttention")
    model.eval()
    orig = maybe_compile(model, args, "ha")
    for S in context_lens:
        rep = time_prefill(model, prefill_inputs[S], args.prefill_warmup, args.prefill_repeats)
        results["ha"][f"prefill_{S}"] = rep
        log(f"[ha] ctx={S}: {rep}")
    if args.decode:
        if decode_ctx is not None:
            rep = time_decode_long(model, dev, decode_ctx, args.decode_tokens,
                                   args.decode_warmup, args.decode_repeats, tokenizer)
        else:
            rep = time_decode(model, tokenizer, dev, args.decode_prompt,
                              args.decode_tokens, args.decode_warmup, args.decode_repeats)
        results["ha"]["decode"] = rep
        log(f"[ha] decode: {rep}")
    if loss_ids is not None:
        results["ha"]["loss"] = teacher_forcing_loss(model, loss_ids)
        log(f"[ha] teacher-forcing loss (len={args.loss_len}): {results['ha']['loss']:.6f}")
    uncompile(model, orig)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- Reference HA (loss-equality reference) ----------------
    if loss_ids is not None and Path(args.ref_ha_file).resolve() != Path(args.ha_file).resolve():
        log("\n========== REFERENCE HA (loss reference) ==========")
        R.HierarchicalGlobalAttention = load_attention_class(args.ref_ha_file)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype, attn_implementation=args.attn_impl, trust_remote_code=True
        ).to(dev)
        model.config.use_cache = True
        R.replace_attention_modules(model, build_replace_args(args))
        model.eval()
        results["ref_ha_loss"] = teacher_forcing_loss(model, loss_ids)
        log(f"[ref_ha] teacher-forcing loss (len={args.loss_len}): {results['ref_ha_loss']:.6f}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- Summary ----------------
    log("\n========== SUMMARY (HA vs dense) ==========")
    for S in context_lens:
        d = results["dense"][f"prefill_{S}"]["prefill_median_s"]
        h = results["ha"][f"prefill_{S}"]["prefill_median_s"]
        log(f"  prefill ctx={S:6d}: dense={d*1e3:8.2f} ms  ha={h*1e3:8.2f} ms  "
            f"speedup={d/h:5.2f}x")
    if args.decode and "decode" in results["dense"] and "decode" in results["ha"]:
        dd, hh = results["dense"]["decode"], results["ha"]["decode"]
        d = dd["decode_tokens_per_s"]
        h = hh["decode_tokens_per_s"]
        ctx = dd.get("ctx_len", "n/a")
        log(f"  decode tok/s (ctx={ctx}): dense={d:8.2f}  ha={h:8.2f}  speedup={h/d:5.2f}x")
        if "gen_tokens_per_s" in dd:
            log(f"  gen   tok/s (incl prefill): dense={dd['gen_tokens_per_s']:8.2f}  "
                f"ha={hh['gen_tokens_per_s']:8.2f}  speedup={hh['gen_tokens_per_s']/dd['gen_tokens_per_s']:5.2f}x")
    if "loss" in results["ha"]:
        hl = results["ha"]["loss"]
        line = f"  loss(len={args.loss_len}): ha={hl:.6f}"
        if "dense" in results and "loss" in results["dense"]:
            line += f"  dense={results['dense']['loss']:.6f}"
        if "ref_ha_loss" in results:
            diff = abs(hl - results["ref_ha_loss"])
            verdict = "OK (matches reference)" if diff < 1e-2 else "MISMATCH"
            line += f"  ref_ha={results['ref_ha_loss']:.6f}  |diff|={diff:.2e}  -> {verdict}"
        log(line)


if __name__ == "__main__":
    main()
