"""Benchmark one QLoRA fine-tuning **micro-step** with routed (patched) attention vs the dense
base attention, on the *same* model weights.

The routed surgery (`replace_qwen_attention_with_router`) reuses the original q/k/v/o projections
*by reference* and only changes *what each query attends to*.  To isolate the pure cost of routing
we build the model once (routed) and flip attention off with the fine-tune script's
``dense_attention`` context manager — both regimes run the identical 4-bit base + LoRA adapters,
so any time/memory difference is attributable to routing alone, not to different weights.

Per regime we time a real training step (forward + backward + optimizer step) and record peak GPU
memory, then print a side-by-side table and write a TSV.

Run from the repo root::

    python -m ExistingModelFineTuning.Qwen3LongContext.benchmark_finetune_routed_vs_dense --selfcheck
    python -m ExistingModelFineTuning.Qwen3LongContext.benchmark_finetune_routed_vs_dense --seq-len 1024 --repeats 10

ponytail: bf16 autocast with a plain ``optimizer.step()`` (no GradScaler).  Comparing both regimes
under the same path keeps it apples-to-apples; the fp16/scaler training path is intentionally not
benchmarked here.  Upgrade path: add ``--fp16`` and wrap both phases in a shared GradScaler.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import torch

# Reuse the fine-tune harness so the model, routing toggle and seq-len rules stay the single
# source of truth (works from the repo root via -m or as a direct script).
try:
    from .finetune_qwen06b_qlora_routed import (  # type: ignore
        build_blocks,
        build_model,
        chunked_causal_lm_loss,
        dense_attention,
        read_token_stats,
        reset_routers,
        routing_defaults,
        set_token_stats,
    )
    from .qwen_routed_attention import QwenRoutedAttention  # type: ignore
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from finetune_qwen06b_qlora_routed import (  # type: ignore
        build_blocks,
        build_model,
        chunked_causal_lm_loss,
        dense_attention,
        read_token_stats,
        reset_routers,
        routing_defaults,
        set_token_stats,
    )
    from qwen_routed_attention import QwenRoutedAttention  # type: ignore

import bitsandbytes as bnb
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_step(
    model: Any,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    compute_dtype: torch.dtype,
    *,
    routed: bool,
    warmup: int,
    repeats: int,
    loss_chunk_size: int,
) -> Dict[str, float]:
    """Median forward/backward/optimizer timings (ms) and peak GPU memory (GB) for one regime.

    Phases are bracketed with ``torch.cuda.synchronize`` so each measured slice is the real GPU
    cost.  Warmup runs full steps first so the paged AdamW optimizer state is already allocated
    before timing — otherwise the first step's lazy state alloc would inflate the optimizer slice.
    """
    fwd_ms: List[float] = []
    bwd_ms: List[float] = []
    opt_ms: List[float] = []
    total_ms: List[float] = []
    peak_gb = 0.0

    def _one_step() -> torch.Tensor:
        if routed:
            reset_routers(model)  # the router KV store is stateful; start each step clean
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=compute_dtype):
            loss = chunked_causal_lm_loss(model, batch, batch, loss_chunk_size, train=True)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during benchmark step")
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(warmup):
        _one_step()
    _sync()

    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if routed:
            reset_routers(model)
        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=compute_dtype):
            loss = chunked_causal_lm_loss(model, batch, batch, loss_chunk_size, train=True)
        _sync()
        t1 = time.perf_counter()

        loss.backward()
        _sync()
        t2 = time.perf_counter()

        optimizer.step()
        _sync()
        t3 = time.perf_counter()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during benchmark step")
        fwd_ms.append((t1 - t0) * 1e3)
        bwd_ms.append((t2 - t1) * 1e3)
        opt_ms.append((t3 - t2) * 1e3)
        total_ms.append((t3 - t0) * 1e3)
        if torch.cuda.is_available():
            peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1024**3)

    def _median(xs: List[float]) -> float:
        return sorted(xs)[len(xs) // 2]

    total_med = _median(total_ms)
    tokens = batch.shape[0] * batch.shape[1]
    return {
        "forward_ms": _median(fwd_ms),
        "backward_ms": _median(bwd_ms),
        "opt_ms": _median(opt_ms),
        "total_ms": total_med,
        "tokens_per_s": tokens / (total_med / 1e3),
        "peak_gb": peak_gb,
    }


def _print_table(seq_len: int, routed: Dict[str, float], base: Dict[str, float],
                 tok_stats: Dict[str, float] | None = None) -> None:
    rows = [
        ("forward_ms", "Forward (ms)"),
        ("backward_ms", "Backward (ms)"),
        ("opt_ms", "Optimizer (ms)"),
        ("total_ms", "Total step (ms)"),
        ("tokens_per_s", "Throughput (tok/s)"),
        ("peak_gb", "Peak GPU mem (GB)"),
    ]
    print(f"\n=== Fine-tune micro-step: routed vs dense base (seq_len={seq_len}) ===")
    print(f"{'Metric':<20}{'Routed':>14}{'Base (dense)':>16}{'Routed/Base':>14}")
    for key, label in rows:
        r, b = routed[key], base[key]
        ratio = (r / b) if b else float("nan")
        print(f"{label:<20}{r:>14.3f}{b:>16.3f}{ratio:>14.3f}")
    if tok_stats is not None:
        # Routed column = real KV tokens a query attends; Base column = dense causal budget;
        # Routed/Base = density (1 - saving).
        print(f"{'Attended KV/query':<20}{tok_stats['attended']:>14.3f}"
              f"{tok_stats['dense']:>16.3f}{tok_stats['density']:>14.3f}")


def _write_tsv(path: str, seq_len: int, routed: Dict[str, float], base: Dict[str, float]) -> None:
    keys = ["forward_ms", "backward_ms", "opt_ms", "total_ms", "tokens_per_s", "peak_gb"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("regime\tseq_len\t" + "\t".join(keys) + "\n")
        for regime, m in (("routed", routed), ("base", base)):
            fh.write(f"{regime}\t{seq_len}\t" + "\t".join(f"{m[k]:.6f}" for k in keys) + "\n")
    print(f"[tsv] wrote {path}")


def run(args) -> Dict[str, Dict[str, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA benchmarking.")
    knobs = routing_defaults()
    chunk_size = knobs["chunk_size"]
    window = (knobs["keep_first"] + knobs["keep_last"]) * chunk_size
    if args.seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be a multiple of chunk_size ({chunk_size}).")
    if args.seq_len <= window:
        raise ValueError(
            f"seq_len ({args.seq_len}) must exceed the resident windows ({window}) so routing engages."
        )
    if not (os.path.isfile(args.data_path) and os.path.getsize(args.data_path) > 0):
        raise FileNotFoundError(f"Training text not found or empty: {args.data_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    compute_dtype = torch.bfloat16

    print(f"[setup] model={args.model} seq_len={args.seq_len} batch={args.batch_size} "
          f"device={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    block = build_blocks(text, tokenizer, args.seq_len)[: args.batch_size].to(device)
    if block.shape[0] < args.batch_size:
        raise ValueError(
            f"Text yields only {block.shape[0]} block(s); need {args.batch_size} for the batch."
        )

    model, n = build_model(args, compute_dtype)
    print(f"[model] wrapped {n} attention layers with the router")
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr, weight_decay=0.0)

    # Routed regime: the model is already wrapped after build_model.  Collect the routed
    # attended-KV budget over the timed routed step so the table reports how sparse it ran.
    set_token_stats(model, True)
    routed = time_step(model, block, optimizer, compute_dtype,
                       routed=True, warmup=args.warmup, repeats=args.repeats,
                       loss_chunk_size=args.loss_chunk_size)
    tok_stats = read_token_stats(model)
    set_token_stats(model, False)
    torch.cuda.empty_cache()

    # Dense base: same weights, routing toggled off; CM restores the router afterward.
    with dense_attention(model, knobs):
        base = time_step(model, block, optimizer, compute_dtype,
                         routed=False, warmup=args.warmup, repeats=args.repeats,
                         loss_chunk_size=args.loss_chunk_size)
    torch.cuda.empty_cache()

    _print_table(args.seq_len, routed, base, tok_stats)
    if args.out_tsv:
        _write_tsv(args.out_tsv, args.seq_len, routed, base)
    return {"routed": routed, "base": base, "_model": model}


def selfcheck(args) -> None:
    """Fast assert-based check: both regimes produce finite timings + real peak memory, and the
    router is re-wrapped after the dense-attention context (toggle round-trips cleanly)."""
    args.seq_len = 512  # 8 chunks > 4-chunk window -> routing engages, fast
    args.warmup = 1
    args.repeats = 2
    out = run(args)
    routed, base, model = out["routed"], out["base"], out["_model"]
    for name, m in (("routed", routed), ("base", base)):
        for k in ("forward_ms", "backward_ms", "opt_ms", "total_ms", "tokens_per_s"):
            assert m[k] > 0 and m[k] == m[k], f"{name}.{k} not positive/finite: {m[k]}"
        assert m["peak_gb"] > 0, f"{name}.peak_gb not measured: {m['peak_gb']}"
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    assert all(
        isinstance(layer.self_attn, QwenRoutedAttention) for layer in base_model.model.layers
    ), "router was not restored after the dense-attention context"
    print("[selfcheck] OK: routed & base timings finite, peak memory measured, router restored")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data-path", default=os.path.join(_REPO_ROOT, "TrainData", "The-Master-and-Margarita.txt"))
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--loss-chunk-size", type=int, default=512, help="tokens per lm_head+CE slice; smaller = less VRAM in the vocab dim")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-tsv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_finetune_routed_vs_dense.tsv"))
    # LoRA (consumed by build_model)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--selfcheck", action="store_true", help="run the fast assert-based self-check and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selfcheck:
        selfcheck(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
