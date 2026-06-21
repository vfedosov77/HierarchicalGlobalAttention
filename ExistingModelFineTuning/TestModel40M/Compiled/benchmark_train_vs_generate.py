#!/usr/bin/env python3
"""
Benchmark training vs. generation (big-block preprocessing) speed.

This compares, for the *same* SmallLM architecture used in
``finetune_small_model_bf16.py``:

  * "HA"    : the Hierarchical Global Attention (Exact-Q fused) module
              (``use_global=True``), and
  * "Dense" : regular RoPE dense attention, obtained by running the very
              same module with ``use_global=False`` (it then falls back to a
              causal ``F.scaled_dot_product_attention`` over RoPE q/k).

Two regimes are timed per model:

  * "train"    : forward + backward over the whole block (model.train(),
                 gradients enabled) -- this is the fine-tuning step cost.
  * "generate" : forward-only prefill of the whole block under
                 ``torch.inference_mode`` (model.eval()).  These attention
                 modules have no KV cache; decoding is done by running on the
                 full available prefix, so "big-block preprocessing" is exactly
                 a single forward over the block.

Both models are compiled with ``torch.compile`` exactly like the fine-tune
script.  A Markdown report is written next to this script.

Example:
    python benchmark_train_vs_generate.py \
        --seq-lens 2048 4096 8192 \
        --batch-size 1 --iters 10 --warmup 3 --precision fp32
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import platform
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Backport the Inductor codegen fix used by the training scripts so that
# torch.compile of the fused attention does not crash on torch 2.10/2.11.
try:
    import torch_inductor_patch

    torch_inductor_patch.apply()
except Exception as exc:  # pragma: no cover - patch is best-effort
    print(f"[warn] torch_inductor_patch not applied: {exc}")

# Import GlobalAttention from the same place the fine-tune script uses, with a
# fallback to the flat module layout in this directory.
_IMPORT_ERR = None
GlobalAttention = None
for _modpath in (
    "Compiled.HierarchicalGlobalAttentionFusedExactQ",
    "HierarchicalGlobalAttentionFusedExactQ",
):
    try:
        module = __import__(_modpath, fromlist=["GlobalAttention"])
        GlobalAttention = getattr(module, "GlobalAttention")
        break
    except Exception as exc:  # pragma: no cover - depends on local layout
        _IMPORT_ERR = exc
if GlobalAttention is None:
    raise RuntimeError(f"Could not import GlobalAttention: {_IMPORT_ERR}")


# -----------------------------------------------------------------------------
# Model / data defaults (mirrored from finetune_small_model_bf16.py)
# -----------------------------------------------------------------------------
HIDDEN_DIM = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
VOCAB_SIZE = 50257  # gpt2 vocab

CHUNK_SIZE = 64
GROUP_SIZE = 16
TOPK_CHUNKS = 20
TOPK_GROUPS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Architecture (identical to the fine-tune script)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 / rms) * self.weight.float()
        return out.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn_module: nn.Module, dff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_result = self.self_attn(self.norm1(x))
        h = attn_result[0] if isinstance(attn_result, (tuple, list)) else attn_result
        x = x + self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)
        return x


class SmallLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        kv_heads: int,
        num_layers: int,
        dff: int,
        attn_factory,
        dropout: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim, attn_factory(layer_idx=i), dff, dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1))
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(use_global: bool, dropout: float = 0.0) -> nn.Module:
    def attn_factory(layer_idx: int):
        return GlobalAttention(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=dropout,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            use_global=use_global,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=TOPK_CHUNKS,
            topk_groups=TOPK_GROUPS,
            return_router_stats=False,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            q_norm=None,
            k_norm=None,
        )

    return SmallLM(
        VOCAB_SIZE,
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        NUM_LAYERS,
        DFF,
        attn_factory,
        dropout=dropout,
        ignore_index=-100,
    )


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------
def autocast_context(enabled: bool, dtype: Optional[torch.dtype]):
    if enabled and DEVICE == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return contextlib.nullcontext()


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def make_batch(batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(1234 + seq_len)
    ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len + 1), generator=g)
    inputs = ids[:, :-1].contiguous().to(DEVICE)
    targets = ids[:, 1:].contiguous().to(DEVICE)
    return inputs, targets


def time_train(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    iters: int,
    warmup: int,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> float:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None

    def one_step():
        for p in params:
            p.grad = None
        with autocast_context(amp_enabled, amp_dtype):
            _, loss = model(inputs, labels=targets)
        loss.backward()

    for _ in range(warmup):
        one_step()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        one_step()
    _sync()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def time_generate(
    model: nn.Module,
    inputs: torch.Tensor,
    iters: int,
    warmup: int,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> float:
    model.eval()

    def one_step():
        with torch.inference_mode(), autocast_context(amp_enabled, amp_dtype):
            model(inputs)

    for _ in range(warmup):
        one_step()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        one_step()
    _sync()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def detect_attn_path(model: nn.Module) -> str:
    raw = getattr(model, "_orig_mod", model)
    try:
        attn = raw.layers[0].self_attn
        return getattr(attn, "_last_path", "n/a")
    except Exception:
        return "n/a"


# -----------------------------------------------------------------------------
# Benchmark driver
# -----------------------------------------------------------------------------
def run_benchmark(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    amp_enabled = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None

    if amp_enabled and not (DEVICE == "cuda" and torch.cuda.is_bf16_supported()):
        print("[warn] bf16 requested but not supported; falling back to fp32.")
        amp_enabled, amp_dtype = False, None

    configs = [
        ("HA (use_global=True)", True),
        ("Dense RoPE (use_global=False)", False),
    ]

    results: List[Dict] = []
    for label, use_global in configs:
        print(f"\n=== Building {label} ===")
        model = build_model(use_global, dropout=args.dropout).to(DEVICE)
        n_params = model.count_parameters()
        if args.compile:
            model = torch.compile(model)
            print("  compiled with torch.compile")

        for seq_len in args.seq_lens:
            inputs, targets = make_batch(args.batch_size, seq_len)
            tokens = args.batch_size * seq_len

            print(f"  seq_len={seq_len}: timing train ...", flush=True)
            try:
                t_train = time_train(
                    model, inputs, targets, args.iters, args.warmup, amp_enabled, amp_dtype
                )
                train_path = detect_attn_path(model)
            except Exception as exc:
                print(f"    train failed: {exc}")
                t_train, train_path = float("nan"), "error"

            print(f"  seq_len={seq_len}: timing generate ...", flush=True)
            try:
                t_gen = time_generate(
                    model, inputs, args.iters, args.warmup, amp_enabled, amp_dtype
                )
                gen_path = detect_attn_path(model)
            except Exception as exc:
                print(f"    generate failed: {exc}")
                t_gen, gen_path = float("nan"), "error"

            train_tps = tokens / t_train if t_train == t_train and t_train > 0 else float("nan")
            gen_tps = tokens / t_gen if t_gen == t_gen and t_gen > 0 else float("nan")
            # How many times faster a forward-only generate step is vs a full train step.
            gen_speedup = t_train / t_gen if (t_gen == t_gen and t_gen > 0) else float("nan")

            results.append(
                {
                    "model": label,
                    "use_global": use_global,
                    "params": n_params,
                    "seq_len": seq_len,
                    "batch": args.batch_size,
                    "tokens": tokens,
                    "train_ms": t_train * 1e3,
                    "gen_ms": t_gen * 1e3,
                    "train_tps": train_tps,
                    "gen_tps": gen_tps,
                    "gen_vs_train_speedup": gen_speedup,
                    "attn_path": gen_path if gen_path != "n/a" else train_path,
                }
            )
            print(
                f"    train={t_train*1e3:8.2f} ms ({train_tps:,.0f} tok/s)  "
                f"generate={t_gen*1e3:8.2f} ms ({gen_tps:,.0f} tok/s)  "
                f"generate is {gen_speedup:.2f}x faster than train"
            )

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "precision": "bf16" if amp_enabled else "fp32",
        "compiled": bool(args.compile),
        "batch_size": args.batch_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "seq_lens": list(args.seq_lens),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arch": {
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "kv_heads": KV_HEADS,
            "num_layers": NUM_LAYERS,
            "dff": DFF,
            "chunk_size": CHUNK_SIZE,
            "group_size": GROUP_SIZE,
            "topk_chunks": TOPK_CHUNKS,
            "topk_groups": TOPK_GROUPS,
        },
    }
    return results, meta


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(results: List[Dict], meta: Dict, path: str) -> None:
    lines: List[str] = []
    lines.append("# Training vs. Generation (big-block preprocessing) speed benchmark\n")
    lines.append(f"_Generated: {meta['timestamp']}_\n")

    lines.append("## Environment\n")
    lines.append(f"- Device: `{meta['device']}`")
    lines.append(f"- PyTorch: `{meta['torch']}` | Python: `{meta['python']}`")
    lines.append(f"- Precision: `{meta['precision']}` | torch.compile: `{meta['compiled']}`")
    lines.append(
        f"- Batch size: `{meta['batch_size']}` | timed iters: `{meta['iters']}` | "
        f"warmup iters: `{meta['warmup']}`"
    )
    a = meta["arch"]
    lines.append(
        f"- Architecture: hidden={a['hidden_dim']}, heads={a['num_heads']}, "
        f"kv_heads={a['kv_heads']}, layers={a['num_layers']}, dff={a['dff']}, "
        f"chunk={a['chunk_size']}, group={a['group_size']}, "
        f"topk_chunks={a['topk_chunks']}, topk_groups={a['topk_groups']}\n"
    )

    lines.append("## What is measured\n")
    lines.append(
        "- **train**: one forward + backward over the whole block "
        "(`model.train()`, gradients on). This is the per-step fine-tuning cost.\n"
        "- **generate (big-block preprocessing)**: forward-only over the whole "
        "block under `torch.inference_mode()` (`model.eval()`). These attention "
        "modules keep no KV cache, so prefilling a prompt is a single forward "
        "over the full block.\n"
        "- **Dense RoPE** is the same module run with `use_global=False`, which "
        "falls back to a causal `scaled_dot_product_attention` over RoPE q/k.\n"
    )

    lines.append("## Results\n")
    header = (
        "| Model | seq_len | tokens | train ms | train tok/s | "
        "generate ms | generate tok/s | train/generate |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            f"| {r['model']} | {r['seq_len']} | {r['tokens']:,} | "
            f"{r['train_ms']:.2f} | {r['train_tps']:,.0f} | "
            f"{r['gen_ms']:.2f} | {r['gen_tps']:,.0f} | "
            f"{r['gen_vs_train_speedup']:.2f}x |"
        )
    lines.append("")

    # HA vs Dense comparison per seq_len / regime.
    lines.append("## HA vs. Dense RoPE (same regime)\n")
    lines.append("| seq_len | regime | HA ms | Dense ms | HA/Dense |")
    lines.append("|---:|---|---:|---:|---:|")
    by_key: Dict[Tuple[int, bool], Dict] = {(r["seq_len"], r["use_global"]): r for r in results}
    seq_lens = sorted({r["seq_len"] for r in results})
    for seq_len in seq_lens:
        ha = by_key.get((seq_len, True))
        dense = by_key.get((seq_len, False))
        if not ha or not dense:
            continue
        for regime, key in (("train", "train_ms"), ("generate", "gen_ms")):
            ha_ms = ha[key]
            dn_ms = dense[key]
            ratio = ha_ms / dn_ms if dn_ms and dn_ms == dn_ms else float("nan")
            lines.append(
                f"| {seq_len} | {regime} | {ha_ms:.2f} | {dn_ms:.2f} | {ratio:.2f}x |"
            )
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- `train/generate` is how many times faster a forward-only generate step "
        "is than a full training step; it should roughly reflect that a training "
        "step adds a backward pass on top of the forward.\n"
        "- `HA/Dense` < 1 means hierarchical attention is faster than dense.\n"
        "- All numbers are wall-clock averages over the timed iterations after "
        "warmup; `torch.cuda.synchronize()` brackets each timed region.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {path}")


# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                   help="fp32 keeps the HA fused Triton path active; bf16 uses autocast.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Compile both models with torch.compile, like the fine-tune script.")
    p.add_argument("--report", type=str, default=os.path.join(SCRIPT_DIR, "benchmark_report.md"))
    return p.parse_args()


def main() -> None:
    if DEVICE != "cuda":
        print("[warn] CUDA not available; the fused HA path requires CUDA fp32. "
              "Results will use the reference/fallback path.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    print("Benchmark configuration:")
    print(f"  seq_lens={args.seq_lens} batch_size={args.batch_size} "
          f"iters={args.iters} warmup={args.warmup} precision={args.precision} "
          f"compile={args.compile}")

    results, meta = run_benchmark(args)
    write_report(results, meta, args.report)


if __name__ == "__main__":
    main()
