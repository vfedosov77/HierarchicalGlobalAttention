#!/usr/bin/env python3
"""Block-wise long-context LoRA fine-tuning of Qwen3 with HGA routed attention.

What this does
--------------
* Replaces every ``self_attn`` with the router-backed sparse attention (no SDPA fallback) and
  trains **LoRA only** (attn ± mlp).
* Trains on **fixed-length sequences** (default 16K tokens) drawn from **three mixed sources**
  (the novel + a long-context corpus + a dialog corpus), batched ``--batch-size`` sequences at a
  time (default 3 → 48K tokens / step).
* Processes each sequence **block by block** (``keep_last + --block-extra`` chunks per block).
  Previous blocks stay in the **RAM/VRAM KV cache** (detached); only the hot window carries
  gradients.  Each block does its own ``backward`` and then the cache graph is **detached**
  (truncated BPTT), so activation memory is bounded by one block regardless of sequence length.
* Per-sequence cache isolation is automatic: every sequence is its own batch row in the store, so
  chunks never mix across sequences.

Model is swappable: ``Qwen/Qwen3-0.6B`` (default), ``Qwen/Qwen3-8B``, ``Qwen/Qwen3-14B``, and
``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`` (MoE/FP8, for a 32GB RTX 5090) are all supported — the
attention surgery and LoRA targets adapt to dense vs. MoE automatically.

Example (real run, RTX 5090):
    python train.py --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --device cuda \
        --seq-len 16384 --batch-size 3 --keep-last 8 --cold vram --lr 3e-5
Example (offline CPU smoke):
    python train.py --device cpu --sources mm --seq-len 512 --batch-size 3 \
        --keep-last 3 --max-steps 2
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLC = os.path.dirname(_HERE)
_EFT = os.path.dirname(_QLC)
_ROOT = os.path.dirname(_EFT)
for _p in (_HERE, _QLC, _EFT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from routed_attention import (  # noqa: E402
    patch_qwen3_with_router, hybrid_grad_store_factory, vram_hybrid_store_factory,
)
from streaming import _segment_plan  # noqa: E402
from data import build_sources, MixedBatcher  # noqa: E402

MM_FILE = os.path.join(_ROOT, "The-Master-and-Margarita.txt")
PARQUET_GLOB = os.path.join(_ROOT, "fineweb_sample", "sample", "10BT", "*.parquet")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build_lora_targets(config, override: str | None) -> list[str]:
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    is_moe = bool(getattr(config, "num_experts", 0)) or \
        any("moe" in a.lower() for a in (getattr(config, "architectures", None) or []))
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # MoE: attention only (targeting every expert's projections would explode the LoRA count).
    return attn if is_moe else attn + ["gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    # sequence / routing
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--block-extra", type=int, default=1,
                    help="block size in chunks = keep_last + block_extra")
    ap.add_argument("--topk-chunks", type=int, default=32)
    ap.add_argument("--topk-groups", type=int, default=64)
    ap.add_argument("--cold", choices=["ram", "vram"], default="vram",
                    help="where detached cold KV lives: host RAM (offload) or VRAM (inference-style)")
    # optimization
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=20)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", default=None, help="comma list to override auto targets")
    # data
    ap.add_argument("--sources", default="mm,long,dialog")
    ap.add_argument("--mix", default="10,1,1", help="sampling weights matching --sources")
    ap.add_argument("--dialog-name", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--dialog-split", default="train_sft")
    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--save", default=None, help="dir to save the LoRA adapter at the end")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    assert args.seq_len % args.chunk_size == 0, "seq_len must be a multiple of chunk_size"
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # run dtype for activations / KV (fp8 weights still produce bf16 activations on the 5090)
    if args.dtype == "auto":
        run_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    else:
        run_dtype = DTYPES[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"Loading {args.model} (run_dtype={run_dtype}, device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = dict(attn_implementation="eager")
    if args.dtype != "auto":
        load_kwargs["dtype"] = run_dtype          # let transformers keep fp8/native dtype on 'auto'
    elif device.type == "cpu":
        load_kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
    model.config.use_cache = False

    # --- HGA surgery + LoRA ----------------------------------------------------------------
    store_factory = vram_hybrid_store_factory if args.cold == "vram" else hybrid_grad_store_factory
    controller = patch_qwen3_with_router(
        model, chunk_size=args.chunk_size, group_size=args.group_size,
        keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk_chunks, topk_groups=args.topk_groups,
        store_factory=store_factory,
    )
    targets = build_lora_targets(model.config, args.lora_targets)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA targets={targets}  trainable params={n_train:,}  cold-tier={args.cold.upper()}")

    # --- data ------------------------------------------------------------------------------
    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    weights = [float(x) for x in args.mix.split(",")]
    assert len(names) == len(weights), "--mix must match --sources"
    sources, weights, used = build_sources(
        names, weights, tokenizer=tokenizer, seq_len=args.seq_len,
        mm_path=MM_FILE, parquet_glob=PARQUET_GLOB,
        dialog_name=args.dialog_name, dialog_split=args.dialog_split, seed=args.seed,
    )
    print(f"Active sources={used}  weights={weights}")
    batcher = MixedBatcher(sources, weights, args.batch_size, args.seq_len, seed=args.seed)

    # --- optimizer + schedule --------------------------------------------------------------
    fused_ok = device.type == "cuda"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), fused=fused_ok)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # block plan: first block = 1 chunk (force incremental, never the start_pos==0 vectorized
    # prefill), then blocks of (keep_last + block_extra) chunks.
    n_chunks = args.seq_len // args.chunk_size
    block_chunks = args.keep_last + args.block_extra
    plan = _segment_plan(n_chunks, block_chunks)
    T = args.seq_len
    print(f"seq_len={T}  n_chunks={n_chunks}  block={block_chunks} chunks "
          f"({block_chunks * args.chunk_size} tok)  blocks/seq={len(plan)}  "
          f"tokens/step={T * args.batch_size}")

    if args.compile:
        model = torch.compile(model)

    # --- training loop ---------------------------------------------------------------------
    model.train()
    V = model.config.vocab_size
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(args.max_steps):
        batch = batcher.next_batch().to(device)            # [B, T]
        B = batch.shape[0]
        controller.begin(batch_size=B, dtype=run_dtype, device=device, start_pos=0)
        opt.zero_grad(set_to_none=True)

        t0 = time.time()
        step_loss = 0.0
        tok = 0
        for seg_chunks in plan:
            seg_len = seg_chunks * args.chunk_size
            controller.start_pos = tok
            seg_ids = batch[:, tok : tok + seg_len]
            tgt = batch[:, tok + 1 : tok + seg_len + 1]    # next-token targets (may be short at end)
            w = tgt.shape[1]
            position_ids = torch.arange(tok, tok + seg_len, device=device).unsqueeze(0).expand(B, -1)
            out = model(input_ids=seg_ids, position_ids=position_ids, use_cache=False)
            logits = out.logits[:, :w].reshape(-1, V)
            loss = ce(logits.float(), tgt.reshape(-1)) * (w / (T - 1))
            loss.backward()
            controller.detach_cache()                       # truncated BPTT across blocks
            step_loss += float(loss.detach())
            tok += seg_len

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            tps = (T * B) / dt
            print(f"step {step:5d}  loss {step_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{dt:.2f}s  {tps:,.0f} tok/s")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        print(f"saved LoRA adapter to {args.save}")


if __name__ == "__main__":
    main()
