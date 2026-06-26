#!/usr/bin/env python3
"""Block-wise long-context LoRA fine-tuning of Qwen3 with HGA routed attention.

What this does
--------------
* Replaces every ``self_attn`` with the router-backed sparse attention (no SDPA fallback) and
  trains **LoRA only** (attn ± mlp).
* Trains on **fixed-length sequences** (default 16K tokens) drawn from **three mixed sources**
  (the novel + a long-context corpus + a dialog corpus), batched ``--batch-size`` sequences at a
  time (default 3 → 48K tokens / step).
* Processes each sequence **block by block**.  A block is ``--block-chunks`` chunks (default 4, a
  few; capped at ``keep_last``): the model forwards **only that block's chunks**, while all earlier
  context comes
  from the **detached** KV cache (host RAM + the bounded VRAM cache).  So **only the current
  block's chunks carry gradients and activations** — everything before is stop-grad.  Each block
  runs its own ``backward`` and **accumulates** gradients into the LoRA params; the cache graph is
  then **detached** (truncated BPTT).  Because every query attends to a *fixed* routed window
  (``topk_chunks`` + the sink/recent windows, never the whole past), activation **and** VRAM stay
  flat regardless of sequence length — no checkpointing needed.
* Cold-tier placement (``--cold``): with ``ram`` (default) only the **hot window** (the last
  ``keep_last`` chunks, which carry gradients) plus — if ``--vram-cache-chunks > 0`` — a bounded
  **LRU VRAM cache** of the *most-recently-routed* cold chunks stay in **VRAM**; every other
  chunk's K/V + summaries live in **host RAM** and are pulled to VRAM only when a routed chunk is
  not already cached (the cache keeps the *most-useful* chunks resident, not just the last few).
  The cache is a **speed-vs-VRAM** knob: it avoids re-copying recurring chunks over PCIe but
  competes with the activation budget, so it is **off by default in training** (stream from RAM =
  flat, OOM-proof VRAM on any model/seq/GPU).  Raise it to fill spare VRAM — e.g. a 0.6B model at
  16K/batch-3 plateaus near 11GB of activations, leaving room for ~100 chunks on a 16GB card; each
  cached chunk costs ``B*kv_heads*C*head_dim*2`` bytes per layer.  Use ``vram`` only when the whole
  cold tier comfortably fits in VRAM (e.g. a 32GB RTX 5090); the LRU cache is then a no-op.
* Per-sequence cache isolation is automatic: every sequence is its own batch row in the store, so
  chunks never mix across sequences.

Model is swappable: ``Qwen/Qwen3-0.6B`` (default), ``Qwen/Qwen3-8B``, ``Qwen/Qwen3-14B``, and
``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`` (MoE/FP8, for a 32GB RTX 5090) are all supported — the
attention surgery and LoRA targets adapt to dense vs. MoE automatically.

Example (limited VRAM — cold cache offloaded to host RAM, only hot window in VRAM):
    python train.py --model Qwen/Qwen3-0.6B --device cuda \
        --seq-len 16384 --batch-size 3 --keep-last 8 --cold ram --lr 3e-5
Example (ample VRAM, RTX 5090 — whole cache resident in VRAM):
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
    patch_qwen3_with_router, make_hybrid_store_factory,
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
    ap.add_argument("--block-chunks", type=int, default=4,
                    help="chunks forwarded+backward per block = the grad/activation window "
                         "(a few; capped at keep_last). Earlier context is detached cache. "
                         "Activation ~ block_chunks * routed_window, so keep this small.")
    ap.add_argument("--topk-chunks", type=int, default=20)
    ap.add_argument("--topk-groups", type=int, default=32)
    ap.add_argument("--cold", choices=["ram", "vram"], default="ram",
                    help="cold KV placement: 'ram' keeps only the hot window + the VRAM working-set "
                         "cache in VRAM and offloads older chunks to host RAM (default); 'vram' keeps "
                         "the whole cache in VRAM (only when it fits, e.g. a 32GB GPU)")
    ap.add_argument("--vram-cache-chunks", type=int, default=0,
                    help="bounded LRU VRAM cache of most-useful cold chunks (0=off, stream from "
                         "RAM). Speed knob that competes with the activation budget; raise it to "
                         "fill spare VRAM. Each chunk costs B*kv_heads*C*head_dim*2 bytes/layer.")
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

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # A pre-quantized checkpoint (e.g. the FP8 30B) carries its own weight dtype — never override
    # it.  For every other model, `--dtype auto` must still pick a *compute* dtype explicitly:
    # `from_pretrained` defaults to fp32 when no dtype is given (the config's bf16 is ignored),
    # which would silently run the whole model in fp32 and blow up VRAM.
    cfg_peek = AutoConfig.from_pretrained(args.model)
    is_quantized = getattr(cfg_peek, "quantization_config", None) is not None

    print(f"Loading {args.model} (run_dtype={run_dtype}, device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = dict(attn_implementation="eager")
    if is_quantized:
        pass                                      # keep the checkpoint's native (fp8/quant) dtype
    elif args.dtype != "auto":
        load_kwargs["dtype"] = run_dtype
    elif device.type == "cpu":
        load_kwargs["dtype"] = torch.float32
    else:
        load_kwargs["dtype"] = run_dtype          # auto + cuda -> bf16 (NOT fp32, the silent default)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
    model.config.use_cache = False

    # --- HGA surgery + LoRA ----------------------------------------------------------------
    store_factory = make_hybrid_store_factory(cold=args.cold, vram_cache_chunks=args.vram_cache_chunks)
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
    cache_desc = f"{args.vram_cache_chunks} chunks" if args.cold == "ram" and args.vram_cache_chunks > 0 else "off"
    print(f"LoRA targets={targets}  trainable params={n_train:,}  cold-tier={args.cold.upper()}  "
          f"vram-cache={cache_desc}")

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
    # prefill), then blocks of `block_chunks` (the grad/activation window; a few chunks, capped
    # at keep_last so a block never evicts its own hot chunks mid-block).
    n_chunks = args.seq_len // args.chunk_size
    block_chunks = min(args.block_chunks, args.keep_last)
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
        for bi, seg_chunks in enumerate(plan):
            seg_len = seg_chunks * args.chunk_size
            controller.start_pos = tok
            seg_ids = batch[:, tok : tok + seg_len]
            tgt = batch[:, tok + 1 : tok + seg_len + 1]    # next-token targets (may be short at end)
            w = tgt.shape[1]
            position_ids = torch.arange(tok, tok + seg_len, device=device).unsqueeze(0).expand(B, -1)
            out = model(input_ids=seg_ids, position_ids=position_ids, use_cache=False)
            logits = out.logits[:, :w].reshape(-1, V)
            loss = ce(logits.float(), tgt.reshape(-1)) * (w / (T - 1))
            if os.environ.get("KVR_MEMDBG") and device.type == "cuda":
                print(f"  [memdbg] block {bi:2d} POST-FWD alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB seg_len={seg_len}", flush=True)
            loss.backward()
            controller.detach_cache()                       # truncated BPTT across blocks
            step_loss += float(loss.detach())
            tok += seg_len
            if os.environ.get("KVR_MEMDBG") and device.type == "cuda":
                print(f"  [memdbg] block {bi:2d} start_pos={controller.start_pos:6d} "
                      f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

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
