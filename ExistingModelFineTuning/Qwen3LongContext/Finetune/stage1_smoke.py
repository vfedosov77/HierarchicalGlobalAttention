#!/usr/bin/env python3
"""Stage 1 smoke test: Qwen3 + HGA routed attention + LoRA, one training step.

Verifies the drop-in mechanics end-to-end:
  * every ``self_attn`` is routed (no SDPA path exists), and the router actually closes chunks;
  * LoRA (attn + mlp) is the ONLY trainable part — base weights stay frozen with ``grad is None``;
  * a forward + backward over a multi-chunk sequence produces finite loss and non-zero LoRA grads.

Runs on CPU by default (the GPU may be busy with another job); point ``--model`` at a bigger Qwen3
and ``--device cuda`` for the real thing.  Defaults to the cached ``Qwen/Qwen3-0.6B`` (same dense
Qwen3 attention as 14B: q_norm/k_norm, GQA).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLC = os.path.dirname(_HERE)
_EFT = os.path.dirname(_QLC)
_ROOT = os.path.dirname(_EFT)
for _p in (_HERE, _QLC, _EFT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from routed_attention import patch_qwen3_with_router  # noqa: E402

MM_FILE = os.path.join(_ROOT, "The-Master-and-Margarita.txt")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--topk-chunks", type=int, default=32)
    ap.add_argument("--topk-groups", type=int, default=64)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()

    assert args.seq_len % args.chunk_size == 0, "seq_len must be a multiple of chunk_size"
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"Loading {args.model} (dtype={dtype}, device={device}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False

    # --- HGA surgery: every self_attn now always routes (no SDPA fallback) ----------------
    controller = patch_qwen3_with_router(
        model, chunk_size=args.chunk_size, group_size=args.group_size,
        keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk_chunks, topk_groups=args.topk_groups,
    )
    print(f"Patched {model.config.num_hidden_layers} attention layers with router.")

    # --- LoRA on attn + mlp; only LoRA trainable ------------------------------------------
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    assert n_trainable == n_lora and n_lora > 0, "only LoRA params must be trainable"
    print(f"trainable params: {n_trainable:,} (all LoRA: {n_lora == n_trainable})")

    # --- one batch from The-Master-and-Margarita ------------------------------------------
    with open(MM_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= args.seq_len + 1, "M&M text too short for seq_len"
    input_ids = ids[: args.seq_len].unsqueeze(0).to(device)
    labels = input_ids.clone()

    # --- forward + backward ----------------------------------------------------------------
    model.train()
    controller.begin(batch_size=1, dtype=dtype, device=device, start_pos=0)
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    print(f"loss = {float(loss.detach()):.4f}")
    loss.backward()

    # router actually closed chunks (proof the routed path ran, not a no-op) ---------------
    n_closed = controller.store.num_closed_chunks(0)
    print(f"router layer-0 closed chunks: {n_closed} (expected ~{args.seq_len // args.chunk_size})")

    # gradient audit: LoRA gets grad, base stays frozen (grad is None) ----------------------
    lora_grad_sum = 0.0
    lora_with_grad = 0
    base_with_grad = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                lora_with_grad += 1
                lora_grad_sum += float(p.grad.abs().sum())
        else:
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                base_with_grad += 1

    print(f"LoRA tensors with non-zero grad: {lora_with_grad}")
    print(f"base tensors with grad        : {base_with_grad} (must be 0)")
    print(f"sum|LoRA grad|                 : {lora_grad_sum:.4e}")

    ok = (torch.isfinite(loss) and lora_with_grad > 0 and base_with_grad == 0
          and n_closed >= args.seq_len // args.chunk_size - 1)
    print("STAGE 1 SMOKE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
