#!/usr/bin/env python3
"""Stage 2 smoke test: hybrid grad store + streaming forward.

Verifies the long-context training mechanics that bound VRAM:
  * the sequence is fed in chunk-aligned segments through the **incremental** router path
    (never the vectorized ``start_pos == 0`` prefill);
  * the :class:`HybridGradKVCacheStore` partitions KV correctly — the last ``keep_last`` closed
    chunks form a **grad-carrying hot window on the compute device**, while every older chunk is
    **detached on host RAM** (``requires_grad == False``, ``grad_fn is None``);
  * backward over the streamed logits flows gradient into **LoRA only** (base frozen), proving the
    "only the last N chunks carry K/V gradients, the rest live on RAM" training regime.

Runs on CPU by default (GPU may be busy).  On CPU, compute_device == ram_device, so the cold/hot
split is asserted via ``grad_fn`` (the real invariant); on CUDA it is additionally a VRAM/RAM split.
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

from routed_attention import (  # noqa: E402
    patch_qwen3_with_router, hybrid_grad_store_factory, vram_hybrid_store_factory,
)
from streaming import streaming_forward  # noqa: E402

MM_FILE = os.path.join(_ROOT, "The-Master-and-Margarita.txt")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq-len", type=int, default=768)       # 12 chunks of 64
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--block-chunks", type=int, default=8)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=4)
    ap.add_argument("--topk-chunks", type=int, default=8)
    ap.add_argument("--topk-groups", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--cold", choices=["ram", "vram"], default="ram",
                    help="where detached cold chunks live: host RAM (offload) or VRAM (inference-style)")
    args = ap.parse_args()

    assert args.seq_len % args.chunk_size == 0, "seq_len must be a multiple of chunk_size"
    n_chunks = args.seq_len // args.chunk_size
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

    # --- HGA surgery with the Stage-2 hybrid store ----------------------------------------
    store_factory = vram_hybrid_store_factory if args.cold == "vram" else hybrid_grad_store_factory
    controller = patch_qwen3_with_router(
        model, chunk_size=args.chunk_size, group_size=args.group_size,
        keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk_chunks, topk_groups=args.topk_groups,
        store_factory=store_factory,
    )
    print(f"Patched {model.config.num_hidden_layers} attention layers "
          f"(hybrid grad store, cold tier = {args.cold.upper()}).")

    # --- LoRA on attn + mlp; only LoRA trainable ------------------------------------------
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    assert n_trainable == n_lora and n_lora > 0, "only LoRA params must be trainable"
    print(f"trainable params: {n_trainable:,} (all LoRA)")

    # --- one batch from The-Master-and-Margarita ------------------------------------------
    with open(MM_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= args.seq_len + 1, "M&M text too short for seq_len"
    input_ids = ids[: args.seq_len].unsqueeze(0).to(device)
    labels = input_ids.clone()

    # --- streaming forward (segment-wise, incremental router path) ------------------------
    model.train()
    logits, loss = streaming_forward(
        model, controller, input_ids,
        chunk_size=args.chunk_size, block_chunks=args.block_chunks,
        device=device, dtype=dtype, labels=labels,
    )
    print(f"loss = {float(loss.detach()):.4f}  logits={tuple(logits.shape)}")

    # --- store partition audit (BEFORE backward) ------------------------------------------
    store = controller.store
    st = store._layer(0)
    n_closed = store.num_closed_chunks(0)
    n_evicted = st.cold_tk.n
    n_hot = len(st.hot)
    expected_evicted = max(0, n_chunks - args.keep_last)
    print(f"layer-0: closed={n_closed} cold(RAM)={n_evicted} hot(grad)={n_hot} "
          f"(expect closed={n_chunks}, cold={expected_evicted}, hot={args.keep_last})")

    cold_k = st.cold_tk.data
    cold_detached = (cold_k is not None
                     and not cold_k.requires_grad and cold_k.grad_fn is None
                     and cold_k.device == store.ram_device)
    hot_grad = all(k.grad_fn is not None and k.device == store.compute_device
                   for k, _ in st.hot)
    print(f"cold token K/V: detached on RAM  -> {cold_detached}")
    print(f"hot  token K/V: grad on device   -> {hot_grad}")

    # routing table tiny + on the compute device (scanned every step) ----------------------
    table_on_device = st.chunk_k.data is not None and st.chunk_k.data.device == store.compute_device
    summaries_on_ram = st.group_k.data is not None and st.group_k.data.device == store.ram_device
    print(f"chunk_k routing table on compute device: {table_on_device}")
    print(f"group summaries on RAM                 : {summaries_on_ram}")

    # --- backward + gradient audit --------------------------------------------------------
    loss.backward()
    lora_with_grad = base_with_grad = 0
    lora_grad_sum = 0.0
    for n, p in model.named_parameters():
        g = p.grad
        if g is not None and float(g.abs().sum()) > 0:
            if "lora_" in n:
                lora_with_grad += 1
                lora_grad_sum += float(g.abs().sum())
            else:
                base_with_grad += 1
    print(f"LoRA tensors with non-zero grad: {lora_with_grad}")
    print(f"base tensors with grad         : {base_with_grad} (must be 0)")
    print(f"sum|LoRA grad|                 : {lora_grad_sum:.4e}")

    ok = (torch.isfinite(loss)
          and n_closed == n_chunks
          and n_evicted == expected_evicted
          and n_hot == args.keep_last
          and cold_detached and hot_grad
          and table_on_device and summaries_on_ram
          and lora_with_grad > 0 and base_with_grad == 0)
    print("STAGE 2 SMOKE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
