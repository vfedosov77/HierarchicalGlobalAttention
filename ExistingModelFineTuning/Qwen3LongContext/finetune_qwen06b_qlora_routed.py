"""QLoRA fine-tune Qwen3-0.6B **with the routed sparse attention active during training**.

The repo's ``replace_qwen_attention_with_router`` swaps every ``Qwen3Attention`` for a drop-in
that reuses the original q/k/v/o projections (here their 4-bit + LoRA versions) *by reference*
and only changes *what each query attends to*.  A fresh full-sequence forward (``start_pos==0``,
multi-chunk) takes the vectorized chunk-parallel path, which is differentiable: routing
*selection* runs under ``no_grad`` (hard top-k), but the K/V gathers and the attention math are
not, so gradients reach the LoRA adapters on q/k/v/o (and, via the residual stream, the MLP).

Methodology: QLoRA (4-bit NF4 frozen base + LoRA adapters, paged 8-bit AdamW).

Run from the repo root::

    python -m ExistingModelFineTuning.Qwen3LongContext.finetune_qwen06b_qlora_routed --smoke
    python -m ExistingModelFineTuning.Qwen3LongContext.finetune_qwen06b_qlora_routed \
        --data-path TrainData/The-Master-and-Margarita.txt --epochs 3

Reloading the adapter for inference: build the base model, apply
``replace_qwen_attention_with_router`` with the *same* routing knobs, then
``PeftModel.from_pretrained(base, out_dir)`` — the adapter keys carry the wrapper's ``.orig.``
prefix, so the surgery must be applied before loading.

ponytail: gradient checkpointing is intentionally OFF — it would recompute the attention
forward and double-seed the stateful router store.  Bounded VRAM is kept instead by resetting
the per-layer routers before every micro-forward (store holds at most one sequence).  This caps
the sequence length: ``--seq-len 1024`` fits a 16 GB Turing card; 2048 OOMs without
checkpointing.  Upgrade path: thread ``populate_store=False`` through
``QwenRoutedAttention.forward`` to drop the seeding entirely and re-enable checkpointing for
longer ``--seq-len``.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import math
import os
import random
import sys
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- routed-attention surgery (works from repo root or this folder) -------------------------
try:
    from .qwen_routed_attention import (  # type: ignore
        QwenRoutedAttention,
        replace_qwen_attention_with_router,
        restore_original_attention,
    )
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_routed_attention import (  # type: ignore
        QwenRoutedAttention,
        replace_qwen_attention_with_router,
        restore_original_attention,
    )

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =================================================================================================
# Data
# =================================================================================================
def build_blocks(text: str, tokenizer, seq_len: int) -> torch.Tensor:
    """Tokenize the whole text and pack it into ``[N, seq_len]`` non-overlapping blocks."""
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    n_blocks = len(ids) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"Text has {len(ids)} tokens but seq_len={seq_len}; need at least one full block."
        )
    ids = torch.tensor(ids[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    return ids


# =================================================================================================
# Router bookkeeping
# =================================================================================================
def reset_routers(model: torch.nn.Module) -> None:
    """Clear every wrapped layer's KV store so it never accumulates across steps."""
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            r = getattr(m, "_kv_router", None)
            if r is not None:
                r.reset()


# =================================================================================================
# Validation against the dense-attention baseline (same trained weights, router toggled off)
# =================================================================================================
@contextlib.contextmanager
def dense_attention(model: Any, knobs: Dict[str, Any]):
    """Temporarily run the *current* model with original dense attention, then restore routing.

    ``restore_original_attention`` puts each ``QwenRoutedAttention.orig`` back as ``self_attn``.
    Because ``orig`` still holds the 4-bit base + LoRA projections by reference, the baseline is
    the *same trained weights* under plain causal attention — isolating the pure effect of
    routing. Re-wrapping afterwards adds no learned parameters, so optimizer references to the
    LoRA tensors stay valid.  Operates on ``get_base_model()`` because ``_iter_attention_layers``
    needs the underlying ``Qwen3*ForCausalLM`` decoder, not the PEFT wrapper.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    restore_original_attention(base)
    try:
        yield
    finally:
        replace_qwen_attention_with_router(base, **knobs)


@torch.no_grad()
def _avg_loss(model: Any, blocks: torch.Tensor, device: torch.device,
              compute_dtype: torch.dtype, routed: bool) -> float:
    """Mean next-token CE over the held-out blocks (one block at a time to bound VRAM)."""
    total = 0.0
    for i in range(blocks.shape[0]):
        block = blocks[i : i + 1].to(device)
        if routed:
            reset_routers(model)  # the router store is stateful; start each block clean
        with torch.autocast("cuda", dtype=compute_dtype):
            loss = model(input_ids=block, labels=block).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss on block {i}")
        total += loss.item()
    return total / max(1, blocks.shape[0])


@torch.no_grad()
def evaluate(model: Any, val_blocks: torch.Tensor, device: torch.device,
             compute_dtype: torch.dtype, knobs: Dict[str, Any], opt_step: int) -> Dict[str, float]:
    """Compare routed-attention loss/perplexity against the dense baseline on the held-out blocks."""
    was_training = model.training
    model.eval()
    try:
        routed_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=True)
        with dense_attention(model, knobs):
            base_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=False)
    finally:
        if was_training:
            model.train()

    metrics = {
        "routed_loss": routed_loss,
        "routed_ppl": math.exp(min(routed_loss, 30.0)),
        "base_loss": base_loss,
        "base_ppl": math.exp(min(base_loss, 30.0)),
        "delta_loss": routed_loss - base_loss,
    }
    print(
        f"[val step {opt_step}] routed_loss={metrics['routed_loss']:.4f} ppl={metrics['routed_ppl']:.3f} | "
        f"base_loss={metrics['base_loss']:.4f} ppl={metrics['base_ppl']:.3f} | "
        f"delta(routed-base)={metrics['delta_loss']:+.4f} over {val_blocks.shape[0]} blocks"
    )
    return metrics



# =================================================================================================
# Model assembly (QLoRA + routed attention)
# =================================================================================================
def routing_defaults() -> Dict[str, Any]:
    """The ``qwen_routed_attention`` routing geometry — the single source of truth.

    Pulled straight from ``replace_qwen_attention_with_router``'s signature so ``build_model``,
    the validation baseline toggle, and the seq-len checks all agree without duplicating values.
    """
    sig = inspect.signature(replace_qwen_attention_with_router)
    keys = ("chunk_size", "group_size", "keep_first", "keep_last", "topk_chunks", "topk_groups")
    return {k: sig.parameters[k].default for k in keys}


def build_model(args, compute_dtype: torch.dtype):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Routed attention surgery first: the wrapper holds q/k/v/o by reference, so the LoRA
    # layers injected next are exactly what the router calls -> adapters train through routing.
    n = replace_qwen_attention_with_router(model, **routing_defaults())
    if n == 0:
        raise RuntimeError("No attention layers were wrapped; check the model architecture.")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, n


# =================================================================================================
# Training
# =================================================================================================
def train(args) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA training.")
    knobs = routing_defaults()
    chunk_size, keep_first, keep_last = knobs["chunk_size"], knobs["keep_first"], knobs["keep_last"]
    if args.seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be a multiple of chunk_size ({chunk_size}).")
    window = (keep_first + keep_last) * chunk_size
    if args.seq_len <= window:
        raise ValueError(
            f"seq_len ({args.seq_len}) must exceed the resident windows ({window}) so routing engages."
        )
    if not (os.path.isfile(args.data_path) and os.path.getsize(args.data_path) > 0):
        raise FileNotFoundError(f"Training text not found or empty: {args.data_path}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"[setup] model={args.model} seq_len={args.seq_len} dtype={compute_dtype} device={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    blocks = build_blocks(text, tokenizer, args.seq_len)
    print(f"[data] {args.data_path}: {blocks.numel()} tokens -> {blocks.shape[0]} blocks of {args.seq_len}")

    # Hold out the last val_blocks rows for validation so the dense-vs-routed comparison runs on
    # text the adapters never trained on.
    val_blocks = None
    if args.save_every > 0 and args.val_blocks > 0:
        if blocks.shape[0] <= args.val_blocks + 1:
            raise ValueError(
                f"Need > {args.val_blocks + 1} blocks to hold out {args.val_blocks} for validation; "
                f"got {blocks.shape[0]}. Lower --val-blocks or use a longer text."
            )
        val_blocks = blocks[-args.val_blocks :].clone()
        blocks = blocks[: -args.val_blocks]
        print(f"[data] holding out {val_blocks.shape[0]} blocks for validation -> {blocks.shape[0]} train blocks")

    model, n_wrapped = build_model(args, compute_dtype)
    print(f"[model] wrapped {n_wrapped} attention layers with the router")
    model.print_trainable_parameters()
    model.train()

    loader = DataLoader(TensorDataset(blocks), batch_size=args.batch_size, shuffle=True, drop_last=True)
    micro_per_epoch = len(loader)
    opt_steps_per_epoch = max(1, micro_per_epoch // args.accum)
    total_opt_steps = opt_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_opt_steps = min(total_opt_steps, args.max_steps)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, total_opt_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    print(f"[train] {args.epochs} epoch(s), {total_opt_steps} optimizer steps (accum={args.accum})")

    opt_step = 0
    running = 0.0
    last_loss = float("nan")
    done = False
    t0 = time.time()
    for epoch in range(args.epochs):
        if done:
            break
        optimizer.zero_grad(set_to_none=True)
        for i, (batch,) in enumerate(loader):
            reset_routers(model)  # store stays bounded to one sequence
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=compute_dtype):
                out = model(input_ids=batch, labels=batch)
                loss = out.loss / args.accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} micro {i}: {loss.item()}")
            scaler.scale(loss).backward()
            running += loss.item() * args.accum

            if (i + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    last_loss = running / (args.accum * args.log_every)
                    tok_s = (args.log_every * args.accum * args.batch_size * args.seq_len) / (time.time() - t0)
                    print(f"[step {opt_step}/{total_opt_steps}] loss={last_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} {tok_s:.0f} tok/s")
                    running = 0.0
                    t0 = time.time()

                if args.save_every > 0 and opt_step % args.save_every == 0:
                    if val_blocks is not None:
                        evaluate(model, val_blocks, device, compute_dtype, routing_defaults(), opt_step)
                    model.save_pretrained(args.output_dir)
                    print(f"[save] adapter -> {args.output_dir} (step {opt_step})")

                if opt_step >= total_opt_steps:
                    done = True
                    break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] final adapter -> {args.output_dir}")
    return last_loss


# =================================================================================================
# Smoke self-check (the one runnable check)
# =================================================================================================
def smoke(args) -> None:
    """Overfit a tiny fixed slice for a few steps; assert the loss drops clearly."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the smoke check.")
    torch.manual_seed(0)
    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    args.seq_len = 512  # 8 chunks > 4-chunk windows -> routing engages, fast
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    block = build_blocks(text, tokenizer, args.seq_len)[:1].to(device)  # one fixed block

    model, n = build_model(args, compute_dtype)
    assert n == int(model.config.num_hidden_layers), f"wrapped {n} != {model.config.num_hidden_layers} layers"
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable LoRA params"
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    losses: List[float] = []
    for step in range(args.smoke_steps):
        reset_routers(model)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=compute_dtype):
            loss = model(input_ids=block, labels=block).loss
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        print(f"[smoke {step + 1}/{args.smoke_steps}] loss={loss.item():.4f}")

    assert losses[-1] < 0.8 * losses[0], f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"[smoke] OK: loss {losses[0]:.3f} -> {losses[-1]:.3f} over {args.smoke_steps} steps")

    # Exercise the routed-vs-dense validation path: both losses must be finite and the router
    # must be re-wrapped after the dense-attention context manager (toggle round-trips cleanly).
    metrics = evaluate(model, block, device, compute_dtype, routing_defaults(), opt_step=args.smoke_steps)
    assert math.isfinite(metrics["routed_loss"]) and math.isfinite(metrics["base_loss"]), "non-finite validation loss"
    base = model.get_base_model()
    assert all(
        isinstance(layer.self_attn, QwenRoutedAttention)
        for layer in base.model.layers
    ), "router was not restored after dense-attention validation"
    print(f"[smoke] OK: validation routed={metrics['routed_loss']:.3f} base={metrics['base_loss']:.3f} delta={metrics['delta_loss']:+.3f}")


# =================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data-path", default=os.path.join(_REPO_ROOT, "TrainData", "The-Master-and-Margarita.txt"))
    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "ExistingModelFineTuning", "Qwen3LongContext", "qwen06b_routed_qlora_adapter"))
    p.add_argument("--seq-len", type=int, default=1024)  # 2048 OOMs on a 16 GB Turing card (no grad ckpt)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full schedule")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=200, help="save the adapter and run routed-vs-dense validation every N optimizer steps (0 disables both)")
    p.add_argument("--val-blocks", type=int, default=4, help="held-out blocks (from the end of the text) for validation")
    p.add_argument("--seed", type=int, default=1332)
    p.add_argument("--fp16", action="store_true", help="fp16 compute + GradScaler (Turing tensor-core path)")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # smoke
    p.add_argument("--smoke", action="store_true", help="run the overfit self-check and exit")
    p.add_argument("--smoke-steps", type=int, default=40)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
