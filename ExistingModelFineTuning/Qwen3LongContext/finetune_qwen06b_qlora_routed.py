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
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

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
# Memory-frugal loss: chunked lm_head + cross-entropy (no full [B, S, vocab] logits tensor)
# =================================================================================================
def chunked_causal_lm_loss(
    model: Any, input_ids: torch.Tensor, labels: torch.Tensor, chunk_size: int, *, train: bool
) -> torch.Tensor:
    """Next-token CE that never materializes the full ``[B, S, vocab]`` logits tensor.

    Runs the decoder backbone once for hidden states, then streams ``lm_head`` + cross-entropy
    over ``chunk_size``-token slices and sums, dividing by the valid-label count at the end.  This
    keeps memory flat in the (huge) vocab dimension -- the dominant OOM driver at long ``seq_len``
    -- at the cost of a Python loop over a handful of slices.  Semantics match
    ``model(labels=...).loss`` (next-token shift, ``ignore_index=-100``, mean over non-ignored).

    In the training path each slice is wrapped in ``checkpoint`` so the per-slice logits are
    recomputed in backward instead of being held; in eval (under ``no_grad``) it runs plain.
    """
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = backbone.model(input_ids=input_ids, use_cache=False)[0]  # [B, S, D]; LoRA applies in-place
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    lm_head = backbone.lm_head

    def _slice_loss(h: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(lm_head(h).float(), lbl, ignore_index=-100, reduction="sum")

    total = shift_hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, shift_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, shift_hidden.shape[0])
        lbl = shift_labels[start:end]
        valid = int((lbl != -100).sum().item())
        if valid == 0:
            continue
        h = shift_hidden[start:end]
        slice_loss = checkpoint(_slice_loss, h, lbl, use_reentrant=False) if train else _slice_loss(h, lbl)
        total = total + slice_loss
        count += valid
    if count == 0:
        raise RuntimeError("No valid label tokens in the block")
    return total / count


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


def set_token_stats(model: torch.nn.Module, on: bool) -> None:
    """Enable/disable routed attended-token collection and clear the per-layer accumulators."""
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            m.collect_token_stats = on
            m._stat_visible = None
            m._stat_dense = 0
            m._stat_queries = 0


def read_token_stats(model: torch.nn.Module) -> Optional[Dict[str, float]]:
    """Aggregate routed attended-KV stats across all wrapped layers, or ``None`` if uncollected.

    Reports, averaged over layers and queries: ``attended`` (real KV tokens a query attends),
    the dense causal reference ``dense``, and the resulting ``density`` / ``saving`` fractions.
    Sums the on-device counters first so only a single host sync (``.item()``) is paid per read.
    """
    visible_t: Optional[torch.Tensor] = None
    dense = queries = 0
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            if m._stat_visible is not None:
                visible_t = m._stat_visible if visible_t is None else visible_t + m._stat_visible
            dense += m._stat_dense
            queries += m._stat_queries
    if queries == 0 or dense == 0 or visible_t is None:
        return None
    visible = int(visible_t.item())
    return {
        "attended": visible / queries,
        "dense": dense / queries,
        "density": visible / dense,
        "saving": 1.0 - visible / dense,
    }


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
              compute_dtype: torch.dtype, routed: bool, loss_chunk_size: int,
              stock_base: bool = False) -> float:
    """Mean next-token CE over the held-out blocks (one block at a time to bound VRAM).

    ``stock_base`` runs under ``model.disable_adapter()``, which zeroes the LoRA contribution so
    the forward uses the *un-fine-tuned* 4-bit base weights — the reference for "did fine-tuning
    help?" with no second model to load (PEFT just toggles a flag on the same LoRA layers).
    """
    adapter_off = model.disable_adapter() if stock_base else contextlib.nullcontext()
    total = 0.0
    with adapter_off:
        for i in range(blocks.shape[0]):
            block = blocks[i : i + 1].to(device)
            if routed:
                reset_routers(model)  # the router store is stateful; start each block clean
            with torch.autocast("cuda", dtype=compute_dtype):
                loss = chunked_causal_lm_loss(model, block, block, loss_chunk_size, train=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss on block {i}")
            total += loss.item()
    return total / max(1, blocks.shape[0])


@torch.no_grad()
def evaluate(model: Any, val_blocks: torch.Tensor, device: torch.device,
             compute_dtype: torch.dtype, knobs: Dict[str, Any], opt_step: int,
             loss_chunk_size: int = 512) -> Dict[str, float]:
    """Three-way perplexity comparison on the held-out blocks, all on the *same* model object:

    * **routed** — the fine-tuned adapter with active routed sparse attention (the deploy regime).
    * **dense_adapter** — the *same* trained weights under plain dense attention (isolates the
      cost of routing alone).
    * **stock_base** — the un-fine-tuned 4-bit base under dense attention (LoRA disabled): the
      reference for whether fine-tuning improved perplexity on the novel.

    Two deltas are reported: ``routed − stock_base`` (fine-tuning benefit, should go negative as
    training progresses) and ``routed − dense_adapter`` (routing cost at the current weights).
    """
    was_training = model.training
    model.eval()
    try:
        # Measure the routed attention's real-token budget on the routed pass only.
        set_token_stats(model, True)
        routed_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=True, loss_chunk_size=loss_chunk_size)
        tok_stats = read_token_stats(model)
        set_token_stats(model, False)
        with dense_attention(model, knobs):
            dense_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=False, loss_chunk_size=loss_chunk_size)
            stock_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=False,
                                   loss_chunk_size=loss_chunk_size, stock_base=True)
    finally:
        if was_training:
            model.train()

    metrics = {
        "routed_loss": routed_loss,
        "routed_ppl": math.exp(min(routed_loss, 30.0)),
        "dense_loss": dense_loss,
        "dense_ppl": math.exp(min(dense_loss, 30.0)),
        "stock_loss": stock_loss,
        "stock_ppl": math.exp(min(stock_loss, 30.0)),
        "finetune_gain": routed_loss - stock_loss,   # < 0 ⇒ fine-tuning helped
        "routing_cost": routed_loss - dense_loss,     # routed vs same-weights dense
    }
    if tok_stats is not None:
        metrics.update({
            "attended_kv": tok_stats["attended"],
            "dense_kv": tok_stats["dense"],
            "density": tok_stats["density"],
            "saving": tok_stats["saving"],
        })
    attended_line = (
        f"\n    attended KV/query={tok_stats['attended']:.1f} (dense={tok_stats['dense']:.1f}, "
        f"density={tok_stats['density'] * 100:.1f}%, saving={tok_stats['saving'] * 100:.1f}%)"
        if tok_stats is not None else ""
    )
    print(
        f"[val step {opt_step}] over {val_blocks.shape[0]} blocks\n"
        f"    routed(adapter) loss={metrics['routed_loss']:.4f} ppl={metrics['routed_ppl']:.3f}\n"
        f"    dense(adapter)  loss={metrics['dense_loss']:.4f} ppl={metrics['dense_ppl']:.3f}\n"
        f"    stock(base)     loss={metrics['stock_loss']:.4f} ppl={metrics['stock_ppl']:.3f}\n"
        f"    finetune_gain(routed-stock)={metrics['finetune_gain']:+.4f} | "
        f"routing_cost(routed-dense)={metrics['routing_cost']:+.4f}"
        + attended_line
    )
    return metrics


def _maybe_save_best(model: Any, output_dir: str, metrics: Dict[str, float], best_val: float,
                     opt_step: int) -> float:
    """Save the adapter to ``<output_dir>/best`` whenever held-out routed loss improves.

    Domain-fit on a single novel can start overfitting late, so the lowest-held-out-loss adapter
    is kept separately from the always-overwritten final one. Returns the (possibly updated) best.
    """
    val = metrics["routed_loss"]
    if val < best_val:
        best_dir = os.path.join(output_dir, "best")
        model.save_pretrained(best_dir)
        print(f"[best] new best held-out routed_loss={val:.4f} (was {best_val:.4f}) -> {best_dir} (step {opt_step})")
        return val
    return best_val



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


def load_routed_base(args, compute_dtype: torch.dtype):
    """Load the 4-bit base and apply the routed-attention surgery (no LoRA yet).

    Shared by training (``build_model`` injects LoRA on top) and ``--evaluate-only`` (which loads
    a trained adapter on top via ``PeftModel.from_pretrained``). The surgery must come *before*
    either, so the adapter keys carry the wrapper's ``.orig.`` prefix consistently.
    """
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

    # Training/benchmark always run one full-sequence forward from position 0, which reads nothing
    # from the router KV store (the routed output is fully determined by assemble_routed_kv).
    # Disabling store seeding therefore cannot change the attention result; it only drops the
    # detached KV copy decode would need and makes the routed forward stateless -- which is what
    # lets gradient checkpointing recompute it safely in backward (the documented upgrade path).
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            m.populate_store = False
    return model, n


def build_model(args, compute_dtype: torch.dtype):
    model, n = load_routed_base(args, compute_dtype)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
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

    # Hold out the last val_blocks rows for validation so the base-vs-fine-tuned comparison runs
    # on text the adapters never trained on. Decoupled from --save-every: a final validation runs
    # at the end of training regardless of how often the periodic save fires.
    val_blocks = None
    if args.val_blocks > 0:
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
    set_token_stats(model, True)  # tally the routed attended-KV budget for the step log

    opt_step = 0
    running = 0.0
    last_loss = float("nan")
    best_val = float("inf")
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
                loss = chunked_causal_lm_loss(model, batch, batch, args.loss_chunk_size, train=True) / args.accum
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
                    tok = read_token_stats(model)
                    set_token_stats(model, True)  # reset the accumulation window
                    kv = (f" | attended {tok['attended']:.0f}/{tok['dense']:.0f} KV ({tok['saving'] * 100:.0f}% saved)"
                          if tok is not None else "")
                    print(f"[step {opt_step}/{total_opt_steps}] loss={last_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} {tok_s:.0f} tok/s{kv}")
                    running = 0.0
                    t0 = time.time()

                if args.save_every > 0 and opt_step % args.save_every == 0:
                    if val_blocks is not None:
                        metrics = evaluate(model, val_blocks, device, compute_dtype, routing_defaults(), opt_step,
                                           loss_chunk_size=args.loss_chunk_size)
                        best_val = _maybe_save_best(model, args.output_dir, metrics, best_val, opt_step)
                    ckpt_dir = os.path.join(args.output_dir, "last")
                    model.save_pretrained(ckpt_dir)
                    print(f"[save] adapter -> {ckpt_dir} (step {opt_step})")
                    set_token_stats(model, True)  # evaluate() re-wraps attention; resume the step-log tally

                if opt_step >= total_opt_steps:
                    done = True
                    break

    # Final validation always runs (independent of --save-every) so every run ends with the
    # base-vs-fine-tuned table, and the best adapter reflects the last weights too.
    if val_blocks is not None:
        metrics = evaluate(model, val_blocks, device, compute_dtype, routing_defaults(), opt_step,
                           loss_chunk_size=args.loss_chunk_size)
        best_val = _maybe_save_best(model, args.output_dir, metrics, best_val, opt_step)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] final adapter -> {args.output_dir}")
    if val_blocks is not None and math.isfinite(best_val):
        print(f"[done] best held-out routed_loss={best_val:.4f} -> {os.path.join(args.output_dir, 'best')}")
    return last_loss


# =================================================================================================
# Standalone validation of an already-trained adapter (no training)
# =================================================================================================
def evaluate_only(args) -> Dict[str, float]:
    """Load a saved adapter and run the three-way base-vs-fine-tuned comparison on held-out text.

    Builds the 4-bit base + routed surgery (same geometry as training), loads the adapter with
    ``PeftModel.from_pretrained`` (the surgery must precede the load so the ``.orig.`` adapter keys
    line up), then evaluates the routed adapter, the same-weights dense adapter, and the stock
    base on the last ``--val-blocks`` blocks of the novel — the same unseen tail training held out.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit evaluation.")
    adapter_dir = args.adapter_path or args.output_dir
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    knobs = routing_defaults()
    chunk_size = knobs["chunk_size"]
    if args.seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be a multiple of chunk_size ({chunk_size}).")
    if not (os.path.isfile(args.data_path) and os.path.getsize(args.data_path) > 0):
        raise FileNotFoundError(f"Evaluation text not found or empty: {args.data_path}")
    if args.val_blocks <= 0:
        raise ValueError("--val-blocks must be > 0 for --evaluate-only.")

    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    blocks = build_blocks(text, tokenizer, args.seq_len)
    if blocks.shape[0] <= args.val_blocks:
        raise ValueError(f"Need > {args.val_blocks} blocks; got {blocks.shape[0]}. Lower --val-blocks.")
    val_blocks = blocks[-args.val_blocks :].clone()
    print(f"[eval] {adapter_dir} on the last {val_blocks.shape[0]} blocks of {args.data_path} (seq_len={args.seq_len})")

    base, n = load_routed_base(args, compute_dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    print(f"[eval] wrapped {n} attention layers; loaded adapter from {adapter_dir}")
    metrics = evaluate(model, val_blocks, device, compute_dtype, knobs, opt_step=0,
                       loss_chunk_size=args.loss_chunk_size)
    return metrics


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
            loss = chunked_causal_lm_loss(model, block, block, args.loss_chunk_size, train=True)
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

    # Exercise the three-way validation path: all losses must be finite and the router must be
    # re-wrapped after the dense-attention context manager (dense<->routed and adapter on/off
    # toggles round-trip cleanly).
    metrics = evaluate(model, block, device, compute_dtype, routing_defaults(), opt_step=args.smoke_steps,
                       loss_chunk_size=args.loss_chunk_size)
    assert all(math.isfinite(metrics[k]) for k in ("routed_loss", "dense_loss", "stock_loss")), "non-finite validation loss"
    base = model.get_base_model()
    assert all(
        isinstance(layer.self_attn, QwenRoutedAttention)
        for layer in base.model.layers
    ), "router was not restored after dense-attention validation"
    print(
        f"[smoke] OK: validation routed={metrics['routed_loss']:.3f} dense={metrics['dense_loss']:.3f} "
        f"stock={metrics['stock_loss']:.3f} gain={metrics['finetune_gain']:+.3f}"
    )


# =================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data-path", default=os.path.join(_REPO_ROOT, "TrainData", "The-Master-and-Margarita.txt"))
    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "checkpoints"))
    # Recommended domain-fit config for a 16 GB Turing card with gradient checkpointing on.
    # 4096 = 64 chunks (routing engages strongly) and fits (~4 GB peak); longer blocks mean fewer
    # blocks, so accum is lowered to keep enough optimizer updates per epoch.
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--loss-chunk-size", type=int, default=512, help="tokens per lm_head+CE slice; smaller = less VRAM in the vocab dim")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full schedule")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=20, help="save the adapter and run three-way validation every N optimizer steps (0 disables periodic saves; the final validation still runs)")
    p.add_argument("--val-blocks", type=int, default=6, help="held-out blocks (from the end of the text) for validation")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--fp16", action="store_true", help="fp16 compute + GradScaler (Turing tensor-core path, ~2x throughput)")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # evaluation of an already-trained adapter (no training)
    p.add_argument("--evaluate-only", action="store_true", help="load a saved adapter and run the three-way base-vs-fine-tuned comparison, then exit")
    p.add_argument("--adapter-path", default=None, help="adapter dir for --evaluate-only (default: --output-dir)")
    # smoke
    p.add_argument("--smoke", action="store_true", help="run the overfit self-check and exit")
    p.add_argument("--smoke-steps", type=int, default=40)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        smoke(args)
    elif args.evaluate_only:
        evaluate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
