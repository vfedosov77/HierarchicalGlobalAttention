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

Training mixes the held-out-free prefix of ``--data-path`` with Hugging Face FineWeb blocks at a
Master:FineWeb ratio of 1:10 by default (``--fineweb-ratio 10``).  The FineWeb side is **streamed
fresh every epoch** from one persistent, seeded-shuffled stream (``--fineweb-shuffle-buffer``), so
the web-text regulariser keeps real diversity instead of replaying a single fixed up-front draw;
only one epoch's FineWeb is resident at a time.  Validation stays strictly on
``--val-data-path`` (default: the same Master-and-Margarita file); when train and validation paths
are the same file, the last ``--val-blocks`` Master blocks are removed from training before FineWeb
is added.

Reloading the adapter for inference: build the base model, apply
``replace_qwen_attention_with_router`` with the *same* routing knobs, then
``PeftModel.from_pretrained(base, out_dir)`` — the adapter keys carry the wrapper's ``.orig.``
prefix, so the surgery must be applied before loading.

``--attn-mode dense`` trains the *same* QLoRA adapters under the model's original dense attention
(the router surgery is skipped), giving an apples-to-apples "same LoRA, same data, dense
attention" counterpart to the routed run for the head-to-head loss/speed comparison.  A dense run
saves plain adapter keys (no ``.orig.`` prefix), so reload it onto an *un-surgered* base; the
three-way validation still reports routed (HGA applied zero-shot), dense (deploy), and stock
(initial) losses on the same held-out tail.

ponytail: by default training runs full fresh-sequence forwards with ``populate_store=False`` on
every routed layer — the vectorized assembly never reads the stateful KV store, which keeps
non-reentrant gradient checkpointing safe and VRAM bounded.  Pass ``--train-seg-len`` to switch on
the author's blocked-inference streaming path instead: the sequence is fed in segments with the
KV store accumulating across them (old K/V resident in the ``--cache-location`` tier, e.g. RAM)
and each segment runs its own backward (truncated BPTT), so VRAM stays bounded to one segment.
After the dense-validation toggle re-wraps attention, the active mode must be restored before
training resumes (see ``apply_segment_mode``).
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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

import bitsandbytes as bnb
from datasets import load_dataset
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
FINEWEB_RATIO = 10
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_SPLIT = "train"
FINEWEB_FIELD = "text"
FINEWEB_STREAM_RETRIES = 8       # transient HF read-timeout retries before giving up
FINEWEB_STREAM_BACKOFF = 5.0     # base seconds for exponential backoff (capped at 60s)


# =================================================================================================
# Data
# =================================================================================================
def load_text(path: str) -> str:
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        raise FileNotFoundError(f"Text file not found or empty: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def encode_no_special(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])


def build_blocks(text: str, tokenizer, seq_len: int) -> torch.Tensor:
    """Tokenize the whole text and pack it into ``[N, seq_len]`` non-overlapping blocks."""
    ids = encode_no_special(tokenizer, text)
    n_blocks = len(ids) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"Text has {len(ids)} tokens but seq_len={seq_len}; need at least one full block."
        )
    ids = torch.tensor(ids[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    return ids


def alnum_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalnum() for ch in text) / max(1, len(text))


def extract_text(example: Dict[str, Any], field: str) -> str:
    if field and isinstance(example.get(field), str):
        return example[field]
    strings = [value for value in example.values() if isinstance(value, str)]
    return max(strings, key=len) if strings else ""


def _fineweb_config(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" or stripped.lower() in {"none", "null"} else stripped


def _open_fineweb_stream(
    *, dataset_name: str, dataset_config: Optional[str], split: str,
    seed: Optional[int], shuffle_buffer: int, skip: int = 0,
):
    """Open the streaming FineWeb split (optionally seeded-shuffled, optionally fast-forwarded).

    ``skip`` drops the first ``skip`` examples *after* shuffling, so rebuilding with the same
    ``seed``/``shuffle_buffer`` reproduces the identical global order and ``.skip(seen)`` resumes
    exactly where a dropped connection left off.
    """
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    config = _fineweb_config(dataset_config)
    dataset = load_dataset(dataset_name, config, **kwargs) if config else load_dataset(dataset_name, **kwargs)
    if shuffle_buffer > 0 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip > 0:
        dataset = dataset.skip(skip)
    return dataset


def iter_fineweb_blocks(
    tokenizer,
    seq_len: int,
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    field: str,
    seed: Optional[int] = None,
    shuffle_buffer: int = 0,
) -> Iterator[torch.Tensor]:
    """Stream FineWeb text into fixed-size LM blocks, matching the neighboring project pattern.

    ``shuffle_buffer > 0`` turns on ``IterableDataset.shuffle(seed, buffer_size)`` so the draw is a
    seeded random window over the shard rather than the deterministic head of the stream (and so
    ``--seed`` actually controls *which* FineWeb data is seen).
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer has no eos_token_id; cannot separate FineWeb documents.")
    stream_kwargs = dict(dataset_name=dataset_name, dataset_config=dataset_config, split=split,
                         seed=seed, shuffle_buffer=shuffle_buffer)
    iterator = iter(_open_fineweb_stream(**stream_kwargs))
    seen = 0  # examples consumed so far; lets us .skip() back to here after a dropped connection

    buffer: List[int] = []
    while True:
        while len(buffer) >= seq_len:
            out = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yield torch.tensor(out, dtype=torch.long)

        # ponytail: HF streaming GETs time out mid-shard on a flaky link; a multi-hour run must not
        # die on a transient read.  Retry with capped exponential backoff, rebuilding the stream and
        # .skip()-ing past `seen` (same seed+shuffle ⇒ identical order ⇒ exact resume, no data loss).
        # Ceiling: FINEWEB_STREAM_RETRIES attempts; a genuinely persistent outage still raises.
        for attempt in range(FINEWEB_STREAM_RETRIES + 1):
            try:
                example = next(iterator)
                break
            except StopIteration:
                return
            except Exception as exc:  # network/IO read errors from the streaming reader
                if attempt == FINEWEB_STREAM_RETRIES:
                    raise
                wait = min(FINEWEB_STREAM_BACKOFF * (2 ** attempt), 60.0)
                print(f"[data] FineWeb stream error ({type(exc).__name__}: {exc}); "
                      f"retry {attempt + 1}/{FINEWEB_STREAM_RETRIES} in {wait:.0f}s "
                      f"(resuming after {seen} examples)")
                time.sleep(wait)
                iterator = iter(_open_fineweb_stream(**stream_kwargs, skip=seen))
        seen += 1
        text = extract_text(example, field)
        if len(text) < 200 or alnum_fraction(text) < 0.45:
            continue
        ids = encode_no_special(tokenizer, text)
        if len(ids) < 64:
            continue
        buffer.extend(ids)
        buffer.append(int(eos))


def make_fineweb_iter(args, tokenizer) -> Iterator[torch.Tensor]:
    """One persistent, seeded-shuffled FineWeb block stream.

    Draining it across epochs yields *fresh* blocks each epoch (the stream keeps advancing), so the
    web-text regulariser is real-diversity streaming rather than a single fixed up-front draw reused
    every epoch.  ``--fineweb-shuffle-buffer`` sets the seeded shuffle window (0 = deterministic
    stream head, lowest memory).
    """
    return iter_fineweb_blocks(
        tokenizer,
        args.seq_len,
        dataset_name=FINEWEB_DATASET,
        dataset_config=FINEWEB_CONFIG,
        split=FINEWEB_SPLIT,
        field=FINEWEB_FIELD,
        seed=args.seed,
        shuffle_buffer=getattr(args, "fineweb_shuffle_buffer", 0),
    )


def pull_fineweb_blocks(fineweb_iter: Iterator[torch.Tensor], target_blocks: int,
                        *, label: Optional[str] = None) -> torch.Tensor:
    """Pull exactly ``target_blocks`` blocks from a (persistent) FineWeb iterator.

    Reusing the same iterator across epochs returns fresh blocks each call.  Raises if the stream is
    exhausted before ``target_blocks`` (lower ``--epochs`` or ``--fineweb-ratio``).
    """
    blocks: List[torch.Tensor] = []
    log_every = max(1, min(100, target_blocks // 10 or 1))
    for block in fineweb_iter:
        blocks.append(block)
        if len(blocks) == target_blocks:
            break
        if label and len(blocks) % log_every == 0:
            print(f"[data] FineWeb {label}: collected {len(blocks)}/{target_blocks} blocks")
    if len(blocks) != target_blocks:
        raise RuntimeError(
            f"FineWeb produced {len(blocks)} blocks, expected {target_blocks}. "
            "The shard is exhausted; lower --epochs or --fineweb-ratio (or check dataset access)."
        )
    return torch.stack(blocks, dim=0)


def collect_fineweb_blocks(args, tokenizer, target_blocks: int) -> torch.Tensor:
    """Single fresh draw of ``target_blocks`` FineWeb blocks (one-shot / single-epoch helper)."""
    if target_blocks <= 0:
        return torch.empty((0, args.seq_len), dtype=torch.long)
    print(
        f"[data] FineWeb: pulling {target_blocks} blocks from "
        f"{FINEWEB_DATASET}/{FINEWEB_CONFIG or '<default>'}:{FINEWEB_SPLIT} "
        f"(streaming=True, field={FINEWEB_FIELD})"
    )
    return pull_fineweb_blocks(make_fineweb_iter(args, tokenizer), target_blocks, label="collect")


def build_master_train_val_blocks(args, tokenizer) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    master_blocks = build_blocks(load_text(args.data_path), tokenizer, args.seq_len)
    print(f"[data] master train source {args.data_path}: {master_blocks.numel()} tokens -> {master_blocks.shape[0]} blocks")
    if args.val_blocks <= 0:
        print("[data] validation disabled (--val-blocks <= 0)")
        return master_blocks, None

    same_file = os.path.exists(args.val_data_path) and os.path.samefile(args.data_path, args.val_data_path)
    if same_file:
        if master_blocks.shape[0] <= args.val_blocks + 1:
            raise ValueError(
                f"Need > {args.val_blocks + 1} Master blocks to hold out {args.val_blocks} for validation; "
                f"got {master_blocks.shape[0]}. Lower --val-blocks or use a longer text."
            )
        val_blocks = master_blocks[-args.val_blocks :].clone()
        train_blocks = master_blocks[: -args.val_blocks]
        print(
            f"[data] validation: held out {val_blocks.shape[0]} pure Master blocks from tail -> "
            f"{train_blocks.shape[0]} Master train blocks"
        )
        return train_blocks, val_blocks

    val_source_blocks = build_blocks(load_text(args.val_data_path), tokenizer, args.seq_len)
    if val_source_blocks.shape[0] < args.val_blocks:
        raise ValueError(
            f"Need at least {args.val_blocks} validation blocks from {args.val_data_path}; "
            f"got {val_source_blocks.shape[0]}."
        )
    val_blocks = val_source_blocks[-args.val_blocks :].clone()
    print(
        f"[data] validation: {val_blocks.shape[0]} pure blocks from {args.val_data_path}; "
        f"Master train source remains {master_blocks.shape[0]} blocks"
    )
    return master_blocks, val_blocks


def build_mixed_train_blocks(args, tokenizer, master_train_blocks: torch.Tensor) -> torch.Tensor:
    fineweb_ratio = getattr(args, "fineweb_ratio", FINEWEB_RATIO)
    if fineweb_ratio < 0:
        raise ValueError("--fineweb-ratio must be >= 0")
    target_fineweb_blocks = master_train_blocks.shape[0] * fineweb_ratio
    if target_fineweb_blocks == 0:
        print(f"[data] training: {master_train_blocks.shape[0]} Master blocks, FineWeb disabled")
        return master_train_blocks
    fineweb_blocks = collect_fineweb_blocks(args, tokenizer, target_fineweb_blocks)
    blocks = torch.cat([master_train_blocks, fineweb_blocks], dim=0)
    print(
        f"[data] training mix: {master_train_blocks.shape[0]} Master blocks + "
        f"{fineweb_blocks.shape[0]} FineWeb blocks -> {blocks.shape[0]} total "
        f"(Master:FineWeb = 1:{fineweb_ratio})"
    )
    return blocks


# =================================================================================================
# Memory-frugal loss: chunked lm_head + cross-entropy (no full [B, S, vocab] logits tensor)
# =================================================================================================
def chunked_causal_lm_loss(
    model: Any, input_ids: torch.Tensor, labels: torch.Tensor, chunk_size: int, *, train: bool,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Next-token CE that never materializes the full ``[B, S, vocab]`` logits tensor.

    Runs the decoder backbone once for hidden states, then streams ``lm_head`` + cross-entropy
    over ``chunk_size``-token slices and sums, dividing by the valid-label count at the end.  This
    keeps memory flat in the (huge) vocab dimension -- the dominant OOM driver at long ``seq_len``
    -- at the cost of a Python loop over a handful of slices.  Semantics match
    ``model(labels=...).loss`` (next-token shift, ``ignore_index=-100``, mean over non-ignored).

    In the training path each slice is wrapped in ``checkpoint`` so the per-slice logits are
    recomputed in backward instead of being held; in eval (under ``no_grad``) it runs plain.

    When ``position_ids`` is given (segmented streaming), it is forwarded to the backbone together
    with a matching ``cache_position`` so the routed attention sees the correct absolute
    ``start_pos`` (and RoPE) for a mid-sequence segment; otherwise the backbone defaults to
    ``arange(0, S)`` (a single fresh-sequence forward).
    """
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    extra: Dict[str, Any] = {}
    if position_ids is not None:
        extra["position_ids"] = position_ids
        extra["cache_position"] = position_ids.reshape(-1)
    hidden = backbone.model(input_ids=input_ids, use_cache=False, **extra)[0]  # [B, S, D]; LoRA in-place
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


def set_populate_store(model: torch.nn.Module, on: bool) -> None:
    """Enable/disable KV-store seeding on all routed attention wrappers."""
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            m.populate_store = on


def set_training_segments(model: torch.nn.Module, on: bool) -> None:
    """Enable/disable the per-forward store ``rewind`` (segmented-streaming training)."""
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            m.training_segments = on


def commit_routers(model: torch.nn.Module) -> None:
    """Commit each routed layer's store at a segment boundary (detach prefix; truncate BPTT)."""
    for m in model.modules():
        if isinstance(m, QwenRoutedAttention):
            r = getattr(m, "_kv_router", None)
            if r is not None:
                r.commit(m.layer_idx)


def apply_segment_mode(model: torch.nn.Module, n_seg: int) -> None:
    """Toggle the routed wrappers between single-forward and segmented-streaming training.

    ``n_seg > 1`` ⇒ the author's blocked-inference pattern: seed the store (``populate_store``) and
    ``rewind`` it each forward (``training_segments``) so segments stream through the cache tier.
    ``n_seg == 1`` ⇒ the stateless single full-sequence forward (store untouched).
    """
    segmented = n_seg > 1
    set_populate_store(model, segmented)
    set_training_segments(model, segmented)


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
def attention_mode(model: Any, knobs: Dict[str, Any], *, routed: bool, cache_location: str = "vram"):
    """Force the base model into routed (``routed=True``) or dense (``routed=False``) attention for
    the body, then restore whatever wrapping state it started in.

    Symmetric — it records the current state first, so it works whether the model was trained
    routed (starts wrapped) or dense (starts unwrapped).  Wrapping reuses the original q/k/v/o
    projections + LoRA *by reference* (``replace_qwen_attention_with_router`` adds no learned
    params), and ``restore_original_attention`` puts each ``QwenRoutedAttention.orig`` back, so the
    *same trained weights* are measured under both attentions and optimizer references stay valid.
    Operates on ``get_base_model()`` (the underlying ``Qwen3*ForCausalLM`` decoder, not the PEFT
    wrapper).  ``cache_location`` is preserved across a re-wrap so training resumes on the same KV
    tier (RAM offload survives).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    was_routed = any(isinstance(layer.self_attn, QwenRoutedAttention) for layer in base.model.layers)

    def _wrap() -> None:
        replace_qwen_attention_with_router(base, cache_location=cache_location, **knobs)
        set_populate_store(base, False)

    if routed and not was_routed:
        _wrap()
        revert = "unwrap"
    elif not routed and was_routed:
        restore_original_attention(base)
        revert = "wrap"
    else:
        revert = None
    try:
        yield
    finally:
        if revert == "unwrap":
            restore_original_attention(base)
        elif revert == "wrap":
            _wrap()


@contextlib.contextmanager
def dense_attention(model: Any, knobs: Dict[str, Any], cache_location: str = "vram"):
    """Run the body with original dense attention, then restore the starting state (thin wrapper
    over :func:`attention_mode` kept for the benchmark script's import)."""
    with attention_mode(model, knobs, routed=False, cache_location=cache_location):
        yield


@torch.no_grad()
def _avg_loss(model: Any, blocks: torch.Tensor, device: torch.device,
              compute_dtype: torch.dtype, routed: bool, loss_chunk_size: int,
              stock_base: bool = False, eval_seg_len: int = 0) -> float:
    """Mean next-token CE over the held-out blocks (one block at a time to bound VRAM).

    ``stock_base`` runs under ``model.disable_adapter()``, which zeroes the LoRA contribution so
    the forward uses the *un-fine-tuned* 4-bit base weights — the reference for "did fine-tuning
    help?" with no second model to load (PEFT just toggles a flag on the same LoRA layers).

    ``eval_seg_len > 0`` streams the *routed* pass in segments (see ``_segmented_eval_loss``) so the
    full-length routed assembly never OOMs on a small card; ``0`` (default) keeps the single
    full-sequence forward.  Ignored on the dense/stock passes (those are already windowed).
    """
    adapter_off = model.disable_adapter() if stock_base else contextlib.nullcontext()
    total = 0.0
    with adapter_off:
        for i in range(blocks.shape[0]):
            block = blocks[i : i + 1].to(device)
            if routed:
                reset_routers(model)  # the router store is stateful; start each block clean
            if routed and 0 < eval_seg_len < block.shape[1]:
                total += _segmented_eval_loss(model, block, loss_chunk_size, eval_seg_len, compute_dtype)
                continue
            with torch.autocast("cuda", dtype=compute_dtype):
                loss = chunked_causal_lm_loss(model, block, block, loss_chunk_size, train=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss on block {i}")
            total += loss.item()
    return total / max(1, blocks.shape[0])


@torch.no_grad()
def evaluate(model: Any, val_blocks: torch.Tensor, device: torch.device,
             compute_dtype: torch.dtype, knobs: Dict[str, Any], opt_step: int,
             loss_chunk_size: int = 512, cache_location: str = "vram",
             dense_eval_len: int = 0, eval_seg_len: int = 0,
             attn_mode: str = "routed") -> Dict[str, float]:
    """Three-way perplexity comparison on the held-out blocks, all on the *same* model object:

    * **routed** — the fine-tuned adapter with active routed sparse attention (the deploy regime).
    * **dense_adapter** — the *same* trained weights under plain dense attention (isolates the
      cost of routing alone).
    * **stock_base** — the un-fine-tuned 4-bit base under dense attention (LoRA disabled): the
      reference for whether fine-tuning improved perplexity on the novel.

    Two deltas are reported: ``routed − stock_base`` (fine-tuning benefit, should go negative as
    training progresses) and ``routed − dense_adapter`` (routing cost at the current weights).

    The dense baselines run a full S²-attention forward, which OOMs at long seq_len on a small card
    (16384 wants a 16 GiB attention matrix — the exact regime routing exists to avoid).  Set
    ``dense_eval_len`` to run them (and a matching routed reference, so the two deltas stay
    apples-to-apples) on the last ``dense_eval_len`` tokens of each block instead; ``0`` (default)
    uses the whole block (== seq_len).  ``routed_loss`` stays the full-length deploy metric and
    drives best-checkpoint selection regardless.
    """
    was_training = model.training
    model.eval()
    # Window the dense baselines (and the matching routed reference) to a length the GPU can fit a
    # full S² forward for.  ponytail: tail slice — the last dense_eval_len tokens carry the most
    # routed context anyway.  dense_eval_len<=0 (default) ⇒ the window is the whole block.
    win = val_blocks.shape[1] if dense_eval_len <= 0 else min(int(dense_eval_len), val_blocks.shape[1])
    dense_blocks = val_blocks if win == val_blocks.shape[1] else val_blocks[:, -win:]
    try:
        # Routed pass — wrap on demand so this works for a dense-trained model too (the CM restores
        # the starting state).  Measure the routed attention's real-token budget here only.  On a
        # dense run this wrap is *extra* VRAM on top of the dense model, so tolerate OOM: the routed
        # metric is a cross-check there, while the dense deploy loss (below) is what matters.
        try:
            with attention_mode(model, knobs, routed=True, cache_location=cache_location):
                set_token_stats(model, True)
                routed_loss = _avg_loss(model, val_blocks, device, compute_dtype, routed=True,
                                        loss_chunk_size=loss_chunk_size, eval_seg_len=eval_seg_len)
                tok_stats = read_token_stats(model)
                set_token_stats(model, False)
                routed_ref = (routed_loss if win == val_blocks.shape[1]
                              else _avg_loss(model, dense_blocks, device, compute_dtype, routed=True,
                                             loss_chunk_size=loss_chunk_size, eval_seg_len=eval_seg_len))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            routed_loss = routed_ref = float("nan")
            tok_stats = None
            print(f"[val step {opt_step}] routed pass OOM (on-demand router wrap) — "
                  f"reporting routed as NaN; dense/stock metrics stand")
        try:
            with attention_mode(model, knobs, routed=False, cache_location=cache_location):
                dense_loss = _avg_loss(model, dense_blocks, device, compute_dtype, routed=False, loss_chunk_size=loss_chunk_size)
                stock_loss = _avg_loss(model, dense_blocks, device, compute_dtype, routed=False,
                                       loss_chunk_size=loss_chunk_size, stock_base=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            dense_loss = stock_loss = routed_ref = float("nan")
            print(f"[val step {opt_step}] dense baselines OOM even at window={win} "
                  f"(full S² forward) — reporting dense/stock as NaN; routed metric stands")
    finally:
        if was_training:
            model.train()

    # Which loss drives best-checkpoint selection: the deploy regime you trained in.
    primary_loss = dense_loss if attn_mode == "dense" else routed_loss

    metrics = {
        "routed_loss": routed_loss,
        "routed_ppl": math.exp(min(routed_loss, 30.0)),
        "dense_loss": dense_loss,
        "dense_ppl": math.exp(min(dense_loss, 30.0)),
        "stock_loss": stock_loss,
        "stock_ppl": math.exp(min(stock_loss, 30.0)),
        "primary_loss": primary_loss,  # routed_loss (routed mode) or dense_loss (dense mode)
        # Deltas use routed_ref (routed loss on the SAME windowed blocks as the dense baselines) so
        # the comparison is apples-to-apples; routed_loss above is the full-length deploy metric.
        "dense_eval_len": win,
        "finetune_gain": routed_ref - stock_loss,   # < 0 ⇒ fine-tuning helped
        "routing_cost": routed_ref - dense_loss,     # routed vs same-weights dense
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
    win_note = "" if win == val_blocks.shape[1] else f" @{win}-tok window"
    deploy_note = f" [deploy: {'dense' if attn_mode == 'dense' else 'routed'}]"
    print(
        f"[val step {opt_step}] over {val_blocks.shape[0]} blocks{deploy_note}\n"
        f"    routed(adapter) loss={metrics['routed_loss']:.4f} ppl={metrics['routed_ppl']:.3f} (full {val_blocks.shape[1]} tok)\n"
        f"    dense(adapter)  loss={metrics['dense_loss']:.4f} ppl={metrics['dense_ppl']:.3f}{win_note}\n"
        f"    stock(base)     loss={metrics['stock_loss']:.4f} ppl={metrics['stock_ppl']:.3f}{win_note}\n"
        f"    finetune_gain(routed-stock)={metrics['finetune_gain']:+.4f} | "
        f"routing_cost(routed-dense)={metrics['routing_cost']:+.4f}{win_note}"
        + attended_line
    )
    return metrics


def _maybe_save_best(model: Any, output_dir: str, metrics: Dict[str, float], best_val: float,
                     opt_step: int) -> float:
    """Save the adapter to ``<output_dir>/best`` whenever the held-out deploy loss improves.

    The deploy loss is the regime you trained in (``primary_loss``: routed for an HGA run, dense
    for a dense run).  Domain-fit on a single novel can start overfitting late, so the
    lowest-held-out-loss adapter is kept separately from the always-overwritten final one.
    Returns the (possibly updated) best.
    """
    val = metrics["primary_loss"]
    if val < best_val:
        best_dir = os.path.join(output_dir, "best")
        model.save_pretrained(best_dir)
        print(f"[best] new best held-out deploy loss={val:.4f} (was {best_val:.4f}) -> {best_dir} (step {opt_step})")
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

    if getattr(args, "attn_mode", "routed") == "dense":
        # Dense baseline: skip the router surgery entirely so the QLoRA adapters injected next
        # train under the model's original causal attention.  This is the "same LoRA, same data,
        # dense attention" counterpart to the routed run; the three-way validation re-wraps the
        # model on demand to still report the routed (HGA) metric.  No .orig. adapter-key prefix.
        return model, 0

    # Routed attention surgery first: the wrapper holds q/k/v/o by reference, so the LoRA
    # layers injected next are exactly what the router calls -> adapters train through routing.
    n = replace_qwen_attention_with_router(
        model, cache_location=getattr(args, "cache_location", "vram"), **routing_defaults()
    )
    if n == 0:
        raise RuntimeError("No attention layers were wrapped; check the model architecture.")

    # Training/benchmark always run one full-sequence forward from position 0, which reads nothing
    # from the router KV store (the routed output is fully determined by assemble_routed_kv).
    # Disabling store seeding therefore cannot change the attention result; it only drops the
    # detached KV copy decode would need and makes the routed forward stateless -- which is what
    # lets gradient checkpointing recompute it safely in backward (the documented upgrade path).
    set_populate_store(model, False)
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


def _segmented_backward(
    model: Any, input_ids: torch.Tensor, loss_chunk_size: int, seg_len: int,
    scaler: Any, accum: int, compute_dtype: torch.dtype,
) -> float:
    """Blocked forward+backward over ``seg_len`` segments — the author's inference loop + backward.

    Mirrors ``chat_qwen30b_fp8._generate_iter``'s blocked prefill: the sequence is fed in
    ``seg_len`` blocks with growing absolute ``position_ids`` and ``populate_store=True``, so the
    routed KV store accumulates across segments (old K/V resident in the ``--cache-location`` tier,
    e.g. RAM).  Each segment runs its own backward and the boundary is committed (live windows
    detached), so the autograd graph — and therefore VRAM — stays bounded to one segment instead
    of the whole sequence.  Cross-segment gradient is truncated (TBPTT): a segment reads the past
    as detached, stop-gradient KV from the store.

    Returns the mean next-token CE over the sequence (for logging), not the back-scaled value.
    """
    S = input_ids.shape[1]
    device = input_ids.device
    n_seg = (S + seg_len - 1) // seg_len
    ce_sum = 0.0
    for s in range(0, S, seg_len):
        e = min(s + seg_len, S)
        seg = input_ids[:, s:e]
        pos = torch.arange(s, e, device=device).unsqueeze(0)
        with torch.autocast("cuda", dtype=compute_dtype):
            ce = chunked_causal_lm_loss(model, seg, seg, loss_chunk_size, train=True, position_ids=pos)
        if not torch.isfinite(ce):
            raise RuntimeError(f"Non-finite loss on segment [{s}:{e}]: {ce.item()}")
        # 1/n_seg keeps the summed-gradient magnitude on par with a single full-sequence mean.
        scaler.scale(ce / (n_seg * accum)).backward()
        commit_routers(model)  # freeze this segment as the next segment's stop-gradient prefix
        ce_sum += ce.item()
    return ce_sum / n_seg


@torch.no_grad()
def _segmented_eval_loss(
    model: Any, input_ids: torch.Tensor, loss_chunk_size: int, seg_len: int,
    compute_dtype: torch.dtype,
) -> float:
    """Blocked no-grad forward over ``seg_len`` segments — the eval twin of ``_segmented_backward``.

    The routed validation pass otherwise runs ONE full-sequence forward, whose ``assemble_routed_kv``
    gathers every query's chunks at once and OOMs on a small card at long ``seq_len`` (the exact
    regime routing exists to avoid).  Streaming the sequence in ``seg_len`` blocks with growing
    absolute ``position_ids`` and ``populate_store=True`` lets the routed KV store accumulate old
    K/V in the ``--cache-location`` tier (e.g. RAM), so each forward only assembles ``seg_len``
    queries and VRAM stays bounded to one segment.  No backward, no scaler: ``commit_routers`` here
    just advances the store's committed prefix between segments.

    ponytail: segment mean (not token-weighted) matches ``_segmented_backward``; equal-length
    segments make it ~exact.  Upgrade path = return (sum, count) from ``chunked_causal_lm_loss``.
    """
    S = input_ids.shape[1]
    device = input_ids.device
    n_seg = (S + seg_len - 1) // seg_len
    ce_sum = 0.0
    set_populate_store(model, True)  # seed the store so segments stream through the cache tier
    try:
        for s in range(0, S, seg_len):
            e = min(s + seg_len, S)
            seg = input_ids[:, s:e]
            pos = torch.arange(s, e, device=device).unsqueeze(0)
            with torch.autocast("cuda", dtype=compute_dtype):
                ce = chunked_causal_lm_loss(model, seg, seg, loss_chunk_size, train=False, position_ids=pos)
            if not torch.isfinite(ce):
                raise RuntimeError(f"Non-finite eval loss on segment [{s}:{e}]: {ce.item()}")
            commit_routers(model)  # advance the committed prefix for the next segment
            ce_sum += ce.item()
    finally:
        set_populate_store(model, False)  # restore the stateless single-forward default
    return ce_sum / n_seg


def _validate_seg_len(name: str, seg_len: int, seq_len: int, chunk_size: int) -> None:
    """Shared check for a streaming segment length: must tile ``seq_len`` in whole chunks."""
    if seg_len <= 0:
        return
    if seq_len % seg_len != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of {name} ({seg_len}).")
    if seg_len % chunk_size != 0:
        raise ValueError(f"{name} ({seg_len}) must be a multiple of chunk_size ({chunk_size}).")


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
    # Segmented streaming training (the author's blocked-inference pattern): each segment must be a
    # whole number of chunks and tile seq_len exactly, and batching is single-sequence so the store
    # state belongs to one sequence at a time.  The routed eval segment length (weak-card opt-in)
    # shares the same tiling constraints.
    _validate_seg_len("--train-seg-len", args.train_seg_len, args.seq_len, chunk_size)
    _validate_seg_len("--eval-seg-len", args.eval_seg_len, args.seq_len, chunk_size)
    if args.attn_mode == "dense" and (args.train_seg_len > 0 or args.eval_seg_len > 0):
        raise ValueError(
            "--train-seg-len/--eval-seg-len stream through the routed KV store, which --attn-mode "
            "dense does not build; drop them for the dense baseline run."
        )
    n_seg = (args.seq_len // args.train_seg_len) if args.train_seg_len > 0 else 1
    if n_seg > 1 and args.batch_size != 1:
        raise ValueError("Segmented training (--train-seg-len < seq_len) requires --batch-size 1.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"[setup] model={args.model} seq_len={args.seq_len} dtype={compute_dtype} device={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    master_train_blocks, val_blocks = build_master_train_val_blocks(args, tokenizer)

    # FineWeb is streamed fresh every epoch from one persistent, seeded-shuffled stream so the
    # web-text regulariser keeps real diversity instead of replaying a single fixed up-front draw.
    # The per-epoch block count is constant (master + master*ratio), so the LR schedule length is
    # known up front; only one epoch's FineWeb is resident at a time (memory bounded to one epoch).
    if args.fineweb_ratio < 0:
        raise ValueError("--fineweb-ratio must be >= 0")
    target_fineweb_blocks = master_train_blocks.shape[0] * args.fineweb_ratio
    fineweb_iter = make_fineweb_iter(args, tokenizer) if target_fineweb_blocks > 0 else None
    blocks_per_epoch = master_train_blocks.shape[0] + target_fineweb_blocks
    if fineweb_iter is not None:
        print(f"[data] per-epoch fresh mix: {master_train_blocks.shape[0]} Master + "
              f"{target_fineweb_blocks} fresh FineWeb = {blocks_per_epoch} blocks/epoch "
              f"(Master:FineWeb = 1:{args.fineweb_ratio}, shuffle_buffer={args.fineweb_shuffle_buffer})")
    else:
        print(f"[data] training: {blocks_per_epoch} Master blocks/epoch, FineWeb disabled")

    model, n_wrapped = build_model(args, compute_dtype)
    if args.attn_mode == "dense":
        print("[model] dense baseline: router surgery skipped; LoRA trains under original attention")
    else:
        print(f"[model] wrapped {n_wrapped} attention layers with the router")
    apply_segment_mode(model, n_seg)
    if n_seg > 1:
        print(f"[train] segmented streaming: {n_seg} x {args.train_seg_len}-token segments "
              f"(cache_location={args.cache_location}); old K/V offloaded to the routed store")
    model.print_trainable_parameters()
    model.train()

    micro_per_epoch = blocks_per_epoch // args.batch_size  # DataLoader(drop_last=True) length
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
        # Fresh FineWeb draw for this epoch (the persistent stream keeps advancing); master is
        # fixed.  Rebuilding the loader also reshuffles the master/FineWeb interleave.
        if fineweb_iter is not None:
            epoch_blocks = torch.cat(
                [master_train_blocks,
                 pull_fineweb_blocks(fineweb_iter, target_fineweb_blocks, label=f"epoch {epoch + 1}")],
                dim=0,
            )
        else:
            epoch_blocks = master_train_blocks
        loader = DataLoader(TensorDataset(epoch_blocks), batch_size=args.batch_size,
                            shuffle=True, drop_last=True)
        optimizer.zero_grad(set_to_none=True)
        for i, (batch,) in enumerate(loader):
            reset_routers(model)  # store stays bounded to one sequence
            batch = batch.to(device)
            if n_seg > 1:
                # Author's blocked-inference path: stream the sequence through the cache tier in
                # seg_len segments, each with its own backward (TBPTT) — VRAM bounded to a segment.
                running += _segmented_backward(model, batch, args.loss_chunk_size, args.train_seg_len,
                                               scaler, args.accum, compute_dtype)
            else:
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
                                           loss_chunk_size=args.loss_chunk_size, cache_location=args.cache_location,
                                           dense_eval_len=args.dense_eval_len, eval_seg_len=args.eval_seg_len,
                                           attn_mode=args.attn_mode)
                        best_val = _maybe_save_best(model, args.output_dir, metrics, best_val, opt_step)
                    ckpt_dir = os.path.join(args.output_dir, "last")
                    model.save_pretrained(ckpt_dir)
                    print(f"[save] adapter -> {ckpt_dir} (step {opt_step})")
                    apply_segment_mode(model, n_seg)  # evaluate() re-wrapped attention; restore mode
                    set_token_stats(model, True)  # evaluate() re-wraps attention; resume the step-log tally

                if opt_step >= total_opt_steps:
                    done = True
                    break

    # Final validation always runs (independent of --save-every) so every run ends with the
    # base-vs-fine-tuned table, and the best adapter reflects the last weights too.
    if val_blocks is not None:
        metrics = evaluate(model, val_blocks, device, compute_dtype, routing_defaults(), opt_step,
                           loss_chunk_size=args.loss_chunk_size, cache_location=args.cache_location,
                           dense_eval_len=args.dense_eval_len, eval_seg_len=args.eval_seg_len,
                           attn_mode=args.attn_mode)
        best_val = _maybe_save_best(model, args.output_dir, metrics, best_val, opt_step)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] final adapter -> {args.output_dir}")
    if val_blocks is not None and math.isfinite(best_val):
        print(f"[done] best held-out deploy loss={best_val:.4f} -> {os.path.join(args.output_dir, 'best')}")
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
    if args.val_blocks <= 0:
        raise ValueError("--val-blocks must be > 0 for --evaluate-only.")
    _validate_seg_len("--eval-seg-len", args.eval_seg_len, args.seq_len, chunk_size)

    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    blocks = build_blocks(load_text(args.val_data_path), tokenizer, args.seq_len)
    if blocks.shape[0] < args.val_blocks:
        raise ValueError(f"Need at least {args.val_blocks} validation blocks; got {blocks.shape[0]}.")
    val_blocks = blocks[-args.val_blocks :].clone()
    print(f"[eval] {adapter_dir} on the last {val_blocks.shape[0]} pure validation blocks of {args.val_data_path} (seq_len={args.seq_len})")

    base, n = load_routed_base(args, compute_dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    wrap_note = (f"wrapped {n} attention layers" if args.attn_mode != "dense"
                 else "dense (un-surgered) base")
    print(f"[eval] {wrap_note}; loaded adapter from {adapter_dir}")
    metrics = evaluate(model, val_blocks, device, compute_dtype, knobs, opt_step=0,
                       loss_chunk_size=args.loss_chunk_size, cache_location=args.cache_location,
                       dense_eval_len=args.dense_eval_len, eval_seg_len=args.eval_seg_len,
                       attn_mode=args.attn_mode)
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
    block = build_blocks(load_text(args.data_path), tokenizer, args.seq_len)[:1].to(device)  # one fixed block

    model, n = build_model(args, compute_dtype)
    expected = 0 if args.attn_mode == "dense" else int(model.config.num_hidden_layers)
    assert n == expected, f"wrapped {n} != expected {expected} (attn_mode={args.attn_mode})"
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
                       loss_chunk_size=args.loss_chunk_size, attn_mode=args.attn_mode)
    assert all(math.isfinite(metrics[k]) for k in ("routed_loss", "dense_loss", "stock_loss")), "non-finite validation loss"
    base = model.get_base_model()
    want_routed = args.attn_mode != "dense"
    assert all(
        isinstance(layer.self_attn, QwenRoutedAttention) == want_routed
        for layer in base.model.layers
    ), f"attention not restored to {'routed' if want_routed else 'dense'} after validation"
    print(
        f"[smoke] OK: validation routed={metrics['routed_loss']:.3f} dense={metrics['dense_loss']:.3f} "
        f"stock={metrics['stock_loss']:.3f} gain={metrics['finetune_gain']:+.3f}"
    )


# =================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data-path", default=os.path.join(_REPO_ROOT, "TrainData", "The-Master-and-Margarita.txt"))
    p.add_argument("--val-data-path", default=os.path.join(_REPO_ROOT, "TrainData", "The-Master-and-Margarita.txt"),
                   help="pure validation text path; defaults to Master and Margarita.  If it is the "
                        "same file as --data-path, the held-out tail is removed from Master training "
                        "before FineWeb blocks are added.")
    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "checkpoints"))
    p.add_argument("--attn-mode", choices=("routed", "dense"), default="routed",
                   help="train the QLoRA adapters under 'routed' HGA sparse attention (default) or "
                        "'dense' original attention — the apples-to-apples same-LoRA/same-data "
                        "baseline the author asked for.  Dense skips the router surgery (saves plain, "
                        "no-'.orig.' adapter keys) and is incompatible with --train-seg-len; the "
                        "three-way validation still re-wraps on demand to report routed/dense/stock.")
    # Recommended domain-fit config for a 16 GB Turing card with gradient checkpointing on.
    # 4096 = 64 chunks (routing engages strongly) and fits (~4 GB peak); longer blocks mean fewer
    # blocks, so accum is lowered to keep enough optimizer updates per epoch.
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--cache-location", choices=("vram", "ram", "fs"), default="vram",
                   help="cold KV record tier for the routed store: 'vram' (all on GPU), 'ram' (host "
                        "memory, routed chunks pulled to VRAM), or 'fs' (RAM-bounded page cache with "
                        "disk spillover).  Only reduces training VRAM together with --train-seg-len "
                        "(the store is otherwise untouched on a single full-sequence forward).")
    p.add_argument("--train-seg-len", type=int, default=0,
                   help="segment length for streaming long-context training (the author's blocked-"
                        "inference pattern + per-segment backward): the sequence is fed in chunks of "
                        "this many tokens with the routed KV store accumulating across segments, so "
                        "old K/V lives in the --cache-location tier (use 'ram') and VRAM stays bounded "
                        "to one segment.  Must divide --seq-len and be a multiple of chunk_size (64).  "
                        "0 (default) = single full-sequence forward (no streaming).  Requires "
                        "--batch-size 1.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--loss-chunk-size", type=int, default=512, help="tokens per lm_head+CE slice; smaller = less VRAM in the vocab dim")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full schedule")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=20, help="save the adapter and run three-way validation every N optimizer steps (0 disables periodic saves; the final validation still runs)")
    p.add_argument("--val-blocks", type=int, default=6, help="held-out pure validation blocks from the end of --val-data-path")
    p.add_argument("--dense-eval-len", type=int, default=0,
                   help="cap the dense validation baselines (and the matching routed reference for the "
                        "two deltas) to the last N tokens of each held-out block; the full S² dense "
                        "forward OOMs at long seq_len on a small card (16384 wants ~16 GiB), so set e.g. "
                        "4096 to keep the comparison alive.  0 (default) = use the whole block (seq_len). "
                        "routed_loss stays full-length and drives best-checkpoint selection regardless.")
    p.add_argument("--eval-seg-len", type=int, default=0,
                   help="stream the routed validation pass in segments of this many tokens (weak-card "
                        "opt-in): the full-length routed forward assembles every query's chunks at once "
                        "and OOMs at long seq_len on a small card, so feed it in blocks with the KV store "
                        "accumulating across them (use --cache-location ram) to bound VRAM to one segment. "
                        "Must divide --seq-len and be a multiple of chunk_size (64).  0 (default) = single "
                        "full-sequence routed eval (assumes a card with enough VRAM; no flag needed).")
    p.add_argument("--fineweb-ratio", type=int, default=FINEWEB_RATIO,
                   help="FineWeb blocks per one Master train block; 10 means a 1:10 Master:FineWeb "
                        "training mix, 0 disables FineWeb and restores local-text-only training.")
    p.add_argument("--fineweb-shuffle-buffer", type=int, default=1000,
                   help="buffer_size for the streaming FineWeb .shuffle(--seed): each epoch draws a "
                        "seeded-random window of the shard instead of the deterministic stream head. "
                        "0 disables shuffling (deterministic, lowest memory).")
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
