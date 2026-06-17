#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from ExistingModelFineTuning.HierarchicalGlobalAttentionFusedExactQ import GlobalAttention


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_dtype(dtype: str) -> torch.dtype | str:
    dtype = dtype.lower()
    if dtype == "auto":
        return "auto"
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported dtype {dtype}; use auto/bf16/fp16/fp32")
    return table[dtype]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_report(model: Optional[nn.Module] = None) -> Dict[str, float]:
    rep: Dict[str, float] = {}
    if model is not None and hasattr(model, "get_memory_footprint"):
        try:
            rep["model_footprint_gb"] = float(model.get_memory_footprint()) / 1024**3
        except Exception:
            pass
    if torch.cuda.is_available():
        rep.update(
            cuda_allocated_gb=torch.cuda.memory_allocated() / 1024**3,
            cuda_reserved_gb=torch.cuda.memory_reserved() / 1024**3,
            cuda_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
            cuda_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
        )
    return rep


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def unwrap_compile_forward(model: nn.Module, original_forward: Optional[Any]) -> None:
    if original_forward is not None:
        model.forward = original_forward  # type: ignore[method-assign]


def compile_forward_inplace(model: nn.Module, args: argparse.Namespace, tag: str) -> Optional[Any]:
    """Compile model.forward in place, so generate() also calls compiled forward."""
    if not args.compile:
        return None
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")

    try:
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, args.dynamo_cache_size_limit)
    except Exception:
        pass

    log(f"[compile] compiling {tag} forward with torch.compile(mode={args.compile_mode})")
    original_forward = model.forward
    model.forward = torch.compile(  # type: ignore[method-assign]
        original_forward,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
        dynamic=args.compile_dynamic,
    )
    return original_forward


def first_param_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    try:
        p = next(module.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


# -----------------------------------------------------------------------------
# Cache utilities for chunked loss
# -----------------------------------------------------------------------------


_WARNED_CACHE_NOT_GROWING = False


def cache_seq_len(past_key_values: Any) -> Optional[int]:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) > 0 and torch.is_tensor(first[0]):
            return int(first[0].shape[-2])
    if hasattr(past_key_values, "key_cache"):
        kc = getattr(past_key_values, "key_cache")
        if isinstance(kc, list) and kc and torch.is_tensor(kc[0]):
            return int(kc[0].shape[-2])
    if hasattr(past_key_values, "layers"):
        layers = getattr(past_key_values, "layers")
        if layers:
            layer0 = layers[0]
            for attr in ("keys", "key_cache"):
                if hasattr(layer0, attr):
                    t = getattr(layer0, attr)
                    if torch.is_tensor(t):
                        return int(t.shape[-2])
                    if isinstance(t, list) and t and torch.is_tensor(t[0]):
                        return int(t[0].shape[-2])
    return None


def detach_cache_inplace(past_key_values: Any) -> Any:
    """Best-effort detach for HF DynamicCache / legacy tuple caches."""
    if past_key_values is None:
        return None
    if torch.is_tensor(past_key_values):
        return past_key_values.detach()
    if isinstance(past_key_values, tuple):
        return tuple(detach_cache_inplace(x) for x in past_key_values)
    if isinstance(past_key_values, list):
        return [detach_cache_inplace(x) for x in past_key_values]

    for attr in ("key_cache", "value_cache"):
        if hasattr(past_key_values, attr):
            cache = getattr(past_key_values, attr)
            if isinstance(cache, list):
                for i, t in enumerate(cache):
                    if torch.is_tensor(t):
                        cache[i] = t.detach()

    if hasattr(past_key_values, "layers"):
        for layer in getattr(past_key_values, "layers"):
            for attr in ("keys", "values", "key_cache", "value_cache"):
                if hasattr(layer, attr):
                    t = getattr(layer, attr)
                    if torch.is_tensor(t):
                        setattr(layer, attr, t.detach())
                    elif isinstance(t, list):
                        for i, v in enumerate(t):
                            if torch.is_tensor(v):
                                t[i] = v.detach()
    return past_key_values


# -----------------------------------------------------------------------------
# Direct attention replacement
# -----------------------------------------------------------------------------


def copy_linear(dst: nn.Linear, src: nn.Linear, name: str) -> None:
    if tuple(dst.weight.shape) != tuple(src.weight.shape):
        raise ValueError(
            f"{name}.weight shape mismatch: custom {tuple(dst.weight.shape)} vs "
            f"original {tuple(src.weight.shape)}. For Qwen3-0.6B expected approximately: "
            "q_proj=[2048,1024], k_proj/v_proj=[1024,1024], o_proj=[1024,2048]. "
            "Your GlobalAttention must create projections with exactly the same shapes."
        )
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None:
            if src.bias is not None:
                if tuple(dst.bias.shape) != tuple(src.bias.shape):
                    raise ValueError(f"{name}.bias shape mismatch: {dst.bias.shape} vs {src.bias.shape}")
                dst.bias.copy_(src.bias)
            else:
                dst.bias.zero_()


def validate_qwen_forward_signature(attn: nn.Module, strict: bool) -> None:
    """
    Warn/fail if GlobalAttention.forward does not look Qwen-compatible.
    This is only signature-level validation; a tiny warmup forward later catches runtime errors.
    """
    sig = inspect.signature(attn.forward)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # Qwen may pass hidden_states as keyword; old forward(x, rotary_data, **kw) is not enough.
    required_or_kwargs = ["hidden_states", "position_embeddings", "attention_mask"]
    missing = [k for k in required_or_kwargs if k not in params and not has_kwargs]

    # If **kw exists but hidden_states does not, Python accepts it, but your function must read it.
    suspicious = has_kwargs and "hidden_states" not in params

    if missing or suspicious:
        msg = (
            "GlobalAttention.forward may not be Qwen-compatible. Recommended signature:\n"
            "  forward(self, hidden_states, position_embeddings=None, attention_mask=None, "
            "position_ids=None, past_key_values=None, use_cache=None, cache_position=None, **kw)\n"
            f"Current signature: {sig}"
        )
        if strict:
            raise TypeError(msg)
        log("[warn] " + msg)


def make_global_attention_from_qwen(original_attn: nn.Module, config: Any, layer_idx: int, args: argparse.Namespace) -> GlobalAttention:
    hidden_size = int(getattr(config, "hidden_size"))
    num_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    attention_dropout = float(getattr(config, "attention_dropout", 0.0))

    if args.bias_mode == "original":
        use_bias_q = getattr(original_attn.q_proj, "bias", None) is not None
        use_bias_k = getattr(original_attn.k_proj, "bias", None) is not None
        use_bias_v = getattr(original_attn.v_proj, "bias", None) is not None
        use_bias_o = getattr(original_attn.o_proj, "bias", None) is not None
    else:
        use_bias_q = args.use_bias_q
        use_bias_k = args.use_bias_k
        use_bias_v = args.use_bias_v
        use_bias_o = args.use_bias_o

    extra_kwargs = json.loads(args.global_attention_extra_kwargs or "{}")
    attn = GlobalAttention(
        d_model=hidden_size,
        nhead=num_heads,
        kv_heads=num_kv_heads,
        dropout=attention_dropout if args.dropout < 0 else args.dropout,
        use_bias_q=use_bias_q,
        use_bias_k=use_bias_k,
        use_bias_v=use_bias_v,
        use_bias_o=use_bias_o,
        **extra_kwargs,
    )

    # Attach Qwen-compatible metadata directly to the custom module.
    # These are useful for custom forward/cache/generation logic and mirror HF attention attrs.
    attn.config = config
    attn.layer_idx = layer_idx
    attn.layer_type = getattr(original_attn, "layer_type", None)
    attn.hidden_size = hidden_size
    attn.num_heads = num_heads
    attn.num_key_value_heads = num_kv_heads
    attn.num_key_value_groups = num_heads // num_kv_heads
    attn.head_dim = head_dim
    attn.scaling = head_dim**-0.5
    attn.attention_dropout = attention_dropout
    attn.is_causal = True
    attn.sliding_window = getattr(original_attn, "sliding_window", None)

    # Qwen3 has q/k RMSNorm inside attention. Register directly on GlobalAttention.
    if hasattr(original_attn, "q_norm"):
        attn.q_norm = copy.deepcopy(original_attn.q_norm)
    if hasattr(original_attn, "k_norm"):
        attn.k_norm = copy.deepcopy(original_attn.k_norm)

    # Copy q/k/v/o weights directly into GlobalAttention linears.
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(attn, proj_name):
            raise AttributeError(f"GlobalAttention has no .{proj_name}; required for weight copying")
        copy_linear(getattr(attn, proj_name), getattr(original_attn, proj_name), proj_name)

    validate_qwen_forward_signature(attn, strict=args.strict_forward_signature)
    return attn


def iter_attention_slots(model: nn.Module) -> Iterable[Tuple[nn.Module, str, nn.Module, int]]:
    """Yield (parent_module, child_name, attention_module, layer_idx)."""
    # Common HF decoder layout: model.model.layers[i].self_attn.
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if all(hasattr(attn, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    yield layer, "self_attn", attn, i
        return

    # Generic fallback.
    idx = 0
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in {"self_attn", "attention", "attn"} and all(
                hasattr(child, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            ):
                yield module, child_name, child, idx
                idx += 1


def replace_attention_modules(model: nn.Module, args: argparse.Namespace) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model has no config")

    count = 0
    slots = list(iter_attention_slots(model))
    if not slots:
        raise RuntimeError("No attention modules with q_proj/k_proj/v_proj/o_proj were found")

    for parent, child_name, original_attn, layer_idx in slots:
        dev, dtype = first_param_device_dtype(original_attn)
        custom_attn = make_global_attention_from_qwen(original_attn, config, layer_idx, args)
        custom_attn.to(device=dev, dtype=dtype)
        setattr(parent, child_name, custom_attn)
        count += 1

    return count


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


class TokenBatcher:
    def __init__(self, token_buffer: torch.Tensor, context_len: int, batch_size: int, seed: int = 0) -> None:
        if token_buffer.dim() != 1:
            raise ValueError("token_buffer must be a 1D tensor")
        if token_buffer.numel() < context_len + 2:
            raise ValueError("token_buffer too small")
        self.tokens = token_buffer.cpu().long()
        self.context_len = context_len
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def next_batch(self) -> torch.Tensor:
        max_start = self.tokens.numel() - (self.context_len + 1)
        rows = []
        for _ in range(self.batch_size):
            s = self.rng.randint(0, max_start)
            rows.append(self.tokens[s : s + self.context_len + 1])
        return torch.stack(rows, dim=0)


def text_file_iter(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def get_text_iter(args: argparse.Namespace, split: str) -> Iterable[str]:
    if split == args.train_split and args.train_text_file:
        return text_file_iter(args.train_text_file)
    if split == args.eval_split and args.eval_text_file:
        return text_file_iter(args.eval_text_file)

    if load_dataset is None:
        raise RuntimeError("datasets is not installed; install it or pass --train-text-file/--eval-text-file")

    cfg = args.dataset_config if args.dataset_config not in {"", "none", "None", None} else None
    ds = load_dataset(args.dataset_name, cfg, split=split, streaming=args.streaming_dataset)

    def gen() -> Iterator[str]:
        for row in ds:
            text = row.get(args.text_field, None)
            if isinstance(text, str) and text.strip():
                yield text

    return gen()


def build_token_buffer(tokenizer: Any, args: argparse.Namespace, split: str, needed_tokens: int) -> torch.Tensor:
    log(f"[data] building {split} token buffer: need {needed_tokens:,} tokens")
    ids: List[int] = []
    eos = tokenizer.eos_token_id
    text_iter = get_text_iter(args, split)

    for txt in text_iter:
        encoded = tokenizer(txt, add_special_tokens=False).input_ids
        if encoded:
            ids.extend(encoded)
            if eos is not None:
                ids.append(eos)
        if len(ids) >= needed_tokens:
            break

    if len(ids) < needed_tokens:
        raise RuntimeError(
            f"Not enough tokens for split={split}: got {len(ids):,}, need {needed_tokens:,}. "
            "Use a larger dataset/text file or reduce --context-len/--eval-batches/--token-buffer-tokens."
        )
    log(f"[data] {split}: collected {len(ids):,} tokens")
    return torch.tensor(ids[:needed_tokens], dtype=torch.long)


def make_eval_batches(eval_tokens: torch.Tensor, context_len: int, batch_size: int, num_batches: int) -> List[torch.Tensor]:
    needed = num_batches * batch_size * (context_len + 1)
    if eval_tokens.numel() < needed:
        raise ValueError(f"eval_tokens too small: got {eval_tokens.numel()}, need {needed}")
    batches: List[torch.Tensor] = []
    offset = 0
    for _ in range(num_batches):
        rows = []
        for _ in range(batch_size):
            rows.append(eval_tokens[offset : offset + context_len + 1])
            offset += context_len + 1
        batches.append(torch.stack(rows, dim=0))
    return batches


# -----------------------------------------------------------------------------
# Loss routines
# -----------------------------------------------------------------------------


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def call_model_for_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool,
    past_key_values: Any = None,
) -> Any:
    kwargs: Dict[str, Any] = {"input_ids": input_ids, "use_cache": use_cache}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    # Qwen3 supports logits_to_keep; not all models do.
    try:
        return model(**kwargs, logits_to_keep=0)
    except TypeError:
        return model(**kwargs)


def lm_loss_full(
    model: nn.Module,
    batch_plus_one: torch.Tensor,
    backward: bool = False,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    inp = batch_plus_one[:, :-1]
    tgt = batch_plus_one[:, 1:]
    out = call_model_for_logits(model, inp, use_cache=False)
    loss = ce_loss(out.logits, tgt)
    if backward:
        (loss * loss_scale).backward()
    return loss.detach()


def lm_loss_chunked(
    model: nn.Module,
    batch_plus_one: torch.Tensor,
    chunk_len: int,
    use_cache: bool,
    detach_kv_between_chunks: bool,
    backward: bool = False,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute CE over a context_len sequence while only materializing logits for chunk_len tokens.

    With use_cache=True, each chunk attends to previous chunks. During training,
    detach_kv_between_chunks=True gives truncated BPTT and frees previous graphs.
    Set it false for exact gradients, but memory rises strongly.
    """
    inp_all = batch_plus_one[:, :-1]
    tgt_all = batch_plus_one[:, 1:]
    bsz, seq_len = inp_all.shape
    past = None
    total_tokens = 0
    total_loss_value = 0.0
    graph_loss: Optional[torch.Tensor] = None

    for start in range(0, seq_len, chunk_len):
        end = min(start + chunk_len, seq_len)
        inp = inp_all[:, start:end]
        tgt = tgt_all[:, start:end]

        out = call_model_for_logits(model, inp, use_cache=use_cache, past_key_values=past)
        logits = out.logits
        loss = ce_loss(logits, tgt)
        ntok = tgt.numel()
        total_tokens += ntok
        total_loss_value += float(loss.detach()) * ntok

        if backward:
            weighted = loss * (ntok / (bsz * seq_len)) * loss_scale
            if detach_kv_between_chunks:
                weighted.backward()
            else:
                graph_loss = weighted if graph_loss is None else graph_loss + weighted

        past = getattr(out, "past_key_values", None) if use_cache else None
        if use_cache and past is not None:
            global _WARNED_CACHE_NOT_GROWING
            got_len = cache_seq_len(past)
            if got_len is not None and got_len < end and not _WARNED_CACHE_NOT_GROWING:
                log(
                    "[warn] chunked loss requested cache, but past_key_values length "
                    f"is {got_len} after processing {end} tokens. If this is the custom model, "
                    "GlobalAttention may not update the HF cache; chunked loss will then be truncated."
                )
                _WARNED_CACHE_NOT_GROWING = True

        del logits, out, loss
        if detach_kv_between_chunks and past is not None:
            past = detach_cache_inplace(past)

    if backward and not detach_kv_between_chunks and graph_loss is not None:
        graph_loss.backward()

    return torch.tensor(total_loss_value / max(total_tokens, 1), device=batch_plus_one.device)


def batch_lm_loss(
    model: nn.Module,
    batch_plus_one: torch.Tensor,
    args: argparse.Namespace,
    backward: bool = False,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    if args.loss_chunk_len and args.loss_chunk_len > 0 and args.loss_chunk_len < args.context_len:
        return lm_loss_chunked(
            model=model,
            batch_plus_one=batch_plus_one,
            chunk_len=args.loss_chunk_len,
            use_cache=args.chunked_loss_use_cache,
            detach_kv_between_chunks=args.detach_kv_between_chunks,
            backward=backward,
            loss_scale=loss_scale,
        )
    return lm_loss_full(model, batch_plus_one, backward=backward, loss_scale=loss_scale)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    eval_batches: Sequence[torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, float]:
    model.eval()
    reset_peak_memory()
    sync()
    t0 = time.perf_counter()
    losses = []
    for b in eval_batches:
        b = b.to(device, non_blocking=True)
        loss = batch_lm_loss(model, b, args, backward=False)
        losses.append(float(loss.detach().cpu()))
        del b, loss
    sync()
    elapsed = time.perf_counter() - t0
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    mem = memory_report(model)
    rep = {"loss": mean_loss, "ppl": ppl, "seconds": elapsed, **mem}
    log(f"[eval:{tag}] loss={mean_loss:.6f} ppl={ppl:.3f} sec={elapsed:.2f} mem={mem}")
    return rep


# -----------------------------------------------------------------------------
# Training routines
# -----------------------------------------------------------------------------


def set_requires_grad_for_stage(model: nn.Module, stage: str, train_layernorms: bool = False) -> int:
    for p in model.parameters():
        p.requires_grad_(False)

    train_count = 0
    for name, p in model.named_parameters():
        should_train = False
        if stage == "kv":
            should_train = ("self_attn" in name) and (".k_proj." in name or ".v_proj." in name)
        elif stage == "attn_mlp":
            should_train = ("self_attn" in name) or (".mlp." in name)
            if not train_layernorms and ("q_norm" in name or "k_norm" in name or "layernorm" in name or "norm" in name):
                should_train = False if ".mlp." not in name else should_train
        else:
            raise ValueError(stage)
        if should_train:
            p.requires_grad_(True)
            train_count += p.numel()
    return train_count


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected")
    return torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)


def train_stage(
    model: nn.Module,
    batcher: TokenBatcher,
    device: torch.device,
    args: argparse.Namespace,
    stage: str,
    steps: int,
) -> Dict[str, Any]:
    if steps <= 0:
        return {"steps": 0}

    n_trainable = set_requires_grad_for_stage(model, stage, args.train_layernorms_in_stage2)
    optimizer = make_optimizer(model, args)
    model.train()
    log(f"[train:{stage}] trainable params: {n_trainable:,}; optimizer steps: {steps}")

    reset_peak_memory()
    step_losses: List[float] = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            batch = batcher.next_batch().to(device, non_blocking=True)
            loss = batch_lm_loss(
                model,
                batch,
                args,
                backward=True,
                loss_scale=1.0 / args.grad_accum_steps,
            )
            accum_loss += float(loss.detach().cpu()) / args.grad_accum_steps
            del batch, loss

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        step_losses.append(accum_loss)

        if step == 1 or step % args.log_every == 0 or step == steps:
            sync()
            recent_n = min(len(step_losses), args.log_every)
            recent = sum(step_losses[-recent_n:]) / recent_n
            log(f"[train:{stage}] step={step}/{steps} loss={accum_loss:.6f} recent={recent:.6f}")

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    sync()
    elapsed = time.perf_counter() - t0
    rep = {
        "steps": steps,
        "trainable_params": n_trainable,
        "last_loss": step_losses[-1] if step_losses else None,
        "mean_loss": sum(step_losses) / max(len(step_losses), 1),
        "seconds": elapsed,
        **memory_report(model),
    }
    log(f"[train:{stage}] done: {rep}")
    return rep


# -----------------------------------------------------------------------------
# Generation benchmark
# -----------------------------------------------------------------------------


@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, float]:
    model.eval()
    inputs = tokenizer(args.generation_prompt, return_tensors="pt", add_special_tokens=True).to(device)

    gen_kwargs = dict(
        max_new_tokens=args.generation_tokens,
        do_sample=False,
        use_cache=args.generation_use_cache,
        pad_token_id=tokenizer.eos_token_id,
    )

    for _ in range(args.generation_warmup):
        _ = model.generate(**inputs, **gen_kwargs)
    sync()

    reset_peak_memory()
    times: List[float] = []
    new_tokens: List[int] = []
    for _ in range(args.generation_repeats):
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        new_tokens.append(int(out.shape[-1] - inputs["input_ids"].shape[-1]))
        del out

    total_toks = sum(new_tokens)
    total_sec = sum(times)
    rep = {
        "seconds_mean": total_sec / max(len(times), 1),
        "new_tokens_mean": total_toks / max(len(times), 1),
        "tokens_per_second": total_toks / max(total_sec, 1e-9),
        **memory_report(model),
    }
    log(f"[gen:{tag}] {rep}")
    return rep


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_training_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    tag: str = "final",
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The custom module is not reconstructed by plain AutoModelForCausalLM.
    # To load: instantiate base model, run replace_attention_modules(), then load this state_dict.
    ckpt_path = out_dir / f"custom_attention_{tag}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "note": "Instantiate base model, replace attention with this script, then load model_state_dict.",
        },
        ckpt_path,
    )
    tokenizer.save_pretrained(out_dir / "tokenizer")
    save_json(out_dir / f"metrics_{tag}.json", metrics)
    save_json(out_dir / "run_args.json", vars(args))

    try:
        model.save_pretrained(out_dir / "hf_save_pretrained", safe_serialization=True)
    except Exception as e:
        log(f"[save] save_pretrained failed, custom torch checkpoint is still saved: {e!r}")
    log(f"[save] checkpoint saved to {ckpt_path}")


def maybe_load_custom_checkpoint(model: nn.Module, path: Optional[str]) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    log(f"[resume] loaded {path}; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"[resume] first missing keys: {missing[:10]}")
    if unexpected:
        log(f"[resume] first unexpected keys: {unexpected[:10]}")


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model / custom attention.
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="./qwen3_06b_global_attention_ft")
    p.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume-custom-checkpoint", default=None)

    p.add_argument("--global-attention-extra-kwargs", default="{}", help="JSON dict passed to GlobalAttention constructor")
    p.add_argument("--dropout", type=float, default=-1.0, help="negative means use model config attention_dropout")
    p.add_argument("--bias-mode", default="requested", choices=["requested", "original"])
    p.add_argument("--use-bias-q", type=str2bool, default=True)
    p.add_argument("--use-bias-k", type=str2bool, default=True)
    p.add_argument("--use-bias-v", type=str2bool, default=True)
    p.add_argument("--use-bias-o", type=str2bool, default=False)
    p.add_argument(
        "--strict-forward-signature",
        action="store_true",
        help="Fail before training if GlobalAttention.forward signature does not look Qwen-compatible.",
    )

    # Data.
    p.add_argument("--dataset-name", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--text-field", default="text")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--streaming-dataset", action="store_true")
    p.add_argument("--train-text-file", default=None)
    p.add_argument("--eval-text-file", default=None)
    p.add_argument("--token-buffer-tokens", type=int, default=4_194_304)

    # Context / loss chunking.
    p.add_argument("--context-len", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--loss-chunk-len", type=int, default=2048, help="0/full means materialize full-sequence logits")
    p.add_argument("--chunked-loss-use-cache", type=str2bool, default=True)
    p.add_argument(
        "--detach-kv-between-chunks",
        type=str2bool,
        default=True,
        help="Training memory saver: truncated BPTT across chunks. For exact gradients set false, but memory rises.",
    )

    # Training.
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--stage1-steps", type=int, default=1000)
    p.add_argument("--stage2-steps", type=int, default=1000)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--train-layernorms-in-stage2", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=0)

    # Compile.
    p.add_argument("--compile", type=str2bool, default=True)
    p.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--compile-fullgraph", action="store_true")
    p.add_argument("--compile-dynamic", type=str2bool, default=False)
    p.add_argument("--dynamo-cache-size-limit", type=int, default=256)

    # Generation benchmark.
    p.add_argument("--generation-prompt", default="Give me a concise explanation of rotary position embeddings.")
    p.add_argument("--generation-tokens", type=int, default=128)
    p.add_argument("--generation-repeats", type=int, default=3)
    p.add_argument("--generation-warmup", type=int, default=1)
    p.add_argument("--generation-use-cache", type=str2bool, default=True)

    # Flow switches.
    p.add_argument("--skip-original-bench", action="store_true")
    p.add_argument("--skip-generation-bench", action="store_true")
    p.add_argument("--skip-replaced-pre-ft-loss", action="store_true")
    p.add_argument("--save-after-stage1", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device(args.device or device_name())
    dtype = parse_dtype(args.torch_dtype)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    log(f"[env] device={dev}, dtype={dtype}, model={args.model_name}")
    log(f"[env] context_len={args.context_len}, loss_chunk_len={args.loss_chunk_len}")
    log(f"[data] dataset={args.dataset_name}, config={args.dataset_config}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_needed = args.eval_batches * args.batch_size * (args.context_len + 1)
    train_needed = max(args.token_buffer_tokens, args.batch_size * (args.context_len + 1) * 2)
    eval_tokens = build_token_buffer(tokenizer, args, args.eval_split, eval_needed)
    train_tokens = build_token_buffer(tokenizer, args, args.train_split, train_needed)
    eval_batches = make_eval_batches(eval_tokens, args.context_len, args.batch_size, args.eval_batches)
    batcher = TokenBatcher(train_tokens, args.context_len, args.batch_size, seed=args.seed + 1)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(dev)
    model.config.use_cache = bool(args.generation_use_cache or args.chunked_loss_use_cache)

    metrics: Dict[str, Any] = {"args": vars(args), "memory_after_load": memory_report(model)}
    log(f"[model] loaded; memory={metrics['memory_after_load']}")

    if not args.skip_original_bench:
        original_forward = compile_forward_inplace(model, args, "original")
        with torch.no_grad():
            tiny = eval_batches[0][:1, : min(128, args.context_len + 1)].to(dev)
            _ = lm_loss_full(model, tiny, backward=False)
            del tiny
        metrics["loss_original"] = evaluate_loss(model, eval_batches, dev, args, "original")
        if not args.skip_generation_bench:
            metrics["generation_original"] = benchmark_generation(model, tokenizer, dev, args, "original")
        unwrap_compile_forward(model, original_forward)

    n_replaced = replace_attention_modules(model, args)
    log(f"[replace] replaced {n_replaced} attention modules with direct GlobalAttention instances")
    maybe_load_custom_checkpoint(model, args.resume_custom_checkpoint)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    replaced_forward = compile_forward_inplace(model, args, "direct_global_attention")
    with torch.no_grad():
        tiny = eval_batches[0][:1, : min(128, args.context_len + 1)].to(dev)
        _ = lm_loss_full(model, tiny, backward=False)
        del tiny

    if not args.skip_replaced_pre_ft_loss:
        metrics["loss_replaced_pre_ft"] = evaluate_loss(model, eval_batches, dev, args, "replaced_pre_ft")

    metrics["train_stage1_kv"] = train_stage(model, batcher, dev, args, "kv", args.stage1_steps)
    metrics["loss_after_stage1_kv"] = evaluate_loss(model, eval_batches, dev, args, "after_stage1_kv")
    if args.save_after_stage1:
        save_training_checkpoint(model, tokenizer, args, metrics, tag="after_stage1_kv")

    metrics["train_stage2_attn_mlp"] = train_stage(model, batcher, dev, args, "attn_mlp", args.stage2_steps)
    metrics["loss_final"] = evaluate_loss(model, eval_batches, dev, args, "final")

    if not args.skip_generation_bench:
        metrics["generation_final"] = benchmark_generation(model, tokenizer, dev, args, "final")

    unwrap_compile_forward(model, replaced_forward)
    save_training_checkpoint(model, tokenizer, args, metrics, tag="final")

    log("[done] summary:")
    for key in (
        "loss_original",
        "loss_replaced_pre_ft",
        "loss_after_stage1_kv",
        "loss_final",
        "generation_original",
        "generation_final",
    ):
        if key in metrics:
            log(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
