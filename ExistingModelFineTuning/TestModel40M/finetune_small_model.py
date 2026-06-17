#!/usr/bin/env python3
"""
Fine-tune a SmallLM with HierarchicalAttention initialized from a trained DenseAttention checkpoint.

Typical use:
    python finetune_ha_from_dense_bf16.py \
        --dense-checkpoint speed_run_dense_muon_final.pt \
        --train-scope attention \
        --precision bf16

The checkpoint is expected to come from the matching DenseAttention SmallLM, where
q_proj/k_proj/v_proj/o_proj and the rest of the model have the same parameter names.
"""

import argparse
import contextlib
import gc
import inspect
import logging
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

try:
    from torch.optim import Muon  # PyTorch nightly / recent versions
except Exception:  # pragma: no cover - depends on local PyTorch build
    Muon = None

from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
from transformers import GPT2TokenizerFast

import ExistingModelFineTuning.torch_inductor_patch as path
path.apply()

from ExistingModelFineTuning.HierarchicalGlobalAttentionFusedExactQ import GlobalAttention
#from RotaryGQASDPA_new import RotaryGQASDPA


# -----------------------------------------------------------------------------
# Model / data defaults
# -----------------------------------------------------------------------------
HIDDEN_DIM = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
MAX_LEN = 8192

BATCH_SIZE = 2
ACCUM_STEPS = 3

# Fine-tuning defaults are intentionally lower than pretraining defaults.
TARGET_TOKENS = 200_000_000
LR = 5e-5
MUON_LR = 0.002
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
MIN_LR_RATIO = 0.1

LOG_EVERY = 10
EVAL_EVERY = 250
SAVE_EVERY = 2_000

CAUSALITY_CHECK_EVERY = 1_000
CAUSALITY_CHECK_LEN = 4096
CAUSALITY_CHECK_TOL = 2e-3

CHUNK_SIZE = 64
GROUP_SIZE = 16
TOPK_CHUNKS = 20
TOPK_GROUPS = 32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("ha_finetune_from_dense")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logger("finetune_ha_from_dense.log")


# -----------------------------------------------------------------------------
# Precision / stability helpers
# -----------------------------------------------------------------------------
def resolve_amp_config(
        precision: str,
        device: str,
        require_bf16: bool = False,
) -> Tuple[bool, Optional[torch.dtype]]:
    """Keep parameters/optimizer state in FP32; use BF16 only in autocast forward."""
    if precision == "fp32":
        return False, None
    if precision != "bf16":
        raise ValueError(f"Unsupported precision: {precision}")
    if device != "cuda":
        msg = "BF16 autocast requested, but CUDA is unavailable; falling back to FP32."
        if require_bf16:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False, None
    if not torch.cuda.is_bf16_supported():
        msg = "This CUDA device does not report native BF16 support; falling back to FP32."
        if require_bf16:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False, None
    return True, torch.bfloat16


def autocast_context(device: str, enabled: bool, dtype: Optional[torch.dtype]):
    if enabled and device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return contextlib.nullcontext()


def assert_finite_tensor(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"Non-finite {name} detected: {t.detach()}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def prepare_fineweb_files(
        data_dir: str,
        n_files: int,
        hf_token: Optional[str],
) -> List[str]:
    os.makedirs(data_dir, exist_ok=True)

    files: List[str] = []
    for i in range(n_files):
        filename = f"sample/10BT/{i:03d}_00000.parquet"
        local_path = os.path.join(data_dir, filename)
        files.append(local_path)
        if not os.path.exists(local_path):
            logger.info(f"Downloading FineWeb file {filename}...")
            hf_hub_download(
                repo_id="HuggingFaceFW/fineweb",
                filename=filename,
                repo_type="dataset",
                local_dir=data_dir,
                token=hf_token,
            )
    return files


class FineWebIterable(IterableDataset):
    """
    Same packing logic as the dense pretraining script:
    - tokenize documents without special tokens;
    - skip very short documents;
    - concatenate into a token buffer;
    - emit contiguous input/target chunks of length max_len.
    """

    def __init__(self, tokenizer: GPT2TokenizerFast, max_len: int, parquet_files: List[str]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.parquet_files = parquet_files
        self.dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            streaming=True,
        )["train"]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        sharded_ds = self.dataset.shard(num_shards=num_workers, index=worker_id)
        buf: List[int] = []

        for example in sharded_ds:
            text = example["text"]
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(tokens) < 1025:
                continue

            buf += tokens

            while len(buf) > self.max_len:
                chunk = buf[: self.max_len + 1]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
                buf = buf[self.max_len + 1:]

            if len(buf) < 1025:
                buf = []


# -----------------------------------------------------------------------------
# SmallLM with pluggable attention
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduction in FP32 is more stable for long-context BF16 training.
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


def _filter_constructor_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass exact kwargs when supported, but tolerate local constructor drift."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kwargs:
        return kwargs

    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        logger.warning(f"GlobalAttention constructor does not accept {dropped}; dropping them")
    return filtered


def build_ha_model(vocab_size: int, pad_token_id: int, dropout: float) -> nn.Module:
    def ha_attn_factory(layer_idx: int):
        kwargs = dict(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=dropout,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            use_global=True,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=TOPK_CHUNKS,
            topk_groups=TOPK_GROUPS,
            return_router_stats=False,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            q_norm=None,
            k_norm=None,
        )
        return GlobalAttention(**_filter_constructor_kwargs(GlobalAttention, kwargs)) #RotaryGQASDPA(**_filter_constructor_kwargs(RotaryGQASDPA, kwargs)) 
            

    return SmallLM(
        vocab_size,
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        NUM_LAYERS,
        DFF,
        ha_attn_factory,
        dropout=dropout,
        ignore_index=pad_token_id,
    )


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", getattr(model, "module", model))


def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_state_dict_keys(state: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode == "strip_compile":
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if mode == "add_compile":
        return {k if k.startswith("_orig_mod.") else f"_orig_mod.{k}": v for k, v in state.items()}
    if mode == "strip_module":
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if mode == "strip_compile_strip_module":
        out = {}
        for k, v in state.items():
            k2 = k.replace("_orig_mod.", "", 1).replace("module.", "", 1)
            out[k2] = v
        return out
    return state


def extract_model_state(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload is not a state dict or a dict with key 'model'.")
    if not all(isinstance(k, str) for k in payload.keys()):
        raise RuntimeError("Checkpoint state dict keys are not strings.")
    return payload


def load_state_dict_flexible(
        model: nn.Module,
        path: str,
        map_location: str = "cpu",
        strict: bool = True,
) -> None:
    payload = safe_torch_load(path, map_location=map_location)
    state = extract_model_state(payload)

    variants = [
        ("strip_compile", normalize_state_dict_keys(state, "strip_compile")),
        ("as_is", state),
        ("strip_module", normalize_state_dict_keys(state, "strip_module")),
        ("strip_compile_strip_module", normalize_state_dict_keys(state, "strip_compile_strip_module")),
        ("add_compile", normalize_state_dict_keys(state, "add_compile")),
    ]

    last_error: Optional[Exception] = None
    raw_model = unwrap_model(model)
    model_keys = set(raw_model.state_dict().keys())

    for mode, candidate in variants:
        try:
            result = raw_model.load_state_dict(candidate, strict=strict)
            loaded_keys = model_keys.intersection(candidate.keys())
            logger.info(
                f"Loaded checkpoint {path} with key mode={mode}, strict={strict}, "
                f"matched_tensors={len(loaded_keys)}/{len(model_keys)}"
            )
            if not strict:
                logger.info(f"Missing keys: {len(result.missing_keys)}; unexpected keys: {len(result.unexpected_keys)}")
                if result.missing_keys:
                    logger.info(f"First missing keys: {result.missing_keys[:20]}")
                if result.unexpected_keys:
                    logger.info(f"First unexpected keys: {result.unexpected_keys[:20]}")
                if len(loaded_keys) == 0:
                    raise RuntimeError("Partial load matched zero tensors; refusing to continue.")
            return
        except RuntimeError as exc:
            last_error = exc
            break

    raise RuntimeError(f"Could not load checkpoint {path}: {last_error}")


def resolve_existing_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(SCRIPT_DIR, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Checkpoint not found: {path}. Tried: " + ", ".join(os.path.abspath(c) for c in candidates)
    )


def resolve_dense_checkpoint(args: argparse.Namespace, hf_token: Optional[str]) -> str:
    if args.dense_checkpoint_repo:
        if not args.dense_checkpoint_filename:
            raise ValueError("--dense-checkpoint-filename is required when --dense-checkpoint-repo is used")
        logger.info(
            f"Downloading dense checkpoint {args.dense_checkpoint_filename} "
            f"from {args.dense_checkpoint_repo}..."
        )
        return hf_hub_download(
            repo_id=args.dense_checkpoint_repo,
            filename=args.dense_checkpoint_filename,
            repo_type=args.dense_checkpoint_repo_type,
            token=hf_token,
            local_dir=args.checkpoint_cache_dir,
        )
    return resolve_existing_path(args.dense_checkpoint)


def save_model_state(model: nn.Module, path: str) -> None:
    torch.save(unwrap_model(model).state_dict(), path)


def save_training_checkpoint(
        model: nn.Module,
        optimizers: Tuple[optim.Optimizer, ...],
        schedulers: Tuple[LambdaLR, ...],
        path: str,
        opt_step: int,
        args: argparse.Namespace,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "step": opt_step,
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "args": vars(args),
    }
    torch.save(payload, path)


def count_parameters(model: nn.Module) -> int:
    raw = unwrap_model(model)
    if hasattr(raw, "count_parameters"):
        return int(raw.count_parameters())
    return sum(p.numel() for p in raw.parameters())


# -----------------------------------------------------------------------------
# Trainable scopes
# -----------------------------------------------------------------------------
def normalize_train_scope(scope: str) -> str:
    scope = scope.lower().replace("_", "-")
    aliases = {
        "only-kq": "kq",
        "kq-only": "kq",
        "qk": "kq",
        "only-qk": "kq",
        "attn": "attention",
        "attention-mlp": "attention-mlp",
        "attn-mlp": "attention-mlp",
        "whole-model": "full",
        "all": "full",
    }
    return aliases.get(scope, scope)


def should_train_parameter(name: str, scope: str) -> bool:
    scope = normalize_train_scope(scope)
    if scope == "full":
        return True

    is_attn = ".self_attn." in name
    is_qk = is_attn and any(token in name for token in ("q_proj", "k_proj", "q_norm", "k_norm"))
    is_mlp = ".ffn." in name
    is_block_norm = ".norm1." in name or ".norm2." in name

    if scope == "kq":
        return is_qk
    if scope == "attention":
        return is_attn
    if scope == "attention-mlp":
        return is_attn or is_mlp or is_block_norm
    raise ValueError(f"Unknown train scope: {scope}")


def configure_trainable_parameters(model: nn.Module, scope: str) -> Tuple[int, int]:
    raw = unwrap_model(model)
    total_params = 0
    trainable_params = 0
    trainable_tensors: List[str] = []

    for name, param in raw.named_parameters():
        total_params += param.numel()
        trainable = should_train_parameter(name, scope)
        param.requires_grad_(trainable)
        if trainable:
            trainable_params += param.numel()
            trainable_tensors.append(name)

    if trainable_params == 0:
        raise RuntimeError(f"Train scope '{scope}' selected zero parameters.")

    logger.info(
        f"Train scope={normalize_train_scope(scope)}: "
        f"trainable_params={trainable_params:,}/{total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.2f}%), "
        f"trainable_tensors={len(trainable_tensors)}"
    )
    logger.info(f"First trainable tensors: {trainable_tensors[:30]}")
    return trainable_params, total_params


# -----------------------------------------------------------------------------
# Optimizers / schedulers
# -----------------------------------------------------------------------------
def make_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    actual_warmup = min(max(1, warmup_steps), max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < actual_warmup:
            return float(step + 1) / float(actual_warmup)
        progress = float(step - actual_warmup) / float(max(1, total_steps - actual_warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def build_muon_adamw_optimizer_and_scheduler(
        model: nn.Module,
        base_lr: float,
        muon_lr: float,
        weight_decay: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
):
    if Muon is None:
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build. Use --optimizer adamw.")

    adamw_nodecay_names = [
        "embedding",
        "bias",
        "LayerNorm",
        "norm1",
        "norm2",
        "final_norm",
        "lm_head",
    ]

    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []

    for name, param in unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in adamw_nodecay_names):
            adamw_nodecay_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_decay_params.append(param)

    optimizers: List[optim.Optimizer] = []
    schedulers: List[LambdaLR] = []
    lr_lambda = make_lr_lambda(total_steps, warmup_steps, min_lr_ratio)

    if muon_params:
        muon_optimizer = Muon(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.0,
            ns_steps=5,
        )
        optimizers.append(muon_optimizer)
        schedulers.append(LambdaLR(muon_optimizer, lr_lambda=lr_lambda))

    adamw_groups = []
    if adamw_nodecay_params:
        adamw_groups.append({"params": adamw_nodecay_params, "weight_decay": 0.0})
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": weight_decay})
    if adamw_groups:
        adamw_optimizer = optim.AdamW(adamw_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
        optimizers.append(adamw_optimizer)
        schedulers.append(LambdaLR(adamw_optimizer, lr_lambda=lr_lambda))

    if not optimizers:
        raise RuntimeError("No trainable parameters were assigned to any optimizer.")

    logger.info(
        "Optimizer split: "
        f"Muon tensors={len(muon_params)}, "
        f"AdamW no-decay tensors={len(adamw_nodecay_params)}, "
        f"AdamW decay tensors={len(adamw_decay_params)}"
    )
    return tuple(optimizers), tuple(schedulers)


def build_adamw_optimizer_and_scheduler(
        model: nn.Module,
        base_lr: float,
        weight_decay: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
):
    no_decay = ["embedding", "bias", "LayerNorm", "norm", "final_norm", "lm_head"]
    nodecay_params = []
    decay_params = []

    for name, param in unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if nodecay_params:
        groups.append({"params": nodecay_params, "weight_decay": 0.0})
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if not groups:
        raise RuntimeError("No trainable parameters were assigned to AdamW.")

    optimizer = optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(total_steps, warmup_steps, min_lr_ratio))
    logger.info(f"AdamW split: no_decay tensors={len(nodecay_params)}, decay tensors={len(decay_params)}")
    return (optimizer,), (scheduler,)


def fast_forward_schedulers(schedulers: Iterable[LambdaLR], start_step: int) -> None:
    if start_step <= 0:
        return
    for _ in range(start_step):
        for scheduler in schedulers:
            scheduler.step()


def optimizer_lrs(optimizers: Tuple[optim.Optimizer, ...]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for idx, optimizer in enumerate(optimizers):
        key = optimizer.__class__.__name__.lower()
        if key in result:
            key = f"{key}_{idx}"
        result[key] = f"{optimizer.param_groups[0]['lr']:.2e}"
    return result


# -----------------------------------------------------------------------------
# Loss / eval / causality check
# -----------------------------------------------------------------------------
def extract_loss(result: Any) -> torch.Tensor:
    if isinstance(result, dict):
        loss = result.get("loss")
    elif hasattr(result, "loss"):
        loss = getattr(result, "loss")
    elif isinstance(result, (tuple, list)):
        loss = result[1] if len(result) > 1 else None
    else:
        loss = None
    if loss is None:
        raise RuntimeError("loss is None; labels were probably not passed or model output is unexpected.")
    return loss.view(-1).mean()


def extract_logits(result: Any) -> torch.Tensor:
    if isinstance(result, dict):
        logits = result.get("logits")
    elif hasattr(result, "logits"):
        logits = getattr(result, "logits")
    elif isinstance(result, (tuple, list)):
        logits = None
        for item in result:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                logits = item
                break
    elif isinstance(result, torch.Tensor) and result.ndim >= 3:
        logits = result
    else:
        logits = None
    if logits is None:
        raise RuntimeError("Could not extract logits from model output.")
    return logits


def mean_ce_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def stable_loss_from_output(result: Any, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    try:
        logits = extract_logits(result)
    except RuntimeError:
        loss = extract_loss(result).float()
    else:
        loss = mean_ce_from_logits(logits, labels, ignore_index=ignore_index)
    loss = loss.view(-1).mean().float()
    assert_finite_tensor(loss, "loss")
    return loss


@torch.inference_mode()
def evaluate_loss(
        model: nn.Module,
        eval_batches_cpu: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: str,
        ignore_index: int,
        amp_enabled: bool,
        amp_dtype: Optional[torch.dtype],
) -> Optional[float]:
    if not eval_batches_cpu:
        return None

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    try:
        for inputs_cpu, targets_cpu in eval_batches_cpu:
            inputs = inputs_cpu.to(device, non_blocking=True).to(torch.long)
            targets = targets_cpu.to(device, non_blocking=True).to(torch.long)
            with autocast_context(device, amp_enabled, amp_dtype):
                result = model(inputs, labels=targets)
            loss = stable_loss_from_output(result, targets, ignore_index=ignore_index)
            total_loss += float(loss.item())
            total_batches += 1
            del inputs, targets, result, loss
    finally:
        model.train(was_training)
    if total_batches == 0:
        return None
    return total_loss / total_batches


@torch.no_grad()
def validate_causality_loss_match(
        name: str,
        model: nn.Module,
        validation_inputs: torch.Tensor,
        validation_targets: torch.Tensor,
        device: str,
        opt_step: int,
        check_len: int,
        ignore_index: int,
        tolerance: float,
        amp_enabled: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[bool, Dict[str, float]]:
    if check_len <= 0:
        raise ValueError("check_len must be > 0")

    check_len = min(check_len, int(validation_inputs.numel()), int(validation_targets.numel()))
    if check_len <= 0:
        raise RuntimeError("Validation sequence is empty.")

    eval_model = unwrap_model(model)
    was_training = model.training
    model.eval()
    eval_model.eval()

    try:
        inputs = validation_inputs[:check_len].unsqueeze(0).to(device, non_blocking=True).to(torch.long)
        targets = validation_targets[:check_len].unsqueeze(0).to(device, non_blocking=True).to(torch.long)

        with autocast_context(device, amp_enabled, amp_dtype):
            full_logits = extract_logits(eval_model(inputs))
        full_loss_t = mean_ce_from_logits(full_logits, targets, ignore_index=ignore_index)
        full_loss = float(full_loss_t.item())
        del full_logits, full_loss_t

        token_loss_sum = torch.zeros((), device=inputs.device, dtype=torch.float64)
        token_count = 0
        for pos in range(check_len):
            target = targets[:, pos]
            valid_count = int((target != ignore_index).sum().item())
            if valid_count == 0:
                continue

            with autocast_context(device, amp_enabled, amp_dtype):
                prefix_logits = extract_logits(eval_model(inputs[:, : pos + 1]))
            last_logits = prefix_logits[:, -1, :]
            token_loss = F.cross_entropy(
                last_logits.float(),
                target,
                ignore_index=ignore_index,
                reduction="sum",
            )
            token_loss_sum += token_loss.double()
            token_count += valid_count
            del prefix_logits, last_logits, token_loss

        if token_count == 0:
            raise RuntimeError("All validation targets were ignored; choose a different validation sequence.")

        generation_loss = float((token_loss_sum / token_count).item())
        abs_diff = abs(full_loss - generation_loss)
        rel_diff = abs_diff / max(abs(generation_loss), 1e-12)
        ok = abs_diff <= tolerance

        metrics = {
            "full_loss": full_loss,
            "generation_loss": generation_loss,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "tokens": float(token_count),
        }
        log_fn = logger.info if ok else logger.error
        log_fn(
            f"{name}: causality_check step={opt_step}, tokens={token_count}, "
            f"full_loss={full_loss:.8f}, token_by_token_loss={generation_loss:.8f}, "
            f"abs_diff={abs_diff:.3e}, rel_diff={rel_diff:.3e}, "
            f"tol={tolerance:.3e}, status={'PASS' if ok else 'FAIL'}"
        )
        return ok, metrics
    finally:
        model.train(was_training)
        eval_model.train(was_training)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_model(
        name: str,
        model: nn.Module,
        dataloader: DataLoader,
        total_steps: int,
        device: str,
        args: argparse.Namespace,
        start_step: int = 0,
        ignore_index: int = -100,
        amp_enabled: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
) -> List[Tuple[int, float, Optional[float]]]:
    model.to(device)
    model.train()

    if args.optimizer == "muon":
        optimizers, schedulers = build_muon_adamw_optimizer_and_scheduler(
            model,
            base_lr=args.lr,
            muon_lr=args.muon_lr,
            weight_decay=args.weight_decay,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    elif args.optimizer == "adamw":
        optimizers, schedulers = build_adamw_optimizer_and_scheduler(
            model,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {args.optimizer}")

    fast_forward_schedulers(schedulers, start_step)
    if start_step > 0:
        logger.info(f"{name}: resumed scheduler at step {start_step}, lr={optimizer_lrs(optimizers)}")

    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    eval_batches_cpu: List[Tuple[torch.Tensor, torch.Tensor]] = []
    validation_inputs_cpu: Optional[torch.Tensor] = None
    validation_targets_cpu: Optional[torch.Tensor] = None

    for _ in range(max(0, args.eval_batches)):
        try:
            eval_inputs, eval_targets = next(data_iter)
        except StopIteration:
            break
        eval_batches_cpu.append((eval_inputs.detach().cpu().to(torch.long), eval_targets.detach().cpu().to(torch.long)))

    if eval_batches_cpu:
        validation_inputs_cpu = eval_batches_cpu[0][0][0].detach().cpu().to(torch.long)
        validation_targets_cpu = eval_batches_cpu[0][1][0].detach().cpu().to(torch.long)
        logger.info(
            f"Reserved {len(eval_batches_cpu)} eval batch(es); "
            f"causality check uses first {min(args.causality_check_len, validation_inputs_cpu.numel())} tokens."
        )
        initial_eval = evaluate_loss(
            model,
            eval_batches_cpu,
            device,
            ignore_index,
            amp_enabled,
            amp_dtype,
        )
        if initial_eval is not None:
            logger.info(f"{name}: initial_eval_loss={initial_eval:.6f}")
    else:
        logger.warning(f"{name}: could not reserve eval batches; validation loss disabled")
        initial_eval = None

    loss_history: List[Tuple[int, float, Optional[float]]] = []
    step_loss = 0.0
    micro_step = 0
    opt_step = start_step
    start_time = time.time()
    remaining = max(0, total_steps - start_step)
    pbar = tqdm(total=remaining, desc=f"{name}-{args.optimizer}-{normalize_train_scope(args.train_scope)}", unit="step")

    while opt_step < total_steps:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            logger.info("Dataloader exhausted before reaching target_steps.")
            break

        inputs = inputs.to(device, non_blocking=True).to(torch.long)
        targets = targets.to(device, non_blocking=True).to(torch.long)

        with autocast_context(device, amp_enabled, amp_dtype):
            result = model(inputs, labels=targets)
        loss = stable_loss_from_output(result, targets, ignore_index=ignore_index)
        loss_value = float(loss.detach().item())

        (loss / args.accum_steps).backward()
        step_loss += loss_value
        micro_step += 1

        del inputs, targets, result, loss

        if micro_step % args.accum_steps != 0:
            continue

        avg_loss = step_loss / args.accum_steps
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                args.grad_clip,
                error_if_nonfinite=True,
            )
        except TypeError:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                args.grad_clip,
            )
            assert_finite_tensor(grad_norm, "gradient norm")

        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        opt_step += 1
        step_loss = 0.0

        eval_loss: Optional[float] = None
        if args.eval_every > 0 and opt_step % args.eval_every == 0 and eval_batches_cpu:
            eval_loss = evaluate_loss(
                model,
                eval_batches_cpu,
                device,
                ignore_index,
                amp_enabled,
                amp_dtype,
            )
            if eval_loss is not None:
                logger.info(f"{name}: opt_step={opt_step}, eval_loss={eval_loss:.6f}")

        loss_history.append((opt_step, avg_loss, eval_loss))

        if opt_step % args.log_every == 0:
            elapsed = time.time() - start_time
            tokens_done = max(1, opt_step - start_step) * args.batch_size * args.accum_steps * args.max_len
            tps = tokens_done / max(elapsed, 1e-9)
            postfix = {
                "loss": f"{avg_loss:.4f}",
                "tok/s": f"{tps:.0f}",
                "grad": f"{float(grad_norm):.2f}",
            }
            if eval_loss is not None:
                postfix["eval"] = f"{eval_loss:.4f}"
            postfix.update(optimizer_lrs(optimizers))
            pbar.set_postfix(postfix)
            logger.info(
                f"{name}: opt_step={opt_step}, loss={avg_loss:.4f}, "
                f"grad_norm_before_clip={float(grad_norm):.2f}, lr={optimizer_lrs(optimizers)}"
            )

        pbar.update(1)

        if (
                args.causality_check_every > 0
                and args.causality_check_len > 0
                and opt_step % args.causality_check_every == 0
                and validation_inputs_cpu is not None
                and validation_targets_cpu is not None
        ):
            ok, _ = validate_causality_loss_match(
                name=name,
                model=model,
                validation_inputs=validation_inputs_cpu,
                validation_targets=validation_targets_cpu,
                device=device,
                opt_step=opt_step,
                check_len=args.causality_check_len,
                ignore_index=ignore_index,
                tolerance=args.causality_check_tol,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if not ok and args.causality_check_fail_fast:
                raise RuntimeError(
                    f"{name}: causality check failed at step {opt_step}. "
                    "Full-sequence teacher-forced loss differs from token-by-token generation loss."
                )

        if args.save_every > 0 and opt_step % args.save_every == 0:
            model_path = os.path.join(SCRIPT_DIR, f"speed_run_{name}_{args.optimizer}_{normalize_train_scope(args.train_scope)}_step{opt_step}.pt")
            state_path = os.path.join(SCRIPT_DIR, f"speed_run_{name}_{args.optimizer}_{normalize_train_scope(args.train_scope)}_step{opt_step}_train_state.pt")
            save_model_state(model, model_path)
            save_training_checkpoint(model, optimizers, schedulers, state_path, opt_step, args)
            logger.info(f"{name}: saved model checkpoint: {model_path}")
            logger.info(f"{name}: saved training state: {state_path}")

    eval_loss = evaluate_loss(
        model,
        eval_batches_cpu,
        device,
        ignore_index,
        amp_enabled,
        amp_dtype,
    )


    logger.info(f"{name}: Final eval, eval_loss={eval_loss:.6f}")
    
    pbar.close()
    elapsed = time.time() - start_time
    trained_steps = max(1, opt_step - start_step)
    tokens_per_step = args.batch_size * args.accum_steps * args.max_len
    logger.info(
        f"{name}: finished opt_step={opt_step} in {elapsed / 60:.1f} min "
        f"({trained_steps * tokens_per_step / max(elapsed, 1e-9):.0f} tok/s)"
    )
    return loss_history


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_tokenizer(tokenizer_path: str) -> GPT2TokenizerFast:
    if os.path.exists(tokenizer_path):
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.save_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune HierarchicalAttention SmallLM initialized from DenseAttention SmallLM checkpoint"
    )

    parser.add_argument("--dense-checkpoint", type=str,
                        help="Local dense SmallLM final checkpoint. Relative paths are checked from CWD and script dir.")
    parser.add_argument("--dense-checkpoint-repo", type=str, default=None,
                        help="Optional HF repo id to download the dense checkpoint from.")
    parser.add_argument("--dense-checkpoint-filename", type=str, default=None,
                        help="Filename inside --dense-checkpoint-repo.")
    parser.add_argument("--dense-checkpoint-repo-type", choices=["model", "dataset", "space"], default="model")
    parser.add_argument("--checkpoint-cache-dir", type=str, default=None)
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="Use strict=False for loading dense weights. Useful only if HA has extra non-dense parameters.")

    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--n-files", type=int, default=6)
    parser.add_argument("--data-dir", type=str, default="fineweb_sample")
    parser.add_argument("--tokenizer-path", type=str, default="gpt2_tokenizer")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    parser.add_argument("--train-scope", type=str, default="qk",
                        choices=["kq", "only-kq", "qk", "attention", "attn", "attention-mlp", "attn-mlp", "full", "all"],
                        help="What to fine-tune: kq, attention, attention-mlp, or full.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--muon-lr", type=float, default=MUON_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--min-lr-ratio", type=float, default=MIN_LR_RATIO)
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Only advances the LR schedule; model weights still come from --dense-checkpoint.")

    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--compile-ha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-batches", type=int, default=2,
                        help="Batches reserved from the stream for quick loss tracking. 0 disables eval.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--causality-check-every", type=int, default=CAUSALITY_CHECK_EVERY,
                        help="0 disables teacher-forced-vs-token-by-token causality check.")
    parser.add_argument("--causality-check-len", type=int, default=CAUSALITY_CHECK_LEN)
    parser.add_argument("--causality-check-tol", type=float, default=CAUSALITY_CHECK_TOL)
    parser.add_argument("--causality-check-fail-fast", action="store_true")

    args = parser.parse_args()
    args.train_scope = normalize_train_scope(args.train_scope)

    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be > 0")
    if args.batch_size <= 0 or args.accum_steps <= 0 or args.max_len <= 0:
        raise ValueError("--batch-size, --accum-steps, and --max-len must be > 0")
    if args.max_len != MAX_LEN:
        logger.warning(
            f"--max-len={args.max_len} differs from architecture MAX_LEN={MAX_LEN}. "
            "This script keeps the model architecture fixed; data chunks use --max-len."
        )

    tokens_per_step = args.batch_size * args.accum_steps * args.max_len
    target_steps = args.target_tokens // tokens_per_step
    if target_steps <= args.resume_step:
        raise ValueError(
            f"Target steps ({target_steps}) must be greater than --resume-step ({args.resume_step}). "
            "Increase --target-tokens."
        )

    set_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.info("HF_TOKEN is not set. Public datasets/checkpoints may still work.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    amp_enabled, amp_dtype = resolve_amp_config(args.precision, DEVICE, require_bf16=args.require_bf16)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Script directory: {SCRIPT_DIR}")
    logger.info(f"GlobalAttention import: {inspect.getfile(GlobalAttention) if GlobalAttention is not None else 'FAILED'}")
    logger.info(
        f"Batch={args.batch_size}, accum={args.accum_steps}, seq_len={args.max_len}, "
        f"tokens/step={tokens_per_step}, target_steps={target_steps}, "
        f"optimizer={args.optimizer}, lr={args.lr:.2e}, muon_lr={args.muon_lr:.2e}, "
        f"grad_clip={args.grad_clip}, train_scope={args.train_scope}, "
        f"precision={args.precision}, amp_enabled={amp_enabled}, amp_dtype={amp_dtype}"
    )

    tokenizer = build_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size}; PAD token id: {tokenizer.pad_token_id}")

    dense_ckpt_path = resolve_dense_checkpoint(args, hf_token)
    logger.info(f"Dense checkpoint resolved to: {dense_ckpt_path}")

    parquet_files = prepare_fineweb_files(args.data_dir, args.n_files, hf_token)
    dataset = FineWebIterable(tokenizer, args.max_len, parquet_files)
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    model_ha = build_ha_model(vocab_size, tokenizer.pad_token_id, dropout=args.dropout)
    logger.info(f"HA SmallLM: {count_parameters(model_ha):,} parameters")

    load_state_dict_flexible(
        model_ha,
        dense_ckpt_path,
        map_location="cpu",
        strict=not args.allow_partial_load,
    )
    
    configure_trainable_parameters(model_ha, args.train_scope)

    if args.compile_ha:
        model_ha = torch.compile(model_ha)
        logger.info("Compiled HA model with torch.compile")

    logger.info("=" * 60)
    logger.info(
        f"Fine-tuning HA SmallLM from dense checkpoint for {target_steps - args.resume_step} optimizer steps "
        f"({args.target_tokens / 1e6:.0f}M target tokens), scope={args.train_scope}"
    )
    logger.info("=" * 60)

    history = train_model(
        "ha_from_dense",
        model_ha,
        dataloader,
        target_steps,
        DEVICE,
        args,
        start_step=args.resume_step,
        ignore_index=tokenizer.pad_token_id,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    scope = normalize_train_scope(args.train_scope)
    final_ckpt = os.path.join(SCRIPT_DIR, f"speed_run_ha_from_dense_{args.optimizer}_{scope}_final.pt")
    final_state = os.path.join(SCRIPT_DIR, f"speed_run_ha_from_dense_{args.optimizer}_{scope}_final_train_state.pt")
    save_model_state(model_ha, final_ckpt)
    logger.info(f"HA: saved final model checkpoint to {final_ckpt}")

    # Save a lightweight final training state too. Optimizer state is useful if you continue the same run.
    # Optimizers/schedulers are saved inside periodic checkpoints during training; final model-only is the main artifact.
    torch.save({"model": unwrap_model(model_ha).state_dict(), "args": vars(args), "history": history}, final_state)
    logger.info(f"HA: saved final train metadata to {final_state}")

    history_path = os.path.join(SCRIPT_DIR, f"speed_run_ha_from_dense_{args.optimizer}_{scope}_loss.tsv")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("step\ttrain_loss\teval_loss\n")
        for step, train_loss, eval_loss in history:
            eval_text = "" if eval_loss is None else f"{eval_loss:.8f}"
            f.write(f"{step}\t{train_loss:.8f}\t{eval_text}\n")
    logger.info(f"HA: saved loss history to {history_path}")


if __name__ == "__main__":
    main()
