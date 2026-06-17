import os
import math
import gc
import time
import logging
import argparse
import contextlib
import inspect
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader

from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download


import torch_inductor_patch

# Backport the upstream fix for an Inductor codegen crash (no_x_dim scan
# kernels) hit when compiling GlobalAttentionFused on torch 2.11.
torch_inductor_patch.apply()

# Optional local comparison model pieces kept from the original second script.
# HA training below uses Model20M, same as the first script.
try:
    from HierarchicalGlobalAttentionFusedExactQ import GlobalAttention
except Exception:  # Not needed when HA uses Model20M.
    GlobalAttention = None

# -----------------------------------------------------------------------------
# Hyperparameters aligned with the first script
# -----------------------------------------------------------------------------
HIDDEN_DIM = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
MAX_LEN = 8192

# Same effective batch as the first script: 1 * 6 * 8192 = 49,152 tokens/step.
BATCH_SIZE = 3
ACCUM_STEPS = 2
TOKENS_PER_STEP = BATCH_SIZE * ACCUM_STEPS * MAX_LEN

LR = 2e-4
MUON_LR = 0.007
GRAD_CLIP = 15.0
TARGET_TOKENS = 2_000_000_000
TARGET_STEPS = TARGET_TOKENS // TOKENS_PER_STEP
WARMUP_STEPS = 500
MIN_LR_RATIO = 0.2

LOG_EVERY = 10
SAVE_EVERY = 10_000

# Compares teacher-forced full-sequence loss with prefix-by-prefix next-token
# loss on one fixed validation sequence. Use --causality-check-len 8192 for a
# full-context check, but that is intentionally not the default because it means
# 8192 forward passes every validation.
CAUSALITY_CHECK_EVERY = 1_000
CAUSALITY_CHECK_LEN = 2048
CAUSALITY_CHECK_TOL = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 64
GROUP_SIZE = 4
TOPK_CHUNKS = 20
TOPK_GROUPS = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(log_file: str = "train_small_model_fixed_muon.log") -> logging.Logger:
    logger = logging.getLogger("speed_run_fixed")
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


logger = setup_logger()


# -----------------------------------------------------------------------------
# Precision / stability helpers
# -----------------------------------------------------------------------------
def resolve_amp_config(
        precision: str,
        device: str,
        require_bf16: bool = False,
) -> Tuple[bool, Optional[torch.dtype]]:
    """
    Best-practice BF16 setup:
    - keep model parameters and optimizer state in FP32;
    - use autocast only for the forward pass;
    - keep the loss in FP32.
    """
    if precision == "fp32":
        return False, None

    if precision != "bf16":
        raise ValueError(f"Unsupported precision: {precision}")

    if device != "cuda":
        msg = "BF16 autocast was requested, but CUDA is not available; using FP32."
        if require_bf16:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False, None

    if not torch.cuda.is_bf16_supported():
        msg = "This CUDA device does not report native BF16 support; using FP32."
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
            logger.info(f"Downloading {filename}...")
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
    Same packing logic as the first script:
    - tokenize documents without special tokens;
    - keep documents >=1025 tokens;
    - concatenate into a buffer;
    - emit contiguous max_len input/target chunks.

    This avoids the second script's old behavior of discarding every document
    shorter than MAX_LEN + 1.
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
# Optional Dense/SmallLM code kept from the second script
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm is numerically sensitive; compute the reduction in FP32 even
        # under BF16 autocast, then cast activations back to the input dtype.
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


class DenseAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            kv_heads: int = 2,
            dropout: float = 0.0,
            theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = d_model // nhead
        self.theta = theta
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(nhead * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        # Build rotary embeddings in FP32 for long-context stability, then cast
        # the rotated q/k back to the autocast dtype.
        cos, sin = self._get_rotary(seq, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = self._apply_rotary(q.float(), cos, sin).to(dtype=q.dtype)
        k = self._apply_rotary(k.float(), cos, sin).to(dtype=k.dtype)

        repeat_factor = self.nhead // self.kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.nhead * self.head_dim)
        return self.o_proj(out), {}

    def _get_rotary(self, seq_len: int, device: torch.device):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (DenseAttention._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half_dim = x.shape[-1] // 2
        return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn_module: nn.Module, dff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(self.norm1(x))
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
            # Cross entropy should be accumulated in FP32 for stable BF16 training.
            loss = self.criterion(logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1))
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Optimizers / schedulers
# -----------------------------------------------------------------------------
def make_lr_lambda(total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
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
        min_lr_ratio: float,
):
    """Same Muon/AdamW parameter split as the first script."""
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

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in adamw_nodecay_names):
            adamw_nodecay_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_decay_params.append(param)

    if not muon_params:
        raise RuntimeError("No 2D parameters were assigned to Muon; check model parameter names/shapes.")

    muon_optimizer = Muon(
        muon_params,
        lr=muon_lr,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
        ns_steps=5,
    )

    adamw_groups = []
    if adamw_nodecay_params:
        adamw_groups.append({"params": adamw_nodecay_params, "weight_decay": 0.0})
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": weight_decay})

    adamw_optimizer = optim.AdamW(adamw_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)

    lr_lambda = make_lr_lambda(total_steps, min_lr_ratio)
    muon_scheduler = LambdaLR(muon_optimizer, lr_lambda=lr_lambda)
    adamw_scheduler = LambdaLR(adamw_optimizer, lr_lambda=lr_lambda)

    logger.info(
        "Optimizer split: "
        f"Muon tensors={len(muon_params)}, "
        f"AdamW no-decay tensors={len(adamw_nodecay_params)}, "
        f"AdamW decay tensors={len(adamw_decay_params)}"
    )
    return (muon_optimizer, adamw_optimizer), (muon_scheduler, adamw_scheduler)


def build_adamw_optimizer_and_scheduler(
        model: nn.Module,
        base_lr: float,
        weight_decay: float,
        total_steps: int,
        min_lr_ratio: float,
):
    no_decay = ["embedding", "bias", "LayerNorm", "norm", "final_norm", "lm_head"]
    nodecay_params = []
    decay_params = []

    for name, param in model.named_parameters():
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

    optimizer = optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(total_steps, min_lr_ratio))
    return (optimizer,), (scheduler,)


def fast_forward_schedulers(schedulers: Iterable[LambdaLR], start_step: int) -> None:
    if start_step <= 0:
        return
    for _ in range(start_step):
        for scheduler in schedulers:
            scheduler.step()


def optimizer_lrs(optimizers: Tuple[optim.Optimizer, ...]) -> Dict[str, str]:
    if len(optimizers) == 2:
        return {
            "lr_muon": f"{optimizers[0].param_groups[0]['lr']:.2e}",
            "lr_adamw": f"{optimizers[1].param_groups[0]['lr']:.2e}",
        }
    return {"lr": f"{optimizers[0].param_groups[0]['lr']:.2e}"}


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def normalize_state_dict_keys(state: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode == "strip_compile":
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if mode == "add_compile":
        return {k if k.startswith("_orig_mod.") else f"_orig_mod.{k}": v for k, v in state.items()}
    if mode == "strip_module":
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if mode == "add_compile_strip_module":
        return {
            (k.replace("module.", "", 1) if k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def load_state_dict_flexible(model: nn.Module, path: str, map_location: str = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} did not contain a state dict.")

    variants = [
        state,
        normalize_state_dict_keys(state, "strip_module"),
        normalize_state_dict_keys(state, "strip_compile"),
        normalize_state_dict_keys(state, "add_compile"),
    ]

    last_error: Optional[Exception] = None
    for candidate in variants:
        try:
            model.load_state_dict(candidate)
            logger.info(f"Loaded checkpoint: {path}")
            return
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not load checkpoint {path}: {last_error}")


def count_parameters(model: nn.Module) -> int:
    raw = getattr(model, "_orig_mod", getattr(model, "module", model))
    if hasattr(raw, "count_parameters"):
        return int(raw.count_parameters())
    return sum(p.numel() for p in raw.parameters())


# -----------------------------------------------------------------------------
# Training
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
    """Prefer recomputing CE from logits in FP32; fall back to model-provided loss."""
    try:
        logits = extract_logits(result)
    except RuntimeError:
        loss = extract_loss(result).float()
    else:
        loss = mean_ce_from_logits(logits, labels, ignore_index=ignore_index)

    loss = loss.view(-1).mean().float()
    assert_finite_tensor(loss, "loss")
    return loss


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
    """
    Detects future-token leakage by comparing two losses on the same sequence:
    1. full teacher-forced loss: model sees the whole input sequence at once;
    2. token-by-token generation loss: for position i, model sees only input[:i+1]
       and predicts target[i] from the last logit.

    For a causal model in eval mode these should match up to numerical noise.
    A much lower full-sequence loss is a strong signal that attention can see future
    tokens during teacher-forced training.
    """
    if check_len <= 0:
        raise ValueError("check_len must be > 0")

    check_len = min(check_len, int(validation_inputs.numel()), int(validation_targets.numel()))
    if check_len <= 0:
        raise RuntimeError("Validation sequence is empty.")

    # Avoid triggering many torch.compile specializations for prefix lengths 1..N.
    # OptimizedModule keeps the eager module in _orig_mod and shares parameters.
    eval_model = getattr(model, "_orig_mod", model)
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


def train_model(
        name: str,
        model: nn.Module,
        dataloader: DataLoader,
        total_steps: int,
        device: str,
        save_every: int,
        optimizer_kind: str,
        start_step: int = 0,
        ignore_index: int = -100,
        amp_enabled: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        causality_check_every: int = CAUSALITY_CHECK_EVERY,
        causality_check_len: int = CAUSALITY_CHECK_LEN,
        causality_check_tol: float = CAUSALITY_CHECK_TOL,
        causality_check_fail_fast: bool = False,
) -> List[Tuple[int, float]]:
    model.to(device)
    model.train()

    if optimizer_kind == "muon":
        optimizers, schedulers = build_muon_adamw_optimizer_and_scheduler(
            model,
            base_lr=LR,
            muon_lr=MUON_LR,
            weight_decay=0.1,
            total_steps=total_steps,
            min_lr_ratio=MIN_LR_RATIO,
        )
    elif optimizer_kind == "adamw":
        optimizers, schedulers = build_adamw_optimizer_and_scheduler(
            model,
            base_lr=LR,
            weight_decay=0.1,
            total_steps=total_steps,
            min_lr_ratio=MIN_LR_RATIO,
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {optimizer_kind}")

    fast_forward_schedulers(schedulers, start_step)
    if start_step > 0:
        logger.info(f"{name}: resumed scheduler at step {start_step}, lr={optimizer_lrs(optimizers)}")

    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    loss_history: List[Tuple[int, float]] = []
    step_loss = 0.0
    micro_step = 0
    opt_step = start_step
    start_time = time.time()
    data_iter = iter(dataloader)
    validation_inputs_cpu: Optional[torch.Tensor] = None
    validation_targets_cpu: Optional[torch.Tensor] = None

    if causality_check_every > 0 and causality_check_len > 0:
        try:
            validation_inputs, validation_targets = next(data_iter)
            if validation_inputs.shape[0] > 0:
                validation_inputs_cpu = validation_inputs[0].detach().cpu().to(torch.long)
                validation_targets_cpu = validation_targets[0].detach().cpu().to(torch.long)
                logger.info(
                    f"{name}: reserved one causality validation sequence with "
                    f"{validation_inputs_cpu.numel()} tokens; "
                    f"checking first {min(causality_check_len, validation_inputs_cpu.numel())} tokens "
                    f"every {causality_check_every} optimizer steps"
                )
        except StopIteration:
            logger.warning(f"{name}: could not reserve causality validation sequence; dataloader is empty")

    remaining = total_steps - start_step
    pbar = tqdm(total=remaining, desc=f"{name}-{optimizer_kind}", unit="step")

    while opt_step < total_steps:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            break

        inputs = inputs.to(device, non_blocking=True).to(torch.long)
        targets = targets.to(device, non_blocking=True).to(torch.long)

        with autocast_context(device, amp_enabled, amp_dtype):
            result = model(inputs, labels=targets)
        loss = stable_loss_from_output(result, targets, ignore_index=ignore_index)
        loss_value = float(loss.detach().item())

        (loss / ACCUM_STEPS).backward()
        step_loss += loss_value
        micro_step += 1

        if micro_step % ACCUM_STEPS == 0:
            avg_loss = step_loss / ACCUM_STEPS
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP, error_if_nonfinite=True
                )
            except TypeError:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                assert_finite_tensor(grad_norm, "gradient norm")

            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            opt_step += 1
            step_loss = 0.0
            loss_history.append((opt_step, avg_loss))

            if opt_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                tokens_done = max(1, opt_step - start_step) * TOKENS_PER_STEP
                tps = tokens_done / max(elapsed, 1e-9)
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "tok/s": f"{tps:.0f}",
                    "grad": f"{float(grad_norm):.2f}",
                }
                postfix.update(optimizer_lrs(optimizers))
                pbar.set_postfix(postfix)
                logger.info(
                    f"{name}: opt_step={opt_step}, loss={avg_loss:.4f}, "
                    f"grad_norm_before_clip={float(grad_norm):.2f}, lr={optimizer_lrs(optimizers)}"
                )

            pbar.update(1)

            if (
                    causality_check_every > 0
                    and causality_check_len > 0
                    and opt_step % causality_check_every == 0
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
                    check_len=causality_check_len,
                    ignore_index=ignore_index,
                    tolerance=causality_check_tol,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                if not ok and causality_check_fail_fast:
                    raise RuntimeError(
                        f"{name}: causality check failed at step {opt_step}. "
                        "Full-sequence teacher-forced loss differs from token-by-token generation loss."
                    )

            if save_every > 0 and opt_step % save_every == 0:
                ckpt_path = os.path.join(SCRIPT_DIR, f"speed_run_{name}_{optimizer_kind}_step{opt_step}.pt")
                torch.save(getattr(model, "_orig_mod", model).state_dict(), ckpt_path)
                logger.info(f"{name}: saved checkpoint at step {opt_step}: {ckpt_path}")

    pbar.close()
    elapsed = time.time() - start_time
    trained_steps = max(1, total_steps - start_step)
    logger.info(
        f"{name}: finished {trained_steps} scheduled steps in {elapsed / 60:.1f} min "
        f"({trained_steps * TOKENS_PER_STEP / max(elapsed, 1e-9):.0f} tok/s)"
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


def build_model20m(vocab_size: int, tokenizer: GPT2TokenizerFast) -> nn.Module:
    return Model20M(
        vocab_size,
        HIDDEN_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        DFF,
        MAX_LEN,
        DEVICE,
        ignore_index=tokenizer.pad_token_id,
        criterion_reduction="none",
        dropout=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed HA speed run with first-script settings and Muon support")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--resume-ha", type=str, default=None)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--model-path", type=str, default="new_attention_pretrained.pt__")
    parser.add_argument("--no-auto-load-ha", action="store_true")
    parser.add_argument("--compile-ha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-files", type=int, default=6)
    parser.add_argument("--data-dir", type=str, default="Tests/Gpt2LevelModel/fineweb_sample")
    parser.add_argument("--tokenizer-path", type=str, default="gpt2_tokenizer")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="Use BF16 autocast on supported CUDA GPUs; weights/optimizer state stay FP32")
    parser.add_argument("--require-bf16", action="store_true",
                        help="Raise an error instead of falling back to FP32 when BF16 is unavailable")
    parser.add_argument("--compile-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causality-check-every", type=int, default=CAUSALITY_CHECK_EVERY,
                        help="Run the teacher-forced-vs-token-by-token causality check every N optimizer steps; 0 disables it")
    parser.add_argument("--causality-check-len", type=int, default=CAUSALITY_CHECK_LEN,
                        help="Number of tokens from one fixed validation sequence to check")
    parser.add_argument("--causality-check-tol", type=float, default=CAUSALITY_CHECK_TOL,
                        help="Allowed absolute loss difference for the causality check")
    parser.add_argument("--causality-check-fail-fast", action="store_true",
                        help="Raise an error and stop training when the causality check fails")
    parser.add_argument("--train-dense", action="store_true", help="Also train the optional Dense SmallLM baseline")
    args = parser.parse_args()

    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be > 0")

    target_steps = args.target_tokens // TOKENS_PER_STEP
    if target_steps <= 0:
        raise ValueError("Target steps resolved to 0; increase --target-tokens")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.info("HF_TOKEN is not set. Public datasets/models may still work; private/gated downloads will fail.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_enabled, amp_dtype = resolve_amp_config(args.precision, DEVICE, require_bf16=args.require_bf16)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Script directory: {SCRIPT_DIR}")
    logger.info(
        f"Batch={BATCH_SIZE}, accum={ACCUM_STEPS}, seq_len={MAX_LEN}, "
        f"tokens/step={TOKENS_PER_STEP}, target_steps={target_steps}, "
        f"grad_clip={GRAD_CLIP}, min_lr_ratio={MIN_LR_RATIO}, optimizer={args.optimizer}, "
        f"precision={args.precision}, amp_enabled={amp_enabled}, amp_dtype={amp_dtype}"
    )
    logger.info(
        f"Causality check: every={args.causality_check_every}, "
        f"len={args.causality_check_len}, tol={args.causality_check_tol:.3e}, "
        f"fail_fast={args.causality_check_fail_fast}"
    )

    tokenizer = build_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size}; PAD token id: {tokenizer.pad_token_id}")

    parquet_files = prepare_fineweb_files(args.data_dir, args.n_files, hf_token)
    dataset = FineWebIterable(tokenizer, MAX_LEN, parquet_files)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=(DEVICE == "cuda"),
    )

    if True:
        def dense_attn_factory(layer_idx: int):
            return DenseAttention(HIDDEN_DIM, NUM_HEADS, KV_HEADS, dropout=0.0)

        model_dense = SmallLM(
            vocab_size,
            HIDDEN_DIM,
            NUM_HEADS,
            KV_HEADS,
            NUM_LAYERS,
            DFF,
            dense_attn_factory,
            dropout=0.0,
            ignore_index=tokenizer.pad_token_id,
        )
        model_dense.to(DEVICE)
        if args.compile_dense:
            model_dense = torch.compile(model_dense)
        logger.info(f"Dense model: {count_parameters(model_dense):,} parameters")
        train_model(
            "dense",
            model_dense,
            dataloader,
            target_steps,
            DEVICE,
            args.save_every,
            optimizer_kind=args.optimizer,
            start_step=args.resume_step,
            ignore_index=tokenizer.pad_token_id,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            causality_check_every=args.causality_check_every,
            causality_check_len=args.causality_check_len,
            causality_check_tol=args.causality_check_tol,
            causality_check_fail_fast=args.causality_check_fail_fast,
        )
        dense_final_ckpt = os.path.join(SCRIPT_DIR, f"speed_run_dense_{args.optimizer}_final.pt")
        torch.save(getattr(model_dense, "_orig_mod", model_dense).state_dict(), dense_final_ckpt)
        logger.info(f"Dense: saved final checkpoint to {dense_final_ckpt}")
        del model_dense
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()