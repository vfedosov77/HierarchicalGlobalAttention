#!/usr/bin/env python3
"""
Benchmark generation quality and speed: Dense vs HierarchicalGlobalAttention.

This script uses the 40M SmallLM model to compare:
  1) Standard dense attention (loaded from speed_run_dense_muon_final.pt)
  2) Hierarchical Global Attention (fine-tuned from the same dense checkpoint)

Modes:
  --mode finetune   Fine-tune the HA model from the dense checkpoint and save it.
  --mode benchmark  Compare generation quality and speed between both models.

Generation tests:
  - Token-by-token generation (no context, measuring per-token latency)
  - Prefill + generation (existing context, measuring throughput)
  - Quality: perplexity on validation data, output token agreement

Usage:
    # Step 1: Fine-tune HA model (skip if checkpoint exists)
    python benchmark_generation.py --mode finetune --target-tokens 50000000

    # Step 2: Benchmark
    python benchmark_generation.py --mode benchmark --gen-tokens 128 --context-len 512
"""

import argparse
import contextlib
import gc
import inspect
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
from transformers import GPT2TokenizerFast, DynamicCache

# Add parent dirs to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.insert(0, ROOT_DIR)

from ExistingModelFineTuning.HierarchicalGlobalAttentionRouted import HierarchicalGlobalAttentionRouted

HA_IMPLS = {
    "routed": HierarchicalGlobalAttentionRouted,
}

# -----------------------------------------------------------------------------
# Model / data defaults
# -----------------------------------------------------------------------------
HIDDEN_DIM = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
MAX_LEN = 8192

CHUNK_SIZE = 64
GROUP_SIZE = 16
TOPK_CHUNKS = 20
TOPK_GROUPS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DENSE_CHECKPOINT = os.path.join(PARENT_DIR, "speed_run_dense_muon_final.pt")
HA_CHECKPOINT = os.path.join(SCRIPT_DIR, "ha_finetuned_from_dense.pt")


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Standard GQA with RoPE and SDPA. Supports KV cache for generation."""

    def __init__(self, d_model: int, nhead: int, kv_heads: int = 2,
                 dropout: float = 0.0, theta: float = 1_000_000.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = d_model // nhead
        self.theta = theta
        self.dropout_p = dropout
        self.num_key_value_groups = nhead // kv_heads

        self.q_proj = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(nhead * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_key_value=None,
                cache_position=None, **kwargs) -> Tuple[torch.Tensor, Any]:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        if cache_position is not None:
            pos = cache_position
        else:
            start = 0
            if past_key_value is not None and hasattr(past_key_value, 'get_seq_length'):
                start = past_key_value.get_seq_length()
            pos = torch.arange(start, start + seq, device=x.device)

        cos, sin = self._get_rotary(int(pos.max().item()) + 1, x.device)
        cos = cos[pos].unsqueeze(0).unsqueeze(0)
        sin = sin[pos].unsqueeze(0).unsqueeze(0)
        q = self._apply_rotary(q.float(), cos, sin).to(dtype=q.dtype)
        k = self._apply_rotary(k.float(), cos, sin).to(dtype=k.dtype)

        if past_key_value is not None:
            k_cache, v_cache = past_key_value.update(k, v, 0)
            k, v = k_cache, v_cache

        rep = self.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Causal LM: SDPA's bottom-right alignment makes is_causal=True correct for
        # prefill (q_len==k_len) and decode (q_len<k_len: the new tokens see all past).
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.nhead * self.head_dim)
        return self.o_proj(out), past_key_value

    def _get_rotary(self, seq_len: int, device: torch.device):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class DenseAttentionLayered(DenseAttention):
    """DenseAttention that stores KV in per-layer slots of a shared DynamicCache."""

    def __init__(self, layer_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, past_key_value=None,
                cache_position=None, **kwargs) -> Tuple[torch.Tensor, Any]:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        if cache_position is not None:
            pos = cache_position
        else:
            start = 0
            if past_key_value is not None:
                try:
                    start = past_key_value.get_seq_length(self.layer_idx)
                except TypeError:
                    start = past_key_value.get_seq_length()
            pos = torch.arange(start, start + seq, device=x.device)

        cos, sin = self._get_rotary(int(pos.max().item()) + 1, x.device)
        cos = cos[pos].unsqueeze(0).unsqueeze(0)
        sin = sin[pos].unsqueeze(0).unsqueeze(0)
        q = self._apply_rotary(q.float(), cos, sin).to(dtype=q.dtype)
        k = self._apply_rotary(k.float(), cos, sin).to(dtype=k.dtype)

        if past_key_value is not None:
            k_cache, v_cache = past_key_value.update(k, v, self.layer_idx)
            k, v = k_cache, v_cache

        rep = self.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Causal LM: SDPA's bottom-right alignment makes is_causal=True correct for
        # prefill (q_len==k_len) and decode (q_len<k_len: the new tokens see all past).
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.nhead * self.head_dim)
        return self.o_proj(out), past_key_value


class HAAttentionWrapper(nn.Module):
    """Wraps a hierarchical attention impl, computing position_embeddings internally."""

    def __init__(self, layer_idx: int, impl: str = "new", **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        ga_kwargs = dict(kwargs)
        ga_kwargs['layer_idx'] = layer_idx
        self.attn = HA_IMPLS[impl](**ga_kwargs)

    def forward(self, x: torch.Tensor, past_key_value=None,
                cache_position=None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, S, _ = x.shape
        device = x.device

        # Determine positions
        if cache_position is not None:
            pos = cache_position
        else:
            start = 0
            if past_key_value is not None:
                try:
                    start = past_key_value.get_seq_length(self.layer_idx)
                except TypeError:
                    start = past_key_value.get_seq_length()
            pos = torch.arange(start, start + S, device=device)

        # Build position embeddings (cos, sin) in the format HierarchicalGlobalAttention expects
        cos, sin = self._get_rotary(int(pos.max().item()) + 1, device)
        # Select positions: [S, D] -> [B, S, D]
        cos_pos = cos[pos].unsqueeze(0).expand(B, -1, -1)
        sin_pos = sin[pos].unsqueeze(0).expand(B, -1, -1)
        position_embeddings = (cos_pos, sin_pos)

        out, _ = self.attn(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_value=past_key_value,
            cache_position=pos,
        )
        return out, past_key_value

    def _get_rotary(self, seq_len: int, device: torch.device):
        head_dim = self.attn.head_dim
        theta = self.attn.theta
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    # Expose projection parameters at the wrapper level for checkpoint loading
    @property
    def q_proj(self):
        return self.attn.q_proj

    @property
    def k_proj(self):
        return self.attn.k_proj

    @property
    def v_proj(self):
        return self.attn.v_proj

    @property
    def o_proj(self):
        return self.attn.o_proj


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn_module: nn.Module, dff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_value=None,
                cache_position=None) -> Tuple[torch.Tensor, Any]:
        h, past_key_value = self.self_attn(
            self.norm1(x), past_key_value=past_key_value,
            cache_position=cache_position)
        x = x + self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)
        return x, past_key_value


class SmallLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int,
                 kv_heads: int, num_layers: int, dff: int, attn_factory,
                 dropout: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, attn_factory(layer_idx=i), dff, dropout=dropout)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                past_key_value=None, cache_position=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x, past_key_value = layer(x, past_key_value=past_key_value,
                                      cache_position=cache_position)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1))
        return logits, loss, past_key_value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
def build_dense_model(vocab_size: int, pad_token_id: int) -> SmallLM:
    def factory(layer_idx: int):
        return DenseAttentionLayered(
            layer_idx=layer_idx,
            d_model=HIDDEN_DIM, nhead=NUM_HEADS, kv_heads=KV_HEADS,
        )
    return SmallLM(vocab_size, HIDDEN_DIM, NUM_HEADS, KV_HEADS, NUM_LAYERS,
                   DFF, factory, ignore_index=pad_token_id)


def build_ha_model(vocab_size: int, pad_token_id: int,
                   cache_location: str = "vram") -> SmallLM:
    def factory(layer_idx: int):
        return HAAttentionWrapper(
            layer_idx=layer_idx, impl="routed",
            d_model=HIDDEN_DIM, nhead=NUM_HEADS, kv_heads=KV_HEADS,
            dropout=0.0, use_bias_q=False, use_bias_k=False,
            use_bias_v=False, use_bias_o=False, causal=True,
            chunk_size=CHUNK_SIZE, group_size=GROUP_SIZE,
            topk_chunks=TOPK_CHUNKS, topk_groups=TOPK_GROUPS,
            return_router_stats=False, head_dim=HIDDEN_DIM // NUM_HEADS,
            qk_norm=False, cache_location=cache_location,
        )
    return SmallLM(vocab_size, HIDDEN_DIM, NUM_HEADS, KV_HEADS, NUM_LAYERS,
                   DFF, factory, ignore_index=pad_token_id)


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------
def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
        return payload
    raise RuntimeError("Cannot extract state dict from checkpoint.")


def load_checkpoint(model: nn.Module, path: str, strict: bool = True) -> None:
    """Load checkpoint with flexible key normalization."""
    payload = safe_torch_load(path)
    state = extract_model_state(payload)

    def _strip_compile(s):
        return {k.replace("_orig_mod.", ""): v for k, v in s.items()}

    def _map_to_wrapper(s):
        """Map self_attn.X -> self_attn.attn.X for HA wrapper."""
        mapped = {}
        for k, v in s.items():
            k = k.replace("_orig_mod.", "")
            if ".self_attn." in k and ".self_attn.attn." not in k:
                parts = k.split(".self_attn.")
                mapped[parts[0] + ".self_attn.attn." + parts[1]] = v
            else:
                mapped[k] = v
        return mapped

    model_keys = set(model.state_dict().keys())

    # Try each normalization, pick the one with best key overlap
    candidates = [
        ("strip_compile", _strip_compile(state)),
        ("as_is", state),
        ("wrapper_map", _map_to_wrapper(state)),
    ]

    best_candidate = None
    best_overlap = -1
    for name, candidate in candidates:
        overlap = len(model_keys & set(candidate.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = (name, candidate)

    if best_candidate is None:
        raise RuntimeError(f"Could not load checkpoint {path}")

    name, candidate = best_candidate
    result = model.load_state_dict(candidate, strict=strict)
    matched = len(model_keys & set(candidate.keys()))
    print(f"  Loaded ({name}): {matched}/{len(model_keys)} keys matched")
    if not strict and result.missing_keys:
        print(f"  Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class FineWebIterable(IterableDataset):
    def __init__(self, tokenizer, max_len: int, parquet_files: List[str]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = load_dataset(
            "parquet", data_files={"train": parquet_files}, streaming=True,
        )["train"]

    def __iter__(self):
        buf: List[int] = []
        for example in self.dataset:
            tokens = self.tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            if len(tokens) < 128:
                continue
            buf += tokens
            while len(buf) > self.max_len:
                chunk = buf[:self.max_len + 1]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))
                buf = buf[self.max_len + 1:]
            if len(buf) < 128:
                buf = []


def get_data_files(data_dir: str, n_files: int = 2) -> List[str]:
    os.makedirs(data_dir, exist_ok=True)
    files = []
    for i in range(n_files):
        filename = f"sample/10BT/{i:03d}_00000.parquet"
        local_path = os.path.join(data_dir, filename)
        files.append(local_path)
        if not os.path.exists(local_path):
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id="HuggingFaceFW/fineweb", filename=filename,
                repo_type="dataset", local_dir=data_dir,
            )
    return files


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------
def finetune_ha(args):
    """Fine-tune HA model from dense checkpoint."""
    print("=" * 60)
    print("FINE-TUNING HA MODEL FROM DENSE CHECKPOINT")
    print("=" * 60)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Build HA model and load dense weights
    model = build_ha_model(vocab_size, tokenizer.pad_token_id,
                           cache_location=args.cache_location)
    print(f"HA model parameters: {model.count_parameters():,}")

    dense_path = args.dense_checkpoint
    if not os.path.exists(dense_path):
        print(f"Dense checkpoint not found at {dense_path}")
        print("Run: python ../../prepare_model.py")
        return

    print(f"Loading dense checkpoint: {dense_path}")
    load_checkpoint(model, dense_path, strict=False)

    model.to(DEVICE)
    model.train()

    # Only train attention parameters (k/q projections + any HA-specific params)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "self_attn" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Data
    seq_len = min(args.max_len, MAX_LEN)
    parquet_files = get_data_files(args.data_dir, n_files=args.n_files)
    dataset = FineWebIterable(tokenizer, seq_len, parquet_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1,
                            pin_memory=(DEVICE == "cuda"), prefetch_factor=2,
                            persistent_workers=True)

    tokens_per_step = args.batch_size * args.accum_steps * seq_len
    total_steps = args.target_tokens // tokens_per_step
    print(f"Total steps: {total_steps}, tokens/step: {tokens_per_step}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    data_iter = iter(dataloader)
    opt_step = 0
    micro_step = 0
    step_loss = 0.0
    start_time = time.time()

    # The HA training (no-cache) path uses a fused Triton kernel that requires
    # fp32 inputs, so fine-tuning runs in fp32 (no bf16 autocast).
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pbar = tqdm(total=total_steps, desc="HA fine-tune")
    while opt_step < total_steps:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        _, loss, _ = model(inputs, labels=targets)

        (loss / args.accum_steps).backward()
        step_loss += loss.item()
        micro_step += 1

        if micro_step % args.accum_steps != 0:
            continue

        avg_loss = step_loss / args.accum_steps
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        opt_step += 1
        step_loss = 0.0

        if opt_step % 50 == 0:
            elapsed = time.time() - start_time
            tps = opt_step * tokens_per_step / elapsed
            pbar.set_postfix(loss=f"{avg_loss:.4f}", tps=f"{tps:.0f}")

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time
    print(f"Fine-tuning done in {elapsed/60:.1f} min, final loss={avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), HA_CHECKPOINT)
    print(f"Saved HA checkpoint: {HA_CHECKPOINT}")


# -----------------------------------------------------------------------------
# Generation utilities
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_tokens_dense(model: SmallLM, input_ids: torch.Tensor,
                          num_tokens: int) -> Tuple[List[int], float]:
    """Generate tokens one-by-one with KV cache. Returns tokens and total time."""
    model.eval()
    device = input_ids.device
    B = input_ids.shape[0]
    assert B == 1

    generated = []
    cache = DynamicCache()

    # Prefill
    cache_position = torch.arange(input_ids.shape[1], device=device)
    logits, _, cache = model(input_ids, past_key_value=cache, cache_position=cache_position)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token.item())

    # Generate
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    for i in range(num_tokens - 1):
        pos = input_ids.shape[1] + i
        cache_position = torch.tensor([pos], device=device)
        logits, _, cache = model(
            next_token.unsqueeze(1), past_key_value=cache, cache_position=cache_position)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    return generated, elapsed


@torch.no_grad()
def generate_tokens_ha(model: SmallLM, input_ids: torch.Tensor,
                       num_tokens: int) -> Tuple[List[int], float]:
    """Generate tokens one-by-one with HGA cache. Returns tokens and total time."""
    model.eval()
    device = input_ids.device
    B = input_ids.shape[0]
    assert B == 1

    generated = []
    cache = DynamicCache()

    # Prefill
    cache_position = torch.arange(input_ids.shape[1], device=device)
    logits, _, cache = model(input_ids, past_key_value=cache, cache_position=cache_position)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token.item())

    # Generate
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    for i in range(num_tokens - 1):
        pos = input_ids.shape[1] + i
        cache_position = torch.tensor([pos], device=device)
        logits, _, cache = model(
            next_token.unsqueeze(1), past_key_value=cache, cache_position=cache_position)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    return generated, elapsed


@torch.no_grad()
def measure_prefill_speed(model: SmallLM, input_ids: torch.Tensor,
                          use_cache: bool = True) -> Tuple[float, Any]:
    """Measure prefill time. Returns (time_seconds, cache)."""
    model.eval()
    device = input_ids.device
    cache = DynamicCache() if use_cache else None
    cache_position = torch.arange(input_ids.shape[1], device=device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    _, _, cache = model(input_ids, past_key_value=cache, cache_position=cache_position)

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    return elapsed, cache


@torch.no_grad()
def compute_perplexity(model: SmallLM, input_ids: torch.Tensor,
                       targets: torch.Tensor) -> float:
    """Compute perplexity on a batch (no cache, full sequence)."""
    model.eval()
    logits, loss, _ = model(input_ids, labels=targets)
    return math.exp(loss.item())


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
def _avg(xs):
    return sum(xs) / len(xs)


def benchmark(args):
    """Compare dense and one/both HA implementations on generation speed and quality."""
    print("=" * 60)
    print("BENCHMARK: Dense vs Hierarchical Global Attention")
    print("=" * 60)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    if not os.path.exists(args.dense_checkpoint):
        print(f"ERROR: Dense checkpoint not found: {args.dense_checkpoint}")
        print("Run: python ../../prepare_model.py")
        return

    # --- Build + load models -------------------------------------------------
    print("\n[1] Building models...")
    dense_model = build_dense_model(vocab_size, tokenizer.pad_token_id)
    load_checkpoint(dense_model, args.dense_checkpoint, strict=False)
    dense_model.to(DEVICE).eval()

    ha_models = {}  # name -> model
    m = build_ha_model(vocab_size, tokenizer.pad_token_id,
                       cache_location=args.cache_location)
    ha_ckpt = args.ha_checkpoint
    if os.path.exists(ha_ckpt):
        load_checkpoint(m, ha_ckpt, strict=False)
    else:
        load_checkpoint(m, args.dense_checkpoint, strict=False)
    ha_models["HA-Routed"] = m.to(DEVICE).eval()

    print(f"  Dense params: {dense_model.count_parameters():,}")
    print(f"  HA impls: {list(ha_models)}  |  cache_location={args.cache_location}  |  Device: {DEVICE}")

    # --- Test data -----------------------------------------------------------
    print("\n[2] Preparing test data...")
    parquet_files = get_data_files(args.data_dir, n_files=1)
    dataset = FineWebIterable(tokenizer, args.context_len + args.gen_tokens + 1, parquet_files)
    data_iter = iter(dataset)
    test_seqs = []
    for _ in range(args.num_samples):
        try:
            test_seqs.append(next(data_iter))
        except StopIteration:
            break
    if not test_seqs:
        print("ERROR: Could not load test data")
        return
    print(f"  Loaded {len(test_seqs)} test sequences, length={test_seqs[0][0].shape[0]}")

    context_len = min(args.context_len, test_seqs[0][0].shape[0] - args.gen_tokens)
    gen_tokens = args.gen_tokens
    eval_len = min(args.context_len + args.gen_tokens, test_seqs[0][0].shape[0])

    all_models = {"Dense": dense_model, **ha_models}

    # --- Test 1: Perplexity --------------------------------------------------
    print("\n[3] Perplexity (full sequence, teacher-forced)...")
    ppls = {name: [] for name in all_models}
    for inp, tgt in test_seqs:
        inp_dev = inp[:eval_len].unsqueeze(0).to(DEVICE)
        tgt_dev = tgt[:eval_len].unsqueeze(0).to(DEVICE)
        for name, model in all_models.items():
            ppls[name].append(compute_perplexity(model, inp_dev, tgt_dev))
    for name in all_models:
        print(f"  {name:14s} ppl = {_avg(ppls[name]):.2f}")

    # --- Test 2: Token-by-token generation -----------------------------------
    print(f"\n[4] Generation: {gen_tokens} tokens from {context_len}-token context...")
    warmup_input = test_seqs[0][0][:16].unsqueeze(0).to(DEVICE)
    for model in all_models.values():
        _ = generate_tokens_dense(model, warmup_input, 4)

    gen_times = {name: [] for name in all_models}
    gen_first = {name: None for name in all_models}
    for idx, (inp, tgt) in enumerate(test_seqs):
        context = inp[:context_len].unsqueeze(0).to(DEVICE)
        for name, model in all_models.items():
            toks, dt = generate_tokens_dense(model, context, gen_tokens)
            gen_times[name].append(dt)
            if idx == 0:
                gen_first[name] = toks
    # agreement vs dense
    dense_toks = gen_first["Dense"]
    for name in all_models:
        tps = gen_tokens / _avg(gen_times[name])
        if name == "Dense":
            print(f"  {name:14s} {_avg(gen_times[name])*1000:7.1f} ms  {tps:7.1f} tok/s")
        else:
            agree = sum(1 for a, b in zip(dense_toks, gen_first[name]) if a == b) / len(dense_toks)
            print(f"  {name:14s} {_avg(gen_times[name])*1000:7.1f} ms  {tps:7.1f} tok/s   "
                  f"agree-vs-dense {agree*100:.0f}%")

    # --- Test 3: Prefill speed -----------------------------------------------
    print(f"\n[5] Prefill speed (context_len={context_len})...")
    prefill_times = {name: [] for name in all_models}
    for inp, _ in test_seqs:
        context = inp[:context_len].unsqueeze(0).to(DEVICE)
        for name, model in all_models.items():
            dt, _ = measure_prefill_speed(model, context)
            prefill_times[name].append(dt)
    for name in all_models:
        print(f"  {name:14s} {_avg(prefill_times[name])*1000:7.1f} ms  "
              f"({context_len/_avg(prefill_times[name]):.0f} tok/s)")

    # --- Summary + old-vs-new verdict ---------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in all_models:
        print(f"  {name:14s} ppl {_avg(ppls[name]):6.2f} | "
              f"gen {gen_tokens/_avg(gen_times[name]):6.1f} tok/s | "
              f"prefill {context_len/_avg(prefill_times[name]):6.0f} tok/s")

    if "Dense" in all_models and "HA-Routed" in all_models:
        d_gen = gen_tokens / _avg(gen_times["Dense"])
        ha_gen = gen_tokens / _avg(gen_times["HA-Routed"])
        d_pf = context_len / _avg(prefill_times["Dense"])
        ha_pf = context_len / _avg(prefill_times["HA-Routed"])
        print(f"\n  Decode:  Dense {d_gen:.1f} tok/s  HA-Routed {ha_gen:.1f} tok/s  "
              f"-> {ha_gen/d_gen:.2f}x")
        print(f"  Prefill: Dense {d_pf:.0f} tok/s  HA-Routed {ha_pf:.0f} tok/s  "
              f"-> {ha_pf/d_pf:.2f}x")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Dense vs HA generation")
    parser.add_argument("--mode", choices=["finetune", "benchmark"], default="benchmark")
    parser.add_argument("--ha-impl", choices=["routed"], default="routed",
                        help="Hierarchical attention implementation (exact-chunk-routed).")
    parser.add_argument("--cache-location", choices=["vram", "ram"], default="vram",
                        help="Generation KV-cache tier for the new routed impl.")

    # Paths
    parser.add_argument("--dense-checkpoint", type=str, default=DENSE_CHECKPOINT)
    parser.add_argument("--ha-checkpoint", type=str, default=HA_CHECKPOINT)
    parser.add_argument("--data-dir", type=str, default="fineweb_sample")

    # Benchmark params
    parser.add_argument("--context-len", type=int, default=512,
                        help="Number of context tokens for generation")
    parser.add_argument("--gen-tokens", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of test sequences")

    # Fine-tune params
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n-files", type=int, default=2)

    args = parser.parse_args()

    if args.mode == "finetune":
        finetune_ha(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
