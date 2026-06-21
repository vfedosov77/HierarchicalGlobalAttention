#!/usr/bin/env python3
"""Compare the next-token loss of the fine-tuned 40M checkpoint under three
attention regimes on a single real FineWeb sequence:

  * "dense"            : native dense causal RoPE attention
                         (``GlobalAttention(use_global=False)`` -> SDPA).
  * "ha+summaries"     : the hierarchical Exact-Q fused attention with the
                         group summaries kept in the softmax (original).
  * "ha_token_only"    : the same hierarchical routing, but with the group
                         summaries removed from the attention scores so only
                         exact token K/V contribute (``attend_summaries=False``)
                         -- mirrors ``HierarchicalGlobalAttentionRouted``.

The hypothesis is that the summaries inject approximate vectors a dense
checkpoint never trained on, so removing them brings the hierarchical loss back
close to the native dense loss.

Run:
    python benchmark_no_summaries_loss.py \
        --checkpoint ../NotOptimized/ha_finetuned_from_dense.pt \
        --seq-len 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# repo root = Compiled -> TestModel40M -> ExistingModelFineTuning -> <root>
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from HierarchicalGlobalAttentionFusedExactQ import GlobalAttention

try:
    from ExistingModelFineTuning.HierarchicalGlobalAttentionRouted import (
        HierarchicalGlobalAttentionRouted,
    )
    _HAS_ROUTED = True
except Exception as _exc:  # pragma: no cover - optional dependency (KvRouter)
    HierarchicalGlobalAttentionRouted = None  # type: ignore
    _HAS_ROUTED = False
    print(f"[warn] routed impl unavailable: {_exc}")

# -----------------------------------------------------------------------------
# Architecture (identical to finetune_small_model.py / benchmark_train_vs_generate.py)
# -----------------------------------------------------------------------------
HIDDEN_DIM = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
VOCAB_SIZE = 50257

CHUNK_SIZE = 64
GROUP_SIZE = 16
TOPK_CHUNKS = 20
TOPK_GROUPS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT = os.path.join(SCRIPT_DIR, "..", "NotOptimized", "ha_finetuned_from_dense.pt")


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


class AttnWrapper(nn.Module):
    """Nests the attention under ``.attn`` so checkpoint keys match
    ``layers.<i>.self_attn.attn.<q,k,v,o>_proj.weight`` and lets the attention
    build its own RoPE (rotary_data=None) with the trained theta."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class RoutedWrapper(nn.Module):
    """Faithful copy of ``benchmark_generation.HAAttentionWrapper`` (it nests the
    routed attention under ``.attn`` and builds RoPE internally).  ``use_cache``
    selects which path of the routed module runs:

    * ``False`` -> ``past_key_value=None`` -> the routed module's training/eval
      branch, which is **plain causal SDPA (full dense)**.  This is exactly what
      ``benchmark_generation.compute_perplexity`` measures.
    * ``True``  -> a fresh ``DynamicCache`` per call -> the sparse chunk-routed
      prefill path actually runs.
    """

    def __init__(self, attn: nn.Module, use_cache: bool):
        super().__init__()
        self.attn = attn
        self.use_cache = use_cache

    def _get_rotary(self, seq_len: int, device: torch.device):
        head_dim = self.attn.head_dim
        theta = self.attn.theta
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        device = x.device
        cos, sin = self._get_rotary(S, device)
        pe = (cos.unsqueeze(0).expand(B, -1, -1), sin.unsqueeze(0).expand(B, -1, -1))
        if self.use_cache:
            from transformers import DynamicCache
            cache = DynamicCache()
            pos = torch.arange(S, device=device)
            out, _ = self.attn(hidden_states=x, position_embeddings=pe,
                               past_key_value=cache, cache_position=pos)
        else:
            out, _ = self.attn(hidden_states=x, position_embeddings=pe,
                               past_key_value=None)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn_module: nn.Module, dff: int):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, attn_factory, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList(
            [DecoderLayer(HIDDEN_DIM, attn_factory(i), DFF) for i in range(NUM_LAYERS)]
        )
        self.final_norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
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
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1)
            )
        return logits, loss


def build_routed_model(use_cache: bool,
                       topk_chunks: int = TOPK_CHUNKS,
                       topk_groups: int = TOPK_GROUPS) -> SmallLM:
    def attn_factory(layer_idx: int):
        attn = HierarchicalGlobalAttentionRouted(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=0.0,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=topk_chunks,
            topk_groups=topk_groups,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            qk_norm=False,
            layer_idx=layer_idx,
            cache_location="vram",
            num_layers=NUM_LAYERS,
        )
        return RoutedWrapper(attn, use_cache=use_cache)

    return SmallLM(attn_factory)


def build_model(use_global: bool, attend_summaries: bool,
                topk_chunks: int = TOPK_CHUNKS, topk_groups: int = TOPK_GROUPS) -> SmallLM:
    def attn_factory(layer_idx: int):
        attn = GlobalAttention(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=0.0,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            use_global=use_global,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=topk_chunks,
            topk_groups=topk_groups,
            return_router_stats=False,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            q_norm=None,
            k_norm=None,
        )
        attn.attend_summaries = attend_summaries
        return AttnWrapper(attn)

    return SmallLM(attn_factory)


# -----------------------------------------------------------------------------
# Checkpoint / data helpers
# -----------------------------------------------------------------------------
def load_checkpoint(model: nn.Module, path: str, remap_self_attn: bool = False) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    state = {k.replace("_orig_mod.", "", 1): v for k, v in payload.items()}
    if remap_self_attn:
        # Fine-tune checkpoints store attention directly under ``self_attn``
        # (no AttnWrapper), so insert the ``.attn`` infix this benchmark uses.
        remapped = {}
        for k, v in state.items():
            if ".self_attn." in k and ".self_attn.attn." not in k:
                k = k.replace(".self_attn.", ".self_attn.attn.", 1)
            remapped[k] = v
        state = remapped
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The only acceptable "missing" key is the tied lm_head weight.
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing:
        raise RuntimeError(f"Missing checkpoint tensors: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors: {unexpected[:10]}")
    matched = len(model.state_dict()) - len(missing)
    print(f"[load] matched {matched}/{len(model.state_dict())} tensors "
          f"(missing tied lm_head: {'lm_head.weight' in missing})")


def get_fineweb_sequence(seq_len: int, doc_index: int) -> torch.Tensor:
    """Return ``seq_len + 1`` contiguous GPT-2 tokens from one FineWeb doc."""
    import pandas as pd
    from transformers import GPT2TokenizerFast

    candidates = [
        os.path.join(SCRIPT_DIR, "..", "NotOptimized", "fineweb_sample",
                     "sample", "10BT", "000_00000.parquet"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "fineweb_sample",
                     "sample", "10BT", "000_00000.parquet"),
    ]
    parquet = next((p for p in candidates if os.path.exists(p)), None)
    if parquet is None:
        raise FileNotFoundError(f"No FineWeb parquet found. Tried: {candidates}")
    print(f"[data] reading {os.path.abspath(parquet)}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    df = pd.read_parquet(parquet, columns=["text"])
    need = seq_len + 1
    skipped = 0
    for i in range(doc_index, len(df)):
        text = df["text"].iloc[i]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= need:
            print(f"[data] using doc #{i} (len={len(ids)} tokens), skipped {skipped} short docs")
            return torch.tensor(ids[:need], dtype=torch.long)
        skipped += 1
    raise RuntimeError(f"No document with >= {need} tokens found from index {doc_index}.")


@torch.no_grad()
def eval_loss(model: SmallLM, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
    model.eval()
    logits, loss = model(inputs, labels=targets)
    # per-token NLL for diagnostics
    nll = nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size).float(),
        targets.reshape(-1),
        reduction="none",
    )
    return float(loss.item()), nll.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ha-checkpoint", default=None,
                        help="Optional separate checkpoint loaded into the HA/routed "
                             "regimes (e.g. a QK fine-tuned model). Its "
                             "'self_attn.*' keys are remapped to 'self_attn.attn.*'. "
                             "Dense always uses --checkpoint.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--doc-index", type=int, default=0)
    parser.add_argument("--topk-chunks", type=int, default=TOPK_CHUNKS)
    parser.add_argument("--topk-groups", type=int, default=TOPK_GROUPS)
    args = parser.parse_args()

    if DEVICE != "cuda":
        print("[warn] CUDA not available; the fused HA path requires CUDA/fp32.")

    torch.manual_seed(0)
    seq = get_fineweb_sequence(args.seq_len, args.doc_index)
    inputs = seq[:-1].unsqueeze(0).to(DEVICE)
    targets = seq[1:].unsqueeze(0).to(DEVICE)
    print(f"[data] seq_len={inputs.shape[1]} (input) on {DEVICE}\n")

    regimes = [
        ("dense",                 "fused",  dict(use_global=False, attend_summaries=True)),
        ("ha+summaries",          "fused",  dict(use_global=True,  attend_summaries=True)),
        ("ha_token_only",         "fused",  dict(use_global=True,  attend_summaries=False)),
    ]
    if _HAS_ROUTED:
        # The HAAttentionWrapper from benchmark_generation.py.  No-cache == what
        # that script's perplexity actually measures (dense SDPA fallback);
        # with-cache == the real sparse chunk-routed path.
        regimes += [
            ("ha_routed(no-cache)",   "routed", dict(use_cache=False)),
            ("ha_routed(cache)",      "routed", dict(use_cache=True)),
        ]

    print(f"[cfg] topk_chunks={args.topk_chunks} topk_groups={args.topk_groups} "
          f"(chunk={CHUNK_SIZE} group={GROUP_SIZE})\n")

    results = {}
    dense_nll = None
    for name, kind, cfg in regimes:
        if kind == "routed":
            model = build_routed_model(topk_chunks=args.topk_chunks,
                                       topk_groups=args.topk_groups, **cfg).to(DEVICE)
        else:
            model = build_model(topk_chunks=args.topk_chunks,
                                topk_groups=args.topk_groups, **cfg).to(DEVICE)
        model = model.float()
        if name != "dense" and args.ha_checkpoint:
            load_checkpoint(model, args.ha_checkpoint, remap_self_attn=True)
        else:
            load_checkpoint(model, args.checkpoint)
        try:
            loss, nll = eval_loss(model, inputs, targets)
        except Exception as exc:
            print(f"[{name:>20}] FAILED: {exc}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        ppl = float(torch.exp(torch.tensor(loss)))
        try:
            path = model.layers[0].self_attn.attn._last_path
        except Exception:
            path = "n/a"
        results[name] = (loss, ppl, nll)
        if name == "dense":
            dense_nll = nll
        print(f"[{name:>20}] loss={loss:.5f}  ppl={ppl:8.3f}  attn_path={path}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n================ summary (4K FineWeb sequence) ================")
    dl = results["dense"][0]
    print(f"{'regime':>22} | {'loss':>9} | {'ppl':>9} | {'Δloss vs dense':>14}")
    print("-" * 66)
    for name, _, _ in regimes:
        if name not in results:
            continue
        loss, ppl, _ = results[name]
        print(f"{name:>22} | {loss:9.5f} | {ppl:9.3f} | {loss - dl:+14.5f}")

    if dense_nll is not None:
        print("\nmean |per-token NLL diff vs dense|:")
        for name, _, _ in regimes:
            if name == "dense" or name not in results:
                continue
            diff = (results[name][2] - dense_nll).abs().mean().item()
            print(f"  {name:>22}: {diff:.5f}")


if __name__ == "__main__":
    main()
