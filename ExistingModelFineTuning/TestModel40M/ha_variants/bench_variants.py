#!/usr/bin/env python3
"""Benchmark harness for HA attention variants (decode speed + quality).

For each variant module (a file exporting ``GlobalAttention``) this:
  * builds the 40M SmallLM with that attention,
  * loads the HA finetuned checkpoint,
  * measures decode (token-by-token) throughput,
  * measures decode fidelity vs the fused prefill path (argmax agreement +
    logit MSE) and perplexity.

The dense model is included as a speed/quality reference.

Usage:
  python bench_variants.py --variants v0_baseline v1_fast --context-len 512 --gen-tokens 128
"""
import argparse
import importlib.util
import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast, DynamicCache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # ha_variants
TM_DIR = os.path.dirname(SCRIPT_DIR)                         # TestModel40M
EMFT_DIR = os.path.dirname(TM_DIR)                           # ExistingModelFineTuning
ROOT_DIR = os.path.dirname(EMFT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TM_DIR)

import benchmark_generation as bg  # reuse SmallLM, dense model, data, checkpoint utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Efficient HA wrapper: precomputes a rotary cache, no per-token rebuild / sync.
# -----------------------------------------------------------------------------
class FastHAWrapper(nn.Module):
    _rope_cache = {}  # (head_dim, theta, device) -> (cos[L,D], sin[L,D]), grown lazily

    def __init__(self, GlobalAttention, layer_idx: int, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        ga_kwargs = dict(kwargs)
        ga_kwargs['layer_idx'] = layer_idx
        self.attn = GlobalAttention(**ga_kwargs)

    def _rotary(self, end: int, device):
        head_dim = self.attn.head_dim
        theta = self.attn.theta
        key = (head_dim, theta, str(device))
        cached = FastHAWrapper._rope_cache.get(key)
        target = max(end, 8192)
        if cached is None or cached[0].shape[0] < target:
            L = target
            half = head_dim // 2
            inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
            t = torch.arange(L, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cached = (emb.cos(), emb.sin())
            FastHAWrapper._rope_cache[key] = cached
        return cached

    def forward(self, x, past_key_value=None, cache_position=None, **kwargs):
        B, S, _ = x.shape
        device = x.device
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

        # Index a pre-grown rope table (no .item() sync -> CUDA-graph safe).
        cos_t, sin_t = self._rotary(8192, device)
        cos = cos_t[pos].unsqueeze(0).expand(B, -1, -1)
        sin = sin_t[pos].unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(
            hidden_states=x, position_embeddings=(cos, sin), attention_mask=None,
            past_key_value=past_key_value, cache_position=pos,
        )
        return out, past_key_value

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


def load_variant(name: str):
    path = os.path.join(SCRIPT_DIR, f"{name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"ha_variant_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GlobalAttention


def build_ha_model_variant(GlobalAttention, vocab_size: int, pad_token_id: int) -> bg.SmallLM:
    def factory(layer_idx: int):
        return FastHAWrapper(
            GlobalAttention, layer_idx=layer_idx,
            d_model=bg.HIDDEN_DIM, nhead=bg.NUM_HEADS, kv_heads=bg.KV_HEADS,
            dropout=0.0, use_bias_q=False, use_bias_k=False,
            use_bias_v=False, use_bias_o=False, causal=True,
            chunk_size=bg.CHUNK_SIZE, group_size=bg.GROUP_SIZE,
            topk_chunks=bg.TOPK_CHUNKS, topk_groups=bg.TOPK_GROUPS,
            return_router_stats=False, head_dim=bg.HIDDEN_DIM // bg.NUM_HEADS,
            qk_norm=False,
        )
    return bg.SmallLM(vocab_size, bg.HIDDEN_DIM, bg.NUM_HEADS, bg.KV_HEADS,
                      bg.NUM_LAYERS, bg.DFF, factory, ignore_index=pad_token_id)


# -----------------------------------------------------------------------------
# Generation / quality helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def decode_generate(model, input_ids, num_tokens) -> Tuple[List[int], float]:
    device = input_ids.device
    cache = DynamicCache()
    cache_position = torch.arange(input_ids.shape[1], device=device)
    logits, _, cache = model(input_ids, past_key_value=cache, cache_position=cache_position)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = [next_token.item()]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(num_tokens - 1):
        pos = input_ids.shape[1] + i
        cache_position = torch.tensor([pos], device=device)
        logits, _, cache = model(next_token.unsqueeze(1), past_key_value=cache,
                                 cache_position=cache_position)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return generated, time.perf_counter() - start


_HGA_BUFFERS = (
    "chunk_k", "group_k", "cur_chunk_raw", "cur_chunk_rope",
    "cur_group_raw", "cur_group_rope", "chunk_smax", "group_smax", "kbuf", "vbuf",
)


@torch.no_grad()
def _snapshot_hga(cache):
    snaps = []
    for st in (getattr(cache, "_hga", None) or []):
        if st is None:
            continue
        d = {}
        for name in _HGA_BUFFERS:
            t = getattr(st, name, None)
            if t is not None:
                d[name] = (t, t.clone())
        snaps.append(d)
    return snaps


@torch.no_grad()
def _restore_hga(snaps):
    for d in snaps:
        for _name, (t, c) in d.items():
            t.copy_(c)


@torch.no_grad()
def decode_generate_cudagraph(model, input_ids, num_tokens) -> Tuple[List[int], float]:
    """Generate with a captured CUDA graph for the per-token decode step.

    Requires the variant's decode path to be static-shape and sync-free
    (v2_static and descendants).  Returns (tokens, steady-state replay time).
    """
    device = input_ids.device
    P = input_ids.shape[1]
    cache = DynamicCache()
    logits, _, cache = model(input_ids, past_key_value=cache,
                             cache_position=torch.arange(P, device=device))
    t0 = logits[:, -1, :].argmax(dim=-1)  # [1]

    if not (getattr(cache, "_hga", None)):
        raise RuntimeError("variant did not seed HGA decode state; cannot CUDA-graph")

    snaps = _snapshot_hga(cache)
    static_token = t0.view(1, 1).clone()
    static_pos = torch.tensor([P], device=device, dtype=torch.long)

    # Warmup (side stream) then restore mutated state before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(static_token, past_key_value=cache, cache_position=static_pos)
    torch.cuda.current_stream().wait_stream(s)
    _restore_hga(snaps)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_logits, _, _ = model(static_token, past_key_value=cache,
                                    cache_position=static_pos)

    # Clean timed replay run from the post-prefill state.
    _restore_hga(snaps)
    cur = t0
    out_tokens = torch.empty(num_tokens - 1, dtype=torch.long, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(num_tokens - 1):
        static_token.copy_(cur.view(1, 1))
        static_pos.fill_(P + i)
        g.replay()
        cur = static_logits[:, -1, :].argmax(dim=-1)
        out_tokens[i] = cur
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return [t0.item()] + out_tokens.tolist(), elapsed


@torch.no_grad()
def decode_logits_teacher_forced(model, input_ids, prefill_len) -> torch.Tensor:
    """Return decode-path logits for positions [prefill_len .. L-1] feeding the
    ground-truth tokens (teacher forced)."""
    device = input_ids.device
    L = input_ids.shape[1]
    cache = DynamicCache()
    cache_position = torch.arange(prefill_len, device=device)
    logits, _, cache = model(input_ids[:, :prefill_len], past_key_value=cache,
                             cache_position=cache_position)
    out = [logits[:, -1, :]]
    for pos in range(prefill_len, L - 1):
        cache_position = torch.tensor([pos], device=device)
        logits, _, cache = model(input_ids[:, pos:pos + 1], past_key_value=cache,
                                 cache_position=cache_position)
        out.append(logits[:, -1, :])
    return torch.stack(out, dim=1)  # [1, L-prefill_len, V]


@torch.no_grad()
def prefill_logits(model, input_ids) -> torch.Tensor:
    logits, _, _ = model(input_ids)
    return logits


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["v0_baseline"])
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--include-dense", action="store_true")
    ap.add_argument("--check-quality", action="store_true")
    ap.add_argument("--cudagraph", action="store_true",
                    help="also measure CUDA-graph decode for static variants")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Test data
    parquet_files = bg.get_data_files(os.path.join(TM_DIR, "fineweb_sample"), n_files=1)
    total_len = args.context_len + args.gen_tokens + 1
    dataset = bg.FineWebIterable(tok, total_len, parquet_files)
    it = iter(dataset)
    seqs = []
    for _ in range(args.num_samples):
        try:
            inp, tgt = next(it)
            seqs.append(inp)
        except StopIteration:
            break
    print(f"Loaded {len(seqs)} sequences of length {seqs[0].shape[0]}")

    ctx_len = args.context_len
    gen = args.gen_tokens

    def speed_quality(model, label, is_ha):
        # warmup
        w = seqs[0][:16].unsqueeze(0).to(DEVICE)
        decode_generate(model, w, 4)
        times = []
        for inp in seqs:
            context = inp[:ctx_len].unsqueeze(0).to(DEVICE)
            _, t = decode_generate(model, context, gen)
            times.append(t)
        avg = sum(times) / len(times)
        print(f"  [{label}] decode: {avg*1000:.1f} ms, {gen/avg:.1f} tok/s")

        if args.check_quality and is_ha:
            agall, mse_all = [], []
            for inp in seqs:
                seq = inp[:ctx_len + gen].unsqueeze(0).to(DEVICE)
                pf = prefill_logits(model, seq)[:, ctx_len - 1:-1, :]  # predicts ctx..end
                dl = decode_logits_teacher_forced(model, seq, ctx_len)
                n = min(pf.shape[1], dl.shape[1])
                a = (pf[:, :n].argmax(-1) == dl[:, :n].argmax(-1)).float().mean().item()
                mse = (pf[:, :n].float() - dl[:, :n].float()).pow(2).mean().item()
                agall.append(a)
                mse_all.append(mse)
            print(f"  [{label}] decode-vs-prefill argmax agree: {100*sum(agall)/len(agall):.2f}%, "
                  f"logit MSE: {sum(mse_all)/len(mse_all):.4e}")
        return gen / avg

    def cudagraph_speed_quality(model, label):
        # Reference tokens from eager static decode (exact to baseline).
        ref_ctx = seqs[0][:ctx_len].unsqueeze(0).to(DEVICE)
        ref_tokens, _ = decode_generate(model, ref_ctx, gen)
        try:
            times, match = [], []
            for inp in seqs:
                context = inp[:ctx_len].unsqueeze(0).to(DEVICE)
                cg_tokens, t = decode_generate_cudagraph(model, context, gen)
                times.append(t)
            avg = sum(times) / len(times)
            cg_tokens, _ = decode_generate_cudagraph(model, ref_ctx, gen)
            m = sum(1 for a, b in zip(ref_tokens, cg_tokens) if a == b) / len(ref_tokens)
            print(f"  [{label}+cudagraph] decode: {avg*1000:.1f} ms, {gen/avg:.1f} tok/s; "
                  f"token match vs eager: {100*m:.2f}%")
            return gen / avg
        except Exception as e:
            print(f"  [{label}+cudagraph] FAILED: {repr(e)[:200]}")
            return None

    results = {}
    if args.include_dense:
        dense = bg.build_dense_model(tok.vocab_size, tok.pad_token_id)
        bg.load_checkpoint(dense, bg.DENSE_CHECKPOINT, strict=False)
        dense.to(DEVICE).eval()
        results["dense"] = speed_quality(dense, "dense", is_ha=False)
        del dense
        torch.cuda.empty_cache()

    for name in args.variants:
        GA = load_variant(name)
        model = build_ha_model_variant(GA, tok.vocab_size, tok.pad_token_id)
        bg.load_checkpoint(model, bg.HA_CHECKPOINT, strict=False)
        model.to(DEVICE).eval()
        # Tighten static decode buffers for this run (variants that support it).
        for layer in model.layers:
            if hasattr(layer.self_attn.attn, "decode_max_seq"):
                layer.self_attn.attn.decode_max_seq = ctx_len + gen + bg.CHUNK_SIZE
        results[name] = speed_quality(model, name, is_ha=True)
        if args.cudagraph and hasattr(model.layers[0].self_attn.attn, "decode_max_seq"):
            r = cudagraph_speed_quality(model, name)
            if r is not None:
                results[name + "+cudagraph"] = r
        del model
        torch.cuda.empty_cache()

    print("\n=== SUMMARY (decode tok/s) ===")
    base = results.get("v0_baseline")
    for k, v in results.items():
        rel = f" ({v/base:.2f}x vs baseline)" if base and k != "v0_baseline" else ""
        print(f"  {k:20s}: {v:8.1f} tok/s{rel}")


if __name__ == "__main__":
    main()
