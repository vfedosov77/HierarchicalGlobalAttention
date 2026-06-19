#!/usr/bin/env python3
"""Self-contained ``torch.compile`` decode benchmark.

This file depends on **nothing in the project except the attention file**
(``v2_static.py``).  It builds a tiny decoder LM with *random* weights (this is a
pure speed benchmark, so quality is irrelevant) and measures single-token decode
throughput for two attention variants:

  * ``dense``  - a standard GQA + RoPE attention with a static KV buffer
                 (written inline here, compile / cuda-graph friendly).
  * ``ha``     - the hierarchical ``GlobalAttention`` from ``v2_static.py``
                 driven through its sync-free static-decode fast path.

For each variant it times three execution modes:

  1. ``eager``       - plain python-driven token loop.
  2. ``compile``     - ``torch.compile(mode="reduce-overhead")`` which lets
                       TorchInductor place the decode step inside CUDA graphs
                       automatically (the "native" alternative to a manual
                       capture).
  3. ``cudagraph``   - a manual ``torch.cuda.CUDAGraph`` capture/replay of the
                       decode step (the proven "superior" reference target).

Run:
    python bench_compile.py
    python bench_compile.py --variants ha --context-len 2048 --gen 256
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- import ONLY the attention class from the attention file -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from v2_static import GlobalAttention  # noqa: E402

try:
    from transformers import DynamicCache  # noqa: E402
except Exception:  # pragma: no cover
    DynamicCache = None  # type: ignore


# ===========================================================================
# Model config (matches the 40M test model; weights are random here)
# ===========================================================================

HIDDEN = 384
NUM_HEADS = 6
KV_HEADS = 2
NUM_LAYERS = 8
DFF = 2048
HEAD_DIM = 64
CHUNK_SIZE = 64
GROUP_SIZE = 16
TOPK_CHUNKS = 20
TOPK_GROUPS = 32
VOCAB = 50257
THETA = 1_000_000.0
DEVICE = "cuda"
DTYPE = torch.float32  # the HA fused prefill path requires fp32


# ===========================================================================
# Shared building blocks
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = self.weight.float() * x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_rope_table(max_seq: int, head_dim: int, theta: float,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()  # [max_seq, head_dim]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D]; cos/sin broadcastable to [.., S, D]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


# ===========================================================================
# Dense attention with a static KV buffer (compile / cuda-graph friendly)
# ===========================================================================

class DenseStaticAttention(nn.Module):
    def __init__(self, max_seq: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.rep = NUM_HEADS // KV_HEADS
        self.scale = HEAD_DIM ** -0.5
        self.max_seq = max_seq
        self.register_buffer("kbuf", torch.zeros(1, KV_HEADS, max_seq, HEAD_DIM, dtype=DTYPE), persistent=False)
        self.register_buffer("vbuf", torch.zeros(1, KV_HEADS, max_seq, HEAD_DIM, dtype=DTYPE), persistent=False)

    def _rep(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.rep == 1 else t.repeat_interleave(self.rep, dim=1)

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos.view(1, 1, S, HEAD_DIM), sin.view(1, 1, S, HEAD_DIM))
        k = apply_rope(k, cos.view(1, 1, S, HEAD_DIM), sin.view(1, 1, S, HEAD_DIM))
        self.kbuf[:, :, :S, :] = k
        self.vbuf[:, :, :S, :] = v
        out = F.scaled_dot_product_attention(q, self._rep(k), self._rep(v), is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, NUM_HEADS * HEAD_DIM))

    def decode(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               pos: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.q_proj(x).view(B, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        cview = cos.view(1, 1, 1, HEAD_DIM)
        sview = sin.view(1, 1, 1, HEAD_DIM)
        q = apply_rope(q, cview, sview)
        k = apply_rope(k, cview, sview)
        idx = pos.view(1)
        self.kbuf.index_copy_(2, idx, k)
        self.vbuf.index_copy_(2, idx, v)
        kk = self._rep(self.kbuf)                              # [B,H,max_seq,D]
        vv = self._rep(self.vbuf)
        scores = torch.einsum("bhd,bhsd->bhs", q.squeeze(2), kk) * self.scale  # [B,H,max_seq]
        mask = torch.arange(self.max_seq, device=x.device) <= pos
        scores = scores.masked_fill(~mask.view(1, 1, -1), float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.einsum("bhs,bhsd->bhd", probs, vv)          # [B,H,D]
        return self.o_proj(out.reshape(B, 1, NUM_HEADS * HEAD_DIM))


# ===========================================================================
# HA attention wrapper (sync-free RoPE table, drives v2_static fast path)
# ===========================================================================

class HAStaticAttention(nn.Module):
    def __init__(self, layer_idx: int, max_seq: int) -> None:
        super().__init__()
        self.attn = GlobalAttention(
            d_model=HIDDEN, nhead=NUM_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            chunk_size=CHUNK_SIZE, group_size=GROUP_SIZE,
            topk_chunks=TOPK_CHUNKS, topk_groups=TOPK_GROUPS,
            use_bias_q=False, use_bias_k=False, use_bias_v=False, use_bias_o=False,
            qk_norm=False, theta=THETA, layer_idx=layer_idx,
            decode_max_seq=max_seq,
        )

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache, pos: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        cos3 = cos.view(1, S, HEAD_DIM).expand(B, S, HEAD_DIM)
        sin3 = sin.view(1, S, HEAD_DIM).expand(B, S, HEAD_DIM)
        out, _ = self.attn(x, (cos3, sin3), None, cache, pos)
        return out

    def decode(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               cache, pos: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cos3 = cos.view(1, 1, HEAD_DIM).expand(B, 1, HEAD_DIM)
        sin3 = sin.view(1, 1, HEAD_DIM).expand(B, 1, HEAD_DIM)
        out, _ = self.attn(x, (cos3, sin3), None, cache, pos.view(1))
        return out


# ===========================================================================
# Tiny decoder LM
# ===========================================================================

class TinyLM(nn.Module):
    def __init__(self, variant: str, max_seq: int) -> None:
        super().__init__()
        self.variant = variant
        self.max_seq = max_seq
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList()
        for li in range(NUM_LAYERS):
            attn = (DenseStaticAttention(max_seq) if variant == "dense"
                    else HAStaticAttention(li, max_seq))
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(HIDDEN),
                "attn": attn,
                "norm2": RMSNorm(HIDDEN),
                "ffn": SwiGLU(HIDDEN, DFF),
            }))
        self.final_norm = RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        cos, sin = build_rope_table(max_seq, HEAD_DIM, THETA, torch.device(DEVICE))
        self.register_buffer("cos_tab", cos, persistent=False)
        self.register_buffer("sin_tab", sin, persistent=False)
        self._cache = None  # HA only

    # -- prefill (eager) -------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        pos = torch.arange(S, device=input_ids.device)
        cos = self.cos_tab[pos]
        sin = self.sin_tab[pos]
        if self.variant == "ha":
            self._cache = DynamicCache()
        for layer in self.layers:
            h = layer["norm1"](x)
            if self.variant == "dense":
                a = layer["attn"].prefill(h, cos, sin)
            else:
                a = layer["attn"].prefill(h, cos, sin, self._cache, pos)
            x = x + a
            x = x + layer["ffn"](layer["norm2"](x))
        x = self.final_norm(x)
        return self.lm_head(x[:, -1:, :])

    # -- single-token decode (this is what gets compiled / captured) -----
    def decode_step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.embed(token)
        idx = pos.view(1)
        cos = self.cos_tab.index_select(0, idx)[0]  # [D], sync-free
        sin = self.sin_tab.index_select(0, idx)[0]
        for layer in self.layers:
            h = layer["norm1"](x)
            if self.variant == "dense":
                a = layer["attn"].decode(h, cos, sin, pos)
            else:
                a = layer["attn"].decode(h, cos, sin, self._cache, pos)
            x = x + a
            x = x + layer["ffn"](layer["norm2"](x))
        x = self.final_norm(x)
        return self.lm_head(x)


# ===========================================================================
# Benchmark drivers
# ===========================================================================

def _greedy_next(logits: torch.Tensor) -> torch.Tensor:
    # logits [B,1,V] -> token [B,1]
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.no_grad()
def run_eager(model: TinyLM, input_ids: torch.Tensor, gen: int) -> Tuple[float, torch.Tensor]:
    B, S = input_ids.shape
    logits = model.prefill(input_ids)
    tok = _greedy_next(logits)
    toks = [tok]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(gen):
        pos = torch.tensor(S + i, device=input_ids.device)
        logits = model.decode_step(tok, pos)
        tok = _greedy_next(logits)
        toks.append(tok)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return gen / dt, torch.cat(toks, dim=1)


@torch.no_grad()
def run_compile(model: TinyLM, input_ids: torch.Tensor, gen: int,
                warmup: int = 8) -> Tuple[float, torch.Tensor]:
    B, S = input_ids.shape
    logits = model.prefill(input_ids)
    tok = _greedy_next(logits)
    decode_fn = torch.compile(model.decode_step, mode="reduce-overhead", fullgraph=False)
    toks = [tok]
    # warmup: triggers compilation + cuda-graph recording (positions advance).
    for i in range(warmup):
        pos = torch.tensor(S + i, device=input_ids.device)
        logits = decode_fn(tok, pos)
        tok = _greedy_next(logits)
        toks.append(tok)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(gen):
        pos = torch.tensor(S + warmup + i, device=input_ids.device)
        logits = decode_fn(tok, pos)
        tok = _greedy_next(logits)
        toks.append(tok)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return gen / dt, torch.cat(toks, dim=1)


@torch.no_grad()
def run_cudagraph(model: TinyLM, input_ids: torch.Tensor, gen: int,
                  warmup: int = 3) -> Tuple[float, torch.Tensor]:
    B, S = input_ids.shape
    logits = model.prefill(input_ids)
    tok0 = _greedy_next(logits)

    static_token = tok0.clone()
    static_pos = torch.tensor(S, device=input_ids.device)

    # Snapshot the clean post-prefill state.  Both warmup and the capture itself
    # execute the decode body once and mutate persistent state, so we restore
    # back to this snapshot before the timed replay loop.
    snap = _snapshot(model)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(warmup):
            static_pos.fill_(S + i)
            _ = model.decode_step(static_token, static_pos)
    torch.cuda.current_stream().wait_stream(s)
    _restore(model, snap)

    g = torch.cuda.CUDAGraph()
    static_pos.fill_(S)
    with torch.cuda.graph(g):
        static_logits = model.decode_step(static_token, static_pos)
    _restore(model, snap)  # undo the capture's one-step mutation

    toks = [tok0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(gen):
        static_pos.fill_(S + i)
        g.replay()
        nxt = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        static_token.copy_(nxt)
        toks.append(nxt.clone())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = torch.cat(toks, dim=1)
    g.reset()
    del g, static_logits
    torch.cuda.synchronize()
    return gen / dt, out


@torch.no_grad()
def _snapshot(model: TinyLM) -> dict:
    snap: dict = {"buf": {}, "hga": None}
    for name, buf in model.named_buffers():
        snap["buf"][name] = buf.clone()
    cache = model._cache
    hga = getattr(cache, "_hga", None) if cache is not None else None
    if hga:
        states = []
        for st in hga:
            if st is None:
                states.append(None)
                continue
            states.append({s: (getattr(st, s).clone() if torch.is_tensor(getattr(st, s)) else None)
                           for s in st.__slots__})
        snap["hga"] = states
    return snap


@torch.no_grad()
def _restore(model: TinyLM, snap: dict) -> None:
    bufs = dict(model.named_buffers())
    for name, val in snap["buf"].items():
        bufs[name].copy_(val)
    if snap["hga"] is not None and model._cache is not None:
        for st, sd in zip(model._cache._hga, snap["hga"]):
            if st is None or sd is None:
                continue
            for s, val in sd.items():
                if val is not None:
                    getattr(st, s).copy_(val)


def build_model(variant: str, max_seq: int) -> TinyLM:
    torch.manual_seed(0)
    model = TinyLM(variant, max_seq).to(DEVICE).to(DTYPE).eval()
    return model


def _run_single(variant: str, mode: str, S: int, gen: int, warmup: int,
                max_seq: int) -> Tuple[float, int]:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, S), device=DEVICE)
    model = build_model(variant, max_seq)
    if mode == "eager":
        tps, toks = run_eager(model, input_ids, gen)
    elif mode == "compile":
        tps, toks = run_compile(model, input_ids, gen, warmup=warmup)
    else:
        tps, toks = run_cudagraph(model, input_ids, gen, warmup=3)
    # checksum over the first (1 + gen) generated tokens (common to all modes)
    chk = int(toks[0, : 1 + gen].to(torch.int64).sum().item())
    return tps, chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["dense", "ha"],
                    choices=["dense", "ha"])
    ap.add_argument("--modes", nargs="+", default=["eager", "compile", "cudagraph"],
                    choices=["eager", "compile", "cudagraph"])
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    # Internal: run exactly one (variant, mode) in this process, print a result
    # line, and exit.  Used to fully isolate CUDA-graph state between modes
    # (torch.compile's cuda-graph pool otherwise breaks a later manual capture).
    ap.add_argument("--_child", nargs=2, default=None,
                    metavar=("VARIANT", "MODE"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    S = args.context_len
    gen = args.gen
    headroom = max(args.warmup, 8) + gen + CHUNK_SIZE + 8
    max_seq = S + headroom

    # ---- child: run one config, emit a parseable line ----
    if args._child is not None:
        variant, mode = args._child
        try:
            tps, chk = _run_single(variant, mode, S, gen, args.warmup, max_seq)
            print(f"RESULT {variant} {mode} {tps:.3f} {chk}")
        except Exception as e:
            print(f"RESULT {variant} {mode} FAIL {type(e).__name__}:{e}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # ---- parent: dispatch each (variant, mode) as an isolated subprocess ----
    import subprocess
    print(f"device={torch.cuda.get_device_name(0)}  dtype={DTYPE}")
    print(f"context={S}  gen={gen}  max_seq={max_seq}")
    print(f"{'variant':>8} {'mode':>10} {'tok/s':>10} {'rel':>7}  {'match':>6}")
    print("-" * 50)

    common = [
        "--context-len", str(S), "--gen", str(gen), "--warmup", str(args.warmup),
    ]
    for variant in args.variants:
        base = None
        ref_chk = None
        for mode in args.modes:
            cmd = [sys.executable, os.path.abspath(__file__), *common,
                   "--_child", variant, mode]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = None
            for ln in proc.stdout.splitlines():
                if ln.startswith("RESULT "):
                    line = ln
            if line is None:
                tail = proc.stdout.strip().splitlines()[-1:] or proc.stderr.strip().splitlines()[-1:]
                print(f"{variant:>8} {mode:>10} {'FAIL':>10}   {tail}")
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[3] != "FAIL":
                tps = float(parts[3])
                chk = int(parts[4])
                if base is None:
                    base = tps
                if ref_chk is None:
                    ref_chk = chk
                match = "ref" if chk == ref_chk and mode == args.modes[0] else (
                    "yes" if chk == ref_chk else "NO")
                print(f"{variant:>8} {mode:>10} {tps:>10.1f} {tps / base:>6.2f}x  {match:>6}")
            else:
                print(f"{variant:>8} {mode:>10} {'FAIL':>10}   {' '.join(parts[3:])}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
