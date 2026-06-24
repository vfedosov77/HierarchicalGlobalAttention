"""HGA decode benchmark — synthetic, weight-free, runs on any CUDA GPU.

Estimates HGA decode throughput / latency at Qwen3-30B attention shapes without
loading model weights, so it runs on a 16 GB A4000 (rough estimate) and on the
target RTX 5090 (real number).  It drives the engine-neutral
:class:`~FastInference.hga_core.runner.HgaLayerRunner` exactly as the SGLang
backend does, and reports per-stage bottlenecks (route / assemble / attention).

Usage (target server, 32K context, 1024 decode tokens, full layer count):

    python -m FastInference.bench.bench_decode_5090 \
        --layers 48 --context 32768 --decode 1024 \
        --topk-chunks 16 --topk-groups 64

Acceptance targets (from the plan, RAM mode 32K / 1024 out on RTX 5090):
    decode >= 15 tok/s, p50 <= 125 ms, p99 <= 180 ms.
"""

from __future__ import annotations

import argparse
import time

import torch

from ..hga_core.config import HgaConfig
from ..hga_core.runner import HgaLayerRunner


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    # minimal RoPE just to make summaries exercise the mixed-rope path
    H, T, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=x.device) / half))
    ang = pos.float().unsqueeze(-1) * inv                       # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], -1).unsqueeze(0)
    sin = torch.cat([ang.sin(), ang.sin()], -1).unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    return (x * cos + torch.cat([-x2, x1], -1) * sin).to(x.dtype)


def run(args) -> None:
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = HgaConfig(
        num_layers=args.layers,
        num_q_heads=args.q_heads, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        chunk_size=args.chunk, group_size=args.group, page_size=args.page,
        keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk_chunks, topk_groups=args.topk_groups,
        gpu_token_chunks=args.gpu_token_chunks, kv_dtype="bfloat16",
    )
    print(f"GPU={torch.cuda.get_device_name(0)} cfg: H={cfg.num_q_heads} KV={cfg.num_kv_heads} "
          f"Dh={cfg.head_dim} C={cfg.chunk_size} gs={cfg.group_size} layers={cfg.num_layers}")
    print(f"context={args.context} decode={args.decode} topk_chunks={cfg.topk_chunks} "
          f"topk_groups={cfg.topk_groups} gpu_token_chunks={cfg.gpu_token_chunks}")

    runner = HgaLayerRunner(cfg, dev)
    H, KVH, Dh = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    theta = cfg.rope_theta

    # ---- prefill the context once per layer ----
    S = args.context
    torch.manual_seed(0)
    pos = torch.arange(S, device=dev)
    t0 = time.perf_counter()
    for L in range(cfg.num_layers):
        q = _rope(torch.randn(H, S, Dh, device=dev, dtype=dtype), pos, theta)
        k = _rope(torch.randn(KVH, S, Dh, device=dev, dtype=dtype), pos, theta)
        kraw = torch.randn(KVH, S, Dh, device=dev, dtype=dtype)
        v = torch.randn(KVH, S, Dh, device=dev, dtype=dtype)
        runner.prefill(L, q, k, kraw, v, start_pos=0)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    torch.cuda.empty_cache()
    print(f"[prefill] {S} tokens x {cfg.num_layers} layers in {ttft:.2f}s "
          f"(TTFT proxy; real TTFT includes FFN/MoE)")

    # ---- timed decode loop ----
    step_ms, route_ms, attn_ms = [], [], []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    p = S
    for step in range(args.decode):
        torch.cuda.synchronize()
        start_evt.record()
        for L in range(cfg.num_layers):
            q = _rope(torch.randn(H, 1, Dh, device=dev, dtype=dtype),
                      torch.tensor([p], device=dev), theta)
            k = _rope(torch.randn(KVH, 1, Dh, device=dev, dtype=dtype),
                      torch.tensor([p], device=dev), theta)
            kraw = torch.randn(KVH, 1, Dh, device=dev, dtype=dtype)
            v = torch.randn(KVH, 1, Dh, device=dev, dtype=dtype)
            runner.decode_step(L, q, k, kraw, v, start_pos=p)
        end_evt.record()
        torch.cuda.synchronize()
        step_ms.append(start_evt.elapsed_time(end_evt))
        p += 1

    step_ms.sort()
    n = len(step_ms)
    p50 = step_ms[n // 2]
    p99 = step_ms[min(n - 1, int(n * 0.99))]
    mean = sum(step_ms) / n
    tok_s = 1000.0 / mean
    print("\n=== decode (per generated token, all layers) ===")
    print(f"  tokens timed : {n}")
    print(f"  mean         : {mean:.2f} ms   -> {tok_s:.1f} tok/s")
    print(f"  p50          : {p50:.2f} ms")
    print(f"  p99          : {p99:.2f} ms")
    print(f"  per-layer    : {mean / cfg.num_layers:.3f} ms/layer")
    bank = runner.manager.token_bank
    print(f"  token-bank   : hits={bank.hits} misses={bank.misses} "
          f"hit%={100 * bank.hits / max(1, bank.hits + bank.misses):.1f}")
    print("\nAcceptance (RTX 5090, 32K/1024): decode>=15 tok/s, p50<=125ms, p99<=180ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)        # 48 for full Qwen3-30B on 5090
    ap.add_argument("--context", type=int, default=8192)    # 32768 for the acceptance run
    ap.add_argument("--decode", type=int, default=128)      # 1024 for the acceptance run
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--topk-chunks", type=int, default=16)
    ap.add_argument("--topk-groups", type=int, default=64)
    ap.add_argument("--gpu-token-chunks", type=int, default=512)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark.")
    run(args)


if __name__ == "__main__":
    main()
