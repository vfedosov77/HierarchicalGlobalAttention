"""Compare VLM prefill + diffusion kernels to SDPA / Flash Attention.

    python -m SpecializedKernels.Len3000HeadDim128.bench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SpecializedKernels.Len3000HeadDim128 import (
    diffusion_cross_attention,
    prompt_chunk_keys,
    vlm_prefill_attention,
)


def _median_us(fn, warmup: int = 20, iters: int = 60) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2] * 1e3


def _row(name: str, us: float, ref: float) -> None:
    print(f"{name:<42} {us:8.1f} us   {ref / us:5.2f}x vs dense")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=3000)
    p.add_argument("--q-diff", type=int, default=64)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    b, h, kvh, d, s, qd = 1, args.heads, args.kv_heads, 128, args.seqlen, args.q_diff
    dtype = torch.bfloat16
    scale = d ** -0.5
    print(f"GPU  {torch.cuda.get_device_name(0)}")
    print(f"VLM  B={b} S={s} Hq={h} Hkv={kvh} D={d} topk={args.topk} causal")
    print(f"Diff Q={qd} K={s + qd} non-causal routed prefix (no record_attention)")
    print()

    q = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, kvh, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, kvh, s, d, device="cuda", dtype=dtype)

    def sdpa_vlm():
        F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)

    def routed_vlm():
        vlm_prefill_attention(q, k, v, softmax_scale=scale, topk=args.topk)

    q_d = torch.randn(b, h, qd, d, device="cuda", dtype=dtype)
    k_d = torch.randn(b, kvh, s + qd, d, device="cuda", dtype=dtype)
    v_d = torch.randn(b, kvh, s + qd, d, device="cuda", dtype=dtype)

    def sdpa_diff():
        F.scaled_dot_product_attention(q_d, k_d, v_d, is_causal=False, scale=scale, enable_gqa=True)

    ck_d = prompt_chunk_keys(k_d, n_prompt=s)

    def routed_diff():
        diffusion_cross_attention(q_d, k_d, v_d, softmax_scale=scale, topk=args.topk, n_prompt=s)

    def routed_diff_cached():
        diffusion_cross_attention(
            q_d, k_d, v_d, softmax_scale=scale, topk=args.topk, n_prompt=s, chunk_k=ck_d,
        )

    for _ in range(10):
        routed_vlm()
        sdpa_vlm()
        routed_diff()
        routed_diff_cached()
        sdpa_diff()
    torch.cuda.synchronize()

    vlm_sdpa = _median_us(sdpa_vlm)
    vlm_rt = _median_us(routed_vlm)
    print("VLM prefill (BHSD, no transpose)")
    _row("SDPA causal GQA", vlm_sdpa, vlm_sdpa)
    _row("vlm_prefill_attention", vlm_rt, vlm_sdpa)
    print()
    diff_sdpa = _median_us(sdpa_diff)
    diff_rt = _median_us(routed_diff)
    diff_ck = _median_us(routed_diff_cached)
    print("Diffusion cross-attn (Q=64, no record_attention)")
    _row("SDPA full 64×(S+64)", diff_sdpa, diff_sdpa)
    _row("routed (recompute chunk means)", diff_rt, diff_sdpa)
    _row("routed (chunk_k reused / graph)", diff_ck, diff_sdpa)

    try:
        from flash_attn import flash_attn_func

        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.repeat_interleave(h // kvh, dim=1).transpose(1, 2).contiguous()
        v_fa = v.repeat_interleave(h // kvh, dim=1).transpose(1, 2).contiguous()

        def fa_vlm():
            flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=scale)

        for _ in range(8):
            fa_vlm()
        fa_us = _median_us(fa_vlm)
        print()
        print("Reference: FlashAttention on VLM shape (includes their layout)")
        _row("flash_attn_func causal", fa_us, fa_us)
        _row("vlm_prefill vs that FA", vlm_rt, fa_us)
    except Exception as exc:
        print(f"\nFlashAttention skip: {exc}")


if __name__ == "__main__":
    main()
