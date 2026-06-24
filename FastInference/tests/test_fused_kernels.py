"""Fused-kernel + runner equivalence checks.

Run directly:

    python -m FastInference.tests.test_fused_kernels [cuda|cpu]
"""

from __future__ import annotations

import sys

import torch

from ..hga_core.config import HgaConfig
from ..hga_core.kernels.decode_attention import (
    fused_decode_attention,
    _torch_decode_attention,
    HAS_TRITON,
)
from ..hga_core.runner import HgaLayerRunner


def test_fused_decode_matches_reference(device: str) -> None:
    torch.manual_seed(0)
    B, H, L, R, Dh = 1, 8, 1, 777, 128
    dev = torch.device(device)
    q = torch.randn(B, H, L, Dh, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, R, Dh, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, R, Dh, device=dev, dtype=torch.bfloat16)
    mask = torch.rand(B, H, L, R, device=dev) > 0.3
    mask[..., 0] = True  # guarantee >=1 visible key
    scale = Dh ** -0.5

    ref = _torch_decode_attention(q, k, v, mask, scale)
    got = fused_decode_attention(q, k, v, mask, scale)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"[fused_decode] triton={HAS_TRITON and device=='cuda'} max_abs_err={err:.3e}")
    assert err < 2e-2, f"fused decode kernel diverged: {err}"

    # multi-token block
    q2 = torch.randn(B, H, 16, Dh, device=dev, dtype=torch.bfloat16)
    m2 = torch.rand(B, H, 16, R, device=dev) > 0.3
    m2[..., 0] = True
    ref2 = _torch_decode_attention(q2, k, v, m2, scale)
    got2 = fused_decode_attention(q2, k, v, m2, scale)
    err2 = (got2.float() - ref2.float()).abs().max().item()
    print(f"[fused_decode L=16] max_abs_err={err2:.3e}")
    assert err2 < 2e-2, f"fused decode (L=16) diverged: {err2}"
    print("  PASS")


def test_runner_dense_equivalence(device: str) -> None:
    """Keep-everything limit: HGA decode == dense causal attention (token-level)."""
    torch.manual_seed(0)
    dev = torch.device(device)
    cfg = HgaConfig(
        num_layers=1, num_q_heads=4, num_kv_heads=4, head_dim=64,
        chunk_size=16, group_size=8,
        keep_first=0, keep_last=0,
        topk_chunks=999, topk_groups=999,
        kv_dtype="float16",
    )
    # disable summaries so the routed pool is exact token-level (dense-equivalent limit)
    runner = HgaLayerRunner(cfg, dev)
    H, Dh = cfg.num_q_heads, cfg.head_dim
    S = cfg.chunk_size * 6                      # 6 full chunks
    q = torch.randn(H, S, Dh, device=dev, dtype=torch.float16)
    k = torch.randn(H, S, Dh, device=dev, dtype=torch.float16)
    v = torch.randn(H, S, Dh, device=dev, dtype=torch.float16)

    # dense causal reference
    scale = cfg.scale
    scores = torch.einsum("hqd,hkd->hqk", q.float(), k.float()) * scale
    causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    ref = torch.einsum("hqk,hkd->hqd", torch.softmax(scores, dim=-1), v.float())

    # HGA prefill, token-level (summaries OFF, all routed groups opened exactly) ->
    # must reduce to dense causal attention.
    out = runner.prefill(0, q, k, k, v, start_pos=0, use_summaries=False)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"[runner dense-equiv token-level] max_abs_err={err:.3e}")
    assert err < 5e-2, f"token-level HGA decode should match dense causal: {err}"
    print("  PASS")


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")
    test_fused_decode_matches_reference(device)
    test_runner_dense_equivalence(device)
    print("DONE")


if __name__ == "__main__":
    main()
