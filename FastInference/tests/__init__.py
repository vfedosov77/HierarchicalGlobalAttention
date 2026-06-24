"""FastInference test suite (run directly, not via pytest).

    python -m FastInference.tests.test_fused_kernels

Validates the engine-neutral HGA core on whatever GPU is present (A4000 here,
RTX 5090 on the target server):

1. ``fused_decode_attention`` matches the exact PyTorch reference (masked softmax).
2. The full ``HgaLayerRunner`` decode path reduces to dense causal attention in
   the keep-everything limit (the dense-equivalence contract).
"""
