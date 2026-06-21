# Training vs. Generation (big-block preprocessing) speed benchmark

_Generated: 2026-06-21 15:07:04_

## Environment

- Device: `NVIDIA RTX A4000`
- PyTorch: `2.10.0+cu128` | Python: `3.12.3`
- Precision: `fp32` | torch.compile: `True`
- Batch size: `1` | timed iters: `10` | warmup iters: `3`
- Architecture: hidden=384, heads=6, kv_heads=2, layers=8, dff=2048, chunk=64, group=16, topk_chunks=20, topk_groups=32

## What is measured

- **train**: one forward + backward over the whole block (`model.train()`, gradients on). This is the per-step fine-tuning cost.
- **generate (big-block preprocessing)**: forward-only over the whole block under `torch.inference_mode()` (`model.eval()`). These attention modules keep no KV cache, so prefilling a prompt is a single forward over the full block.
- **Dense RoPE** is the same module run with `use_global=False`, which falls back to a causal `scaled_dot_product_attention` over RoPE q/k.

## Results

| Model | seq_len | tokens | train ms | train tok/s | generate ms | generate tok/s | train/generate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HA (use_global=True) | 2048 | 2,048 | 51.08 | 40,091 | 16.65 | 123,037 | 3.07x |
| HA (use_global=True) | 4096 | 4,096 | 102.20 | 40,078 | 33.86 | 120,965 | 3.02x |
| HA (use_global=True) | 8192 | 8,192 | 205.84 | 39,798 | 69.85 | 117,275 | 2.95x |
| HA (use_global=True) | 16384 | 16,384 | 431.52 | 37,968 | 158.82 | 103,161 | 2.72x |
| Dense RoPE (use_global=False) | 2048 | 2,048 | 55.54 | 36,877 | 16.42 | 124,754 | 3.38x |
| Dense RoPE (use_global=False) | 4096 | 4,096 | 139.88 | 29,283 | 42.76 | 95,792 | 3.27x |
| Dense RoPE (use_global=False) | 8192 | 8,192 | 408.70 | 20,044 | 124.43 | 65,838 | 3.28x |
| Dense RoPE (use_global=False) | 16384 | 16,384 | 1376.76 | 11,900 | 427.24 | 38,349 | 3.22x |

## HA vs. Dense RoPE (same regime)

| seq_len | regime | HA ms | Dense ms | HA/Dense |
|---:|---|---:|---:|---:|
| 2048 | train | 51.08 | 55.54 | 0.92x |
| 2048 | generate | 16.65 | 16.42 | 1.01x |
| 4096 | train | 102.20 | 139.88 | 0.73x |
| 4096 | generate | 33.86 | 42.76 | 0.79x |
| 8192 | train | 205.84 | 408.70 | 0.50x |
| 8192 | generate | 69.85 | 124.43 | 0.56x |
| 16384 | train | 431.52 | 1376.76 | 0.31x |
| 16384 | generate | 158.82 | 427.24 | 0.37x |

## Notes

- `train/generate` is how many times faster a forward-only generate step is than a full training step; it should roughly reflect that a training step adds a backward pass on top of the forward.
- `HA/Dense` < 1 means hierarchical attention is faster than dense.
- All numbers are wall-clock averages over the timed iterations after warmup; `torch.cuda.synchronize()` brackets each timed region.
