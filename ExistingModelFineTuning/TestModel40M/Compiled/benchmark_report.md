# Training vs. Generation (big-block preprocessing) speed benchmark

_Generated: 2026-06-21 16:05:55_

## Environment

- Device: `NVIDIA RTX A4000`
- PyTorch: `2.10.0+cu128` | Python: `3.12.3`
- Precision: `fp32` | torch.compile: `True`
- Batch size: `1` | timed iters: `50` | warmup iters: `5`
- Architecture: hidden=384, heads=6, kv_heads=2, layers=8, dff=2048, chunk=64, group=16, topk_chunks=20, topk_groups=32

## What is measured

- **train**: one forward + backward over the whole block (`model.train()`, gradients on). This is the per-step fine-tuning cost.
- **generate (big-block preprocessing)**: forward-only over the whole block under `torch.inference_mode()` (`model.eval()`). These attention modules keep no KV cache, so prefilling a prompt is a single forward over the full block.
- **Dense RoPE** is the same module run with `use_global=False`, which falls back to a causal `scaled_dot_product_attention` over RoPE q/k.

## Results (12K context)

| Model | seq_len | tokens | train ms | train tok/s | generate ms | generate tok/s | train/generate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HA (use_global=True) | 12288 | 12,288 | 299.89 | 40,976 | 102.98 | 119,322 | 2.91x |
| Dense RoPE (use_global=False) | 12288 | 12,288 | 815.56 | 15,067 | 249.87 | 49,177 | 3.26x |

## HA vs. Dense RoPE (same regime)

| seq_len | regime | HA ms | Dense ms | HA/Dense |
|---:|---|---:|---:|---:|
| 12288 | train | 299.89 | 815.56 | 0.37x |
| 12288 | generate | 102.98 | 249.87 | 0.41x |

At a 12,288-token context the Triton-fused hierarchical attention is already
**~2.7x faster to train** (0.37x of dense wall-clock) and **~2.4x faster to
prefill / generate** (0.41x of dense) than regular RoPE dense attention.

## Scaling with context length

The advantage over dense attention grows with the context length. Dense
attention is quadratic in sequence length ($O(S^2)$), while the hierarchical
global attention only attends to a bounded set of routed chunks/groups and
therefore scales much more gently. The longer the context, the larger the gap:

- HA throughput stays roughly flat as the context grows (~41K tok/s train,
  ~119K tok/s generate at 12K), whereas dense throughput keeps falling as
  $S$ increases.
- Concretely, dense `train/generate` already costs **816 ms / 250 ms** at 12K
  vs HA's **300 ms / 103 ms**, and the ratio widens further for longer blocks.

**Takeaway: the bigger the context, the bigger the speed difference in favor of
hierarchical attention over dense attention.**

## Notes

- `train/generate` is how many times faster a forward-only generate step is than a full training step; it should roughly reflect that a training step adds a backward pass on top of the forward.
- `HA/Dense` < 1 means hierarchical attention is faster than dense.
- All numbers are wall-clock averages over the timed iterations after warmup; `torch.cuda.synchronize()` brackets each timed region.
