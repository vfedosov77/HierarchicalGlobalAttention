# Qwen3-30B-A3B-Instruct-2507-FP8 — Long-Context Results (RTX 5090)

Hierarchical KV-routing lets the FP8 30B run long context on a single **RTX 5090 (32 GB)** by
keeping the full KV-cache in host RAM and pulling only a bounded working set to VRAM each step.
**Peak VRAM stays ~30 GB regardless of context length** — bounded by the weights, not the
sequence — so 32K fits comfortably, with the KV record living in host RAM instead.

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB (SM120; FP8 matmul via Triton fallback — DeepGEMM needs Hopper/Blackwell) |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (~29 GB weights), `attn_implementation="sdpa"` baseline |
| Attention | `QwenRoutedAttention` (drop-in; original FP8 projections reused by reference, no fine-tuning) |
| Cache | `RamKVCacheStore` — full KV record in host RAM; KV-head-granular LRU VRAM bank for the cold gather |
| Routing | group-level: `chunk_size=64`, `group_size=16`, `keep_first=2`, `keep_last=8`, `topk_chunks=20`, `topk_groups=32` (each query opens `topk_groups//2 = 16` groups ≈ 256 routed-middle tokens/step) |
| Precision | bf16 activations, fp32 attention softmax (`use_summaries=False` — real token K/V only) |
| Env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, prefill `block=16` |

## Quality (vs. dense baseline)

Teacher-forced over a 4096-token context, greedy next-token agreement + per-token perplexity
vs. the unmodified FP8 model (measured):

| attention | greedy-match (all) | match in routing tail (>256) | perplexity |
|-----------|:---:|:---:|:---:|
| dense baseline | 100% | 100% | **1.261** |
| routed (group-level, active config) | **99.76%** | **100%** | **1.262** |
| routed (whole-chunk, `group_size=64`) | 99.78% | 100% | 1.263 |

The routed model matches the dense baseline essentially perfectly across the routing-active
region; the sub-1% deltas are confined to the first few hundred tokens (near-tie logits / bf16
noise).

## Memory & speed

VRAM is bounded by the weights at every context length; the KV-cache grows in **host RAM**
instead. Decode throughput is **flat in context length** — the per-step working set
(sinks + local + routed-middle ≈ 1.3K keys) does not grow with the sequence.

| context | peak VRAM | KV record (host RAM) | decode tok/s | notes |
|--------:|:---:|:---:|:---:|---|
| 4K  | ~29.2 GB | 0.43 GB | **3.6** | measured (bank off, safe default) |
| 8K  | ~29.2 GB | 0.86 GB | **3.6** | measured |
| 32K | ~29.2 GB · bounded | 3.34 GB | ~3.6 (working-set-bounded) | VRAM bounded as 4K/8K |

- **Peak VRAM**: ~29.2 GB (≈ the 29 GB weights + small activation + the resident chunk-summary
  routing table), **flat in context length** because the working set is independent of the
  sequence. Well under the 32 GB card at 32K.
- **KV record (host RAM)**: full cold KV-cache footprint = `chunks × 6.68 MB`
  (`chunks = ctx / 64`), i.e. 0.43 / 0.86 / 3.34 GB at 4K / 8K / 32K. This is the cost that
  *would* otherwise sit in VRAM — moved to host RAM, so context length is bounded by RAM, not the
  GPU.
- **Decode tok/s**: ~3.6, flat across context (constant working set; the only context-growing
  cost is the chunk-routing scan over the resident chunk-summary table — negligible vs. the MoE
  forward).

### Optional VRAM chunk bank (decode accelerator)

A bounded LRU VRAM bank can keep the routed chunks resident so the per-token cold gather does not
re-copy KV from host RAM (99.6% hit-rate observed), which **roughly doubles decode to ~6 tok/s**
(measured 6.05 / 6.02 at 4K / 8K, peak ~29.9 GB). The bank **auto-sizes to the genuinely-free
VRAM** and is deliberately conservative: on the memory-tight 30B (~3 GB free after the FP8
weights) there is usually no safe room, so it **stays off by default** — VRAM stays bounded and
decode never OOMs, which is the priority. It engages automatically on configs/GPUs with more
headroom. Set `VRAM_CACHE_RESERVE_GB` lower (carefully) to trade activation headroom for the bank.

### Routed vs. dense decode (4K, measured)

| | decode tok/s | peak VRAM |
|---|:---:|:---:|
| dense (full attention) | 9.86 | 29.5 GB |
| routed RAM (bank off, safe default) | ~3.6 | ~29.2 GB |
| routed RAM + VRAM bank (when it fits) | 5.64 | 29.9 GB |

At 4K the full KV still fits in VRAM, so dense is faster (no PCIe, no per-layer routing overhead).
The routed path's value is **bounded VRAM at long context** (and host-RAM-resident KV), not raw
4K decode speed.

## Reproduce

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
    --tokens 4096 --block 16                 # quality vs dense (table above)

python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
    --speed-variants --block 16 --topk 8 --ctx-sizes 4096 8192   # prefill+decode per context

python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
    --bench --tokens 4096 --block 16         # dense vs routed decode + bank hit-rate

python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
    --ram --block 16 --ctx-sizes 2048 32768  # 32K bounded-VRAM retrieval (irrelevant-prefix QA)
```

> All numbers above are bf16-activation / fp32-softmax on a single RTX 5090. Prefill is fed in
> small blocks because the FP8 matmul autotuner OOMs on large prefill matmuls in the ~3 GB free
> after the weights; the vectorized chunk-parallel prefill path is exercised on the 40M benchmark
> (`../TestModel40M`) where there is no such memory pressure.
