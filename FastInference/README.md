# FastInference — production HGA inference backends

Engine-native **Hierarchical Global Attention (HGA)** for high-throughput
long-context inference. This directory is the production path that complements
the research/training reference in
[ExistingModelFineTuning](../ExistingModelFineTuning). The first engine is
**SGLang**; **vLLM** is planned next.

> The HF replacement in `ExistingModelFineTuning` remains the correctness and
> training reference. `FastInference` has **no** dependency on `DynamicCache` and
> does **no** Qwen module monkey-patching.

## Why SGLang first

SGLang has an explicit attention-backend extension path — implement
`forward_extend`, `forward_decode`, `init_forward_metadata`, then add CUDA-graph
capture/replay — and already ships concepts close to HGA (HiCache GPU/host/storage
KV hierarchy; HiSparse decode-side sparse KV). vLLM (PagedAttention, prefix
caching, chunked prefill, FP8 KV, KV offloading) is a heavier integration and
lands second.

## Architecture

```
SGLang / vLLM scheduler
        |
        v
Qwen3-HGA model runner
        |
        v
HGA attention backend            FastInference/sglang_backend
  forward_extend  (prefill)
  forward_decode  (decode)
  CUDA-graph metadata (v1)
        |
        v
HGA layer runner                 FastInference/hga_core/runner.py
  route (chunk top-k -> group top-k)
  assemble routed KV
  fused decode attention
        |
        v
HGA cache manager                FastInference/hga_core/cache
  GPU : chunk summaries (always), group summaries (<= limit),
        sink/local windows, small token page bank
  CPU : full closed-token KV in pinned RAM
  NVMe: cold/idle spill only (L3, disabled in v0)
```

### Page layout

| param        | v0 value | notes |
|--------------|----------|-------|
| `group_size` | 16       | tokens per group |
| `chunk_size` | 64       | 4 groups / chunk |
| `page_size`  | 16       | benchmark 64 later (kernel perf vs prefix-cache granularity tradeoff) |

## Module map

| Path | Role |
|------|------|
| `hga_core/config.py` | `HgaConfig` — geometry + tiered-cache budgets |
| `hga_core/route_metadata.py` | `RouteMetadata` — engine-neutral routing selection |
| `hga_core/summaries.py` | mixed-RoPE chunk/group summary builder |
| `hga_core/kernels/decode_attention.py` | **fused** flash-style decode attention (Triton + torch) |
| `hga_core/kernels/route_topk.py` | chunk / group top-k scoring + selection |
| `hga_core/cache/gpu_summary_store.py` | `HgaGpuSummaryStore` (chunk/group summaries on GPU) |
| `hga_core/cache/gpu_token_bank.py` | `HgaGpuTokenBank` (LRU token pages, protects sink/local/current) |
| `hga_core/cache/pinned_host_kv.py` | `HgaPinnedHostKV` (full closed KV in pinned RAM) |
| `hga_core/cache/cold_storage.py` | `HgaColdStorageAdapter` (L3 spill, **stub in v0**) |
| `hga_core/cache/manager.py` | `HgaCacheManager` (ties tiers, assembles routed KV) |
| `hga_core/runner.py` | `HgaLayerRunner` (engine-neutral prefill/decode orchestration) |
| `sglang_backend/hga_attention_backend.py` | SGLang `AttentionBackend` adapter |
| `sglang_backend/register.py` | register under `--attention-backend hga` |
| `tests/test_fused_kernels.py` | kernel + dense-equivalence checks |
| `bench/bench_decode_5090.py` | weight-free decode benchmark (any GPU) |
| `scripts/run_on_5090.sh` | one-shot correctness + benchmark on the target server |

## v0 status (verified on an RTX A4000 16 GB)

The biggest decode-speed lever — replacing `RoutedKV.attend`'s
`einsum -> mask -> softmax -> einsum` with one fused online-softmax kernel — is
done and validated. The opened-group gather is fully vectorized (no Python
segment loops).

| Check | Result |
|-------|--------|
| `fused_decode_attention` vs exact reference (bf16) | max abs err **3.8e-6** (L=1), **4.9e-4** (L=16) |
| Full `HgaLayerRunner`, token-level, vs dense causal SDPA | max abs err **9.7e-4** |
| Decode, 2 layers @ 4K (A4000) | **229 tok/s**, p50 4.0 ms, 99% token-bank hit |
| Decode, 48 layers @ 8K (A4000, attention only) | **10.3 tok/s**, p50 96 ms, p99 104 ms, 98.9% bank hit |

The A4000 number is attention-only and Python-routing-bound; the RTX 5090 (faster
SMs, more VRAM, larger token bank) is expected to clear the acceptance targets.

### Acceptance targets (RTX 5090, RAM mode, 32K ctx / 1024 out)

```
decode >= 15 tok/s   p50 <= 125 ms   p99 <= 180 ms   TTFT <= 120 s
```

## Run

### Validate + benchmark (this box or the 5090)

```bash
# correctness
python -m FastInference.tests.test_fused_kernels cuda

# decode benchmark (weight-free; scales to any GPU)
python -m FastInference.bench.bench_decode_5090 --layers 48 --context 32768 --decode 1024
```

### One-shot on the RTX 5090 server

```bash
bash FastInference/scripts/run_on_5090.sh /path/to/venv/bin/activate
```

### Register the SGLang backend (when running a real model)

```python
from FastInference.sglang_backend import register_hga_backend, configure_hga
configure_hga(chunk_size=64, group_size=16, topk_chunks=16, topk_groups=64)
register_hga_backend("hga")      # then launch sglang with --attention-backend hga
```

## Roadmap

* **v0 (this drop)** — DCA off, native HGA, RAM+VRAM fast path, eager
  `forward_extend`/`forward_decode`, fused decode kernel, vectorized gather,
  batch=1 (+ request-loop batch>1). FS disabled.
* **v0.x** — CUDA-graph capture/replay (SGLang's required 2nd step), fused
  chunk/group top-k + fused gather+attention, FP8 KV with **fused** dequant, batch>1
  with static scratch buffers.
* **v1** — YaRN / RoPE-scaling compatibility, FS L3 tier (port the paged
  writer-pool store from `ExistingModelFineTuning/KvRouter`), async next-token
  prefetch tuning.
* **v2** — fused DCA kernel; vLLM backend (PagedAttention block <-> HGA
  chunk/group mapping, offloading connector).

### Known v0 limitations

* SGLang applies RoPE **before** `backend.forward`, so the adapter currently only
  receives RoPE-applied `k` and passes it as `k_raw` (summary approximation). A
  pre-RoPE model-runner hook is needed for exact summaries — see
  `sglang_backend/hga_attention_backend.py`.
* FS/NVMe spill is intentionally a stub (`HgaColdStorageAdapter`) until the
  RAM+VRAM path meets targets.
* CUDA-graph hooks raise `NotImplementedError` until eager decode is fast.
