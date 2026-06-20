# Hierarchical attention as a memory/IO win at 100K–1M context (analysis only)

The earlier experiments showed HA decode cannot beat **flash-attention dense decode**
on *compute* at ≤64K (flash decode is a single bandwidth-optimal kernel, flat ~45 tok/s;
HA's eager routed decode is launch-bound ~13 tok/s).  But that comparison assumes the
**full KV cache is resident in VRAM**.  The real opportunity for HA is the regime where
that assumption breaks: **100K–1M context on memory-limited GPUs**, where dense *must*
offload KV and collapses, while HA's per-token working set stays bounded.

All numbers below are for **Qwen3-8B, bf16**: `L=36` layers, `kv_heads=8`, `head_dim=128`,
`chunk=64`, `group=16` (M=4 groups/chunk), `topk_chunks=20`, `topk_groups=32`.

## 1. The one number that drives everything: per-token KV size

```
kv_per_token = 2(K,V) × kv_heads × head_dim × L × 2 bytes
             = 2 × 8 × 128 × 36 × 2 = 147,456 B = 144 KiB / token
```

Dense self-attention attends to **all** cached tokens every step, so the resident KV (and,
when offloaded, the bytes read per generated token) is `144 KiB × context`:

| context | dense KV (resident **and** read/token) |
|--------:|---------------------------------------:|
| 8 K     | 1.1 GiB |
| 32 K    | 4.5 GiB |
| 100 K   | 13.7 GiB |
| 1 M     | **144 GiB** |

## 2. HA's working set is bounded, not O(context)

HA keeps **summaries** resident and, per step, touches only the routed tokens:

* `chunk_k` (key summary, 1 per 64 tokens, K only): `8×128×36×2 = 72 KiB / chunk`.
* `group_k`+`group_v` (4 groups/chunk, K and V): `4×2×72 KiB = 576 KiB / chunk`.

| context | n_chunks | chunk_k resident | +group summaries | full summaries | dense KV |
|--------:|---------:|-----------------:|-----------------:|---------------:|---------:|
| 100 K   | 1 563    | 110 MiB          | +880 MiB         | ~1.0 GiB       | 13.7 GiB |
| 1 M     | 16 384   | **1.1 GiB**      | +9.0 GiB         | ~10.1 GiB      | 144 GiB  |

Routing only needs `chunk_k` resident (**1.1 GiB for 1M context**). `group_k/group_v` are
needed for just the top-20 routed chunks → fetch `20×576 KiB ≈ 11.5 MiB`. The full token KV
lives on **CPU RAM or NVMe**, and only the *opened* tokens are pulled in.

**Per generated token, HA touches:**
* opened token KV: `topk_groups × group × kv_per_token` = `32 × 16 × 144 KiB / 64`-ish…
  more precisely per layer `512 tokens × (kv_heads×head_dim×2×2B)=2 MiB`, ×36 = **~75 MiB/token**
  if fetched per token. With **chunk-shared routing** the routed set is stable across the 64
  tokens of a chunk, so load the union **once per chunk** and reuse → amortized **~5 MiB/token**.

So HA's per-token IO is **~5 MiB regardless of context length**, vs dense's `144 KiB × context`.

## 3. Offload IO time per token (the decisive comparison)

Bandwidths: CPU↔GPU PCIe4 x16 ≈ 26 GB/s; NVMe PCIe4 ≈ 5 GB/s; SATA SSD ≈ 0.5 GB/s.

| context | dense read/tok | dense via PCIe | dense via NVMe4 | HA read/tok | HA via PCIe | HA via NVMe4 |
|--------:|---------------:|---------------:|----------------:|------------:|------------:|-------------:|
| 100 K   | 13.7 GiB       | 0.53 s         | 2.7 s           | ~5 MiB      | 0.2 ms      | 1.0 ms |
| 1 M     | 144 GiB        | **5.5 s**      | **29 s**        | ~5 MiB      | 0.2 ms      | 1.0 ms |

HA's IO is **constant ~5 MiB/token**; dense's grows with context to **GiB/token**. At 1M,
offloaded dense is **5–29 s/token**; HA's IO is **sub-millisecond** and fully hideable behind
compute via prefetch.

## 4. End-to-end decode estimate (memory-limited GPU)

HA decode is compute-bound by its eager routing (~70 ms/token measured at 1.7B; estimate
**~100–150 ms/token at 8B**, launch-bound, roughly context-independent — the `chunk_k` scan
grows as `context/64` but a 16K-row scan is a cheap matmul + topk). IO (~1 ms) hides behind it.

| 8B, 1M context        | resident feasible?            | decode speed |
|-----------------------|-------------------------------|--------------|
| Dense, fits in VRAM   | needs **144 GiB** KV → **no** on any consumer GPU | — |
| Dense, KV offloaded   | yes                           | **0.03–0.2 tok/s** (IO-bound) |
| **HA, offloaded**     | yes (1.1 GiB resident summaries) | **~7–10 tok/s** (compute-bound) |

→ **HA is ~50–300× faster than offloaded dense at 1M, and the only option that fits at all.**

## 5. When is it profitable? The crossover

Dense stays fast only while its KV fits in **free** VRAM. The threshold context is:

```
ctx_dense_max ≈ (VRAM_free) / 144 KiB
```

| GPU (8B bf16, ~16 GiB weights) | free VRAM | dense max ctx (resident) | beyond that |
|--------------------------------|----------:|-------------------------:|-------------|
| 16 GB                          | ~0        | weights barely fit       | offload immediately |
| 24 GB                          | ~8 GiB    | ~58 K tokens             | dense → seconds/token |
| 32 GB (RTX 5090)               | ~16 GiB   | ~116 K tokens            | dense → seconds/token |

**Past ~100 K context on a 24–32 GB GPU, dense must offload and collapses to <1 tok/s, while
HA keeps ~10 tok/s.** This is exactly the user's "100K and bigger is a game changer," and it is
*more* pronounced on weaker GPUs (smaller free VRAM → earlier crossover). HA also enables
context lengths (≥1M) that dense simply cannot hold at all.

## 6. Why it works: bounded working set

The chunked/hierarchical design turns the per-step working set from **O(context)** (dense:
read all KV) into **O(topk)** ≈ constant (HA: scan small summaries + read ~20 routed chunks).
A constant working set is what makes CPU/disk offload *usable*: you stream a few MiB per token
instead of the entire cache.

## 7. Engineering required (sketch)

1. **KV store backend.** Token K/V on CPU **pinned** memory (or `mmap`'d NVMe), laid out
   **chunk- and group-contiguous** so an opened group is a sequential read. Keep `chunk_k`
   resident on GPU (≤1.1 GiB at 1M); `group_k/group_v` resident if they fit, else fetched for
   the 20 routed chunks.
2. **Streaming prefill.** Process the context in blocks; for each closed chunk write its token
   K/V to the store and its summaries to the resident tables. Never hold the full context
   resident (the current pure-torch routed prefill OOM'd at 64K — it must be chunk-streamed).
3. **Chunk-granular decode + prefetch.** Route once per chunk (reuse the running-max union),
   async-fetch that chunk's opened-token KV (pinned→GPU) while the previous chunk's 64 tokens
   compute → IO fully hidden behind the ~100 ms compute.
4. **(Optional) second hierarchy level.** The `chunk_k` routing scan is O(context/64); beyond
   ~1–10M add super-chunk summaries to keep routing sublinear.

## 8. Caveats / honesty

* The bottleneck in this regime is HA's **eager decode compute (~100 ms/token)**, not IO.
  Optimizing it (fewer ops) raises throughput but is *not* what wins here — dense's offload IO
  (seconds/token) is the thing being beaten. HA wins on offload even un-optimized.
* This is a **memory/IO** win, not a compute win. With the full KV resident and flash attention,
  dense decode is faster (≤64K results). HA's advantage begins exactly where the KV no longer
  fits.
* Accuracy is a separate axis: HA attends to a routed subset, so quality vs full attention is the
  model's trained approximation tradeoff — unrelated to this memory analysis.
* Numbers scale with model: bigger `L`/`kv_heads` raise `kv_per_token` and shift the crossover to
  *shorter* context (dense gets memory-bound sooner → HA wins sooner).
