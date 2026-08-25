# turing1 (Xeon Gold 6148 × 2, 80 threads, V100 16 GB)

Measured 2026-08-19. Server has **no internet**; sources, portable cmake, and a patched llama.cpp tree were copied in.

## Hardware

| | |
|---|---|
| CPU | 2× Intel Xeon Gold 6148 @ 2.40 GHz (20c/40t each, AVX-512, **no VNNI**) |
| NUMA | 2 nodes × ~192 GB |
| GPU | Tesla V100-PCIE-16 GB (driver 580, no CUDA toolkit installed) |

## 1-level vs 2-level (HGA attention kernel)

Same routed **chunk** count. 1-level opens every group in those chunks (~8 % tokens). 2-level opens groups of 16 (~4 % tokens). Qwen3.8-27B gated-attention dims: 24/4 heads, `head_dim=256`. Synthetic QKV, 40 physical cores, OpenMP + AVX-512.

| seq | level | chunks | groups | att. tokens | frac | prefill ms | decode µs | attn tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 1 | 4 | 722 | 0.175 | 395 | 780 | 1282 |
| 4096 | 2 | 1 | 1 | 674 | 0.164 | 338 | 632 | 1583 |
| 8192 | 1 | 1 | 4 | 722 | 0.088 | 754 | 729 | 1372 |
| 8192 | 2 | 1 | 1 | 674 | 0.082 | 687 | 669 | 1494 |
| 16384 | 1 | 10 | 40 | 1298 | 0.079 | 1692 | 888 | 1127 |
| 16384 | 2 | 10 | 10 | 818 | 0.050 | 1497 | 647 | 1547 |
| **32768** | **1** | **31** | 124 | 2642 | **0.081** | 5054 | 2743 | 365 |
| **32768** | **2** | **31** | 42 | 1330 | **0.041** | 3250 | 1638 | **610** |

At 32K, **two levels are 1.67× faster decode and 1.55× faster prefill** than one level, with the same 31 routed chunks. Below ~8K the sink+local window already covers most of the 8 % budget, so the group win is small.

Unit tests on the same box: dense-equivalence max abs err `4.7e-4`; both levels select the same chunk count.

## L2-tiled flash kernel (INT8, 32-core grid)

Same box, **40 physical cores**, `OMP_PLACES=cores`. Log line:

`hga L2 tiles: kv_heads=4  q_tiles=2  k_tiles=4  tasks=32  threads=40`

Each core owns ½ of one KV-head’s Q (3 of 6 GQA heads) and ¼ of that head’s keys; 4 KV heads × 2 × 4 = 32 tasks. Softmax partials from the 4 K-tiles are merged (flash-attention combine). Tests: dense `4.7e-4`, I8 vs F16 cosine `0.99999`.

| seq | level | prec | att. tokens | prefill ms | decode µs | attn tok/s |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 1 | i8 | 722 | 423 | 649 | 1541 |
| 4096 | 2 | i8 | 674 | 382 | 632 | 1582 |
| 4096 | 2 | f16 | 674 | 492 | 545 | 1835 |
| 8192 | 1 | i8 | 722 | 802 | 721 | 1388 |
| 8192 | 2 | i8 | 674 | 824 | 613 | 1631 |
| 8192 | 2 | f16 | 674 | 959 | 568 | 1760 |
| 16384 | 1 | i8 | 1298 | 1775 | 1255 | 797 |
| 16384 | 2 | i8 | 818 | 1619 | 892 | 1122 |
| 16384 | 2 | f16 | 818 | 1979 | 780 | 1283 |
| **32768** | **1** | i8 | 2642 | 5245 | 2666 | 375 |
| **32768** | **2** | i8 | 1330 | **3891** | 1617 | 619 |
| **32768** | **2** | f16 | 1330 | 4514 | **1389** | **720** |

INT8 prefill is **0.78–0.86×** the F16 time (faster). INT8 decode is **1.08–1.16×** F16 (slower — quantize + K-tile merge). Versus the earlier F16-KV kernel, 32K 2-level prefill is not yet faster (3891 ms vs 3250 ms); the 2D split is the right L2 layout but the merge still costs.

## llama.cpp

Patched `qwen35` full-attention path, built **on the box** (portable cmake 3.28.3, AVX-512, no VNNI, no CUDA toolkit):

```
~/HGA/Inference/Qwen3_8_27B/third_party/llama.cpp/build/bin/llama-cli
  --hga --hga-levels {1|2} --no-kv-offload --numa distribute -t 40 -tb 80
```

`--hga`, `--hga-levels`, `--hga-frac-l1/l2`, `--no-kv-offload` all show up in `--help`.

## End-to-end Qwen3.8-27B Q4_K_M (CPU + V100)

Copied onto the box: CUDA 12.5 toolkit (`~/opt/cuda-12.5`, sm_70), Unsloth `Qwen3.8-27B-Q4_K_M.gguf` (15.9 GB), patched llama.cpp with `libggml-cuda.so`.

Do **not** pass `-ngl 99` — 17 GB of weights plus compute buffers do not fit in 16 GB, and current llama.cpp `--fit` **aborts** if the user forced `n_gpu_layers`. Omit `-ngl` so auto-fit packs FFN/GDN into the V100. `--no-kv-offload` keeps KV + HGA softmax on the CPU.

Earlier `llama-completion` with a **20-token** prompt (overhead-dominated): dense **56.9 / 6.77** tok/s, HGA-2 **8.76 / 6.79**.

Integrated **INT8 + L2-tiled HGA** into the same llama.cpp binary (`prec=i8`, `--hga --hga-levels 2`). Long prompts, 40 cores, KV on CPU, FFN/GDN on V100:

| ctx (prompt tokens) | mode | prefill tok/s | decode tok/s |
|---|---|---:|---:|
| 1989 | dense (GPU FA) | **427** | 5.50 |
| 1989 | HGA 2-level INT8 | 73.1 | **5.73** |
| 3969 | dense | **426** | 4.78 |
| 3969 | HGA 2-level INT8 | 72.5 | **5.57** |
| 7709 | dense | **405** | 3.84 |
| 7709 | HGA 2-level INT8 | 69.8 | **5.58** |

Decode with HGA stays ~**5.6 tok/s** as context grows; dense falls (5.5 → 3.8 at 8K) because GPU FA still attends every token. Prefill is still faster **without** HGA (~**420 tok/s** vs ~**70 tok/s**) because the 16 full-attn layers run on CPU instead of V100 flash-attn. FFN on the V100 still dominates decode.

Binary: `~/HGA/Inference/Qwen3_8_27B/third_party/llama.cpp/build/bin/llama-completion`  
Env: `source ~/HGA/Inference/Qwen3_8_27B/scripts/env_turing1.sh`
