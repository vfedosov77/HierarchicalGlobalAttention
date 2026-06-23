# Hierarchical Global Attention (HGA)

**Metaphor**

Imagine you're reading a very long book and someone asks you a question. You don't re-read every single page to answer it. Instead, you remember a few key chapters that seem relevant, flip to those, and scan the important paragraphs.
HGA does the same thing for an AI model. Instead of looking at every previous word it has ever seen (which fills up memory fast), it first picks the most relevant chunks of past context, then zooms in on the most important groups within those chunks. Everything else stays in cheaper storage (RAM) until needed.
It reuses the model's existing "reading comprehension" weights to decide what's relevant—no extra training required.

**Long-context attention for pretrained grouped-query transformers when GPU memory is the bottleneck.**

Hierarchical Global Attention (HGA) is a drop-in replacement for attention in pretrained GQA-style transformer models. It keeps the original `q_proj`, `k_proj`, `v_proj`, and `o_proj` weights unchanged, adds no calibration weights in the current exact-token path, and changes how old K/V cache is selected and stored.

The practical goal is simple: keep model weights on GPU, keep most old K/V cache in RAM or, later, NVMe, and pull only a small routed working set into VRAM. This is especially useful for models such as `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, where FP8 weights nearly consume a 32 GB GPU and dense KV cache becomes the long-context blocker.

## Highlights

| Area | Current status |
|---|---|
| Large-model demo | Qwen3-30B-A3B-Instruct-2507-FP8 with RAM-cached routed K/V |
| Tested long context | 32K-token context on a 32 GB RTX 5090-class GPU |
| Weight compatibility | Original Q/K/V/O projections are reused directly |
| Fine-tuning required | No fine-tuning required for the Qwen3 RAM-cache demo |
| Attention math | Two-level routing selects chunks, then groups; final attention uses exact token K/V |
| 40M validation | Dense-to-routed copy: `+0.01828` nats at 8192 tokens without any fine-tuning |
| 12K fused benchmark | `2.72x` faster train step and `2.43x` faster forward-only prefill than dense RoPE |

> **Research status:** this repository is an active research prototype, not yet a polished inference server. The Qwen3 demo is the most important system result. The 40M model is mainly a correctness and benchmark harness. We plan a production-level support based on vLLM/SGLang.

## Core idea

Dense attention asks every token to attend to every previous token. HGA asks two cheaper questions first:

> **Which old chunks are relevant to the current chunk?**  
> **Which groups inside those chunks should be opened to real token K/V?**

The context is split into fixed-size chunks, typically 64 tokens. Each chunk can be split into groups, for example 4 groups of 16 tokens. Nearby tokens in the current block often need almost the same remote context, so the router can select chunks and groups once for the current block and reuse that working set.

For the Qwen3 RAM-cache chatbot, the current intended hierarchy is:

```text
chunk_size   = 64 tokens
group_size   = 16 tokens   # 4 groups per chunk
keep_first   = 2 chunks    # 128 sink tokens
keep_last    = 8 chunks    # 512 recent local tokens
topk_chunks  = 16 chunks   # candidate remote chunks
topk_groups  = 64 groups   # about 1024 routed remote tokens
```

So a whole 64-token block can work with roughly a thousand routed remote tokens, plus deterministic sink/local/current context, instead of materializing attention over the whole history.

## How HGA works

1. **Partition context into chunks and groups.**  
   Closed context is represented as 64-token chunks. In the group-level Qwen path, each chunk is further split into smaller groups, typically 16 tokens per group.

2. **Build routing summaries from existing projections.**  
   HGA does not add learned summary projections. Chunk and group summaries are pooled from the model's own projected K/V tensors. For RoPE models, summaries preserve positional geometry using the mixed-RoPE rule used in the implementation.

3. **Keep deterministic windows.**  
   The first chunks are kept as attention sinks, and the most recent chunks are kept as exact local context.

4. **Route chunks first.**  
   The router scores the current query block against compact chunk key summaries and selects the most relevant old chunks.

5. **Open groups inside selected chunks.**  
   Inside the selected chunks, the router scores group summaries and opens only the strongest groups to exact token-level attention. This is the second hierarchy level and is important for keeping the VRAM working set small.

6. **Attend over real token K/V.**  
   In the pretrained exact-token path, summaries are used for selection only. The final softmax is computed over real token keys and values from the opened groups, plus sink/local/current context.

7. **Store old K/V outside VRAM.**  
   The tiered cache keeps tiny routing tables and hot windows on GPU. Old token K/V can live in host RAM. The storage interface is designed so the cold token record can later be backed by NVMe.

## Why this helps

For a large model, the weights can already occupy almost all GPU memory. Dense decoding then fails not because the model weights do not fit, but because the KV cache grows with context length.

HGA makes VRAM use depend mostly on:

```text
model weights + hot local chunks + routed chunk/group working set
```

instead of:

```text
model weights + full dense KV cache
```

This is why the Qwen3-30B FP8 demo can run long contexts on a 32 GB GPU while keeping the full old KV history in host RAM.

## Quick start

### 1. Clone

```bash
git clone https://github.com/vfedosov77/HierarchicalGlobalAttention.git
cd HierarchicalGlobalAttention
```

### 2. Install dependencies

Use a recent CUDA-enabled PyTorch build. Exact package versions depend on your CUDA setup.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U torch triton transformers datasets accelerate safetensors huggingface_hub tqdm
```

For Qwen3 FP8 experiments, make sure your `transformers` version supports the selected Qwen3 MoE FP8 checkpoint.

### 3. Run Qwen3-30B FP8 with RAM-cached HGA (tested on RTX 5090 32 GB)

```bash
python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8
```

Browser UI:

```bash
python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8 --ui
```

The script loads:

```text
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

and replaces every attention layer with the routed exact-token attention wrapper. In the group-level configuration, routing is two-level: chunk selection first, then group opening inside selected chunks.

### 4. Run Qwen router self-tests

These tests do not require loading the full 30B model when `--selftest-only` is used.

```bash
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --selftest-only
```

RAM-cache 32K-style test:

```bash
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
  --ram \
  --ctx-sizes 32768 \
  --variant group \
  --keep-first 2 \
  --keep-last 8 \
  --topk 16
```

For `--variant group`, the test path uses `group_size=16` and `topk_groups=4*topk`, so the 16 selected chunks expose up to 64 groups, or about 1024 routed remote tokens.

Decode speed comparison:

```bash
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed \
  --bench \
  --tokens 8192 \
  --variant group \
  --keep-first 2 \
  --keep-last 8 \
  --topk 16
```

## 40M SmallLM validation

The 40M model is useful because it is small enough to run correctness and speed experiments quickly.

### Routing quality without fine-tuning

```bash
python ExistingModelFineTuning/TestModel40M/test_routed_keep.py \
  --seq-len 8192 \
  --keep-first 6 \
  --keep-last 6 \
  --num-batches 20
```

Reported result:

| Model | Loss, nats | Perplexity |
|---|---:|---:|
| Dense causal SDPA | 3.70516 | 40.657 |
| Chunk-routed, same weights | 3.72344 | 41.406 |
| Difference | `+0.01828` | `+1.8%` |

This isolates the routing approximation: both models use the same dense checkpoint weights, and no HGA fine-tuning is applied.

### Fused 12K benchmark

```bash
python ExistingModelFineTuning/TestModel40M/Compiled/benchmark_train_vs_generate.py \
  --seq-lens 12288 \
  --batch-size 1 \
  --precision fp32
```

Reported on an NVIDIA RTX A4000, PyTorch 2.10, `torch.compile`, fp32:

| Model | Train ms | Train tok/s | Forward-only ms | Forward-only tok/s |
|---|---:|---:|---:|---:|
| HGA, Triton fused | 299.89 | 40,976 | 102.98 | 119,322 |
| Dense RoPE | 815.56 | 15,067 | 249.87 | 49,177 |

At 12,288 tokens this is about `2.72x` faster for train steps and `2.43x` faster for forward-only prefill.

## Repository map

```text
.
├── README.md
├── prepare_model.py
└── ExistingModelFineTuning
    ├── KvRouter
    │   ├── chunk_router.py       # chunk/group router and routed attention assembly
    │   ├── cache_store.py        # VRAM/RAM tiered K/V storage interface
    │   └── vectorized.py         # vectorized prefill assembly
    ├── Qwen3LongContext
    │   ├── chat_qwen30b_fp8.py   # Qwen3-30B FP8 chat with RAM-cached HGA
    │   ├── qwen_routed_attention.py
    │   └── test_qwen30b_routed.py
    ├── TestModel40M
    │   ├── README.md
    │   ├── test_routed_keep.py
    │   ├── Compiled             # Triton-fused HGA benchmark path
    │   └── NotOptimized         # eager reference/correctness path
    └── OtherTestedIdeas         # archived experiments and older prototypes
```

## Main implementation files

### Qwen3 large-model path

- `ExistingModelFineTuning/Qwen3LongContext/chat_qwen30b_fp8.py`  
  Interactive terminal/browser chat for Qwen3-30B FP8 with RAM-cached K/V.

- `ExistingModelFineTuning/Qwen3LongContext/qwen_routed_attention.py`  
  Drop-in wrapper around Qwen3 attention. It reuses the original projections and norms, and changes only what each query attends to.

- `ExistingModelFineTuning/Qwen3LongContext/test_qwen30b_routed.py`  
  Self-tests, RAM-cache tests, and dense-vs-routed comparisons.

### Router and storage

- `ExistingModelFineTuning/KvRouter/chunk_router.py`  
  Selects chunks/groups, assembles routed K/V, and applies causal masks.

- `ExistingModelFineTuning/KvRouter/cache_store.py`  
  Abstracts where K/V lives: VRAM hot windows, RAM cold storage, and future NVMe-backed storage.

### 40M benchmark path

- `ExistingModelFineTuning/TestModel40M/Compiled/HierarchicalGlobalAttentionFusedExactQ.py`  
  Triton-fused HGA implementation for the benchmark path.

- `ExistingModelFineTuning/TestModel40M/Compiled/benchmark_report.md`  
  Recorded 12K training/prefill benchmark.

- `ExistingModelFineTuning/TestModel40M/README.md`  
  Detailed explanation and reproducibility commands for the 40M experiments.

## Current limitations

- The repository is research code, not a pip-installable package yet.
- The Qwen3 path is an inference prototype and is not integrated with vLLM or SGLang.
- Keep the root README and `Qwen3LongContext/chat_qwen30b_fp8.py` aligned if the default chatbot switches between whole-chunk and group-level routing.
- The RAM cache is implemented; the NVMe backend is a planned extension.
- Quality/speed trade off through `keep_first`, `keep_last`, `topk_chunks`, `topk_groups`, and `group_size`.
- The 40M fused benchmark is currently the cleanest speed benchmark; the large Qwen3 path is mainly a system feasibility demo.
- Some folders contain older experimental ideas. Treat `OtherTestedIdeas/` as archive unless you are debugging history.

## Recommended citation

A paper draft is in preparation. For now, cite the repository:

```bibtex
@software{hga_2026,
  title  = {Hierarchical Global Attention},
  author = {Woernle, Frank and Fedosov, Vladimir and Grinenko, Artemiy},
  year   = {2026},
  url    = {https://github.com/vfedosov77/HierarchicalGlobalAttention}
}
```

## License

MIT. Consider adding a top-level `LICENSE` file so GitHub detects the license automatically.
