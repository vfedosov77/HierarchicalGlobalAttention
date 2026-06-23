# Qwen3-30B Long-Context via Hierarchical KV-Routing

Run **Qwen3-30B-A3B-Instruct-2507-FP8** at long context on a single 32 GB GPU (e.g. RTX 5090)
by replacing every attention layer with a drop-in, router-backed **sparse attention** whose
KV-cache lives in host RAM. Only a small, bounded working set (a few sink + local + routed
chunks) is pulled to VRAM each step, so **peak VRAM is bounded by the model weights regardless
of context length** — 32K tokens fit in the ~3 GB left after the ~29 GB of FP8 weights.

This is the same Hierarchical Global Attention idea as the 40M benchmark (`../TestModel40M`),
applied to a *pretrained* model with **no fine-tuning**: the original Q/K/V/O projections and
q/k norms are reused **by reference** (FP8-safe — weights are never copied or dequantized).

## How it works

- **Routing selects, attention reads real tokens.** Each query (per attention head) scores
  resident chunk-/group-**key** summaries to pick the top previous chunks/groups, then attention
  is computed over those chunks' **real token K/V** (`RoutedKV.attend(use_summaries=False)`).
  Group *value* summaries are never attended, so a model that never learned the summaries is not
  corrupted.
- **A-shape + routed middle.** The first `KEEP_FIRST` chunks (attention sinks) and last
  `KEEP_LAST` chunks (local context) are always resident; the top-`TOPK_CHUNKS` routed middle
  chunks are an additional recall path (MInference's A-shape + content routing).
- **Two granularities** (set by `GROUP_SIZE`): *group-level* routing (`GROUP_SIZE < CHUNK_SIZE`,
  opens the top `TOPK_GROUPS` groups of the selected chunks) or *whole-chunk* routing
  (`GROUP_SIZE == CHUNK_SIZE`). Both match the dense baseline closely (see results); the chat
  defaults to group-level.
- **RAM-cached KV, KV-head VRAM bank.** The full KV record lives in host RAM
  (`RamKVCacheStore`). Routing is per query-head, but the cold-tier gather and its bounded LRU
  VRAM bank are **KV-head-granular** (the `rep` query heads of a GQA group share KV), so
  consecutive decode steps reuse resident chunks (~99% hit rate) and only newly-required chunks
  cross PCIe.
- **Independent group-summary cache.** Group-level routing only needs each routed chunk's tiny
  *group summaries* (`M·Dh`, ≈16× smaller than its `C·Dh` token K/V), so summaries get their own,
  much larger LRU VRAM cache (`VRAM_SUMMARY_CHUNKS`). The per-step routing decision is therefore
  served from VRAM and never drags whole token chunks across PCIe just to score them — the token
  bank only ever loads the chunks whose groups are actually *opened* and attended.
- **Fast prefill.** A fresh multi-chunk block takes the vectorized chunk-parallel path
  (`KvRouter/vectorized.py`); decode streams chunk-by-chunk. Both are exact vs. dense causal
  SDPA at full coverage (selftests below).

## Files

| File | Purpose |
|------|---------|
| `qwen_routed_attention.py` | `QwenRoutedAttention` drop-in + `replace_qwen_attention_with_router` / `restore_original_attention` model surgery. |
| `chat_qwen30b_fp8.py` | Interactive chat (terminal default, or `--ui` browser chat — stdlib HTTP + SSE, no deps). |
| `test_qwen30b_routed.py` | Offline selftests (no model) + quality (`compare_on_qwen`), 32K RAM (`--ram`), and speed (`--bench`, `--speed-variants`) harnesses. |
| `try_qwen30b_fp8.py` | Minimal smoke-test of loading and a single forward. |

The routing/cache engine itself lives in `../KvRouter/` (`ChunkRouter`, `RouterConfig`,
`KVCacheStore`, `vectorized.py`).

## Quickstart

```bash
pip install -r ../../requirements.txt      # repo-root requirements.txt
# (the FP8 matmul kernels are fetched from the HF Hub on first run)

cd ~/HierarchicalGlobalAttention

# Terminal chat:
python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8

# Browser UI (binds 0.0.0.0; prints the LAN URL):
python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8 --ui

# Offline correctness selftests (no model download):
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --selftest-only

# Quality vs. dense baseline at 4K (greedy-match + perplexity, group vs whole-chunk):
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --tokens 4096 --block 16

# 32K irrelevant-prefix retrieval test + bounded-VRAM check:
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --ram --block 64 \
    --ctx-sizes 2048 32768

# Decode speed: dense vs routed RAM+cache (reports bank hit-rate):
python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --bench --tokens 4096 --block 16
```

> **FP8 memory note (RTX 5090 / SM120):** the FP8 weights leave only ~3 GB free, and the FP8
> Triton matmul autotuner OOMs on a large prefill matmul. Feed prefill in small blocks
> (`--block 16`, or `64` for routed) and keep `PREFILL_BLOCK` modest in the chat. Always run with
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the chat sets this automatically).

## Results

See [`results_qwen30b_32k_rtx5090.md`](results_qwen30b_32k_rtx5090.md) for measured context
length, VRAM, host-RAM and tokens/second on an RTX 5090.

## Configuration (chat)

Edit the constants at the top of `chat_qwen30b_fp8.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `CHUNK_SIZE` | 64 | Tokens per chunk. |
| `GROUP_SIZE` | 16 | `< CHUNK_SIZE` → group-level routing; `== CHUNK_SIZE` → whole-chunk. |
| `KEEP_FIRST` / `KEEP_LAST` | 2 / 8 | Always-resident sink / local chunks. |
| `TOPK_CHUNKS` | 20 | Routed-middle candidate chunks. |
| `TOPK_GROUPS` | 32 | Materialized groups; each query opens `TOPK_GROUPS // 2`. |
| `PREFILL_BLOCK` | 128 | Prefill block size (keep modest on FP8 — see memory note). |
| `VRAM_CACHE_CHUNKS` / `VRAM_CACHE_RESERVE_GB` | 400 / 1.5 | LRU VRAM **token** bank upper bound; the bank auto-sizes to free VRAM. |
| `VRAM_SUMMARY_CHUNKS` | 8192 | Independent LRU VRAM **group-summary** cache (≈C/M = 16× smaller per chunk). Large ⇒ group routing stays GPU-resident (≈0 misses); auto-shrinks to free VRAM. |
| 'RAM_BUDGET_GB' | 6.0 | Ram cache size(GB). |
