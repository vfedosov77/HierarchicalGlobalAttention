# Hierarchical Global Attention

A causal attention module that replaces standard dense attention in existing transformer models, enabling significantly faster training and inference at long contexts. This repository contains the fine-tuning implementation and benchmarks on a 40M-parameter SmallLM.

## Key Results

Fine-tuning a 40M-parameter model on 16K-token sequences:

| Variant | Speed | VRAM |
|---------|-------|------|
| Dense attention (baseline) | 1.81 s/step | 17 362 MiB |
| Hierarchical attention | 0.54 s/step (1.85 steps/s) | 17 440 MiB |
| **Speedup** | **3.34×** | — |

Memory footprint stays almost identical; the speedup comes entirely from reduced attention compute at long contexts.

## How It Works

Standard causal attention attends every token to every previous token — O(n²) in both time and memory. At 16K tokens this becomes the bottleneck.

Hierarchical Global Attention organizes the past context into a two-level hierarchy:

1. **Chunks** — groups of 64 contiguous tokens. Each chunk is summarized by a single key vector.
2. **Groups** — sub-chunks of 16 tokens. Each group has its own summary key.

For each query position the router:
- Scores all previous chunk summaries and selects the top-K most relevant (default K=20).
- Within each selected chunk, scores its group summaries and opens the top-K groups (default K=32) to full token-level attention.
- Always attends to every token in the current (local) chunk exactly.

This keeps attention cost roughly O(n · √n) instead of O(n²) for long sequences, while preserving the model's ability to retrieve any token in the full context when the router promotes it.

The module (`HierarchicalGlobalAttentionExactQ.py`) has the same `q_proj / k_proj / v_proj / o_proj` parameter layout as standard GQA attention, so a trained dense checkpoint loads into the HA model directly with no key remapping.

A Triton-fused fast path (`HierarchicalGlobalAttentionFusedExactQ.py`) is used automatically on CUDA for fp32 when the chunk geometry is supported.

## Repository Layout

```
ExistingModelFineTuning/
    HierarchicalGlobalAttentionExactQ.py        # reference HA module
    HierarchicalGlobalAttentionFusedExactQ.py   # Triton-fused fast path
    torch_inductor_patch.py                     # torch.compile fix for torch 2.11
    train_small_model_dense_bf16.py             # trains the dense baseline (SmallLM 40M)
    finetune_small_model_bf16.py                # fine-tunes dense → HA (this is the main script)
    speed_run_ha_from_dense_adamw_kq_loss.tsv   # training loss history

Qwen3LongContext/                               # upcoming — see below
```

Model checkpoints are hosted on Hugging Face (see below).

## Quick Start

### 1. Install dependencies

```bash
pip install torch triton transformers datasets huggingface_hub tqdm
```

Triton is required for the fused fast path. torch.compile is strongly recommended (`--compile-ha` flag, on by default).

### 2. Download checkpoints (or train from scratch)

```bash
python prepare_model.py          # downloads both checkpoints into ExistingModelFineTuning/
python prepare_model.py --dense-only  # dense baseline only
```

Or train from scratch:

```bash
cd ExistingModelFineTuning
python train_small_model_dense_bf16.py --precision bf16 --optimizer muon
```

This trains a 40M-parameter decoder (8 layers, 384 hidden, 6/2 GQA heads, 8K context) on FineWeb-10BT and saves `speed_run_dense_muon_final.pt`.

### 3. Fine-tune to Hierarchical Attention

```bash
python finetune_small_model_bf16.py \
    --dense-checkpoint speed_run_dense_muon_final.pt \
    --train-scope kq \
    --precision bf16 \
    --optimizer adamw
```

`--train-scope kq` freezes all weights except the query and key projections, which is sufficient for the router to learn the new attention pattern. Use `--train-scope attention` to also fine-tune V and O projections.

To load the dense checkpoint directly from Hugging Face:

```bash
python finetune_small_model_bf16.py \
    --dense-checkpoint-repo vfedosov/HierarchicalGlobalAttention \
    --dense-checkpoint-filename speed_run_dense_muon_final.pt \
    --train-scope kq \
    --precision bf16
```

### 4. Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--train-scope` | `kq` | `kq` / `attention` / `attention-mlp` / `full` |
| `--max-len` | 8192 | Sequence length during fine-tuning |
| `--target-tokens` | 200M | Total tokens seen during fine-tuning |
| `--precision` | `fp32` | `bf16` for ~2× memory reduction |
| `--compile-ha` | on | `--no-compile-ha` to disable torch.compile |

## Checkpoints on Hugging Face

The dense baseline and fine-tuned HA checkpoints are available at:

**[vfedosov/HierarchicalGlobalAttention](https://huggingface.co/vfedosov/HierarchicalGlobalAttention)**

| File | Description |
|------|-------------|
| `speed_run_dense_muon_final.pt` | Dense SmallLM 40M baseline (trained with Muon) |
| `speed_run_ha_from_dense_adamw_kq_final.pt` | HA SmallLM fine-tuned from the dense checkpoint (AdamW, kq scope) |

Both files are raw `state_dict` tensors loadable with `torch.load(..., weights_only=True)`.

## Causality Verification

The fine-tuning script periodically checks that teacher-forced full-sequence loss and token-by-token generation loss match within a tight tolerance. This is a necessary condition for a correct causal implementation and catches any future-token leakage at training time.

## Upcoming: Qwen 3 0.6B Long-Context (128K tokens)

The `Qwen3LongContext/` directory will contain experiments applying Hierarchical Global Attention to [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The HA module already separates processing for high-frequency and low-frequency RoPE components. Slowing down low-frequency rotation rates 4× extends the effective context from 32K to ~128K tokens at negligible cost, since the long-range routing is already handled by the hierarchical router rather than full attention. Results and scripts will be added as experiments complete.

## License

MIT
