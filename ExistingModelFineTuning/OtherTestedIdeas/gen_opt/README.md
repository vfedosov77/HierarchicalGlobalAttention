# gen_opt — pure-torch generation-optimized HierarchicalGlobalAttention

Experimental variants that optimize **generation** on long context **without
Triton and without `torch.compile`/CUDA-graphs** (pure eager PyTorch).  The
original `ExistingModelFineTuning/HierarchicalGlobalAttention.py` is left
untouched; these are drop-in replacements loaded by file path via
`benchmark_v2_vs_dense.py --ha-file ...`.

## Design (ideas borrowed from `HierarchicalGlobalAttentionTriton.py`)

* **Prefill = the training path.** A new token sequence is processed by the dense
  routed-attention forward (`_forward_dense`), the same code used for training,
  padding the sequence to full chunks.  It is a **verbatim port** of the reference
  `HierarchicalGlobalAttention._forward_dense`, so teacher-forcing loss is
  **bit-identical** to the original (see `test_equivalence.py`).  No Triton kernel.
* **Efficient cache** (same shape as the Triton version): per layer the state
  keeps all token KV, closed-chunk key summaries `chunk_k`, per-group key
  summaries `group_k`, the active chunk's running raw/rope sums, and running-max
  routing scores (`chunk_smax`/`group_smax`) = the cumulative "opened ids".
* **Per-token decode** updates only the active chunk/group summaries, routes the
  new token's top-k chunks/groups (unioned with previously opened ids), builds one
  score row, softmaxes, and multiplies by the gathered values — filling the cache
  on the way.

## Variants

| file              | prefill         | decode path                                        |
|-------------------|-----------------|----------------------------------------------------|
| `vt_torch.py`     | pure-torch dense| reference branchless static decode (from v2_static)|
| `vt_fast.py`      | pure-torch dense| op-count-reduced decode: cached aranges/masks/inv_freq, GQA einsums instead of `repeat_interleave` |
| `vt_dyn.py`       | pure-torch dense| dynamic per-token decode (operates over actual #chunks; no branchless CUDA-graph scaffolding) |

All three are bit-identical to the original on the prefill path
(`python -m ExistingModelFineTuning.gen_opt.test_equivalence`).

## Results (Qwen3-0.6B, fp32, RTX 5090, 8K context, eager)

| metric                         | dense   | HA (pure-torch) | speedup |
|--------------------------------|---------|-----------------|---------|
| **prefill 8K**                 | 976 ms  | **406 ms**      | **2.40x** |
| decode tok/s (8K ctx)          | 45      | 14              | 0.31x   |

### Why decode does not beat dense here — and where it will

For a 0.6B model, per-token decode is dominated by the **shared** cost (lm_head +
MLP + projections ≈ 20 ms/token); dense self-attention over 8K keys is one fused
SDPA kernel and is nearly free, so dense decode barely changes with context
(44.7 tok/s @ short ctx → 45 tok/s @ 8K).  Profiling shows HA decode is
**CPU-launch-bound**: ~50 tiny ops per layer × 28 layers, while the GPU sits idle
(~0.38 ms of GPU work per layer).  In pure eager (no CUDA graphs to amortize
launches) that op count, not the FLOPs, sets the speed.  HA pays the same shared
cost **plus** routing ops, so it cannot beat dense on per-token decode at this
size.

The decode win is an **architecture-at-scale** property: it appears once dense
attention dominates per-token cost (large models / much longer context, where
dense attention FLOPs and the O(context) KV-cache concat outgrow the fixed MLP
cost — and where dense fp32 prefill already OOMs at 32K on a 32 GB GPU).

### The real 8K win: prefill / total generation

Processing an 8K context is **2.40x faster** with the pure-torch routed prefill.
For a generation of `n` tokens, total time ≈ `prefill + n·decode`:

* `n < ~12`: HA total is faster (prefill dominates).
* prefill-only / scoring / short answers over long context: HA clearly wins.

## Run

```bash
source ../../my_env/bin/activate   # from this dir; torch 2.11 cu128, RTX 5090
cd <repo-root>
python -m ExistingModelFineTuning.Qwen3LongContext.benchmark_v2_vs_dense \
    --ha-file ExistingModelFineTuning/gen_opt/vt_fast.py \
    --ref-ha-file ExistingModelFineTuning/gen_opt/vt_torch.py \
    --context-lens 8192 --compile false \
    --decode true --decode-context-len 8192 --decode-tokens 64 \
    --check-loss true --loss-len 2048
```

The benchmark reports prefill speedup, decode tok/s, total-generation tok/s, and
a teacher-forcing loss-equality check against the reference HA.
