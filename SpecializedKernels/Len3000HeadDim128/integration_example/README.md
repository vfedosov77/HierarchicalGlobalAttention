# HGA on stock Alpamayo 1.5 (no ROS)

Drop-in example: official `Alpamayo1_5` + **this folder’s own kernel**.
**No ROS node, no speculative CoT, no distilled-FP8 overlay, no camera
pruning.** Attention is HGA. Diffusion uses the same 10-step Euler
expert as ROS, including the **CUDA-graph Euler loop** that makes ROS
report ~117 ms instead of ~300 ms eager.

The ROS-optimized node must keep using the parent files
(`Len3000HeadDim128/kernel.py`, `attention.py`, `alpamayo_adapter.py`).
Those are not the files this example imports.

| File | Who uses it |
|---|---|
| `../kernel.py` + `../attention.py` + `../alpamayo_adapter.py` | ROS |
| `kernel.py` + `attention.py` + `adapter.py` + `cache.py` + `graphed.py` | `run_nogener.py` |
| `eval_mcap_routes.py` | dense vs HGA ADE/FDE on a real rosbag |

## No generation

**This example never runs VLM decode / Chain-of-Causation.**

`sample_no_generation` does:

1. **VLM prefill only** — `model.vlm(..., use_cache=True)` on the visual
   tokens + text prompt. `max_new_tokens = 0`. No `<|traj_future_start|>`
   is produced.
2. **Expert diffusion** — 10 Euler steps, 64 action tokens, using that
   prefill KV cache.

That is the same-S comparison used in the ROS writeup (`opt_no_cot`).
CoT-on is **not** a fair length match: HGA would reuse a long CoC prefix
while dense often EOS-stops early.

HGA is wired into **both**:

| Call | Backend after `adapt_alpamayo(model)` |
|---|---|
| Causal VLM **prefill** (GQA 32/8, `head_dim=128`) | `vlm_prefill_attention` (true 2-level, 192 tokens) |
| Expert **diffusion** (GQA **16/8**, 64 queries × prefix) | pack-and-reuse `diffusion_cross_attention` |

The vision encoder (`head_dim=72`) is **not** HGA. Optional
`--fast-vision` only swaps ViT Flash-2 + fp32 RoPE for batched SDPA +
bf16 RoPE.

Decode (`q_len ≠ k_len`) is unused here because nothing is generated.

## Why ROS diffusion is ~117 ms and eager stock was ~300 ms

Same 10 Euler steps, same 36-layer expert, same 64 action tokens. ROS
is not running a different denoiser. The node wraps it in
`GraphedTrajectoryDecoder` (`cuda_graph_diffusion.py`):

| | ROS node (bag, no-CoT) | This example, **eager** | This example, **CUDA-graph** |
|---|---:|---:|---:|
| Dense SDPA | **123 ms** | 311 ms | **169 ms** |
| HGA 2L 192 | **117 ms** | 305 ms | **85 ms** |

The ~180 ms gap is host launch: 10 × `expert.forward` × 36 HF layers
from Python, plus `DynamicCache` cat/crop. A CUDA graph records the
whole Euler loop once and replays it. That is what `graphed.py` does
here (no `alpamayo_ros` import).

ROS dense is still a bit faster than our graphed dense (123 vs 169)
because the node also applies the **distilled FP8 overlay** on expert
linears. HGA here is faster than ROS HGA (85 vs 117): we drop the
all-attend 4D mask (HF otherwise expands GQA and takes the math SDPA
kernel) and pack the 176 routed tokens instead of re-gathering.

`--eager` turns the graph off if you want the official
`model.diffusion.sample` loop.

## Why diffusion needs its own integration (besides the graph)

Stock Alpamayo (this path) does **not** run ROS `record_attention`.
Dense expert attention is already a 64×3K GQA SDPA. Isolated, that is
~44 µs. The ROS 2L kernel is ~56 µs — it wins on the bag by replacing
`record_attention` + SDPA (~160 µs), not by beating bare SDPA.

An earlier attempt reused **route indices** every 3 Euler steps. That
was the wrong thing to reuse:

1. Chunk/group scoring is only ~8 µs of a ~64 µs launch. Skipping it
   leaves the 64×256 **scattered gather-attend** of 11 groups from the
   3K cache. End-to-end diffusion stayed **+20 ms** vs dense
   (`logs/integration_nogener_route3.log`).
2. Alpamayo 1.5’s expert is GQA **16/8** (`hidden_size=2048`). The
   adapter used to size buffers from the VLM text config (**32/8**), so
   the preallocated BSHD output never matched and was re-allocated
   every layer.
3. Each Euler step `torch.cat`s 64 tokens onto the VLM
   `DynamicCache` and `crop`s them back — 36 layers × 10 steps. The ROS
   node already avoided that with `StaticExpertCache`.
4. The official 4D prefix mask is all-attend when there is no CoT
   (`offset == prefill_len`). Hugging Face still treats a 4D float mask
   as “expand GQA and take the math SDPA kernel” (~267 µs isolated).
   HGA ignores that mask; the example now passes `attention_mask=None`
   into the expert.

This folder therefore ships a **custom diffusion path**:

1. **Fixed-slot cache** (`cache.py`) — copy the VLM prefix once into
   `[B, Hkv, n_prompt+64, D]`. Each layer overwrites the last 64 slots.
   No cat, no crop.
2. **Pack-and-reuse** — on Euler steps 0, 3, 6, 9 the kernel routes
   11 groups **and packs** those 176 tokens into a contiguous per-head
   buffer. The other six steps only attend the packed prefix + the 64
   live diffusion keys (`HGA_DIFF_ROUTE_EVERY=3`).
3. **Expert-sized buffers** — 16 query heads, 8 KV heads.

VLM prefill stays the same true 2-level kernel as ROS (means + Triton
chunk top-4 + 16-group attend).

## Install

You need the **stock** Alpamayo 1.5 Python package (the public
`alpamayo1_5` tree, e.g. `Alpamayo-ROS/src`), a checkpoint, and this
repo on `PYTHONPATH`.

```bash
export PYTHONPATH=/path/to/Alpamayo-ROS/src:/path/to/HierarchicalGlobalAttention
# or:  --alpamayo-src /path/to/Alpamayo-ROS/src

# Local official-fp8 dir, or the HF id:
export ALPAMAYO_MODEL=/path/to/Alpamayo-1.5-10B-official-fp8
# export ALPAMAYO_MODEL=nvidia/Alpamayo-1.5-10B
```

## Run

```bash
# HGA only (prefill + diffusion)
python -m SpecializedKernels.Len3000HeadDim128.integration_example.run_nogener \
    --model "$ALPAMAYO_MODEL"

# Dense vs HGA on the same prompt (fair, no CoT)
python -m SpecializedKernels.Len3000HeadDim128.integration_example.run_nogener \
    --model "$ALPAMAYO_MODEL" --mode compare --warmup 1 --iters 3

# Real clip instead of synthetic cameras (timing only)
python -m SpecializedKernels.Len3000HeadDim128.integration_example.run_nogener \
    --model "$ALPAMAYO_MODEL" --clip-pt /path/to/sample_clip_data.pt --mode compare

# Dense vs HGA **route error** on a real ROS bag (cameras + future odometry)
python -m SpecializedKernels.Len3000HeadDim128.integration_example.eval_mcap_routes \
    --model "$ALPAMAYO_MODEL" \
    --mcap /path/to/rosbag2_20260805_164006_0.mcap
```

Without `--clip-pt` the timing script builds **synthetic** 4×4 camera
frames so you can time the kernels without PhysicalAI / a rosbag.

First iteration is Triton / FA compile. Quote the **median of timed
iters after warmup**.

## Measured (this machine)

Stock `Alpamayo1_5`, official-fp8 weights, **no generation**, synthetic
4 cameras × 4 frames (**S = 2929**). RTX 5090. `run_nogener.py --mode
compare --warmup 1 --iters 3`. Median of the 3 timed iters after
warmup (first call captures the graph). Expert dense backend is
**SDPA**. Diffusion is the **CUDA-graph** 10-step Euler loop.

| (median, ms) | VLM prefill | Diffusion (graph) | Predict |
|---|---:|---:|---:|
| Dense (FA-2 VLM / SDPA expert) | 269 | 169 | 438 |
| **HGA 2L 192, pack every 3** | **259** | **85** | **345** |
| Δ HGA − dense | **−10** | **−84** | **−93** |

ROS bag, same GPU, no-CoT, also CUDA-graphed: dense **123** / HGA
**117** (plus distilled FP8 overlay, S≈2324). Eager stock (no graph)
was dense 311 / HGA 305.

Log: `logs/integration_nogener_graph.log`.

On this CUDA-graph path the **diffusion** win is the large one
(−84 ms). VLM prefill is still a bit faster (−10 ms); vision + MLP
dominate that slice.

## Route quality vs dense (real bag)

Same no-CoT `sample_no_generation` path, official-fp8 weights, RTX 5090.
Bag `rosbag2_20260805_164006_0.mcap` (the ROS timing bag). Three cameras
(left / front-wide / right), matching `opt_ignore_front_tele`. Ground
truth is the next **6.4 s** of `/sensors/odometry` in the t0 ego frame
(64 waypoints × 0.1 s). Five windows: early turn, sharp turn, exit
turn, stopped/creep, gentle curve.

ADE / FDE in **metres**. Δ = HGA − dense (negative = HGA closer to GT).
`|H−d|` is the mean waypoint gap between the two predictions.

| Case | ADE dense | ADE HGA | Δ ADE | FDE dense | FDE HGA | Δ FDE | \|H−d\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| early turn | 9.51 | **9.14** | **−0.36** | 25.0 | **18.5** | **−6.5** | 10.1 |
| sharp turn | **6.05** | 12.80 | +6.75 | **15.9** | 30.4 | +14.5 | 13.4 |
| exit turn / straighten | 9.96 | **1.30** | **−8.66** | 28.8 | **5.0** | **−23.8** | 9.6 |
| stopped / creep | **1.25** | 1.84 | +0.58 | **5.4** | 7.6 | +2.3 | 0.6 |
| gentle curve *(both fail)* | **95.4** | 102.6 | +7.2 | **178** | 198 | +19.9 | 9.1 |
| **mean, all 5** | 24.4 | 25.5 | +1.1 | 50.6 | 51.9 | +1.3 | 8.6 |
| **mean, drop failed curve** | 6.69 | **6.27** | **−0.42** | 18.8 | **15.4** | **−3.4** | 8.4 |

Short horizon, failed curve dropped: ADE@1s **0.32 m dense / 0.48 m HGA**,
ADE@3s **1.60 / 2.09**.

HGA and dense stay near each other (~9 m mean gap over 6.4 s) and
trade wins. The gentle-curve window is a **shared** failure (both
~170° the wrong way, FDE ~180–200 m), not an HGA-only miss. Open-loop,
no CoT, no ROS F-theta remapping — 6.4 s FDE is noisy; 1–3 s is the
more honest driving number.

Log: `logs/eval_mcap_routes.log`.

Reproduce timing and bag ADE:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=/path/to/Alpamayo-ROS/src:/path/to/HierarchicalGlobalAttention \
  python -m SpecializedKernels.Len3000HeadDim128.integration_example.run_nogener \
    --model /path/to/Alpamayo-1.5-10B-official-fp8 --mode compare

HF_HUB_OFFLINE=1 PYTHONPATH=/path/to/Alpamayo-ROS/src:/path/to/HierarchicalGlobalAttention \
  python -m SpecializedKernels.Len3000HeadDim128.integration_example.eval_mcap_routes \
    --model /path/to/Alpamayo-1.5-10B-official-fp8 \
    --mcap /path/to/rosbag2_20260805_164006_0.mcap
```

Isolated kernels (expert 16/8, S=2929, Q=64):

```bash
python SpecializedKernels/Len3000HeadDim128/integration_example/bench_diff.py
```

## Minimal hook

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from SpecializedKernels.Len3000HeadDim128.integration_example.adapter import (
    adapt_alpamayo,
)
from SpecializedKernels.Len3000HeadDim128.integration_example.nogener import (
    sample_no_generation,
)

model = Alpamayo1_5.from_pretrained(ckpt, dtype=torch.bfloat16).cuda()
adapt_alpamayo(model)          # VLM prefill + expert diffusion
pred_xyz, pred_rot, extra = sample_no_generation(model, data)
# extra["stats"]["generated"] == 0
```

`adapt_alpamayo` registers the HF attention backends, preallocates
group/chunk means and packed prefix buffers, and (via
`sample_no_generation`) copies the VLM cache into a fixed-slot expert
cache, then **CUDA-graphs** the 10-step Euler loop. Diffusion routing
is recomputed every 3 Euler steps; the packed 176 prefix tokens are
reused on the other two (`HGA_DIFF_ROUTE_EVERY=3`).

Do **not** call `sample_trajectories_from_data_with_vlm_rollout(...,
max_generation_length=256)` if you want this comparison: that is CoT
generate, and sequence lengths diverge.

## What this is not

- Not `alpamayo_optimized_node.py` (no ROS, no speculative decode, no
  `record_attention` top-k). The Euler CUDA graph **is** included
  (`graphed.py`); it is the same idea as ROS
  `GraphedTrajectoryDecoder`.
- Not a CoT-text quality claim. The bag ADE table above is **no-CoT**
  open-loop vs odometry.
- Not vision HGA (`head_dim=72`).

Kernel details and ROS bag numbers:
[`../README.md`](../README.md), `/home/vladimir/Alpamayo-ROS/HGAintegration.md`.
