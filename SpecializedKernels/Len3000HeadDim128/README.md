# Len3000HeadDim128

Alpamayo VLM + diffusion attention. Tensors are already RoPE’d
`[B, H, S, 128]` (no Flash transpose).

**What this folder runs now:** true two-level HGA — 128-token chunks,
then 16-token groups, **192 tokens** (4 previous chunks, 11 groups +
current). That is what `adapt_alpamayo()` installs.

Older one-level and nested kernels are in
[`other_variants/`](other_variants/). The live ROS path is this
folder’s true 2L 192.

`head_dim` must be 128. Vision encoder (`head_dim=72`) is not patched.
Decode (`q_len ≠ k_len`) falls back to SDPA.

| Function | When |
|---|---|
| `vlm_prefill_attention` | Causal VLM prefill, GQA 32/8 |
| `diffusion_cross_attention` | 64 expert queries; no `record_attention` |

## Wire into Alpamayo

After load, before the first infer / CUDA-graph capture:

```python
from SpecializedKernels.Len3000HeadDim128.alpamayo_adapter import adapt_alpamayo

adapt_alpamayo(model)
```

Stock Alpamayo (no ROS, **no CoT generation**) is in
[`integration_example/`](integration_example/). That folder ships its
**own** kernel (`integration_example/kernel.py`) for route-reuse
experiments. ROS must keep using this folder’s `kernel.py` /
`attention.py` / `alpamayo_adapter.py` only.

This registers the HF backends, preallocates group/chunk means, the
chunk-route table, and BSHD outputs, sets `diffusion_kv_top_fraction=1`,
and refreshes prefix means once per frame.

```python
from SpecializedKernels.Len3000HeadDim128 import (
    vlm_prefill_attention,
    diffusion_cross_attention,
)

attn = vlm_prefill_attention(q, k, v, softmax_scale=scale)
attn = diffusion_cross_attention(q, k, v, n_prompt=prefill_len, softmax_scale=scale)
```

`record_attention` is a dense 64×3K softmax (~119 µs) used only to pick
tokens, plus ~50 µs SDPA ≈ 160 µs. The diffusion kernel replaces both.

Setup for bag numbers: RTX 5090, official-fp8 Alpamayo 1.5 +
distilled FP8 overlay, bag `rosbag2_20260805_164006_0.mcap`,
`bmw_car_eval_v4.yaml`, 12 inferences, `opt_adaptive_cameras:=false`,
`opt_ignore_front_tele:=true`, repair=0. **Headline ROS pair uses
`opt_no_cot:=true` (max_gen=0, CoC=0 on both sides).** Median of
frames 3–12 (frame 1 is compile). Isolated: GQA 32/8, S=2688, Q=64,
bf16. **Diff** in kernel tables is the **diffusion** kernel, not
“difference vs SDPA”.

Longer ROS writeup: `/home/vladimir/Alpamayo-ROS/HGAintegration.md`.

---

## Why this variant (decision trail)

### 1. One-level 64 / top-3 was the first ROS win

Dense expert path = `record_attention` + SDPA. One-level 64, top-3
previous blocks + current (~256 VLM tokens, ~8.5% at S=3000) replaced
that.

| Isolated | Routed | Dense |
|---|---:|---:|
| VLM prefill | 180–280 µs | 400–506 µs SDPA |
| Diffusion, reused chunk means | 58–75 µs | ~50 µs SDPA alone; **~160 µs record+SDPA** |

| ROS median predict, frames 3–12 | ms |
|---|---:|
| Dense + `record_attention` | **390** |
| One-level 64 / top-3 | **373 (−17 ms)** |

Almost all 15–17 ms is the expert. VLM is ~215–225 ms of vision + MLP.
`repair_tokens=0` still leaves a CoC prefix in the KV cache. Default
CoT-on is **not** a fair length match: HGA reuses **256** CoC tokens,
dense EOS-stops at **15**. `opt_no_cot` (`max_gen=0`, CoC=0) is the
same sequence on both sides.

### 2. Same top-k at C=128 is not a fair test

Fixed `topk=3` opens 4×128 = **512** tokens vs 4×64 = **256**. That is
why C=128 looked slow (415 µs), not because 128-tiles are a bad shape.

Unfair sweep (`topk=3` for every block):

| Block | VLM tokens | VLM µs | L1 vs SDPA | Diff µs |
|---:|---:|---:|---:|---:|
| 16 | 64 | 223 | 0.155 | 111 |
| 32 | 128 | 196 | 0.106 | 75 |
| 64 | 256 | 212 | 0.069 | 80 |
| 128 | **512** | 415 | 0.040 | 92 |

At a **shared 256-token** budget the ranking flips (larger tiles win):

| Variant | VLM tok | VLM µs | vs SDPA | L1 | Diff µs |
|---|---:|---:|---:|---:|---:|
| 1-level 32, top-7+cur | 256 | 262 | 1.53× | **0.066** | 65 |
| 1-level 64, top-3+cur | 256 | 182 | 2.21× | 0.069 | 58 |
| 1-level 128, top-1+cur | 256 | **138** | **2.91×** | 0.077 | 53 |
| 1-level 128, top-3+cur *(unfair)* | 512 | 371 | 1.08× | 0.040 | 65 |

ROS C=32 was 378 ms vs C=64 **373 ms**. One-level stayed on 64/top-3
until two-level quality was measured.

### 3. Nested coarse-then-fine inside the old ROS kernels lost

Score all coarse means per query tile, then fine tiles inside winners.
`tl.dot` needs N≥16, so 4- and 8-token tiles pad to 16.

At 256 VLM tokens (isolated):

| Variant | VLM µs | L1 | Diff µs |
|---|---:|---:|---:|
| 1-level 64 | 182 | 0.069 | 58 |
| 1-level 128 | 138 | 0.077 | 53 |
| nested 32/4 (14×4) | 1342 | 0.075 | 171 |
| nested 64/8 (6×4) | 508 | 0.076 | 102 |
| nested 128/8 (2×8) | 168 | 0.080 | 84 |
| nested 128/16 (2×4) | 142 | 0.080 | 68 |
| nested 32/8 (3×2), 80 tok | 508 | 0.148 | 235 |

32/4 and 64/8 were slower than dense SDPA. 128/16 was close to
one-level 128 on speed and a bit worse on L1.

On the bag, nested 128/16 and 64/8 did **not** beat one-level 64:

| | tokens | predict | VLM | Diff | Isolated VLM L1 |
|---|---:|---:|---:|---:|---:|
| one-level 64 @ 8% | 256 | **373** | 225 | **114** | **0.069** |
| nested 128/16 @ 8% | 256 | 386 | 224 | 131 | 0.080 |
| nested 128/16 @ 4% | 128 | 398 | 232 | 135 | 0.121 |
| nested 128/16 @ 2% | 64 | 380 | 224 | 125 | 0.161 |
| nested 64/8 @ 4% | 128 | 397 | 230 | 133 | 0.119 |

Decision: drop nested-in-kernel 32/4, 64/8, 128/16 for ROS. Extra
route/attend steps cost more than they save at 3K.

### 4. True two-level (chunk then group) is more precise

Paper layout, not the nested kernel above:

1. Each **128-token chunk** picks 4 previous chunks (shared table).
2. Each **16-token group** scores only groups inside those chunks +
   current, attends the winners + the current group.

This can open 16-token details inside several distant 128-chunks.
One-level 64/top-2 can open only two whole 64-blocks.

**Quality at 192 tokens** (exact chunk top-4):

| 192-token setup | i.i.d. random L1 (cos) | Structured: 4 distant chunks, 2 hot groups L1 (cos) |
|---|---|---|
| 1-level 64, top-2 | 0.086 (0.583) | 0.047 (0.541) |
| **true 2L, 11 groups + current** | **0.082 (0.615)** | **0.030 (0.713)** |

Two-level is already a bit better on white-noise Q/K. When important
tokens sit in **distant** chunks, it wins clearly (0.030 vs 0.047,
cosine 0.71 vs 0.54).

The hierarchy shape is fixed (128 → 16, top-4 then top-11); only
scores change. Live code is specialized for that one shape.

A Triton chunk router agrees with exact PyTorch top-4 on **99.8%**
of late (chunk, head) pairs (same set). The ~0.2% 4th-place swaps
are near-ties; i.i.d. L1 vs SDPA is 0.092 vs 0.082 for exact top-4.
Eager `torch.topk` on the 21×21 table was ~200 µs — not usable.
Structured-distant quality is why 2L stays (see below).

### 5. Specialized 128/16 kernels (full path, including routing)

Older isolated tables compared 2L **attend only** to 1L-64 (route
inside Triton). Eager PyTorch chunk top-4 then made full-path 2L
**~2×** slower than 1L (459 vs 233 µs at S=2688). That router is
replaced by hierarchy-specific Triton:

1. **Means** — one CTA per 128-token KV chunk writes 8 group means
   and 1 chunk mean (one K read).
2. **VLM chunk route** — one CTA per query chunk × head: mean-Q vs
   the 32-slot chunk-K table, bf16 `tl.dot`, top-4.
3. **VLM attend** — one CTA per 16-token group: 16×64 group-score,
   then 11 coalesced 16×16 attends + causal current. A 16×256
   gather-attend was **slower** (uncoalesced K/V).
4. **Diffusion** — chunk top-4 is fused into the 64-query kernel
   (prefix means reused).

Remeasured RTX 5090, GQA 32/8, bf16, `bench_fused.py`:

**VLM prefill, full call** (means + route + attend every layer):

| S | 1L-64 top-3+cur (256 tok) | 2L 11+cur (192 tok) | 2L / 1L |
|---:|---:|---:|---:|
| 2324 | 261 | **199** | **0.76×** (faster) |
| 2688 | 231 | **212** | **0.92×** (faster) |
| 3000 | 291 | **230** | **0.79×** (faster) |

Breakdown at S=2688: means 31 + Triton top-4 33 + attend 162 ≈ 212.

**Diffusion, prefix means reused:**

| S | 1L-64 reused | 2L fused route+attend | 2L / 1L |
|---:|---:|---:|---:|
| 2324 | 54 | **57** | 1.06× |
| 2688 | 53 | **56** | 1.05× |
| 3000 | 54 | **57** | 1.07× |

Quality at 192 tokens (i.i.d. L1 is Triton top-4; structured numbers
are the exact-top-4 measurement):

| Variant | Tokens | L1 i.i.d. (cos) | L1 structured (cos) |
|---|---:|---|---|
| 1-level 64 top-3 | 256 | 0.069 | — |
| 1-level 64 top-2 | 192 | 0.086 (0.583) | 0.047 (0.541) |
| **true 2L, 11 groups + current** | **192** | **0.092 (0.570)** / exact 0.082 (0.615) | **0.030 (0.713)** |
| true 2L, 7 groups | 128 | 0.117 | — |
| true 2L, 15 groups | 256 | 0.068 | — |

2L / 192 is the **precision** choice at 192 tokens (distant 16-token
detail) and, with these kernels, **faster than dense on the bag**.

Fair ROS bag (`opt_no_cot:=true`, CoC=0 both sides, n=3–12): true 2L
192 **predict 363 ms** vs dense + `record_attention` **376 ms**
(−13 ms). VLM **215 vs 222**, diffusion **117 vs 123.5**. HGA is
faster on every frame 3–12 for predict, VLM, and diffusion. The
expert win is replacing `record_attention` + SDPA; the VLM win is
the routed prefill at the same S.

### 6. FP8 attention and residuals (not used on the live 2L path)

Official-fp8 linears emit **bf16** activations. Native FP8 pointers
(no in-kernel cast) are faster if Q/K/V are already e4m3: VLM 185 vs
253 µs, diffusion 71 vs 113 µs. Packing Q+K+V every VLM layer (~122 µs)
eats the VLM win. Diffusion prefix packed once ≈ 5 ms over 360 calls,
not 15. Live 2L stays on bf16 pointers.

Unscaled FP8 residuals NaN at layer 6 (activation ~26k vs e4m3 max
448). Clamps stop NaNs, move the trajectory by metres (mean |Δ| 1.86 m),
and erase the 15 ms (bag 392 vs 373). Not enabled.

---

## ROS bag summary (this folder)

End-to-end predict, same bag and flags as above. Median frames 3–12.

**Fair pair — `opt_no_cot:=true`, CoC=0 both sides**
(`logs/ros_2l_nocot_20260815_172613.log`,
`logs/ros_dense_nocot_20260815_172801.log`):

| | predict | VLM | Diff | Isolated VLM L1 |
|---|---:|---:|---:|---:|
| Dense + `record_attention` | 376 | 222 | 123.5 | 0 |
| **True 2L 192 (this folder)** | **363 (−13)** | **215 (−7)** | **117 (−6.5)** | **0.092 i.i.d. / 0.030 structured** |

CoT-on (`repair_tokens=0` only) is **not** the same S: 2L reuses
CoC=256, dense EOS-stops at CoC=15. That pair was predict 376 vs 383
(VLM 227 vs 224) — the extra ~241 cached CoC tokens on the HGA side
hid the VLM win. Logs: `logs/ros_2l_fused_20260815_154434.log`,
`logs/ros_dense_20260815_154635.log`.

At the **same 192 tokens**, 2L is more precise than 1L-64 top-2
(0.082 exact / 0.030 structured vs 0.086 / 0.047). Isolated
attention (S=2688): 2L **212 µs** vs 1L-256 **231 µs** vs SDPA
**400 µs**.

---

## What not to use these kernels for

- VLM decode (`q_len=1`) — SDPA.
- Vision encoder (`head_dim=72`).
- Sequences longer than 4096 tokens.
- The old FA-shaped wrapper (`other_variants/flash_attn.py`) — extra
  transposes.
