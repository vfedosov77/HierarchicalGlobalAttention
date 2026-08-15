# Len3000HeadDim128 — Alpamayo VLM + diffusion

Two functions for the Alpamayo optimized node. Both take the tensors Qwen and
the action expert already have after `q_proj` + RoPE: **`[B, H, S, 128]`**.
No FlashAttention `[B, S, H, D]` transpose on the inputs.

CoT is **off** on the specialized VLM path. Prefill is visual tokens plus any
extra prompt tokens (traj history, short text). Decode (`q_len ≠ k_len`) falls
back to SDPA.

| Function | When | Q | K/V | Mask |
|---|---|---|---|---|
| `vlm_prefill_attention` | VLM **prefill** (visual + extra tokens) | `S ≈ 3K` causal | same `S`, GQA 32/8 | none (pad, don’t mask) |
| `diffusion_cross_attention` | Expert denoiser, every Euler step | `Q = 64` waypoints, **non-causal** | `n_prompt + 64` (VLM prefix then diffusion slots) | none |

**Default routing (what Alpamayo-project uses):** one-level, **64-token** block means,
**top-3** previous / prefix blocks plus the local tile (~256 VLM tokens,
~8.5% of a 3000-token prefix). `head_dim` must be **128**.

Qwen3-VL-8B language model: **36 layers, 32 Q heads, 8 KV heads, `head_dim=128`**.
The vision encoder is **`head_dim=72`** — these kernels do not apply there.

Optional experiments (env, forwarded by Alpamayo `docker/run.sh`):

| Env | Meaning |
|---|---|
| `HGA_ROUTE_BLOCK=16\|32\|64\|128` | one-level block size (default 64) |
| `HGA_HIERARCHY=128/16` or `64/8` | nested two-level (coarse/fine) |
| `HGA_SPARSITY=8\|4\|2` | token budget for a hierarchy (~256 / 128 / 64 VLM tokens) |
| `HGA_DOT_DTYPE=bf16\|fp8\|fp8e4\|fp8e5` | tensor-core dtype for QK / PV |

Full Alpamayo bag writeup: [`Alpamayo-project/HGAintegration.md`]
(or `Alpamayo-project/HGAintegration.md`).

## What `record_attention` is (and why we drop it)

In `cuda_graph_diffusion.py`, when `kv_top_fraction < 1`, each
refresh still runs **dense** SDPA, then:

```python
scores = einsum("bhgqd,bhkd->bhgqk", query, key)  # 64 queries × ~3072 keys
importance = softmax(scores).sum(...)             # which prefix tokens mattered
cache.select_prompt(importance)                   # gather those K/V
```

That full 64×3K softmax exists **only to pick tokens**. Measured on an RTX 5090
that pick step is **~119 µs**, plus **~50 µs** SDPA → **~160 µs** per refresh.

`diffusion_cs_attention` **replaces both**: it picks chunks from the
block-mean keys and attends them in one launch. Do **not** attach
`_alpamayo_topk_cache` / `record_attention`. Use a static prompt cache
(`kv_top_fraction=1` / `StaticExpertCache`).

## Wire into Alpamayo-project

After the model is loaded, **before the first infer** (so diffusion CUDA graphs
capture these kernels):

```python
from SpecializedKernels.Len3000HeadDim128.alpamayo_adapter import adapt_alpamayo

adapt_alpamayo(model)       # Alpamayo1_5
# or adapt_alpamayo(wrapper)    # AlpamayoOptimizedModelWrapper
# or adapt_alpamayo(predictor)  # FP8OptimizedPredictor
```

`adapt_alpamayo` registers the HF backends, preallocates `chunk_k` + `[B,S,H,D]`
outputs, sets `diffusion_kv_top_fraction=1`, and rebuilds per-layer prefix key
averages once per camera frame.

Lower-level (without the Alpamayo-project wrapper):

```python
from SpecializedKernels.Len3000HeadDim128 import attach_to_alpamayo, set_diffusion_prompt

attach_to_alpamayo(model.vlm, model.expert, topk=3)
set_diffusion_prompt(model.expert, expert_cache.layers[0].keys, n_prompt=prefill_len)
```

Keep `attention_mask=None` and `expert_non_causal_attention=True` (already the
default). Or call the kernels yourself (already BHSD):

```python
from SpecializedKernels.Len3000HeadDim128 import (
    vlm_prefill_attention,
    diffusion_cross_attention,
    prompt_chunk_keys,
)

attn = vlm_prefill_attention(q, k, v, softmax_scale=scale, topk=3)
ck = prompt_chunk_keys(k_cache, n_prompt=prefill_len)  # once per frame
attn = diffusion_cross_attention(
    q, k, v, softmax_scale=scale, topk=3, n_prompt=prefill_len, chunk_k=ck,
)
```

Pass `out=` and `chunk_k=` to reuse buffers inside a CUDA graph.

**Diffusion, 10 Euler steps:** prefix K does not change. Build chunk keys once
when the VLM cache is copied into the expert. Recomputing the mean every step
costs ~100 µs and erases the gain.

## Results (RTX 5090)

### Isolated kernels, Alpamayo shapes (GQA 32/8, S=2688–3000, Q=64, bf16)

| Path | Time | vs dense |
|---|---:|---:|
| VLM SDPA | 400–506 µs | — |
| **`vlm_prefill_attention` (C=64, top-3)** | **180–280 µs** | **~1.6–2.2×** |
| FlashAttention same VLM shape | ~519 µs | — |
| Diffusion SDPA 64×3K | ~50 µs | — |
| `record_attention` only | ~119 µs | — |
| Old path: record + SDPA | **~160 µs** | — |
| **Routed diffusion, `chunk_k` reused** | **58–75 µs** | **~2.5× vs record+SDPA** |

**Diff** in later tables is the diffusion kernel, not “difference vs SDPA”.

Reproduce:

```bash
python -m SpecializedKernels.Len3000HeadDim128.bench
```

### Match token budget when comparing block sizes

Same `topk=3` at C=128 opens **512** tokens vs **256** at C=64 — that is why
128 looked slow. At a shared **256-token** VLM budget:

| Variant | VLM µs | vs SDPA ~402 µs | mean \|err\| vs SDPA |
|---|---:|---:|---:|
| 1-level 32, top-7+cur | 262 | 1.53× | **0.066** |
| 1-level 64, top-3+cur | 182 | 2.21× | 0.069 |
| **1-level 128, top-1+cur** | **138** | **2.91×** | 0.077 |
| 1-level 128, top-3+cur (512 tok) | 371 | 1.08× | 0.040 |

Larger one-level tiles are faster at the same token count. Alpamayo-project still defaults
to 64/top-3 (see bag table).

### Nested two-level (coarse then fine, in these kernels)

At the same 256 tokens, 32/4 and 64/8 are slower than dense SDPA (`tl.dot`
pads 4- and 8-token tiles to 16). 128/16 is close to one-level 128 (142 vs
138 µs) and a bit less accurate. On the Alpamayo-project bag, 128/16 @ 8/4/2% and 64/8 @
4% are all **slower** than one-level 64 (386–398 vs 373 ms). Extra
route/attend steps cost more than they save at 3K / 5090.

### True two-level (chunk then group) — standalone, not in Alpamayo-project

Paper layout: each 128-chunk picks 4 previous chunks; each 16-group routes
only inside those chunks + current.

| Variant | Tokens | Kernel µs | + fused chunk route |
|---|---:|---:|---:|
| 1-level 64 top-3 | 256 | **181** | — |
| 1-level 64 top-2 | 192 | **154** | — |
| true 2L, 15 groups | 256 | 208 | 240 |
| true 2L, 11 groups | 192 | 169 | 200 |
| true 2L, 7 groups | 128 | 130 | 161 |

Not faster than one-level at 3K (more CTAs, many 16×16 attends). Quality
**is** better when important groups sit in **distant** chunks (structured
test: L1 0.030 vs 0.047 for 1-level 64/top-2 at 192 tokens). On i.i.d.
random Q/K the two are close; quote quality from an **exact** top-4 route
(a 99%-agree fused router flipped that ranking).

Right design for long context / KV offload, not for beating a one-level
specialized kernel on a 3K in-VRAM working set.

### Alpamayo project bag (official-fp8, 12 frames, repair=0)

| (median predict, n=3–12) | ms |
|---|---:|
| Dense + `record_attention` | **390** |
| **One-level 64 / top-3 (default)** | **373 (−17 ms)** |
| Nested 128/16 @ 8% | 386 |
| Nested 128/16 @ 4% / 64/8 @ 4% | 398 / 397 |
| HGA + FP8 residual clamps | 392 |

Almost all of the 15 ms is the expert (no `record_attention`). Timed frames
already skip CoT decode (`recalculated=0`).

### FP8

Official-fp8 linears emit **bf16** activations. Native FP8 *pointers* (no
in-kernel cast) are faster if Q/K/V are already e4m3: VLM **185 vs 253 µs**,
diffusion **71 vs 113 µs**. Packing Q+K+V every VLM layer (~122 µs) eats that
win, so **VLM stays on bf16 pointers**. Diffusion can pack prefix K/V once
per frame (~5 ms over 360 calls, not 15).

Unscaled FP8 **residuals** NaN at layer 6 (activation ~26k vs e4m3 max 448).
Clamps stop NaNs, drift the trajectory by metres, and erase the 15 ms.

`HGA_DOT_DTYPE=bf16` forces bf16 everywhere.

## What not to use these for

- VLM **decode** (`q_len=1`) — leave SDPA/Flash.
- Vision encoder (`head_dim=72`).
- Sequences longer than 4096 tokens.
- The FA-shaped `flash_attn_func` wrapper inside Alpamayo — extra transposes.
  Use `vlm_prefill_attention` / `attach_to_alpamayo` instead.
