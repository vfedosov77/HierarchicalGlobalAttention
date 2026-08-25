# How this 16 GB build works

This note is for people reviewing or modifying the tree. Day-to-day usage is
in the [README](README.md).

## Why HGA

Qwen3.8-27B Q4_K_M is about 15.3 GiB. A 16 GB card can hold the weights and
almost nothing else. Dense attention KV for 256K tokens does not fit.

HGA does not change `q_proj` / `k_proj` / `v_proj` / `o_proj`. It changes
**which past tokens** those projections attend:

1. Context is split into 64-token chunks, each with 16-token groups.
2. Chunk and group **summaries** are pooled from the model's own keys.
3. The first 2 chunks (sink) and last 7 chunks (local) are always attended.
4. From the middle, routing keeps about 8% of chunks and opens about 4% of
   groups (floors: 3 chunks / 6 groups; caps: 20 chunks / 32 groups).
5. Softmax always uses the original token K/V of the selected spans.

Persistent KV lives in host RAM as INT8 with a per-vector scale. Routing and
softmax stay F32.

Qwen3.8-27B has 64 layers; only 16 are full attention
(`3, 7, 11, …, 63`). The other 48 are Gated DeltaNet. HGA only replaces
those 16 attention blocks.

## 16 GB weight packing

Almost every layer stays CUDA-resident, including `lm_head`. Eight layer
pairs are **exchange slots**:

```text
0↔32  4↔36  8↔40  12↔44  16↔48  20↔52  24↔56  28↔60
```

During prefill all eight stream. After the prompt, six leftover pairs are
pinned on the GPU and only `16↔48` and `28↔60` keep streaming (two paced
H2D streams). CUDA graphs stay off because independent copy streams are
illegal inside a capture.

`--load-mode none` plus the 16 MiB chunked loader in `apply_hga.py` keeps
host RAM from staging whole tensors (lm_head is ~1 GiB).

## Prefill (the ~20% win)

Each 64-token query chunk still routes independently, including every GQA
query head. For the physical 512-token ubatch the selected groups are
**united once per KV head** into one historical INT8 image:

```text
K/V  [n_kv_heads, history_capacity, head_dim]   INT8 + F32 scales
```

`history_capacity` is independent of the current prefix (two routing
envelopes, then `HGA_GPU_PREFILL_MAX_KEYS=3200`). Direct K/V of the current
ubatch stays on CUDA and is padded to 768 so a short final chunk does not
allocate a new compute buffer. A four-byte `history_valid` input hides
columns past the current prefix.

Tensor shapes stay fixed, but **reusing the graph while that valid-history
boundary grows changes answers**. llama.cpp therefore rebuilds at ubatch
starts `0, 768, …` until the historical image is saturated, then reuses.
The public LongBench-E passage-retrieval gate (`tools/bench_longbench_retrieval.py`)
is 6/6 with this policy and 2/6 if the position-zero graph is reused
unconditionally.

CUDA flash-attention then runs on `history + padded direct` with one shared
causal mask. Unused INT8 payload is not cleared; a zero scale makes it
dequantize to zero.

## Decode

Width-3 verify (K=2 MTP proposals + the target token) uses the tiled INT8
kernel: 4 KV heads × as many key tiles as threads allow (max 8). Each
worker dequantizes a K/V vector once and reuses it across the six GQA heads
and up to three tokens. Partial online-softmax states merge after the
parallel pass.

llama.cpp's outer batch pool is one thread (`HGA_THREADS_BATCH=1`) so it
does not spin while HGA's OpenMP team runs.

VERIFY weight uploads are **paced**: each A/B image is split into segments
submitted at post-HGA boundaries so a 200+ MiB copy cannot stall the tiny
activation H2D.

The API defaults `HGA_SPEC=0` so leftover pin has VRAM margin on cards that
also drive a display. One-shot benches use `HGA_SPEC=2`.

## Files that implement this

| File | Role |
|---|---|
| `cpp/src/hga.cpp` | routing, INT8 cache, united prefill image, tiled decode |
| `cpp/include/hga/profile.h` | Qwen3.8-27B shape + 16 full-attn layers + 8 pairs |
| `llama.cpp-hga/llama-hga.cpp` | ggml custom ops, graph reuse, CUDA flash-attn |
| `llama.cpp-hga/hga-weight-swap.cpp` | exchange slots, leftover pin, paced H2D |
| `scripts/apply_hga.py` | patches llama.cpp (VMM shrink, chunked load, Qwen hooks) |
| `scripts/run_hga.sh` | placement flags (`-ot`, `--fit`, `--no-kv-offload`) |
