# Qwen3.8-27B HGA optimization details

This reference captures the investigation completed on 2026-08-29 for the 16 GB Qwen3.8-27B HGA deployment. Read it when diagnosing or changing this model's performance path.

## Contents

1. Repository map and model facts
2. Known-good benchmark protocol
3. What the investigation proved
4. Split-FFN implementation and correctness traps
5. Deployment and benchmark bugs fixed
6. Matched results and interpretation
7. Current runtime recommendation
8. Validation and future experiments

## Repository map and model facts

Run commands from `Inference/Qwen3_8_27B_VRAM16GB` unless noted otherwise.

Important files:

| Purpose | Path |
|---|---|
| Deployment entry point | `deployment/deploy.py` |
| Runtime environment | `deployment/run-api.sh` |
| API 8K benchmark | `tools/bench_8k_api.py` |
| Offline benchmark support | `tools/bench_8k.py`, `tools/bench.py` |
| Whole-layer exchange | `llama.cpp-hga/hga-weight-swap.cpp` |
| Split-FFN scheduler | `llama.cpp-hga/hga-split-ffn.cpp` |
| Qwen graph construction | `llama.cpp-hga/patched/qwen35.cpp` |
| Recurrent rollback fix | `llama.cpp-hga/patched/llama-memory-recurrent.cpp` |
| Durable llama.cpp patching | `scripts/apply_hga.py` |
| Split tests | `tests/test_split_ffn.py` |
| Deploy tests | `tests/test_deploy_threads.py` |
| Benchmark tests | `tests/test_bench.py` |

Model and schedule facts:

- Main model: Qwen3.8-27B, 64 layers, hidden size 5120, dense SwiGLU intermediate size 17408.
- Quantization under test: UD-Q4_K_M; tensor types can vary, so derive byte sizes from GGML traits rather than assuming a four-bit ratio.
- Whole-layer VERIFY exchange pairs: layers `16 <-> 48` and `24 <-> 56`.
- Default split width: 1024 intermediate channels, exactly 17 tiles because `17408 / 1024 = 17`.
- Split FFN applies only to DECODE/VERIFY. PREFILL and the whole-layer fallback remain unchanged.
- Speculative `HGA_SPEC=K` drafts K tokens and verifies with width K+1.

## Known-good benchmark protocol

### Establish the configuration

Pass performance settings to `deploy.py`; it writes them to the effective API environment. Example baseline:

```bash
HGA_THREADS=24 \
HGA_SPEC=2 \
HGA_SPLIT_FFN=0 \
python3 deployment/deploy.py --skip-calibrate --skip-smoke
```

Candidate configuration:

```bash
HGA_THREADS=24 \
HGA_SPEC=3 \
HGA_SPLIT_FFN=0 \
HGA_SPLIT_FFN_TILE_CHANNELS=1024 \
python3 deployment/deploy.py --skip-calibrate --skip-smoke
```

`deployment/run-api.sh` sets `GGML_CUDA_DISABLE_GRAPHS=1`. CUDA graph capture is intentionally disabled because independent H2D copy streams and event coordination are incompatible with the intended capture path. This does not mean the GGML compute graph is running from RAM.

After deployment, verify both health and the actual process configuration:

```bash
curl -fsS http://127.0.0.1:8081/health
tr '\0' '\n' </proc/$(pgrep -n llama-server)/environ \
  | rg '^(HGA_SPEC|HGA_SPLIT_FFN|HGA_SPLIT_FFN_TILE_CHANNELS|HGA_THREADS|GGML_CUDA_DISABLE_GRAPHS)='
```

Use the specific process PID if more than one `llama-server` exists. Inspect `~/.config/hga-qwen38/api.env`, but do not treat it as proof that a stale live process restarted.

### Run a matched 8K test

Use the same nonce for every configuration and restart the server between candidates:

```bash
python3 tools/bench_8k_api.py \
  --url http://127.0.0.1:8081 \
  --nonce hga-ab-20260829 \
  --n-predict 64 \
  --json /tmp/hga-candidate.json
```

The fixed nonce makes prompt contents identical while the harness enforces an uncached measurement. Avoid `--stable-prompt` for A/B runs: repeated stable prompts may hit the lazy prefix cache. The benchmark should report about 8010 prompt tokens and 64 generated tokens; preserve the returned timing fields and speculative statistics.

For each run record:

- prefill milliseconds and tokens/s;
- decode milliseconds and tokens/s;
- generated-token count and termination status;
- accepted and drafted speculative tokens;
- mean accepted draft length;
- target graph batch count and mean target time when logged;
- graph BUILD/HIT counts and total build time;
- residency census, pin-check output, CUDA allocation failures, and GPU memory use.

### Avoid misleading comparisons

- A random nonce per run prevents cache hits but changes tokenization and generation. Use a fixed nonce for a controlled A/B.
- Restarting only the gateway is insufficient; restart `llama-server` to reset prefix and graph state.
- A candidate with lower speculative acceptance can appear slower for reasons unrelated to its kernels. Compare target-batch counts.
- `copy_ms` and `wait_ms` currently measure host-side enqueue/event insertion, not elapsed GPU DMA. Do not derive PCIe GiB/s from them.
- Long 8K prefill dominates total wall time. Report decode improvement separately from total request improvement.

## What the investigation proved

### No unintended layer was pushed to RAM

At initialization the target tensor census reported:

```text
resident GPU: 607
resident host: 0
exchange host initially: 224
host-ok: 35
hga-pin: ok no expected-GPU tensor on CUDA_Host/CPU
```

The 224 initial exchange tensors are the deliberately streamable images, not a surprise offload. After the split path runs, 12 FFN tensors may remain on host: `ffn_up`, `ffn_gate`, and `ffn_down` for layers 16, 24, 48, and 56. Those are exactly the intended four split-streamed layers. All non-exchange tensors expected on GPU remain there.

Whole-layer streaming similarly keeps the inactive side of each pair in pinned host memory. Therefore, “some tensors are on host” alone is not a bug. A bug exists only when the census reports an expected-GPU tensor on CUDA host or CPU memory, a pin failure, or a CUDA allocation fallback.

At idle, the final whole-layer K=3 deployment used approximately 15.68 GiB and left about 0.37 GiB free. During decode CUDA may report nearly zero transient free memory without an OOM. Look for actual allocation errors or fallback logs rather than interpreting low free memory as proof of offload.

### CPU scratch is expected and small

For K=2, target graph allocation was approximately:

```text
GPU compute buffer: 49.3 MiB
CPU compute scratch: 2.6 MiB
```

K=3 was approximately 52.5 MiB GPU and 2.7 MiB CPU. The CPU scratch serves HGA route/norm work and is not a model layer. Graph metadata living on CPU is also normal; inspect tensor backends and compute buffers to determine actual offload.

### Extra graph rebuilds were not the bottleneck

Matched whole and split runs each showed:

```text
compute graphs: 16 BUILD, 10 HIT
reserve graphs: 21 BUILD, 87 HIT
```

Measured graph build time:

| Path | Compute build | Reserve build | Total |
|---|---:|---:|---:|
| Whole layer | 33.621 ms | 24.846 ms | 58.467 ms |
| Split FFN | 33.012 ms | 24.334 ms | 57.346 ms |

About 58 ms over a roughly 47-second request is near 0.1% of wall time. Steady decode reused the K=2 three-token and K=3 four-token target/MTP graph shapes. Graph reconstruction was therefore neither a per-token regression nor a material bottleneck in this test.

CUDA graphs being disabled is distinct from GGML graph caching. CUDA capture is off, but the cached GGML graph shapes are reused. Do not describe this as “the graph was pushed into RAM.”

### Corrected split FFN is slower on this host

The split path reduces approximate H2D bytes per streamed pass from 875 MiB to 602.3 MiB, about 31%. It nevertheless fragments a dense FFN into 17 up/gate/down graph groups and adds scheduling, event, launch, and accumulator overhead. At matched speculative acceptance, this cost exceeded the transfer saving.

This conclusion is hardware- and implementation-specific. Keep the path available for slower PCIe systems or future fused kernels, but do not enable it by default without a new end-to-end win.

## Split-FFN implementation and correctness traps

### Intended memory layout

For each exchange pair:

- Keep both layers' non-FFN core tensors resident on GPU.
- Keep quantized FFN tile images in pinned host memory.
- Reuse a bounded GPU tile bank within the existing whole-layer pair budget.
- Pack each tile as aligned `ffn_up`, `ffn_gate`, then repacked `ffn_down` fragments.
- Derive sizes and block alignment from the actual GGML tensor types.
- Allocate one contiguous `cudaMallocHost` image per layer rather than many small pinned allocations.
- Give each exchange pair its own CUDA copy stream and ready/free events.

The 1024-channel width must remain exact for the current implementation. An experimental 2048-channel tail created unstable graph shapes and initially broke acceptance. Padding restored correctness but reduced decode to 5.62 tokens/s, so the extension was removed.

### Tile lifetime bug that was fixed

The original implementation inserted `ggml_scale(acc, 1.0)` as a final-tile marker. CUDA fusion could eliminate or move this identity node, letting the scheduler overwrite the final tile slot while the graph still consumed it.

The corrected scheduler marks the final tile pending and releases it at the next stable `norm-(layer+1)` graph callback. The fuse-prone identity scale markers were removed from both `patched/qwen35.cpp` and the durable `scripts/apply_hga.py` template. Any later graph refactor must preserve this lifetime boundary or replace it with an equally reliable event recorded after the last down-projection/accumulation use.

### Stream ownership correction

The task design called for pair-local copy streams, but the first implementation used one global CUDA copy stream. The scheduler now owns one copy stream per pair. This is structurally correct and prevents unnecessary cross-pair serialization, although the matched benchmark still did not beat whole-layer exchange.

### Correctness signals

Treat these as failures even when a request completes:

- speculative acceptance collapses to zero;
- target-batch count rises unexpectedly;
- output differs materially for the fixed prompt;
- the server aborts during short final-batch cleanup;
- a tile slot is reused before the last down projection and accumulation completes;
- a quantized tile boundary is not block aligned;
- PREFILL takes the split path;
- the split allocator exceeds the former whole-layer pair budget;
- a split-plan rejection fails to fall back to whole-layer exchange.

## Deployment and benchmark bugs fixed

### Split mode was never active in the first comparison

The persisted `~/.config/hga-qwen38/api.env` contained `HGA_SPLIT_FFN=0`. Consequently, early “before” and “after” measurements exercised the same whole-layer path. Always inspect both process environment and startup plan logs for the selected scheduler.

### Local fallback did not restart an existing server

When user systemd was unavailable, `deploy.py` previously invoked the local start helper without first stopping the old PID. Rewriting `api.env` therefore did not change the live configuration. The fallback now performs stop followed by start. Future edits must retain this behavior.

### HGA source changes did not trigger a rebuild

`deploy.py` previously skipped compilation whenever `llama-server` already existed. It now compares the binary mtime with HGA C/C++ headers/sources and `scripts/apply_hga.py`, rebuilding when an input is newer. Do not use `--skip-build` after changing these sources.

### Speculative cleanup could abort

Padded K+1 cleanup followed by speculative rejection can request two recurrent-memory rollbacks before the next graph. The old code replaced the first rollback snapshot with the second and could abort with:

```text
common/common.cpp:1628 failed to remove sequence
```

`patched/llama-memory-recurrent.cpp` now composes rollback indices, bounded by `n_rs_seq`; the same change lives in `scripts/apply_hga.py` so rebuilding third-party llama.cpp does not discard it.

### The API benchmark could accidentally cache-hit

`tools/bench_8k_api.py` now supports `--nonce`. The chosen nonce is written to JSON, allowing exact reproduction. `--stable-prompt` remains available for cache-specific testing but is inappropriate for uncached performance A/B work.

## Matched results and interpretation

All principal runs used nonce `hga-ab-20260829`, an approximately 8010-token prompt, 64 generated tokens, and a server restart between candidates.

| Configuration | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | Speculative detail |
|---|---:|---:|---:|---:|---|
| Whole layer, K=2 baseline | 37977.73 | 210.91 | 9399.76 | 6.70 | accepted 40/45 |
| Split 1024, K=2, corrected global stream | 37714.62 | 212.38 | 9751.94 | 6.46 | accepted 41/43 |
| Split 1024, K=2, pair-local streams | 37640.93 | 212.80 | 10142.27 | 6.21 | accepted 40/45; 23 target batches |
| Whole layer, K=3 | 37768.20 | 212.08 | 7908.36 | **7.97** | accepted 45/52; mean draft 3.50; 18 target batches |

Relative to whole-layer K=2:

- Whole-layer K=3 improved decode throughput by about 19.0%.
- Prefill changed by about 0.6%, which is effectively flat without repeated evidence.
- Total prefill-plus-decode time improved from 47.38 seconds to 45.68 seconds, about 3.6%, because long prefill dominates the request.
- Corrected pair-local split FFN was about 7.3% slower in decode despite reducing transferred bytes.

The winning K=3 run used verify width four and did not show the cleanup failure. Retest if the model, quantization, driver, GPU, context policy, or prompt distribution changes.

## Current runtime recommendation

Recommended environment for the measured host:

```text
HGA_SPEC=3
HGA_SPLIT_FFN=0
HGA_SPLIT_FFN_TILE_CHANNELS=1024
HGA_THREADS=24
HGA_F16_TRANSPORT=1
GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto
GGML_CUDA_DISABLE_GRAPHS=1
```

`HGA_SPLIT_FFN_TILE_CHANNELS` is retained even while split mode is off so enabling the experiment has a known default. If K=3 causes a VERIFY pin OOM on another 16 GB card, compare K=2 and K=0 rather than allowing an unexplained CPU fallback.

Do not claim PREFILL was optimized by the K=3 change; it was not. A future prefill effort should profile the fixed-shape GPU prefill, route/norm CPU work, host bandwidth, and 768-token chunk scheduling independently from split FFN.

## FP16 activation transport experiment

### What is actually FP32

The model file is UD-Q4_K_M, so streamed weights are quantized rather than
FP32. The large avoidable FP32 payload was the HGA activation boundary: Q,
K-rope, V, and K-raw traveled from CUDA to the CPU for routing, and the HGA
attention result traveled back to CUDA for the output projection.

A blanket F16 ggml graph is not valid in this llama.cpp revision. Quantized
MMQ/MMVQ destinations, RMS norm, recurrent kernels, softmax, and several CUDA
copy/flash-attention contracts use F32 tensors. `ggml_mul_mat()` also exposes an
F32 result even when CUDA uses F16 tensor-core arithmetic internally. Treat
"FP16 model" here as maximal safe FP16 compute and transport, not a promise
that every CUDA instruction or accumulator is half precision.

### Implemented boundary

`HGA_F16_TRANSPORT=1` is opt-in and changes the large PREFILL boundary only:

- cast Q, K-rope, V, and K-raw from F32 to F16 on CUDA before D2H;
- expand the F16 rows to F32 on the CPU before existing HGA routines;
- leave route IDs and selection metadata integer and keep the persistent KV
  cache INT8 with F32 scales;
- convert the CPU attention result to F16 before H2D, then restore F32 on CUDA
  so the optimized quantized output projection remains selected.

Decode/VERIFY stays on the existing F32 boundary. Its activation copies are
only a few KiB, while forcing the F16 CPU-custom-op edge into reserve/VERIFY
graphs produced an unassigned-backend scheduler assertion. Restricting the
wire to PREFILL avoids that invalid graph and targets the material traffic.

Use `ggml_cpu_fp16_to_fp32()` and `ggml_cpu_fp32_to_fp16()` for the host-side
boundary. The generic ggml row helpers are scalar reference conversions. On
the measured Xeon E5-2695 v4, the CPU backend is compiled with AVX2/F16C; using
its vectorized converters changed this experiment from a regression to a win.

Deployment persists both switches:

```text
HGA_F16_TRANSPORT=1
GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto
```

`GGML_CUDA_CUBLAS_COMPUTE_TYPE=fp16` forces only cuBLAS fallbacks. In `auto`,
quantized cuBLAS paths already select F16 on this GPU, and custom quantized
kernels are unaffected.

### Matched 8K/64 measurements

These runs used nonce `hga-fp16-20260829`, 8013 prompt tokens, 64 generated
tokens, K=3, split FFN off, 24 CPU threads, and a restart between variants.

| Variant | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | Draft accepted/generated |
|---|---:|---:|---:|---:|---:|
| F32 wire, cuBLAS auto | 37242.37 | 215.16 | 7899.89 | 7.97 | 45/54 |
| F32 wire, cuBLAS forced FP16 | 37379.87 | 214.37 | 7961.13 | 7.91 | 45/54 |
| F16 wire with scalar host conversion, forced FP16 | 40655.33 | 197.10 | 7403.05 | 8.51 | 46/51 |
| F16 wire with AVX2/F16C conversion, cuBLAS auto | **35402.43** | **226.34** | 7468.40 | 8.44 | 46/51 |
| F16 wire with AVX2/F16C conversion, forced FP16 | 35619.10 | 224.96 | 7464.77 | 8.44 | 46/51 |

The selected F16/auto result improved matched PREFILL throughput by 5.2% and
reduced total inference time from 45.14 to 42.87 seconds. Do not attribute the
apparent decode improvement to faster decode kernels: FP16 prefill changed the
draft trajectory from 45/54 to 46/51, reducing target VERIFY batches from 18
to 17. Forced cuBLAS FP16 was neutral/slightly slower and is not recommended.

The runtime logs show 768-token wire payloads of 9216 KiB for Q and 1536 KiB
for each K/V/K-raw tensor, half their F32 sizes. The existing profiler reports
432 MiB because it counts logical cast/copy node bytes, so use the named D2H
wire logs when auditing physical payload.

### Residency and graph findings

The startup census rejects the extra-layer-in-RAM hypothesis on the measured
run: 607/607 required resident tensors were on CUDA (11581 MiB), all 224 host
tensors were the intentional exchange set, and no expected-GPU tensor was on
CUDA host or CPU memory. During PREFILL, all 224 exchange tensors were staged
on CUDA.

The F16 wire adds explicit cast nodes, increasing the 768-token target graph
from 4410 to 4474 nodes. Builds occur for expected context/phase/shape recipes;
the final run reported 21 graph reuses. There is no evidence that a compute
graph was pushed into RAM, and graph construction time is milliseconds rather
than the multi-second routing/packing cost.

## Validation and future experiments

### Focused validation

After editing, run:

```bash
python3 -m unittest -v \
  tests/test_f16_transport.py \
  tests/test_split_ffn.py \
  tests/test_deploy_threads.py \
  tests/test_bench.py
python3 tools/bench_8k_api.py --self-test
python3 -m py_compile deployment/deploy.py tools/bench_8k_api.py
git diff --check
```

The completed 2026-08-29 change set passed 36 focused unit tests, the API benchmark self-test, standalone HGA tests, syntax checks, and `git diff --check`.

After rebuilding, confirm the durable patch output contains the same changes as `llama.cpp-hga/patched/*`; otherwise a later setup run can silently restore an old bug.

### Promising next work

Measure before implementing. Likely directions are:

1. Fuse multiple split FFN tile operations or use a custom kernel so reduced H2D traffic does not require 17 fragmented graph groups.
2. Pipeline copies with GPU work using device-side timestamps or CUDA events; current host enqueue timing cannot identify true DMA overlap.
3. Explore larger fixed-shape tiles only if tail handling preserves quantized alignment, graph reuse, and speculative acceptance.
4. Profile prefill separately, especially route/norm CPU time and fixed-shape chunk boundaries.
5. Sweep speculative K around 3 using the same fixed prompt and enough runs to capture acceptance variance; higher K is useful only if fewer target passes outweigh larger verification work and VRAM.

Reject an optimization that merely lowers one internal counter while reducing end-to-end decode throughput, correctness, or service stability.
