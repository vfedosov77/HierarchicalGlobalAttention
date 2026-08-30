---
name: qwen38-hga-optimization
description: Diagnose, benchmark, and optimize the repository's HGA Qwen3.8-27B inference path on 16 GB NVIDIA GPUs, especially VRAM residency, graph reuse, speculative decoding, and split-FFN streaming. Use for performance regressions or deployment tuning in Inference/Qwen3_8_27B_VRAM16GB; do not use for unrelated models or generic llama.cpp tuning.
---

# Qwen3.8 HGA optimization

Optimize the live Qwen3.8-27B HGA deployment using matched measurements, while preserving model correctness, VRAM limits, and the user's existing worktree changes.

## Required context

Before diagnosing or changing this inference path, read [references/optimization-details.md](references/optimization-details.md). It records the model-specific memory layout, benchmark method, fixed bugs, measured alternatives, current recommendation, and relevant source locations.

Work from `Inference/Qwen3_8_27B_VRAM16GB`. Treat the recorded numbers as a 2026-08-29 calibration for an RTX 5060 Ti-class 16 GB host, not as universal performance guarantees.

## Workflow

1. Inspect the current diff and running process before editing. Existing modifications may belong to the user.
2. Record a matched baseline with `deployment/deploy.py` and `tools/bench_8k_api.py`. Use the same fixed `--nonce`, prompt size, generation length, model, and server restart policy for every A/B candidate.
3. Verify the effective process environment and startup log. Do not infer the running configuration merely from `api.env`.
4. Distinguish intentional host-resident exchange weights from an unintended CPU fallback. Use the startup tensor census and pin check described in the reference.
5. Compare graph BUILD/HIT counts and total build time before blaming graph reconstruction. CPU graph metadata is not evidence that compute was offloaded.
6. When changing speculative depth, compare accepted draft tokens, target-batch count, output correctness, and decode throughput together.
7. When changing split FFN, validate tile lifetime, quantization alignment, host pinning, stream/event ownership, numerical output, and the whole-layer fallback before judging speed.
8. Rebuild, redeploy, wait for `/health`, run the matched benchmark, then run the focused tests. Leave the fastest verified safe configuration deployed unless the user requests otherwise.

## Benchmark invariants

- Prefer `--nonce <fixed-value>` for deterministic uncached A/B prompts. `--stable-prompt` can exercise the lazy prefix cache and invalidate a comparison.
- Restart the model between candidate configurations so weights, prefix state, and graph caches begin comparably.
- Do not compare runs with materially different speculative acceptance or target-batch counts as if they measured the same compute schedule.
- Separate prefill and decode. Split FFN is a DECODE/VERIFY optimization and should not be expected to improve PREFILL.
- Preserve raw JSON results and the corresponding server log. Report exact tokens/s and wall times, not only percentages.
- A speed change within ordinary run-to-run noise needs repeated trials before becoming a default.

## Current recommendation

The best matched configuration measured on 2026-08-29 is whole-layer exchange with `HGA_SPEC=3`, `HGA_SPLIT_FFN=0`, `HGA_F16_TRANSPORT=1`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto`, 24 CPU threads, and CUDA graphs disabled. The FP16 activation wire produced 226.34 prefill tokens/s in the fixed 8K/64-token test, versus 215.16 without it. Corrected 1024-channel split FFN reduced H2D bytes but was slower because it fragmented each FFN into 17 graph groups.

Keep split FFN opt-in. Keep FP16 transport limited to the large PREFILL HGA boundary: decode transfers are tiny, and making the full ggml graph FP16 violates CUDA kernel type contracts. Read the FP16 section in the reference before extending this path.

## Completion criteria

An optimization task is complete only when:

- the live process uses the intended configuration;
- `/health` succeeds;
- no expected-GPU tensor unexpectedly resides on CUDA host or CPU memory;
- graph-build evidence is interpreted quantitatively;
- the matched benchmark improves the requested metric without correctness failures;
- focused unit tests, benchmark self-test, syntax checks, and `git diff --check` pass; and
- the final report states what is deployed, measured tradeoffs, and any remaining bottleneck.
