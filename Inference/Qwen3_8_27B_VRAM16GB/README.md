# Qwen3.8-27B on 16 GB VRAM (128K context)

Run **Qwen3.8-27B** with a **128K-token** context on a single **16 GB NVIDIA GPU**.
The recommended AccessPoint window is **128K** (131072 tokens).

Dense KV cache cannot fit that window next to a 27B Q4 model. This tree keeps
the weights on the GPU, stores the KV cache in RAM, and uses Hierarchical
Global Attention (HGA) so each token only attends a small routed working set.

No extra training. The GGUF is the public Unsloth 4-bit file.

| You need | Typical value |
|---|---|
| GPU | 16 GB NVIDIA (RTX A4000, RTX 4080, Tesla V100, …) |
| System RAM | 32 GB or more (KV + GGUF scratch) |
| Disk | ~16 GB for the GGUF, plus a llama.cpp build |
| OS | Linux, CUDA toolkit, cmake, Python 3 |

## Quick start

The intended use is an OpenAI-compatible **AccessPoint** on this GPU:
`llama-server` plus a profile gateway. Point OpenCode or Copilot Chat at it.

From this directory:

```bash
# 1. Download the ~15.3 GiB GGUF (once)
./scripts/download_qwen38.sh

# 2. Clone llama.cpp, apply HGA, build for your GPU
./scripts/setup.sh

# 3. Install, start, and smoke the AccessPoint
export HGA_API_KEY="$(openssl rand -hex 32)"
python3 ./deployment/deploy.py
```

`deploy.py` checks for ≥16 GB VRAM, writes systemd user units with **this**
tree’s absolute paths, starts the backend + gateway, and smokes
`/v1/chat/completions`. Base URL: `http://127.0.0.1:8080/v1`. The key is
stored mode 0600 in `~/.config/hga-qwen38/api.env`.

`setup.sh` detects the CUDA architecture with `nvidia-smi`. Re-run it after
pulling C++/patch changes; it is safe on an already-patched llama.cpp tree.
`deploy.py` rebuilds only when `llama-server` is missing; after `setup.sh`,
re-run `deploy.py` to restart the AccessPoint on the new binary.

### Adjust the AccessPoint

Re-running `deploy.py` rewrites units/env and restarts the services:

```bash
set -a; . ~/.config/hga-qwen38/api.env; set +a
python3 ./deployment/deploy.py                  # same host/port, restart
python3 ./deployment/deploy.py --lan            # listen on 0.0.0.0
python3 ./deployment/deploy.py --port 8080      # gateway port
python3 ./deployment/deploy.py --ctx 131072     # context length (128K, recommended)
python3 ./deployment/deploy.py --no-start       # rewrite files only
python3 ./deployment/deploy.py --skip-build     # never invoke setup.sh
```

`deploy.py` owns host/port/ctx/`HGA_API_KEY` in `~/.config/hga-qwen38/api.env`.
Other packing knobs (`HGA_SPEC`, `HGA_THREADS`, …) belong in that file or
[`scripts/env.sh`](scripts/env.sh). After editing knobs `deploy.py` does not
rewrite, restart without regenerating the env file:

```bash
./deployment/stop-local.sh
./deployment/start-local.sh
```

`stop-local.sh` is the reliable stop from a desktop/IDE shell (it pins the
user systemd bus). Details: [`deployment/README.md`](deployment/README.md).

| Model ID | Thinking | Max output | Use |
|---|---|---:|---|
| `qwen3.8-27b-hga-fast` | off | 131072 | agent loops |
| `qwen3.8-27b-hga-normal` | 512 | 131072 | default chat |
| `qwen3.8-27b-hga-deep` | 4096 | 131072 | long reasoning |

Reasoning remains separately bounded. The usable answer length is the context
space left after the prompt and reasoning tokens; explicitly smaller client
limits are still honored.

### Use it from OpenCode

The AccessPoint is OpenAI-compatible:

- Base URL: `http://127.0.0.1:8080/v1`
- Chat: `http://127.0.0.1:8080/v1/chat/completions`
- Auth: `Authorization: Bearer` plus the key in
  `~/.config/hga-qwen38/api-key` (written by `deploy.py`)

Point OpenCode at that URL by adding a **local** provider in
`~/.config/opencode/opencode.json`. This is the `hga-local` block used on this
16 GB host (it can sit next to other providers). Use an absolute path to the
key file:

```json
{
  "provider": {
    "hga-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local Qwen HGA (16 GB)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "{file:/home/<user>/.config/hga-qwen38/api-key}"
      },
      "models": {
        "qwen3.8-27b-hga-fast": {
          "name": "local Qwen3.8-27B HGA (fast)",
          "reasoning": false,
          "limit": { "context": 131072, "output": 131072 }
        },
        "qwen3.8-27b-hga-normal": {
          "name": "local Qwen3.8-27B HGA (normal reasoning)",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "limit": { "context": 131072, "output": 131072 }
        },
        "qwen3.8-27b-hga-deep": {
          "name": "local Qwen3.8-27B HGA (deep reasoning)",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "limit": { "context": 131072, "output": 131072 }
        }
      }
    }
  },
  "model": "hga-local/qwen3.8-27b-hga-fast"
}
```

A complete snippet is [`examples/opencode.json`](examples/opencode.json). Merge
the `hga-local` provider into your existing config; set `"model"` to
`hga-local/qwen3.8-27b-hga-fast` (or `…-normal` / `…-deep`). Restart OpenCode
after editing.

Smoke without OpenCode:

```bash
set -a; . ~/.config/hga-qwen38/api.env; set +a
curl -sS http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $HGA_API_KEY"
```

Copilot Chat: [`examples/chatLanguageModels.json`](examples/chatLanguageModels.json).

## Speed check

Parser checks (no GPU, no GGUF):

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
python3 tools/bench.py --self-test
```

Full model, ~2K prefill then 64 generated tokens:

```bash
source ./scripts/env.sh
python3 tools/bench.py
```

8K prefill:

```bash
python3 tools/bench256_8k.py
```

On a dual-socket Xeon + 16 GB V100 the current **768/3200** prefill path is
about **410 tok/s**. A 12-thread workstation is CPU-bound on HGA; the GPU
packing is the same.

## Quality check (LongBench-E)

Unconditional reuse of the position-zero prefill graph while visible history
grows is fast but **wrong**: LongBench-E passage retrieval dropped from 6/6
to 2/6. Production rebuilds until the historical boundary saturates, then
reuses. Confirm with the public dataset:

```bash
./scripts/download_longbench.sh
python3 tools/bench_longbench_retrieval.py \
  --data ~/data/LongBench/passage_retrieval_en_e.jsonl
```

Parser only (no GPU): `python3 tools/bench_longbench_retrieval.py --self-test`

The default six-example subset must score **6/6**.

## One-shot CLI (optional)

A single prompt without the AccessPoint. Less useful than the API for
OpenCode / Copilot:

```bash
source ./scripts/env.sh
./scripts/run_hga.sh "Summarize the idea of hierarchical attention in one paragraph."
```

## Immutable graph recipe cache

PREFILL and VERIFY still use separate fresh ggml schedulers and CUDA scratch
arenas. Before a scheduler is destroyed, this tree now saves an immutable
host-side recipe for each exact graph shape and restores the graph to that
recipe:

- scheduler-inserted cross-backend `src[]` tensors are removed;
- graph-owned scratch `buffer`, `data`, and backend `extra` pointers are reset;
- explicit HGA CPU/CUDA placement is replayed into the fresh scheduler.

The cache retains no executable scheduler object and no CUDA scratch. It is
bounded to eight compute and eight reserve recipes by default. Recipes are
matched with llama.cpp's normal graph-input compatibility checks plus the HGA
phase and `op_offload` mode, so the growing-history PREFILL graph continues to
rebuild until its visibility boundary is compatible. Saturated PREFILL shapes,
VERIFY shapes, and reserve shapes are then reused across scheduler teardown.

`HGA_GRAPH_RECIPE_CACHE=0` disables the cache for an A/B control.
`HGA_GRAPH_RECIPE_CACHE_MAX` changes the per-cache recipe limit (minimum 2).
Logs use `hga-graph: ... BUILD` and `hga-graph: ... HIT` lines.

Local RTX A4000 validation used seven consecutive growing API turns. All
PREFILL-to-DECODE-to-PREFILL switches completed, prompt caching grew from 793
to 4699 tokens, and the saturated 768-token PREFILL recipe was restored after
scheduler destruction.

## Lazy recurrent prefix cache

The API keeps the general llama.cpp context checkpoints disabled because they
split every prompt near its end and slow normal prefill. Instead, HGA records
the token text of recent prompt blocks in an eight-entry index keyed by a hash
of the first 256 tokens. When a later prompt has the same key and diverges
after a common prefix, that one request stops at the intersection and stores a
recurrent-state checkpoint there. Later sibling prompts restore that state and
reuse the resident INT8 HGA KV prefix without re-prefilling it.

When a cache-enabled completion finishes, the API also stores a recurrent
checkpoint for every token that was actually evaluated and indexes that final
prompt state. A following chat turn can therefore restore the completed
conversation immediately. The last sampled token is deliberately excluded if
it has not yet been evaluated; the continuation evaluates that token together
with the newly appended turn, preserving the logits/state boundary.

`HGA_LAZY_PREFIX_CACHE=0` disables this cache. A positive value controls both
the number of indexed prompt families and the global checkpoint limit. Logs
use `hga-prefix: INTERSECT`, `STORE`, `FINISH`, and `HIT` lines. The first
prefill of a prompt family is unchanged; state capture happens after generation
finishes or at a common-prefix intersection discovered by a later branch.
On the tested 27B model each recurrent checkpoint is about 150 MiB of host RAM,
so the default limit of eight allows roughly 1.2 GiB for saved states.

Speculative `HGA_SPEC=2` validation also covers repeated three-token target
VERIFY and MTP graphs. Custom-op userdata is owned by the immutable graph
recipe and contains copied phase/layer values, never pointers to the temporary
graph-build parameters. The MTP graph sets its HGA phase before building its
position inputs, so its V tensor is staged to the CPU before the HGA custom op.
Three consecutive cache-enabled API requests completed locally; the logs
showed both target and MTP `compute HIT` entries after PREFILL/VERIFY switches.

## What HGA does on 16 GB

```text
GPU  16 GB   Q4 weights + 8 streamed layer-pair slots + flash-attn working set
RAM          INT8 KV cache, chunk/group summaries, HGA routing
```

- 16 of 64 layers are full attention; the rest are Gated DeltaNet.
- Each query attends: 2 sink chunks + 7 local chunks + ~4% routed mid-context.
- Prefill: CPU routes every 64-token chunk, then CUDA flash-attention runs
  on one **united INT8 historical image**. Tensor shapes stay fixed (768
  direct + 3200 historical). The graph is rebuilt while the valid-history
  boundary grows, then reused after saturation.
- Decode: tiled INT8 kernel on the CPU.
- K=2 verify: CPU HGA routing packs the selected historical INT8 K/V, then
  CUDA flash attention evaluates `softmax(QKᵀ)V` together with direct K/V
  from the verify batch. `HGA_GPU_VERIFY=0` selects the former CPU-attention
  control path.
- After the prompt, six of eight exchange pairs are pinned on the GPU;
  two pairs keep streaming so leftover VRAM still fits.

Details for reviewers: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Layout

```text
cpp/                 standalone HGA core (no GGUF, no CUDA)
llama.cpp-hga/       llama.cpp glue, weight streamer, patched Qwen sources
scripts/setup.sh     clone + patch + build llama.cpp
scripts/env.sh       16 GB defaults (threads, ubatch 768, max keys 3200)
scripts/run_hga.sh   launcher used by the AccessPoint (also one-shot CLI)
scripts/apply_hga.py patches a llama.cpp checkout
tools/bench.py       2K-prefill / 64-gen speed test
tools/bench_longbench_retrieval.py  public LongBench-E quality gate
deployment/deploy.py install / restart the OpenAI AccessPoint
examples/            OpenCode and Copilot client snippets
ARCHITECTURE.md      packing, routing, and prefill/decode notes
baselines/vram16/    functional speed floors, not peak claims
```

`third_party/llama.cpp/` is created by `setup.sh` and is gitignored.

## Useful knobs

| Variable | Default | Meaning |
|---|---|---|
| `HGA_MODEL` | `~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | GGUF path |
| `HGA_CTX` | `2048` CLI / `131072` API (128K, recommended) | context length |
| `HGA_THREADS` | physical cores | HGA OpenMP team |
| `HGA_UBATCH` | `768` | prefill graph width |
| `HGA_GPU_PREFILL_MAX_KEYS` | `3200` | historical KV columns on CUDA |
| `HGA_GPU_VERIFY` | `1` | CPU routing + CUDA attention for K>1 VERIFY |
| `HGA_GPU_VERIFY_MAX_KEYS` | prefill max | maximum fixed VERIFY history width |
| `HGA_SPEC` | `2` CLI / `0` API | MTP draft tokens; API default is off for VRAM margin |
| `HGA_LOAD_MODE` | `none` | chunked 16 MiB GGUF→VRAM load |

```bash
# AccessPoint: add HGA_SPEC=2 to ~/.config/hga-qwen38/api.env, then
./deployment/stop-local.sh && ./deployment/start-local.sh

# one-shot CLI only
HGA_CTX=32768 ./scripts/run_hga.sh
```

## License

Apache 2.0. llama.cpp keeps its own MIT license in `third_party/llama.cpp`.
