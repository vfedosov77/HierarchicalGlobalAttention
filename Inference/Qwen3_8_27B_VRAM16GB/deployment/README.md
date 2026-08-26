# OpenAI-compatible AccessPoint (16 GB CUDA)

`llama-server` plus a small profile gateway. Packing matches the local
benches: UD-Q4_K_M, HGA-2 INT8 KV, physical-core HGA threads, K=2-capable
VERIFY, two paced streams, chunked GGUF load, CUDA graphs off, `--jinja`
for tool calling. `--parallel 1`: one long agent conversation.

The gateway keeps the prompt/KV cache intact and exposes profiles as
ordinary model IDs:

* `GET /health`
* `GET /v1/models`
* `POST /v1/chat/completions`
* `POST /v1/completions`

Loopback by default. Bearer auth is required.

## Deploy

```bash
export HGA_API_KEY="$(openssl rand -hex 32)"
python3 deployment/deploy.py
```

That checks for ≥16 GB VRAM, writes systemd user units with **this** tree’s
absolute paths, starts the backend + gateway, and smokes `/v1/chat/completions`.

```bash
python3 deployment/deploy.py --no-start   # install files only
python3 deployment/deploy.py --lan        # listen on the LAN
deployment/start-local.sh                 # no systemd user bus
deployment/stop-local.sh
```

Recommended context is **128K** (131072). KV/HGA stay on CPU. Prefill ubatch is 768 with a
3200-key historical cap; the graph rebuilds until valid history saturates.
Speculative MTP defaults to **K=3** draft tokens (`HGA_SPEC=2`, verify width 4).
Set `HGA_SPEC=2` if leftover VERIFY pin OOMs.

| Model ID | Thinking | Max output | Use |
|---|---|---:|---|
| `qwen3.8-27b-hga-fast` | off | 131072 | agent loops |
| `qwen3.8-27b-hga-normal` | 512 | 131072 | default chat |
| `qwen3.8-27b-hga-deep` | 4096 | 131072 | long reasoning |

`qwen3.8-27b-hga` maps to `normal`. Tool requests are forced to
`parallel_tool_calls=false`. `cache_prompt=true` is injected so a growing
conversation reuses its KV prefix.

Reasoning remains separately bounded. The usable answer length is the context
space left after the prompt and reasoning tokens; explicitly smaller client
limits are still honored.

Sibling agent prompts use the lazy recurrent prefix cache. The first prompt is
indexed by its first 256 tokens without saving state. The first suffix branch
creates one checkpoint at the discovered common-prefix boundary; later sibling
branches restore it. Set `HGA_LAZY_PREFIX_CACHE=0` for an A/B control.

Each completed cache-enabled request also stores a checkpoint at its last
evaluated token. Consequently, the next turn of the same growing conversation
restores the finished-chat state and prefills only the newly appended tokens.

Client snippets: [`../examples/`](../examples/).
