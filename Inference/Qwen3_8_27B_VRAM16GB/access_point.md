# HGA AccessPoint (16 GB CUDA)

OpenAI-compatible API for OpenCode and GitHub Copilot Chat on this machine.

- Base URL: `http://127.0.0.1:8080/v1`
- Chat completions: `http://127.0.0.1:8080/v1/chat/completions`
- Models: `qwen3.8-27b-hga-fast`, `qwen3.8-27b-hga-normal`, `qwen3.8-27b-hga-deep`
- Auth: `Authorization: Bearer $HGA_API_KEY` (file `/home/q548040/.config/hga-qwen38/api.env`, mode 0600)
- GPU: 16376 MiB  |  HGA threads: 12  |  context: 262144

Do not commit the API key. Source it with:

```bash
set -a; . ~/.config/hga-qwen38/api.env; set +a
```

## Start / stop

```bash
# systemd user services (if a user session bus exists):
systemctl --user start hga-qwen38.service hga-qwen38-gateway.service
systemctl --user stop  hga-qwen38.service hga-qwen38-gateway.service

# portable foreground/background helper:
/home/q548040/projects/HierarchicalGlobalAttention/Inference/Qwen3_8_27B_VRAM16GB/deployment/start-local.sh
/home/q548040/projects/HierarchicalGlobalAttention/Inference/Qwen3_8_27B_VRAM16GB/deployment/stop-local.sh
```

## OpenCode

Merge `examples/opencode.json` into `~/.config/opencode/opencode.json`.
Provider `hga` talks to `http://127.0.0.1:8080/v1`. Default model: `hga/qwen3.8-27b-hga-fast`.

## GitHub Copilot (VS Code Chat)

1. Command Palette → **Chat: Manage Language Models** → **Add Models** → **Custom Endpoint**.
2. API type: **Chat Completions**.
3. Paste `/home/q548040/projects/HierarchicalGlobalAttention/Inference/Qwen3_8_27B_VRAM16GB/deployment/clients/chatLanguageModels.json` (replace the API key prompt).
4. Enable **toolCalling** models for Copilot agent mode.

Copilot CLI:

```bash
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8080/v1
export COPILOT_PROVIDER_API_KEY="$HGA_API_KEY"
export COPILOT_MODEL=qwen3.8-27b-hga-fast
```

Inline Copilot completions still use GitHub-hosted models; this endpoint is for Chat / agent.

## Smoke

```bash
HGA_API_KEY="$HGA_API_KEY" python3 /home/q548040/projects/HierarchicalGlobalAttention/Inference/Qwen3_8_27B_VRAM16GB/deployment/smoke.py --url http://127.0.0.1:8080
```
