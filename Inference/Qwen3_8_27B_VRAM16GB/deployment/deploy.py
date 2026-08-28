#!/usr/bin/env python3
"""Install the HGA OpenAI-compatible AccessPoint on a 16 GB CUDA GPU host.

Default is local install on this machine (OpenCode / GitHub Copilot).
Use --remote only to push to another host over SSH.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path.home() / ".config" / "hga-qwen38"
USER_SYSTEMD = Path.home() / ".config" / "systemd" / "user"
MIN_VRAM_MIB = 15000


def run(command: list[str], **kw) -> subprocess.CompletedProcess:
    print("==> " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kw)


def gpu_memory_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    sizes = []
    for line in (out.stdout or "").splitlines():
        try:
            sizes.append(int(float(line.strip())))
        except ValueError:
            continue
    return max(sizes) if sizes else 0


def require_16gb_gpu() -> int:
    mib = gpu_memory_mib()
    if mib < MIN_VRAM_MIB:
        raise SystemExit(
            f"need a CUDA GPU with at least ~16 GiB; nvidia-smi reports {mib} MiB"
        )
    return mib


def detect_threads() -> int:
    env = os.environ.get("HGA_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 12)


def python_bin() -> str:
    return str(Path(sys.executable).resolve())


def llama_cpp_dir() -> Path:
    env = os.environ.get("HGA_LLAMA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "third_party" / "llama.cpp"


def llama_server_bin() -> Path:
    return llama_cpp_dir() / "build" / "bin" / "llama-server"


def gguf_path() -> Path:
    env = os.environ.get("HGA_MODEL")
    if env:
        return Path(env).expanduser()
    stamp = ROOT / "third_party" / "gguf.path"
    if stamp.is_file():
        text = stamp.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return Path.home() / "models" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q4_K_M.gguf"


def ensure_gguf() -> None:
    """Resolve weights via setup.sh (local lookup, download only if missing)."""
    env = os.environ.copy()
    env["HGA_SETUP_ONLY"] = "gguf"
    print("==> resolving GGUF via scripts/setup.sh (skip download if already on this host)", flush=True)
    run(["bash", str(ROOT / "scripts" / "setup.sh")], env=env)
    stamp = ROOT / "third_party" / "gguf.path"
    if stamp.is_file():
        found = stamp.read_text(encoding="utf-8").strip()
        if found:
            os.environ["HGA_MODEL"] = found
    path = gguf_path()
    try:
        ok = path.is_file() and path.stat().st_size > 1_000_000_000
    except OSError:
        ok = False
    if not ok:
        raise SystemExit(
            f"missing GGUF; put Qwen3.8-27B-UD-Q4_K_M.gguf in the current directory, "
            f"{ROOT}, or ~/models/ (or set HGA_MODEL)"
        )


def ensure_llama_server(skip_build: bool) -> None:
    """Run scripts/setup.sh when llama-server is not built yet.

    deploy.py is the user-facing entry point; setup.sh is the clone/patch/build
    helper it invokes. CUDA arch is detected inside setup.sh from nvidia-smi.
    """
    server = llama_server_bin()
    if server.is_file():
        return
    if skip_build:
        raise SystemExit(f"missing {server}; run scripts/setup.sh")
    print(
        "==> llama-server not found; running scripts/setup.sh "
        "(clone llama.cpp, apply HGA, build for this machine's GPU(s))",
        flush=True,
    )
    run(["bash", str(ROOT / "scripts" / "setup.sh")])
    if not server.is_file():
        raise SystemExit(f"scripts/setup.sh finished but {server} is still missing")


def render_backend_unit(root: Path) -> str:
    script = root / "deployment" / "run-api.sh"
    log = "%h/.config/hga-qwen38/server.log"
    return f"""[Unit]
Description=Qwen3.8-27B HGA llama-server backend (16 GB)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/hga-qwen38/api.env
WorkingDirectory={root}
ExecStart={script}
Restart=on-failure
RestartSec=10
TimeoutStartSec=20min
TimeoutStopSec=90s
KillSignal=SIGTERM
LimitNOFILE=65536
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"""


def render_gateway_unit(root: Path, host: str = "127.0.0.1", port: int = 8080) -> str:
    script = root / "deployment" / "api_gateway.py"
    log = "%h/.config/hga-qwen38/gateway.log"
    return f"""[Unit]
Description=HGA Qwen OpenAI API profile gateway
After=hga-qwen38.service
Wants=hga-qwen38.service

[Service]
Type=simple
EnvironmentFile=%h/.config/hga-qwen38/api.env
WorkingDirectory={root}
ExecStart={python_bin()} {script} --host {host} --port {port}
Restart=on-failure
RestartSec=3
TimeoutStopSec=30s
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"""


def write_api_env(args: argparse.Namespace, key: str, threads: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "api.env"
    backend = f"http://127.0.0.1:{args.backend_port}"
    lines = [
        f"HGA_API_KEY={key}",
        f"HGA_API_HOST={args.host_address}",
        f"HGA_API_PORT={args.port}",
        f"HGA_BACKEND_PORT={args.backend_port}",
        f"HGA_BACKEND_URL={backend}",
        f"HGA_ROOT={ROOT}",
        f"HGA_THREADS={threads}",
        f"HGA_CTX={args.ctx}",
        f"HGA_GPU_PREFILL={1 if args.hga_gpu_prefill else 0}",
        "HGA_GPU_PREFILL_MIN_KEYS=1552",
        "HGA_GPU_PREFILL_MAX_KEYS=2560",
        f"HGA_SPEC={os.environ.get('HGA_SPEC', '2')}",
        f"HGA_MODEL={os.environ.get('HGA_MODEL', str(Path.home() / 'models' / 'Qwen3.8-27B-GGUF' / 'Qwen3.8-27B-UD-Q4_K_M.gguf'))}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    key_file = CONFIG_DIR / "api-key"
    key_file.write_text(key + "\n", encoding="utf-8")
    key_file.chmod(0o600)
    return path


def write_systemd_units(host: str = "127.0.0.1", port: int = 8080) -> None:
    USER_SYSTEMD.mkdir(parents=True, exist_ok=True)
    backend = USER_SYSTEMD / "hga-qwen38.service"
    gateway = USER_SYSTEMD / "hga-qwen38-gateway.service"
    backend.write_text(render_backend_unit(ROOT), encoding="utf-8")
    gateway.write_text(render_gateway_unit(ROOT, host, port), encoding="utf-8")
    (ROOT / "deployment" / "hga-qwen38.service").write_text(
        render_backend_unit(ROOT), encoding="utf-8"
    )
    (ROOT / "deployment" / "hga-qwen38-gateway.service").write_text(
        render_gateway_unit(ROOT, host, port), encoding="utf-8"
    )


def _user_runtime_dir() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("XDG_RUNTIME_DIR"):
        candidates.append(Path(os.environ["XDG_RUNTIME_DIR"]))
    candidates.append(Path(f"/run/user/{os.getuid()}"))
    seen: set[str] = set()
    for runtime in candidates:
        key = str(runtime)
        if key in seen:
            continue
        seen.add(key)
        if (runtime / "bus").exists():
            return runtime
    return None


def systemd_user_env() -> dict[str, str] | None:
    """Env that talks to the persistent user systemd, not a stale session bus.

    Desktop and IDE shells often keep DBUS_SESSION_BUS_ADDRESS pointed at a
    GNOME session bus under /tmp. systemctl --user then tries to activate
    org.freedesktop.systemd1 on that bus and fails with:
      Reload daemon failed: Process org.freedesktop.systemd1 exited with status 1
    """
    if shutil.which("systemctl") is None:
        return None
    runtime = _user_runtime_dir()
    if runtime is None:
        return None
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = str(runtime)
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime / 'bus'}"
    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show", "--property=Version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return env


def systemd_user_available() -> bool:
    return systemd_user_env() is not None


def wait_http_ok(url: str, key: str, timeout: int = 300) -> None:
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    last = None
    headers = {"Authorization": f"Bearer {key}"}
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"==> ready {url}", flush=True)
                    return
                last = response.status
        except Exception as exc:  # noqa: BLE001 — poll until llama-server binds
            last = exc
        time.sleep(2)
    log = CONFIG_DIR / "server.log"
    hint = ""
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        abort = [
            ln
            for ln in tail
            if "GGML_ASSERT" in ln
            or "no GPU backend" in ln
            or "failed to load" in ln
            or "missing model" in ln
        ]
        if abort:
            hint = "\n" + "\n".join(abort[-8:])
        hint += f"\nsee {log}"
    raise SystemExit(f"timed out waiting for {url}: {last}{hint}")


def start_systemd(env: dict[str, str]) -> None:
    run(["systemctl", "--user", "daemon-reload"], env=env)
    try:
        run(["loginctl", "enable-linger", os.environ.get("USER", "")], env=env)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("warning: could not enable linger; services may stop at logout", flush=True)
    run(["systemctl", "--user", "enable", "hga-qwen38.service", "hga-qwen38-gateway.service"], env=env)
    run(["systemctl", "--user", "restart", "hga-qwen38.service"], env=env)
    run(["systemctl", "--user", "restart", "hga-qwen38-gateway.service"], env=env)
    run(
        ["systemctl", "--user", "--no-pager", "status", "hga-qwen38.service", "hga-qwen38-gateway.service"],
        env=env,
    )


def public_url(args: argparse.Namespace) -> str:
    host = args.host_address
    if host in {"0.0.0.0", "::"}:
        host = socket.gethostname()
        try:
            host = socket.gethostbyname(host)
        except OSError:
            host = "127.0.0.1"
    return f"http://{host}:{args.port}"


def write_client_examples(base: str) -> None:
    clients = ROOT / "deployment" / "clients"
    clients.mkdir(parents=True, exist_ok=True)
    import json

    copilot = [
        {
            "name": "HGA Qwen3.8-27B",
            "vendor": "customendpoint",
            "apiType": "chat-completions",
            "apiKey": "${input:hgaApiKey}",
            "models": [
                {
                    "id": "qwen3.8-27b-hga-fast",
                    "name": "HGA Fast",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
                {
                    "id": "qwen3.8-27b-hga-normal",
                    "name": "HGA Normal",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
                {
                    "id": "qwen3.8-27b-hga-deep",
                    "name": "HGA Deep",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
            ],
        }
    ]
    (clients / "chatLanguageModels.json").write_text(
        json.dumps(copilot, indent=2) + "\n", encoding="utf-8"
    )
    (CONFIG_DIR / "clients").mkdir(parents=True, exist_ok=True)
    shutil.copy(clients / "opencode.json", CONFIG_DIR / "clients" / "opencode.json")
    shutil.copy(clients / "chatLanguageModels.json", CONFIG_DIR / "clients" / "chatLanguageModels.json")


def write_access_point(args: argparse.Namespace, vram_mib: int, threads: int) -> None:
    base = public_url(args)
    text = f"""# HGA AccessPoint (16 GB CUDA)

OpenAI-compatible API for OpenCode and GitHub Copilot Chat on this machine.

- Base URL: `{base}/v1`
- Chat completions: `{base}/v1/chat/completions`
- Models: `qwen3.8-27b-hga-fast`, `qwen3.8-27b-hga-normal`, `qwen3.8-27b-hga-deep`
- Auth: `Authorization: Bearer $HGA_API_KEY` (file `{CONFIG_DIR / "api.env"}`, mode 0600)
- GPU: {vram_mib} MiB  |  HGA threads: {threads}  |  context: {args.ctx}

Do not commit the API key. Source it with:

```bash
set -a; . ~/.config/hga-qwen38/api.env; set +a
```

## Start / stop

```bash
# preferred (systemd units and start-local.sh pid files):
{ROOT}/deployment/stop-local.sh
{ROOT}/deployment/start-local.sh

# raw systemctl — pin the persistent user bus. Desktop/IDE shells often keep
# DBUS_SESSION_BUS_ADDRESS pointed at a GNOME session bus under /tmp, which
# makes systemctl --user fail with org.freedesktop.systemd1 exited status 1:
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
systemctl --user start hga-qwen38.service hga-qwen38-gateway.service
systemctl --user stop  hga-qwen38.service hga-qwen38-gateway.service
```

## OpenCode

Merge the `hga-local` provider from `examples/opencode.json` into
`~/.config/opencode/opencode.json`. It talks to `{base}/v1` with
`apiKey` `{{file:~/.config/hga-qwen38/api-key}}` (use an absolute path).
Default model: `hga-local/qwen3.8-27b-hga-fast`. Restart OpenCode after editing.

## GitHub Copilot (VS Code Chat)

1. Command Palette → **Chat: Manage Language Models** → **Add Models** → **Custom Endpoint**.
2. API type: **Chat Completions**.
3. Paste `{ROOT}/deployment/clients/chatLanguageModels.json` (replace the API key prompt).
4. Enable **toolCalling** models for Copilot agent mode.

Copilot CLI:

```bash
export COPILOT_PROVIDER_BASE_URL={base}/v1
export COPILOT_PROVIDER_API_KEY="$HGA_API_KEY"
export COPILOT_MODEL=qwen3.8-27b-hga-fast
```

Inline Copilot completions still use GitHub-hosted models; this endpoint is for Chat / agent.

## Smoke

```bash
HGA_API_KEY="$HGA_API_KEY" python3 {ROOT}/deployment/smoke.py --url {base}
```
"""
    (ROOT / "access_point.md").write_text(text, encoding="utf-8")
    (CONFIG_DIR / "access_point.md").write_text(text, encoding="utf-8")


def install_local(args: argparse.Namespace) -> int:
    key = os.environ.get("HGA_API_KEY")
    if not key:
        raise SystemExit("HGA_API_KEY must be set; refusing to deploy an unauthenticated API")
    if any(c.isspace() for c in key):
        raise SystemExit("HGA_API_KEY may not contain whitespace")
    vram = require_16gb_gpu()
    threads = int(os.environ.get("HGA_THREADS") or detect_threads())
    for script in ROOT.glob("scripts/*.sh"):
        script.chmod(script.stat().st_mode | 0o111)
    for script in ROOT.glob("deployment/*.sh"):
        script.chmod(script.stat().st_mode | 0o111)
    ensure_gguf()
    ensure_llama_server(args.skip_build)
    write_api_env(args, key, threads)
    write_systemd_units(args.host_address, args.port)
    write_client_examples(public_url(args) if args.host_address != "0.0.0.0" else f"http://127.0.0.1:{args.port}")
    write_access_point(args, vram, threads)
    print(f"installed AccessPoint files under {CONFIG_DIR}", flush=True)
    if args.no_start:
        print("skipping start (--no-start). Use deployment/start-local.sh or systemctl --user.", flush=True)
        return 0
    systemd_env = None if args.foreground_helper else systemd_user_env()
    if systemd_env is not None:
        try:
            start_systemd(systemd_env)
        except subprocess.CalledProcessError as exc:
            print(
                f"warning: systemd --user failed ({exc}); falling back to deployment/start-local.sh",
                flush=True,
            )
            run(["bash", str(ROOT / "deployment" / "start-local.sh")])
    else:
        if not args.foreground_helper:
            print(
                "systemd --user is not reachable (stale session bus or no user manager); "
                "starting with deployment/start-local.sh",
                flush=True,
            )
        run(["bash", str(ROOT / "deployment" / "start-local.sh")])
    if not args.skip_smoke:
        print("==> waiting for llama-server to finish loading (~6-20s)", flush=True)
        wait_http_ok(f"http://127.0.0.1:{args.backend_port}/health", key, timeout=300)
        wait_http_ok(f"http://127.0.0.1:{args.port}/health", key, timeout=60)
        smoke_env = os.environ.copy()
        smoke_env["HGA_API_KEY"] = key
        url = f"http://127.0.0.1:{args.port}"
        run([python_bin(), str(ROOT / "deployment" / "smoke.py"), "--url", url], env=smoke_env)
    print(f"AccessPoint ready: {public_url(args)}/v1  (see {ROOT / 'access_point.md'})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", default=True, help="Install on this 16 GB GPU host (default)")
    parser.add_argument("--host-address", default=os.environ.get("HGA_API_HOST", "127.0.0.1"))
    parser.add_argument("--lan", action="store_true", help="Listen on 0.0.0.0 instead of loopback")
    parser.add_argument("--port", type=int, default=int(os.environ.get("HGA_API_PORT", "8080")))
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("HGA_BACKEND_PORT", "8081")))
    parser.add_argument("--ctx", type=int, default=int(os.environ.get("HGA_CTX", "131072")))
    parser.add_argument(
        "--hga-gpu-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU-routing / CUDA-attention prefill (default: enabled)",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--no-start", action="store_true", help="Write units and client configs only")
    parser.add_argument("--foreground-helper", action="store_true", help="Use start-local.sh even if systemd exists")
    args = parser.parse_args()
    if args.lan:
        args.host_address = "0.0.0.0"
    return install_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
