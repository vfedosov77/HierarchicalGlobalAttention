#!/usr/bin/env python3
"""Minimal authenticated OpenAI API smoke test for the HGA service."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request(url: str, key: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 90) -> tuple[int, dict[str, Any]]:
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
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


def wait_model_on_gpu(timeout: int, min_mib: int = 10000) -> None:
    """llama-server answers /health before leftover VERIFY pin finishes."""
    deadline = time.monotonic() + timeout
    last = 0
    while time.monotonic() < deadline:
        last = gpu_used_mib()
        if last >= min_mib:
            print(f"==> GPU model resident ({last} MiB)", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"GPU memory stayed at {last} MiB; llama-server did not finish loading")


def wait_health(base: str, key: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last: object = None
    while time.monotonic() < deadline:
        try:
            status, body = request(base + "/health", key, timeout=10)
            if 200 <= status < 300 and (not body or body.get("status") in (None, "ok")):
                return
            last = {"status": status, "body": body}
        except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"service did not become healthy within {timeout}s: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("HGA_API_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="qwen3.8-27b-hga-fast")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--report", default="/tmp/hga-api-smoke.json")
    args = parser.parse_args()
    key = os.environ.get("HGA_API_KEY")
    if not key:
        raise SystemExit("HGA_API_KEY is required")
    base = args.url.rstrip("/")
    wait_health(base, key, args.timeout)
    wait_model_on_gpu(args.timeout)
    status, models = request(base + "/v1/models", key, timeout=30)
    if status != 200 or not isinstance(models.get("data"), list):
        raise RuntimeError(f"/v1/models failed: HTTP {status}: {models}")
    chat: dict[str, Any] = {}
    status = 0
    last_error: object = None
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: HGA API ready"}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    for attempt in range(1, 4):
        try:
            print(f"==> smoke chat attempt {attempt}", flush=True)
            status, chat = request(base + "/v1/chat/completions", key, "POST", payload, timeout=300)
            if status == 200:
                break
            last_error = {"status": status, "body": chat}
        except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
            last_error = str(exc)
            status, chat = 0, {"error": last_error}
        wait_model_on_gpu(args.timeout)
    if status != 200:
        raise RuntimeError(f"/v1/chat/completions failed after retries: {last_error or chat}")
    choices = chat.get("choices") if isinstance(chat, dict) else None
    message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    # Qwen3 may spend a very short completion budget entirely in its reasoning
    # channel.  Both fields are part of llama-server's OpenAI-compatible chat
    # response and either proves that inference completed.
    text = content if isinstance(content, str) and content.strip() else reasoning
    if status != 200 or not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"/v1/chat/completions failed: HTTP {status}: {chat}")
    report = {"passed": True, "url": base, "model": args.model, "model_count": len(models["data"]), "response": text}
    with open(args.report, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("HGA_API_SMOKE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
