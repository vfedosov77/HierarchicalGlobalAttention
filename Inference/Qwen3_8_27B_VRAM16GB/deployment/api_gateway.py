#!/usr/bin/env python3
"""OpenAI-compatible profile gateway for the single HGA llama-server slot.

OpenCode 1.x discovers model variants but does not forward arbitrary variant
request bodies to compatible endpoints.  This gateway makes the profiles real
API model IDs and injects the llama.cpp-specific request fields itself.
"""
from __future__ import annotations

import argparse
import hmac
import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

BACKEND = os.environ.get("HGA_BACKEND_URL", "http://127.0.0.1:8081")
API_KEY = os.environ.get("HGA_API_KEY", "")
OUTPUT_TOKEN_LIMIT = 131072

PROFILES: dict[str, dict[str, Any]] = {
    "qwen3.8-27b-hga": {"thinking": True, "thinking_budget": 512, "max_tokens": OUTPUT_TOKEN_LIMIT},
    "qwen3.8-27b-hga-fast": {"thinking": False, "max_tokens": OUTPUT_TOKEN_LIMIT},
    "qwen3.8-27b-hga-normal": {"thinking": True, "thinking_budget": 512, "max_tokens": OUTPUT_TOKEN_LIMIT},
    "qwen3.8-27b-hga-deep": {"thinking": True, "thinking_budget": 4096, "max_tokens": OUTPUT_TOKEN_LIMIT},
}
UPSTREAM_MODEL = "qwen3.8-27b-hga"
AUTH_HEADER_NAMES = (
    "x-api-key",
    "api-key",
    "apikey",
    "x-goog-api-key",
)


def request_api_keys(headers: Any, path: str) -> list[str]:
    """Collect candidate API keys from Copilot/OpenAI/Anthropic header styles."""
    keys: list[str] = []
    auth = str(headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        keys.append(auth[7:].strip())
    elif auth:
        keys.append(auth)
    for name in AUTH_HEADER_NAMES:
        value = str(headers.get(name) or "").strip()
        if value:
            keys.append(value)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    for name in ("api_key", "api-key", "key"):
        values = query.get(name) or []
        if values and values[0]:
            keys.append(values[0].strip())
    cleaned: list[str] = []
    seen: set[str] = set()
    for key in keys:
        key = key.strip().strip('"').strip("'")
        if key and key not in seen:
            seen.add(key)
            cleaned.append(key)
    return cleaned


def key_matches(supplied: str, expected: str) -> bool:
    if not expected or len(supplied) != len(expected):
        return False
    return hmac.compare_digest(supplied, expected)


def read_chunked_body(rfile: Any) -> bytes:
    chunks: list[bytes] = []
    while True:
        line = rfile.readline()
        if not line:
            break
        size_token = line.split(b";", 1)[0].strip()
        try:
            size = int(size_token, 16)
        except ValueError:
            break
        if size == 0:
            while True:
                trailer = rfile.readline()
                if trailer in (b"\r\n", b"\n", b""):
                    break
            break
        chunks.append(rfile.read(size))
        rfile.read(2)
    return b"".join(chunks)


def apply_profile(body: dict[str, Any]) -> dict[str, Any]:
    selected = str(body.get("model") or UPSTREAM_MODEL)
    profile = PROFILES.get(selected)
    if profile is None:
        raise ValueError(f"unknown HGA model profile: {selected}")
    result = dict(body)
    result["model"] = UPSTREAM_MODEL
    # Ask llama-server to reuse an exact prefix.  Do not force id_slot=0:
    # OpenCode also makes independent title/compaction requests, and pinning
    # those to the agent slot can attach an unrelated stale context.
    result["cache_prompt"] = True
    result.pop("id_slot", None)
    # Qwen3.8's Muse Glimmer tool format permits another assistant message
    # after each tool call when parallel calls are enabled. In practice the
    # model repeats the same call until max_tokens instead of yielding to the
    # agent. This deployment is a sequential agent endpoint, so require one
    # call/result round trip at a time even when a client enables it.
    if result.get("tools"):
        result["parallel_tool_calls"] = False
    kwargs = dict(result.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = bool(profile["thinking"])
    result["chat_template_kwargs"] = kwargs
    if profile["thinking"]:
        result["thinking_budget_tokens"] = profile["thinking_budget"]
    else:
        result.pop("thinking_budget_tokens", None)
    if result.get("max_tokens") is None and result.get("max_completion_tokens") is not None:
        result["max_tokens"] = result["max_completion_tokens"]
    result.pop("max_completion_tokens", None)
    requested = result.get("max_tokens", profile["max_tokens"])
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = profile["max_tokens"]
    result["max_tokens"] = max(1, min(requested, profile["max_tokens"]))
    return result


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("hga-gateway: " + fmt % args + "\n")

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        finally:
            self.close_connection = True

    def authorized(self) -> bool:
        if not API_KEY:
            return False
        expected = API_KEY.strip()
        return any(key_matches(supplied, expected) for supplied in request_api_keys(self.headers, self.path))

    def read_request_body(self) -> bytes:
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            return read_chunked_body(self.rfile)
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return b""
        try:
            length = int(length_header)
        except ValueError:
            return b""
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def require_auth(self) -> bool:
        if self.authorized():
            return True
        present = [
            name
            for name in ("Authorization", *AUTH_HEADER_NAMES)
            if self.headers.get(name)
        ]
        candidates = request_api_keys(self.headers, self.path)
        hint = ""
        if any(c in {"apiKey", "${apiKey}"} for c in candidates):
            hint = " (Copilot sent an unexpanded apiKey placeholder; remove requestHeaders.Authorization)"
        sys.stderr.write(
            "hga-gateway: auth rejected path=%s headers=%s candidate_lens=%s expected_len=%d%s\n"
            % (
                self.path.split("?", 1)[0],
                ",".join(present) or "none",
                ",".join(str(len(k)) for k in candidates) or "none",
                len(API_KEY),
                hint,
            )
        )
        self.send_json(401, {"error": {"message": "invalid API key", "type": "authentication_error"}})
        return False

    def do_GET(self) -> None:
        self.read_request_body()
        if self.path.split("?", 1)[0] == "/health":
            self.proxy(None)
            return
        if self.path.split("?", 1)[0].rstrip("/") == "/v1/models":
            if not self.require_auth():
                return
            data = [{"id": model, "object": "model", "owned_by": "hga"} for model in PROFILES]
            self.send_json(200, {"object": "list", "data": data})
            return
        if not self.require_auth():
            return
        self.proxy(None)

    def do_POST(self) -> None:
        raw = self.read_request_body()
        if not self.require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path.startswith("/v1/chat/completions") or path.startswith("/v1/completions"):
            try:
                body = apply_profile(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, ValueError) as exc:
                self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                return
            raw = json.dumps(body).encode("utf-8")
        self.proxy(raw)

    def proxy(self, data: bytes | None) -> None:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        for name in ("Content-Type", "Accept"):
            if self.headers.get(name):
                headers[name] = self.headers[name]
        stream = False
        if data:
            try:
                stream = bool(json.loads(data.decode("utf-8")).get("stream"))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                stream = False
        request = urllib.request.Request(BACKEND + self.path, data=data, headers=headers, method=self.command)
        try:
            response = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            self.send_json(503, {"error": {"message": f"HGA backend unavailable: {reason}", "type": "server_error"}})
            return
        try:
            if stream:
                self.send_response(response.status)
                content_type = response.headers.get("Content-Type", "text/event-stream")
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            body = response.read()
            self.send_response(response.status)
            content_type = response.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (http.client.RemoteDisconnected, ConnectionError, TimeoutError, OSError) as exc:
            if not getattr(self, "wfile", None):
                return
            try:
                self.send_json(503, {"error": {"message": f"HGA backend closed: {exc}", "type": "server_error"}})
            except Exception:
                pass
        finally:
            response.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HGA_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HGA_API_PORT", "8080")))
    args = parser.parse_args()
    if not API_KEY:
        raise SystemExit("HGA_API_KEY is required")
    server = ThreadingHTTPServer((args.host, args.port), GatewayHandler)
    print(f"hga-gateway: listening on {args.host}:{args.port}; backend={BACKEND}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
