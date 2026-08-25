#!/usr/bin/env python3
"""Verify HGA API file comprehension without any OpenAI tool-call messages.

This isolates model generation from an agent client's tool-result plumbing:
the requested file is supplied verbatim as normal user text and the response
must summarize it without requesting or invoking a tool.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def post(url: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=root / "README.md")
    parser.add_argument("--url", default=os.environ.get("HGA_API_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="qwen3.8-27b-hga-fast")
    parser.add_argument("--report", default="/tmp/hga-supplied-file-test.json")
    args = parser.parse_args()
    key = os.environ.get("HGA_API_KEY")
    if not key:
        raise SystemExit("HGA_API_KEY is required")
    content = args.file.read_text(encoding="utf-8")
    response = post(args.url.rstrip("/"), key, {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Read the supplied file and summarize its purpose in two sentences. Do not call tools.\n\n" + content},
        ],
    })
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content")
    passed = isinstance(text, str) and bool(text.strip()) and not message.get("tool_calls")
    report = {
        "passed": passed,
        "model": args.model,
        "file": str(args.file),
        "response": text,
        "finish_reason": choice.get("finish_reason"),
        "timings": response.get("timings"),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("HGA_SUPPLIED_FILE_TEST " + json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
