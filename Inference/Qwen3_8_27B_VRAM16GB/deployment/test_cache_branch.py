#!/usr/bin/env python3
"""Verify lazy-prefix branch discovery and finished-chat continuation hits."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Any


def post(url: str, key: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def completion(
    url: str,
    key: str,
    prompt: str | list[int],
    timeout: int,
    *,
    return_tokens: bool = False,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.0,
        "cache_prompt": True,
        "id_slot": 0,
        "stream": False,
        "return_tokens": return_tokens,
    }
    return post(url, key, "/completion", payload, timeout)


def tokenize(url: str, key: str, text: str, timeout: int, *, add_special: bool) -> list[int]:
    result = post(
        url,
        key,
        "/tokenize",
        {"content": text, "add_special": add_special},
        timeout,
    )
    return result["tokens"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    key = os.environ.get("HGA_API_KEY", "")
    if not key:
        raise SystemExit("HGA_API_KEY is required")

    unit = (
        "Checkpoint branch validation uses deterministic text so tokenization "
        "and the reusable prefix remain stable across both requests. "
    )
    base = "Continue this synthetic document briefly.\n\n" + unit * 160
    first = completion(args.url, key, base + "\nFirst branch.\n", args.timeout)
    # The second request discovers the shared prefix and creates one checkpoint
    # there. It can still reprocess the prefix once; the third request must hit.
    second = completion(args.url, key, base + "\nSecond branch.\n", args.timeout)
    third_text = base + "\nThird branch.\n"
    third = completion(args.url, key, third_text, args.timeout, return_tokens=True)

    # A finished request stores the state through its last evaluated generated
    # token. Send token IDs here to make that boundary exact and independent of
    # text-tokenizer concatenation behavior.
    third_tokens = tokenize(args.url, key, third_text, args.timeout, add_special=True)
    generated_tokens = third.get("tokens") or []
    suffix_tokens = tokenize(
        args.url,
        key,
        "\nContinue this conversation.\n",
        args.timeout,
        add_special=False,
    )
    continuation = completion(
        args.url,
        key,
        third_tokens + generated_tokens + suffix_tokens,
        args.timeout,
    )

    first_timing = first.get("timings") or {}
    second_timing = second.get("timings") or {}
    third_timing = third.get("timings") or {}
    continuation_timing = continuation.get("timings") or {}
    report = {
        "first_prompt_n": first_timing.get("prompt_n"),
        "first_cache_n": first_timing.get("cache_n"),
        "second_prompt_n": second_timing.get("prompt_n"),
        "second_cache_n": second_timing.get("cache_n"),
        "third_prompt_n": third_timing.get("prompt_n"),
        "third_cache_n": third_timing.get("cache_n"),
        "finish_checkpoint_tokens": len(third_tokens) + max(0, len(generated_tokens) - 1),
        "continuation_prompt_n": continuation_timing.get("prompt_n"),
        "continuation_cache_n": continuation_timing.get("cache_n"),
    }
    report["passed"] = (
        isinstance(report["first_prompt_n"], int)
        and isinstance(report["second_prompt_n"], int)
        and isinstance(report["third_prompt_n"], int)
        and isinstance(report["third_cache_n"], int)
        and isinstance(report["continuation_prompt_n"], int)
        and isinstance(report["continuation_cache_n"], int)
        and report["first_prompt_n"] > 512
        and report["second_prompt_n"] > 512
        and report["third_cache_n"] > 512
        and report["third_prompt_n"] < 64
        and len(generated_tokens) == 8
        and report["continuation_cache_n"] >= report["finish_checkpoint_tokens"]
        and report["continuation_prompt_n"] <= len(suffix_tokens) + 2
    )
    print("HGA_CACHE_BRANCH_TEST " + json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
