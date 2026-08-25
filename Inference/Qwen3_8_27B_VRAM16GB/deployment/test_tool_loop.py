#!/usr/bin/env python3
"""Reproduce OpenCode's README tool-call continuation against the HGA API.

The test deliberately performs the same OpenAI chat sequence used by an
agent: ask to read a file, return every requested ``read`` tool result (the
whole, line-numbered README), then ask the model to continue.  It records raw
tool calls before validating them, which makes truncated JSON and duplicate
calls diagnosable without relying on the OpenCode UI.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a text file at the supplied absolute file path.",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
            "additionalProperties": False,
        },
    },
}


def post(base: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def opencode_read_output(path: Path) -> str:
    """Match the line-numbered envelope OpenCode gives its read tool result."""
    lines = path.read_text(encoding="utf-8").splitlines()
    numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1))
    return f"<path>{path}</path>\n<type>file</type>\n<content>\n{numbered}\n</content>"


def tool_result(call: dict[str, Any], expected_path: Path, output: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the tool message plus a diagnostic record for one model call."""
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name")
    raw_args = function.get("arguments", "")
    diagnostic: dict[str, Any] = {
        "id": call.get("id"), "name": name, "arguments_raw": raw_args,
        "arguments_valid": False, "accepted": False,
    }
    try:
        arguments = json.loads(raw_args)
        diagnostic["arguments"] = arguments
        diagnostic["arguments_valid"] = isinstance(arguments, dict)
    except (TypeError, json.JSONDecodeError) as exc:
        diagnostic["parse_error"] = str(exc)
    if name == "read" and diagnostic["arguments_valid"] and diagnostic["arguments"].get("filePath") == str(expected_path):
        content = output
        diagnostic["accepted"] = True
        diagnostic["result"] = "read README"
    else:
        content = "The arguments provided to the tool are invalid."
        diagnostic["result"] = "invalid tool call"
    return {"role": "tool", "tool_call_id": call.get("id", "missing-id"), "content": content}, diagnostic


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=root / "README.md")
    parser.add_argument("--url", default=os.environ.get("HGA_API_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="qwen3.8-27b-hga-fast")
    parser.add_argument("--max-turns", type=int, default=2, help="Bound the expensive tool-result continuation.")
    parser.add_argument(
        "--disallow-parallel-tool-calls", action="store_true",
        help="Send OpenAI's standard parallel_tool_calls=false control.",
    )
    parser.add_argument("--report", default="/tmp/hga-tool-loop-test.json")
    args = parser.parse_args()
    key = os.environ.get("HGA_API_KEY")
    if not key:
        raise SystemExit("HGA_API_KEY is required")
    path = args.file.resolve()
    full_output = opencode_read_output(path)
    messages: list[dict[str, Any]] = [{
        "role": "user",
        # READ_TOOL requires an absolute path. Supplying it also keeps this
        # reproducer independent of an agent client's system-prompt cwd.
        "content": f"{path} - read that file and say what it is about",
    }]
    turns: list[dict[str, Any]] = []
    for number in range(1, args.max_turns + 1):
        request: dict[str, Any] = {
            "model": args.model,
            "temperature": 0,
            "max_tokens": 384,
            "stream": False,
            "tools": [READ_TOOL],
            "messages": messages,
        }
        if args.disallow_parallel_tool_calls:
            request["parallel_tool_calls"] = False
        response = post(args.url.rstrip("/"), key, request)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        turn = {
            "turn": number,
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_call_count": len(calls),
            "raw_tool_calls": calls,
            "timings": response.get("timings"),
        }
        messages.append(message)
        if not calls:
            turns.append(turn)
            break
        diagnostics: list[dict[str, Any]] = []
        for call in calls:
            result, diagnostic = tool_result(call, path, full_output)
            messages.append(result)
            diagnostics.append(diagnostic)
        turn["tool_diagnostics"] = diagnostics
        turns.append(turn)

    duplicate_count = sum(turn["tool_call_count"] for turn in turns)
    invalid = [entry for turn in turns for entry in turn.get("tool_diagnostics", []) if not entry["accepted"]]
    report = {
        "model": args.model,
        "parallel_tool_calls": not args.disallow_parallel_tool_calls,
        "file": str(path),
        "full_tool_output_bytes": len(full_output.encode("utf-8")),
        "turns": turns,
        "total_tool_calls": duplicate_count,
        "invalid_tool_calls": invalid,
        # Retain the old key for consumers of reports from earlier revisions.
        "malformed_tool_calls": [entry for entry in invalid if not entry["arguments_valid"]],
        "passed": not invalid and duplicate_count == 1 and bool(turns) and not turns[-1]["raw_tool_calls"],
        "verdict": "PASS" if not invalid and duplicate_count == 1 and bool(turns) and not turns[-1]["raw_tool_calls"] else "MODEL_TOOL_LOOP_OR_TRUNCATION",
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("HGA_TOOL_LOOP_TEST " + json.dumps({
        "verdict": report["verdict"], "turns": len(turns), "tool_calls": duplicate_count,
        "invalid": len(invalid), "report": args.report,
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
