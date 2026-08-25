#!/usr/bin/env python3
"""Persistent mixed-prefill/speculative-generation benchmark for a 16 GB GPU.

Unlike bench.py's one-shot 2K -> 64 measurement, this keeps one llama-server
slot alive and submits the complete growing conversation on every request.
With cache_prompt=true the server reports exactly how many new prompt tokens
were evaluated (timings.prompt_n) and how many prior tokens were reused
(timings.cache_n).  It therefore exercises HGA cache append/truncate, graph
reuse, and speculative VERIFY after more than one prefill block.

Run this on the 16 GB GPU host after `scripts/setup.sh`:

    python3 tools/mixed_turn_bench.py --scenario all
    python3 tools/mixed_turn_bench.py --scenario long
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_hga.sh"
SENTENCE = (
    "HGA mixed-turn validation records deterministic synthetic context so that "
    "each appended prefill block has stable tokenization and a distinct turn. "
)
PREAMBLE = "Summarize each new numbered note in one short sentence.\n\n"

# The short cases deliberately straddle 768-token physical PREFILL batches and
# 64-token HGA chunks.  The mixed case also alternates tiny and large appends.
SCENARIOS: dict[str, list[tuple[int, int]]] = {
    "smoke": [(767, 3), (1, 64), (769, 2), (2048, 128), (4095, 16)],
    "mixed": [(256, 16), (1536, 96), (768, 256), (4096, 32), (1024, 128), (8192, 64)],
    # Long mixed-turn load against the recommended 128K AccessPoint window.
    # The generated output is deliberately varied as well.
    "long": [(32768, 32), (32768, 64), (32768, 128), (32768, 16),
             (32768, 256), (32768, 64), (32768, 32), (28672, 128)],
}


def http_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(url: str, timeout: int) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def tokenize_count(url: str, content: str, timeout: int) -> int:
    response = http_json(url + "/tokenize", {"content": content}, timeout)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("/tokenize did not return a token list")
    return len(tokens)


def build_block(url: str, tokens: int, turn: int, timeout: int) -> str:
    # Exact target counts are not assumed: tokenizer boundaries are model
    # dependent.  timings.prompt_n is the authoritative measured value.
    prefix = f"\n[{turn}] "
    unit_tokens = tokenize_count(url, prefix + SENTENCE, timeout) - tokenize_count(url, prefix, timeout)
    copies = max(1, (tokens + unit_tokens - 1) // max(1, unit_tokens))
    return prefix + (SENTENCE * copies)


def server_env(ctx: int, port: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HGA_SERVER": "1",
        "HGA_CTX": str(ctx),
        "HGA_BATCH": "768",
        "HGA_UBATCH": "768",
        "HGA_PREFILL_UBATCH": "768",
        "HGA_N": "512",
        "HGA_SPEC": "2",
        "HGA_THREADS": os.environ.get("HGA_THREADS", str(os.cpu_count() or 12)),
        "HGA_THREADS_BATCH": "1",
        "HGA_GPU_PREFILL": os.environ.get("HGA_GPU_PREFILL", "1"),
        "HGA_GPU_PREFILL_MAX_KEYS": os.environ.get("HGA_GPU_PREFILL_MAX_KEYS", "3200"),
        "HGA_VERIFY_BATCH": "1",
        "HGA_VERIFY_TILES": "1",
        "HGA_VERIFY_ROWS": "0",
        "HGA_STREAM_ASYNC": "1",
        "HGA_STREAM_PACED": "1",
        # Server requests retain a larger response buffer than the one-shot
        # CLI.  Two streams pin six leftover pairs and leave ~10 MiB on the
        # V100, below the next server graph's working allocation.  Three
        # streams/five pins leaves enough headroom while remaining speculative.
        "HGA_VERIFY_STREAMS": "3",
        "HGA_PIN_CHECK": "1",
        # A single slot is intentional: this is a sequential conversation,
        # not a continuous-batching throughput test.
        "HGA_EXTRA": (
            f"--host 127.0.0.1 --port {port} --parallel 1 "
            "--no-warmup --no-context-shift --ignore-eos"
        ),
    })
    return env


def start_server(ctx: int, port: int, log: Path, startup_timeout: int) -> subprocess.Popen[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(LAUNCHER)], cwd=ROOT, env=server_env(ctx, port),
        stdout=stream, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if health(url, 3):
            stream.close()
            return proc
        if proc.poll() is not None:
            stream.close()
            raise RuntimeError(f"llama-server exited early ({proc.returncode}); see {log}")
        time.sleep(1)
    os.killpg(proc.pid, signal.SIGTERM)
    stream.close()
    raise RuntimeError(f"llama-server did not become healthy within {startup_timeout}s; see {log}")


def run_scenario(name: str, turns: list[tuple[int, int]], url: str, timeout: int) -> dict[str, Any]:
    conversation = PREAMBLE
    records: list[dict[str, Any]] = []
    for index, (requested_prefill, requested_generate) in enumerate(turns, start=1):
        conversation += build_block(url, requested_prefill, index, timeout)
        started = time.monotonic()
        response = http_json(url + "/completion", {
            "prompt": conversation,
            "n_predict": requested_generate,
            "temperature": 0.0,
            "cache_prompt": True,
            "id_slot": 0,
            "stream": False,
        }, timeout)
        wall_s = time.monotonic() - started
        timings = response.get("timings") or {}
        generated = response.get("content") or ""
        if not isinstance(generated, str):
            generated = str(generated)
        row = {
            "turn": index,
            "requested_prefill_tokens": requested_prefill,
            "requested_generate_tokens": requested_generate,
            "prompt_n": timings.get("prompt_n"),
            "cache_n": timings.get("cache_n"),
            "prefill_tok_s": timings.get("prompt_per_second"),
            "generated_n": timings.get("predicted_n"),
            "generate_tok_s": timings.get("predicted_per_second"),
            "wall_s": round(wall_s, 3),
            "stop": response.get("stop"),
        }
        records.append(row)
        print(
            "TURN " + json.dumps(row, sort_keys=True), flush=True,
        )
        # The returned completion is part of the next request's exact prefix;
        # this is what makes cache_n prove session continuity.
        conversation += generated

    failures: list[str] = []
    for row in records:
        if not isinstance(row["prompt_n"], int) or row["prompt_n"] <= 0:
            failures.append(f"turn {row['turn']}: no newly-prefilled tokens")
        if row["turn"] > 1 and (not isinstance(row["cache_n"], int) or row["cache_n"] <= 0):
            failures.append(f"turn {row['turn']}: prompt cache was not reused")
        if row["prefill_tok_s"] is not None and row["prefill_tok_s"] < 180:
            failures.append(f"turn {row['turn']}: prefill below 180 tok/s ({row['prefill_tok_s']})")
        # Some speculative tasks legitimately vary, but 10 tok/s is the
        # practical regression floor requested for mixed production traffic.
        if row["generate_tok_s"] is not None and row["generate_tok_s"] < 10:
            failures.append(f"turn {row['turn']}: generate below 10 tok/s ({row['generate_tok_s']})")
    return {"scenario": name, "turns": records, "passed": not failures, "errors": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("smoke", "mixed", "long", "all"), default="all")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-turn HTTP timeout")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--json", default="/tmp/hga_mixed_turns.json")
    parser.add_argument("--log", default="/tmp/hga_mixed_turns_server.log")
    args = parser.parse_args()
    if not Path(os.environ.get("HGA_MODEL", os.path.expanduser("~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"))).is_file():
        raise SystemExit("missing Qwen3.8-27B-UD-Q4_K_M.gguf")

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    ctx = 131072 if "long" in names else 32768
    proc = start_server(ctx, args.port, Path(args.log), args.startup_timeout)
    url = f"http://127.0.0.1:{args.port}"
    result: dict[str, Any] = {"ctx": ctx, "scenarios": [], "server_log": args.log}
    try:
        for name in names:
            result["scenarios"].append(run_scenario(name, SCENARIOS[name], url, args.timeout))
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    result["passed"] = all(item["passed"] for item in result["scenarios"])
    Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT {json.dumps({'passed': result['passed'], 'json': args.json})}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
