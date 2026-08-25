#!/usr/bin/env python3
"""Public LongBench-E passage-retrieval quality check.

This is the gate that caught unconditional GPU-prefill graph reuse: the
model still ran fast, but retrieval dropped from 6/6 to 2/6.  The production
path rebuilds until the historical-validity boundary saturates, then reuses.

The input is the official ``passage_retrieval_en_e.jsonl`` file (see
``scripts/download_longbench.sh``).  A default six-example subset covers
answers near the beginning, middle, and end of the 30 passages.  Each
capacity gets a fresh llama-server process and every request starts a new
prompt, so KV state cannot leak between examples.

    python3 tools/bench_longbench_retrieval.py --self-test
    python3 tools/bench_longbench_retrieval.py --data ~/data/LongBench/passage_retrieval_en_e.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_hga.sh"
DEFAULT_INDICES = (200, 201, 202, 205, 208, 209)
PROMPT = (
    "/no_think\nHere are 30 paragraphs from Wikipedia, along with an abstract. Please "
    "determine which paragraph the abstract is from.\n\n{context}\n\n"
    "The following is an abstract.\n\n{input}\n\nPlease enter the number of "
    "the paragraph that the abstract is from. The answer format must be like "
    '"Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
)


def http_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("HGA_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def http_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers=http_headers(), method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def healthy(url: str) -> bool:
    try:
        request = urllib.request.Request(
            url.rstrip("/") + "/health", headers=http_headers())
        with urllib.request.urlopen(request, timeout=3) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10,
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


def server_env(capacity: int, ubatch: int, port: int, ctx: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HGA_SERVER": "1", "HGA_CTX": str(ctx), "HGA_BATCH": "768",
        "HGA_UBATCH": str(ubatch), "HGA_PREFILL_UBATCH": str(ubatch),
        "HGA_N": "32",
        "HGA_SPEC": "0",
        "HGA_THREADS": os.environ.get("HGA_THREADS", str(os.cpu_count() or 12)),
        "HGA_THREADS_BATCH": "1",
        "HGA_STREAM_ASYNC": "1", "HGA_STREAM_PACED": "1",
        # Two VERIFY streams: six leftover pairs pin on 16 GB. Three streams
        # leave less decode VRAM.
        "HGA_VERIFY_STREAMS": os.environ.get("HGA_VERIFY_STREAMS", "2"),
        "HGA_PIN_CHECK": "1",
        "HGA_FIT_TARGET": "0",
        "HGA_LOAD_MODE": os.environ.get("HGA_LOAD_MODE", "none"),
        "HGA_GPU_PREFILL": "1", "HGA_GPU_PREFILL_MIN_KEYS": "1552",
        "HGA_GPU_PREFILL_MAX_KEYS": str(capacity),
        "HGA_EXTRA": (
            f"--host 127.0.0.1 --port {port} --parallel 1 --no-warmup "
            "--no-context-shift --ignore-eos --reasoning off --jinja"
        ),
    })
    return env


def start_server(capacity: int, ubatch: int, port: int, ctx: int, log: Path,
                 startup_timeout: int) -> tuple[subprocess.Popen[str], Any]:
    stream = log.open("w", encoding="utf-8")
    used = gpu_used_mib()
    if used > 4000:
        stream.close()
        raise RuntimeError(
            f"GPU already has {used} MiB in use; a second Qwen3.8-27B load "
            "will OOM on 16 GB. Stop the AccessPoint first:\n"
            f"  {ROOT}/deployment/stop-local.sh\n"
            "or point this script at the running server:\n"
            "  --url http://127.0.0.1:8080"
        )
    proc = subprocess.Popen(
        [str(LAUNCHER)], cwd=ROOT, env=server_env(capacity, ubatch, port, ctx),
        stdout=stream, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    deadline = time.monotonic() + startup_timeout
    url = f"http://127.0.0.1:{port}"
    while time.monotonic() < deadline:
        if healthy(url):
            return proc, stream
        if proc.poll() is not None:
            stream.flush()
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            stream.close()
            raise RuntimeError(
                f"server exited with {proc.returncode}; see {log}\n"
                + "\n".join(tail)
            )
        time.sleep(1)
    os.killpg(proc.pid, signal.SIGTERM)
    stream.close()
    raise RuntimeError(f"server startup timeout; see {log}")


def stop_server(proc: subprocess.Popen[str], stream: Any, port: int) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    # run_hga may leave llama-server as a reparented process after its launcher
    # shell handles SIGTERM. Kill only the server owned by this benchmark port.
    pattern = f"llama-server.*--port {port}"
    subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
    time.sleep(2)
    subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    time.sleep(2)
    stream.close()


def answer_number(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def score_output(output: str, expected: int | None) -> tuple[int | None, float]:
    predicted_numbers = [int(value) for value in re.findall(r"\d+", output)]
    predicted = predicted_numbers[0] if predicted_numbers else None
    score = (
        predicted_numbers.count(expected) / len(predicted_numbers)
        if predicted_numbers and expected is not None
        else 0.0
    )
    return predicted, score


def self_test() -> int:
    assert answer_number("Paragraph 12") == 12
    assert answer_number("The answer is 3.") == 3
    predicted, score = score_output("Paragraph 5", 5)
    assert predicted == 5 and score == 1.0
    predicted, score = score_output("Paragraph 2 then 5", 5)
    assert predicted == 2 and score == 0.5
    predicted, score = score_output("no number", 5)
    assert predicted is None and score == 0.0
    print("self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="parser checks only; no GGUF, no server")
    parser.add_argument("--data")
    parser.add_argument(
        "--capacities",
        default="3200",
        help="comma-separated HGA_GPU_PREFILL_MAX_KEYS values; 3200 is the "
             "quality-validated 16 GB default (PoC A/B used 4096,3200,2048)",
    )
    parser.add_argument(
        "--ubatch", type=int, choices=(256, 512, 768, 1024), default=768,
        help="physical prefill ubatch; all capacities use this value",
    )
    parser.add_argument("--indices", default=",".join(map(str, DEFAULT_INDICES)))
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument(
        "--url",
        help="use an already-running llama-server / AccessPoint "
             "(do not spawn a second 16 GB load). Example: "
             "http://127.0.0.1:8081 or http://127.0.0.1:8080",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--json", default="/tmp/hga_longbench_capacity.json")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.data:
        raise SystemExit("pass --data PATH or --self-test")

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    indices = [int(value) for value in args.indices.split(",")]
    cases = [(index, rows[index]) for index in indices]
    capacities = [int(value) for value in args.capacities.split(",")]
    result: dict[str, Any] = {
        "benchmark": "LongBench-E/passage_retrieval_en",
        "indices": indices, "ctx": args.ctx, "ubatch": args.ubatch,
        "capacities": [],
    }

    def run_cases(url: str, capacity: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, case in cases:
            prompt = PROMPT.format(context=case["context"], input=case["input"])
            tokenized = http_json(url.rstrip("/") + "/tokenize", {"content": prompt}, 60)
            prompt_tokens = len(tokenized.get("tokens") or [])
            if prompt_tokens + 16 > args.ctx:
                raise RuntimeError(
                    f"case {index} has {prompt_tokens} tokens, over ctx={args.ctx}")
            started = time.monotonic()
            response = http_json(url.rstrip("/") + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16, "temperature": 0.0, "seed": 1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }, args.timeout)
            wall = time.monotonic() - started
            choices = response.get("choices") or []
            message = choices[0].get("message") if choices else {}
            output = str((message or {}).get("content") or "").strip()
            expected = answer_number(case["answers"][0])
            predicted, score = score_output(output, expected)
            timings = response.get("timings") or {}
            record = {
                "index": index, "prompt_tokens": prompt_tokens,
                "expected": expected, "predicted": predicted,
                "score": score, "correct": score == 1.0, "output": output,
                "wall_s": round(wall, 3),
                "prefill_tok_s": timings.get("prompt_per_second"),
            }
            records.append(record)
            print("CASE " + json.dumps({"capacity": capacity, **record}),
                  flush=True)
        return records

    def report(capacity: int, records: list[dict[str, Any]], log: str | None) -> None:
        correct = sum(record["correct"] for record in records)
        score = sum(record["score"] for record in records) / len(records)
        result["capacities"].append({
            "capacity": capacity, "correct": correct, "total": len(records),
            "accuracy": score, "exact_accuracy": correct / len(records),
            "cases": records,
            "server_log": log,
        })
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"CAPACITY {capacity}: {correct}/{len(records)}", flush=True)

    if args.url:
        url = args.url.rstrip("/")
        if not healthy(url):
            raise SystemExit(f"no healthy server at {url}")
        records = run_cases(url, capacities[0] if capacities else 0)
        report(capacities[0] if capacities else 0, records, None)
    else:
        for run, capacity in enumerate(capacities):
            port = args.port + run
            log = Path(f"/tmp/hga_longbench_u{args.ubatch}_k{capacity}.log")
            proc, stream = start_server(
                capacity, args.ubatch, port, args.ctx, log, args.startup_timeout)
            try:
                records = run_cases(f"http://127.0.0.1:{port}", capacity)
            finally:
                stop_server(proc, stream, port)
            report(capacity, records, str(log))

    print("RESULT " + json.dumps({
        row["capacity"]: row["accuracy"] for row in result["capacities"]
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
