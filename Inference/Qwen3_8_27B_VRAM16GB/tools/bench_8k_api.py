#!/usr/bin/env python3
"""8K prefill / 64-token generate against the live local AccessPoint.

Does not start, stop, rebuild, or deploy llama-server. The running API
(default http://127.0.0.1:8080) must already be healthy. The prompt body
matches tools/bench_8k.py; a per-run nonce keeps the lazy prefix cache
from turning a re-run into a cache hit.

    python3 tools/bench_8k_api.py --self-test
    python3 tools/bench_8k_api.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

TOOLS = Path(__file__).resolve().parent
QWEN_ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import bench_8k  # noqa: E402

SUITE_NAME = "prefill-8k-ubatch768-gen-64-accesspoint"
CONFIG_DIR = Path.home() / ".config" / "hga-qwen38"
DEFAULT_MODEL_ID = "qwen3.8-27b-hga-fast"
CACHE_N_MAX = 64


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_api_key() -> str:
    for name in ("HGA_API_KEY", "LLAMA_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    key_file = CONFIG_DIR / "api-key"
    if key_file.is_file():
        value = key_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return load_env_file(CONFIG_DIR / "api.env").get("HGA_API_KEY", "").strip()


def default_url() -> str:
    explicit = os.environ.get("HGA_API_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    env = load_env_file(CONFIG_DIR / "api.env")
    host = (
        os.environ.get("HGA_API_HOST", "").strip()
        or env.get("HGA_API_HOST", "").strip()
        or "127.0.0.1"
    )
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = (
        os.environ.get("HGA_API_PORT", "").strip()
        or env.get("HGA_API_PORT", "").strip()
        or "8080"
    )
    return f"http://{host}:{port}"


def normalize_base(url: str) -> str:
    base = url.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def auth_headers(key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def http_json(
    url: str,
    key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
    method: str | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=auth_headers(key),
        method=method or ("GET" if data is None else "POST"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                return response.status, {"raw": body}
            return response.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw[:2000]}
        if not isinstance(body, dict):
            body = {"raw": body}
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def health_state(url: str, key: str) -> str:
    """Return ok / loading / down / unhealthy for a live AccessPoint URL."""
    try:
        status, body = http_json(
            url.rstrip("/") + "/health", key, timeout=5, method="GET"
        )
    except RuntimeError as exc:
        text = str(exc).lower()
        if "refused" in text or "connect" in text or "name or service" in text:
            return "down"
        return "unhealthy"
    if 200 <= status < 300 and (not body or body.get("status") in (None, "ok")):
        return "ok"
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    message = str(err.get("message") or "")
    if status == 503 and "load" in message.lower():
        return "loading"
    return "unhealthy"


def candidate_urls(preferred: str) -> list[str]:
    base = normalize_base(preferred)
    urls = [base]
    env = load_env_file(CONFIG_DIR / "api.env")
    backend = (
        os.environ.get("HGA_BACKEND_URL", "").strip()
        or env.get("HGA_BACKEND_URL", "").strip()
        or "http://127.0.0.1:8081"
    )
    backend = normalize_base(backend)
    if backend not in urls:
        urls.append(backend)
    return urls


def wait_healthy(url: str, key: str, timeout: int) -> bool:
    deadline = time.monotonic() + max(0, timeout)
    while True:
        state = health_state(url, key)
        if state == "ok":
            return True
        if state == "down":
            return False
        if time.monotonic() >= deadline:
            return False
        if state == "loading":
            print(f"==> {url} still loading the model", flush=True)
        time.sleep(2.0)


def pick_live_url(preferred: str, key: str, wait: int) -> str:
    tried: list[str] = []
    for url in candidate_urls(preferred):
        print(f"==> waiting up to {wait}s for {url}", flush=True)
        if wait_healthy(url, key, wait):
            if url != normalize_base(preferred):
                print(
                    f"==> using {url} (preferred {preferred} was not ready)",
                    flush=True,
                )
            return url
        tried.append(url)
    raise RuntimeError(
        "AccessPoint is not healthy at "
        + ", ".join(tried)
        + "\nThis harness never starts llama-server itself."
    )


def build_api_prompt(*, stable: bool = False) -> str:
    body = bench_8k.build_prompt()
    if stable:
        return body
    return f"Speed probe nonce: {uuid.uuid4().hex}\n" + body


def _round_rate(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def perf_from_timings(timings: dict[str, Any]) -> dict[str, Any]:
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    prompt_tok_s = timings.get("prompt_per_second")
    generate_tok_s = timings.get("predicted_per_second")
    if prompt_ms is None and prompt_n and prompt_tok_s:
        prompt_ms = 1000.0 * float(prompt_n) / float(prompt_tok_s)
    if predicted_ms is None and predicted_n and generate_tok_s:
        predicted_ms = 1000.0 * float(predicted_n) / float(generate_tok_s)
    if prompt_tok_s is None and prompt_n and prompt_ms and float(prompt_ms) > 0:
        prompt_tok_s = 1000.0 * float(prompt_n) / float(prompt_ms)
    if generate_tok_s is None and predicted_n and predicted_ms and float(predicted_ms) > 0:
        generate_tok_s = 1000.0 * float(predicted_n) / float(predicted_ms)
    return {
        "prefill_tokens": prompt_n,
        "prefill_ms": _round_rate(prompt_ms),
        "prefill_tok_s": _round_rate(prompt_tok_s),
        "generate_tokens": predicted_n,
        "generate_ms": _round_rate(predicted_ms),
        "generate_tok_s": _round_rate(generate_tok_s),
        "cache_n": timings.get("cache_n"),
    }


def evaluate_cache(perf: dict[str, Any]) -> list[str]:
    cache_n = perf.get("cache_n")
    if isinstance(cache_n, int) and cache_n > CACHE_N_MAX:
        return [
            f"cache_n={cache_n}; AccessPoint reused KV — not a full 8K prefill "
            "(re-run without --stable-prompt, or wait for a different slot prefix)"
        ]
    return []


def live_llama() -> dict[str, Any]:
    info: dict[str, Any] = {
        "pid": None,
        "spec_k": None,
        "ctx": None,
        "port": None,
    }
    try:
        import subprocess

        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return info
    for line in (proc.stdout or "").splitlines():
        if "llama-server" not in line or "-m " not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            info["pid"] = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        match = re.search(r"--spec-draft-n-max\s+(\d+)", cmd)
        if match:
            info["spec_k"] = int(match.group(1))
        match = re.search(r"(?:-c|--ctx-size)\s+(\d+)", cmd)
        if match:
            info["ctx"] = int(match.group(1))
        match = re.search(r"--port\s+(\d+)", cmd)
        if match:
            info["port"] = int(match.group(1))
        break
    return info


def completion_payload(prompt: str, n_predict: int, model: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "max_tokens": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": True,
        "model": model,
    }


def openai_payload(prompt: str, n_predict: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": n_predict,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }


def post_completion(
    base: str,
    key: str,
    prompt: str,
    n_predict: int,
    model: str,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    native = completion_payload(prompt, n_predict, model)
    status, body = http_json(base + "/completion", key, native, timeout=timeout)
    if status == 404:
        status, body = http_json(
            base + "/v1/completions",
            key,
            openai_payload(prompt, n_predict, model),
            timeout=timeout,
        )
        path = "/v1/completions"
    else:
        path = "/completion"
    if status == 401:
        raise RuntimeError(
            "AccessPoint rejected the API key (HTTP 401). "
            "Source ~/.config/hga-qwen38/api.env or pass HGA_API_KEY."
        )
    if not (200 <= status < 300):
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        message = err.get("message") if isinstance(err, dict) else body.get("error")
        raise RuntimeError(f"HTTP {status} {path}: {message or body}")
    return path, body


def print_api_summary(record: dict[str, Any]) -> None:
    perf = record.get("perf") or {}
    llama = record.get("llama") or {}
    print("", flush=True)
    print("=" * 64, flush=True)
    print("AccessPoint 8K measurements  (live API, no new llama process)", flush=True)
    print("=" * 64, flush=True)
    print(f"  url     : {record.get('url')}", flush=True)
    print(f"  path    : {record.get('path')}", flush=True)
    print(
        f"  spec    : K={llama.get('spec_k')}  ctx={llama.get('ctx')}  "
        f"pid={llama.get('pid')}",
        flush=True,
    )
    print(
        f"  prefill : {perf.get('prefill_tokens')} tokens,  "
        f"{perf.get('prefill_ms')} ms,  {perf.get('prefill_tok_s')} tok/s  "
        f"cache_n={perf.get('cache_n')}",
        flush=True,
    )
    print(
        f"  generate: {perf.get('generate_tokens')} tokens,  "
        f"{perf.get('generate_ms')} ms,  {perf.get('generate_tok_s')} tok/s",
        flush=True,
    )
    latency = record.get("latency") or {}
    if latency.get("inference_ms") is not None:
        print(
            f"  inference: {latency['inference_ms']:.1f} ms  "
            "(prefill + generation from llama timings)",
            flush=True,
        )
    if latency.get("wall_ms") is not None:
        print(
            f"  wall    : {latency['wall_ms']:.1f} ms  (HTTP request)",
            flush=True,
        )
    if record.get("passed") is True:
        print("  result  : PASS", flush=True)
    elif record.get("passed") is False:
        print("  result  : FAIL", flush=True)
        for err in record.get("errors") or []:
            print(f"           - {err}", flush=True)
    print("=" * 64, flush=True)


def run_api(args: argparse.Namespace) -> int:
    bench_8k.assert_can_load_model()
    key = load_api_key()
    if not key:
        print(
            "HGA_API_KEY is not set and ~/.config/hga-qwen38/api-key is missing.",
            file=sys.stderr,
        )
        return 1
    try:
        base = pick_live_url(args.url, key, args.wait)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    llama = live_llama()
    prompt = build_api_prompt(stable=args.stable_prompt)
    n_predict = int(args.n_predict)
    print(
        f"==> {SUITE_NAME}: ~{bench_8k.PROMPT_TOKENS_TARGET} prefill + "
        f"{n_predict} generate  url={base}  spec_k={llama.get('spec_k')}",
        flush=True,
    )
    print(
        "==> occupies the single AccessPoint slot until this request finishes",
        flush=True,
    )
    t0 = time.time()
    try:
        path, body = post_completion(
            base, key, prompt, n_predict, args.model, args.timeout
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    wall_s = time.time() - t0
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    perf = perf_from_timings(timings)
    baseline = bench_8k.load_baseline()
    record: dict[str, Any] = {
        "suite": SUITE_NAME,
        "passed": False,
        "errors": [],
        "url": base,
        "path": path,
        "model": args.model,
        "wall_s": round(wall_s, 3),
        "fingerprint": bench_8k.fingerprint(),
        "git_commit": bench_8k.git_commit(),
        "llama": llama,
        "parameters": {
            "n_predict": n_predict,
            "stable_prompt": bool(args.stable_prompt),
            "cache_prompt": False,
            "ignore_eos": True,
            "prompt_sentences": bench_8k.PROMPT_SENTENCES,
        },
        "perf": perf,
        "timings": timings,
        "gates": baseline.get("gates"),
        "reference": baseline.get("reference"),
    }
    record["latency"] = bench_8k.latency_breakdown(record)
    errors: list[str] = []
    if not timings:
        errors.append("response had no llama timings (cannot measure prefill/generate)")
    errors.extend(evaluate_cache(perf))
    errors.extend(bench_8k.evaluate_gates(perf, baseline))
    record["errors"] = errors
    record["passed"] = not errors
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(bench_8k.format_measure_line(perf), flush=True)
    print_api_summary(record)
    print(f"json             {json_path}", flush=True)
    return 0 if record["passed"] else 1


def self_test() -> int:
    sample = {
        "cache_n": 0,
        "prompt_n": 8011,
        "prompt_ms": 16022.0,
        "prompt_per_second": 500.0,
        "predicted_n": 64,
        "predicted_ms": 2128.0,
        "predicted_per_second": 30.08,
    }
    perf = perf_from_timings(sample)
    assert perf["prefill_tokens"] == 8011, perf
    assert perf["generate_tokens"] == 64, perf
    assert perf["prefill_tok_s"] == 500.0, perf
    assert abs(float(perf["generate_tok_s"]) - 30.08) < 0.01, perf
    derived = perf_from_timings({"prompt_n": 100, "prompt_ms": 200.0, "predicted_n": 10, "predicted_ms": 500.0})
    assert derived["prefill_tok_s"] == 500.0, derived
    assert derived["generate_tok_s"] == 20.0, derived

    a = build_api_prompt()
    b = build_api_prompt()
    body = bench_8k.build_prompt()
    assert a != b
    assert a.endswith(body) or body in a
    assert body in a and body in b
    assert bench_8k.PROMPT_SENTENCE * bench_8k.PROMPT_SENTENCES in a
    stable = build_api_prompt(stable=True)
    assert stable == body

    baseline = bench_8k.load_baseline()
    assert bench_8k.evaluate_gates(
        {
            "prefill_tokens": 8011,
            "prefill_tok_s": 400.0,
            "generate_tokens": 64,
            "generate_tok_s": 12.0,
        },
        baseline,
    ) == []
    assert evaluate_cache({"cache_n": 0}) == []
    assert evaluate_cache({"cache_n": 8000})

    url = default_url()
    assert url.startswith("http://"), url
    assert normalize_base("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    urls = candidate_urls("http://127.0.0.1:8080")
    assert urls[0] == "http://127.0.0.1:8080", urls
    assert "http://127.0.0.1:8081" in urls, urls
    print("self-test OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Measure ~8K prefill + 64-token generate on the live AccessPoint. "
            "Never starts llama-server."
        )
    )
    p.add_argument(
        "--url",
        default=default_url(),
        help="AccessPoint base URL (gateway, not a new server)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="profile id for /v1/completions fallback",
    )
    p.add_argument(
        "--n-predict",
        type=int,
        default=bench_8k.GEN_TOKENS,
        help="generated tokens (default: same as bench_8k.py)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP timeout in seconds for the 8K request",
    )
    p.add_argument(
        "--wait",
        type=int,
        default=300,
        help="Seconds to wait if the live server is still loading the model",
    )
    p.add_argument(
        "--json",
        default="/tmp/hga_prefill-8k-api.json",
        help="Where to write the result JSON",
    )
    p.add_argument(
        "--stable-prompt",
        action="store_true",
        help=(
            "use the exact bench_8k.py prompt (may cache-hit on a re-run)"
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser/gate checks without calling the API",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    return run_api(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
