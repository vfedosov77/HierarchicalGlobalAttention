#!/usr/bin/env python3
"""Isolate which cd35566 prefill details still matter on this 16 GB host.

Each variant restores one (or a combined) detail from the CPU-HGA snapshot
onto the current VRAM16 binary. Defaults match the live AccessPoint algorithm
except ubatch=512 (the 8K bench shape) so a single detail can move tok/s.

Run only from Inference/Qwen3_8_27B_VRAM16GB after rebuilding. Stops nothing
itself; the GPU must be free.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_hga.sh"
OUT_DIR = ROOT / "baselines" / "vram16" / "ablation"
MODEL = os.path.expanduser(
    os.environ.get(
        "HGA_MODEL",
        "~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf",
    )
)
PROMPT_PREFIX = "Write a one-paragraph summary of the following notes.\n\n"
PROMPT_SENTENCE = (
    "Hierarchical global attention keeps a sink window, a local window, "
    "and routed mid-context chunks of keys scored by mixed-RoPE summaries. "
)

PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r".*?([\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)"
    r".*?([\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
CPU_STAGE_RE = re.compile(
    r"hga-gpu: CPU stage 16 layers ([\d.]+) ms route=([\d.]+) pack=([\d.]+)"
    r".*?keys=(\d+).*?n_q=(\d+)"
)
UNITED_RE = re.compile(r"hga-gpu: PREFILL united")
SEGMENTED_RE = re.compile(r"hga-gpu: PREFILL(?! united)")
CPU_PROF_RE = re.compile(r"hga-prof prefill chunk")
GRAPH_RE = re.compile(r"PREFILL after sched_reserve")
FAIL_SNIPPETS = (
    "out of memory",
    "failed to allocate",
    "cuda error",
    "failed to load model",
    "error loading",
    "Aborted",
    "SIGSEGV",
    "Segmentation fault",
)

BASE_ENV = {
    "HGA_GPU_PREFILL": "1",
    "HGA_GPU_PREFILL_UNITED": "1",
    "HGA_KEEP_FIRST": "2",
    "HGA_KEEP_LAST": "7",
    "HGA_ROUTE_FLOORS": "1",
    "HGA_OLD_TOPK": "0",
    "HGA_UNION_MULT": "3",
    "HGA_GPU_PREFILL_MIN_KEYS": "1552",
    "HGA_GPU_PREFILL_MAX_KEYS": "2560",
    "HGA_UBATCH": "512",
    "HGA_PREFILL_UBATCH": "512",
    "HGA_PREFILL_STREAM_ASYNC": "0",
    "HGA_PREFILL_STREAM_PACED": "0",
    "HGA_PREFILL_K_TILES": "0",
    "HGA_SPEC": "0",
    "HGA_N": "16",
    "HGA_THREADS": "12",
    "HGA_THREADS_BATCH": "1",
    "HGA_L2_OFF": "1",
    "HGA_LOAD_MODE": "none",
    "HGA_NUMA": "0",
    "HGA_PIN_CHECK": "1",
    "GGML_CUDA_DISABLE_GRAPHS": "1",
    "HGA_STREAM_ASYNC": "1",
    "HGA_STREAM_PACED": "1",
    "HGA_VERIFY_STREAMS": "2",
    "HGA_EXTRA": "--ignore-eos",
}

# One restored cd35566 detail each, plus a combined old-path control.
VARIANTS: list[dict[str, Any]] = [
    {
        "id": "A_current",
        "detail": "control: GPU united INT8, keep_last=7, floors on, ubatch=512",
        "env": {},
    },
    {
        "id": "B_cpu",
        "detail": "CPU HGA flash (cd35566 attention kernel)",
        "env": {"HGA_GPU_PREFILL": "0"},
    },
    {
        "id": "C_keep4",
        "detail": "keep_last=4 (cd35566 local window)",
        "env": {"HGA_KEEP_LAST": "4"},
    },
    {
        "id": "D_floors0",
        "detail": "routing floors/caps off (no min 3/6, no 20/32 cap)",
        "env": {"HGA_ROUTE_FLOORS": "0"},
    },
    {
        "id": "E_oldtopk",
        "detail": "cd35566 top-k formula (frac * n_closed, no floors)",
        "env": {"HGA_OLD_TOPK": "1"},
    },
    {
        "id": "F_nounited",
        "detail": "GPU prefill but segmented per 64-token chunk (no union image)",
        "env": {"HGA_GPU_PREFILL_UNITED": "0"},
    },
    {
        "id": "G_nomin",
        "detail": "no 1552-key graph pad (HGA_GPU_PREFILL_MIN_KEYS=0)",
        "env": {"HGA_GPU_PREFILL_MIN_KEYS": "0"},
    },
    {
        "id": "H_ub256",
        "detail": "API ubatch 256 (current AccessPoint)",
        "env": {"HGA_UBATCH": "256", "HGA_PREFILL_UBATCH": "256"},
    },
    {
        "id": "J_oldcombo",
        "detail": "CPU flash + keep_last=4 + old top-k (closest cd35566)",
        "env": {
            "HGA_GPU_PREFILL": "0",
            "HGA_KEEP_LAST": "4",
            "HGA_OLD_TOPK": "1",
            "HGA_ROUTE_FLOORS": "0",
        },
    },
]

SUITES = {
    "2k": {"sentences": 71, "ctx": 4096, "target_tokens": 2000},
    "8k": {"sentences": 285, "ctx": 8192, "target_tokens": 8000},
    "32k": {"sentences": 1140, "ctx": 32768, "target_tokens": 32000},
}


def gpu_compute_pids() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]


def failure_line(text: str) -> str | None:
    low = text.lower()
    for snippet in FAIL_SNIPPETS:
        if snippet.lower() in low:
            idx = low.find(snippet.lower())
            start = max(0, idx - 80)
            end = min(len(text), idx + 160)
            return " ".join(text[start:end].split())
    return None


def parse_log(text: str) -> dict[str, Any]:
    prompt = None
    for match in PROMPT_RE.finditer(text):
        prompt = {
            "ms": float(match.group(1)),
            "tokens": int(match.group(2)),
            "tok_s": float(match.group(3)),
        }
    gen = None
    for match in EVAL_RE.finditer(text):
        gen = {
            "ms": float(match.group(1)),
            "tokens": int(match.group(2)),
            "tok_s": float(match.group(3)),
        }
    stages = CPU_STAGE_RE.findall(text)
    stage_ms = [float(row[0]) for row in stages]
    route_ms = [float(row[1]) for row in stages]
    pack_ms = [float(row[2]) for row in stages]
    keys = [int(row[3]) for row in stages]
    path = "cpu"
    if UNITED_RE.search(text):
        path = "gpu-united"
    elif "hga_gpu_stage_cpu" in text or SEGMENTED_RE.search(text):
        path = "gpu-segmented"
    return {
        "prefill": prompt,
        "generate": gen,
        "path": path,
        "cpu_stage_count": len(stages),
        "cpu_stage_ms_max": max(stage_ms) if stage_ms else None,
        "cpu_route_ms_max": max(route_ms) if route_ms else None,
        "cpu_pack_ms_max": max(pack_ms) if pack_ms else None,
        "keys_max": max(keys) if keys else None,
        "prefill_graph_reserves": len(GRAPH_RE.findall(text)),
        "cpu_prof_chunks": len(CPU_PROF_RE.findall(text)),
        "failure": failure_line(text),
    }


def build_prompt(sentences: int) -> str:
    return PROMPT_PREFIX + (PROMPT_SENTENCE * sentences)


def run_variant(
    suite: str,
    variant: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    spec = SUITES[suite]
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["HGA_CTX"] = str(spec["ctx"])
    env["HGA_BATCH"] = str(spec["ctx"])
    env["CUDA_HOME"] = env.get("CUDA_HOME") or "/usr/local/cuda"
    if not Path(env["CUDA_HOME"], "lib64").is_dir() and Path("/usr/local/cuda/lib64").is_dir():
        env["CUDA_HOME"] = "/usr/local/cuda"
    llama_bin = str(ROOT / "third_party" / "llama.cpp" / "build" / "bin")
    cuda_lib = str(Path(env["CUDA_HOME"]) / "lib64")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{llama_bin}:" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = env["HGA_THREADS"]
    env["OMP_PLACES"] = "threads"
    env["OMP_PROC_BIND"] = "close"
    env["HGA_MODEL"] = MODEL

    prompt_path = OUT_DIR / f"prompt-{suite}.txt"
    prompt_path.write_text(build_prompt(spec["sentences"]), encoding="utf-8")
    env["HGA_PROMPT_FILE"] = str(prompt_path)

    log_path = OUT_DIR / f"{suite}-{variant['id']}.log"
    print(
        f"==> {suite} {variant['id']}: {variant['detail']}",
        flush=True,
    )
    print(
        "    "
        + " ".join(
            f"{k}={env[k]}"
            for k in (
                "HGA_GPU_PREFILL",
                "HGA_GPU_PREFILL_UNITED",
                "HGA_KEEP_LAST",
                "HGA_ROUTE_FLOORS",
                "HGA_OLD_TOPK",
                "HGA_UBATCH",
                "HGA_GPU_PREFILL_MIN_KEYS",
            )
        ),
        flush=True,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(LAUNCHER)],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout or ""
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        rc = -1
        timed_out = True
    wall_s = time.time() - t0
    log_path.write_text(output, encoding="utf-8", errors="replace")
    parsed = parse_log(output)
    prefill = parsed.get("prefill") or {}
    record = {
        "suite": suite,
        "id": variant["id"],
        "detail": variant["detail"],
        "env": variant["env"],
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": round(wall_s, 2),
        "log": str(log_path),
        **parsed,
        "prefill_tok_s": prefill.get("tok_s"),
        "prefill_tokens": prefill.get("tokens"),
        "prefill_ms": prefill.get("ms"),
    }
    tok = record["prefill_tok_s"]
    print(
        f"    rc={rc} wall={wall_s:.0f}s path={record['path']} "
        f"prefill={tok if tok is not None else 'NA'} tok/s "
        f"tokens={record['prefill_tokens']} fail={record['failure']!r}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("2k", "8k", "32k", "both"), default="both"
    )
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated variant ids, default all",
    )
    args = parser.parse_args()
    if not LAUNCHER.is_file():
        print(f"missing launcher {LAUNCHER}", file=sys.stderr)
        return 1
    if not Path(MODEL).is_file():
        print(f"missing model {MODEL}", file=sys.stderr)
        return 1
    leftover = gpu_compute_pids()
    if leftover:
        print(f"GPU still has compute pids {leftover}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    variants = [v for v in VARIANTS if not wanted or v["id"] in wanted]
    suites = ["8k", "2k"] if args.suite == "both" else [args.suite]
    # Short/long CPU controls unless --only was set.
    if args.suite in ("both", "32k") and not wanted:
        cpu_ids = {"A_current", "B_cpu", "J_oldcombo"}
    else:
        cpu_ids = {v["id"] for v in variants}

    results: list[dict[str, Any]] = []
    for suite in suites:
        for variant in variants:
            if suite in ("2k", "32k") and variant["id"] not in cpu_ids:
                continue
            results.append(run_variant(suite, variant, args.timeout))

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n==> {summary_path}")
    print(f"{'suite':<4} {'id':<14} {'tok/s':>8} {'path':<14} {'ms':>8} {'tokens':>7} detail")
    for row in results:
        tok = row.get("prefill_tok_s")
        tok_s = f"{tok:.1f}" if isinstance(tok, float) else "FAIL"
        ms = row.get("prefill_ms")
        ms_s = f"{ms:.0f}" if isinstance(ms, float) else "-"
        print(
            f"{row['suite']:<4} {row['id']:<14} {tok_s:>8} {row.get('path','?'):<14} "
            f"{ms_s:>8} {str(row.get('prefill_tokens') or '-'):>7} {row['detail']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
