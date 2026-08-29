#!/usr/bin/env python3
"""Run the Qwen3.8-27B 16 GB-VRAM local speed test.

The default 768/3200 GPU-prefill graph is reused only after its valid
historical range is saturated. Reusing the position-zero graph while that
boundary grows regresses retrieval quality.

Suite `prefill-2k-gen-64` starts the oracle launcher (`scripts/run_hga.sh`)
with default HGA packing and K=2 MTP speculative generate, and measures:

  * prefill of about 2000 prompt tokens
  * generate of 64 tokens on that prefilled context (draft-mtp, verify width 3)

This tree is retuned for a 16 GB GPU + 12 logical CPUs. It does not reconstruct
placement flags; those live in `run_hga.sh`. Load time is reported but is not a
throughput gate.

Examples:

    python3 tools/bench.py                  # local 16 GB host
    python3 tools/bench.py --no-hga-gpu-prefill  # CPU control
    python3 tools/bench.py --hga-kernel fused    # diagnostic A/B
    python3 tools/bench.py --self-test           # parser checks, no model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

QWEN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = QWEN_ROOT.parents[1]
DEFAULT_MODEL = os.environ.get(
    "HGA_MODEL",
    os.path.expanduser("~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"),
)
FORBIDDEN_HOSTS: set[str] = set()
SUITE_NAME = "prefill-2k-gen-64"

# Same filler as the 2026-08-21 2k table (71 sentences → 2001 tokens).
PROMPT_PREFIX = "Write a one-paragraph summary of the following notes.\n\n"
PROMPT_SENTENCE = (
    "Hierarchical global attention keeps a sink window, a local window, "
    "and routed mid-context chunks of keys scored by mixed-RoPE summaries. "
)
PROMPT_SENTENCES = 71
PROMPT_TOKENS_TARGET = 2000
GEN_TOKENS = 64
# Vram16Qwen38Profile: K=2 MTP proposals + target token.
MTP_K = 2
VERIFY_WIDTH = 3
# 2000 prompt + 64 gen does not fit in the launcher default ctx of 2048.
CTX = 4096
# 8 prefill exchange pairs; 2 remain streamed in verify.
PREFILL_EXCHANGE_PAIRS = 8
DECODE_STREAM_PAIRS = 2
DECODE_LEFTOVER_PAIRS = 6
PREFILL_PAIR_TAGS = (
    "0-32", "4-36", "8-40", "12-44", "16-48", "20-52", "24-56", "28-60",
)
DECODE_STREAM_TAGS = ("16-48", "24-56")
DECODE_LEFTOVER_TAGS = ("0-32", "4-36", "8-40", "12-44", "20-52", "28-60")
DECODE_STREAM_TAGS_3 = ("12-44", "16-48", "24-56")
DECODE_LEFTOVER_TAGS_3 = ("0-32", "4-36", "8-40", "20-52", "28-60")
HGA_KERNEL_MODES = ("tiled", "fused", "rows", "per-token")
HGA_STREAM_BLOCK_MODES = (0, 2, 3)
GPU_PREFILL_MIN_KEYS = 1552
GPU_PREFILL_MAX_KEYS = 3200
UBATCH = 768

PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r".*?([\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs"
    r".*?([\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
LOAD_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)
HGA_MEASURE_RE = re.compile(
    r"HGA_MEASURE\s+"
    r"prefill_tokens=(?P<prefill_tokens>\d+)\s+"
    r"prefill_ms=(?P<prefill_ms>[\d.]+)\s+"
    r"prefill_tok_s=(?P<prefill_tok_s>[\d.]+)\s+"
    r"generate_tokens=(?P<generate_tokens>\d+)\s+"
    r"generate_ms=(?P<generate_ms>[\d.]+)\s+"
    r"generate_tok_s=(?P<generate_tok_s>[\d.]+)"
)
FAIL_SNIPPETS = (
    "out of memory",
    "failed to allocate",
    "the prompt exceeds",
    "cuda error",
    "failed to load model",
    "error loading",
    "aborted",
    "invalid argument",
)


def configure_benchmark_hga_env(env: dict[str, str], kernel: str) -> None:
    """Select one reproducible HGA short-batch implementation for the suite."""
    if kernel not in HGA_KERNEL_MODES:
        raise ValueError(f"unknown HGA kernel mode: {kernel}")

    # Keep llama.cpp's outer pool away from HGA's internal OpenMP region.  Set
    # these explicitly: inherited shell values made otherwise identical bench
    # invocations silently exercise different kernels and thread layouts.
    env["HGA_THREADS"] = os.environ.get("HGA_THREADS", "12")
    env["HGA_THREADS_BATCH"] = "1"
    env["HGA_VERIFY_BATCH"] = "0" if kernel == "per-token" else "1"
    env["HGA_VERIFY_TILES"] = "0" if kernel in ("fused", "per-token") else "1"
    env["HGA_VERIFY_ROWS"] = "1" if kernel == "rows" else "0"
    env["HGA_STREAM_ASYNC"] = "1"
    env["HGA_L2_OFF"] = "1"
    env["HGA_LOAD_MODE"] = "none"


def baseline_path() -> Path:
    return QWEN_ROOT / "baselines" / "vram16" / "prefill-2k-gen-64.json"


def load_baseline() -> dict[str, Any]:
    path = baseline_path()
    if not path.is_file():
        raise FileNotFoundError(f"missing baseline {path}")
    return json.loads(path.read_text())


def build_prompt(sentences: int = PROMPT_SENTENCES) -> str:
    return PROMPT_PREFIX + (PROMPT_SENTENCE * sentences)


def parse_llama_perf(text: str) -> dict[str, Any]:
    """Extract llama.cpp perf_context lines. Missing fields stay None."""
    load_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    prefill_tokens: Optional[int] = None
    prefill_tok_s: Optional[float] = None
    generate_ms: Optional[float] = None
    generate_runs: Optional[int] = None
    generate_tok_s: Optional[float] = None

    for raw in text.splitlines():
        line = raw.strip()
        m = PROMPT_RE.search(line)
        if m:
            prefill_ms = float(m.group(1))
            prefill_tokens = int(m.group(2))
            prefill_tok_s = float(m.group(3))
            continue
        m = EVAL_RE.search(line)
        if m:
            generate_ms = float(m.group(1))
            generate_runs = int(m.group(2))
            generate_tok_s = float(m.group(3))
            continue
        m = LOAD_RE.search(line)
        if m and "prompt eval" not in line and load_ms is None:
            load_ms = float(m.group(1))

    return {
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens": prefill_tokens,
        "prefill_tok_s": prefill_tok_s,
        "generate_ms": generate_ms,
        "generate_tokens": generate_runs,
        "generate_tok_s": generate_tok_s,
    }


def _census_after(text: str, title: str) -> str:
    needle = f"census after {title}"
    idx = text.find(needle)
    if idx < 0:
        return ""
    return text[idx:idx + 3000]


def parse_pin_log(text: str) -> dict[str, Any]:
    """Read prefill vs decode exchange-pin transitions from llama logs."""
    pins: dict[str, Any] = {
        "prefill_pairs": None,
        "lm_head_cuda": False,
        "lm_head_on_host": "output.weight still on host" in text,
        "decode_stream_pairs": None,
        "decode_total_pairs": None,
        "leftover_pinned": None,
        "leftover_total": None,
        "leftover_skipped": "PIN-STEP skip leftover pairs" in text,
        "leftover_pushed": None,
        "leftover_resident_host": None,
        "leftover_exchange_host": None,
        "lm_head_buf": None,
    }
    m = re.search(r"PREFILL\s+exchange\s+(\d+)\s+pairs\s+lm_head CUDA", text)
    if m:
        pins["prefill_pairs"] = int(m.group(1))
        pins["lm_head_cuda"] = True
    if pins["lm_head_on_host"]:
        pins["lm_head_cuda"] = False
    m = re.search(r"VERIFY stream\s+(\d+)/(\d+)\s+pairs", text)
    if m:
        pins["decode_stream_pairs"] = int(m.group(1))
        pins["decode_total_pairs"] = int(m.group(2))
    m = re.search(r"PIN-STEP summary\s+pinned\s+(\d+)/(\d+)\s+leftover pairs", text)
    if m:
        pins["leftover_pinned"] = int(m.group(1))
        pins["leftover_total"] = int(m.group(2))
    leftover = _census_after(text, "leftover VERIFY pins")
    if leftover:
        m = re.search(r"PUSHED=(\d+)", leftover)
        if m:
            pins["leftover_pushed"] = int(m.group(1))
        m = re.search(r"resident\s+(\d+) GPU /\s+(\d+) host", leftover)
        if m:
            pins["leftover_resident_host"] = int(m.group(2))
        m = re.search(r"exchange\s+(\d+) GPU /\s+(\d+) host", leftover)
        if m:
            pins["leftover_exchange_host"] = int(m.group(2))
        m = re.search(r"lm_head\s+buf=(\S+)\s+(\S+)", leftover)
        if m:
            pins["lm_head_buf"] = m.group(1)
            if "host" in m.group(1).lower() or "host" in m.group(2).lower():
                pins["lm_head_cuda"] = False
                pins["lm_head_on_host"] = True
            elif "CUDA" in m.group(1) or "CUDA" in m.group(2):
                pins["lm_head_cuda"] = True
    return pins


def parse_split_log(text: str) -> dict[str, Any]:
    """Whole-layer vs split-FFN H2D metrics from HGA_STREAM_TIMING logs."""
    info: dict[str, Any] = {
        "enabled": False,
        "pairs": [],
        "h2d_mib_pass": None,
        "copy_ms_pass": None,
        "wait_ms_pass": None,
        "deadline_misses_pass": None,
        "h2d_gib_s": None,
        "fallback": None,
    }
    if "hga-split: fallback" in text:
        m = re.search(r"hga-split: fallback pair=(\S+) layer=(\S+) reason=(.+)", text)
        if m:
            info["fallback"] = {
                "pair": m.group(1),
                "layer": m.group(2),
                "reason": m.group(3).strip(),
            }
    pair_re = re.compile(
        r"hga-split: pair=(\S+) budget=([\d.]+)MiB coreA=([\d.]+) coreB=([\d.]+) "
        r"tile=([\d.]+) slots=(\d+)/(\d+)"
    )
    for m in pair_re.finditer(text):
        info["enabled"] = True
        info["pairs"].append(
            {
                "tag": m.group(1),
                "budget_mib": float(m.group(2)),
                "core_a_mib": float(m.group(3)),
                "core_b_mib": float(m.group(4)),
                "tile_mib": float(m.group(5)),
                "slots": int(m.group(6)),
                "n_tiles": int(m.group(7)),
            }
        )
    passes = list(
        re.finditer(
            r"hga-split: pass h2d=([\d.]+)MiB copy_ms=([\d.]+) wait_ms=([\d.]+) "
            r"deadline_misses=(\d+)(?: h2d_gib_s=([\d.]+))?",
            text,
        )
    )
    if passes:
        m = passes[-1]
        info["h2d_mib_pass"] = float(m.group(1))
        info["copy_ms_pass"] = float(m.group(2))
        info["wait_ms_pass"] = float(m.group(3))
        info["deadline_misses_pass"] = int(m.group(4))
        if m.group(5):
            info["h2d_gib_s"] = float(m.group(5))
    return info


def evaluate_pin_gates(pins: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    want = baseline.get("pins") or {}
    prefill_n = int(want.get("prefill_pairs", PREFILL_EXCHANGE_PAIRS))
    decode_n = int(want.get("decode_stream_pairs", DECODE_STREAM_PAIRS))
    leftover_n = int(want.get("leftover_pairs", DECODE_LEFTOVER_PAIRS))
    errors: list[str] = []
    if pins.get("prefill_pairs") != prefill_n:
        errors.append(
            f"prefill exchange pairs {pins.get('prefill_pairs')} != {prefill_n}"
        )
    if not pins.get("lm_head_cuda") or pins.get("lm_head_on_host"):
        errors.append("lm_head is not CUDA-resident")
    if pins.get("decode_stream_pairs") != decode_n:
        errors.append(
            f"decode stream pairs {pins.get('decode_stream_pairs')} != {decode_n}"
        )
    if pins.get("leftover_skipped"):
        errors.append("leftover VERIFY CUDA pin was skipped")
    if pins.get("leftover_pinned") != leftover_n or pins.get("leftover_total") != leftover_n:
        errors.append(
            f"leftover CUDA pin {pins.get('leftover_pinned')}/{pins.get('leftover_total')} "
            f"!= {leftover_n}/{leftover_n}"
        )
    if pins.get("leftover_pushed") != 0:
        errors.append(f"PUSHED={pins.get('leftover_pushed')} after leftover pin")
    if pins.get("leftover_resident_host") != 0:
        errors.append(
            f"{pins.get('leftover_resident_host')} resident tensors still on host after leftover pin"
        )
    if pins.get("leftover_exchange_host") != 0:
        errors.append(
            f"{pins.get('leftover_exchange_host')} exchange tensors still on host after leftover pin"
        )
    return errors


ALLOC_GRAPH_RE = re.compile(
    r"ubatch before alloc_graph.*?n_tok=(\d+).*?n_ubatch=(\d+).*?phase=(\d+)\s+ctx=(\d+)"
)
GRAPHS_REUSED_RE = re.compile(r"graphs reused\s*=\s*(\d+)", re.IGNORECASE)
N_DRAFTED_RE = re.compile(r"n_drafted\s*=\s*(\d+)", re.IGNORECASE)
N_ACCEPT_RE = re.compile(r"n_accept\s*=\s*(\d+)", re.IGNORECASE)
ACCEPT_PCT_RE = re.compile(r"accept\s*=\s*([\d.]+)\s*%", re.IGNORECASE)
SPEC_STEPS_RE = re.compile(r"hga-prof spec TOTAL steps=(\d+)")
SPEC_TOTAL_MS_RE = re.compile(
    r"hga-prof spec TOTAL steps=\d+\s+generate=([\d.]+)\s*ms"
)
SPEC_STEP_BATCH_RE = re.compile(r"n_tgt_batch=(\d+)")
DECODED_RE = re.compile(r"decoded\s+(\d+)\s+tokens", re.IGNORECASE)


def parse_graph_log(text: str) -> dict[str, Any]:
    """Count ggml plan builds. alloc_graph runs only when can_reuse failed."""
    prefill: list[dict[str, int]] = []
    decode: list[dict[str, int]] = []
    for m in ALLOC_GRAPH_RE.finditer(text):
        rec = {
            "n_tok": int(m.group(1)),
            "n_ubatch": int(m.group(2)),
            "phase": int(m.group(3)),
            "ctx": int(m.group(4)),
        }
        # ctx!=0 is the MTP draft context; the gate is the target graph.
        if rec["ctx"] != 0:
            continue
        if rec["phase"] == 1:
            prefill.append(rec)
        elif rec["phase"] in (2, 3):
            decode.append(rec)
    reused = None
    m = GRAPHS_REUSED_RE.search(text)
    if m:
        reused = int(m.group(1))
    return {
        "prefill_builds": len(prefill),
        "decode_builds": len(decode),
        "prefill_extra_rebuilds": max(0, len(prefill) - 1),
        "decode_extra_rebuilds": max(0, len(decode) - 1),
        "prefill_allocs": prefill,
        "decode_allocs": decode,
        "graphs_reused": reused,
        "graphs_disabled": "graphs_disabled=1" in text or "graphs_disabled=true" in text.lower(),
    }


def evaluate_graph_gates(graphs: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    want = baseline.get("graphs") or {}
    prefill_max = int(want.get("prefill_builds_max", 1))
    decode_max = int(want.get("decode_builds_max", 1))
    errors: list[str] = []
    pb = graphs.get("prefill_builds")
    db = graphs.get("decode_builds")
    if not pb:
        errors.append("no PREFILL ggml graph build (missing alloc_graph phase=1)")
    elif pb > prefill_max:
        errors.append(
            f"PREFILL graph built {pb} time(s), "
            f"exceeding the safe maximum {prefill_max}"
        )
    if not db:
        errors.append("no DECODE/VERIFY ggml graph build (missing alloc_graph phase=2/3)")
    elif db > decode_max:
        errors.append(
            f"DECODE graph built {db} time(s), "
            f"exceeding the maximum {decode_max}"
        )
    if not graphs.get("graphs_disabled"):
        errors.append("CUDA graphs were not disabled (need graphs_disabled=1)")
    verify_w = int(want.get("verify_width", VERIFY_WIDTH))
    decode_widths = sorted({a["n_ubatch"] for a in graphs.get("decode_allocs") or []})
    if decode_widths and decode_widths != [verify_w]:
        errors.append(
            f"VERIFY graph width {decode_widths} != [{verify_w}] (K+1 speculative batch)"
        )
    reused = graphs.get("graphs_reused")
    reused_max = want.get("reused_max")
    if reused_max is not None and reused is not None and reused > int(reused_max):
        errors.append(
            f"graphs reused {reused} > {reused_max} "
            "(looks like one-token greedy, not speculative verify)"
        )
    return errors


def parse_spec_log(text: str) -> dict[str, Any]:
    """Draft/accept counters from llama-speculative-simple."""
    drafted = None
    accepted = None
    accept_pct = None
    m = N_DRAFTED_RE.search(text)
    if m:
        drafted = int(m.group(1))
    m = N_ACCEPT_RE.search(text)
    if m:
        accepted = int(m.group(1))
    m = ACCEPT_PCT_RE.search(text)
    if m:
        accept_pct = float(m.group(1))
    steps = None
    m = SPEC_STEPS_RE.search(text)
    if m:
        steps = int(m.group(1))
    batches = [int(x) for x in SPEC_STEP_BATCH_RE.findall(text)]
    gen_ms = None
    m = SPEC_TOTAL_MS_RE.search(text)
    if m:
        gen_ms = float(m.group(1))
    decoded = None
    m = DECODED_RE.search(text)
    if m:
        decoded = int(m.group(1))
    return {
        "k": MTP_K,
        "n_drafted": drafted,
        "n_accept": accepted,
        "accept_pct": accept_pct,
        "verify_steps": steps,
        "verify_batches": batches,
        "generate_ms": gen_ms,
        "decoded": decoded,
        "used": drafted is not None and drafted > 0,
    }


def fill_perf_from_spec(perf: dict[str, Any], spec: dict[str, Any]) -> None:
    """llama-speculative-simple often omits the eval-time line; use spec totals."""
    if perf.get("generate_tok_s") is not None:
        return
    n = spec.get("decoded")
    if n is None:
        n = spec.get("n_accept")
    ms = spec.get("generate_ms")
    if n and ms and ms > 0:
        perf["generate_tokens"] = n
        perf["generate_ms"] = ms
        perf["generate_tok_s"] = round(1000.0 * n / ms, 2)


def evaluate_spec_gates(spec: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    want = baseline.get("spec") or {}
    errors: list[str] = []
    if not spec.get("used"):
        errors.append(
            "speculative decoding was not used (no n_drafted>0); "
            "need HGA_SPEC=2 / llama-speculative-simple"
        )
        return errors
    if spec.get("n_accept") is None:
        errors.append("missing n_accept from speculative log")
    k_want = int(want.get("k", MTP_K))
    if spec.get("k") != k_want:
        errors.append(f"spec K={spec.get('k')} != {k_want}")
    return errors


ALLOC_OOM_EVENT_RE = re.compile(
    r"hga-alloc-oom:\s+(\d+)\s+module=(\S+)\s+who=(\S+)\s+mib=([\d.]+)\s+"
    r"free_before=([\d.]+)\s+free_after=([\d.]+)\s+ok=(\d+)"
)
ALLOC_OOM_LIVE_RE = re.compile(
    r"hga-alloc-oom:\s+module=(\S+)\s+n=(\d+)\s+mib=([\d.]+)"
)
ALLOC_LINE_RE = re.compile(
    r"hga-alloc:\s+#(\d+)\s+module=(\S+)\s+who=(\S+)\s+mib=([\d.]+)\s+"
    r"free_before=([\d.]+)\s+free_after=([\d.]+)\s+ok=(\d+)"
)
PIN_STEP_NEED_RE = re.compile(
    r"PIN-STEP\s+\d+/\d+\s+leftover pair\s+(\S+)\s+need=([\d.]+)\s*MiB\s+free=([\d.]+)\s*MiB"
)
PIN_STEP_OK_RE = re.compile(
    r"PIN-STEP\s+\d+/\d+\s+leftover pair\s+(\S+)\s+CUDA alloc\s+([\d.]+)\s*MiB ok\s+"
    r"free_after=([\d.]+)"
)
STREAM_ALLOC_RE = re.compile(
    r"stream\s+(\S+)\s+CUDA alloc\s+([\d.]+)\s*MiB\s+(ok|failed)"
)
STREAM_FREE_RE = re.compile(
    r"CUDA memory after stream slot alloc(?: FAIL)?:\s+free\s+([\d.]+)"
)


def parse_oom_alloc_dump(text: str) -> dict[str, Any]:
    """Prefer the C++ ring dump; otherwise reconstruct from PIN-STEP / stream logs."""
    events: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    why = None
    blocks = list(re.finditer(
        r"hga-alloc-oom: BEGIN why=(\S+)(.*?)hga-alloc-oom: END",
        text, re.S,
    ))
    chunk = blocks[-1].group(0) if blocks else text
    if blocks:
        why = blocks[-1].group(1)
    if why is None and re.search(r"out of memory|cudaMalloc failed|alloc.*FAIL", text, re.I):
        why = "cuda-oom"
    for m in ALLOC_OOM_EVENT_RE.finditer(chunk):
        events.append({
            "seq": int(m.group(1)),
            "module": m.group(2),
            "who": m.group(3),
            "mib": float(m.group(4)),
            "free_before": float(m.group(5)),
            "free_after": float(m.group(6)),
            "ok": int(m.group(7)),
        })
    for m in ALLOC_OOM_LIVE_RE.finditer(chunk):
        live.append({
            "module": m.group(1),
            "n": int(m.group(2)),
            "mib": float(m.group(3)),
        })
    if not events:
        for m in ALLOC_LINE_RE.finditer(text):
            events.append({
                "seq": int(m.group(1)),
                "module": m.group(2),
                "who": m.group(3),
                "mib": float(m.group(4)),
                "free_before": float(m.group(5)),
                "free_after": float(m.group(6)),
                "ok": int(m.group(7)),
            })
    if not events:
        events = _reconstruct_allocs_from_swap_log(text)
    return {"why": why, "events": events[-20:], "live": live}


def _reconstruct_allocs_from_swap_log(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seq = 0
    pending_free: Optional[float] = None
    for raw in text.splitlines():
        m = STREAM_FREE_RE.search(raw)
        if m:
            pending_free = float(m.group(1))
        m = STREAM_ALLOC_RE.search(raw)
        if m:
            seq += 1
            mib = float(m.group(2))
            ok = 1 if m.group(3) == "ok" else 0
            fa = pending_free if pending_free is not None else 0.0
            fb = fa + mib if ok else fa
            events.append({
                "seq": seq,
                "module": "stream-slot",
                "who": m.group(1),
                "mib": mib,
                "free_before": fb,
                "free_after": fa,
                "ok": ok,
            })
            continue
        m = PIN_STEP_NEED_RE.search(raw)
        if m:
            seq += 1
            events.append({
                "seq": seq,
                "module": "leftover-pin",
                "who": m.group(1),
                "mib": float(m.group(2)),
                "free_before": float(m.group(3)),
                "free_after": float(m.group(3)),
                "ok": 0,
            })
            continue
        m = PIN_STEP_OK_RE.search(raw)
        if m and events and events[-1]["module"] == "leftover-pin" and events[-1]["who"] == m.group(1):
            events[-1]["ok"] = 1
            events[-1]["mib"] = float(m.group(2))
            events[-1]["free_after"] = float(m.group(3))
    return events


def print_oom_allocs(dump: dict[str, Any]) -> None:
    events = dump.get("events") or []
    why = dump.get("why")
    title = "last 20 CUDA allocs"
    if why:
        title = f"last 20 CUDA allocs  WHY={why}"
    print("", flush=True)
    print("********** CUDA ALLOCATION LEDGER (last 20) **********", flush=True)
    print(f"  allocs  : {title}", flush=True)
    if not events:
        print("           (none parsed from llama log)", flush=True)
        return
    print(
        f"           {'#':>4}  {'module':<16}  {'who':<16}  {'MiB':>8}  "
        f"{'free_before':>11}  {'free_after':>10}  ok",
        flush=True,
    )
    for e in events[-20:]:
        print(
            f"           {e.get('seq', 0):4d}  {str(e.get('module', '')):<16}  "
            f"{str(e.get('who', '')):<16}  {float(e.get('mib') or 0):8.2f}  "
            f"{float(e.get('free_before') or 0):11.0f}  "
            f"{float(e.get('free_after') or 0):10.0f}  {e.get('ok')}",
            flush=True,
        )
    live = dump.get("live") or []
    if live:
        print("           live by module:", flush=True)
        for L in live:
            print(
                f"             {L['module']:<16}  n={L['n']}  {L['mib']:.1f} MiB",
                flush=True,
            )
    print("********** END CUDA ALLOCATION LEDGER **********", flush=True)


def failure_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        low = line.lower()
        if any(s in low for s in FAIL_SNIPPETS):
            return line.strip()
    return None


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def cpuinfo_text() -> str:
    path = Path("/proc/cpuinfo")
    return path.read_text() if path.is_file() else ""


def nvidia_query() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (out.stdout or "").strip()


def fingerprint() -> dict[str, Any]:
    return {
        "hostname": hostname(),
        "cpu": next(
            (ln.split(":")[-1].strip() for ln in cpuinfo_text().splitlines()
             if "model name" in ln.lower()),
            "",
        ),
        "gpu": nvidia_query(),
    }


def assert_can_load_model() -> None:
    host = hostname()
    if host in FORBIDDEN_HOSTS:
        raise SystemExit(
            f"refusing to load Qwen3.8-27B on {host}."
        )
    gpu = nvidia_query()
    if not gpu:
        raise SystemExit(
            f"refusing to load Qwen3.8-27B on {host}: no NVIDIA GPU reported."
        )


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kill_leftover_llama() -> None:
    for name in ("llama-completion", "llama-speculative-simple", "llama-cli"):
        subprocess.run(
            ["pkill", "-f", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1)


def gpu_compute_pids() -> list[str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]


def parse_measure_line(text: str) -> Optional[dict[str, Any]]:
    match = HGA_MEASURE_RE.search(text)
    if not match:
        return None
    return {
        "prefill_tokens": int(match.group("prefill_tokens")),
        "prefill_ms": float(match.group("prefill_ms")),
        "prefill_tok_s": float(match.group("prefill_tok_s")),
        "generate_tokens": int(match.group("generate_tokens")),
        "generate_ms": float(match.group("generate_ms")),
        "generate_tok_s": float(match.group("generate_tok_s")),
    }


def format_measure_line(perf: dict[str, Any]) -> str:
    return (
        "HGA_MEASURE "
        f"prefill_tokens={perf.get('prefill_tokens')} "
        f"prefill_ms={perf.get('prefill_ms')} "
        f"prefill_tok_s={perf.get('prefill_tok_s')} "
        f"generate_tokens={perf.get('generate_tokens')} "
        f"generate_ms={perf.get('generate_ms')} "
        f"generate_tok_s={perf.get('generate_tok_s')}"
    )


def print_local_summary(record: dict[str, Any]) -> None:
    """Always print tok/s on the machine that launched the script."""
    perf = record.get("perf") or {}
    print("", flush=True)
    print("=" * 64, flush=True)
    print("local 16 GB measurements  (this PC's console)", flush=True)
    print("=" * 64, flush=True)
    print(
        f"  prefill : {perf.get('prefill_tokens')} tokens,  "
        f"{perf.get('prefill_ms')} ms,  {perf.get('prefill_tok_s')} tok/s",
        flush=True,
    )
    print(
        f"  generate: {perf.get('generate_tokens')} tokens,  "
        f"{perf.get('generate_ms')} ms,  {perf.get('generate_tok_s')} tok/s",
        flush=True,
    )
    load = perf.get("load_ms")
    if load is not None:
        print(f"  load    : {load:.1f} ms  (excluded from speed gates)", flush=True)
    pins = record.get("pins") or {}
    if pins:
        print(
            f"  pins    : prefill stream {pins.get('prefill_pairs')} pairs; "
            f"decode stream {pins.get('decode_stream_pairs')}/"
            f"{pins.get('decode_total_pairs')} ; "
            f"leftover GPU {pins.get('leftover_pinned')}/{pins.get('leftover_total')}",
            flush=True,
        )
        print(
            f"  lm_head : {'CUDA' if pins.get('lm_head_cuda') and not pins.get('lm_head_on_host') else 'NOT CUDA'}"
            + (f"  buf={pins['lm_head_buf']}" if pins.get("lm_head_buf") else ""),
            flush=True,
        )
    graphs = record.get("graphs") or {}
    if graphs:
        print(
            f"  graphs  : prefill builds {graphs.get('prefill_builds')} "
            f"(extra {graphs.get('prefill_extra_rebuilds')}); "
            f"decode builds {graphs.get('decode_builds')} "
            f"(extra {graphs.get('decode_extra_rebuilds')}); "
            f"reused {graphs.get('graphs_reused')}",
            flush=True,
        )
    spec = record.get("spec") or {}
    if spec:
        print(
            f"  spec    : K={spec.get('k')}  drafted={spec.get('n_drafted')}  "
            f"accepted={spec.get('n_accept')}  accept={spec.get('accept_pct')}%  "
            f"verify_steps={spec.get('verify_steps')}",
            flush=True,
        )
    oom = record.get("oom_allocs") or {}
    print_oom_allocs(oom)
    passed = record.get("passed")
    if passed is True:
        print("  result  : PASS", flush=True)
    elif passed is False:
        print("  result  : FAIL", flush=True)
        for err in record.get("errors") or []:
            print(f"           - {err}", flush=True)
    print("=" * 64, flush=True)


def evaluate_gates(perf: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    gates = baseline["gates"]
    errors: list[str] = []

    pt = perf.get("prefill_tokens")
    if pt is None:
        errors.append("missing prefill token count (prompt eval line)")
    elif not (gates["prompt_tokens_min"] <= pt <= gates["prompt_tokens_max"]):
        errors.append(
            f"prefill tokens {pt} outside "
            f"[{gates['prompt_tokens_min']}, {gates['prompt_tokens_max']}]"
        )

    gt = perf.get("generate_tokens")
    if gt is None:
        errors.append("missing generate run count (eval time line)")
    elif gt < gates["gen_tokens_min"]:
        errors.append(
            f"generate tokens {gt} < {gates['gen_tokens_min']}"
        )

    pts = perf.get("prefill_tok_s")
    if pts is None:
        errors.append("missing prefill tok/s")
    elif pts < gates["prefill_tok_s_min"]:
        errors.append(
            f"prefill {pts:.2f} tok/s < floor {gates['prefill_tok_s_min']}"
        )

    gts = perf.get("generate_tok_s")
    if gts is None:
        errors.append("missing generate tok/s")
    elif gts < gates["generate_tok_s_min"]:
        errors.append(
            f"generate {gts:.2f} tok/s < floor {gates['generate_tok_s_min']}"
        )
    return errors


def print_report(record: dict[str, Any]) -> None:
    perf = record["perf"]
    print("", flush=True)
    print(f"suite            {record['suite']}", flush=True)
    print(f"host             {record['fingerprint']['hostname']}", flush=True)
    print(f"gpu              {record['fingerprint']['gpu']}", flush=True)
    load = perf.get("load_ms")
    print(
        f"load             {load:.1f} ms" if load is not None else "load             n/a",
        flush=True,
    )
    print(
        f"prefill          {perf.get('prefill_tokens')} tokens  "
        f"{perf.get('prefill_ms')} ms  "
        f"{perf.get('prefill_tok_s')} tok/s",
        flush=True,
    )
    print(
        f"generate         {perf.get('generate_tokens')} tokens  "
        f"{perf.get('generate_ms')} ms  "
        f"{perf.get('generate_tok_s')} tok/s",
        flush=True,
    )
    spec = record.get("spec") or {}
    if spec:
        print(
            f"spec             K={spec.get('k')}  drafted={spec.get('n_drafted')}  "
            f"accepted={spec.get('n_accept')}  accept={spec.get('accept_pct')}%  "
            f"verify_steps={spec.get('verify_steps')}",
            flush=True,
        )
    ref = record.get("reference") or {}
    if ref:
        print(
            f"reference        prefill {ref.get('prefill_tok_s')} tok/s  "
            f"generate {ref.get('generate_tok_s')} tok/s  "
            f"({ref.get('source', '')})",
            flush=True,
        )
    if record["passed"]:
        print("result           PASS", flush=True)
    else:
        print("result           FAIL", flush=True)
        for err in record["errors"]:
            print(f"  - {err}", flush=True)


def run_local(args: argparse.Namespace) -> int:
    assert_can_load_model()
    baseline = load_baseline()
    stream_block = args.hga_stream_block
    verify_streams = args.hga_verify_streams
    stream_paced = (
        args.hga_stream_paced
        and not args.hga_stream_sync
        and args.hga_stream_chunk_mib == 0
        and stream_block == 0
        and verify_streams in (2, 3)
    )
    if stream_block:
        # Group counts, not layer-pair counts: the packed block is one stream
        # group and replaces stream_block of the default eight pairs.
        baseline["pins"] = dict(baseline.get("pins") or {})
        baseline["pins"].update({
            "prefill_pairs": 9 - stream_block,
            "decode_stream_pairs": 1,
            "leftover_pairs": 8 - stream_block,
        })
    elif verify_streams == 3:
        baseline["pins"] = dict(baseline.get("pins") or {})
        baseline["pins"].update({
            "decode_stream_pairs": 3,
            "leftover_pairs": 5,
        })
    model = Path(os.environ.get("HGA_MODEL", DEFAULT_MODEL))
    if not model.is_file():
        print(
            f"missing model {model} — copy Qwen3.8-27B-UD-Q4_K_M.gguf. "
            "See scripts/download_qwen38.sh.",
            file=sys.stderr,
        )
        return 1

    launcher = QWEN_ROOT / "scripts" / "run_hga.sh"
    if not os.access(launcher, os.X_OK):
        launcher.chmod(0o755)

    kill_leftover_llama()
    leftover = gpu_compute_pids()
    if leftover:
        print(
            f"GPU still has compute pids {leftover}; "
            "will not start a second 27B job.",
            file=sys.stderr,
        )
        return 1

    prompt_file = Path(args.prompt_file)
    prompt_file.write_text(build_prompt(), encoding="utf-8")
    log_path = Path(args.log)
    json_path = Path(args.json)

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    if not Path(env["CUDA_HOME"], "lib64").is_dir() and Path("/usr/local/cuda/lib64").is_dir():
        env["CUDA_HOME"] = "/usr/local/cuda"
    cuda_lib = str(Path(env["CUDA_HOME"]) / "lib64")
    llama_bin = str(QWEN_ROOT / "third_party" / "llama.cpp" / "build" / "bin")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{llama_bin}:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["HGA_CTX"] = str(CTX)
    env["HGA_BATCH"] = str(CTX)
    env["HGA_N"] = str(GEN_TOKENS)
    env["HGA_PROMPT_FILE"] = str(prompt_file)
    # llama-speculative-simple rejects -no-cnv (cli-only). --ignore-eos is common.
    env["HGA_EXTRA"] = "--ignore-eos"
    env["HGA_SPEC"] = str(MTP_K)
    configure_benchmark_hga_env(env, args.hga_kernel)
    env["OMP_PLACES"] = "threads"
    env["OMP_PROC_BIND"] = "close"
    env["HGA_NUMA"] = "0"
    env["HGA_UBATCH"] = str(UBATCH)
    env["HGA_PREFILL_UBATCH"] = str(UBATCH)
    env["HGA_PREFILL_K_TILES"] = "1" if args.hga_prefill_k_tiles else "0"
    env["HGA_GPU_PREFILL"] = "1" if args.hga_gpu_prefill else "0"
    env["HGA_GPU_PREFILL_MIN_KEYS"] = str(GPU_PREFILL_MIN_KEYS)
    env["HGA_GPU_PREFILL_MAX_KEYS"] = str(GPU_PREFILL_MAX_KEYS)
    env["HGA_PREFILL_STREAM_ASYNC"] = (
        "1"
        if (args.hga_prefill_stream_async or args.hga_prefill_stream_paced)
        and stream_block == 0
        else "0"
    )
    env["HGA_PREFILL_STREAM_PACED"] = (
        "1" if args.hga_prefill_stream_paced and stream_block == 0 else "0"
    )
    env["HGA_STREAM_CHUNK_MIB"] = str(args.hga_stream_chunk_mib)
    env["HGA_STREAM_PACED"] = "1" if stream_paced else "0"
    if args.hga_stream_sync:
        env["HGA_STREAM_ASYNC"] = "0"
        env["HGA_STREAM_TIMING"] = "1"
    else:
        env.pop("HGA_STREAM_TIMING", None)
    # GPU-resident except the exchange slots; lm_head stays CUDA. The default
    # streams 8 prefill pairs and 2 verify pairs; --hga-stream-block selects
    # the coarse packed A/B experiment instead.
    env.pop("HGA_STREAM_2", None)
    env.pop("HGA_STREAM_UNIFORM", None)
    env["HGA_VERIFY_STREAMS"] = str(verify_streams)
    env["HGA_STREAM_BLOCK"] = str(stream_block)
    env.pop("HGA_OT_CPU", None)
    env.pop("HGA_LMHEAD_CPU", None)
    env.pop("HGA_NO_RESIDENT_FFN", None)
    env["HGA_PIN_CHECK"] = "1"

    print(
        f"==> {SUITE_NAME}: ctx={CTX} n={GEN_TOKENS} "
        f"HGA_SPEC={MTP_K} verify_width={VERIFY_WIDTH} "
        f"hga_kernel={args.hga_kernel} prompt_file={prompt_file} model={model}",
        flush=True,
    )
    prefill_h2d = (
        "paced"
        if env["HGA_PREFILL_STREAM_PACED"] == "1"
        else "event-immediate"
        if env["HGA_PREFILL_STREAM_ASYNC"] == "1"
        else "callback"
    )
    print(
        "==> HGA short-batch: "
        f"batch={env['HGA_VERIFY_BATCH']} tiles={env['HGA_VERIFY_TILES']} "
        f"rows={env['HGA_VERIFY_ROWS']} internal_threads={env['HGA_THREADS']} "
        f"outer_batch_threads={env['HGA_THREADS_BATCH']} "
        f"prefill_k_tiles={env['HGA_PREFILL_K_TILES']} "
        f"gpu_prefill={env['HGA_GPU_PREFILL']} "
        f"prefill_h2d={prefill_h2d} "
        f"verify_h2d={'paced' if stream_paced else 'immediate'}",
        flush=True,
    )
    print(
        "==> pin plan: all layers CUDA except exchange; lm_head CUDA",
        flush=True,
    )
    if stream_block:
        n_left = 8 - stream_block
        last_b = 31 + stream_block
        print(
            f"==> PREFILL stream {1 + n_left} groups; packed "
            f"1:{stream_block}-32:{last_b} plus {n_left} single pairs",
            flush=True,
        )
        print(
            f"==> DECODE stream 1 packed group ({stream_block} layer pairs); "
            f"leftover pin CUDA {n_left} groups",
            flush=True,
        )
    else:
        stream_tags = DECODE_STREAM_TAGS_3 if verify_streams == 3 else DECODE_STREAM_TAGS
        leftover_tags = DECODE_LEFTOVER_TAGS_3 if verify_streams == 3 else DECODE_LEFTOVER_TAGS
        print(
            f"==> PREFILL stream {PREFILL_EXCHANGE_PAIRS} pairs "
            + " ".join(PREFILL_PAIR_TAGS),
            flush=True,
        )
        print(
            f"==> DECODE  stream {verify_streams} pairs "
            + " ".join(stream_tags)
            + " ; leftover pin CUDA "
            + " ".join(leftover_tags),
            flush=True,
        )
    print(
        "==> starting llama-speculative-simple locally "
        "(load, then ~2000-token prefill, then 64-token generate). "
        "tok/s will print on this console when it finishes.",
        flush=True,
    )
    t0 = time.time()
    proc = subprocess.run(
        [str(launcher)],
        cwd=str(QWEN_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    wall_s = time.time() - t0
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8", errors="replace")

    fail = failure_line(output)
    perf = parse_llama_perf(output)
    pins = parse_pin_log(output)
    split = parse_split_log(output)
    graphs = parse_graph_log(output)
    spec = parse_spec_log(output)
    fill_perf_from_spec(perf, spec)
    oom_allocs = parse_oom_alloc_dump(output)
    llama_bin_path = Path(llama_bin) / "llama-speculative-simple"
    if not llama_bin_path.is_file():
        llama_bin_path = Path(llama_bin) / "llama-completion"
    record: dict[str, Any] = {
        "suite": SUITE_NAME,
        "passed": False,
        "errors": [],
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 3),
        "fingerprint": fingerprint(),
        "model": str(model),
        "git_commit": git_commit(),
        "llama_sha256": file_sha256(llama_bin_path),
        "parameters": {
            "ctx": CTX,
            "ubatch": int(env.get("HGA_UBATCH", "768")),
            "n_predict": GEN_TOKENS,
            "hga_spec": MTP_K,
            "verify_width": VERIFY_WIDTH,
            "prompt_sentences": PROMPT_SENTENCES,
            "hga_levels": 2,
            "hga_i8": True,
            "hga_kernel": args.hga_kernel,
            "hga_verify_batch": env["HGA_VERIFY_BATCH"] == "1",
            "hga_verify_tiles": env["HGA_VERIFY_TILES"] == "1",
            "hga_verify_rows": env["HGA_VERIFY_ROWS"] == "1",
            "hga_prefill_k_tiles": env["HGA_PREFILL_K_TILES"] == "1",
            "hga_gpu_prefill": env["HGA_GPU_PREFILL"] == "1",
            "hga_gpu_prefill_min_keys": int(env["HGA_GPU_PREFILL_MIN_KEYS"]),
            "hga_gpu_prefill_max_keys": int(env["HGA_GPU_PREFILL_MAX_KEYS"]),
            "hga_prefill_stream_async": env["HGA_PREFILL_STREAM_ASYNC"] == "1",
            "hga_prefill_stream_paced": env["HGA_PREFILL_STREAM_PACED"] == "1",
            "hga_stream_async": env["HGA_STREAM_ASYNC"] == "1",
            "hga_stream_block": stream_block,
            "hga_verify_streams": verify_streams,
            "hga_stream_chunk_mib": args.hga_stream_chunk_mib,
            "hga_stream_paced": stream_paced,
            "threads": int(env.get("HGA_THREADS", "12")),
            "verify_k_tiles": int(env.get("HGA_VERIFY_K_TILES", "3")),
            "threads_batch": int(env.get("HGA_THREADS_BATCH", "1")),
            "ignore_eos": True,
            "no_conversation": True,
        },
        "perf": perf,
        "pins": pins,
        "split": split,
        "graphs": graphs,
        "spec": spec,
        "oom_allocs": oom_allocs,
        "gates": baseline["gates"],
        "reference": baseline.get("reference"),
        "log": str(log_path),
    }

    errors: list[str] = []
    if proc.returncode != 0:
        errors.append(f"launcher exit {proc.returncode}")
    if fail:
        errors.append(fail)
    errors.extend(evaluate_gates(perf, baseline))
    errors.extend(evaluate_pin_gates(pins, baseline))
    errors.extend(evaluate_graph_gates(graphs, baseline))
    errors.extend(evaluate_spec_gates(spec, baseline))
    record["errors"] = errors
    record["passed"] = not errors
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(format_measure_line(perf), flush=True)
    print_report(record)
    print_local_summary(record)
    print(f"json             {json_path}", flush=True)
    print(f"log              {log_path}", flush=True)
    return 0 if record["passed"] else 1


def self_test() -> int:
    default_args = build_parser().parse_args([])
    assert UBATCH == 768
    assert GPU_PREFILL_MAX_KEYS == 3200
    assert default_args.hga_kernel == "tiled", default_args
    assert default_args.hga_stream_block == 0, default_args
    assert default_args.hga_verify_streams == 2, default_args
    assert not default_args.hga_stream_sync, default_args
    assert default_args.hga_stream_chunk_mib == 0, default_args
    assert default_args.hga_stream_paced, default_args
    assert not default_args.hga_prefill_k_tiles, default_args
    assert default_args.hga_gpu_prefill, default_args
    assert not default_args.hga_prefill_stream_async, default_args
    assert not default_args.hga_prefill_stream_paced, default_args
    kernel_env = {
        "HGA_THREADS": "2",
        "HGA_THREADS_BATCH": "80",
        "HGA_VERIFY_BATCH": "0",
        "HGA_VERIFY_TILES": "0",
        "HGA_VERIFY_ROWS": "1",
    }
    configure_benchmark_hga_env(kernel_env, default_args.hga_kernel)
    assert kernel_env == {
        "HGA_THREADS": os.environ.get("HGA_THREADS", "12"),
        "HGA_THREADS_BATCH": "1",
        "HGA_VERIFY_BATCH": "1",
        "HGA_VERIFY_TILES": "1",
        "HGA_VERIFY_ROWS": "0",
        "HGA_STREAM_ASYNC": "1",
        "HGA_L2_OFF": "1",
        "HGA_LOAD_MODE": "none",
    }, kernel_env

    sample = """
llama_perf_context_print:        load time =   12345.67 ms
llama_perf_context_print: prompt eval time =   10204.12 ms /  2001 tokens (    5.10 ms per token,   196.10 tokens per second)
llama_perf_context_print:        eval time =    5289.26 ms /    64 runs   (   82.64 ms per token,    12.10 tokens per second)
llama_perf_context_print:       total time =   15500.00 ms /  2065 tokens
"""
    perf = parse_llama_perf(sample)
    assert perf["load_ms"] == 12345.67, perf
    assert perf["prefill_tokens"] == 2001, perf
    assert perf["prefill_tok_s"] == 196.10, perf
    assert perf["generate_tokens"] == 64, perf
    assert perf["generate_tok_s"] == 12.10, perf

    prompt_only = (
        "llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens "
        "(10.00 ms per token, 100.00 tokens per second)\n"
    )
    p2 = parse_llama_perf(prompt_only)
    assert p2["generate_tok_s"] is None, p2
    assert p2["prefill_tokens"] == 10, p2

    oom = failure_line("ggml_gallocr: failed to allocate CUDA0 buffer")
    assert oom is not None

    baseline = load_baseline()
    good = {
        "prefill_tokens": 2001,
        "prefill_tok_s": 196.0,
        "generate_tokens": 64,
        "generate_tok_s": 12.1,
    }
    assert evaluate_gates(good, baseline) == []
    bad = dict(good, prefill_tok_s=10.0)
    errs = evaluate_gates(bad, baseline)
    assert errs, errs

    text = build_prompt()
    assert text.startswith("Write a one-paragraph")
    assert text.count(PROMPT_SENTENCE) == PROMPT_SENTENCES

    measured = {
        "prefill_tokens": 2001,
        "prefill_ms": 10204.12,
        "prefill_tok_s": 196.0,
        "generate_tokens": 64,
        "generate_ms": 5289.26,
        "generate_tok_s": 12.1,
    }
    line = format_measure_line(measured)
    parsed = parse_measure_line("noise\n" + line + "\n")
    assert parsed is not None and parsed["prefill_tok_s"] == 196.0

    pin_sample = (
        "hga-swap: PREFILL  exchange 8 pairs  lm_head CUDA  in 245.3 ms\n"
        "hga-swap: VERIFY stream 2/8 pairs; 6 leftover pairs deferred\n"
        "hga-swap: PIN-STEP summary  pinned 6/6 leftover pairs; "
        "stream fallback=0; no unrecovered OOM\n"
        "hga-pin: census after leftover VERIFY pins  phase=VERIFY  tensors=851  "
        "GPU=850/15345.7MiB  CUDA_Host=1/682.0MiB  CPU=0/0.0MiB  PUSHED=0  MOVE=140\n"
        "hga-pin:   resident  626 GPU /   0 host  (n=626)\n"
        "hga-pin:   exchange  224 GPU /   0 host  (n=224)\n"
    )
    pins = parse_pin_log(pin_sample)
    assert evaluate_pin_gates(pins, baseline) == [], pins
    one_each = (
        "hga: graphs_disabled=1\n"
        "ubatch before alloc_graph n_tok=768 n_ubatch=768 n_omax=1 n_rs=0 phase=1 ctx=0\n"
        "ubatch before alloc_graph n_tok=3 n_ubatch=3 n_omax=1 n_rs=2 phase=3 ctx=0\n"
        "graphs reused = 22\n"
    )
    graphs = parse_graph_log(one_each)
    assert evaluate_graph_gates(graphs, baseline) == [], graphs
    safe_prefill = dict(graphs, prefill_builds=4, prefill_extra_rebuilds=3)
    assert evaluate_graph_gates(safe_prefill, baseline) == [], safe_prefill
    too_many_prefill = dict(graphs, prefill_builds=5, prefill_extra_rebuilds=4)
    assert evaluate_graph_gates(too_many_prefill, baseline), too_many_prefill
    spec_sample = "n_drafted = 33\nn_accept  = 30\naccept    = 90.909%\n"
    spec = parse_spec_log(spec_sample)
    assert evaluate_spec_gates(spec, baseline) == [], spec
    greedy = parse_spec_log("graphs reused = 64\n")
    assert evaluate_spec_gates(greedy, baseline), greedy
    oom_sample = (
        "hga-alloc-oom: BEGIN why=leftover-pin\n"
        "hga-alloc-oom: last20\n"
        "hga-alloc-oom:  1  module=lm_head  who=output.weight  mib=994.60  "
        "free_before=4000  free_after=3000  ok=1\n"
        "hga-alloc-oom:  2  module=leftover-pin  who=24-56  mib=457.07  "
        "free_before=744  free_after=744  ok=0\n"
        "hga-alloc-oom: live by module\n"
        "hga-alloc-oom:  module=lm_head  n=1  mib=994.6\n"
        "hga-alloc-oom: END\n"
    )
    dump = parse_oom_alloc_dump(oom_sample)
    assert dump["events"][-1]["ok"] == 0 and dump["events"][-1]["who"] == "24-56"
    print("self-test OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Measure default-packing 2000-token prefill + 64-token generate "
            "on a 16 GB GPU host."
        )
    )
    p.add_argument(
        "command",
        nargs="?",
        choices=("run",),
        default="run",
        help="run the local speed test (default)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Ignored; this harness always runs on this machine",
    )
    p.add_argument(
        "--json",
        default="/tmp/hga_prefill-2k-gen-64.json",
        help="Where to write the result JSON",
    )
    p.add_argument(
        "--log",
        default="/tmp/hga_prefill-2k-gen-64.log",
        help="Where to write combined llama stdout/stderr",
    )
    p.add_argument(
        "--prompt-file",
        default="/tmp/hga_prefill_2k.txt",
        help="Prompt path written before the run",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds allowed for the llama process itself",
    )
    p.add_argument(
        "--hga-kernel",
        choices=HGA_KERNEL_MODES,
        default="tiled",
        help=(
            "HGA implementation for 1-3-token decode/verify: tiled is the "
            "optimized 4-KV-head x 3-key-tile default; the other modes are "
            "diagnostic A/B paths"
        ),
    )
    p.add_argument(
        "--hga-stream-block",
        type=int,
        choices=HGA_STREAM_BLOCK_MODES,
        default=0,
        help=(
            "experimental coarse weight exchange: pack 2 or 3 consecutive "
            "layer pairs into one slot/copy; 0 keeps the default two streams"
        ),
    )
    p.add_argument(
        "--hga-verify-streams",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "number of independent default exchange pairs retained in VERIFY; "
            "3 is a transfer-volume A/B diagnostic"
        ),
    )
    p.add_argument(
        "--hga-stream-sync",
        action="store_true",
        help=(
            "diagnostic synchronized exchange callbacks with per-boundary "
            "remaining-copy wait timing"
        ),
    )
    p.add_argument(
        "--hga-stream-chunk-mib",
        type=int,
        choices=(0, 4, 8, 16, 32, 64),
        default=0,
        help=(
            "split large verify weight uploads into this many MiB per DMA so "
            "latency-critical activation H2D copies can interleave; 0 is one DMA"
        ),
    )
    p.add_argument(
        "--hga-prefill-k-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "prefill HGA layout: split each KV head into 2 query tiles x 5 "
            "key tiles (40 tasks; leave off unless you have 40 physical cores)"
        ),
    )
    p.add_argument(
        "--hga-gpu-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "route every 64-token chunk on CPU, unite one capacity-bounded "
            "history per KV head, and run CUDA attention with a reusable "
            "K/V width (default: enabled)"
        ),
    )
    p.add_argument(
        "--hga-prefill-stream-async",
        action="store_true",
        help=(
            "experimental prefill streamer: replace synchronizing scheduler "
            "callbacks with inline CUDA events and submit each image immediately"
        ),
    )
    p.add_argument(
        "--hga-prefill-stream-paced",
        action="store_true",
        help=(
            "experimental prefill streamer: arm at released layer slots and "
            "submit one whole weight image after each completed HGA layer"
        ),
    )
    p.add_argument(
        "--hga-stream-paced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "split each layer upload into four segments and submit one segment "
            "after each completed HGA layer (default: enabled)"
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser/gate checks without loading a model",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    return run_local(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"timed out: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
