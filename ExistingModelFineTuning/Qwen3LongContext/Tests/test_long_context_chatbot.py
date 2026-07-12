#!/usr/bin/env python3
"""Long-context stability tests for the chatbot's hierarchical KV-routing attention.

These reproduce the *exact* code path ``chat_qwen30b_fp8.py`` uses — same
``replace_qwen_attention_with_router`` config, same blocked-prefill + greedy-decode loop
(``_generate_iter``), same multi-turn prefix-reuse (``_plan_prefill``) and teardown
(``_dispose_cache``) — and drive it through the modes that have been crashing:

  * ``single``    — one prompt with a very long context (default 50K tokens) + a short decode.
  * ``sequenced`` — several big appended turns growing past 256K tokens (the prefix-HIT
                    incremental path), each followed by a short decode.
  * ``decode``    — a moderate context followed by a long decode (chunks close mid-decode,
                    so the append/spill/evict path runs thousands of times).
  * ``reset``     — build context, ``/reset`` (dispose the store), rebuild — exercises the
                    FS teardown + "no spill files left behind" guarantee.

The cold-KV tier defaults to ``fs`` (the NvM/disk-spillover mode the question is about); pass
``--cache ram|vram`` to check the others.

Isolation & safety
------------------
Each scenario runs in its **own subprocess** (``--worker``), so a CUDA OOM or a host-RAM
OOM-kill takes down only that child — never this process or a Jupyter kernel sharing the box.
A lightweight **RSS watchdog** thread aborts the child if its resident memory crosses
``--mem-limit-gb`` (default: 80% of total RAM), so the child self-terminates *before* the
Linux OOM killer can pick a different victim (e.g. your running Jupyter).  This is why the
suite is safe to run on the same machine as the app.

Model
-----
By default a **tiny random-weight Qwen3-MoE** stands in for the 30B (``head_dim`` / GQA / MoE /
RoPE structure identical, weights random): it exercises every routing / DCA / FS-spill / multi-turn
code path at 256K+ tokens in seconds and within a few GB, so it is the CI / smoke regression guard.
``--real`` loads the actual ``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`` instead (run that only on a
box where the model fits comfortably — see the README's FP8 memory note).

Usage
-----
    # Fast proxy-model suite (safe here; fs tier, forced disk spill):
    python -m ExistingModelFineTuning.Qwen3LongContext.Tests.test_long_context_chatbot

    # One scenario, custom size:
    python -m ExistingModelFineTuning.Qwen3LongContext.Tests.test_long_context_chatbot \
        --only sequenced --max-ctx 300000

    # Against the real 30B (heavy — run on a dedicated box):
    python -m ExistingModelFineTuning.Qwen3LongContext.Tests.test_long_context_chatbot --real
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

# Set before CUDA initialises (mirrors the chatbot) — must precede any torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Host-RAM watchdog: self-terminate before the OS OOM killer hits another process.
# ---------------------------------------------------------------------------
def _rss_gb() -> float:
    """Resident set size of this process, in GB (Linux /proc; 0.0 if unavailable)."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as fh:
            pages = int(fh.read().split()[1])  # resident pages
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (OSError, ValueError, IndexError):
        return 0.0


def _start_rss_watchdog(limit_gb: float) -> None:
    if limit_gb <= 0:
        return

    def _watch() -> None:
        while True:
            rss = _rss_gb()
            if rss > limit_gb:
                sys.stderr.write(
                    f"\n[watchdog] RSS {rss:.1f}GB exceeded limit {limit_gb:.1f}GB — "
                    f"aborting this worker to protect the host.\n"
                )
                sys.stderr.flush()
                os._exit(137)  # SIGKILL-like; only this child dies
            time.sleep(0.5)

    threading.Thread(target=_watch, name="rss-watchdog", daemon=True).start()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _build_proxy_model(device: str):
    """Tiny random-weight Qwen3-MoE with the real structural knobs (head_dim/GQA/MoE/RoPE)."""
    import torch
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=2048, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1, norm_topk_prob=True,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(device).to(torch.bfloat16).eval()
    return model, cfg.vocab_size, None


def _build_real_model(device: str):
    """Load the real FP8 30B exactly as the chatbot does."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as chat

    tok = AutoTokenizer.from_pretrained(chat.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        chat.MODEL, torch_dtype="auto", device_map=device, attn_implementation="sdpa"
    ).eval()
    torch.cuda.synchronize()
    return model, tok.vocab_size, tok


class _StubTok:
    """Minimal tokenizer for the proxy model: random ids, no real decode needed."""
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):  # noqa: D401
        return ""


# ---------------------------------------------------------------------------
# One worker = one scenario in its own process.
# ---------------------------------------------------------------------------
def _worker(args) -> int:
    import torch
    from transformers import DynamicCache
    from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as chat
    from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
        replace_qwen_attention_with_router,
    )

    _start_rss_watchdog(args.mem_limit_gb)
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    if args.real:
        model, vocab, tok = _build_real_model(device)
    else:
        model, vocab, _ = _build_proxy_model(device)
        tok = _StubTok()

    # Identical knobs to the chatbot, but with a configurable ram budget / cache dir so the proxy
    # can force genuine disk spill, and so we never collide with the user's live chat spill dir.
    # NB: the spill dir MUST be on a real disk — never a tmpfs like /tmp (RAM-backed: it would put
    # the "disk" tier back in RAM and re-introduce the very host-RAM OOM these tests guard against).
    fs_dir = os.path.join(args.fs_cache_dir, f"kvr_test_{args.scenario}_{os.getpid()}")
    n = replace_qwen_attention_with_router(
        model, cache_location=args.cache,
        keep_first=chat.KEEP_FIRST, keep_last=chat.KEEP_LAST, topk_chunks=chat.TOPK_CHUNKS,
        topk_groups=chat.TOPK_GROUPS, chunk_size=chat.CHUNK_SIZE, group_size=chat.GROUP_SIZE,
        vram_cache_chunks=chat.VRAM_CACHE_CHUNKS, vram_summary_chunks=chat.VRAM_SUMMARY_CHUNKS,
        vram_cache_reserve_gb=chat.VRAM_CACHE_RESERVE_GB,
        ram_budget_gb=args.ram_budget_gb, fs_cache_dir=fs_dir,
        dca_chunk=chat.DCA_CHUNK, dca_local=chat.DCA_LOCAL,
    )
    print(f"[worker {args.scenario}] model={'real' if args.real else 'proxy'} "
          f"cache={args.cache} layers={n} block={args.block} "
          f"ram_budget={args.ram_budget_gb}GB max_ctx={args.max_ctx} decode={args.decode}",
          flush=True)

    torch.manual_seed(1)

    def make_block(n_tok: int):
        return torch.randint(0, vocab, (n_tok,), device=device).tolist()

    def run_turn(cache, cached_ids, new_ids, decode):
        input_ids = torch.tensor([new_ids], device=device)
        cache, prefill_start = chat._plan_prefill(cache, cached_ids, new_ids)
        stats = {}
        for item in chat._generate_iter(model, tok, input_ids, max_new=decode,
                                        block=args.block, cache=cache, prefill_start=prefill_start):
            if isinstance(item, dict):
                stats = item
        return cache, stats["cached_ids"], stats

    def assert_bounded(stats, where):
        # VRAM must stay bounded by the weights regardless of context (the whole point of routing).
        assert stats["peak_gb"] < args.peak_vram_limit_gb, \
            f"{where}: peak VRAM {stats['peak_gb']:.1f}GB exceeded {args.peak_vram_limit_gb}GB"

    t0 = time.perf_counter()
    cache, cached_ids = None, []

    if args.scenario == "single":
        new_ids = make_block(args.max_ctx)
        cache, cached_ids, stats = run_turn(cache, cached_ids, new_ids, args.decode)
        print(f"  ctx={stats['n_ctx']} out={stats['n_out']} ttft={stats['ttft']:.1f}s "
              f"peak={stats['peak_gb']:.2f}GB rss={_rss_gb():.1f}GB", flush=True)
        assert stats["n_out"] >= 1, "no tokens produced"
        assert_bounded(stats, "single")

    elif args.scenario == "sequenced":
        per_turn = max(chat.CHUNK_SIZE, args.max_ctx // args.turns)
        for turn in range(args.turns):
            new_ids = cached_ids + make_block(per_turn)
            cache, cached_ids, stats = run_turn(cache, cached_ids, new_ids, args.decode)
            print(f"  turn {turn}: ctx={stats['n_ctx']} out={stats['n_out']} "
                  f"ttft={stats['ttft']:.1f}s peak={stats['peak_gb']:.2f}GB "
                  f"rss={_rss_gb():.1f}GB cached={len(cached_ids)}", flush=True)
            assert stats["n_out"] >= 1, f"turn {turn}: no tokens produced"
            assert_bounded(stats, f"sequenced turn {turn}")
        assert len(cached_ids) >= args.max_ctx, \
            f"sequenced only reached {len(cached_ids)} < {args.max_ctx} tokens"

    elif args.scenario == "decode":
        new_ids = make_block(args.max_ctx)
        cache, cached_ids, stats = run_turn(cache, cached_ids, new_ids, args.decode)
        print(f"  ctx={stats['n_ctx']} out={stats['n_out']} ttft={stats['ttft']:.1f}s "
              f"peak={stats['peak_gb']:.2f}GB rss={_rss_gb():.1f}GB", flush=True)
        assert stats["n_out"] == args.decode, \
            f"decode produced {stats['n_out']} != requested {args.decode}"
        assert_bounded(stats, "decode")

    elif args.scenario == "reset":
        spill_dirs = []
        for cycle in range(2):
            new_ids = make_block(args.max_ctx)
            cache, cached_ids, stats = run_turn(None, [], new_ids, args.decode)
            router = getattr(cache, "_kv_router", None)
            store = getattr(router, "store", None)
            disk = getattr(store, "disk", None)
            if disk is not None:
                spill_dirs.append(disk.dir)
            print(f"  cycle {cycle}: ctx={stats['n_ctx']} peak={stats['peak_gb']:.2f}GB "
                  f"rss={_rss_gb():.1f}GB", flush=True)
            assert_bounded(stats, f"reset cycle {cycle}")
            chat._dispose_cache(cache)  # simulates /reset
            cache, cached_ids = None, []
        # The fs tier must remove its spill dirs on dispose/close (no files left behind).
        if args.cache == "fs":
            for d in spill_dirs:
                assert not os.path.exists(d), f"reset left spill dir behind: {d}"
            print("  [ok] fs spill dirs removed on dispose", flush=True)

    else:
        raise ValueError(f"unknown scenario {args.scenario}")

    print(f"[worker {args.scenario}] OK in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Orchestrator: spawn one subprocess per scenario.
# ---------------------------------------------------------------------------
SCENARIOS = ["single", "sequenced", "decode", "reset"]


def _spawn(scenario: str, base_args) -> bool:
    cmd = [
        sys.executable, "-m",
        "ExistingModelFineTuning.Qwen3LongContext.Tests.test_long_context_chatbot",
        "--worker", "--scenario", scenario,
        "--cache", base_args.cache, "--block", str(base_args.block),
        "--ram-budget-gb", str(base_args.ram_budget_gb),
        "--max-ctx", str(base_args.max_ctx), "--turns", str(base_args.turns),
        "--decode", str(base_args.decode if scenario != "decode" else base_args.decode_long),
        "--peak-vram-limit-gb", str(base_args.peak_vram_limit_gb),
        "--mem-limit-gb", str(base_args.mem_limit_gb),
        "--fs-cache-dir", base_args.fs_cache_dir,
    ]
    if base_args.real:
        cmd.append("--real")
    # `decode` and `single`/`reset` want different context sizes; keep them modest where decode is long.
    if scenario == "decode":
        cmd[cmd.index("--max-ctx") + 1] = str(base_args.decode_ctx)
    print(f"\n=== scenario: {scenario} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=_REPO)
    ok = proc.returncode == 0
    print(f"=== scenario {scenario}: {'PASS' if ok else f'FAIL (exit {proc.returncode})'} "
          f"in {time.perf_counter() - t0:.1f}s ===", flush=True)
    return ok


def _default_fs_cache_dir() -> str:
    """Real-disk root for spill files (mirrors the chatbot's ~/.cache default — never tmpfs)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "kvr_test_cache")


def _default_mem_limit_gb() -> float:
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        return round(total * 0.8, 1)
    except (OSError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scenario", choices=SCENARIOS, help="(worker) scenario to run")
    ap.add_argument("--real", action="store_true",
                    help="Load the real Qwen3-30B-A3B-FP8 instead of the tiny proxy model")
    ap.add_argument("--cache", choices=("fs", "ram", "vram"), default="fs",
                    help="Cold-KV tier to test (default: fs — the NvM/disk-spillover mode)")
    ap.add_argument("--only", choices=SCENARIOS, help="Run only this scenario")
    ap.add_argument("--block", type=int, default=None,
                    help="Prefill block (default: chatbot PREFILL_BLOCK)")
    ap.add_argument("--ram-budget-gb", type=float, default=None,
                    help="fs-tier host-RAM budget. Proxy default forces spill; real default = chatbot")
    ap.add_argument("--max-ctx", type=int, default=262144,
                    help="Context tokens for single/sequenced (default 256K)")
    ap.add_argument("--turns", type=int, default=6, help="Turns for the sequenced scenario")
    ap.add_argument("--decode", type=int, default=8, help="Decode tokens for short-decode scenarios")
    ap.add_argument("--decode-long", type=int, default=2000,
                    help="Decode tokens for the long-decode scenario")
    ap.add_argument("--decode-ctx", type=int, default=8192,
                    help="Context for the long-decode scenario (kept modest so decode dominates)")
    ap.add_argument("--peak-vram-limit-gb", type=float, default=None,
                    help="Assert peak VRAM stays below this (default: proxy 4GB / real 31GB)")
    ap.add_argument("--mem-limit-gb", type=float, default=_default_mem_limit_gb(),
                    help="RSS watchdog ceiling per worker (0 disables). Default: 80%% of total RAM")
    ap.add_argument("--fs-cache-dir", default=_default_fs_cache_dir(),
                    help="Root for fs-tier spill files. MUST be a real disk, never tmpfs (/tmp). "
                         "Default: $XDG_CACHE_HOME/kvr_test_cache or ~/.cache/kvr_test_cache")
    args = ap.parse_args()

    # Resolve model-dependent defaults.
    from ExistingModelFineTuning.Qwen3LongContext import chat_qwen30b_fp8 as chat
    if args.block is None:
        args.block = chat.PREFILL_BLOCK
    if args.ram_budget_gb is None:
        args.ram_budget_gb = chat.RAM_BUDGET_GB if args.real else 0.02  # tiny → force spill on proxy
    if args.peak_vram_limit_gb is None:
        args.peak_vram_limit_gb = 31.0 if args.real else 4.0

    if args.worker:
        sys.exit(_worker(args))

    scenarios = [args.only] if args.only else SCENARIOS
    print(f"Long-context chatbot stability suite — model={'REAL 30B' if args.real else 'proxy'}, "
          f"cache={args.cache}, scenarios={scenarios}", flush=True)
    results = {s: _spawn(s, args) for s in scenarios}
    print("\n" + "=" * 60)
    for s, ok in results.items():
        print(f"  {s:10s} : {'PASS' if ok else 'FAIL'}")
    n_fail = sum(1 for ok in results.values() if not ok)
    print("=" * 60)
    if n_fail:
        print(f"{n_fail}/{len(results)} scenario(s) FAILED")
        sys.exit(1)
    print(f"All {len(results)} scenario(s) passed.")


if __name__ == "__main__":
    main()
