"""Synthetic RULER-style long-context retrieval suite for the routed 8B QLoRA model.

Three needle-in-a-haystack tasks (a distinctive fact buried in unrelated filler, then a question),
scored by exact-match recall of the planted answer — the standard RULER recipe, self-contained so
no external dataset download is needed:

* **passkey**    — one secret code buried in the filler; recover it.
* **multikey**   — several ``<key> -> <value>`` facts at spread depths; recover one queried key.
* **multivalue** — one key with several distinct values; recover *all* of them (recall).

Each task is generated at several context lengths (``--ctx-sizes``) and evaluated under up to three
regimes on a single loaded model (mirrors the fine-tune script's three-way validation):

* **ft-routed**   — the fine-tuned adapter with routed sparse attention (the deploy regime).
* **stock-routed**— the same routed attention with the LoRA adapter disabled (un-fine-tuned base).
* **dense**       — the fine-tuned weights under the original dense attention, run only where the
                    full S² forward still fits (long contexts OOM — that is the point of routing).

This extends v1's single 64K needle check to a proper scored suite at 8B.  It reuses the fine-tune
harness's model loading / attention toggling and the 30B test harness's blocked cache-backed
``greedy_generate`` + ``filler_text`` so the routing geometry and decode path stay the single source
of truth (build on ``compare_ram``).

Run from the repo root::

    python -m ExistingModelFineTuning.Qwen3LongContext.ruler_eval --selftest   # no model, fast
    python -m ExistingModelFineTuning.Qwen3LongContext.ruler_eval \
        --adapter-path checkpoints/qwen8b_routed/best --cache-location ram \
        --ctx-sizes 4096 8192 16384 32768
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Callable, Dict, List, Tuple

import torch

# Reuse the fine-tune harness (model load + routed/dense toggle) and the 30B test harness's blocked
# decode + filler generator so the routing geometry and decode path stay the single source of truth.
try:
    from .finetune_qwen06b_qlora_routed import (  # type: ignore
        attention_mode,
        load_routed_base,
        routing_knobs,
    )
    from .test_qwen30b_routed import filler_text, greedy_generate  # type: ignore
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from finetune_qwen06b_qlora_routed import (  # type: ignore
        attention_mode,
        load_routed_base,
        routing_knobs,
    )
    from test_qwen30b_routed import filler_text, greedy_generate  # type: ignore

from peft import PeftModel
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Distinctive keys/values so the model cannot guess them from the filler (RULER's "hard distractor"
# principle): rare-ish nouns as keys, random digit strings as values.
_KEYS = ["falcon", "meridian", "cobalt", "lantern", "quartz", "harbor", "cipher", "willow",
         "nimbus", "orchard", "granite", "vellum"]


def _rand_code(rng: random.Random, digits: int = 7) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(digits))


def _insert_needles(tok, ctx_tokens: int, needles: List[str], rng: random.Random) -> str:
    """Embed ``needles`` at spread depths inside ~``ctx_tokens`` tokens of unrelated filler.

    Each needle lands at an evenly spaced fractional depth (jittered) so retrieval is tested across
    the whole context, not just near the query — the RULER depth-sweep idea in miniature.
    """
    filler = filler_text(tok, ctx_tokens)
    ids = tok(filler, add_special_tokens=False).input_ids
    n = len(ids)
    k = len(needles)
    # Evenly spaced anchor depths in (0.05, 0.95), lightly jittered, kept sorted for a clean splice.
    depths = sorted(min(0.95, max(0.05, (j + 1) / (k + 1) + rng.uniform(-0.03, 0.03)))
                    for j in range(k))
    pieces: List[str] = []
    prev = 0
    for depth, needle in zip(depths, needles):
        cut = min(n, max(prev, int(n * depth)))
        pieces.append(tok.decode(ids[prev:cut]))
        pieces.append("\n" + needle + "\n")
        prev = cut
    pieces.append(tok.decode(ids[prev:]))
    return "".join(pieces)


# =================================================================================================
# Task builders — each returns (prompt_text, expected_answers, scorer)
#   scorer(generated_text) -> recall fraction in [0, 1] (1.0 == fully correct)
# =================================================================================================
def _contains_all(gen: str, answers: List[str]) -> float:
    got = sum(1 for a in answers if a.lower() in gen.lower())
    return got / max(1, len(answers))


def build_passkey(tok, ctx: int, rng: random.Random) -> Tuple[str, List[str], Callable[[str], float]]:
    code = _rand_code(rng)
    needle = f"The special passkey is {code}. Remember it carefully."
    body = _insert_needles(tok, ctx, [needle], rng)
    prompt = (body + "\n\nUsing only the text above, answer concisely.\n"
              "Question: What is the special passkey?\nAnswer: The special passkey is")
    return prompt, [code], lambda g: _contains_all(g, [code])


def build_multikey(tok, ctx: int, rng: random.Random, num_keys: int
                   ) -> Tuple[str, List[str], Callable[[str], float]]:
    keys = rng.sample(_KEYS, k=min(num_keys, len(_KEYS)))
    pairs = {key: _rand_code(rng) for key in keys}
    needles = [f"The magic number for {key} is {val}." for key, val in pairs.items()]
    body = _insert_needles(tok, ctx, needles, rng)
    target = rng.choice(keys)
    prompt = (body + "\n\nUsing only the text above, answer concisely.\n"
              f"Question: What is the magic number for {target}?\n"
              f"Answer: The magic number for {target} is")
    ans = pairs[target]
    return prompt, [ans], lambda g: _contains_all(g, [ans])


def build_multivalue(tok, ctx: int, rng: random.Random, num_values: int
                     ) -> Tuple[str, List[str], Callable[[str], float]]:
    key = rng.choice(_KEYS)
    vals = [_rand_code(rng) for _ in range(num_values)]
    needles = [f"One secret token for {key} is {v}." for v in vals]
    body = _insert_needles(tok, ctx, needles, rng)
    prompt = (body + "\n\nUsing only the text above, answer concisely.\n"
              f"Question: List all secret tokens for {key}.\nAnswer: The secret tokens for {key} are")
    return prompt, list(vals), lambda g: _contains_all(g, vals)


TASKS = {
    "passkey": lambda tok, ctx, rng, a: build_passkey(tok, ctx, rng),
    "multikey": lambda tok, ctx, rng, a: build_multikey(tok, ctx, rng, a.num_keys),
    "multivalue": lambda tok, ctx, rng, a: build_multivalue(tok, ctx, rng, a.num_values),
}


# =================================================================================================
# Model + regimes
# =================================================================================================
def _load(args, compute_dtype):
    """Load the 4-bit base (+ routed surgery) and, if given, the fine-tuned adapter on top."""
    base, n = load_routed_base(args, compute_dtype)
    adapter_dir = args.adapter_path
    if adapter_dir:
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        model = PeftModel.from_pretrained(base, adapter_dir)
        print(f"[load] adapter from {adapter_dir} on {n} routed layers")
        has_adapter = True
    else:
        model = base
        print(f"[load] stock base only ({n} routed layers); no adapter (ft-routed == stock-routed)")
        has_adapter = False
    model.eval()
    return model, has_adapter


def run(args) -> Dict[str, Dict[str, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit RULER evaluation.")
    device = torch.device("cuda")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    knobs = routing_knobs(args)
    model, has_adapter = _load(args, compute_dtype)

    tasks = args.tasks.replace(",", " ").split()
    ctx_sizes = [int(c) for c in str(args.ctx_sizes).replace(",", " ").split()]

    # regime -> task -> list of per-ctx recall (averaged over --samples)
    scores: Dict[str, Dict[str, List[float]]] = {}

    def _score_regime(regime: str, generate) -> None:
        bucket = scores.setdefault(regime, {t: [] for t in tasks})
        for ctx in ctx_sizes:
            for task in tasks:
                rng = random.Random(args.seed + ctx + hash(task) % 10_000)
                recalls = []
                for _ in range(args.samples):
                    prompt, answers, scorer = TASKS[task](tokenizer, ctx, rng, args)
                    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    gen = generate(ids)
                    recalls.append(scorer(gen))
                avg = sum(recalls) / len(recalls)
                bucket[task].append(avg)
                print(f"[{regime}] ctx={ctx:>6} {task:<11} recall={avg * 100:5.1f}% "
                      f"({ids.shape[1]} tok)", flush=True)

    def _gen(ids):
        torch.cuda.empty_cache()
        try:
            return greedy_generate(model, tokenizer, ids, args.max_new, args.block)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return ""  # OOM at this ctx -> empty answer scores 0 (recorded, not fatal)

    # ft-routed (adapter on, routed attention as loaded)
    _score_regime("ft-routed" if has_adapter else "stock-routed", _gen)

    # stock-routed (adapter disabled) — only meaningful when an adapter is present.
    if has_adapter:
        def _gen_stock(ids):
            with model.disable_adapter():
                return _gen(ids)
        _score_regime("stock-routed", _gen_stock)

    # dense (same weights, routing off) — only where the full S² forward fits; catch OOM per ctx.
    if not args.no_dense:
        def _gen_dense(ids):
            try:
                with attention_mode(model, knobs, routed=False, cache_location=args.cache_location):
                    return _gen(ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return ""
        _score_regime("dense", _gen_dense)

    _print_summary(scores, ctx_sizes, tasks)
    if args.out_tsv:
        _write_tsv(args.out_tsv, scores, ctx_sizes, tasks)
    return scores


def _print_summary(scores, ctx_sizes, tasks) -> None:
    print("\n=== RULER recall (%) by regime / task / context ===")
    header = "regime          task         " + "".join(f"{c:>9}" for c in ctx_sizes)
    print(header)
    for regime, bucket in scores.items():
        for task in tasks:
            vals = bucket[task]
            cells = "".join(f"{v * 100:>8.1f}%" if i < len(vals) else f"{'--':>9}"
                            for i, v in enumerate(vals))
            print(f"{regime:<16}{task:<13}{cells}")


def _write_tsv(path: str, scores, ctx_sizes, tasks) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("regime\ttask\t" + "\t".join(str(c) for c in ctx_sizes) + "\n")
        for regime, bucket in scores.items():
            for task in tasks:
                vals = bucket[task]
                fh.write(f"{regime}\t{task}\t" + "\t".join(f"{v:.4f}" for v in vals) + "\n")
    print(f"[tsv] wrote {path}")


def selftest(args) -> None:
    """Fast, model-free check that task construction + scoring are correct (the one runnable check)."""
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(0)
    ctx = 512
    for task in ("passkey", "multikey", "multivalue"):
        prompt, answers, scorer = TASKS[task](tok, ctx, rng, args)
        assert answers, f"{task}: no expected answers"
        # Every planted answer must actually appear in the prompt (needle really inserted).
        for a in answers:
            assert a in prompt, f"{task}: planted answer {a!r} missing from prompt"
        # A perfect answer scores 1.0; an empty answer scores 0.0 (exact-match recall works).
        assert scorer(" ".join(answers)) == 1.0, f"{task}: full answer did not score 1.0"
        assert scorer("nothing here") == 0.0, f"{task}: empty match did not score 0.0"
        # Needles must sit inside the filler, not all bunched at the very end near the question.
        depth = prompt.find(answers[0]) / max(1, len(prompt))
        assert depth < 0.95, f"{task}: needle too close to the end (depth {depth:.2f})"
        print(f"[selftest] {task:<11} OK ({len(answers)} needle(s), first-needle depth {depth:.2f})")
    print("[selftest] OK: all tasks build valid needle prompts and exact-match scoring is correct")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter-path", default=None,
                   help="fine-tuned routed adapter dir; omit to score the stock base only.")
    p.add_argument("--attn-mode", choices=("routed", "dense"), default="routed",
                   help="surgery mode for the loaded model (keep 'routed' for the routed regimes).")
    p.add_argument("--cache-location", choices=("vram", "ram", "fs"), default="ram",
                   help="cold KV tier for the routed store; 'ram' keeps long-context VRAM bounded.")
    p.add_argument("--ctx-sizes", default="4096 8192 16384 32768",
                   help="comma/space-separated context lengths to test.")
    p.add_argument("--tasks", default="passkey multikey multivalue",
                   help="comma/space-separated subset of: passkey, multikey, multivalue.")
    p.add_argument("--num-keys", type=int, default=4, help="key/value facts planted for multikey.")
    p.add_argument("--num-values", type=int, default=4, help="values planted for multivalue.")
    p.add_argument("--samples", type=int, default=3, help="random samples averaged per (ctx, task).")
    p.add_argument("--max-new", type=int, default=48, help="tokens to generate for the answer.")
    p.add_argument("--block", type=int, default=64, help="prefill/decode block size for greedy_generate.")
    p.add_argument("--no-dense", action="store_true", help="skip the dense regime entirely.")
    p.add_argument("--fp16", action="store_true", help="fp16 compute (Turing tensor-core path).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-tsv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_eval.tsv"))
    # routing sparsity overrides (0 = locked defaults) — kept in sync with the fine-tune script.
    p.add_argument("--topk-chunks", type=int, default=0)
    p.add_argument("--topk-groups", type=int, default=0)
    # LoRA fields consumed by load_routed_base's shared path (unused here but kept for signature parity).
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--selftest", action="store_true", help="run the model-free construction check and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        selftest(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
