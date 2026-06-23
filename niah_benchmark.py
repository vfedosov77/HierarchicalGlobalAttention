#
"""  Needle-in-a-Haystack benchmark for Qwen3-30B-A3B with HGA RAM-cached router.

Tests whether the model can retrieve a hidden fact planted at various depths
inside a long filler context, at multiple total context lengths.

Usage (on the remote server, inside the HierarchicalGlobalAttention repo):
    python niah_benchmark.py
    python niah_benchmark.py --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
    python niah_benchmark.py --context-lengths 8192 16384 32768 --depths 10 25 50 75 90
    python niah_benchmark.py --mode dense          # compare against dense (may OOM at 32K)
    python niah_benchmark.py --no-hga --max-len 8192  # dense only, short context

Results are saved to niah_results.json and niah_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ---------------------------------------------------------------------------
# Default config — mirrors chat_qwen30b_fp8.py
# ---------------------------------------------------------------------------
MODEL_DEFAULT    = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
CHUNK_SIZE       = 64
KEEP_FIRST       = 2
KEEP_LAST        = 8
TOPK_CHUNKS      = 16
PREFILL_BLOCK    = 64
VRAM_CACHE_CHUNKS        = 500
VRAM_CACHE_RESERVE_GB    = 1.5

# ---------------------------------------------------------------------------
# Haystack / needle templates
# ---------------------------------------------------------------------------

# A pool of distinct "needles" so multiple runs don't share the same fact.
NEEDLES = [
    ("magic_code",        "The secret activation code is BRAVO-SEVEN-NINER-DELTA.",
                          r"BRAVO.SEVEN.NINER.DELTA"),
    ("treasure_city",     "The treasure is buried in the city of Marcellinville.",
                          r"Marcellinville"),
    ("password",          "The database password is Xq72#mPluto!9.",
                          r"Xq72.mPluto.9"),
    ("meeting_time",      "The critical meeting is scheduled for 03:47 AM on a Tuesday.",
                          r"03:47"),
    ("agent_name",        "The undercover agent's codename is Crimson Falcon.",
                          r"Crimson Falcon"),
    ("launch_key",        "The launch authorization key is ZETA-FOUR-KILO-ECHO.",
                          r"ZETA.FOUR.KILO.ECHO"),
    ("hidden_number",     "The winning lottery number hidden in this document is 847293.",
                          r"847293"),
    ("rare_element",      "The newly discovered element has been named Velorium.",
                          r"Velorium"),
    ("safe_combination",  "The safe combination is 19-left, 34-right, 7-left.",
                          r"19.*34.*7"),
    ("secret_planet",     "Astronomers have secretly named the new planet Zorbifax.",
                          r"Zorbifax"),
]

# Filler text paragraph — repeated to fill context.
_FILLER_PARA = (
    "The study of distributed systems requires careful consideration of consistency, "
    "availability, and partition tolerance. Engineers must balance these properties "
    "against latency requirements and operational complexity. Various consensus "
    "algorithms such as Raft and Paxos have been developed to address these challenges. "
    "Modern cloud infrastructure relies heavily on these foundations to deliver "
    "reliable services at scale. The emergence of microservices architectures has "
    "further complicated the landscape, requiring sophisticated orchestration tools "
    "and observability platforms. Research continues into new approaches that can "
    "reduce coordination overhead while maintaining strong correctness guarantees. "
)


def build_haystack(
    tokenizer,
    needle_text: str,
    target_tokens: int,
    depth_pct: float,
    seed: int = 42,
) -> str:
    """Build a filler document of ~target_tokens with needle inserted at depth_pct."""
    random.seed(seed)

    # Estimate tokens per filler paragraph.
    para_toks = len(tokenizer(_FILLER_PARA, add_special_tokens=False)["input_ids"])
    needle_toks = len(tokenizer(needle_text, add_special_tokens=False)["input_ids"])
    filler_needed = max(1, target_tokens - needle_toks)
    n_paras = max(1, filler_needed // para_toks + 1)

    paras = [_FILLER_PARA] * n_paras
    insert_idx = max(0, min(int(len(paras) * depth_pct / 100), len(paras) - 1))
    paras.insert(insert_idx, f"\n\n[IMPORTANT FACT]: {needle_text}\n\n")

    return "\n".join(paras)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def greedy_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new: int,
    prefill_block: int,
) -> str:
    """Blocked prefill + greedy decode, returns generated text."""
    with torch.inference_mode():
        device = prompt_ids.device
        eos_ids = (
            {tokenizer.eos_token_id}
            if isinstance(tokenizer.eos_token_id, int)
            else set(tokenizer.eos_token_id or [])
        )
        cache = DynamicCache()
        S = prompt_ids.shape[1]

        last_logits = None
        for s in range(0, S, prefill_block):
            e = min(s + prefill_block, S)
            cp = torch.arange(s, e, device=device)
            out = model(
                input_ids=prompt_ids[:, s:e],
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            last_logits = out.logits[:, -1]

        gen_ids: list[int] = []
        p = S
        nxt = int(last_logits.argmax(-1))
        for _ in range(max_new):
            if nxt in eos_ids:
                break
            gen_ids.append(nxt)
            cp = torch.tensor([p], device=device)
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            nxt = int(out.logits[:, -1].argmax(-1))
            p += 1
            if len(gen_ids) >= max_new:
                break

    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    model_mode:   str        # "hga" or "dense"
    context_len:  int
    depth_pct:    float
    needle_key:   str
    needle_text:  str
    answer_regex: str
    prompt_tokens: int
    response:     str
    hit:          bool       # needle found in response
    elapsed_s:    float
    ttft_s:       float


def run_trial(
    model,
    tokenizer,
    model_mode: str,
    context_len: int,
    depth_pct: float,
    needle_key: str,
    needle_text: str,
    answer_regex: str,
    prefill_block: int,
    max_new: int,
    seed: int,
    thinking: bool,
) -> TrialResult:
    haystack = build_haystack(
        tokenizer, needle_text, context_len, depth_pct, seed=seed
    )

    # Build the chat prompt.
    question = (
        f"Based on the document above, what is {needle_key.replace('_', ' ')}? "
        f"Reply with only the exact value, nothing else."
    )
    messages = [
        {"role": "user", "content": haystack + "\n\n" + question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    actual_tokens = input_ids.shape[1]

    t0 = time.perf_counter()

    # TTFT: time until first token (full prefill).
    with torch.inference_mode():
        device = input_ids.device
        cache = DynamicCache()
        S = input_ids.shape[1]
        last_logits = None
        for s in range(0, S, prefill_block):
            e = min(s + prefill_block, S)
            cp = torch.arange(s, e, device=device)
            out = model(
                input_ids=input_ids[:, s:e],
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            last_logits = out.logits[:, -1]
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Decode.
        eos_ids = (
            {tokenizer.eos_token_id}
            if isinstance(tokenizer.eos_token_id, int)
            else set(tokenizer.eos_token_id or [])
        )
        gen_ids: list[int] = []
        p = S
        nxt = int(last_logits.argmax(-1))
        for _ in range(max_new):
            if nxt in eos_ids:
                break
            gen_ids.append(nxt)
            cp = torch.tensor([p], device=device)
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            nxt = int(out.logits[:, -1].argmax(-1))
            p += 1

    elapsed = time.perf_counter() - t0
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    hit = bool(re.search(answer_regex, response, re.IGNORECASE))

    return TrialResult(
        model_mode=model_mode,
        context_len=context_len,
        depth_pct=depth_pct,
        needle_key=needle_key,
        needle_text=needle_text,
        answer_regex=answer_regex,
        prompt_tokens=actual_tokens,
        response=response,
        hit=hit,
        elapsed_s=elapsed,
        ttft_s=ttft,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gb(x: int) -> float:
    return x / 1024 ** 3


def load_model_hga(model_name: str, tokenizer):
    """Load model and patch with HGA RAM-cached router."""
    from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
        replace_qwen_attention_with_router,
    )
    print(f"Loading {model_name} with HGA router ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    n = replace_qwen_attention_with_router(
        model,
        mode="exact",
        cache_location="ram",
        keep_first=KEEP_FIRST,
        keep_last=KEEP_LAST,
        topk_chunks=TOPK_CHUNKS,
        chunk_size=CHUNK_SIZE,
        vram_cache_chunks=VRAM_CACHE_CHUNKS,
        vram_cache_reserve_gb=VRAM_CACHE_RESERVE_GB,
    )
    torch.cuda.synchronize()
    print(
        f"  Loaded in {time.perf_counter() - t0:.1f}s, HGA on {n} layers. "
        f"VRAM: {gb(torch.cuda.memory_allocated()):.1f} / "
        f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB",
        flush=True,
    )
    return model


def load_model_dense(model_name: str):
    """Load model without HGA (dense attention). May OOM at long contexts."""
    print(f"Loading {model_name} DENSE (no HGA) ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize()
    print(
        f"  Loaded in {time.perf_counter() - t0:.1f}s. "
        f"VRAM: {gb(torch.cuda.memory_allocated()):.1f} / "
        f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB",
        flush=True,
    )
    return model


def main():
    ap = argparse.ArgumentParser(description="Needle-in-a-Haystack benchmark for HGA")
    ap.add_argument("--model",           default=MODEL_DEFAULT)
    ap.add_argument("--mode",            choices=["hga", "dense", "both"], default="hga",
                    help="Which attention mode to test.")
    ap.add_argument("--context-lengths", type=int, nargs="+",
                    default=[8192, 16384, 32768],
                    help="Context lengths in tokens to test.")
    ap.add_argument("--depths",          type=float, nargs="+",
                    default=[5, 25, 50, 75, 95],
                    help="Needle depth as %% of document (0=start, 100=end).")
    ap.add_argument("--needles",         type=str, nargs="+",
                    default=None,
                    help="Needle keys to use (default: first 3). "
                         f"Available: {[n[0] for n in NEEDLES]}")
    ap.add_argument("--max-new",         type=int, default=64,
                    help="Max new tokens for the answer.")
    ap.add_argument("--thinking",        action="store_true",
                    help="Enable Qwen3 thinking mode.")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--output",          default="niah_results",
                    help="Output file stem (saves .json and .csv).")
    args = ap.parse_args()

    # Select needles.
    needle_pool = {n[0]: n for n in NEEDLES}
    if args.needles:
        selected = [needle_pool[k] for k in args.needles if k in needle_pool]
    else:
        selected = NEEDLES[:3]   # default: first 3 needles

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    modes = ["hga", "dense"] if args.mode == "both" else [args.mode]
    all_results: list[TrialResult] = []

    for mode in modes:
        if mode == "hga":
            model = load_model_hga(args.model, tokenizer)
        else:
            model = load_model_dense(args.model)

        total = len(args.context_lengths) * len(args.depths) * len(selected)
        done  = 0

        for ctx_len in args.context_lengths:
            for depth in args.depths:
                for needle_key, needle_text, answer_regex in selected:
                    done += 1
                    print(
                        f"[{mode}] Trial {done}/{total}: "
                        f"ctx={ctx_len}, depth={depth}%, needle={needle_key}",
                        flush=True,
                    )
                    try:
                        result = run_trial(
                            model=model,
                            tokenizer=tokenizer,
                            model_mode=mode,
                            context_len=ctx_len,
                            depth_pct=depth,
                            needle_key=needle_key,
                            needle_text=needle_text,
                            answer_regex=answer_regex,
                            prefill_block=PREFILL_BLOCK,
                            max_new=args.max_new,
                            seed=args.seed,
                            thinking=args.thinking,
                        )
                        status = "HIT ✓" if result.hit else "MISS ✗"
                        print(
                            f"  {status} | {result.prompt_tokens} tokens | "
                            f"TTFT {result.ttft_s:.1f}s | "
                            f"response: {result.response[:80]!r}",
                            flush=True,
                        )
                        all_results.append(result)
                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM — skipping this trial.", flush=True)
                        all_results.append(
                            TrialResult(
                                model_mode=mode,
                                context_len=ctx_len,
                                depth_pct=depth,
                                needle_key=needle_key,
                                needle_text=needle_text,
                                answer_regex=answer_regex,
                                prompt_tokens=-1,
                                response="OOM",
                                hit=False,
                                elapsed_s=0.0,
                                ttft_s=0.0,
                            )
                        )
                        torch.cuda.empty_cache()

        # Free model before loading the next mode.
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    json_path = args.output + ".json"
    csv_path  = args.output + ".csv"

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    fieldnames = list(asdict(all_results[0]).keys()) if all_results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([asdict(r) for r in all_results])

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<8} {'CtxLen':>8} {'Depth%':>8} {'Hits/Total':>12} {'Pass%':>8}")
    print("-" * 70)

    for mode in modes:
        for ctx_len in args.context_lengths:
            mode_ctx = [r for r in all_results
                        if r.model_mode == mode and r.context_len == ctx_len
                        and r.response != "OOM"]
            if not mode_ctx:
                continue
            for depth in args.depths:
                sub = [r for r in mode_ctx if r.depth_pct == depth]
                hits  = sum(r.hit for r in sub)
                total = len(sub)
                pct   = 100 * hits / total if total else 0
                print(f"{mode:<8} {ctx_len:>8} {depth:>8.0f} "
                      f"{hits:>5}/{total:<6} {pct:>7.1f}%")
            # Per context-length total.
            hits  = sum(r.hit for r in mode_ctx)
            total = len(mode_ctx)
            pct   = 100 * hits / total if total else 0
            print(f"{'':8} {ctx_len:>8} {'ALL':>8} "
                  f"{hits:>5}/{total:<6} {pct:>7.1f}%  ← overall")
            print()

    print(f"\nResults saved to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
