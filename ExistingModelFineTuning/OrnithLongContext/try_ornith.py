#!/usr/bin/env python3
"""Real Ornith-1.0-9B run on a 16GB card: 4-bit NF4 + fp16, HGA on the 8 full-attn layers.

Ornith-1.0-9B is a Qwen3.5 hybrid (24 linear_attention + 8 full_attention layers).  bf16
weights (~18GB) don't fit a 16GB card, so we load the body in **4-bit NF4** with fp16
compute (Turing SM7.5 has no FP8 and slow bf16).  HGA is patched onto the 8 full-attention
layers only; their historical KV lives in host RAM (``cache_location="ram"``) and only the
routed working set is pulled to VRAM, so VRAM stays flat as context grows.

Run (always from repo root, always the project .venv):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -m ExistingModelFineTuning.OrnithLongContext.try_ornith --smoke
    python -m ExistingModelFineTuning.OrnithLongContext.try_ornith --needle --needle-tokens 32768

See docs/ORNITH_HGA.md §8-§9 for the memory budget and acceptance logic.
"""

from __future__ import annotations

import argparse
import os
import time

# Antifragmentation for the tight 16GB card — must be set before the first CUDA alloc.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from .ornith_routed_attention import (
    replace_ornith_attention_with_router,
    restore_ornith_attention,
)

MODEL = "deepreinforce-ai/Ornith-1.0-9B"


def gb(x: int) -> float:
    return x / 1024**3


def _load_model(model_id: str, load: str, attn: str):
    """Load Ornith in 4-bit NF4 (default) or fp16; returns (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # The checkpoint is an image-text-to-text (Qwen3_5ForConditionalGeneration) model; load the
    # generative class via AutoModelForImageTextToText so .generate() and the text stack are intact.
    from transformers import AutoModelForImageTextToText as _AutoGen

    kwargs = dict(trust_remote_code=True, device_map="cuda", attn_implementation=attn)
    if load == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            # Keep the lm_head / norms in fp16 (quantizing them hurts quality for little gain).
            llm_int8_skip_modules=["lm_head", "norm"],
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = _AutoGen.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


def _generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
    print(f"[gen] prompt={n_prompt} tok, new={out.shape[-1] - n_prompt} tok in {dt:.1f}s "
          f"({(out.shape[-1] - n_prompt) / dt:.2f} tok/s)", flush=True)
    return gen


def _blocked_generate(model, tok, prompt: str, max_new_tokens: int, block: int,
                      thinking: bool = False) -> str:
    """Blocked prefill + manual greedy decode for long context.

    HF ``generate`` prefills the whole prompt in one forward — on Ornith that makes the
    *linear_attention* (GatedDeltaNet) layers materialise O(S) chunk intermediates and OOM at
    32K on the torch fallback path.  Feeding the prompt in ``block``-token slices keeps those
    layers' memory bounded (they carry only a constant recurrent state across slices), while the
    HGA-routed full-attention layers offload their KV to RAM either way, so VRAM stays flat.
    """
    messages = [{"role": "user", "content": prompt}]
    tmpl = {}
    if not thinking:
        tmpl["enable_thinking"] = False
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **tmpl)
    except (TypeError, ValueError):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
    n_prompt = ids.shape[-1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pkv = None
        # HGA does not grow the model's KV cache (full-attn KV lives in the router's store), so
        # the model's own position bookkeeping would stall — drive positions explicitly.
        for i in range(0, n_prompt, block):
            chunk = ids[:, i:i + block]
            cp = torch.arange(i, i + chunk.shape[1], device=model.device)
            out = model(input_ids=chunk, past_key_values=pkv, use_cache=True,
                        position_ids=cp.unsqueeze(0), cache_position=cp)
            pkv = out.past_key_values
        t_prefill = time.perf_counter() - t0
        gen_ids = []
        cur = out.logits[:, -1:].argmax(-1)
        for s in range(max_new_tokens):
            gen_ids.append(cur)
            p = n_prompt + s
            out = model(input_ids=cur, past_key_values=pkv, use_cache=True,
                        position_ids=torch.tensor([[p]], device=model.device),
                        cache_position=torch.tensor([p], device=model.device))
            pkv = out.past_key_values
            cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = tok.decode(torch.cat(gen_ids, dim=1)[0], skip_special_tokens=True)
    print(f"[gen] blocked prefill {n_prompt} tok (block={block}) in {t_prefill:.1f}s, "
          f"+{max_new_tokens} new in {dt - t_prefill:.1f}s "
          f"({max_new_tokens / max(dt - t_prefill, 1e-6):.2f} tok/s decode)", flush=True)
    return gen


def _needle_prompt(tok, target_tokens: int, magic: str) -> str:
    """Build a ~target_tokens filler with a single 'magic number' fact buried in the middle."""
    here = os.path.dirname(os.path.abspath(__file__))
    filler_path = os.path.join(os.path.dirname(os.path.dirname(here)), "TrainData",
                               "The-Master-and-Margarita.txt")
    with open(filler_path, "r", encoding="utf-8", errors="ignore") as f:
        book = f.read()
    ids = tok(book, add_special_tokens=False)["input_ids"]
    if len(ids) < target_tokens:
        ids = (ids * (target_tokens // max(1, len(ids)) + 1))
    ids = ids[:target_tokens]
    filler = tok.decode(ids, skip_special_tokens=True)
    needle = f"\n\nRemember this: the secret access code is {magic}.\n\n"
    mid = len(filler) // 2
    return filler[:mid] + needle + filler[mid:] + \
        "\n\nQuestion: What is the secret access code mentioned above? Answer with the code only."


def _verify(model, tok, n_tokens: int, chunk_size: int) -> None:
    """Compare stock vs full-coverage HGA logits on the real model (prefill AND decode).

    Full coverage (keep_first ≥ all chunks token-level, no routed middle, no active-group
    summaries) makes the router reproduce plain dense causal attention, so the patched logits
    must match the stock logits to fp16 precision.  Tests both the single-forward prefill and a
    few autoregressive DECODE steps (the path the needle test actually exercises).
    """
    from .ornith_routed_attention import _iter_ornith_attention_layers, restore_ornith_attention

    here = os.path.dirname(os.path.abspath(__file__))
    book_path = os.path.join(os.path.dirname(os.path.dirname(here)), "TrainData",
                             "The-Master-and-Margarita.txt")
    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        book = f.read()
    ids = tok(book, add_special_tokens=False)["input_ids"][:n_tokens]
    input_ids = torch.tensor([ids], device=model.device)
    n_chunks = (len(ids) + chunk_size - 1) // chunk_size
    n_decode = 8

    def run(tag):
        with torch.inference_mode():
            L = input_ids.shape[1]
            pos = torch.arange(L, device=model.device).unsqueeze(0)
            out = model(input_ids=input_ids, position_ids=pos,
                        cache_position=torch.arange(L, device=model.device), use_cache=True)
            pkv = out.past_key_values
            step_logits = [out.logits[:, -1].float()]
            cur = out.logits[:, -1:].argmax(-1)
            for s in range(n_decode):
                p = L + s
                out = model(input_ids=cur, past_key_values=pkv,
                            position_ids=torch.tensor([[p]], device=model.device),
                            cache_position=torch.tensor([p], device=model.device), use_cache=True)
                pkv = out.past_key_values
                step_logits.append(out.logits[:, -1].float())
                cur = out.logits[:, -1:].argmax(-1)
        return step_logits, out

    stock_steps, stock_out = run("stock")
    # keep the full-sequence prefill logits too
    with torch.inference_mode():
        stock_prefill = model(input_ids=input_ids, use_cache=False).logits.float()

    n = replace_ornith_attention_with_router(
        model, cache_location="ram", chunk_size=chunk_size, group_size=chunk_size,
        keep_first=n_chunks + 2, keep_last=0, topk_chunks=0, topk_groups=0,
    )
    for layer in _iter_ornith_attention_layers(model):
        layer.self_attn._cfg.current_group_summaries = False
    with torch.inference_mode():
        hga_prefill = model(input_ids=input_ids, use_cache=False).logits.float()
    hga_steps, _ = run("hga")

    pre_diff = (hga_prefill - stock_prefill).abs()
    print(f"[verify] {len(ids)} tok ({n_chunks} chunks), patched {n} layers, full coverage:", flush=True)
    print(f"[verify]   PREFILL max|Δ|={pre_diff.max().item():.4f}  mean|Δ|={pre_diff.mean().item():.5f}  "
          f"argmax match={(hga_prefill[:, -1].argmax(-1) == stock_prefill[:, -1].argmax(-1)).item()}", flush=True)
    for i, (a, b) in enumerate(zip(stock_steps, hga_steps)):
        d = (a - b).abs()
        match = (a.argmax(-1) == b.argmax(-1)).item()
        print(f"[verify]   DECODE step {i}: max|Δ|={d.max().item():.4f}  mean|Δ|={d.mean().item():.5f}  "
              f"argmax match={match}", flush=True)
    restore_ornith_attention(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--load", default="4bit", choices=["4bit", "fp16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--smoke", action="store_true", help="short coherent-generation smoke test")
    ap.add_argument("--prompt", default="In two sentences, what is rotary position embedding?")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--needle", action="store_true", help="needle-in-haystack retrieval test")
    ap.add_argument("--needle-tokens", type=int, default=32768)
    ap.add_argument("--needle-new", type=int, default=128, help="decode tokens for the needle answer")
    ap.add_argument("--thinking", action="store_true", help="allow the model's chain-of-thought")
    ap.add_argument("--prefill-block", type=int, default=2048,
                    help="blocked-prefill slice (multiple of chunk_size) to bound linear-layer memory")
    ap.add_argument("--magic", default="ZQ-7731-XK")
    ap.add_argument("--no-patch", action="store_true", help="baseline: do not install HGA")
    ap.add_argument("--verify", action="store_true",
                    help="compare stock vs full-coverage HGA logits on the real model (short ctx)")
    ap.add_argument("--verify-tokens", type=int, default=256)
    # HGA knobs (docs/ORNITH_HGA.md §8).
    ap.add_argument("--cache-location", default="ram", choices=["ram", "fs", "vram"])
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--topk-chunks", type=int, default=16)
    ap.add_argument("--topk-groups", type=int, default=32)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.reset_peak_memory_stats()

    print(f"[load] {args.model} load={args.load} attn={args.attn} "
          f"(PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')})", flush=True)
    t0 = time.perf_counter()
    model, tok = _load_model(args.model, args.load, args.attn)
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter() - t0:.1f}s  "
          f"VRAM allocated={gb(torch.cuda.memory_allocated()):.2f}GB "
          f"reserved={gb(torch.cuda.memory_reserved()):.2f}GB", flush=True)

    if args.verify:
        _verify(model, tok, args.verify_tokens, args.chunk_size)
        return

    if not args.no_patch:
        n = replace_ornith_attention_with_router(
            model, cache_location=args.cache_location,
            chunk_size=args.chunk_size, group_size=args.group_size,
            keep_first=args.keep_first, keep_last=args.keep_last,
            topk_chunks=args.topk_chunks, topk_groups=args.topk_groups,
        )
        print(f"[hga] patched {n} full-attention layers (cache={args.cache_location})", flush=True)

    if args.smoke or not args.needle:
        print("[smoke] generating…", flush=True)
        text = _generate(model, tok, args.prompt, args.max_new_tokens)
        print("=" * 60)
        print(text)
        print("=" * 60)

    if args.needle:
        print(f"[needle] building ~{args.needle_tokens}-token haystack (magic={args.magic})…", flush=True)
        prompt = _needle_prompt(tok, args.needle_tokens, args.magic)
        ans = _blocked_generate(model, tok, prompt, args.needle_new, args.prefill_block,
                                thinking=args.thinking)
        hit = args.magic.replace(" ", "") in ans.replace(" ", "")
        print(f"[needle] answer={ans!r}", flush=True)
        print(f"[needle] {'HIT' if hit else 'MISS'} (magic={args.magic})", flush=True)

    print(f"[mem] peak VRAM allocated={gb(torch.cuda.max_memory_allocated()):.2f}GB "
          f"reserved={gb(torch.cuda.max_memory_reserved()):.2f}GB", flush=True)

    if not args.no_patch:
        restore_ornith_attention(model)


if __name__ == "__main__":
    main()
