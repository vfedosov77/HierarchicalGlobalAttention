#!/usr/bin/env python3
"""Smoke test: load Qwen3-30B-A3B-Instruct-2507-FP8 on a single 32GB GPU and
generate tokens, reporting TTFT and sustained decode speed separately.

Best practices applied:
- device_map="cuda": streams shards straight to GPU (host RAM is only 15GB < 30GB model)
- torch.inference_mode(): no autograd bookkeeping, faster than no_grad
- use_cache=True: KV cache enabled (default)
- enable_thinking=False: skips Qwen3 thinking-mode <think> tokens (otherwise
  they are generated and counted but stripped from display, inflating time)
- do_sample=False: greedy decode for reproducible timing
"""

from __future__ import annotations

import argparse
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def gb(x: int) -> float:
    return x / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--prompt", default="Give me a short explanation of rotary position embeddings.")
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode (off by default)")
    ap.add_argument("--no-cache", action="store_true", help="disable KV cache")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.reset_peak_memory_stats()

    print(f"[load] {MODEL} (device_map=cuda, attn={args.attn})", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation=args.attn,
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(
        f"[mem] after load: allocated={gb(torch.cuda.memory_allocated()):.2f}GB "
        f"reserved={gb(torch.cuda.memory_reserved()):.2f}GB",
        flush=True,
    )

    messages = [{"role": "user", "content": args.prompt}]
    # enable_thinking=False avoids generating <think>...</think> tokens that are
    # counted in throughput but stripped from the printed output.
    template_kwargs = {}
    if not args.thinking:
        try:
            template_kwargs["enable_thinking"] = False
        except Exception:
            pass
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    inputs = tok(text, return_tensors="pt").to("cuda")
    n_prompt = inputs["input_ids"].shape[-1]
    print(f"[gen] prompt tokens: {n_prompt}  thinking={'on' if args.thinking else 'off'}", flush=True)

    # Use a streamer to capture time-to-first-token (TTFT) separately.
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=not args.no_cache,
        streamer=streamer,
    )

    ttft: list[float] = []
    t_start = time.perf_counter()

    def run_generate():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()

    chunks: list[str] = []
    for i, chunk in enumerate(streamer):
        if i == 0:
            torch.cuda.synchronize()
            ttft.append(time.perf_counter() - t_start)
        chunks.append(chunk)

    thread.join()
    torch.cuda.synchronize()
    dt_total = time.perf_counter() - t_start

    generated = "".join(chunks)
    # Count actual output tokens (re-encode to get exact count).
    n_new = len(tok.encode(generated, add_special_tokens=False))

    print(f"[gen] TTFT (time to first token): {ttft[0]:.2f}s", flush=True)
    print(f"[gen] {n_new} output tokens in {dt_total:.2f}s total", flush=True)
    if dt_total > ttft[0] and n_new > 1:
        sustained = (n_new - 1) / (dt_total - ttft[0])
        print(f"[gen] sustained decode: {sustained:.2f} tok/s  (excluding TTFT)", flush=True)
    print(f"[gen] overall throughput: {n_new / dt_total:.2f} tok/s", flush=True)
    print(
        f"[mem] peak: allocated={gb(torch.cuda.max_memory_allocated()):.2f}GB "
        f"reserved={gb(torch.cuda.max_memory_reserved()):.2f}GB",
        flush=True,
    )
    print("=" * 60)
    print(generated)
    print("=" * 60)


if __name__ == "__main__":
    main()
