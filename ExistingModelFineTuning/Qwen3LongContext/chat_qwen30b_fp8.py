#!/usr/bin/env python3
"""Interactive chat with Qwen3-30B-A3B-Instruct-2507-FP8.

Usage:
    source ~/my_env/bin/activate
    cd ~/HierarchicalGlobalAttention
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8

Commands during chat:
    /reset     clear conversation history
    /think     toggle thinking mode on/off (off by default)
    /exit      quit
"""

from __future__ import annotations

import sys
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MAX_NEW_TOKENS = 1024 + 2048


def gb(x: int) -> float:
    return x / 1024**3


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"

    print(f"Loading {MODEL} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s  "
          f"({gb(torch.cuda.memory_allocated()):.1f}GB / {gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB VRAM)\n",
          flush=True)

    history: list[dict] = []
    thinking = False

    print("Commands: /reset  /think  /exit")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye.")
            break
        if user_input == "/reset":
            history.clear()
            print("[history cleared]")
            continue
        if user_input == "/think":
            thinking = not thinking
            print(f"[thinking mode: {'ON' if thinking else 'OFF'}]")
            continue

        history.append({"role": "user", "content": user_input})

        template_kwargs: dict = {"enable_thinking": thinking}
        text = tok.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        inputs = tok(text, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
        )

        t_start = time.perf_counter()
        first_token_time: list[float] = []
        collected: list[str] = []

        def run():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        print("\nAssistant: ", end="", flush=True)
        for i, chunk in enumerate(streamer):
            if i == 0:
                first_token_time.append(time.perf_counter() - t_start)
            print(chunk, end="", flush=True)
            collected.append(chunk)

        thread.join()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start

        reply = "".join(collected)
        n_out = len(tok.encode(reply, add_special_tokens=False))
        sustained = (n_out - 1) / (dt - first_token_time[0]) if n_out > 1 and dt > first_token_time[0] else 0.0

        print(f"\n[{n_out} tokens | TTFT {first_token_time[0]:.1f}s | {sustained:.1f} tok/s]", flush=True)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
