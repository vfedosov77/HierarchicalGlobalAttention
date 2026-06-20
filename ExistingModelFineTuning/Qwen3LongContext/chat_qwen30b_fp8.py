#!/usr/bin/env python3
"""Interactive chat with Qwen3-30B-A3B-Instruct-2507-FP8 on a **RAM-cached KvRouter**.

The attention of every layer is replaced by ``QwenRoutedAttention`` (exact mode) backed by a
``RamKVCacheStore``: the full KV cache lives in host RAM and only the routed chunks (a few sink
+ local + top-k middle chunks) are pulled to VRAM each step.  VRAM use is therefore bounded by
the model weights regardless of context length, so this fits long histories on a 32GB card.

Because the routed attention manages its own KV in the router (not the HF ``DynamicCache``),
this script does **not** use ``model.generate`` — it runs a manual blocked-prefill + streaming
greedy decode loop, passing ``cache_position`` explicitly.

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

import os

# Set before CUDA initialises: avoids the FP8 Triton matmul autotuner OOMing in the small VRAM
# headroom left after the ~29GB of weights, and reduces allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    replace_qwen_attention_with_router,
)


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MAX_NEW_TOKENS = 32*1024

# --- RAM-cached router config (chunk_size 64) ---
CHUNK_SIZE = 64
KEEP_FIRST = 2      # always-resident leading chunks (attention sinks): 128 tokens
KEEP_LAST = 16       # always-resident trailing chunks (local context): 1024 tokens
TOPK_CHUNKS = 20    # routed middle chunks selected per step (per KV-head): up to 1024 tokens
PREFILL_BLOCK = 64  # prefill is fed in blocks of this many tokens (bounds activation peak)
VRAM_CACHE_CHUNKS = 512  # LRU VRAM cache of chunk KV: recurring chunks stay resident (~0.8GB)


def gb(x: int) -> float:
    return x / 1024**3


@torch.inference_mode()
def stream_generate(model, tok, input_ids: torch.Tensor, max_new: int, block: int):
    """Blocked prefill + token-by-token greedy decode, streaming text to stdout as it is produced.

    The router (attached to a fresh ``DynamicCache``) holds the KV in RAM; the cache itself stays
    empty, so we drive positions via explicit ``cache_position``.  Returns (reply, n_tokens, ttft).
    """
    device = input_ids.device
    eos_ids = {tok.eos_token_id} if isinstance(tok.eos_token_id, int) else set()
    cache = DynamicCache()  # router attaches itself here; HF KV cache stays empty
    S = input_ids.shape[1]

    t0 = time.perf_counter()
    last = None
    for s in range(0, S, block):
        e = min(s + block, S)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=input_ids[:, s:e], past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        last = out.logits[:, -1]
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    gen_ids: list[int] = []
    printed = ""
    p = S
    nxt = int(last.argmax(-1))
    for _ in range(max_new):
        if nxt in eos_ids:
            break
        gen_ids.append(nxt)
        text = tok.decode(gen_ids, skip_special_tokens=True)
        print(text[len(printed):], end="", flush=True)
        printed = text
        cp = torch.tensor([p], device=device)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        nxt = int(out.logits[:, -1].argmax(-1))
        p += 1
    return printed, len(gen_ids), ttft


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

    n = replace_qwen_attention_with_router(
        model, mode="exact", cache_location="ram",
        keep_first=KEEP_FIRST, keep_last=KEEP_LAST, topk_chunks=TOPK_CHUNKS, chunk_size=CHUNK_SIZE,
        vram_cache_chunks=VRAM_CACHE_CHUNKS,
    )
    torch.cuda.synchronize()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s  "
          f"({gb(torch.cuda.memory_allocated()):.1f}GB / "
          f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB VRAM)", flush=True)
    print(f"RAM-cached router on {n} layers: keep_first={KEEP_FIRST} ({KEEP_FIRST*CHUNK_SIZE} tok), "
          f"keep_last={KEEP_LAST} ({KEEP_LAST*CHUNK_SIZE} tok), topk_chunks={TOPK_CHUNKS} "
          f"({TOPK_CHUNKS*CHUNK_SIZE} tok); KV cache lives in host RAM.\n", flush=True)

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

        text = tok.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        input_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        n_prompt = input_ids.shape[1]

        print("\nAssistant: ", end="", flush=True)
        t_start = time.perf_counter()
        reply, n_out, ttft = stream_generate(model, tok, input_ids, MAX_NEW_TOKENS, PREFILL_BLOCK)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start

        sustained = (n_out - 1) / (dt - ttft) if n_out > 1 and dt > ttft else 0.0
        print(f"\n[{n_prompt} ctx | {n_out} tokens | TTFT {ttft:.1f}s | {sustained:.1f} tok/s | "
              f"peak {gb(torch.cuda.max_memory_allocated()):.1f}GB]", flush=True)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
