#!/usr/bin/env python3
"""Profile the HA decode path to find hot operations.

Loads the 40M HA model with the baseline (cleaned) attention, prefills a
context, then times token-by-token decode under torch.profiler.
"""
import os
import sys
import time

import torch
from transformers import GPT2TokenizerFast, DynamicCache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TM_DIR = os.path.dirname(SCRIPT_DIR)            # TestModel40M
EMFT_DIR = os.path.dirname(TM_DIR)              # ExistingModelFineTuning
ROOT_DIR = os.path.dirname(EMFT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TM_DIR)

import benchmark_generation as bg

DEVICE = "cuda"


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = bg.build_ha_model(tok.vocab_size, tok.pad_token_id)
    bg.load_checkpoint(model, bg.HA_CHECKPOINT, strict=False)
    model.to(DEVICE).eval()

    ctx_len = 512
    gen = 128
    input_ids = torch.randint(0, tok.vocab_size, (1, ctx_len), device=DEVICE)

    # Warmup + correctness run
    toks, t = bg.generate_tokens_ha(model, input_ids, 8)
    torch.cuda.synchronize()

    # Timed
    toks, t = bg.generate_tokens_ha(model, input_ids, gen)
    print(f"decode {gen} tokens: {t*1000:.1f} ms -> {gen/t:.1f} tok/s")

    # Profile
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        bg.generate_tokens_ha(model, input_ids, 32)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print("\n=== by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))


if __name__ == "__main__":
    main()
