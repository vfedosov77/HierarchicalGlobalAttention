#!/usr/bin/env python3
"""Quick experiment: does torch.compile help baseline HA decode?"""
import os, sys, time
import torch
from transformers import GPT2TokenizerFast, DynamicCache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TM_DIR = os.path.dirname(SCRIPT_DIR)
EMFT_DIR = os.path.dirname(TM_DIR)
ROOT_DIR = os.path.dirname(EMFT_DIR)
sys.path.insert(0, ROOT_DIR); sys.path.insert(0, TM_DIR); sys.path.insert(0, SCRIPT_DIR)
import benchmark_generation as bg
import bench_variants as bv

DEVICE = "cuda"


def run(model, ctx, gen, label):
    w = ctx[:, :16]
    bv.decode_generate(model, w, 4)
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        _, t = bv.decode_generate(model, ctx, gen)
        times.append(t)
    avg = sum(times) / len(times)
    print(f"  [{label}] {avg*1000:.1f} ms, {gen/avg:.1f} tok/s")


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    GA = bv.load_variant("v0_baseline")
    model = bv.build_ha_model_variant(GA, tok.vocab_size, tok.pad_token_id)
    bg.load_checkpoint(model, bg.HA_CHECKPOINT, strict=False)
    model.to(DEVICE).eval()

    ctx = torch.randint(0, tok.vocab_size, (1, 512), device=DEVICE)
    run(model, ctx, 128, "eager")

    # Try compiling each layer's attention forward
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    print(f"Compiling with mode={mode} ...")
    for layer in model.layers:
        layer.self_attn.attn.forward = torch.compile(
            layer.self_attn.attn.forward, mode=mode, dynamic=True)
    try:
        run(model, ctx, 128, f"compiled-{mode}")
    except Exception as e:
        print("compile failed:", repr(e)[:500])


if __name__ == "__main__":
    main()
