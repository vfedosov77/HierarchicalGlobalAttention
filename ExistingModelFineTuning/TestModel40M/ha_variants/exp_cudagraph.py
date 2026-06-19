#!/usr/bin/env python3
"""Validate + time the CUDA-graph decode against eager static decode (v2)."""
import os, sys, time
import torch
from transformers import GPT2TokenizerFast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TM_DIR = os.path.dirname(SCRIPT_DIR)
EMFT_DIR = os.path.dirname(TM_DIR)
ROOT_DIR = os.path.dirname(EMFT_DIR)
sys.path.insert(0, ROOT_DIR); sys.path.insert(0, TM_DIR); sys.path.insert(0, SCRIPT_DIR)
import benchmark_generation as bg
import bench_variants as bv

DEVICE = "cuda"


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "v2_static"
    ctx_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    gen = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    GA = bv.load_variant(variant)
    model = bv.build_ha_model_variant(GA, tok.vocab_size, tok.pad_token_id)
    bg.load_checkpoint(model, bg.HA_CHECKPOINT, strict=False)
    model.to(DEVICE).eval()
    # Tighten static buffers for this run.
    for layer in model.layers:
        if hasattr(layer.self_attn.attn, "decode_max_seq"):
            layer.self_attn.attn.decode_max_seq = ctx_len + gen + bg.CHUNK_SIZE

    ctx = torch.randint(0, tok.vocab_size, (1, ctx_len), device=DEVICE)

    # Eager static
    bv.decode_generate(model, ctx[:, :16], 4)
    torch.cuda.synchronize()
    te = []
    for _ in range(3):
        toks_e, t = bv.decode_generate(model, ctx, gen)
        te.append(t)
    te = min(te)
    print(f"  eager   : {te*1000:.1f} ms, {gen/te:.1f} tok/s")

    # CUDA graph
    try:
        tg_list = []
        toks_g = None
        for _ in range(3):
            toks_g, t = bv.decode_generate_cudagraph(model, ctx, gen)
            tg_list.append(t)
        tg = min(tg_list)
        print(f"  cudagraph: {tg*1000:.1f} ms, {gen/tg:.1f} tok/s  ({te/tg:.2f}x vs eager)")
        match = sum(1 for a, b in zip(toks_e, toks_g) if a == b) / len(toks_e)
        print(f"  token match eager-vs-cudagraph: {100*match:.2f}%  (first diff at "
              f"{next((i for i,(a,b) in enumerate(zip(toks_e,toks_g)) if a!=b), -1)})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CUDA graph failed:", repr(e)[:300])


if __name__ == "__main__":
    main()
