"""Break down VLM time: HGA substeps vs Flash-2 / SDPA, same tensors.

Uses Alpamayo's Qwen3-VL text shape (Hq=32, Hkv=8, D=128, ~3K tokens)
and a real decoder layer (q_proj / RoPE / o_proj / MLP) so only the
attention backend changes.

    /home/vladimir/my_env/bin/python -m SpecializedKernels.Len3000HeadDim128.profile_vlm_parts
    /home/vladimir/my_env/bin/python -m SpecializedKernels.Len3000HeadDim128.profile_vlm_parts --seqlen 3000 --layers 36
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _median_us(fn, warmup: int = 15, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2] * 1e3


class SectionTimer:
    """Accumulate CUDA-event spans; call ``sync()`` before reading."""

    def __init__(self) -> None:
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def span(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timer = self

        class _Ctx:
            def __enter__(self):
                start.record()
                return self

            def __exit__(self, *exc):
                end.record()
                timer._events[name].append((start, end))

        return _Ctx()

    def ms(self, name: str) -> float:
        evs = self._events.get(name, [])
        if not evs:
            return 0.0
        return sum(s.elapsed_time(e) for s, e in evs)

    def reset(self) -> None:
        self._events.clear()


def _wrap_hga(timer: SectionTimer):
    from SpecializedKernels.Len3000HeadDim128 import kernel as kn
    from SpecializedKernels.Len3000HeadDim128.attention import vlm_prefill_attention
    from SpecializedKernels.Len3000HeadDim128.kernel import _vlm_hga2_kernel

    orig_fill = kn.fill_hga_means
    orig_route = kn.chunk_route_vlm_fast
    orig_kern = _vlm_hga2_kernel

    def fill(*a, **k):
        with timer.span("hga_means"):
            return orig_fill(*a, **k)

    def route(*a, **k):
        with timer.span("hga_route"):
            return orig_route(*a, **k)

    class _K:
        def __getitem__(self, grid):
            def launch(*a, **k):
                with timer.span("hga_attend"):
                    return orig_kern[grid](*a, **k)
            return launch

    kn.fill_hga_means = fill
    kn.chunk_route_vlm_fast = route
    import SpecializedKernels.Len3000HeadDim128.attention as at
    at.fill_hga_means = fill
    at.chunk_route_vlm_fast = route
    at._vlm_hga2_kernel = _K()
    return vlm_prefill_attention


def _text_cfg(seqlen: int):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    return Qwen3VLTextConfig(
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=max(4096, seqlen),
        rms_norm_eps=1e-6,
        hidden_act="silu",
        vocab_size=152064,
        rope_parameters={"rope_theta": 5_000_000.0, "rope_type": "default"},
    )


def _rope(seqlen: int, head_dim: int, device, dtype):
    # [1, S, D] — apply_rotary unsqueeze_dim=1 broadcasts over heads.
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    half = head_dim // 2
    freq = 1.0 / (1e6 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.outer(pos, freq)
    emb = torch.cat([ang, ang], dim=-1)
    cos = emb.cos()[None, :, :].to(dtype)
    sin = emb.sin()[None, :, :].to(dtype)
    return cos, sin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=3000)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    s = args.seqlen
    hq, kvh, d = 32, 8, 128
    scale = d ** -0.5
    dtype = torch.bfloat16
    device = torch.device("cuda")
    print(f"GPU  {torch.cuda.get_device_name(0)}")
    print(f"S={s}  Hq={hq} Hkv={kvh} D={d}  layers={args.layers}  bf16")
    try:
        import flash_attn
        print(f"flash_attn {flash_attn.__version__}  torch {torch.__version__}")
        has_fa = True
    except ImportError:
        print(f"flash_attn MISSING  torch {torch.__version__}")
        has_fa = False

    from SpecializedKernels.Len3000HeadDim128.attention import vlm_prefill_attention

    q = torch.randn(1, hq, s, d, device=device, dtype=dtype)
    k = torch.randn(1, kvh, s, d, device=device, dtype=dtype)
    v = torch.randn_like(k)
    out = torch.empty(1, s, hq, d, device=device, dtype=dtype)
    gk = torch.empty(1, kvh, 256, d, device=device, dtype=dtype)
    ck = torch.empty(1, kvh, 32, d, device=device, dtype=dtype)
    rt = torch.empty(1, hq, 32, 4, device=device, dtype=torch.int32)

    timer = SectionTimer()
    _wrap_hga(timer)

    def hga():
        vlm_prefill_attention(q, k, v, softmax_scale=scale, out=out, group_k=gk, chunk_k=ck, route=rt)

    def sdpa():
        F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)

    fa = None
    if has_fa:
        from flash_attn import flash_attn_func
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        def fa():
            flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=scale)

    for _ in range(8):
        hga()
        sdpa()
        if fa:
            fa()
    torch.cuda.synchronize()

    print("\n=== Isolated attention (same Q/K/V, S={}) ===".format(s))
    print(f"  HGA total (means+route+attend)  {_median_us(hga, iters=args.iters):8.1f} µs")
    timer.reset()
    for _ in range(args.iters):
        hga()
    torch.cuda.synchronize()
    print(f"    means                         {timer.ms('hga_means') / args.iters * 1e3:8.1f} µs")
    print(f"    chunk route                   {timer.ms('hga_route') / args.iters * 1e3:8.1f} µs")
    print(f"    attend kernel                 {timer.ms('hga_attend') / args.iters * 1e3:8.1f} µs")
    print(f"  SDPA                            {_median_us(sdpa, iters=args.iters):8.1f} µs")
    if fa:
        print(f"  FlashAttn2 bf16 (dense Alpamayo){_median_us(fa, iters=args.iters):8.1f} µs")

    # One real Qwen3-VL decoder layer: only the attention backend changes.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    from SpecializedKernels.Len3000HeadDim128.attention import (
        VLM_ATTN_NAME,
        attach_to_alpamayo,
        hf_vlm_attention,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cfg = _text_cfg(s)
    cfg._attn_implementation = "flash_attention_2"
    layer = Qwen3VLTextDecoderLayer(cfg, 0).to(device=device, dtype=dtype).eval()
    hidden = torch.randn(1, s, cfg.hidden_size, device=device, dtype=dtype)
    cos, sin = _rope(s, d, device, dtype)
    pos = torch.arange(s, device=device)[None, :]

    # Preallocate HGA buffers on the layer (as adapt_alpamayo does).
    layer.self_attn._alpamayo_group_k = torch.empty(1, kvh, 256, d, device=device, dtype=dtype)
    layer.self_attn._alpamayo_chunk_k = torch.empty(1, kvh, 32, d, device=device, dtype=dtype)
    layer.self_attn._alpamayo_route = torch.empty(1, hq, 32, 4, device=device, dtype=torch.int32)
    layer.self_attn._alpamayo_out_bshd = torch.empty(1, s + 8, hq, d, device=device, dtype=dtype)

    if VLM_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(VLM_ATTN_NAME, hf_vlm_attention)

    def layer_flash():
        layer.self_attn.config._attn_implementation = "flash_attention_2"
        cfg._attn_implementation = "flash_attention_2"
        return layer(hidden, (cos, sin), attention_mask=None, position_ids=pos)

    def layer_hga():
        layer.self_attn.config._attn_implementation = VLM_ATTN_NAME
        cfg._attn_implementation = VLM_ATTN_NAME
        return layer(hidden, (cos, sin), attention_mask=None, position_ids=pos)

    def layer_sdpa():
        layer.self_attn.config._attn_implementation = "sdpa"
        cfg._attn_implementation = "sdpa"
        return layer(hidden, (cos, sin), attention_mask=None, position_ids=pos)

    print("\n=== One Qwen3-VL decoder layer (q_proj+RoPE+attn+o_proj+MLP), S={} ===".format(s))
    for _ in range(6):
        layer_flash()
        layer_hga()
        layer_sdpa()
    torch.cuda.synchronize()
    us_f = _median_us(layer_flash, warmup=8, iters=args.iters)
    us_h = _median_us(layer_hga, warmup=8, iters=args.iters)
    us_s = _median_us(layer_sdpa, warmup=8, iters=args.iters)
    print(f"  layer Flash-2                   {us_f:8.1f} µs")
    print(f"  layer HGA                       {us_h:8.1f} µs")
    print(f"  layer SDPA                      {us_s:8.1f} µs")

    # Sub-module hooks on one layer.
    def _time_parts(run, label: str) -> None:
        t = SectionTimer()
        hs = []

        def pre_attn(m, inp):
            m._e0 = torch.cuda.Event(True)
            m._e1 = torch.cuda.Event(True)
            m._e0.record()

        def post_attn(m, inp, out):
            m._e1.record()
            t._events["attn"].append((m._e0, m._e1))

        def pre_mlp(m, inp):
            m._e0 = torch.cuda.Event(True)
            m._e1 = torch.cuda.Event(True)
            m._e0.record()

        def post_mlp(m, inp, out):
            m._e1.record()
            t._events["mlp"].append((m._e0, m._e1))

        hs.append(layer.self_attn.register_forward_pre_hook(pre_attn))
        hs.append(layer.self_attn.register_forward_hook(post_attn))
        hs.append(layer.mlp.register_forward_pre_hook(pre_mlp))
        hs.append(layer.mlp.register_forward_hook(post_mlp))
        timer.reset()
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
        n = args.iters
        print(f"  {label} split (avg over {n}):")
        print(f"    self_attn (qkv+rope+backend+o) {t.ms('attn') / n * 1e3:8.1f} µs")
        print(f"    mlp                             {t.ms('mlp') / n * 1e3:8.1f} µs")
        if timer.ms("hga_means"):
            print(f"      hga means                     {timer.ms('hga_means') / n * 1e3:8.1f} µs")
            print(f"      hga route                     {timer.ms('hga_route') / n * 1e3:8.1f} µs")
            print(f"      hga attend                    {timer.ms('hga_attend') / n * 1e3:8.1f} µs")
        for h in hs:
            h.remove()

    _time_parts(layer_flash, "Flash-2")
    _time_parts(layer_hga, "HGA")

    nL = args.layers
    print(f"\n=== {nL} layers × one-layer median (same compute as full LM prefill) ===")
    print(f"  Flash-2  {nL * us_f / 1000:6.2f} ms")
    print(f"  HGA      {nL * us_h / 1000:6.2f} ms   delta {nL * (us_h - us_f) / 1000:+6.2f} ms")
    print(f"  SDPA     {nL * us_s / 1000:6.2f} ms")
    print("\nIf this HGA-Flash delta is negative (HGA faster) but ROS vlm= is")
    print("higher, the extra time is outside attention (longer sequence / CoT,")
    print("vision, host) — not the kernel losing to Flash.")


if __name__ == "__main__":
    main()
