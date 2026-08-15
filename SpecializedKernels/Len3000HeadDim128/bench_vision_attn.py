"""Isolated vision-attention microbench: FA / SDPA / RoPE / MLP at Alpamayo shape.

    /home/vladimir/my_env/bin/python -m SpecializedKernels.Len3000HeadDim128.bench_vision_attn
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def median_us(fn, warmup=15, iters=40) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2] * 1e3


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_fp32(q, k, cos, sin):
    qf, kf = q.float(), k.float()
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    qe = (qf * c) + (rotate_half(qf) * s)
    ke = (kf * c) + (rotate_half(kf) * s)
    return qe.to(q.dtype), ke.to(k.dtype)


def rope_bf16(q, k, cos, sin):
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_img, s, h, d = 12, 720, 16, 72
    n_layers = 27
    hidden, inter = 1152, 4304
    print(f"GPU {torch.cuda.get_device_name(0)}  sm={torch.cuda.get_device_capability()}")
    print(f"shape  images={n_img} S={s} H={h} D={d}  layers={n_layers}  bf16")

    # Packed [1, H, N*S, D] as Qwen3-VL vision after transpose
    q = torch.randn(1, h, n_img * s, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.arange(0, n_img * s + 1, s, device=device, dtype=torch.int32)

    # Batched [B, H, S, D]
    qb = torch.randn(n_img, h, s, d, device=device, dtype=dtype)
    kb = torch.randn_like(qb)
    vb = torch.randn_like(qb)

    # Flash layout [B, S, H, D]
    q_fa = qb.transpose(1, 2).contiguous()
    k_fa = kb.transpose(1, 2).contiguous()
    v_fa = vb.transpose(1, 2).contiguous()

    # Packed flash layout [total, H, D]
    q_var = q.transpose(1, 2).contiguous().squeeze(0)  # [N*S, H, D]
    k_var = k.transpose(1, 2).contiguous().squeeze(0)
    v_var = v.transpose(1, 2).contiguous().squeeze(0)

    scale = d ** -0.5
    has_fa = False
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        import flash_attn

        print(f"flash_attn {flash_attn.__version__}")
        has_fa = True
    except Exception as exc:
        print("flash_attn missing:", exc)

    print("\n=== One layer, all 12 images ===")

    def sdpa_batched():
        return F.scaled_dot_product_attention(qb, kb, vb, is_causal=False, scale=scale)

    def sdpa_loop():
        outs = []
        for i in range(n_img):
            outs.append(
                F.scaled_dot_product_attention(
                    q[:, :, i * s : (i + 1) * s],
                    k[:, :, i * s : (i + 1) * s],
                    v[:, :, i * s : (i + 1) * s],
                    is_causal=False,
                    scale=scale,
                )
            )
        return torch.cat(outs, dim=2)

    def sdpa_packed_wrong():
        # One 8640-token softmax (NOT what Qwen does) — lower bound if packed globally
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

    print(f"  SDPA batched B={n_img} S={s}     {median_us(sdpa_batched):8.1f} µs")
    print(f"  SDPA 12 sequential calls         {median_us(sdpa_loop):8.1f} µs")
    print(f"  SDPA packed 8640 (wrong alg)     {median_us(sdpa_packed_wrong):8.1f} µs")

    if has_fa:
        def fa_batched():
            return flash_attn_func(q_fa, k_fa, v_fa, causal=False, softmax_scale=scale)

        def fa_varlen():
            return flash_attn_varlen_func(
                q_var, k_var, v_var,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=s, max_seqlen_k=s,
                causal=False, softmax_scale=scale,
            )

        print(f"  Flash-2 batched B={n_img}          {median_us(fa_batched):8.1f} µs")
        print(f"  Flash-2 varlen (Qwen path)       {median_us(fa_varlen):8.1f} µs")

        for dd in (64, 80, 128):
            qfd = torch.randn(n_img, s, h, dd, device=device, dtype=dtype)
            kfd = torch.randn_like(qfd)
            vfd = torch.randn_like(qfd)
            qvd = torch.randn(n_img * s, h, dd, device=device, dtype=dtype)
            kvd = torch.randn_like(qvd)
            vvd = torch.randn_like(qvd)
            sc = dd ** -0.5

            def fa_d(qfd=qfd, kfd=kfd, vfd=vfd, sc=sc):
                return flash_attn_func(qfd, kfd, vfd, causal=False, softmax_scale=sc)

            def fa_vd(qvd=qvd, kvd=kvd, vvd=vvd, sc=sc):
                return flash_attn_varlen_func(
                    qvd, kvd, vvd, cu, cu, s, s, causal=False, softmax_scale=sc
                )

            try:
                print(f"  Flash-2 batched D={dd:<3}            {median_us(fa_d):8.1f} µs")
                print(f"  Flash-2 varlen  D={dd:<3}            {median_us(fa_vd):8.1f} µs")
            except Exception as exc:
                print(f"  Flash-2 D={dd} FAILED: {exc}")

    # RoPE as in Qwen3-VL vision: q,k [S, H, D], cos [S, D]
    q_rope = q_var  # [N*S, H, D]
    k_rope = k_var
    pos = torch.arange(n_img * s, device=device, dtype=torch.float32)
    half = d // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.outer(pos, freq)
    emb = torch.cat([ang, ang], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    def rope32():
        return rope_fp32(q_rope, k_rope, cos, sin)

    def rope16():
        return rope_bf16(q_rope, k_rope, cos, sin)

    print("\n=== RoPE on packed Q+K (one layer) ===")
    print(f"  apply_rotary fp32 (Qwen)         {median_us(rope32):8.1f} µs")
    print(f"  apply_rotary bf16                {median_us(rope16):8.1f} µs")

    # GEMMs
    x = torch.randn(n_img * s, hidden, device=device, dtype=dtype)
    w_qkv = torch.randn(hidden * 3, hidden, device=device, dtype=dtype)
    w_o = torch.randn(hidden, hidden, device=device, dtype=dtype)
    w1 = torch.randn(inter, hidden, device=device, dtype=dtype)
    w2 = torch.randn(hidden, inter, device=device, dtype=dtype)

    def qkv():
        return F.linear(x, w_qkv)

    def oproj():
        return F.linear(x, w_o)

    def mlp():
        return F.linear(F.gelu(F.linear(x, w1)), w2)

    print("\n=== GEMMs one layer (bf16 linear, 8640 x 1152) ===")
    print(f"  QKV 1152→3456                    {median_us(qkv):8.1f} µs")
    print(f"  o_proj 1152→1152                 {median_us(oproj):8.1f} µs")
    print(f"  MLP 1152→4304→1152               {median_us(mlp):8.1f} µs")

    us_sdpa = median_us(sdpa_batched)
    us_loop = median_us(sdpa_loop)
    us_rope = median_us(rope32)
    us_mlp = median_us(mlp)
    us_fa = median_us(fa_varlen) if has_fa else float("nan")
    print(f"\n=== {n_layers} layers × one-layer median ===")
    print(f"  SDPA batched                     {n_layers * us_sdpa / 1000:6.2f} ms")
    print(f"  SDPA 12 sequential               {n_layers * us_loop / 1000:6.2f} ms")
    if has_fa:
        print(f"  Flash-2 varlen                   {n_layers * us_fa / 1000:6.2f} ms")
    print(f"  RoPE fp32                        {n_layers * us_rope / 1000:6.2f} ms")
    print(f"  MLP                              {n_layers * us_mlp / 1000:6.2f} ms")
    print("\nLive ROS 'attn core' was ~17 ms; 'RoPE fp32' ~14 ms; 'MLP' ~27 ms.")
    print("If Flash-2 varlen here is << 17 ms, the live path is not a good FA kernel")
    print("(fallback loop, D=72 slow path, or extra copies around the hook).")


if __name__ == "__main__":
    main()
