# Other routing variants (not the ROS default)

Moved out of `SpecializedKernels/Len3000HeadDim128/` when that folder
was switched to **true two-level** only (128-token chunks → 16-token
groups, 192 tokens).

| File | What it was |
|---|---|
| `kernel.py` / `attention.py` | One-level 16/32/64/128 and nested 32/8, 64/8, 128/16 |
| `alpamayo_adapter.py` | Adapter with `HGA_ROUTE_BLOCK` / `HGA_HIERARCHY` / FP8 residuals |
| `flash_attn.py` | Flash-shaped BHSD wrapper (extra transpose) |
| `bench.py` | One-level microbench |

The live path is the parent folder: `vlm_prefill_attention`,
`diffusion_cross_attention`, `adapt_alpamayo`.
