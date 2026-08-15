"""Routed attention for Alpamayo VLM prefill and diffusion cross-attn.

Direct BHSD API (no Flash transpose)::

    from SpecializedKernels.Len3000HeadDim128 import (
        vlm_prefill_attention,
        diffusion_cross_attention,
        attach_to_alpamayo,
    )

See ``README.md`` in this folder for how to wire it into Alpamayo.
"""

from .attention import (
    DIFFUSION_ATTN_NAME,
    VLM_ATTN_NAME,
    attach_to_alpamayo,
    diffusion_cross_attention,
    hf_diffusion_attention,
    hf_vlm_attention,
    prompt_chunk_keys,
    set_diffusion_prompt,
    vlm_prefill_attention,
)
from .alpamayo_adapter import adapt_alpamayo
from .flash_attn import flash_attn_func

__all__ = [
    "vlm_prefill_attention",
    "diffusion_cross_attention",
    "prompt_chunk_keys",
    "attach_to_alpamayo",
    "set_diffusion_prompt",
    "adapt_alpamayo",
    "hf_vlm_attention",
    "hf_diffusion_attention",
    "flash_attn_func",
    "VLM_ATTN_NAME",
    "DIFFUSION_ATTN_NAME",
]
