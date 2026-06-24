"""FastInference — production inference backends for Hierarchical Global Attention (HGA).

This package hosts the *engine-neutral* HGA core plus the per-engine attention
backends.  The first supported engine is **SGLang**; vLLM is planned next.

Layout
------
``hga_core``       Engine-neutral HGA: config, route metadata, summary builder,
                   fused decode/route kernels, and the tiered cache manager.
``sglang_backend`` SGLang ``AttentionBackend`` implementation that drives
                   ``hga_core`` from inside the SGLang model runner.

Design rules (v0)
-----------------
* No dependency on HuggingFace ``DynamicCache`` or Qwen monkey-patching — the
  HF replacement in ``ExistingModelFineTuning`` stays as the correctness/training
  reference only.
* Dual Chunk Attention (DCA) is **disabled** in v0 (native HGA, absolute RoPE).
* RAM + VRAM fast path first; filesystem / NVMe is an L3 spill tier added later.
"""

from .hga_core.config import HgaConfig

__all__ = ["HgaConfig"]

__version__ = "0.0.1"
