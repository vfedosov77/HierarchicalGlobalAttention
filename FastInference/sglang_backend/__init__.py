"""SGLang backend for Hierarchical Global Attention.

Public entry points:

* :class:`HgaAttentionBackend`  — the SGLang ``AttentionBackend`` implementation.
* :func:`register_hga_backend`  — register it under a name (default ``"hga"``).
* :func:`configure_hga`         — set HGA config overrides used at instantiation.
"""

from .hga_attention_backend import HgaAttentionBackend, hga_config_from_model_runner
from .register import register_hga_backend, configure_hga

__all__ = [
    "HgaAttentionBackend",
    "hga_config_from_model_runner",
    "register_hga_backend",
    "configure_hga",
]
