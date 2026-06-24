"""Register the HGA backend with SGLang's attention registry.

SGLang resolves attention backends by name (``--attention-backend hga``) through
``ATTENTION_BACKENDS`` in ``sglang.srt.layers.attention.attention_registry``.
Call :func:`register_hga_backend` once at import time (e.g. from a launcher) to
make the name available.  ``cfg_overrides`` are forwarded to
:func:`~FastInference.sglang_backend.hga_attention_backend.hga_config_from_model_runner`.
"""

from __future__ import annotations

from typing import Any, Dict

from .hga_attention_backend import HgaAttentionBackend

_OVERRIDES: Dict[str, Any] = {}


def configure_hga(**cfg_overrides) -> None:
    """Set HGA config overrides applied when the backend is instantiated."""
    _OVERRIDES.clear()
    _OVERRIDES.update(cfg_overrides)


def register_hga_backend(name: str = "hga") -> None:
    try:
        from sglang.srt.layers.attention.attention_registry import (
            register_attention_backend,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sglang is not installed; cannot register the HGA attention backend."
        ) from exc

    @register_attention_backend(name)
    def _create_hga_backend(runner):  # noqa: ANN001
        return HgaAttentionBackend(runner, **_OVERRIDES)

    return None
