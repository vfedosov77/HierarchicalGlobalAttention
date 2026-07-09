"""Gated, zero-overhead NVTX markers for the HGA decode hot path.

Every marker is a **no-op unless** ``HGA_NVTX=1`` (so normal decode / training pay nothing).
``region(name)`` emits an NVTX range so an external profiler (NVIDIA Nsight Systems / nsys)
can attribute GPU+host time to named decode phases (``hga/route``, ``hga/gather_tokens``, ...).

Keeping this in ``KvRouter`` (not the Qwen glue) means the router/store markers import it
by a plain relative import and it travels with the package.
"""
from __future__ import annotations

import contextlib
import os

import torch

# Emit NVTX ranges for the HGA phases so nsys (``--trace=nvtx``) can attribute GPU+host time to
# each phase on its timeline.  Enabled out-of-band via ``HGA_NVTX=1``; off ⇒ zero overhead.
NVTX: bool = bool(int(os.environ.get("HGA_NVTX", "0") or "0")) and torch.cuda.is_available()


@contextlib.contextmanager
def _nvtx(name: str):
    """Push/pop an NVTX range (version-independent; visible to nsys with ``--trace=nvtx``)."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def region(name: str):
    """Named timing region: an NVTX range when ``HGA_NVTX=1``, else a no-op."""
    if NVTX:
        return _nvtx(name)
    return contextlib.nullcontext()
