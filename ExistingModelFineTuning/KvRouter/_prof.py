"""Gated, zero-overhead profiling primitives for the HGA decode hot path.

Every marker here is a **no-op unless** :data:`PROFILE` is turned on (only the
``profile_hga_bottleneck`` script flips it), so normal decode / training pay nothing.

* ``region(name)`` → a ``torch.profiler.record_function(name)`` when profiling, else a
  ``nullcontext`` — lets the profiler attribute host+device time to named decode phases
  (``hga/route``, ``hga/gather_tokens``, ...).
* ``note_sync()`` counts a device→host synchronisation site (``.item()`` / ``.tolist()``)
  so the profiler can report exact D2H syncs/step without a heavyweight sync-debug hook.

Keeping this in ``KvRouter`` (not the Qwen glue) means the router/store markers import it
by a plain relative import and it travels with the package.
"""
from __future__ import annotations

import contextlib
import os

import torch

PROFILE: bool = False
# Emit NVTX ranges for the HGA phases so an external profiler (Nsight Systems / nsys) can attribute
# GPU+host time to each phase on its timeline.  Enabled out-of-band via ``HGA_NVTX=1`` (independent
# of ``PROFILE``, which drives the in-repo torch-profiler harness); off ⇒ zero overhead.
NVTX: bool = bool(int(os.environ.get("HGA_NVTX", "0") or "0")) and torch.cuda.is_available()
_sync_count: int = 0


def enable(on: bool = True) -> None:
    """Turn instrumentation on/off (set once by the profiler around the measured window)."""
    global PROFILE
    PROFILE = bool(on)


@contextlib.contextmanager
def _nvtx(name: str):
    """Push/pop an NVTX range (version-independent; visible to nsys with ``--trace=nvtx``)."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def region(name: str):
    """Named host+device timing region.

    ``HGA_NVTX=1`` ⇒ NVTX range (for nsys); else ``record_function`` while ``PROFILE``; else no-op.
    """
    if NVTX:
        return _nvtx(name)
    if PROFILE:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def note_sync(n: int = 1) -> None:
    """Record ``n`` device→host sync(s) at this call site (only while profiling)."""
    global _sync_count
    if PROFILE:
        _sync_count += n


def reset_syncs() -> None:
    global _sync_count
    _sync_count = 0


def get_syncs() -> int:
    return _sync_count
