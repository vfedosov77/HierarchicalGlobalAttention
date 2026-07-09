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

import torch

PROFILE: bool = False
_sync_count: int = 0


def enable(on: bool = True) -> None:
    """Turn instrumentation on/off (set once by the profiler around the measured window)."""
    global PROFILE
    PROFILE = bool(on)


def region(name: str):
    """Named host+device timing region (``record_function`` while profiling, else no-op)."""
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
