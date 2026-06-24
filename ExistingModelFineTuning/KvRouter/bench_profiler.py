"""Lightweight, opt-in per-stage profiler for the routed-attention hot path.

A single module-level :data:`prof` instance is imported by the hot-path code
(``qwen_routed_attention.py``, ``chunk_router.py``) and by the benchmark harness
(``bench_long_context.py``).  It is **disabled by default** and adds ~zero overhead when off
(``begin``/``end`` early-return and ``section`` returns a shared no-op context manager), so it
never affects normal chat/generation.  The harness enables it for a short, bounded profiling
pass to attribute wall-clock to the stages: qkv, RoPE, chunk top-k, group top-k, KV gather,
attention, MLP.

Timing uses CUDA events (recorded on the stream, no per-call sync → low distortion); the
elapsed times are read once at :meth:`flush`/:meth:`totals` after a single device sync.  On
CPU it falls back to ``perf_counter``.  Because each section allocates a pair of CUDA events,
only profile a *bounded* number of steps (the harness does ~16-32), not a full long decode.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch


class _NullCtx:
    """Shared no-op context manager used when profiling is disabled (no allocation)."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _NullCtx()


class _Section:
    """Context manager that times its body via :meth:`_Profiler.begin`/``end``."""

    __slots__ = ("_prof", "_name")

    def __init__(self, prof: "_Profiler", name: str) -> None:
        self._prof = prof
        self._name = name

    def __enter__(self):
        self._prof.begin(self._name)
        return self

    def __exit__(self, *exc):
        self._prof.end(self._name)
        return False


class _Profiler:
    """Accumulates per-named-stage GPU/CPU time and call counts across a profiling pass."""

    def __init__(self) -> None:
        self.enabled = False
        self.use_cuda = torch.cuda.is_available()
        # name -> pending start (CUDA event or perf_counter float)
        self._pending: Dict[str, object] = {}
        # (name, start_event, stop_event) collected until flush() resolves them
        self._events: List[Tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []
        self._ms: "OrderedDict[str, float]" = OrderedDict()   # accumulated milliseconds
        self._counts: Dict[str, int] = {}

    # -- control -----------------------------------------------------------
    def reset(self) -> None:
        self._pending.clear()
        self._events.clear()
        self._ms.clear()
        self._counts.clear()

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    # -- instrumentation API (called from the hot path) --------------------
    def section(self, name: str):
        if not self.enabled:
            return _NULL
        return _Section(self, name)

    def begin(self, name: str) -> None:
        if not self.enabled:
            return
        self._counts[name] = self._counts.get(name, 0) + 1
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending[name] = ev
        else:
            self._pending[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        start = self._pending.pop(name, None)
        if start is None:
            return
        if self.use_cuda:
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            self._events.append((name, start, stop))  # type: ignore[arg-type]
        else:
            self._ms[name] = self._ms.get(name, 0.0) + (time.perf_counter() - start) * 1e3

    # -- reporting ---------------------------------------------------------
    def flush(self) -> None:
        """Resolve outstanding CUDA events into accumulated milliseconds (one device sync)."""
        if self.use_cuda and self._events:
            torch.cuda.synchronize()
            for name, start, stop in self._events:
                self._ms[name] = self._ms.get(name, 0.0) + start.elapsed_time(stop)
            self._events.clear()

    def totals_ms(self) -> "OrderedDict[str, float]":
        self.flush()
        return OrderedDict(self._ms)

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)


# The single shared instance every module imports.
prof = _Profiler()
