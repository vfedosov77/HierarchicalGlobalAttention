"""Shared disk-manager and I/O helpers for the NVMe/filesystem KV-cache tier.

Design goals:
* **Never behave like a swap file** — explicit ``pread``/``pwrite`` + ``posix_fadvise(DONTNEED)``
  keep the kernel page cache small and the OS responsive.
* **Non-blocking eviction** — spills run on a background writer thread; the compute path only
  pays a cheap RAM→RAM clone for the handoff.
* **No leftover files** — the spill directory is cleaned up on normal exit (``atexit``), on fatal
  signals (SIGINT/SIGTERM/SIGHUP chained to previous handlers), and on ``reset``/``close``.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import threading
import weakref
from queue import Queue
from typing import Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _raw_bytes(t: torch.Tensor) -> memoryview:
    """Read-only ``memoryview`` of a *contiguous* CPU tensor's raw bytes (dtype-agnostic).

    ``numpy()`` rejects bf16, so we reinterpret as ``uint8`` first; the view shares storage, so
    no copy is made.
    """
    return memoryview(t.contiguous().view(torch.uint8).numpy())


def _pwrite_all(fd: int, mv: memoryview, offset: int) -> None:
    n, total = 0, len(mv)
    while n < total:
        n += os.pwrite(fd, mv[n:], offset + n)


def _pread_all(fd: int, mv: memoryview, offset: int) -> None:
    n, total = 0, len(mv)
    while n < total:
        r = os.preadv(fd, [mv[n:]], offset + n)
        if r == 0:
            raise EOFError(f"short read at offset {offset + n} (wanted {total - n} more bytes)")
        n += r


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    """Tell the kernel we will not reuse this file region soon → drop it from the page cache.

    This is the key to *not* behaving like swap: without it, every byte we write/read would sit in
    the kernel page cache as reclaimable/dirty memory and, at long contexts, balloon into the very
    system-wide memory pressure that makes a machine unresponsive.  Best-effort: silently ignored on
    platforms / filesystems that do not support it.
    """
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _is_ram_backed(path: str) -> bool:
    """True if ``path`` lives on a RAM-backed filesystem (tmpfs/ramfs) — e.g. ``/tmp`` on many distros.

    Spilling there would put the "disk" tier back in RAM, defeating the whole point of the fs tier
    and re-introducing the host-RAM pressure it exists to avoid.  Best-effort via ``/proc/mounts``
    (returns ``False`` if it can't be determined).
    """
    try:
        real = os.path.realpath(path)
        best_mp, best_fstype = "", ""
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best_mp):
                    best_mp, best_fstype = mp, fstype
        return best_fstype in ("tmpfs", "ramfs")
    except OSError:
        return False


def _fallocate(fd: int, length: int) -> None:
    """Best-effort reserve ``length`` bytes for ``fd`` up front (preallocation).

    When the target max context is known, reserving the whole file once lets every later spill land
    in already-allocated (ideally contiguous) blocks instead of repeatedly extending the file — fewer
    metadata updates, less fragmentation.  Falls back to ``ftruncate`` (sparse) where
    ``posix_fallocate`` is unavailable, and is silently ignored if neither is supported.
    """
    if length <= 0:
        return
    try:
        os.posix_fallocate(fd, 0, length)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        try:
            os.ftruncate(fd, length)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Shared disk manager
# ---------------------------------------------------------------------------

class _FsDiskManager:
    """Shared owner of the on-disk spill area: a private temp dir, fds, and the writer thread.

    One instance is shared by all layers of a store.  It guarantees the spill files are removed on
    interpreter exit and on fatal signals, so the application never leaves temporary files behind.
    """

    _registry: "weakref.WeakSet[_FsDiskManager]" = weakref.WeakSet()
    _handlers_installed = False
    _orig_handlers: Dict[int, object] = {}
    _reg_lock = threading.Lock()

    def __init__(self, *, root: Optional[str] = None, max_pending: int = 64,
                 num_workers: int = 24) -> None:
        # NOTE: the spill dir must be on *real disk* (NVMe/SSD), never on a tmpfs like ``/tmp`` on
        # many distros — spilling onto tmpfs would put the "disk" tier back in RAM and reintroduce
        # exactly the swap-style memory pressure we are avoiding.  Default: a hidden dir in the cwd.
        if root is None:
            root = os.environ.get("KVR_FS_CACHE_DIR") or os.path.join(os.getcwd(), ".kvr_fscache")
        os.makedirs(root, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="kvr_fscache_", dir=root)
        if _is_ram_backed(self.dir):
            import sys as _sys
            print(f"[kvr] WARNING: fs spill dir {self.dir} is on a RAM-backed filesystem "
                  f"(tmpfs/ramfs). The disk tier will consume host RAM, not disk — this defeats the "
                  f"fs cache and can OOM-kill the process at long context. Point it at a real disk "
                  f"(set fs_cache_dir / $KVR_FS_CACHE_DIR to e.g. ~/.cache or an NVMe scratch mount).",
                  file=_sys.stderr, flush=True)
        self._fds: List[int] = []
        self._fd_lock = threading.Lock()
        # Pool of writer threads: distinct chunks/offsets are independent and ``os.pwrite`` is
        # position-based (no shared file offset), so concurrent spills to the same fd are safe and
        # let the NVMe queue depth stay full.  Overridable via ``KVR_FS_WRITERS``.
        env_nw = os.environ.get("KVR_FS_WRITERS")
        self._num_workers = max(1, int(env_nw)) if env_nw else max(1, int(num_workers))
        self._queue: "Queue[Optional[Callable[[], None]]]" = Queue(
            maxsize=max(max_pending, self._num_workers * 2)
        )
        self._closed = False
        self._threads = [
            threading.Thread(target=self._writer_loop, name=f"kvr-fs-writer-{i}", daemon=True)
            for i in range(self._num_workers)
        ]
        for t in self._threads:
            t.start()
        self._raise_fd_limit()
        with _FsDiskManager._reg_lock:
            _FsDiskManager._registry.add(self)
            self._install_handlers()

    # -- file handles ------------------------------------------------------
    def open_file(self, name: str) -> int:
        path = os.path.join(self.dir, name)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        with self._fd_lock:
            self._fds.append(fd)
        return fd

    @staticmethod
    def _raise_fd_limit() -> None:
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            want = min(hard, max(soft, 8192))
            if want > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except Exception:
            pass

    # -- async writes ------------------------------------------------------
    def submit(self, fn: Callable[[], None]) -> None:
        """Queue a spill closure for the writer thread (blocks only if the queue is full = backpressure)."""
        if self._closed:
            fn()  # drain synchronously during shutdown
            return
        self._queue.put(fn)

    def flush(self) -> None:
        self._queue.join()

    def _writer_loop(self) -> None:
        while True:
            fn = self._queue.get()
            try:
                if fn is None:
                    return
                fn()
            except Exception:  # pragma: no cover - a spill failure must not kill the writer
                import traceback
                import sys

                traceback.print_exc(file=sys.stderr)
            finally:
                self._queue.task_done()

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        with self._fd_lock:
            if self._closed:
                return
            self._closed = True
            fds = list(self._fds)
            self._fds.clear()
        try:
            for _ in self._threads:
                self._queue.put(None)  # one sentinel per worker
            for t in self._threads:
                t.join(timeout=5)
        except Exception:
            pass
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)
        _FsDiskManager._registry.discard(self)

    # -- exit / signal cleanup --------------------------------------------
    @classmethod
    def _install_handlers(cls) -> None:
        if cls._handlers_installed:
            return
        cls._handlers_installed = True
        atexit.register(cls._cleanup_all)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                cls._orig_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, cls._signal_handler)
            except (ValueError, OSError):
                # Not on the main thread, or signal unsupported on this platform → skip; atexit
                # still covers the normal-exit case.
                pass

    @classmethod
    def _cleanup_all(cls) -> None:
        for mgr in list(cls._registry):
            mgr.close()

    @classmethod
    def _signal_handler(cls, signum, frame):  # noqa: ANN001
        cls._cleanup_all()
        prev = cls._orig_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            # Restore the default action and re-raise so the process terminates as expected.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # prev == SIG_IGN → swallow.
