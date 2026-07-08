"""Micro-benchmark: per-token decode latency on the RAM tier, overlap ON vs OFF.

Isolates the routing + host→device (H2D) + attend cost of a single routed attention layer over a
long context held in pinned host-RAM, with the cold KV record on CPU and compute on the GPU — the
exact setting the overlap work targets.  It streams a long prefix through ``decode_block`` to fill
the store, then times many single-token decode steps.

Run:
    python -m ExistingModelFineTuning.KvRouter.Tests.bench_overlap
    python -m ExistingModelFineTuning.KvRouter.Tests.bench_overlap --ctx 16384 --steps 300
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import torch

from ..cache_store import ChunkPlacementPolicy, RamKVCacheStore
from ..chunk_router import ChunkRouter, RouterConfig
from .test_router import _make


class _GpuSampler:
    """Poll NVML GPU utilization (%) and device memory from a side thread during the timed loop.

    ``torch.cuda.utilization`` returns the nvidia-smi metric: percent of the last sample period
    during which at least one kernel was executing.  A higher mean under overlap ON means less
    GPU idle time waiting on CPU routing / H2D.  ``torch.cuda.mem_get_info`` returns (free, total)
    for the whole device from the same NVML source nvidia-smi reads, so ``total - free`` is the
    device-wide "Memory-Usage" column.

    ponytail: NVML samples device-wide and coarsely (~tens of ms); the absolute value is noisy,
    the OFF-vs-ON delta on an otherwise-idle GPU is the signal.  A finer-grained upgrade would
    be a torch.profiler pass summing per-kernel device time.
    """

    def __init__(self, device, period_s: float = 0.004):
        self._dev = device
        self._period = period_s
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._mem_used: list[int] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(torch.cuda.utilization(self._dev))
                free, total = torch.cuda.mem_get_info(self._dev)
                self._mem_used.append(total - free)
            except Exception:
                pass
            time.sleep(self._period)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else float("nan")

    @property
    def peak_mem_gb(self) -> float:
        return max(self._mem_used) / 2**30 if self._mem_used else float("nan")


def _build(cfg: RouterConfig, *, B: int, dtype, device, overlap: bool, num_layers: int,
           scap: int, tcap: int, keep_first: int, keep_last: int) -> ChunkRouter:
    # The store reads HGA_OVERLAP at construction; set it just before building for a clean A/B.
    os.environ["HGA_OVERLAP"] = "1" if overlap else "0"
    store = RamKVCacheStore(
        compute_device=torch.device(device),
        policy=ChunkPlacementPolicy(keep_last=keep_last, keep_first=keep_first, first_token_level=True),
        kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
        groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=dtype,
        pin_memory=True, storage_device=torch.device("cpu"),
        vram_cache_chunks=tcap, vram_summary_chunks=scap, num_layers=num_layers,
        vram_cache_reserve_gb=1.0,
    )
    assert (store._copy_stream is not None) == overlap, "overlap flag did not take effect"
    return ChunkRouter(cfg, store)


def _run(cfg, q, k_rope, k_raw, v, *, ctx: int, steps: int, layers: int, device, dtype, overlap: bool,
         scap: int, tcap: int, keep_first: int, keep_last: int):
    B = q.shape[0]
    dev = torch.device(device)
    router = _build(cfg, B=B, dtype=dtype, device=device, overlap=overlap, num_layers=layers,
                    scap=scap, tcap=tcap, keep_first=keep_first, keep_last=keep_last)

    # Inputs live in host RAM; only per-block/per-step slices ever touch the GPU so the harness
    # itself never holds a ctx-length tensor in VRAM (that would OOM the card before the store does).
    def _g(t, a, b):
        return t[:, :, a:b].to(dev, non_blocking=True)

    # Fill every layer's store with the long prefix, streaming one chunk at a time CPU->GPU.
    C = cfg.chunk_size
    for ly in range(layers):
        p = 0
        while p < ctx:
            take = min(C - (p % C), ctx - p)
            routed = router.decode_block(ly, _g(q, p, p + take), _g(k_rope, p, p + take),
                                         _g(k_raw, p, p + take), _g(v, p, p + take), p)
            routed.attend(_g(q, p, p + take))
            p += take

    def _decode(ly: int, p: int) -> torch.Tensor:
        qs = _g(q, p, p + 1)
        routed = router.decode_block(ly, qs, _g(k_rope, p, p + 1), _g(k_raw, p, p + 1), _g(v, p, p + 1), p)
        return routed.attend(qs)

    # One token step = a forward through all `layers` routed layers; prefetch of layer L overlaps
    # the compute of layers L+1..N (the real cross-layer overlap window in a stacked model).
    warmup = min(20, steps)
    for i in range(warmup):
        for ly in range(layers):
            _decode(ly, ctx + i)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(torch.device(device))
    with _GpuSampler(torch.device(device)) as sampler:
        t0 = time.perf_counter()
        for i in range(warmup, warmup + steps):
            for ly in range(layers):
                _decode(ly, ctx + i)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    util = sampler.mean
    smi_gb = sampler.peak_mem_gb
    torch_gb = torch.cuda.max_memory_reserved(torch.device(device)) / 2**30

    st = router.store
    ms = 1e3 * dt / steps
    return ms, util, smi_gb, torch_gb, (st.summary_hits, st.summary_misses, st.cache_hits, st.cache_misses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192, help="context length (tokens) to prefill")
    ap.add_argument("--steps", type=int, default=200, help="timed single-token decode steps")
    ap.add_argument("--layers", type=int, default=8, help="stacked routed layers (Ornith has 8 full-attn)")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--topk-chunks", type=int, default=20)
    ap.add_argument("--topk-groups", type=int, default=32)
    ap.add_argument("--summary-cap", type=int, default=8192)
    ap.add_argument("--token-cap", type=int, default=64,
                    help="VRAM token-bank chunks/layer; small => opened chunks miss => H2D to hide")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_overlap needs a CUDA device (RAM->GPU H2D is the point).")
    device, dtype = "cuda", torch.float16

    cfg = RouterConfig(
        nhead=args.heads, kv_heads=args.kv_heads, head_dim=args.head_dim,
        chunk_size=args.chunk, group_size=args.group,
        topk_chunks=args.topk_chunks, topk_groups=args.topk_groups, theta=1_000_000.0,
    )
    B = 1
    total = args.ctx + args.steps + 32
    # Build synthetic inputs on the HOST; _run streams only slices to the GPU (see _run).
    q, k_rope, k_raw, v = _make(cfg, B, total, "cpu", dtype)

    print(f"ctx={args.ctx} steps={args.steps} layers={args.layers} H={args.heads} KVH={args.kv_heads} "
          f"Dh={args.head_dim} chunk={args.chunk} group={args.group} token_cap={args.token_cap} dtype={dtype}")
    off_ms, off_util, off_smi, off_torch, off_stats = _run(
        cfg, q, k_rope, k_raw, v, ctx=args.ctx, steps=args.steps, layers=args.layers,
        device=device, dtype=dtype, overlap=False, scap=args.summary_cap,
        tcap=args.token_cap, keep_first=args.keep_first, keep_last=args.keep_last)
    on_ms, on_util, on_smi, on_torch, on_stats = _run(
        cfg, q, k_rope, k_raw, v, ctx=args.ctx, steps=args.steps, layers=args.layers,
        device=device, dtype=dtype, overlap=True, scap=args.summary_cap,
        tcap=args.token_cap, keep_first=args.keep_first, keep_last=args.keep_last)

    def _fmt(s):
        sh, sm, th, tm = s
        s_hr = 100.0 * sh / max(1, sh + sm)
        t_hr = 100.0 * th / max(1, th + tm)
        return f"summary {s_hr:.1f}% hit, token {t_hr:.1f}% hit"

    speedup = 100.0 * (off_ms - on_ms) / off_ms
    print(f"[overlap OFF] {off_ms:.3f} ms/token   GPU util {off_util:.1f}%   "
          f"mem {off_smi:.2f} GB smi / {off_torch:.2f} GB torch   ({_fmt(off_stats)})")
    print(f"[overlap ON ] {on_ms:.3f} ms/token   GPU util {on_util:.1f}%   "
          f"mem {on_smi:.2f} GB smi / {on_torch:.2f} GB torch   ({_fmt(on_stats)})")
    print(f"[delta] {off_ms - on_ms:+.3f} ms/token  ({speedup:+.1f}% latency)   "
          f"GPU util {on_util - off_util:+.1f} pts   mem {on_torch - off_torch:+.2f} GB torch")


if __name__ == "__main__":
    main()
