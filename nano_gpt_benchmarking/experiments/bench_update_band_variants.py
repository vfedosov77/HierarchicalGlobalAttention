"""Decompose ``update_band``: where do the ~60 microseconds go, and what is removable?

``update_band`` is launch/latency-bound (it costs ~0.068 ms on an A4000 and ~0.061 ms on
an H200 -- essentially GPU-independent), so this A4000 is a valid proxy for optimizing it.
The only lever for launch-bound code is FEWER kernels. This script times the current
implementation against variants that drop the two non-fusable data-movement kernels (the
``scatter_reduce_`` that builds the key-major band and the ``argsort`` that compacts it),
so we can see the ceiling of any rewrite:

  baseline    : current update_band (scatter_reduce_ + argsort)
  no_argsort  : compaction via cumsum-scatter instead of argsort (same result, sorted)
  minimal     : KV-major writes only + the scatter_reduce_, NO compaction (timing floor)

Correctness of ``no_argsort`` is checked against baseline (identical full_q_num and, per
key row, identical compacted query set). Run on the target machine.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hga_spda_attention_simplified as M  # noqa: E402

T = 307200
H = 6
D = 128
CH = 64
TOPK = 3
WINDOW = 22
C = T // CH
ITERS = 200
WARMUP = 30
DEV = "cuda"
DT = torch.bfloat16


def _sync_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _update_cumsum(state, sel, slot_valid, window):
    """Same result as update_band, but compact the Q-major band with a cumsum-scatter
    instead of argsort (one prefix-sum + one scatter vs a full sort)."""
    C = state.C
    S = sel.shape[1]
    W = state._band_width(window)
    dev = state.device
    state.full_kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
    state.full_kv_num[0, 0, :] = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32)
    ci = state._arangeC
    j = sel[:, : S - 1].to(torch.long)
    col = ci[:, None] - j - 1
    ok = (slot_valid[:, : S - 1] & (col >= 0) & (col < W)).to(torch.int32)
    band = torch.zeros(C * W, dtype=torch.int32, device=dev)
    band.scatter_reduce_(
        0, (j * W + col.clamp(0, W - 1)).reshape(-1), ok.reshape(-1),
        reduce="amax", include_self=True,
    )
    band_i = band.view(C, W)
    band_k = band_i > 0
    state.full_q_num[0, 0, :] = band_i.sum(dim=1, dtype=torch.int32)
    ranks = band_i.cumsum(dim=1) - 1                                  # [C, W]
    cand_q = ci[:, None] + torch.arange(1, W + 1, device=dev)[None, :]  # [C, W] = j+col+1
    dest = torch.where(band_k, ranks, torch.full_like(ranks, W))        # invalid -> dump col
    scratch = torch.zeros(C, W + 1, dtype=torch.int32, device=dev)
    scratch.scatter_(1, dest, cand_q.clamp(max=C - 1).to(torch.int32))
    state.full_q_idx[0, 0, :, :W] = scratch[:, :W]
    return state.mask


def _update_minimal(state, sel, slot_valid, window):
    """Timing FLOOR: KV-major writes + the key-major scatter_reduce_, but NO compaction
    (leaves full_q_* stale). NOT correct -- only to bound how much compaction costs."""
    C = state.C
    S = sel.shape[1]
    W = state._band_width(window)
    dev = state.device
    state.full_kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
    state.full_kv_num[0, 0, :] = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32)
    ci = state._arangeC
    j = sel[:, : S - 1].to(torch.long)
    col = ci[:, None] - j - 1
    ok = (slot_valid[:, : S - 1] & (col >= 0) & (col < W)).to(torch.int32)
    band = torch.zeros(C * W, dtype=torch.int32, device=dev)
    band.scatter_reduce_(
        0, (j * W + col.clamp(0, W - 1)).reshape(-1), ok.reshape(-1),
        reduce="amax", include_self=True,
    )
    return band


def _row_sets(num_t, idx_t, C):
    num = num_t[0, 0].tolist()
    idx = idx_t[0, 0]
    return [set(idx[i, : num[i]].tolist()) for i in range(C)]


def main():
    torch.manual_seed(0)
    dev = torch.device(DEV)
    print(f"device={torch.cuda.get_device_name(dev)}  T={T} C={C} H={H} D={D} "
          f"chunk={CH} topk={TOPK} window={WINDOW}")

    q = torch.randn(T, H, D, device=dev, dtype=DT)
    k = torch.randn(T, H, D, device=dev, dtype=DT)
    q = torch.nn.functional.rms_norm(q.float(), (D,)).to(DT)
    k = torch.nn.functional.rms_norm(k.float(), (D,)).to(DT)
    scale = 1.0 / (D ** 0.5)
    with torch.no_grad():
        sel, slot_valid = M._route_static(q, k, CH, TOPK, scale, WINDOW)

    # Correctness: baseline vs cumsum on separate states.
    st_base = M._RoutedFlexMaskState(C, T, CH, dev)
    st_cum = M._RoutedFlexMaskState(C, T, CH, dev)
    st_base.update_band(sel, slot_valid, WINDOW)
    _update_cumsum(st_cum, sel, slot_valid, WINDOW)
    torch.cuda.synchronize()
    num_ok = torch.equal(st_base.full_q_num, st_cum.full_q_num)
    sets_ok = _row_sets(st_base.full_q_num, st_base.full_q_idx, C) == \
        _row_sets(st_cum.full_q_num, st_cum.full_q_idx, C)
    kv_ok = torch.equal(st_base.full_kv_idx, st_cum.full_kv_idx) and \
        torch.equal(st_base.full_kv_num, st_cum.full_kv_num)
    print(f"cumsum correctness: q_num={num_ok}  q_sets={sets_ok}  kv={kv_ok}")

    base = torch.compile(st_base.update_band, dynamic=False, fullgraph=True)
    cum = torch.compile(_update_cumsum, dynamic=False, fullgraph=True)
    minim = torch.compile(_update_minimal, dynamic=False, fullgraph=True)

    t_base = _sync_time(lambda: base(sel, slot_valid, WINDOW), ITERS, WARMUP)
    t_cum = _sync_time(lambda: cum(st_cum, sel, slot_valid, WINDOW), ITERS, WARMUP)
    t_min = _sync_time(lambda: minim(st_base, sel, slot_valid, WINDOW), ITERS, WARMUP)

    print(f"\nbaseline (scatter+argsort) : {t_base*1000:7.2f} us/iter")
    print(f"cumsum   (scatter+cumsum)  : {t_cum*1000:7.2f} us/iter   "
          f"({100*(t_base-t_cum)/t_base:+.1f}% vs baseline)")
    print(f"minimal  (no compaction)   : {t_min*1000:7.2f} us/iter   "
          f"(floor; argsort costs ~{(t_base-t_min)*1000:.1f} us)")


if __name__ == "__main__":
    main()
