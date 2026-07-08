"""How big a slice of the routed-attention step is the BlockMask ``update_band``?

The user asked whether an *incremental* mask update (undo the previous topk, then write
only the new topk) could beat the current rebuild-from-band ``update_band``. That only
matters if ``update_band`` is a non-trivial share of the step. This script times, on the
real problem shape, three things in isolation (all compiled the same way the module is):

  1. ``state.update_band(sel, slot_valid, window)``  -- the mask rebuild alone
  2. ``_flex_attend(...)``                            -- attend fwd (includes update_band)
  3. ``_flex_attend`` fwd+bwd                         -- the full training-step cost

Routing (``_route_static``) is done ONCE up front and reused, so we isolate the mask cost.
Run on the target machine:  python bench_update_band_share.py
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
ITERS = 50
WARMUP = 10
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
    return start.elapsed_time(end) / iters  # ms/iter


def main():
    torch.manual_seed(0)
    dev = torch.device(DEV)
    print(f"device={torch.cuda.get_device_name(dev)}  T={T} C={C} H={H} D={D} "
          f"chunk={CH} topk={TOPK} window={WINDOW}")

    if hasattr(M, "set_flex_bwd_block"):
        M.set_flex_bwd_block(CH)

    q = torch.randn(T, H, D, device=dev, dtype=DT)
    k = torch.randn(T, H, D, device=dev, dtype=DT)
    v = torch.randn(T, H, D, device=dev, dtype=DT)
    q = torch.nn.functional.rms_norm(q.float(), (D,)).to(DT)
    k = torch.nn.functional.rms_norm(k.float(), (D,)).to(DT)
    scale = 1.0 / (D ** 0.5)

    state = M._RoutedFlexMaskState(C, T, CH, dev)

    # Route ONCE (eager) -- this is not what we are measuring.
    with torch.no_grad():
        sel, slot_valid = M._route_static(q, k, CH, TOPK, scale, WINDOW)

    # 1) update_band alone, compiled.
    upd = torch.compile(state.update_band, dynamic=False, fullgraph=True)

    def run_update():
        upd(sel, slot_valid, WINDOW)

    t_upd = _sync_time(run_update, ITERS, WARMUP)

    # 2) full attend fwd, compiled (rebuilds the band internally each call).
    attend = torch.compile(M._flex_attend, dynamic=False, fullgraph=True)

    def run_fwd():
        attend(q, k, v, sel, slot_valid, CH, scale, WINDOW, state)

    t_fwd = _sync_time(run_fwd, ITERS, WARMUP)

    # 3) fwd+bwd.
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)

    def run_fwdbwd():
        y = attend(qg, kg, vg, sel, slot_valid, CH, scale, WINDOW, state)
        y.sum().backward()
        qg.grad = kg.grad = vg.grad = None

    t_fb = _sync_time(run_fwdbwd, ITERS, WARMUP)

    print(f"\nupdate_band alone : {t_upd:8.3f} ms/iter")
    print(f"attend fwd        : {t_fwd:8.3f} ms/iter   "
          f"(update_band is {100*t_upd/t_fwd:5.2f}% of fwd)")
    print(f"attend fwd+bwd    : {t_fb:8.3f} ms/iter   "
          f"(update_band is {100*t_upd/t_fb:5.2f}% of fwd+bwd)")


if __name__ == "__main__":
    main()
