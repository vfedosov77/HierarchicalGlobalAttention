"""Benchmark ChatGPT's simplified-path suggestions in isolation, with correctness checks.

Tested on the A4000 (a fine proxy for the launch-bound index work; the attend itself is
compute-bound so its absolute ms differ from H200, but correctness + relative wins carry):

  1. SHRINK metadata: BlockMask index tensors [1,1,C,C] -> [1,1,C,width]
       diag_idx [C,1], full_kv_idx [C,topk], full_q_idx [C,W].
       Checks flex fwd/bwd output equals the full-width mask; times both; reports memory.
  2. ROUTING flatten: score with flattened heads ([C,W+1]) instead of einsum->[H,C,W+1]->mean.
       Checks sel/slot_valid identical; times both.
  3. ROWS_GUARANTEED_SAFE flex kernel option (own-chunk diagonal guarantees >=1 valid key).
       Checks output equals; times fwd/bwd with and without.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hga_spda_attention_simplified as M  # noqa: E402
from torch.nn.attention.flex_attention import BlockMask, flex_attention  # noqa: E402

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
SCALE = 1.0 / (D ** 0.5)


def _sync_time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# ---------------------------------------------------------------------------- routing
def route_flat(q, k, chunk_size, topk, softmax_scale, window):
    """Suggestion 2: flatten heads before scoring -> [C, W+1] directly (no [H,C,W+1]).
    mean_h(dot(q_h,k_h)) == dot(flat(q),flat(k))/H, and /H is a positive monotone scale so
    it cannot change the top-k selection -> dropped."""
    T_, H_, D_ = q.shape
    CH = int(chunk_size)
    C_ = T_ // CH
    dev = q.device
    q_sum = q.view(C_, CH, H_, D_).float().mean(dim=1).reshape(C_, H_ * D_)  # [C, HD]
    k_sum = k.view(C_, CH, H_, D_).float().mean(dim=1).reshape(C_, H_ * D_)
    cidx = torch.arange(C_, device=dev)
    W = max(0, min(int(window), C_ - 1))
    kpad = torch.nn.functional.pad(k_sum, (0, 0, W, 0))          # [W+C, HD]
    kb = kpad.unfold(0, W + 1, 1)                               # [C, HD, W+1]
    w = torch.einsum("cf,cfj->cj", q_sum * float(softmax_scale), kb)  # [C, W+1] head-summed
    past_key = cidx[:, None] - W + torch.arange(W, device=dev)[None, :]
    w = w[:, :W].masked_fill(past_key < 0, float("-inf"))
    topv, topj = w.topk(min(int(topk), W), dim=-1)
    valid_past = torch.isfinite(topv)
    topi = (cidx[:, None] - W + topj).clamp(min=0)
    sel = torch.cat([topi, cidx[:, None]], dim=-1)
    slot_valid = torch.cat([valid_past, torch.ones(C_, 1, dtype=torch.bool, device=dev)], dim=-1)
    return sel, slot_valid


# ---------------------------------------------------------------------------- shrunk mask
class ShrunkMaskState:
    """Suggestion 1: BlockMask metadata sized to the true max row width, not C."""

    def __init__(self, C, T, chunk_size, topk, window, device):
        self.C = C
        self.T = T
        self.CH = chunk_size
        self.device = device
        self._arangeC = torch.arange(C, device=device)
        self.KW = int(topk)                       # full_kv max blocks/row
        self.QW = max(1, min(int(window), C - 1))  # full_q max blocks/row
        ci = torch.arange(C, device=device, dtype=torch.int32)
        self.diag_idx = torch.zeros(1, 1, C, 1, dtype=torch.int32, device=device)
        self.diag_idx[0, 0, :, 0] = ci
        self.diag_num = torch.ones(1, 1, C, dtype=torch.int32, device=device)
        self.full_kv_idx = torch.zeros(1, 1, C, self.KW, dtype=torch.int32, device=device)
        self.full_q_idx = torch.zeros(1, 1, C, self.QW, dtype=torch.int32, device=device)
        self.full_kv_num = torch.zeros(1, 1, C, dtype=torch.int32, device=device)
        self.full_q_num = torch.zeros(1, 1, C, dtype=torch.int32, device=device)
        self.mask = BlockMask(
            seq_lengths=(T, T),
            kv_num_blocks=self.diag_num, kv_indices=self.diag_idx,
            full_kv_num_blocks=self.full_kv_num, full_kv_indices=self.full_kv_idx,
            q_num_blocks=self.diag_num, q_indices=self.diag_idx,
            full_q_num_blocks=self.full_q_num, full_q_indices=self.full_q_idx,
            BLOCK_SIZE=(CH, CH), mask_mod=M._causal_block_mask_mod,
        )

    def update_band(self, sel, slot_valid, window):
        C = self.C
        S = sel.shape[1]
        W = self.QW
        self.full_kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
        self.full_kv_num[0, 0, :] = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32)
        ci = self._arangeC
        j = sel[:, : S - 1].to(torch.long)
        col = ci[:, None] - j - 1
        ok = (slot_valid[:, : S - 1] & (col >= 0) & (col < W)).to(torch.int32)
        band = torch.zeros(C * W, dtype=torch.int32, device=self.device)
        band.scatter_reduce_(0, (j * W + col.clamp(0, W - 1)).reshape(-1), ok.reshape(-1),
                             reduce="amax", include_self=True)
        band_i = band.view(C, W)
        band_k = band_i > 0
        self.full_q_num[0, 0, :] = band_i.sum(dim=-1, dtype=torch.int32)
        ranks = band_i.cumsum(dim=1) - 1
        cand_q = ci[:, None] + torch.arange(1, W + 1, device=self.device)[None, :]
        dest = torch.where(band_k, ranks, torch.full_like(ranks, W))
        scratch = torch.zeros(C, W + 1, dtype=torch.int32, device=self.device)
        scratch.scatter_(1, dest, cand_q.clamp(max=C - 1).to(torch.int32))
        self.full_q_idx[0, 0, :, :W] = scratch[:, :W]
        return self.mask


def attend_with(mask, q, k, v, extra_opts=None):
    opts = {"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH}
    if extra_opts:
        opts.update(extra_opts)
    qf = q.transpose(0, 1)[None]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    y = flex_attention(qf, kf, vf, block_mask=mask, scale=SCALE, kernel_options=opts)
    return y[0].transpose(0, 1)


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

    # ---- Suggestion 2: routing flatten -----------------------------------------------
    with torch.no_grad():
        sel0, sv0 = M._route_static(q, k, CH, TOPK, SCALE, WINDOW)
        sel1, sv1 = route_flat(q, k, CH, TOPK, SCALE, WINDOW)
    route_ok = torch.equal(sel0, sel1) and torch.equal(sv0, sv1)
    r_base = torch.compile(M._route_static, dynamic=False, fullgraph=True)
    r_flat = torch.compile(route_flat, dynamic=False, fullgraph=True)
    t_r0 = _sync_time(lambda: r_base(q, k, CH, TOPK, SCALE, WINDOW))
    t_r1 = _sync_time(lambda: r_flat(q, k, CH, TOPK, SCALE, WINDOW))
    print("\n[2] routing flatten")
    print(f"    sel/slot_valid identical: {route_ok}")
    print(f"    einsum[H,C,W]->mean : {t_r0*1000:7.1f} us")
    print(f"    flattened  [C,W]    : {t_r1*1000:7.1f} us   "
          f"({100*(t_r0-t_r1)/t_r0:+.1f}%)")

    # ---- Suggestion 1: shrink metadata -----------------------------------------------
    st_full = M._RoutedFlexMaskState(C, T, CH, dev)
    st_shr = ShrunkMaskState(C, T, CH, TOPK, WINDOW, dev)
    st_full.update_band(sel0, sv0, WINDOW)
    st_shr.update_band(sel0, sv0, WINDOW)
    torch.cuda.synchronize()

    attend = torch.compile(attend_with, dynamic=False, fullgraph=True)
    yf = attend(st_full.mask, q, k, v)
    ys = attend(st_shr.mask, q, k, v)
    torch.cuda.synchronize()
    fwd_ok = torch.allclose(yf, ys, atol=1e-3, rtol=1e-3)

    def fb(mask):
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        attend(mask, qg, kg, vg).sum().backward()
        return qg.grad, kg.grad, vg.grad

    gf = fb(st_full.mask)
    gs = fb(st_shr.mask)
    bwd_ok = all(torch.allclose(a, b, atol=1e-2, rtol=1e-2) for a, b in zip(gf, gs))

    meta_full = sum(t.numel() * 4 for t in
                    (st_full.diag_idx, st_full.full_kv_idx, st_full.full_q_idx)) / 1e6
    meta_shr = sum(t.numel() * 4 for t in
                   (st_shr.diag_idx, st_shr.full_kv_idx, st_shr.full_q_idx)) / 1e6

    t_ff = _sync_time(lambda: attend(st_full.mask, q, k, v))
    t_fs = _sync_time(lambda: attend(st_shr.mask, q, k, v))

    def run_fb_full():
        attend(st_full.mask, qg, kg, vg).sum().backward()
        qg.grad = kg.grad = vg.grad = None

    def run_fb_shr():
        attend(st_shr.mask, qg, kg, vg).sum().backward()
        qg.grad = kg.grad = vg.grad = None

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    t_bf = _sync_time(run_fb_full)
    t_bs = _sync_time(run_fb_shr)
    print("\n[1] shrink BlockMask metadata")
    print(f"    fwd output equal: {fwd_ok}   bwd grads equal: {bwd_ok}")
    print(f"    metadata: full={meta_full:.1f} MB  shrunk={meta_shr:.3f} MB")
    print(f"    fwd     full={t_ff:6.3f} ms  shrunk={t_fs:6.3f} ms  ({100*(t_ff-t_fs)/t_ff:+.1f}%)")
    print(f"    fwd+bwd full={t_bf:6.3f} ms  shrunk={t_bs:6.3f} ms  ({100*(t_bf-t_bs)/t_bf:+.1f}%)")

    # ---- Suggestion 3: ROWS_GUARANTEED_SAFE ------------------------------------------
    yg = attend_no = None
    attend_safe = torch.compile(
        lambda m, q, k, v: attend_with(m, q, k, v, {"ROWS_GUARANTEED_SAFE": True}),
        dynamic=False, fullgraph=True,
    )
    y_safe = attend_safe(st_full.mask, q, k, v)
    torch.cuda.synchronize()
    safe_ok = torch.allclose(yf, y_safe, atol=1e-3, rtol=1e-3)
    t_safe = _sync_time(lambda: attend_safe(st_full.mask, q, k, v))
    print("\n[3] ROWS_GUARANTEED_SAFE")
    print(f"    output equal: {safe_ok}")
    print(f"    fwd default={t_ff:6.3f} ms  safe={t_safe:6.3f} ms  ({100*(t_ff-t_safe)/t_ff:+.1f}%)")


if __name__ == "__main__":
    main()
