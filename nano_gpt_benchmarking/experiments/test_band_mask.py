"""Validate a C-wide BlockMask whose Q-major (backward) side is built directly from
the routing BAND (O(C*W)) instead of from_kv_blocks' dense transpose (O(C^2 logC)).

Keeps the C-wide index tensors (exactly the validated production shape, so the flex
backward autotune stays valid) but replaces the expensive per-step _transpose_ordered
with a cheap band scatter. Also caches the STATIC diagonal (own-chunk) tensors so they
are not rebuilt every step.

Compares forward + q/k/v grads to production M._flex_attend, compiled, and checks a
SECOND step with a different selection to prove the band update is not stale.
"""
import torch
from _common import M
from torch.nn.attention.flex_attention import BlockMask, flex_attention

DEVICE = "cuda"
T, H, D, CH = 1024, 4, 128, 64
scale = 1.0 / (D ** 0.5)
C = T // CH


def build_sel(topk, shift=0):
    dev = DEVICE
    cidx = torch.arange(C, device=dev)
    pasts = [(cidx - r - shift).clamp(min=0)[:, None] for r in range(1, topk + 1)]
    sel = torch.cat(pasts + [cidx[:, None]], dim=-1)
    sv = torch.ones(C, topk + 1, dtype=torch.bool, device=dev)
    for r in range(1, topk + 1):
        sv[: r + shift, r - 1] = False
    return sel, sv


# --- static diagonal cache (built once per (C, device)) ---
_DIAG_CACHE = {}


def _diag(dev):
    key = (C, dev)
    d = _DIAG_CACHE.get(key)
    if d is None:
        idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=dev)
        idx[0, 0, :, 0] = torch.arange(C, device=dev, dtype=torch.int32)
        num = torch.ones(1, 1, C, dtype=torch.int32, device=dev)
        d = (idx, num)
        _DIAG_CACHE[key] = d
    return d


def band_attend(q, k, v, sel, slot_valid, chunk_size, softmax_scale, window):
    dev = q.device
    Cn, S = sel.shape
    CH = int(chunk_size)
    W = max(1, min(int(window), Cn - 1))
    diag_idx, diag_num = _diag(dev)

    # forward (KV-major), C-wide
    kv_idx = torch.zeros(1, 1, Cn, Cn, dtype=torch.int32, device=dev)
    kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
    kv_num = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32).view(1, 1, Cn)

    # backward (Q-major) transpose, built from the band (O(C*W)), written into C-wide zeros.
    cidx_l = torch.arange(Cn, device=dev)
    band_flat = torch.zeros(Cn * W, dtype=torch.int32, device=dev)
    for slot in range(S - 1):
        j = sel[:, slot].to(torch.long)
        col = cidx_l - j - 1
        ok = (slot_valid[:, slot] & (col >= 0) & (col < W)).to(torch.int32)
        # amax (OR) so an invalid duplicate slot can't clobber a valid edge at the
        # same (key, offset) -- clamp(min=0) collapses out-of-range pasts onto key 0.
        band_flat.scatter_reduce_(0, j * W + col.clamp(0, W - 1), ok,
                                  reduce="amax", include_self=True)
    band_k = (band_flat.view(Cn, W) > 0)
    q_num = band_k.sum(dim=-1, dtype=torch.int32).view(1, 1, Cn)
    order = torch.argsort(band_k.to(torch.int32), dim=1, descending=True, stable=True)
    q_band = (cidx_l[:, None] + order + 1).clamp(max=Cn - 1).to(torch.int32)  # [C,W]
    q_idx = torch.zeros(1, 1, Cn, Cn, dtype=torch.int32, device=dev)
    q_idx[0, 0, :, :W] = q_band

    mask = BlockMask(
        seq_lengths=(T, T),
        kv_num_blocks=diag_num, kv_indices=diag_idx,
        full_kv_num_blocks=kv_num, full_kv_indices=kv_idx,
        q_num_blocks=diag_num, q_indices=diag_idx,
        full_q_num_blocks=q_num, full_q_indices=q_idx,
        BLOCK_SIZE=(CH, CH), mask_mod=M._causal_block_mask_mod,
    )
    qf, kf, vf = (t.transpose(0, 1)[None] for t in (q, k, v))
    y = flex_attention(qf, kf, vf, block_mask=mask, scale=float(softmax_scale),
                       kernel_options={"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH})
    return y[0].transpose(0, 1)


def leaf(t):
    return t.clone().detach().requires_grad_(True)


def compare(ref_fn, new_fn, sel, sv, W, q0, k0, v0, tag):
    qa, ka, va = leaf(q0), leaf(k0), leaf(v0)
    y_ref = ref_fn(qa, ka, va, sel, sv, CH, scale)
    y_ref.float().pow(2).mean().backward()
    qb, kb, vb = leaf(q0), leaf(k0), leaf(v0)
    y_new = new_fn(qb, kb, vb, sel, sv, CH, scale, W)
    y_new.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-9)).item()
    print(f"{tag}: fwd={rel(y_new, y_ref):.2e} "
          f"gq={rel(qb.grad, qa.grad):.2e} gk={rel(kb.grad, ka.grad):.2e} "
          f"gv={rel(vb.grad, va.grad):.2e}")


def main():
    torch.manual_seed(0)
    ref = torch.compile(M._flex_attend, dynamic=False, fullgraph=True)
    new = torch.compile(band_attend, dynamic=False, fullgraph=True)
    q0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    k0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    v0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)

    sel1, sv1 = build_sel(topk=3, shift=0)
    compare(ref, new, sel1, sv1, 8, q0, k0, v0, "step1 topk3")
    # second step, different selection -> proves band update is not stale under reuse
    sel2, sv2 = build_sel(topk=2, shift=1)
    compare(ref, new, sel2, sv2, 8, q0, k0, v0, "step2 topk2 shift1")


if __name__ == "__main__":
    main()
