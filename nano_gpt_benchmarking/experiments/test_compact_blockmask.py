"""Feasibility + correctness of a COMPACT, band-width BlockMask built directly.

The production ``_flex_attend`` builds the routed BlockMask via
``BlockMask.from_kv_blocks``, which internally densifies to ``[1,1,C,C]`` and
transposes it (``_transpose_ordered``) to get the Q-major (backward) side -- an
O(C^2) op paid EVERY step, and the reason every index tensor is C-wide.

Routing is near-diagonal: a query chunk only ever routes to key chunks in
``[c-W, c-1]`` (window W), and each key chunk is only ever selected by queries in
``[j+1, j+W]``. So BOTH the forward (KV-major, width <= topk) and backward
(Q-major, width <= W) index tensors are band-limited -- never C-wide.

This script builds the BlockMask DIRECTLY with band widths (KV width S-1, Q width
W) and checks that compiled ``flex_attention`` (1) accepts differing/compact
widths and (2) matches the full-width ``from_kv_blocks`` path (production
``_flex_attend``) in forward output AND q/k/v gradients.
"""
import torch

from _common import M
from torch.nn.attention.flex_attention import BlockMask, flex_attention

DEVICE = "cuda"
T, H, D, CH = 1024, 4, 128, 64
scale = 1.0 / (D ** 0.5)
C = T // CH


def build_sel(topk=3):
    """Each chunk c routes to c-1..c-topk (clamped, window-bounded)."""
    dev = DEVICE
    cidx = torch.arange(C, device=dev)
    pasts = [(cidx - r).clamp(min=0)[:, None] for r in range(1, topk + 1)]
    topi = torch.cat(pasts, dim=-1)                       # [C, topk]
    sel = torch.cat([topi, cidx[:, None]], dim=-1)        # [C, topk+1] own last
    slot_valid = torch.ones(C, topk + 1, dtype=torch.bool, device=dev)
    for r in range(1, topk + 1):
        slot_valid[:r, r - 1] = False                     # chunk < r has no (c-r)
    return sel, slot_valid


def build_compact_mask(sel, slot_valid, chunk_size, window):
    """Build a BlockMask directly with band-width index tensors (no C^2 transpose).

    Diagonal (own chunk) -> partial blocks (mask_mod causal); its transpose is the
    identity. Routed past chunks -> full blocks; KV-major width = S-1, Q-major
    (transpose) width = W built from the routing band.
    """
    dev = sel.device
    C, S = sel.shape
    CH = int(chunk_size)
    W = max(1, min(int(window), C - 1))
    Skv = max(1, S - 1)
    cidx = torch.arange(C, device=dev, dtype=torch.int32)

    # --- diagonal (own-chunk) partial blocks; transpose(diagonal) == diagonal ---
    # NOTE: the flex lowering requires the partial (sparse) and full index tensors to
    # share the SAME last-dim width on each side, so pad the diagonal to Skv (KV side)
    # and W (Q side) even though only col 0 is used (num_blocks == 1).
    diag_kv = torch.zeros(1, 1, C, Skv, dtype=torch.int32, device=dev)
    diag_kv[0, 0, :, 0] = cidx
    diag_q = torch.zeros(1, 1, C, W, dtype=torch.int32, device=dev)
    diag_q[0, 0, :, 0] = cidx
    diag_num = torch.ones(1, 1, C, dtype=torch.int32, device=dev)

    # --- routed full blocks: KV-major (forward), compact width S-1 ---
    full_kv_indices = sel[:, : S - 1].to(torch.int32).contiguous().view(1, 1, C, Skv)
    full_kv_num = slot_valid[:, : S - 1].sum(dim=-1, dtype=torch.int32).view(1, 1, C)

    # --- routed full blocks: Q-major (backward) transpose, compact width W ---
    # band_k[j, r-1] == key j is selected by query j+r  (r in 1..W).
    cidx_l = torch.arange(C, device=dev)
    band_flat = torch.zeros(C * W, dtype=torch.bool, device=dev)
    for slot in range(S - 1):
        j = sel[:, slot].to(torch.long)                   # key chunk
        col = cidx_l - j - 1                               # r-1
        ok = slot_valid[:, slot] & (col >= 0) & (col < W)
        col_safe = col.clamp(0, W - 1)
        flat = j * W + col_safe                            # [C]
        band_flat.scatter_(0, flat, ok)
    band_k = band_flat.view(C, W)                          # [C, W]

    full_q_num = band_k.sum(dim=-1, dtype=torch.int32).view(1, 1, C)
    order = torch.argsort(band_k.to(torch.int32), dim=1, descending=True, stable=True)
    q_val = (torch.arange(C, device=dev)[:, None] + order + 1).clamp(max=C - 1)
    full_q_indices = q_val.to(torch.int32).contiguous().view(1, 1, C, W)

    return BlockMask(
        seq_lengths=(T, T),
        kv_num_blocks=diag_num, kv_indices=diag_kv,
        full_kv_num_blocks=full_kv_num, full_kv_indices=full_kv_indices,
        q_num_blocks=diag_num, q_indices=diag_q,
        full_q_num_blocks=full_q_num, full_q_indices=full_q_indices,
        BLOCK_SIZE=(CH, CH), mask_mod=M._causal_block_mask_mod,
    )


def dense_ref(q, k, v, sel, slot_valid, chunk_size, softmax_scale):
    """Ground-truth block-sparse attention in fp32 (own chunk causal + routed full)."""
    dev = q.device
    Cn, S = sel.shape
    CH = int(chunk_size)
    Tn = q.shape[0]
    qf = q.float().transpose(0, 1)                        # [H,T,D]
    kf = k.float().transpose(0, 1)
    vf = v.float().transpose(0, 1)
    scores = torch.einsum("htd,hsd->hts", qf, kf) * float(softmax_scale)  # [H,T,T]
    tok_chunk = torch.arange(Tn, device=dev) // CH        # [T]
    allow = torch.zeros(Tn, Tn, dtype=torch.bool, device=dev)
    qc = tok_chunk[:, None]                               # [T,1]
    kc = tok_chunk[None, :]                               # [1,T]
    # own chunk, causal within
    own = (kc == qc) & (torch.arange(Tn, device=dev)[None, :] <= torch.arange(Tn, device=dev)[:, None])
    allow |= own
    for slot in range(S - 1):
        keyc = sel[:, slot][tok_chunk]                    # [T] routed key chunk of each query token
        valid = slot_valid[:, slot][tok_chunk]            # [T]
        allow |= (kc == keyc[:, None]) & valid[:, None]
    scores = scores.masked_fill(~allow[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    y = torch.einsum("hts,hsd->htd", attn, vf)            # [H,T,D]
    return y.transpose(0, 1)                              # [T,H,D]


def compact_attend(q, k, v, sel, slot_valid, chunk_size, softmax_scale, window):
    mask = build_compact_mask(sel, slot_valid, chunk_size, window)
    qf = q.transpose(0, 1)[None]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    y = flex_attention(
        qf, kf, vf, block_mask=mask, scale=float(softmax_scale),
        kernel_options={"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH},
    )
    return y[0].transpose(0, 1)


def run(compiled):
    torch.manual_seed(0)
    sel, sv = build_sel(topk=3)
    W = 8

    def fresh():
        g = lambda: torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        return g(), g(), g()

    # shared inputs (clone so grads are independent per path)
    q0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    k0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    v0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)

    def leaf(t):
        return t.clone().detach().requires_grad_(True)

    ref_fn = M._flex_attend
    new_fn = compact_attend
    if compiled:
        ref_fn = torch.compile(ref_fn, dynamic=False, fullgraph=True)
        new_fn = torch.compile(new_fn, dynamic=False, fullgraph=True)

    qa, ka, va = leaf(q0), leaf(k0), leaf(v0)
    y_ref = ref_fn(qa, ka, va, sel, sv, CH, scale)
    y_ref.float().pow(2).mean().backward()

    qb, kb, vb = leaf(q0), leaf(k0), leaf(v0)
    y_new = new_fn(qb, kb, vb, sel, sv, CH, scale, W)
    y_new.float().pow(2).mean().backward()

    torch.cuda.synchronize()
    yd = dense_ref(q0, k0, v0, sel, sv, CH, scale).float()
    ref_vs_dense = (y_ref.float() - yd).norm() / yd.norm().clamp(min=1e-9)
    new_vs_dense = (y_new.float() - yd).norm() / yd.norm().clamp(min=1e-9)
    fwd_err = (y_new.float() - y_ref.float()).norm() / y_ref.float().norm().clamp(min=1e-9)
    gq_err = (qb.grad.float() - qa.grad.float()).norm() / qa.grad.float().norm().clamp(min=1e-9)
    gk_err = (kb.grad.float() - ka.grad.float()).norm() / ka.grad.float().norm().clamp(min=1e-9)
    gv_err = (vb.grad.float() - va.grad.float()).norm() / va.grad.float().norm().clamp(min=1e-9)
    print(f"compiled={compiled}: fwd(new_vs_ref)={fwd_err:.2e} "
          f"ref_vs_dense={ref_vs_dense:.2e} new_vs_dense={new_vs_dense:.2e} "
          f"gq={gq_err:.2e} gk={gk_err:.2e} gv={gv_err:.2e}")


def main():
    for compiled in (False, True):
        try:
            run(compiled)
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
