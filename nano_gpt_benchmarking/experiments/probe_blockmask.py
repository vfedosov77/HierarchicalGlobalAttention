"""Probe WHY a directly-built BlockMask is wrong under compile vs from_kv_blocks.

Variants (all compiled, compared to fp32 dense ground truth):
  A: production _flex_attend (from_kv_blocks, C-wide)           [known good]
  B: direct BlockMask, C-wide, q-side via _transpose_ordered    (isolates "direct build")
  C: direct BlockMask, compact widths, q-side via band          (the target)
  D: direct BlockMask, compact widths, but partial padded to S-1 width
"""
import torch
from _common import M
from torch.nn.attention.flex_attention import (
    BlockMask, flex_attention, _transpose_ordered,
)

DEVICE = "cuda"
T, H, D, CH = 1024, 4, 128, 64
scale = 1.0 / (D ** 0.5)
C = T // CH
W = 8
TOPK = 3


def build_sel(topk=TOPK):
    dev = DEVICE
    cidx = torch.arange(C, device=dev)
    pasts = [(cidx - r).clamp(min=0)[:, None] for r in range(1, topk + 1)]
    sel = torch.cat(pasts + [cidx[:, None]], dim=-1)
    sv = torch.ones(C, topk + 1, dtype=torch.bool, device=dev)
    for r in range(1, topk + 1):
        sv[:r, r - 1] = False
    return sel, sv


def dense_ref(q, k, v, sel, sv):
    dev = q.device
    S = sel.shape[1]
    qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k, v))
    scores = torch.einsum("htd,hsd->hts", qf, kf) * scale
    tok = torch.arange(T, device=dev) // CH
    ar = torch.arange(T, device=dev)
    allow = (tok[None] == tok[:, None]) & (ar[None] <= ar[:, None])
    for slot in range(S - 1):
        keyc = sel[:, slot][tok]
        val = sv[:, slot][tok]
        allow = allow | ((tok[None] == keyc[:, None]) & val[:, None])
    scores = scores.masked_fill(~allow[None], float("-inf"))
    y = torch.einsum("hts,hsd->htd", torch.softmax(scores, -1), vf)
    return y.transpose(0, 1)


def _run_flex(mask, q, k, v):
    qf, kf, vf = (t.transpose(0, 1)[None] for t in (q, k, v))
    y = flex_attention(qf, kf, vf, block_mask=mask, scale=scale,
                       kernel_options={"fwd_BLOCK_M": CH, "fwd_BLOCK_N": CH})
    return y[0].transpose(0, 1)


def variantB(q, k, v, sel, sv):
    dev = q.device
    S = sel.shape[1]
    kv_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=dev)
    kv_idx[0, 0, :, : S - 1] = sel[:, : S - 1].to(torch.int32)
    kv_num = sv[:, : S - 1].sum(-1, dtype=torch.int32).view(1, 1, C)
    diag_idx = torch.zeros(1, 1, C, C, dtype=torch.int32, device=dev)
    diag_idx[0, 0, :, 0] = torch.arange(C, device=dev, dtype=torch.int32)
    diag_num = torch.ones(1, 1, C, dtype=torch.int32, device=dev)
    q_num, q_idx = _transpose_ordered(diag_num, diag_idx)
    fq_num, fq_idx = _transpose_ordered(kv_num, kv_idx)
    mask = BlockMask(
        seq_lengths=(T, T),
        kv_num_blocks=diag_num, kv_indices=diag_idx,
        full_kv_num_blocks=kv_num, full_kv_indices=kv_idx,
        q_num_blocks=q_num, q_indices=q_idx,
        full_q_num_blocks=fq_num, full_q_indices=fq_idx,
        BLOCK_SIZE=(CH, CH), mask_mod=M._causal_block_mask_mod,
    )
    return _run_flex(mask, q, k, v)


def _compact_q_side(sel, sv, S, Wq):
    dev = sel.device
    cidx_l = torch.arange(C, device=dev)
    band_flat = torch.zeros(C * Wq, dtype=torch.bool, device=dev)
    for slot in range(S - 1):
        j = sel[:, slot].to(torch.long)
        col = cidx_l - j - 1
        ok = sv[:, slot] & (col >= 0) & (col < Wq)
        flat = j * Wq + col.clamp(0, Wq - 1)
        band_flat.scatter_(0, flat, ok)
    band_k = band_flat.view(C, Wq)
    fq_num = band_k.sum(-1, dtype=torch.int32).view(1, 1, C)
    order = torch.argsort(band_k.to(torch.int32), dim=1, descending=True, stable=True)
    q_val = (torch.arange(C, device=dev)[:, None] + order + 1).clamp(max=C - 1)
    return fq_num, q_val.to(torch.int32).contiguous().view(1, 1, C, Wq)


def variantC(q, k, v, sel, sv, pad_partial=False):
    dev = q.device
    S = sel.shape[1]
    cidx = torch.arange(C, device=dev, dtype=torch.int32)
    pw = (S - 1) if pad_partial else 1
    diag_idx = torch.zeros(1, 1, C, pw, dtype=torch.int32, device=dev)
    diag_idx[0, 0, :, 0] = cidx
    diag_num = torch.ones(1, 1, C, dtype=torch.int32, device=dev)
    fkv_idx = sel[:, : S - 1].to(torch.int32).contiguous().view(1, 1, C, S - 1)
    fkv_num = sv[:, : S - 1].sum(-1, dtype=torch.int32).view(1, 1, C)
    # q partial: identity, width matches (pad_partial -> W else 1)
    qw = W if pad_partial else 1
    q_idx = torch.zeros(1, 1, C, qw, dtype=torch.int32, device=dev)
    q_idx[0, 0, :, 0] = cidx
    q_num = torch.ones(1, 1, C, dtype=torch.int32, device=dev)
    fq_num, fq_idx = _compact_q_side(sel, sv, S, W)
    mask = BlockMask(
        seq_lengths=(T, T),
        kv_num_blocks=diag_num, kv_indices=diag_idx,
        full_kv_num_blocks=fkv_num, full_kv_indices=fkv_idx,
        q_num_blocks=q_num, q_indices=q_idx,
        full_q_num_blocks=fq_num, full_q_indices=fq_idx,
        BLOCK_SIZE=(CH, CH), mask_mod=M._causal_block_mask_mod,
    )
    return _run_flex(mask, q, k, v)


def main():
    torch.manual_seed(0)
    sel, sv = build_sel()
    q0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    k0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    v0 = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    yd = dense_ref(q0, k0, v0, sel, sv).float()

    def rel(y):
        return ((y.float() - yd).norm() / yd.norm()).item()

    variants = {
        "A_prod": lambda: M._flex_attend(q0, k0, v0, sel, sv, CH, scale),
        "B_direct_Cwide": lambda: variantB(q0, k0, v0, sel, sv),
        "C_compact": lambda: variantC(q0, k0, v0, sel, sv, pad_partial=False),
        "D_compact_padpartial": lambda: variantC(q0, k0, v0, sel, sv, pad_partial=True),
    }
    for name, fn in variants.items():
        try:
            cfn = torch.compile(fn, dynamic=False, fullgraph=True)
            y = cfn()
            torch.cuda.synchronize()
            print(f"{name:24s} rel_vs_dense={rel(y):.3e}")
        except Exception as e:
            print(f"{name:24s} ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
