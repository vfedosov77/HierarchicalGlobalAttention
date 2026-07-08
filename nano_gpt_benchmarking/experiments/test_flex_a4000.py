"""Test 2: does the FlexAttention block-sparse path run (fwd + bwd) on this box?

This box is an RTX A4000 (CC 8.6, Ampere) -- NOT Hopper (sm90). The production
``_flex_attend`` was written/tuned for sm90 (see ``_patch_flex_bwd_configs`` and the
head_dim==128 backward comment). Here we check the full ``chunk_routed_flex_attention``
call end-to-end, eager AND under ``torch.compile(dynamic=False, fullgraph=True)``,
forward and backward, and compare against a dense reference over the SAME routed
selection (so a mismatch means a flex/kernel bug, not a routing difference).
"""
import torch
import torch.nn.functional as F

from _common import M, make_packed_batch

DEVICE = "cuda"
# Smaller-than-prod shapes so it is quick but keeps head_dim=128 (the tricky one).
T, H, D = 4096, 6, 128
CH = 64
TOPK = 3
WINDOW = 22
MAX_NUM_DOCS = 16
SCALE = 1.0 / (D ** 0.5)


def dense_reference(q, k, v, sel, slot_valid, chunk_size, scale):
    """Reference attend over the exact routed selection (matches _flex_attend semantics):
    each query chunk sees its top-k past chunks (full) + its own chunk (intra causal)."""
    T, H, D = q.shape
    C = T // chunk_size
    S = sel.shape[1]
    dev = q.device
    qf = q.transpose(0, 1)[None].float()  # [1,H,T,D]
    kf = k.transpose(0, 1)[None].float()
    vf = v.transpose(0, 1)[None].float()

    # Build a [T, T] boolean mask from the block selection.
    mask = torch.zeros(T, T, dtype=torch.bool, device=dev)
    q_chunk = torch.arange(T, device=dev) // chunk_size  # [T]
    for slot in range(S):
        kv_chunk = sel[:, slot]                # [C] key chunk id per query chunk
        valid = slot_valid[:, slot]            # [C]
        # For every query token, its query chunk's selected key chunk:
        sel_kchunk = kv_chunk[q_chunk]         # [T]
        sel_valid = valid[q_chunk]             # [T]
        key_in_chunk = (torch.arange(T, device=dev)[None, :] // chunk_size) == sel_kchunk[:, None]
        mask |= key_in_chunk & sel_valid[:, None]
    # Own chunk (last slot) is intra-chunk causal.
    causal = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
    same_chunk = (torch.arange(T, device=dev)[None, :] // chunk_size) == q_chunk[:, None]
    # Remove non-causal own-chunk keys.
    mask = mask & (~same_chunk | causal)

    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn)
    y = torch.einsum("bhqk,bhkd->bhqd", attn, vf)
    return y[0].transpose(0, 1)  # [T,H,D]


def run(compile_it, detach_routed):
    q, k, v, cu = make_packed_batch(T, H, D, n_docs=4, max_num_docs=MAX_NUM_DOCS,
                                    device=DEVICE, seed=1, dtype=torch.bfloat16)
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)

    fn = M.chunk_routed_flex_attention
    if compile_it:
        fn = torch.compile(fn, dynamic=False, fullgraph=True)

    y = fn(q, k, v, cu, chunk_size=CH, topk=TOPK, softmax_scale=SCALE,
           window=WINDOW, detach_routed=detach_routed)
    loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()

    # Reference (recompute selection the same way, no grad).
    with torch.no_grad():
        sel, sv = M._route_static(q.detach(), k.detach(), cu, CH, TOPK, SCALE, WINDOW)
    yref = dense_reference(q.detach(), k.detach(), v.detach(), sel, sv, CH, SCALE)

    err = (y.float() - yref.float()).abs().max().item()
    rel = err / (yref.float().abs().max().item() + 1e-9)
    gnorm = tuple(round(t.grad.float().norm().item(), 6) for t in (q, k, v))
    gfinite = all(torch.isfinite(t.grad).all().item() for t in (q, k, v))
    gnonzero = all(t.grad.float().norm().item() > 0 for t in (q, k, v))
    return err, rel, gnorm, gfinite, gnonzero


def main():
    torch.manual_seed(0)
    for compile_it in (False, True):
        for detach in (True, False):
            tag = f"compile={compile_it} detach_routed={detach}"
            try:
                err, rel, gnorm, gfinite, gnonzero = run(compile_it, detach)
                ok = "OK" if (rel < 5e-2 and gfinite and gnonzero) else "BAD"
                print(f"[{ok}] {tag:40s} fwd max-err={err:.4e} rel={rel:.2e} "
                      f"grad-norms={gnorm} finite={gfinite} nonzero={gnonzero}")
            except Exception:
                import traceback
                print(f"[FAIL] {tag}")
                traceback.print_exc()
                print()


if __name__ == "__main__":
    main()
