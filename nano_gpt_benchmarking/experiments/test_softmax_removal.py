"""Test 3: can the softmax in ``_route_static`` be removed?

Current routing selects the top-k past chunks by HEAD-AVERAGED SOFTMAX weights:
    w[c,j] = mean_h softmax_j(scores[h,c,:])[j]      # scores masked to same-doc window
Proposed: drop softmax and top-k the HEAD-AVERAGED RAW scores (masking + top-k done
BEFORE any softmax), which is what the user suggested:
    w[c,j] = mean_h scores[h,c,j]                     # masked with -inf, then top-k

Since the top-k only picks WHICH chunks to attend, and the flex attend recomputes a
real softmax over the selected tokens, the ONLY thing that can change is the selected
SET. If the sets match, the attention output is bit-identical. We measure per-query-chunk
set agreement (order within top-k is irrelevant: all past chunks are full blocks).

Reasoning: q,k are RMS-normed per head and scaled by 1/sqrt(D), so scores are small.
In that regime softmax(s) ~= uniform + linear(s), so mean-over-heads of softmax ranks
chunks the same as mean-over-heads of raw s -> removal should barely change selection.
"""
import torch
import torch.nn.functional as F

from _common import M, make_packed_batch

DEVICE = "cuda"
CH = 64
TOPK = 3
WINDOW = 22


def route_static_softmax(q, k, cu_seqlens, chunk_size, topk, softmax_scale, window):
    """The ORIGINAL ``_route_static`` (with softmax), kept here for the comparison record.
    Head-averaged per-head softmax weights of the masked band, then top-k of the past."""
    T, H, D = q.shape
    C = T // chunk_size
    dev = q.device
    q_sum = q.view(C, chunk_size, H, D).float().mean(dim=1)
    k_sum = k.view(C, chunk_size, H, D).float().mean(dim=1)

    cidx = torch.arange(C, device=dev)
    last_tok = cidx * chunk_size + (chunk_size - 1)
    doc_ends = cu_seqlens.to(torch.long)[1:]
    doc_of_chunk = (last_tok[:, None] >= doc_ends[None, :]).sum(dim=-1)

    W = max(0, min(int(window), C - 1))
    kpad = F.pad(k_sum.reshape(C, H * D), (0, 0, W, 0))
    kb = kpad.unfold(0, W + 1, 1).view(C, H, D, W + 1)
    scores = torch.einsum("chd,chdj->hcj", q_sum * float(softmax_scale), kb)

    dpad = F.pad(doc_of_chunk + 1, (W, 0))
    d_band = dpad.unfold(0, W + 1, 1)
    allowed = d_band == (doc_of_chunk + 1)[:, None]
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    w = attn.mean(dim=0)
    w = w[:, :W].masked_fill(~allowed[:, :W], -1.0)
    topv, topj = w.topk(min(int(topk), W), dim=-1)
    valid_past = topv >= 0
    topi = (cidx[:, None] - W + topj).clamp(min=0)
    sel = torch.cat([topi, cidx[:, None]], dim=-1)
    slot_valid = torch.cat(
        [valid_past, torch.ones(C, 1, dtype=torch.bool, device=dev)], dim=-1,
    )
    return sel, slot_valid


def route_static_nosoftmax(q, k, cu_seqlens, chunk_size, topk, softmax_scale, window):
    """``_route_static`` with the softmax removed: mask -> head-mean -> top-k on raw scores."""
    T, H, D = q.shape
    C = T // chunk_size
    dev = q.device
    q_sum = q.view(C, chunk_size, H, D).float().mean(dim=1)
    k_sum = k.view(C, chunk_size, H, D).float().mean(dim=1)

    cidx = torch.arange(C, device=dev)
    last_tok = cidx * chunk_size + (chunk_size - 1)
    doc_ends = cu_seqlens.to(torch.long)[1:]
    doc_of_chunk = (last_tok[:, None] >= doc_ends[None, :]).sum(dim=-1)

    W = max(0, min(int(window), C - 1))
    kpad = F.pad(k_sum.reshape(C, H * D), (0, 0, W, 0))
    kb = kpad.unfold(0, W + 1, 1).view(C, H, D, W + 1)
    scores = torch.einsum("chd,chdj->hcj", q_sum * float(softmax_scale), kb)  # [H,C,W+1]

    dpad = F.pad(doc_of_chunk + 1, (W, 0))
    d_band = dpad.unfold(0, W + 1, 1)
    allowed = d_band == (doc_of_chunk + 1)[:, None]              # [C, W+1]

    # NO SOFTMAX: head-mean raw scores, mask (before top-k), then top-k the past band.
    s = scores.mean(dim=0)                                       # [C, W+1]
    s = s.masked_fill(~allowed, float("-inf"))
    s_past = s[:, :W]                                            # drop own chunk (col W)
    topv, topj = s_past.topk(min(int(topk), W), dim=-1)
    valid_past = torch.isfinite(topv)
    topi = (cidx[:, None] - W + topj).clamp(min=0)

    sel = torch.cat([topi, cidx[:, None]], dim=-1)
    slot_valid = torch.cat(
        [valid_past, torch.ones(C, 1, dtype=torch.bool, device=dev)], dim=-1,
    )
    return sel, slot_valid


def set_agreement(selA, svA, selB, svB):
    """Per-query-chunk: does the selected PAST set match (ignoring order & own chunk)?"""
    C, S = selA.shape
    W = S - 1
    agree = 0
    jacc_sum = 0.0
    for c in range(C):
        a = set(int(selA[c, j]) for j in range(W) if bool(svA[c, j]))
        b = set(int(selB[c, j]) for j in range(W) if bool(svB[c, j]))
        if a == b:
            agree += 1
        union = a | b
        jacc_sum += (len(a & b) / len(union)) if union else 1.0
    return agree / C, jacc_sum / C


def main():
    torch.manual_seed(0)
    for T, ndoc in [(8192, 1), (8192, 8), (16384, 32), (16384, 4)]:
        maxd = 128
        q, k, v, cu = make_packed_batch(T, 6, 128, n_docs=ndoc, max_num_docs=maxd,
                                        device=DEVICE, seed=ndoc, dtype=torch.bfloat16)
        scale = 1.0 / (128 ** 0.5)
        with torch.no_grad():
            # selA = ORIGINAL softmax router (kept locally); selB = shipped no-softmax
            # _route_static. Documents how much the softmax removal changes selections.
            selA, svA = route_static_softmax(q, k, cu, CH, TOPK, scale, WINDOW)
            selB, svB = M._route_static(q, k, cu, CH, TOPK, scale, WINDOW)
        exact, jacc = set_agreement(selA, svA, selB, svB)

        # Also measure the resulting attention output difference (compiled flex).
        fa = torch.compile(M._flex_attend, dynamic=False, fullgraph=True)
        with torch.no_grad():
            yA = fa(q, k, v, selA, svA, CH, scale)
            yB = fa(q, k, v, selB, svB, CH, scale)
        out_rel = ((yA.float() - yB.float()).abs().max()
                   / (yA.float().abs().max() + 1e-9)).item()
        print(f"T={T:6d} docs={ndoc:3d}  set-exact-match={exact:6.3f}  "
              f"mean-Jaccard={jacc:6.3f}  out-rel-diff={out_rel:.3e}")


if __name__ == "__main__":
    main()
