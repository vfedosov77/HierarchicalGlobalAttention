"""Test 1: does ``_route_static`` recompile with the number of documents?

Wraps ``_route_static`` in ``torch.compile(dynamic=False, fullgraph=True)`` (the
production compile flags) and feeds it many batches whose *document count* varies
while T, H, D, chunk_size, window, topk and the cu_seqlens LENGTH stay fixed
(exactly what the training loader guarantees: cu_seqlens is padded to a fixed
``max_num_docs``). We count dynamo recompiles via the guard-failure log.

If the count stays at 1 (the initial compile) across all document counts, the
document count does NOT trigger recompiles.
"""
import torch
import torch._dynamo as dynamo

from _common import M, make_packed_batch

DEVICE = "cuda"
T, H, D = 49152, 6, 128
CH = 64
TOPK = 3
WINDOW = 22
MAX_NUM_DOCS = 128
SCALE = 1.0 / (D ** 0.5)


def main():
    torch.manual_seed(0)
    dynamo.reset()

    # Count recompiles by hooking the dynamo recompile reason logger.
    recompiles = []
    from torch._dynamo import convert_frame

    compiled = torch.compile(M._route_static, dynamic=False, fullgraph=True)

    # Track number of distinct compiled graphs via CompileCounterWithBackend-style
    # counter: use torch._dynamo.utils.counters.
    from torch._dynamo.utils import counters
    counters.clear()

    doc_counts = [1, 2, 5, 8, 17, 33, 64, 96, 127, 3, 50, 96, 8]
    for i, nd in enumerate(doc_counts):
        q, k, v, cu = make_packed_batch(
            T, H, D, n_docs=nd, max_num_docs=MAX_NUM_DOCS, device=DEVICE, seed=i,
        )
        sel, slot_valid = compiled(q, k, cu, CH, TOPK, SCALE, WINDOW)
        torch.cuda.synchronize()
        n_compiles = counters["stats"].get("unique_graphs", 0)
        print(f"  batch {i:2d}  n_docs={nd:4d}  cu.shape={tuple(cu.shape)}  "
              f"sel.shape={tuple(sel.shape)}  unique_graphs={n_compiles}")

    print()
    print("dynamo counters['stats']:", dict(counters["stats"]))
    frame_count = counters["stats"].get("unique_graphs", 0)
    if frame_count <= 1:
        print("RESULT: NO recompile with document count (1 graph). ✓")
    else:
        print(f"RESULT: {frame_count} graphs compiled -> investigate recompiles.")


if __name__ == "__main__":
    main()
