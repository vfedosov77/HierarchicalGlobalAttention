# kv_router — tiered KV-cache router for hierarchical attention

Runs the hierarchical chunk-routed attention with the KV cache spread across
**VRAM / RAM / (later) NVMe**, keeping only summaries and a bounded working set resident.
This is the engineering realisation of `../gen_opt/OFFLOAD_ANALYSIS.md`: per-token IO and
resident memory become ~constant instead of `O(context)`, which is what makes 100K–1M
context usable on a memory-limited GPU.

Two responsibilities, two classes:

| class | file | owns |
|-------|------|------|
| `ChunkRouter` | `chunk_router.py` | *what each token attends to* — chunk/group selection + KV assembly + masks |
| `KVCacheStore` (`RamKVCacheStore`) | `cache_store.py` | *where the KV lives and how it moves between tiers* |

The router owns no storage and no projections; the store owns no attention logic. Swap the
store (RAM → NVMe) without touching the router.

## Storage tiers (per layer, GQA / kv-head granularity)

| artefact | shape / chunk | tier | fetched |
|----------|---------------|------|---------|
| `chunk_k` (routing table) | `[B,KVH,Dh]` | **VRAM**, resident, contiguous | scanned every step |
| `group_k`,`group_v` | `[B,KVH,M,Dh]` | **RAM** (pinned) | only for top-k routed chunks |
| `token_k`,`token_v` | `[B,KVH,C,Dh]` | **RAM/NVMe** | only for the top-k *opened* groups |

Plus two always-resident windows kept *live on the GPU* by `ChunkPlacementPolicy`:
the **last `keep_last`** closed chunks (local context) and the **first `keep_first`**
chunks (attention sinks; `first_token_level` chooses full-KV vs summaries-only).

## Routing (two levels, mirrors the training/decode reference)

1. **Chunks.** Score new tokens vs the resident `chunk_k`, take top-`topk_chunks`. The
   first/last windows are excluded from the candidate pool (so they are never
   double-counted) and exposed unconditionally.
2. **Groups.** Score the routed chunks' group summaries, open top-`topk_groups` to exact
   token KV (downloaded RAM/NVMe → VRAM).

The router returns a `RoutedKV(k, v, mask, scale)`: assembled keys/values `[B,H,R,Dh]` and a
per-token visibility mask `[B,H,L,R]`. Call `routed.attend(q)` for the output, or consume
`k/v/mask` directly in a fused kernel.

## Gradient contract

The router **never detaches on the hot path**. A freshly-closed chunk is handed to the
store as the *live, grad-carrying* tensors just computed; they stay live in the
first/last hot windows and are read back through `hot_group_summaries` / `hot_tokens`. The
single detach point is **eviction from the hot window** (`_offload`) — exactly when a chunk
becomes cache-only. Routing scores are computed under `no_grad`, so `chunk_k` is stored
detached. Routed *old* chunks come from the detached cold record (a cache-only
approximation, as in any incremental decode).

→ You can fine-tune over an offloaded cache: gradients flow through the resident window;
older context is frozen, like activation checkpointing at the window boundary.

## Efficiency

* `chunk_k` resident & contiguous → routing is one matmul + `topk` over an `O(context/C)`
  table (1.1 GiB at 1M ctx for Qwen3-8B).
* Cold gather indexes the **CPU** record and issues **one async H2D copy** of just the
  routed/opened slices (pinned memory ⇒ `non_blocking`, overlaps compute).
* **Chunk-shared routing**: routing + fetch happen once per block (= one chunk of tokens),
  amortising IO to ~5 MiB/token regardless of context.
* GQA: stored at `kv_heads`, expanded to `nhead` on the GPU.
* Growing buffers double in capacity → amortised O(1) append.

Further opportunities (not yet done): union-dedup of routed chunks across heads before
gather; double-buffered prefetch of the next chunk's opened tokens during the current
chunk's compute (the `prefetch` hook exists for this); keeping `group_*` GPU-resident when
it fits.

## Extending to NVMe

Subclass `KVCacheStore` (or replace the `cpu_token_*` tensors in `RamKVCacheStore`):
keep `chunk_k` and the hot windows in VRAM/RAM unchanged; back the cold `token_*` record
with an `mmap`'d file laid out **chunk- and group-contiguous** so an opened group is one
sequential read. Override `prefetch` to issue the async read for the routed set and
`gather_tokens` to consume it. The router needs no changes.

## Streaming contract & usage

`decode_block` processes new tokens inside the *current* chunk and closes it on the last
token; feed longer inputs chunk-by-chunk (`prefill` does this). The caller provides
post-projection, post-RoPE `q`/`k_rope`, the pre-RoPE `k_raw` (needed for the mixed-RoPE
summaries), and `v`.

```python
from kv_router import ChunkRouter, RouterConfig, RamKVCacheStore, ChunkPlacementPolicy

cfg = RouterConfig(nhead=H, kv_heads=KVH, head_dim=Dh, chunk_size=64, group_size=16,
                   topk_chunks=20, topk_groups=32)
policy = ChunkPlacementPolicy(keep_last=1, keep_first=4, first_token_level=False)
store = RamKVCacheStore(compute_device=dev, policy=policy, kv_heads=KVH, head_dim=Dh,
                        chunk_size=64, groups_per_chunk=4, batch_size=B, dtype=torch.bfloat16)
router = ChunkRouter(cfg, store)

# one router/store per attention layer (pass distinct layer ids if sharing a store)
routed = router.decode_block(layer=0, q=q, k_rope=k_rope, k_raw=k_raw, v=v, start_pos=pos)
out = routed.attend(q)            # [B, H, L, Dh]
```

Tests: `python -m ExistingModelFineTuning.kv_router.test_router`
(dense-equivalence when everything is token-level, sparse-routing shapes/causality, grad flow).
