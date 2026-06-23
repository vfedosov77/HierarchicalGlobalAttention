"""Tests for the NVMe/filesystem KV-cache tier (:class:`FsKVCacheStore`).

These prove the disk-backed store is a *byte-exact* drop-in for the RAM store even when the host-RAM
budget is tiny enough that most chunks spill to disk and must be reloaded, and that it leaves no
temporary files behind.

What is exercised:

1. **Correctness across spill/reload** — every gather (`gather_group_summaries`, `gather_tokens`,
   `gather_chunk_tokens`, `gather_chunk_tokens_kvh`) returns exactly the RAM-store result, for both
   the bulk-seed and incremental-append ingest paths, with ``ram_cap << n_closed`` so the LRU page
   cache must evict to disk and read back.
2. **RAM bound** — at most ``ram_cap`` chunks are ever resident in host staging.
3. **Cleanup** — ``reset()`` deletes spilled file contents; ``close()`` removes the whole temp dir.

Run:  python -m ExistingModelFineTuning.KvRouter.Tests.test_fs_cache
"""

from __future__ import annotations

import os

import torch

from ..cache_store import ChunkPlacementPolicy, FsKVCacheStore, RamKVCacheStore


# Small shapes so the test is fast and the RAM budget can be made tiny on purpose.
B, H, KVH, Dh, C, M = 1, 8, 2, 16, 8, 2
REP = H // KVH


def _policy() -> ChunkPlacementPolicy:
    return ChunkPlacementPolicy(keep_last=0, keep_first=0, first_token_level=False)


def _ram_budget_for(n_cap: int, num_layers: int = 1, dtype: torch.dtype = torch.float32) -> float:
    """ram_budget_gb that resolves to a per-layer RAM capacity of ``n_cap`` chunks (forces spill).

    The store splits the budget into a group-summary tier and a token tier; here we size both to
    ``n_cap`` chunks, so the budget covers ``n_cap`` token chunks plus ``n_cap`` group chunks.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    token_pc = num_layers * B * KVH * C * Dh * itemsize * 2
    group_pc = num_layers * B * KVH * M * Dh * itemsize * 2
    return (n_cap * (token_pc + group_pc)) / 1024**3


def _group_frac(num_layers: int = 1, dtype: torch.dtype = torch.float32) -> float:
    """Group-tier fraction of the budget that yields equal token & group caps (both = n_cap)."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    token_pc = num_layers * B * KVH * C * Dh * itemsize * 2
    group_pc = num_layers * B * KVH * M * Dh * itemsize * 2
    return group_pc / (token_pc + group_pc)


def _ram_store(dtype=torch.float32, scap=0, tcap=0):
    return RamKVCacheStore(
        compute_device=torch.device("cpu"), policy=_policy(),
        kv_heads=KVH, head_dim=Dh, chunk_size=C, groups_per_chunk=M, batch_size=B, dtype=dtype,
        pin_memory=False, storage_device=torch.device("cpu"),
        vram_cache_chunks=tcap, vram_summary_chunks=scap, num_layers=1, vram_cache_reserve_gb=0.0,
    )


def _fs_store(ram_cap, dtype=torch.float32, scap=0, tcap=0, tmpdir=None, max_ctx=None):
    return FsKVCacheStore(
        compute_device=torch.device("cpu"), policy=_policy(),
        kv_heads=KVH, head_dim=Dh, chunk_size=C, groups_per_chunk=M, batch_size=B, dtype=dtype,
        vram_cache_chunks=tcap, vram_summary_chunks=scap, num_layers=1, vram_cache_reserve_gb=0.0,
        ram_budget_gb=_ram_budget_for(ram_cap, dtype=dtype), fs_group_ram_frac=_group_frac(dtype=dtype),
        fs_cache_dir=tmpdir, max_context_tokens=max_ctx,
    )


def _rand_chunks(N, dtype, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    ck = torch.randn(B, KVH, N, Dh, generator=g, dtype=dtype, device=device)
    gk = torch.randn(B, KVH, N, M, Dh, generator=g, dtype=dtype, device=device)
    gv = torch.randn(B, KVH, N, M, Dh, generator=g, dtype=dtype, device=device)
    tk = torch.randn(B, KVH, N, C, Dh, generator=g, dtype=dtype, device=device)
    tv = torch.randn(B, KVH, N, C, Dh, generator=g, dtype=dtype, device=device)
    return ck, gk, gv, tk, tv


def _seed_both(ram, fs, N, dtype):
    data = _rand_chunks(N, dtype)
    ram.seed_closed_chunks(0, *data)
    fs.seed_closed_chunks(0, *data)


def _append_both(ram, fs, N, dtype):
    ck, gk, gv, tk, tv = _rand_chunks(N, dtype)
    for i in range(N):
        ram.append_closed_chunk(0, ck[:, :, i], gk[:, :, i], gv[:, :, i], tk[:, :, i], tv[:, :, i])
        fs.append_closed_chunk(0, ck[:, :, i], gk[:, :, i], gv[:, :, i], tk[:, :, i], tv[:, :, i])


def _max_err(a, b):
    return (a - b).abs().max().item()


def _check_gathers(ram, fs, N, ram_cap, label):
    g = torch.Generator(device="cpu").manual_seed(123)
    worst = 0.0
    for step in range(40):  # many steps so the cap-limited RAM cache evicts & reloads from disk
        # A single gather must keep all its *distinct* chunks resident at once, so the per-step
        # unique set is bounded by ram_cap (in production ram_cap is thousands, far above this).
        # We pick a small random pool of distinct chunk ids that changes every step → forces the
        # LRU page cache to spill old chunks to disk and reload them.
        pool_n = int(torch.randint(2, ram_cap - 1, (1,), generator=g).item())
        pool = torch.randperm(N, generator=g)[:pool_n]
        K = int(torch.randint(1, 7, (1,), generator=g).item())

        def _pick(shape):
            sel = torch.randint(0, pool_n, shape, generator=g)
            return pool[sel]

        idx = _pick((B, H, K))
        rk, rv = ram.gather_group_summaries(0, idx)
        fk, fv = fs.gather_group_summaries(0, idx)
        worst = max(worst, _max_err(rk, fk), _max_err(rv, fv))

        rk, rv = ram.gather_chunk_tokens(0, idx)
        fk, fv = fs.gather_chunk_tokens(0, idx)
        worst = max(worst, _max_err(rk, fk), _max_err(rv, fv))

        # whole-chunk, KV-head granular
        idx_kvh = _pick((B, KVH, K))
        rk, rv = ram.gather_chunk_tokens_kvh(0, idx_kvh)
        fk, fv = fs.gather_chunk_tokens_kvh(0, idx_kvh)
        worst = max(worst, _max_err(rk, fk), _max_err(rv, fv))

        # opened-group token slices
        gidx = torch.randint(0, M, (B, H, K), generator=g)
        rk, rv = ram.gather_tokens(0, idx, gidx)
        fk, fv = fs.gather_tokens(0, idx, gidx)
        worst = max(worst, _max_err(rk, fk), _max_err(rv, fv))

        # RAM stays bounded (independently per tier).
        rec = fs._rec_for(0)
        assert len(rec.tokens.id2slot) <= rec.tokens.ram_cap, \
            f"{label}: token resident {len(rec.tokens.id2slot)} > cap {rec.tokens.ram_cap}"
        assert len(rec.groups.id2slot) <= rec.groups.ram_cap, \
            f"{label}: group resident {len(rec.groups.id2slot)} > cap {rec.groups.ram_cap}"

    assert worst == 0.0, f"{label}: FS gather differs from RAM gather (max abs err {worst})"
    print(f"  [ok] {label}: gathers byte-exact over 40 spill/reload steps (ram_cap={ram_cap}, N={N})")


def test_seed_path(device="cpu"):
    """Bulk-seed ingest: FS == RAM with most chunks spilled to disk."""
    N, ram_cap = 40, 8
    for dtype in (torch.float32, torch.bfloat16):
        ram = _ram_store(dtype)
        fs = _fs_store(ram_cap, dtype)
        try:
            _seed_both(ram, fs, N, dtype)
            rec = fs._rec_for(0)
            assert len(rec.tokens.on_disk) >= N - ram_cap, "seed did not spill token overflow to disk"
            assert len(rec.groups.on_disk) >= N - ram_cap, "seed did not spill group overflow to disk"
            _check_gathers(ram, fs, N, ram_cap, f"seed/{dtype}")
        finally:
            fs.close()


def test_append_path(device="cpu"):
    """Incremental append ingest: FS == RAM with LRU eviction to disk."""
    N, ram_cap = 36, 10
    for dtype in (torch.float32, torch.bfloat16):
        ram = _ram_store(dtype)
        fs = _fs_store(ram_cap, dtype)
        try:
            _append_both(ram, fs, N, dtype)
            fs.disk.flush()  # drain async spills so on_disk reflects everything evicted
            rec = fs._rec_for(0)
            assert len(rec.tokens.on_disk) >= N - ram_cap, "append did not spill tokens to disk"
            assert len(rec.groups.on_disk) >= N - ram_cap, "append did not spill groups to disk"
            _check_gathers(ram, fs, N, ram_cap, f"append/{dtype}")
        finally:
            fs.close()


def test_with_vram_banks(device="cpu"):
    """The token + summary VRAM banks (here on CPU) feed correctly from the disk-backed record."""
    N, ram_cap = 40, 12
    ram = _ram_store(torch.float32, scap=16, tcap=16)
    fs = _fs_store(ram_cap, torch.float32, scap=16, tcap=16)
    try:
        _seed_both(ram, fs, N, torch.float32)
        _check_gathers(ram, fs, N, ram_cap, "banks")
        fs.disk.flush()  # drain async writes; correctness must already hold via pending buffers
        _check_gathers(ram, fs, N, ram_cap, "banks-after-flush")
    finally:
        fs.close()


def test_prealloc(device="cpu"):
    """max_context_tokens reserves the spill file up front and stays byte-exact with the RAM store."""
    N, ram_cap = 40, 8
    max_ctx = N * C  # reserve room for the full context
    ram = _ram_store(torch.float32)
    fs = _fs_store(ram_cap, torch.float32, max_ctx=max_ctx)
    try:
        # Files are preallocated at construction (before any chunk is ingested/spilled).
        rec = fs._rec_for(0)
        n_chunks = (max_ctx + C - 1) // C
        for tier, cache in (("token", rec.tokens), ("group", rec.groups)):
            p = os.path.join(fs.disk.dir, f"L0.{cache.prefix}")
            assert os.path.exists(p), f"{tier} spill file was not preallocated at construction"
            assert os.path.getsize(p) >= n_chunks * cache.cbytes, \
                f"{tier} file not reserved to full context ({os.path.getsize(p)} < {n_chunks * cache.cbytes})"
        _seed_both(ram, fs, N, torch.float32)
        _check_gathers(ram, fs, N, ram_cap, "prealloc")
        print("  [ok] preallocation reserved spill files up front and gathers stayed byte-exact")
    finally:
        fs.close()


def test_cleanup(device="cpu"):
    """reset() truncates spill files; close() removes the temp dir (no leftover files)."""
    N, ram_cap = 40, 8
    fs = _fs_store(ram_cap, torch.float32)
    spill_dir = fs.disk.dir
    try:
        _seed_both(_ram_store(), fs, N, torch.float32)
        fs.disk.flush()
        files = [f for f in os.listdir(spill_dir) if f.startswith("L0.")]
        sizes = {f: os.path.getsize(os.path.join(spill_dir, f)) for f in files}
        assert any(s > 0 for s in sizes.values()), "expected non-empty spill files after seed"

        fs.reset()
        fs.disk.flush()
        for f in files:
            p = os.path.join(spill_dir, f)
            if os.path.exists(p):
                assert os.path.getsize(p) == 0, f"reset() left data in {f}"
        print("  [ok] reset() truncated all spill files")
    finally:
        fs.close()
    assert not os.path.exists(spill_dir), "close() did not remove the spill directory"
    print("  [ok] close() removed the spill directory")


def main() -> None:
    print("FsKVCacheStore tests")
    test_seed_path()
    test_append_path()
    test_with_vram_banks()
    test_prealloc()
    test_cleanup()
    print("All FS-cache tests passed.")


if __name__ == "__main__":
    main()
