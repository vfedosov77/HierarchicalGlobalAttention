"""Engine-neutral HGA configuration.

A single immutable ``HgaConfig`` describes the routing geometry, the deterministic
sink/local windows, and the tiered-cache budgets.  It is shared verbatim by the
SGLang backend, the (future) vLLM backend, the fused kernels, and the cache
manager so that all paths agree on chunk/group/page geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HgaConfig:
    # ---- model geometry ----
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    # ---- routing geometry (see README page-layout) ----
    chunk_size: int = 64           # tokens per HGA chunk
    group_size: int = 16           # tokens per group (4 groups / chunk by default)
    page_size: int = 16            # KV page granularity for v0 (benchmark 64 later)

    # ---- deterministic windows ----
    keep_first: int = 2            # sink chunks (always visible)
    keep_last: int = 8             # recent local chunks (always visible)

    # ---- top-k budgets ----
    topk_chunks: int = 16          # candidate remote chunks
    topk_groups: int = 64          # opened remote groups (~1024 routed tokens)
    # Per-token *request* budgets for sticky L==1 decode (None -> topk // 2).
    topk_chunks_request: Optional[int] = None
    topk_groups_request: Optional[int] = None
    decode_sticky: bool = True

    # ---- positional ----
    rope_theta: float = 1_000_000.0
    # v0: DCA disabled. dca_chunk == 0 keeps native absolute-RoPE HGA.
    dca_chunk: int = 0
    dca_local: int = 0

    # ---- tiered cache budgets ----
    # Chunk summaries: always GPU-resident.  Group summaries: GPU if context
    # fits ``gpu_summary_chunks``; token K/V live in pinned host RAM with a small
    # GPU token bank (``gpu_token_chunks``).  FS is an opt-in L3 spill tier.
    gpu_summary_chunks: int = 8192
    gpu_token_chunks: int = 512
    host_pinned_gb: float = 12.0
    enable_fs_spill: bool = False
    fs_cache_dir: Optional[str] = None

    # ---- runtime ----
    kv_dtype: str = "bfloat16"     # token K/V storage dtype ("bfloat16"|"float16"|"fp8_e4m3")
    prefetch_stream: bool = True   # async chunk prefetch on a side CUDA stream
    cuda_graph: bool = False       # enable decode CUDA-graph capture (added after eager works)

    # mixed-rope summary cutoff (None -> derived from head_dim / threshold)
    mixed_rope_threshold: float = 0.5
    mixed_rope_cutoff_pair: Optional[int] = None

    def __post_init__(self) -> None:
        if self.chunk_size % self.group_size != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be a multiple of "
                f"group_size ({self.group_size})"
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.dca_chunk and self.dca_chunk % self.chunk_size != 0:
            raise ValueError("dca_chunk must be a multiple of chunk_size")

    # ---- derived geometry ----
    @property
    def groups_per_chunk(self) -> int:
        return self.chunk_size // self.group_size

    @property
    def gqa_rep(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def group_kv_scale(self) -> float:
        # Pooling scale for group/chunk summaries (reference heuristic, kept for
        # checkpoint/behaviour continuity).
        return 1.0 / (self.group_size + math.sqrt(self.group_size))

    @property
    def effective_topk_chunks_request(self) -> int:
        return self.topk_chunks_request if self.topk_chunks_request is not None else self.topk_chunks // 2

    @property
    def effective_topk_groups_request(self) -> int:
        return self.topk_groups_request if self.topk_groups_request is not None else self.topk_groups // 2

    @property
    def mixed_rope_cutoff(self) -> int:
        """Pair index below which summaries keep RoPE phase; above which they are pooled raw."""
        if self.mixed_rope_cutoff_pair is not None:
            return int(self.mixed_rope_cutoff_pair)
        # half the head-dim pairs by default (matches reference threshold=0.5).
        return int(round((self.head_dim // 2) * self.mixed_rope_threshold))

    def routed_token_budget(self) -> int:
        """Worst-case number of token-level keys the fused decode kernel must cover."""
        sink = self.keep_first * self.chunk_size
        local = self.keep_last * self.chunk_size
        opened = self.topk_groups * self.group_size
        active = self.chunk_size
        return sink + local + opened + active

    def estimate_host_kv_chunks(self) -> int:
        """How many chunks of token K/V fit in the pinned-host budget (both K and V)."""
        bytes_per = self._kv_bytes()
        per_chunk = self.num_layers * self.num_kv_heads * self.chunk_size * self.head_dim * 2 * bytes_per
        return max(1, int(self.host_pinned_gb * (1024 ** 3) // per_chunk))

    def _kv_bytes(self) -> int:
        return {"bfloat16": 2, "float16": 2, "fp8_e4m3": 1}.get(self.kv_dtype, 2)
