"""Gradient-preserving, VRAM-only KV-cache store.

This is a *standalone* :class:`KVCacheStore` (it deliberately does **not** inherit from
:class:`RamKVCacheStore` — that store exists to spill a *detached* cold record to host RAM, the
exact opposite of what training a routed KV "memory" needs).  Everything lives on the compute
device and every stored K/V slab is kept as a **grad-carrying** tensor, so the router's gather
paths (``gather_tokens`` / ``gather_chunk_tokens`` / ``hot_tokens`` / ``hot_group_summaries``)
return tensors still connected to whatever produced them.

Why a separate class
--------------------
``VramKVCacheStore`` (a thin ``RamKVCacheStore`` subclass) writes the closed-chunk record with
``.detach()`` and serves routed/opened chunks from that detached record — only the small hot
windows keep gradients.  That is correct for *inference* and for normal HA *prefill training*
(where gradients flow through the live block tensors, not the store).  But a persistent, trainable
KV memory that is *seeded into the store* and then read back through routing needs gradients on
the **value path** of every gathered chunk.  This store provides exactly that, with no CPU record,
no LRU VRAM bank and no PCIe traffic, so it is also strictly simpler/faster than the RAM tier.

Differentiability
-----------------
* ``chunk_k`` (the routing scan table) is stored **detached** — selection (top-k) is
  non-differentiable anyway, and :meth:`chunk_summaries` is documented to return a detached view.
* ``group_k/group_v`` and ``token_k/token_v`` are stored **with grad**.  The router's second-level
  group scoring runs under ``torch.no_grad`` (so the summary grad is unused there), but the
  *attended* token K/V (opened groups, hot windows, whole chunks) flow gradients straight back to
  the source memory parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .chunk_placement_policy import ChunkPlacementPolicy
from .kv_cache_store import KVCacheStore


@dataclass
class _GradLayer:
    """Per-layer grad-carrying backing tensors (``None`` until the first chunk is stored)."""

    chunk_k: Optional[torch.Tensor] = None    # [B, KVH, n, Dh]   detached routing table
    group_k: Optional[torch.Tensor] = None    # [B, KVH, n, M, Dh] grad
    group_v: Optional[torch.Tensor] = None    # [B, KVH, n, M, Dh] grad
    token_k: Optional[torch.Tensor] = None    # [B, KVH, n, C, Dh] grad
    token_v: Optional[torch.Tensor] = None    # [B, KVH, n, C, Dh] grad
    n_closed: int = 0


class VramGradKVCacheStore(KVCacheStore):
    """VRAM-only tiered store that keeps every stored K/V slab differentiable.

    One instance serves all layers.  Drop-in for :class:`VramKVCacheStore` (same constructor
    kwargs); extra keyword arguments accepted by the RAM/VRAM stores (LRU bank sizes, reserve,
    ``num_layers`` …) are accepted and ignored, since there is no cold tier here.
    """

    def __init__(
        self,
        *,
        compute_device: torch.device,
        policy: ChunkPlacementPolicy,
        kv_heads: int,
        head_dim: int,
        chunk_size: int,
        groups_per_chunk: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        **_ignored,
    ) -> None:
        super().__init__(compute_device=compute_device, policy=policy)
        self.kvh = kv_heads
        self.dh = head_dim
        self.C = chunk_size
        self.M = groups_per_chunk
        self.B = batch_size
        self.dtype = dtype
        self._layers: Dict[int, _GradLayer] = {}

    # -- internal ----------------------------------------------------------
    def _layer(self, layer: int) -> _GradLayer:
        st = self._layers.get(layer)
        if st is None:
            st = _GradLayer()
            self._layers[layer] = st
        return st

    def _gather(self, src: torch.Tensor, chunk_idx: torch.Tensor, rep: int) -> torch.Tensor:
        """Gather ``src[B, KVH, n, *tail]`` by ``chunk_idx[B, H, *]`` → ``[B, H, *, tail]``.

        Pure same-device advanced indexing (no copy across tiers); grad flows back to ``src``.
        ``rep = H // KVH`` maps each query head onto its kv-head (GQA).
        """
        B = src.shape[0]
        H = chunk_idx.shape[1]
        extra = chunk_idx.ndim - 2  # trailing index dims after (B, H)
        dev = src.device
        b = torch.arange(B, device=dev).view(B, 1, *([1] * extra))
        kv = (torch.arange(H, device=dev) // rep).view(1, H, *([1] * extra))
        return src[b, kv, chunk_idx]

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._layers.clear()

    def num_closed_chunks(self, layer: int) -> int:
        return self._layer(layer).n_closed

    # -- routing table (HOT) ----------------------------------------------
    def chunk_summaries(self, layer: int) -> Optional[torch.Tensor]:
        st = self._layer(layer)
        if st.chunk_k is None or st.n_closed == 0:
            return None
        return st.chunk_k[:, :, : st.n_closed]

    # -- ingest ------------------------------------------------------------
    def append_closed_chunk(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, Dh]
        group_k: torch.Tensor,   # [B, KVH, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, C, Dh]
    ) -> None:
        st = self._layer(layer)
        ck = chunk_k.detach().unsqueeze(2)         # routing table: detached (top-k is non-diff)
        gk = group_k.unsqueeze(2)                  # value/summary path: grad kept
        gv = group_v.unsqueeze(2)
        tk = token_k.unsqueeze(2)
        tv = token_v.unsqueeze(2)
        if st.chunk_k is None:
            st.chunk_k, st.group_k, st.group_v, st.token_k, st.token_v = ck, gk, gv, tk, tv
        else:
            st.chunk_k = torch.cat([st.chunk_k, ck], dim=2)
            st.group_k = torch.cat([st.group_k, gk], dim=2)
            st.group_v = torch.cat([st.group_v, gv], dim=2)
            st.token_k = torch.cat([st.token_k, tk], dim=2)
            st.token_v = torch.cat([st.token_v, tv], dim=2)
        st.n_closed += 1

    def seed_closed_chunks(
        self,
        layer: int,
        chunk_k: torch.Tensor,   # [B, KVH, N, Dh]
        group_k: torch.Tensor,   # [B, KVH, N, M, Dh]
        group_v: torch.Tensor,   # [B, KVH, N, M, Dh]
        token_k: torch.Tensor,   # [B, KVH, N, C, Dh]
        token_v: torch.Tensor,   # [B, KVH, N, C, Dh]
    ) -> None:
        """Bulk-ingest ``N`` closed chunks (the vectorized prefill seed / memory seed).

        Unlike the RAM store this keeps ``group_*`` / ``token_*`` **with grad** so a seeded,
        trainable KV memory stays differentiable through every later gather.  ``chunk_k`` (the
        scan table) is detached.  Appended after existing chunks (so a memory seed followed by
        streamed sequence chunks share one contiguous index space).
        """
        st = self._layer(layer)
        N = chunk_k.shape[2]
        if N == 0:
            return
        ck = chunk_k.detach()
        if st.chunk_k is None:
            st.chunk_k, st.group_k, st.group_v = ck, group_k, group_v
            st.token_k, st.token_v = token_k, token_v
        else:
            st.chunk_k = torch.cat([st.chunk_k, ck], dim=2)
            st.group_k = torch.cat([st.group_k, group_k], dim=2)
            st.group_v = torch.cat([st.group_v, group_v], dim=2)
            st.token_k = torch.cat([st.token_k, token_k], dim=2)
            st.token_v = torch.cat([st.token_v, token_v], dim=2)
        st.n_closed += N

    # -- fetch into VRAM (grad-preserving) --------------------------------
    def gather_group_summaries(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        return self._gather(st.group_k, chunk_idx, rep), self._gather(st.group_v, chunk_idx, rep)

    def gather_chunk_tokens(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-chunk token K/V for ``chunk_idx[B, H, *]`` → ``[B, H, *, C, Dh]`` (grad kept)."""
        st = self._layer(layer)
        rep = chunk_idx.shape[1] // self.kvh
        return self._gather(st.token_k, chunk_idx, rep), self._gather(st.token_v, chunk_idx, rep)

    def gather_chunk_tokens_kvh(
        self, layer: int, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """KV-head-granular full-chunk token K/V for ``chunk_idx[B, KVH, K]`` → ``[B, KVH, K, C, Dh]``."""
        st = self._layer(layer)
        B, KVH, _ = chunk_idx.shape
        dev = st.token_k.device
        b = torch.arange(B, device=dev).view(B, 1, 1)
        g = torch.arange(KVH, device=dev).view(1, KVH, 1)
        return st.token_k[b, g, chunk_idx], st.token_v[b, g, chunk_idx]

    def gather_tokens(
        self, layer: int, chunk_idx: torch.Tensor, group_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opened-group token K/V for ``(chunk_idx, group_idx)[B, H, Kg]`` → ``[B, H, Kg, gs, Dh]``.

        This is the differentiable hot path for a routed memory: grad flows back to ``token_k`` /
        ``token_v`` (hence to the source memory parameters) for exactly the opened tokens.
        """
        st = self._layer(layer)
        H = chunk_idx.shape[1]
        rep = H // self.kvh
        gs = self.C // self.M
        dev = st.token_k.device
        tok = group_idx.unsqueeze(-1) * gs + torch.arange(gs, device=dev)   # [B, H, Kg, gs]
        cidx = chunk_idx.unsqueeze(-1).expand_as(tok)                       # [B, H, Kg, gs]
        b = torch.arange(self.B, device=dev).view(self.B, 1, 1, 1)
        kv = (torch.arange(H, device=dev) // rep).view(1, H, 1, 1)
        return st.token_k[b, kv, cidx, tok], st.token_v[b, kv, cidx, tok]

    # -- always-resident windows (grad kept) ------------------------------
    def hot_group_summaries(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        return st.group_k[:, :, lo:hi], st.group_v[:, :, lo:hi]

    def hot_tokens(self, layer: int, lo: int, hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self._layer(layer)
        return st.token_k[:, :, lo:hi], st.token_v[:, :, lo:hi]
