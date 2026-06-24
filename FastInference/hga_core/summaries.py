"""Engine-neutral mixed-RoPE summary builder.

HGA does **not** add learned summary projections.  Chunk- and group-level
routing summaries are pooled directly from the model's own projected K/V, with a
mixed-RoPE rule that keeps the low-frequency (position-sensitive) RoPE pairs
phase-correct while letting high-frequency pairs be pooled raw and re-anchored at
the chunk/group endpoint position.

This module ports the reference math (``ChunkRouter._rope_summary_at`` &
friends) into stateless functions so the SGLang/vLLM backends build identical
summaries during prefill without importing the HF reference module.

Conventions
-----------
* ``k_raw``  — pre-RoPE projected keys ``[..., T, Dh]``.
* ``k_rope`` — RoPE-applied keys ``[..., T, Dh]`` (same layout).
* ``v``      — values ``[..., T, Dh]`` (no RoPE).
* Summaries reduce the ``T`` (token) axis of each group/chunk.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .config import HgaConfig


def _inv_freq(head_dim: int, theta: float, device, dtype=torch.float32) -> torch.Tensor:
    half = head_dim // 2
    return 1.0 / (theta ** (torch.arange(half, device=device, dtype=dtype) / half))


def _rotary_for_positions(pos: torch.Tensor, head_dim: int, theta: float, like: torch.Tensor):
    half = head_dim // 2
    inv_freq = _inv_freq(head_dim, theta, like.device)
    freqs = pos.to(device=like.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    view = (1, 1) + tuple(pos.shape) + (head_dim,)
    return emb.cos().reshape(view), emb.sin().reshape(view)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _mix_tokenwise_and_anchor(cfg: HgaConfig, tokenwise: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    half = cfg.head_dim // 2
    cutoff = cfg.mixed_rope_cutoff
    pair_mask = torch.arange(half, device=tokenwise.device) < cutoff
    mask = torch.cat([pair_mask, pair_mask], dim=0)
    view_shape = [1] * tokenwise.ndim
    view_shape[-1] = cfg.head_dim
    return torch.where(mask.view(*view_shape), tokenwise, anchor)


def rope_summary(
    cfg: HgaConfig,
    raw: torch.Tensor,          # [..., T, Dh] pre-rope
    rope: torch.Tensor,         # [..., T, Dh] rope-applied
    reduce_dim: int,
    anchor_pos: torch.Tensor,   # broadcastable to the reduced shape
    scale: float,
) -> torch.Tensor:
    """Mixed-RoPE pooled summary over ``reduce_dim`` (the token axis of a group/chunk)."""
    raw_sum = raw.sum(dim=reduce_dim) * scale
    tokenwise = rope.sum(dim=reduce_dim) * scale
    a_cos, a_sin = _rotary_for_positions(anchor_pos, cfg.head_dim, cfg.rope_theta, raw_sum)
    endpoint = _apply_rotary(raw_sum.float(), a_cos, a_sin).to(dtype=raw_sum.dtype)
    return _mix_tokenwise_and_anchor(cfg, tokenwise, endpoint)


def build_group_summaries(
    cfg: HgaConfig,
    k_raw: torch.Tensor,        # [B, KVH, n_chunks, M, gs, Dh] pre-rope
    k_rope: torch.Tensor,       # [B, KVH, n_chunks, M, gs, Dh] rope-applied
    v: torch.Tensor,            # [B, KVH, n_chunks, M, gs, Dh]
    chunk_start_pos: torch.Tensor,  # [n_chunks] absolute start position of each chunk
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-level K/V summaries ``[B, KVH, n_chunks, M, Dh]``.

    The K summary anchors RoPE at each group's *last* token position; the V
    summary is a plain mean (values carry no positional phase).
    """
    M, gs = cfg.groups_per_chunk, cfg.group_size
    scale = cfg.group_kv_scale
    device = k_raw.device
    # anchor = last token of each group: chunk_start + g*gs + (gs-1)
    g = torch.arange(M, device=device)
    anchor = chunk_start_pos.view(-1, 1) + g.view(1, -1) * gs + (gs - 1)  # [n_chunks, M]
    group_k = rope_summary(cfg, k_raw, k_rope, reduce_dim=-2, anchor_pos=anchor, scale=scale)
    group_v = v.sum(dim=-2) * scale
    return group_k, group_v


def build_chunk_summaries(
    cfg: HgaConfig,
    group_k: torch.Tensor,      # [B, KVH, n_chunks, M, Dh]
) -> torch.Tensor:
    """Chunk-level K summary ``[B, KVH, n_chunks, Dh]`` — mean over the chunk's groups.

    Chunk summaries are the cheapest routing table and are *always* GPU-resident.
    """
    return group_k.mean(dim=-2)
