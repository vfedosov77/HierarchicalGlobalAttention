"""Dual Chunk Attention (DCA) RoPE helpers for training-free context extrapolation.

DCA (An et al., 2024 — ChunkLlama) remaps RoPE positions so attention never sees relative
distances outside the model's pretraining window.  Qwen3-30B-Instruct-2507 does **not** use
DCA natively — it is trained to 262K with standard absolute RoPE in ``Qwen3MoeRotaryEmbedding``.
DCA is only needed when extrapolating **beyond** ``max_position_embeddings`` (or for legacy
short-context checkpoints).

Reference: https://github.com/HKUNLP/ChunkLlama (``chunkqwen_attn_replace.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, rotate_half


LongContextMode = Literal["native", "dca"]

_NEG = -1.0e4


@dataclass(frozen=True)
class DCAConfig:
    """DCA chunk geometry derived from the model's pretraining context length."""

    pretraining_length: int = 32768
    local_window: Optional[int] = None
    rope_theta: float = 10_000_000.0

    @property
    def dca_chunk_size(self) -> int:
        return self.pretraining_length * 3 // 4

    @property
    def local_window_size(self) -> int:
        return self.local_window if self.local_window is not None else max(1, self.pretraining_length // 16)

    @property
    def chunk_len(self) -> int:
        return self.dca_chunk_size - self.local_window_size


def _has_non_default_rope_scaling(config) -> bool:
    """True only for real extrapolation tables (YaRN, linear, …), not HF's default entry."""
    scaling = getattr(config, "rope_scaling", None)
    if not scaling:
        return False
    if isinstance(scaling, dict):
        kind = scaling.get("type") or scaling.get("rope_type") or scaling.get("name")
        return kind not in (None, "", "default")
    return True


def infer_long_context_mode(
    config,
    *,
    target_context: Optional[int] = None,
    force_dca: bool = False,
) -> LongContextMode:
    """Pick ``native`` (model RoPE) vs ``dca`` (ChunkLlama-style extrapolation).

    Qwen3-30B-A3B-Instruct-2507-FP8: ``max_position_embeddings=262144`` with default RoPE
    → ``native`` for any context ≤ 262K.  DCA is selected only when the requested context
    exceeds the checkpoint's native window (or ``force_dca``).
    """
    if force_dca:
        return "dca"
    max_pos = int(getattr(config, "max_position_embeddings", 32768))
    target = int(target_context or max_pos)
    if target <= max_pos and not _has_non_default_rope_scaling(config):
        return "native"
    return "dca"


def resolve_vram_summary_chunks(
    max_context: int,
    chunk_size: int,
    *,
    margin_factor: float = 2.0,
    minimum: int = 8192,
) -> int:
    """Size the group-summary LRU so routing stays GPU-resident for the full context."""
    needed = int((max_context + chunk_size - 1) // chunk_size * margin_factor)
    return max(minimum, needed)


class DCARopeTables:
    """Cached q / qc / k cos-sin tables for one DCA geometry and head dimension."""

    def __init__(
        self,
        cfg: DCAConfig,
        *,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.head_dim = head_dim
        self.max_seq_len = max(1, max_seq_len)
        self.device = device
        self.dtype = dtype
        self._build()

    def _build(self) -> None:
        cfg = self.cfg
        cl = cfg.chunk_len
        seq_len = self.max_seq_len
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(half, device=self.device, dtype=torch.float32) / half)
        )
        q_t = torch.arange(cl, device=self.device, dtype=torch.float32)
        qc_t = (q_t + cl).clamp(max=cfg.dca_chunk_size).float()
        k_t = (torch.arange(seq_len, device=self.device, dtype=torch.float32) % cl)

        def _emb(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

        self.q_cos, self.q_sin = _emb(q_t)
        self.qc_cos, self.qc_sin = _emb(qc_t)
        self.k_cos, self.k_sin = _emb(k_t)

    def maybe_grow(self, seq_len: int) -> "DCARopeTables":
        if seq_len <= self.max_seq_len:
            return self
        return DCARopeTables(
            self.cfg, head_dim=self.head_dim, max_seq_len=seq_len, device=self.device, dtype=self.dtype,
        )


def _gather_rope(
    cos_table: torch.Tensor, sin_table: torch.Tensor, pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index 1-D tables ``[T, Dh]`` with ``pos`` ``[B, S]`` → ``[B, 1, S, Dh]``."""
    cos_g = cos_table[pos].unsqueeze(1)
    sin_g = sin_table[pos].unsqueeze(1)
    return cos_g, sin_g


def apply_dca_key_rope(
    k_raw: torch.Tensor,
    abs_positions: torch.Tensor,
    tables: DCARopeTables,
) -> torch.Tensor:
    """Keys use cyclic intra-chunk positions (``pos % chunk_len``).  ``k_raw``: ``[B, KVH, S, Dh]``."""
    pos = abs_positions % tables.cfg.chunk_len
    cos, sin = _gather_rope(tables.k_cos, tables.k_sin, pos)
    k_rope, _ = apply_rotary_pos_emb(k_raw, k_raw, cos, sin)
    return k_rope


def apply_dca_query_intra(
    q_raw: torch.Tensor,
    abs_positions: torch.Tensor,
    tables: DCARopeTables,
) -> torch.Tensor:
    """Intra-chunk query RoPE (position within the current DCA chunk)."""
    pos = abs_positions % tables.cfg.chunk_len
    cos, sin = _gather_rope(tables.q_cos, tables.q_sin, pos)
    q_rope, _ = apply_rotary_pos_emb(q_raw, q_raw, cos, sin)
    return q_rope


def _inverse_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos - rotate_half(x) * sin


def _path_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (weighted output, log-sum-exp) for one DCA path; empty mask → zeros."""
    if not mask.any():
        z = q.new_zeros(q.shape)
        lse = torch.full((q.shape[0], q.shape[1], q.shape[2]), -math.inf, device=q.device, dtype=torch.float32)
        return z, lse
    scores = torch.einsum("bhld,bhrd->bhlr", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask, _NEG)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhlr,bhrd->bhld", probs, v.float())
    return out, lse


def _merge_lse(paths: list[Tuple[torch.Tensor, torch.Tensor]], out_dtype: torch.dtype) -> torch.Tensor:
    """Log-sum-exp merge of multiple attention paths (ChunkLlama ``merge_attn_outputs``)."""
    outs, lses = zip(*paths)
    stacked_lse = torch.stack(list(lses), dim=0)                        # [P,B,H,L]
    max_lse = stacked_lse.max(dim=0).values
    weights = torch.exp(stacked_lse - max_lse.unsqueeze(0))
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-9)
    merged = sum(o * w.unsqueeze(-1) for o, w in zip(outs, weights.unbind(0)))
    return merged.to(out_dtype)


def dca_attend(
    q_model_rope: torch.Tensor,
    routed,
    *,
    query_abs_start: int,
    tables: DCARopeTables,
    model_cos: torch.Tensor,
    model_sin: torch.Tensor,
    use_summaries: bool = False,
) -> torch.Tensor:
    """DCA three-path attend (intra / successive / inter) with log-sum-exp merge.

    Falls back to plain ``routed.attend`` when ``token_key_positions`` are unavailable.
    """
    key_pos = getattr(routed, "token_key_positions", None)
    if key_pos is None:
        return routed.attend(q_model_rope, use_summaries=use_summaries)

    B, H, L, Dh = q_model_rope.shape
    cfg = tables.cfg
    cl = cfg.chunk_len
    k, v, mask = routed._segments(use_summaries)
    R = k.shape[2]
    device = q_model_rope.device
    out_dtype = v.dtype

    q_abs = torch.arange(query_abs_start, query_abs_start + L, device=device)
    q_chunk = q_abs // cfg.dca_chunk_size
    k_chunk = key_pos // cfg.dca_chunk_size                          # [R]

    cos_m = model_cos.reshape(B, 1, L, Dh)
    sin_m = model_sin.reshape(B, 1, L, Dh)
    q_unrot = _inverse_rotary(q_model_rope.float(), cos_m, sin_m)

    def _q_path(table_cos: torch.Tensor, table_sin: torch.Tensor, pos_2d: torch.Tensor) -> torch.Tensor:
        cos, sin = _gather_rope(table_cos, table_sin, pos_2d)
        qr, _ = apply_rotary_pos_emb(q_unrot, q_unrot, cos, sin)
        return qr

    paths: list[Tuple[torch.Tensor, torch.Tensor]] = []

    # intra — keys in the same DCA chunk as the query
    same = k_chunk.view(1, 1, 1, R) == q_chunk.view(1, 1, L, 1)
    vis = mask & same
    pos_intra = (q_abs % cl).view(1, L).expand(B, L)
    q_intra = _q_path(tables.q_cos, tables.q_sin, pos_intra)
    paths.append(_path_attention(q_intra, k, v, vis, routed.scale))

    # successive — keys in the immediately preceding DCA chunk
    prev_chunk = q_chunk - 1
    succ = (k_chunk.view(1, 1, 1, R) == prev_chunk.view(1, 1, L, 1)) & (prev_chunk >= 0).view(1, 1, L, 1)
    vis_succ = mask & succ
    pos_succ = (q_abs % cl).view(1, L).expand(B, L)
    q_succ = _q_path(tables.qc_cos, tables.qc_sin, pos_succ)
    paths.append(_path_attention(q_succ, k, v, vis_succ, routed.scale))

    # inter — keys two or more DCA chunks behind
    inter = k_chunk.view(1, 1, 1, R) < prev_chunk.view(1, 1, L, 1)
    vis_inter = mask & inter
    fixed = torch.full((B, L), cl - 1, device=device, dtype=torch.long)
    q_inter = _q_path(tables.qc_cos, tables.qc_sin, fixed)
    paths.append(_path_attention(q_inter, k, v, vis_inter, routed.scale))

    return _merge_lse(paths, out_dtype)