"""Dual Chunk Attention (DCA) RoPE helpers for training-free context extrapolation.

DCA (An et al., 2024 — ChunkLlama) remaps RoPE positions so attention never sees relative
distances outside the model's pretraining window.  Qwen3-30B-Instruct-2507 does **not** use
DCA natively — it is trained to 262K with standard absolute RoPE in ``Qwen3MoeRotaryEmbedding``.
DCA is only needed when extrapolating **beyond** ``max_position_embeddings``.

When DCA is active, the decoder's ``rotary_emb`` is patched so ``position_embeddings`` passed
to every layer already use cyclic DCA positions (``pos % chunk_len``).  Attention applies
those embeddings directly; only the three-path attend step needs extra qc tables (built once
via the same ``rotary_emb``, cached on the KV holder).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, rotate_half


LongContextMode = Literal["native", "dca", "hybrid"]

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
    """Pick strategy mode for the requested context ceiling.

    * ``native`` — target ≤ native window (262K for Qwen3-2507).
    * ``hybrid`` — target > native window: DCA ``position_embeddings`` for the full session
      (KV cache must stay on one RoPE scheme).
    """
    if force_dca:
        return "dca"
    max_pos = int(getattr(config, "max_position_embeddings", 32768))
    target = int(target_context or max_pos)
    if target <= max_pos and not _has_non_default_rope_scaling(config):
        return "native"
    return "hybrid"


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


@dataclass
class DCARopeEmbeddings:
    """Cached q / qc / k cos-sin rows — ``chunk_len`` cyclic positions from ``rotary_emb``."""

    cfg: DCAConfig
    q_cos: torch.Tensor
    q_sin: torch.Tensor
    qc_cos: torch.Tensor
    qc_sin: torch.Tensor
    k_cos: torch.Tensor
    k_sin: torch.Tensor


def _rope_rows(
    rotary_emb: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``positions`` ``[N]`` → cos/sin ``[N, Dh]`` via the model's ``Qwen3MoeRotaryEmbedding``."""
    pos = positions.view(1, -1)
    cos, sin = rotary_emb(x, position_ids=pos)
    return cos[0], sin[0]


def build_dca_embeddings(
    rotary_emb: nn.Module,
    x: torch.Tensor,
    cfg: DCAConfig,
) -> DCARopeEmbeddings:
    """Build cyclic DCA tables using the checkpoint's shared ``rotary_emb``."""
    cl = cfg.chunk_len
    dev = x.device
    q_pos = torch.arange(cl, device=dev, dtype=torch.long)
    qc_pos = (q_pos + cl).clamp(max=cfg.dca_chunk_size)
    q_cos, q_sin = _rope_rows(rotary_emb, x, q_pos)
    qc_cos, qc_sin = _rope_rows(rotary_emb, x, qc_pos)
    k_cos, k_sin = _rope_rows(rotary_emb, x, q_pos)
    return DCARopeEmbeddings(
        cfg=cfg, q_cos=q_cos, q_sin=q_sin, qc_cos=qc_cos, qc_sin=qc_sin, k_cos=k_cos, k_sin=k_sin,
    )


def patch_qwen_rotary_emb(model: nn.Module, lc_settings: Any) -> None:
    """Patch ``model.model.rotary_emb`` so ``position_embeddings`` use DCA cyclic positions."""
    inner = getattr(model, "model", None)
    if inner is None:
        return
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None:
        return
    restore_qwen_rotary_emb(model)
    if lc_settings.mode not in ("dca", "hybrid") or lc_settings.dca is None:
        return

    cfg: DCAConfig = lc_settings.dca
    orig_forward = rotary_emb.forward

    def dca_forward(x, position_ids, *args, **kwargs):
        dca_pos = position_ids.remainder(cfg.chunk_len)
        return orig_forward(x, dca_pos, *args, **kwargs)

    rotary_emb.forward = dca_forward  # type: ignore[method-assign]
    rotary_emb._dca_orig_forward = orig_forward  # type: ignore[attr-defined]
    setattr(model, "_long_context_settings", lc_settings)


def restore_qwen_rotary_emb(model: nn.Module) -> None:
    """Restore the checkpoint's unmodified ``rotary_emb.forward``."""
    inner = getattr(model, "model", None)
    if inner is None:
        return
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None:
        return
    orig = getattr(rotary_emb, "_dca_orig_forward", None)
    if orig is not None:
        rotary_emb.forward = orig  # type: ignore[method-assign]
        del rotary_emb._dca_orig_forward  # type: ignore[attr-defined]


def get_shared_dca_embeddings(
    holder: Any,
    rotary_emb: nn.Module,
    x: torch.Tensor,
    cfg: DCAConfig,
) -> DCARopeEmbeddings:
    """One embedding set per cache holder (shared by all attention layers)."""
    emb = getattr(holder, "_dca_rope_embeddings", None)
    if emb is None:
        emb = build_dca_embeddings(rotary_emb, x, cfg)
        setattr(holder, "_dca_rope_embeddings", emb)
    return emb


def _gather_rope(
    cos_table: torch.Tensor, sin_table: torch.Tensor, pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index tables ``[cl, Dh]`` with ``pos`` ``[B, S]`` → ``[B, S, Dh]``."""
    return cos_table[pos], sin_table[pos]


def apply_dca_key_rope(
    k_raw: torch.Tensor,
    abs_positions: torch.Tensor,
    emb: DCARopeEmbeddings,
) -> torch.Tensor:
    """Keys use cyclic intra-chunk positions (``pos % chunk_len``).  ``k_raw``: ``[B, KVH, S, Dh]``."""
    pos = abs_positions % emb.cfg.chunk_len
    cos, sin = _gather_rope(emb.k_cos, emb.k_sin, pos)
    k_rope, _ = apply_rotary_pos_emb(k_raw, k_raw, cos, sin)
    return k_rope


def apply_dca_query_intra(
    q_raw: torch.Tensor,
    abs_positions: torch.Tensor,
    emb: DCARopeEmbeddings,
) -> torch.Tensor:
    """Intra-chunk query RoPE (position within the current DCA chunk)."""
    pos = abs_positions % emb.cfg.chunk_len
    cos, sin = _gather_rope(emb.q_cos, emb.q_sin, pos)
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
    stacked_lse = torch.stack(list(lses), dim=0)
    max_lse = stacked_lse.max(dim=0).values
    weights = torch.exp(stacked_lse - max_lse.unsqueeze(0))
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-9)
    merged = sum(o * w.unsqueeze(-1) for o, w in zip(outs, weights.unbind(0)))
    return merged.to(out_dtype)


def dca_attend(
    q_intra_rope: torch.Tensor,
    routed,
    *,
    query_abs_start: int,
    emb: DCARopeEmbeddings,
    use_summaries: bool = False,
) -> torch.Tensor:
    """DCA three-path attend (intra / successive / inter) with log-sum-exp merge."""
    key_pos = getattr(routed, "token_key_positions", None)
    if key_pos is None:
        return routed.attend(q_intra_rope, use_summaries=use_summaries)

    B, H, L, Dh = q_intra_rope.shape
    cfg = emb.cfg
    cl = cfg.chunk_len
    k, v, mask = routed._segments(use_summaries)
    R = k.shape[2]
    device = q_intra_rope.device
    out_dtype = v.dtype

    q_abs = torch.arange(query_abs_start, query_abs_start + L, device=device)
    q_chunk = q_abs // cfg.dca_chunk_size
    k_chunk = key_pos // cfg.dca_chunk_size

    pos_intra = (q_abs % cl).view(1, L).expand(B, L)
    cos_i, sin_i = _gather_rope(emb.q_cos, emb.q_sin, pos_intra)
    q_unrot = _inverse_rotary(q_intra_rope.float(), cos_i.unsqueeze(1), sin_i.unsqueeze(1))

    def _q_path(table_cos: torch.Tensor, table_sin: torch.Tensor, pos_2d: torch.Tensor) -> torch.Tensor:
        cos, sin = _gather_rope(table_cos, table_sin, pos_2d)
        qr, _ = apply_rotary_pos_emb(q_unrot, q_unrot, cos, sin)
        return qr

    paths: list[Tuple[torch.Tensor, torch.Tensor]] = []

    same = k_chunk.view(1, 1, 1, R) == q_chunk.view(1, 1, L, 1)
    vis = mask & same
    q_intra = _q_path(emb.q_cos, emb.q_sin, pos_intra)
    paths.append(_path_attention(q_intra, k, v, vis, routed.scale))

    prev_chunk = q_chunk - 1
    succ = (k_chunk.view(1, 1, 1, R) == prev_chunk.view(1, 1, L, 1)) & (prev_chunk >= 0).view(1, 1, L, 1)
    vis_succ = mask & succ
    pos_succ = (q_abs % cl).view(1, L).expand(B, L)
    q_succ = _q_path(emb.qc_cos, emb.qc_sin, pos_succ)
    paths.append(_path_attention(q_succ, k, v, vis_succ, routed.scale))

    inter = k_chunk.view(1, 1, 1, R) < prev_chunk.view(1, 1, L, 1)
    vis_inter = mask & inter
    fixed = torch.full((B, L), cl - 1, device=device, dtype=torch.long)
    q_inter = _q_path(emb.qc_cos, emb.qc_sin, fixed)
    paths.append(_path_attention(q_inter, k, v, vis_inter, routed.scale))

    return _merge_lse(paths, out_dtype)