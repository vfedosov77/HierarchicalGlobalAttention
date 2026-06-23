"""Dual Chunk Attention (DCA) for training-free context extrapolation beyond 262K.

The decoder ``rotary_emb`` is patched so ``position_embeddings`` use cyclic positions
(``pos % chunk_len``).  Attention applies those embeddings normally; only the final attend
step over routed KV uses the three-path DCA merge (intra / successive / inter).
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


@dataclass(frozen=True)
class LongContextSettings:
    mode: LongContextMode
    max_context: int
    vram_summary_chunks: int
    native_limit: int
    dca: Optional[DCAConfig] = None

    @property
    def use_dca(self) -> bool:
        return self.mode in ("dca", "hybrid") and self.dca is not None


def _has_non_default_rope_scaling(config) -> bool:
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
    needed = int((max_context + chunk_size - 1) // chunk_size * margin_factor)
    return max(minimum, needed)


def resolve_long_context_settings(
    config,
    *,
    chunk_size: int = 64,
    target_context: Optional[int] = None,
    force_dca: bool = False,
    vram_summary_chunks: Optional[int] = None,
    dca_pretraining_length: Optional[int] = None,
) -> LongContextSettings:
    max_pos = int(getattr(config, "max_position_embeddings", 32768))
    max_context = int(target_context or max_pos)
    mode = infer_long_context_mode(config, target_context=max_context, force_dca=force_dca)
    summary = vram_summary_chunks or resolve_vram_summary_chunks(max_context, chunk_size)
    dca = None
    if mode in ("dca", "hybrid"):
        pretrain = dca_pretraining_length or max_pos
        dca = DCAConfig(
            pretraining_length=pretrain,
            rope_theta=float(getattr(config, "rope_theta", 10_000_000.0)),
        )
    return LongContextSettings(
        mode=mode, max_context=max_context, vram_summary_chunks=summary,
        native_limit=max_pos, dca=dca,
    )


def patch_qwen_rotary_emb(model: nn.Module, lc: LongContextSettings) -> None:
    """Patch ``model.model.rotary_emb`` for cyclic DCA ``position_embeddings``."""
    inner = getattr(model, "model", None)
    if inner is None:
        return
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None:
        return
    restore_qwen_rotary_emb(model)
    if not lc.use_dca or lc.dca is None:
        return

    cfg = lc.dca
    orig_forward = rotary_emb.forward

    def dca_forward(x, position_ids, *args, **kwargs):
        return orig_forward(x, position_ids.remainder(cfg.chunk_len), *args, **kwargs)

    rotary_emb.forward = dca_forward  # type: ignore[method-assign]
    rotary_emb._dca_orig_forward = orig_forward  # type: ignore[attr-defined]
    setattr(model, "_long_context_settings", lc)


def restore_qwen_rotary_emb(model: nn.Module) -> None:
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


@dataclass
class DCARopeEmbeddings:
    cfg: DCAConfig
    q_cos: torch.Tensor
    q_sin: torch.Tensor
    qc_cos: torch.Tensor
    qc_sin: torch.Tensor
    k_cos: torch.Tensor
    k_sin: torch.Tensor


def _rope_rows(rotary_emb: nn.Module, x: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = positions.view(1, -1)
    cos, sin = rotary_emb(x, position_ids=pos)
    return cos[0], sin[0]


def build_dca_embeddings(rotary_emb: nn.Module, x: torch.Tensor, cfg: DCAConfig) -> DCARopeEmbeddings:
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


def get_shared_dca_embeddings(
    holder: Any, rotary_emb: nn.Module, x: torch.Tensor, cfg: DCAConfig,
) -> DCARopeEmbeddings:
    emb = getattr(holder, "_dca_rope_embeddings", None)
    if emb is None:
        emb = build_dca_embeddings(rotary_emb, x, cfg)
        setattr(holder, "_dca_rope_embeddings", emb)
    return emb


def reconstruct_token_key_positions(
    router: Any,
    layer_idx: int,
    *,
    query_abs_start: int,
    q_len: int,
    n_keys: int,
) -> Optional[torch.Tensor]:
    store = router.store
    cfg = router.cfg
    C, gs = cfg.chunk_size, cfg.group_size
    device = store.compute_device

    n_closed = store.num_closed_chunks(layer_idx)
    n = query_abs_start // C
    c0 = query_abs_start % C
    cur_len = c0 + q_len
    if cur_len > C:
        return None

    parts: list[torch.Tensor] = []
    policy = store.policy

    f_lo, f_hi = policy.hot_first_range(n_closed)
    if f_hi > f_lo and policy.first_token_level:
        parts.append(_chunk_range_positions(f_lo, f_hi, C, device))

    l_lo, l_hi = policy.hot_last_range(n_closed)
    if l_hi > l_lo:
        parts.append(_chunk_range_positions(l_lo, l_hi, C, device))

    accounted = sum(p.numel() for p in parts)
    opened_len = max(0, n_keys - accounted - cur_len)
    if opened_len > 0:
        if opened_len % gs != 0:
            return None
        parts.append(torch.zeros(opened_len, dtype=torch.long, device=device))

    parts.append(torch.arange(n * C, n * C + cur_len, device=device))

    positions = torch.cat(parts, dim=0) if parts else None
    if positions is None or positions.numel() != n_keys:
        return None
    return positions


def _chunk_range_positions(lo: int, hi: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    chunks = torch.arange(lo, hi, device=device)
    return (chunks.unsqueeze(1) * chunk_size + torch.arange(chunk_size, device=device)).reshape(-1)


def _gather_rope(cos_table: torch.Tensor, sin_table: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return cos_table[pos], sin_table[pos]


def _inverse_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos - rotate_half(x) * sin


def _path_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Three-path DCA attend (intra / successive / inter) with log-sum-exp merge."""
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