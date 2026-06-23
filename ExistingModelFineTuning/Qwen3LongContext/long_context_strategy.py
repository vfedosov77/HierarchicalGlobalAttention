"""Pluggable long-context strategies for hierarchical Qwen attention.

The KvRouter is responsible only for *which* KV to load; how queries and keys are
position-encoded for contexts beyond the model's comfort zone is handled here.

When DCA is enabled (``dca`` or ``hybrid`` mode), the decoder's ``rotary_emb`` is patched
so ``position_embeddings`` already carry cyclic DCA RoPE.  Strategies apply those embeddings
directly to Q/K; the three-path DCA attend step uses qc tables from the same ``rotary_emb``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from ExistingModelFineTuning.Qwen3LongContext.dca_rope import (
    DCAConfig,
    DCARopeEmbeddings,
    dca_attend,
    get_shared_dca_embeddings,
    infer_long_context_mode,
    resolve_vram_summary_chunks,
)

LongContextMode = Literal["native", "dca", "hybrid"]


@dataclass(frozen=True)
class LongContextSettings:
    """Resolved long-context configuration for a loaded model."""

    mode: LongContextMode
    max_context: int
    vram_summary_chunks: int
    native_limit: int
    dca: Optional[DCAConfig] = None


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


class LongContextStrategy(ABC):
    """How this attention layer position-encodes Q/K and runs the final attend step."""

    @abstractmethod
    def prepare_qk(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        abs_positions: torch.Tensor,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        *,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE-applied ``(q, k)`` ready for the router."""

    @abstractmethod
    def attend(
        self,
        q_rope: torch.Tensor,
        routed: Any,
        *,
        query_abs_start: int,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        router: Any = None,
        layer_idx: int = 0,
        use_summaries: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> torch.Tensor:
        """Attention over a routed KV segment.  ``q_rope``: ``[B, H, L, Dh]``."""

    def validate_position(self, abs_pos: int, max_context: int) -> None:
        if abs_pos >= max_context:
            raise ValueError(
                f"Sequence position {abs_pos} exceeds configured max_context={max_context}. "
                "Raise target_context or switch to a DCA strategy for extrapolation."
            )


class NativeLongContextStrategy(LongContextStrategy):
    """Standard decoder ``position_embeddings`` (absolute RoPE, up to native window)."""

    def __init__(self, max_context: int) -> None:
        self.max_context = max_context

    def prepare_qk(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        abs_positions: torch.Tensor,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        *,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.validate_position(int(abs_positions.reshape(-1)[-1].item()), self.max_context)
        return apply_rotary_pos_emb(q_raw, k_raw, model_cos, model_sin)

    def attend(
        self,
        q_rope: torch.Tensor,
        routed: Any,
        *,
        query_abs_start: int,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        router: Any = None,
        layer_idx: int = 0,
        use_summaries: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> torch.Tensor:
        return routed.attend(q_rope, use_summaries=use_summaries)


class DCALongContextStrategy(LongContextStrategy):
    """DCA via decoder ``position_embeddings`` + three-path attend for routed KV."""

    def __init__(self, cfg: DCAConfig, max_context: int) -> None:
        self.cfg = cfg
        self.max_context = max_context

    def _embeddings(
        self,
        rotary_emb: nn.Module,
        hidden_states: torch.Tensor,
        holder: Any,
    ) -> DCARopeEmbeddings:
        cache_holder = holder if holder is not None else self
        return get_shared_dca_embeddings(cache_holder, rotary_emb, hidden_states, self.cfg)

    def prepare_qk(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        abs_positions: torch.Tensor,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        *,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply DCA ``position_embeddings`` from the patched decoder ``rotary_emb``."""
        self.validate_position(int(abs_positions.reshape(-1)[-1].item()), self.max_context)
        return apply_rotary_pos_emb(q_raw, k_raw, model_cos, model_sin)

    def attend(
        self,
        q_rope: torch.Tensor,
        routed: Any,
        *,
        query_abs_start: int,
        model_cos: torch.Tensor,
        model_sin: torch.Tensor,
        router: Any = None,
        layer_idx: int = 0,
        use_summaries: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        hidden_states: Optional[torch.Tensor] = None,
        holder: Any = None,
    ) -> torch.Tensor:
        if rotary_emb is None or hidden_states is None:
            raise ValueError("DCA attend requires rotary_emb and hidden_states for qc paths")
        emb = self._embeddings(rotary_emb, hidden_states, holder)
        routed = self._attach_key_positions(
            routed, router=router, layer_idx=layer_idx, query_abs_start=query_abs_start, q_len=q_rope.shape[2],
        )
        return dca_attend(
            q_rope, routed,
            query_abs_start=query_abs_start,
            emb=emb,
            use_summaries=use_summaries,
        )

    def _attach_key_positions(
        self,
        routed: Any,
        *,
        router: Any,
        layer_idx: int,
        query_abs_start: int,
        q_len: int,
    ) -> Any:
        if getattr(routed, "token_key_positions", None) is not None:
            return routed
        if router is None:
            return routed
        positions = reconstruct_token_key_positions(
            router, layer_idx, query_abs_start=query_abs_start, q_len=q_len, n_keys=routed.token_k.shape[2],
        )
        if positions is None:
            return routed
        routed.token_key_positions = positions
        return routed


def reconstruct_token_key_positions(
    router: Any,
    layer_idx: int,
    *,
    query_abs_start: int,
    q_len: int,
    n_keys: int,
) -> Optional[torch.Tensor]:
    """Rebuild absolute key positions from the router store layout (no router code changes)."""
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


def make_strategy(settings: LongContextSettings) -> LongContextStrategy:
    if settings.mode in ("dca", "hybrid") and settings.dca is not None:
        return DCALongContextStrategy(settings.dca, settings.max_context)
    return NativeLongContextStrategy(settings.max_context)