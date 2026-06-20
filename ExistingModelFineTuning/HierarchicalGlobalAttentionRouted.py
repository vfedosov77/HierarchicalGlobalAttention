"""Router-backed hierarchical global attention — the simplified replacement.

This is the slimmed-down rewrite of :class:`HierarchicalGlobalAttention`.  All of the
chunk-routing and KV-cache machinery now lives in :class:`ChunkRouter` /
:class:`KVCacheStore` (package ``KvRouter``); this module only

  1. projects Q/K/V and applies RoPE,
  2. gets (or resets) the router that owns the KV cache, and
  3. asks the router to route + attend the new query block.

Both modes go through the *same* router call:

* **training / teacher-forced eval** — a throwaway all-VRAM cache
  (:class:`VramKVCacheStore`); the block spans many chunks → the router takes its fast
  vectorized chunk-parallel path.  No cache is persisted.
* **generation** — a cache attached to the HF ``past_key_values`` object, on the tier
  chosen by ``cache_location`` (``"vram"`` or ``"ram"``).  Prefill (multi-chunk) seeds the
  store via the vectorized path; per-token decode (single-chunk) takes the incremental path.

The router decides single-chunk vs multi-chunk from the tokens' positions, so this class
has a single, branch-free forward body.  Drop-in compatible with the Qwen-style call
convention (``hidden_states`` / ``position_embeddings`` / ``past_key_value`` / ``use_cache``
/ ``cache_position``) as well as the legacy ``x`` / ``rotary_data`` / ``past_key_values``
names.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:  # works whether ExistingModelFineTuning/ or the repo root is on sys.path
    from KvRouter import (
        ChunkPlacementPolicy,
        ChunkRouter,
        RamKVCacheStore,
        RouterConfig,
        VramKVCacheStore,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (
        ChunkPlacementPolicy,
        ChunkRouter,
        RamKVCacheStore,
        RouterConfig,
        VramKVCacheStore,
    )

try:
    from utilities import RMSNorm  # type: ignore
except Exception:  # pragma: no cover - keeps this file standalone
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = x.float()
            out = self.weight.float() * x_fp32 * torch.rsqrt(
                x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return out.to(dtype=x.dtype)


RotaryData = Tuple[torch.Tensor, torch.Tensor]


class HierarchicalGlobalAttentionRouted(nn.Module):
    """Causal hierarchical attention that delegates all KV-cache work to a ChunkRouter."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 16,
        kv_heads: int = 8,
        dropout: float = 0.0,
        use_bias_q: bool = True,
        use_bias_k: bool = True,
        use_bias_v: bool = True,
        use_bias_o: bool = False,
        causal: bool = True,
        use_global: bool = True,
        chunk_size: int = 64,
        group_size: int = 16,
        topk_chunks: int = 20,
        topk_groups: int = 32,
        return_router_stats: bool = False,
        head_dim: Optional[int] = None,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        theta: float = 1_000_000.0,
        mixed_rope_threshold: float = 0.5,
        mixed_rope_cutoff_pair: Optional[int] = None,
        # -- routed-cache specific knobs --
        cache_location: str = "vram",        # generation cache tier: "vram" | "ram"
        max_chunks_at_once: int = 64,         # vectorized multi-chunk window (VRAM bound)
        # keep_first/keep_last default to 0 so the summary decode path reduces exactly to the
        # reference _decode_forward (prev chunk force-included as a routed *summary*, no
        # token-level hot windows) — matching the vectorized prefill/training path.
        keep_first: int = 0,                  # always-resident leading chunks (sinks)
        keep_last: int = 0,                   # always-resident trailing chunks (local ctx)
        first_token_level: bool = False,
        layer_idx: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()
        assert causal, "This implementation is causal-only."
        assert nhead % kv_heads == 0
        head_dim = d_model // nhead if head_dim is None else head_dim
        assert head_dim % 2 == 0, "RoPE needs an even head_dim."
        assert chunk_size % group_size == 0
        assert cache_location in ("vram", "ram")

        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dropout_p = dropout
        self.use_global = use_global
        self.chunk_size = chunk_size
        self.group_size = group_size
        self.groups_per_chunk = chunk_size // group_size
        self.theta = float(theta)
        self.return_router_stats = return_router_stats
        self.cache_location = cache_location
        self.max_chunks_at_once = max_chunks_at_once
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(d_model, nhead * head_dim, bias=use_bias_q)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_k)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=use_bias_v)
        self.o_proj = nn.Linear(nhead * head_dim, d_model, bias=use_bias_o)

        make_norm = lambda: RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.q_norm = q_norm if q_norm is not None else make_norm()
        self.k_norm = k_norm if k_norm is not None else make_norm()

        self._router_cfg = RouterConfig(
            nhead=nhead, kv_heads=kv_heads, head_dim=head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups,
            theta=theta, mixed_rope_threshold=mixed_rope_threshold,
            mixed_rope_cutoff_pair=mixed_rope_cutoff_pair,
        )
        self._policy = ChunkPlacementPolicy(
            keep_last=keep_last, keep_first=keep_first, first_token_level=first_token_level
        )
        # Reused throwaway router for training / cache-less eval (rebuilt if shape changes).
        self._train_router: Optional[ChunkRouter] = None
        self._train_key: Optional[Tuple[int, torch.dtype, torch.device]] = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        rotary_data: Optional[RotaryData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[RotaryData] = None,
        past_key_value: Any = None,
        cache_position: Optional[torch.Tensor] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, Any]:
        x = hidden_states if x is None else x
        if x is None:
            raise ValueError("HierarchicalGlobalAttentionRouted.forward needs `x`/`hidden_states`.")
        rotary = position_embeddings if position_embeddings is not None else rotary_data
        pkv = past_key_values if past_key_values is not None else past_key_value

        B, S, _ = x.shape
        device, dtype = x.device, x.dtype

        cache_mode = bool(use_cache) if use_cache is not None else (pkv is not None and not self.training)
        start_pos = self._resolve_start_pos(cache_position, pkv, cache_mode)

        cos, sin = self._rotary(rotary, S, start_pos, device, dtype)

        q, k_rope, k_raw, v = self._project(x, cos, sin)

        C = self.chunk_size
        if cache_mode:
            router = self._get_decode_router(pkv, B, dtype, device)
            if self.layer_idx == 0 and start_pos == 0:
                router.reset()
            # Decode hot path: a block that stays inside one active chunk is one decode_block —
            # call it directly (skip the route_query_block segment list) to keep this
            # launch-bound step lean.
            if start_pos // C == (start_pos + S - 1) // C:
                routed = router.decode_block(self.layer_idx, q, k_rope, k_raw, v, start_pos)
                out_heads = routed.attend(q, use_summaries=True)
            else:
                segments = router.route_query_block(
                    self.layer_idx, q, k_rope, k_raw, v, start_pos,
                    cos=cos, sin=sin, populate_store=True,
                    max_chunks_at_once=self.max_chunks_at_once,
                )
                out_heads = self._attend_segments(segments, q)
        else:
            router = self._get_train_router(B, dtype, device)
            segments = router.route_query_block(
                self.layer_idx, q, k_rope, k_raw, v, start_pos=0,
                cos=cos, sin=sin, populate_store=False,
                max_chunks_at_once=self.max_chunks_at_once,
            )
            out_heads = self._attend_segments(segments, q)

        out = self.o_proj(out_heads.transpose(1, 2).reshape(B, S, self.nhead * self.head_dim))
        return out, pkv

    def _attend_segments(self, segments, q: torch.Tensor) -> torch.Tensor:
        """Attend the router's routed-KV segments (summary mode → token + group summaries)."""
        B, H, S, Dh = q.shape
        if len(segments) == 1 and segments[0][1] == 0 and segments[0][2] == S:
            return segments[0][0].attend(q, use_summaries=True)  # common case: no copy
        out = q.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            out[:, :, lo:hi] = routed.attend(q[:, :, lo:hi], use_summaries=True)
        return out

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(B, S, H, Dh)).transpose(1, 2)        # [B,H,S,Dh]
        k_raw = self.k_norm(self.k_proj(x).reshape(B, S, KVH, Dh)).transpose(1, 2)   # [B,KVH,S,Dh] pre-rope
        v = self.v_proj(x).reshape(B, S, KVH, Dh).transpose(1, 2)
        q_rope = self._apply_rotary(q.float(), cos, sin).to(dtype=q.dtype)
        k_rope = self._apply_rotary(k_raw.float(), cos, sin).to(dtype=k_raw.dtype)
        return q_rope, k_rope, k_raw, v

    def _resolve_start_pos(self, cache_position: Optional[torch.Tensor], pkv: Any, cache_mode: bool) -> int:
        if not cache_mode:
            return 0
        if cache_position is not None and cache_position.numel() > 0:
            return int(cache_position.reshape(-1)[0].item())
        if pkv is not None:
            try:
                return int(pkv.get_seq_length(self.layer_idx))
            except TypeError:
                try:
                    return int(pkv.get_seq_length())
                except Exception:
                    return 0
            except Exception:
                return 0
        return 0

    def _rotary(
        self, rotary: Optional[RotaryData], S: int, start_pos: int,
        device: torch.device, dtype: torch.dtype,
    ) -> RotaryData:
        """Return cos/sin shaped ``[1, 1, S, Dh]`` covering positions ``[start_pos, start_pos+S)``."""
        if rotary is not None:
            cos, sin = rotary
            return self._normalize_rotary(cos, device, dtype), self._normalize_rotary(sin, device, dtype)
        pos = torch.arange(start_pos, start_pos + S, device=device)
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        freqs = pos.float().unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).view(1, 1, S, self.head_dim), emb.sin().to(dtype).view(1, 1, S, self.head_dim)

    def _normalize_rotary(self, t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = t.to(device=device, dtype=dtype)
        if t.dim() == 2:        # [S, Dh]
            t = t.view(1, 1, *t.shape)
        elif t.dim() == 3:      # [B, S, Dh] -> rotary is batch-invariant, take any row's layout
            t = t[:1].unsqueeze(1)
        elif t.dim() == 4:      # [B, 1, S, Dh] or [1, 1, S, Dh]
            t = t[:1]
        else:
            raise ValueError(f"rotary tensor must be 2/3/4-D, got {t.dim()}-D")
        return t

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    # -- router construction --------------------------------------------
    def _make_store(self, location: str, B: int, dtype: torch.dtype, device: torch.device):
        kwargs = dict(
            compute_device=device, policy=self._policy, kv_heads=self.kv_heads,
            head_dim=self.head_dim, chunk_size=self.chunk_size,
            groups_per_chunk=self.groups_per_chunk, batch_size=B, dtype=dtype,
        )
        if location == "vram":
            return VramKVCacheStore(**kwargs)
        return RamKVCacheStore(**kwargs)

    def _get_train_router(self, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        key = (B, dtype, device)
        if self._train_router is None or self._train_key != key:
            self._train_router = ChunkRouter(self._router_cfg, self._make_store("vram", B, dtype, device))
            self._train_key = key
        self._train_router.reset()
        return self._train_router

    def _get_decode_router(self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        """One ChunkRouter/store shared by all layers, attached to the ``past_key_values`` object."""
        router = getattr(pkv, "_hga_router", None)
        if router is None:
            store = self._make_store(self.cache_location, B, dtype, device)
            router = ChunkRouter(self._router_cfg, store)
            setattr(pkv, "_hga_router", router)
        return router
