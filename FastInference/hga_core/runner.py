"""Engine-neutral HGA layer runner.

``HgaLayerRunner`` orchestrates one attention layer's HGA flow on **plain
tensors** — no ``DynamicCache``, no ``ForwardBatch``, no Qwen monkey-patching.
Both the unit tests (on this box) and the SGLang backend drive it identically:

* :meth:`prefill_chunk` — feed up to one ``chunk_size`` block of new tokens,
  build summaries for any chunk that closes, store token K/V to the tiered cache,
  and return the block's attention output.
* :meth:`decode_step`   — feed ``L`` new tokens inside the current active chunk
  (``L == 1`` is the hot decode path), route (chunk top-k -> group top-k),
  assemble the routed KV, and run the fused decode attention kernel.

v0 scope: batch = 1, DCA disabled (native absolute RoPE), RAM + VRAM tiers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .config import HgaConfig
from .route_metadata import RouteMetadata
from .cache.manager import HgaCacheManager
from .kernels.decode_attention import fused_decode_attention
from .kernels.route_topk import chunk_topk, group_topk


class HgaLayerRunner:
    def __init__(self, cfg: HgaConfig, device: torch.device,
                 manager: Optional[HgaCacheManager] = None):
        self.cfg = cfg
        self.device = device
        self.manager = manager or HgaCacheManager(cfg, device)
        # active (partial) chunk accumulators per layer (KVH-headed)
        self._act_krope: Dict[int, torch.Tensor] = {}
        self._act_kraw: Dict[int, torch.Tensor] = {}
        self._act_v: Dict[int, torch.Tensor] = {}
        self._act_start: Dict[int, int] = {}    # absolute position of active chunk's first token

    def reset(self) -> None:
        self.manager.reset()
        self._act_krope.clear(); self._act_kraw.clear()
        self._act_v.clear(); self._act_start.clear()

    # ------------------------------------------------------------------
    # window policy
    # ------------------------------------------------------------------
    def _windows(self, n_closed: int) -> Tuple[int, int, int, int]:
        cfg = self.cfg
        first_lo, first_hi = 0, min(cfg.keep_first, n_closed)
        last_lo = min(max(cfg.keep_first, n_closed - cfg.keep_last), n_closed)
        last_hi = n_closed
        return first_lo, first_hi, last_lo, last_hi

    # ------------------------------------------------------------------
    # active-chunk accumulation + closing
    # ------------------------------------------------------------------
    def _append_active(self, layer, k_rope, k_raw, v, start_pos):
        if layer not in self._act_krope or self._act_krope[layer] is None:
            self._act_krope[layer] = k_rope
            self._act_kraw[layer] = k_raw
            self._act_v[layer] = v
            self._act_start[layer] = (start_pos // self.cfg.chunk_size) * self.cfg.chunk_size
        else:
            self._act_krope[layer] = torch.cat([self._act_krope[layer], k_rope], dim=-2)
            self._act_kraw[layer] = torch.cat([self._act_kraw[layer], k_raw], dim=-2)
            self._act_v[layer] = torch.cat([self._act_v[layer], v], dim=-2)

    def _maybe_close_active(self, layer: int) -> None:
        C = self.cfg.chunk_size
        kp = self._act_krope.get(layer)
        if kp is None or kp.shape[-2] < C:
            return
        # close exactly one chunk (callers feed <= C at a time)
        start_chunk = self._act_start[layer] // C
        KVH = kp.shape[0]
        Dh = kp.shape[-1]
        kr = self._act_kraw[layer][..., :C, :].reshape(KVH, 1, C, Dh)
        kpc = kp[..., :C, :].reshape(KVH, 1, C, Dh)
        vc = self._act_v[layer][..., :C, :].reshape(KVH, 1, C, Dh)
        self.manager.append_closed_chunk(layer, kr, kpc, vc, start_chunk)
        # drop the closed tokens; keep remainder (none, since callers align)
        rem = kp.shape[-2] - C
        if rem > 0:
            self._act_krope[layer] = self._act_krope[layer][..., C:, :]
            self._act_kraw[layer] = self._act_kraw[layer][..., C:, :]
            self._act_v[layer] = self._act_v[layer][..., C:, :]
            self._act_start[layer] += C
        else:
            self._act_krope[layer] = None
            self._act_kraw[layer] = None
            self._act_v[layer] = None
            self._act_start[layer] += C

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _route(self, layer: int, q: torch.Tensor, n_closed: int) -> RouteMetadata:
        """q: [B,H,L,Dh] (B==1). Build a RouteMetadata from the GPU summaries."""
        cfg = self.cfg
        B, H, L, Dh = q.shape
        device = q.device
        first_lo, first_hi, last_lo, last_hi = self._windows(n_closed)
        mid_lo, mid_hi = first_hi, last_lo
        n_mid = max(0, mid_hi - mid_lo)

        empty = torch.empty(B, H, 0, dtype=torch.long, device=device)
        empty_vis = torch.empty(B, H, L, 0, dtype=torch.bool, device=device)
        if n_mid == 0 or cfg.topk_chunks <= 0:
            return RouteMetadata(
                batch_size=B, num_q_heads=H, num_kv_heads=cfg.num_kv_heads, head_dim=Dh,
                n_closed=n_closed, mid_chunk_idx=empty, mid_vis=empty_vis,
                open_chunk_idx=empty.clone(), open_group_idx=empty.clone(),
                open_vis=empty_vis.clone(), first_lo=first_lo, first_hi=first_hi,
                last_lo=last_lo, last_hi=last_hi, active_start=self._act_start.get(layer, 0),
                use_summaries=True,
            )

        chunk_k_full = self.manager.summaries.chunk_summaries(layer)   # [1,KVH,n,Dh]
        ck = chunk_k_full[:, :, mid_lo:mid_hi]                         # [1,KVH,n_mid,Dh]
        if cfg.gqa_rep > 1:
            ck = ck.repeat_interleave(cfg.gqa_rep, dim=1)
        ck = ck.to(q.dtype)
        prev = n_closed - 1
        force_prev = (prev - mid_lo) if (mid_lo <= prev < mid_hi) else None
        mid_rel, mid_vis, _ = chunk_topk(q, ck, cfg.topk_chunks, cfg.scale, force_prev)
        mid_idx = mid_rel + mid_lo

        gk, _gv = self.manager.summaries.gather_group_summaries(layer, mid_idx)   # [B,H,Kc,M,Dh]
        gk = gk.to(q.dtype)
        open_chunk, open_grp, open_vis = group_topk(
            q, gk, mid_idx, mid_vis, cfg.topk_groups,
            cfg.effective_topk_groups_request, cfg.scale, cfg.groups_per_chunk,
        )
        return RouteMetadata(
            batch_size=B, num_q_heads=H, num_kv_heads=cfg.num_kv_heads, head_dim=Dh,
            n_closed=n_closed, mid_chunk_idx=mid_idx, mid_vis=mid_vis,
            open_chunk_idx=open_chunk, open_group_idx=open_grp, open_vis=open_vis,
            first_lo=first_lo, first_hi=first_hi, last_lo=last_lo, last_hi=last_hi,
            active_start=self._act_start.get(layer, 0),
            use_summaries=True,
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_step(
        self,
        layer: int,
        q: torch.Tensor,        # [H, L, Dh]   rope-applied (B=1)
        k_rope: torch.Tensor,   # [KVH, L, Dh] rope-applied
        k_raw: torch.Tensor,    # [KVH, L, Dh] pre-rope
        v: torch.Tensor,        # [KVH, L, Dh]
        start_pos: int,
        use_summaries: bool = True,
    ) -> torch.Tensor:
        """Route + assemble + fused attention for ``L`` new tokens. Returns ``[H, L, Dh]``."""
        cfg = self.cfg
        self._append_active(layer, k_rope, k_raw, v, start_pos)
        n_closed = self.manager.num_closed_chunks(layer)
        active_off = start_pos - self._act_start[layer]
        qb = q.unsqueeze(0)                                # [1,H,L,Dh]
        route = self._route(layer, qb, n_closed)
        route.use_summaries = use_summaries
        act_k = self._act_krope[layer]                     # [KVH, L_act, Dh]
        act_v = self._act_v[layer]
        k, vv, mask = self.manager.assemble_routed_kv(
            layer, route, act_k, act_v, active_off,
        )
        out = fused_decode_attention(qb, k, vv, mask, cfg.scale)   # [1,H,L,Dh]
        # close chunk if the active accumulator just filled
        self._maybe_close_active(layer)
        return out[0]

    @torch.no_grad()
    def prefill(
        self,
        layer: int,
        q: torch.Tensor,        # [H, S, Dh] rope-applied
        k_rope: torch.Tensor,   # [KVH, S, Dh]
        k_raw: torch.Tensor,    # [KVH, S, Dh]
        v: torch.Tensor,        # [KVH, S, Dh]
        start_pos: int = 0,
        use_summaries: bool = True,
    ) -> torch.Tensor:
        """Prefill ``S`` tokens chunk-by-chunk, populating summaries + host KV.

        Returns the full ``[H, S, Dh]`` attention output.  Feeds the runner in
        ``chunk_size``-aligned sub-blocks so each closing chunk gets summarised.
        """
        C = self.cfg.chunk_size
        S = q.shape[-2]
        outs = []
        pos = start_pos
        i = 0
        while i < S:
            # take up to the next chunk boundary
            room = C - (pos % C)
            step = min(room, S - i)
            outs.append(self.decode_step(
                layer,
                q[..., i:i + step, :], k_rope[..., i:i + step, :],
                k_raw[..., i:i + step, :], v[..., i:i + step, :], pos,
                use_summaries=use_summaries,
            ))
            i += step
            pos += step
        return torch.cat(outs, dim=-2)
