"""Router-backed sparse attention for pretrained Qwen3 (MoE) models.

This swaps each ``Qwen3MoeAttention`` for a drop-in module that keeps the *original* (here
FP8-quantized) Q/K/V/O projections and q/k RMSNorms **by reference** and only changes *what
each query attends to*.  All KV-cache bookkeeping is delegated to the ``KvRouter`` package
(``ChunkRouter`` + ``KVCacheStore``); the cache stays in VRAM for this PoC
(``VramKVCacheStore``) and can later be moved to RAM/NVMe without touching this file.

Two assembly modes, sharing the exact same projections / RoPE / router config so they form an
apples-to-apples comparison (see ``test_qwen30b_routed.py``):

* ``"exact"`` — the new design.  Routing only *selects* chunks; attention is then computed over
  the **actual token K/V** of the selected chunks (first ``keep_first`` sinks, last ``keep_last``
  local chunks, the top-``topk_chunks`` routed middle chunks, and the active partial chunk),
  exactly like ordinary transformer attention.  No summary vector ever enters the attention math,
  so a pretrained model that never learned the summaries is not corrupted.
* ``"summary"`` — what the existing ``HierarchicalGlobalAttentionRouted`` does: the stock
  ``ChunkRouter.decode_block`` exposes routed chunks as *group-summary* K/V (a learned-from-scratch
  approximation).  Kept here only so we can measure how much that approximation costs on a model
  that was not trained for it.

MInference's static **A-shape** pattern (first ``n_init`` sink tokens + last ``n_local`` local
tokens) maps onto ``keep_first`` / ``keep_last`` chunks (chunk_size 64): n_init=128 → keep_first=2,
n_local=128 → keep_last=2 for the PoC test.  The top-k routed middle is an additional recall path
on top of A-shape.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

# --- KvRouter package import (works from repo root or ExistingModelFineTuning/) -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_EFT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EFT)
for _p in (_ROOT, _EFT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from KvRouter import ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore  # type: ignore
    from KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore
    from KvRouter.chunk_router import RoutedKV  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramKVCacheStore, RamKVCacheStore,
    )
    from ExistingModelFineTuning.KvRouter.cache_store import ChunkPlacementPolicy  # type: ignore
    from ExistingModelFineTuning.KvRouter.chunk_router import RoutedKV  # type: ignore

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

_NEG = -1.0e4  # finite mask fill (fp16/bf16-safe)


# =================================================================================================
# Exact router: selection via summaries, attention via real token KV
# =================================================================================================
class ExactChunkRouter(ChunkRouter):
    """``ChunkRouter`` whose ``decode_block_exact`` attends over real token KV (no summaries).

    Reuses the parent's validated bookkeeping (``_append_active``, ``_close_active_chunk``, the
    mixed-RoPE chunk summaries used *only* for routing, the store interaction) and replaces the
    attention-assembly step.
    """

    def decode_block_exact(
        self,
        layer: int,
        q: torch.Tensor,        # [B, H, L, Dh]   rope-applied, head-expanded
        k_rope: torch.Tensor,   # [B, KVH, L, Dh] rope-applied
        k_raw: torch.Tensor,    # [B, KVH, L, Dh] pre-rope (for chunk summary)
        v: torch.Tensor,        # [B, KVH, L, Dh]
        start_pos: int,
    ) -> RoutedKV:
        """Route + assemble exact token KV.  Routing is **per KV-head** (the ``rep`` query heads
        of a GQA group share KV), so each step touches at most ``keep_first+keep_last+KVH·topk``
        distinct chunks — a small, stable working set that the store's VRAM cache keeps resident,
        and the cold-tier gather is KV-head-granular (no per-query-head copy blow-up)."""
        cfg = self.cfg
        C, KVH, rep = cfg.chunk_size, cfg.kv_heads, cfg.rep
        B, H, L, Dh = q.shape
        device = q.device
        n = start_pos // C
        c0 = start_pos % C
        assert c0 + L <= C, "decode_block_exact must stay within one chunk; feed chunk-by-chunk."
        n_closed = self.store.num_closed_chunks(layer)
        assert n_closed == n, f"active chunk {n} != closed {n_closed}; out-of-order block"

        # -- accumulate the active (partial) chunk's KV --
        self._append_active(layer, k_rope, k_raw, v, start_pos)
        act_krope = self._active_krope[layer]   # [B,KVH,cur_len,Dh]
        act_v = self._active_v[layer]
        cur_len = act_krope.shape[2]            # == c0 + L
        q_local = torch.arange(c0, c0 + L, device=device)

        keep_first = self.store.policy.keep_first
        keep_last = self.store.policy.keep_last
        f_hi = min(keep_first, n_closed)
        l_lo = min(max(f_hi, n_closed - keep_last), n_closed) if keep_last > 0 else n_closed
        fixed_ids = list(range(0, f_hi)) + list(range(l_lo, n_closed))   # sinks + local window
        mid_lo, mid_hi = f_hi, l_lo
        n_mid = max(0, mid_hi - mid_lo)

        # -- routed middle, selected per KV-head (pool the rep query heads' scores) --
        parts: List[torch.Tensor] = []
        if fixed_ids:
            parts.append(torch.tensor(fixed_ids, device=device).view(1, 1, -1).expand(B, KVH, -1))
        if n_mid > 0 and cfg.topk_chunks > 0:
            ck = self.store.chunk_summaries(layer)[:, :, mid_lo:mid_hi]      # [B,KVH,n_mid,Dh]
            q_g = q.reshape(B, KVH, rep, L, Dh)
            sc = torch.einsum("bgrld,bgnd->bgrln", q_g, ck) * cfg.scale       # [B,KVH,rep,L,n_mid]
            pooled = sc.amax(dim=2).amax(dim=2)                              # [B,KVH,n_mid]
            Kc = min(cfg.topk_chunks, n_mid)
            _, mid_rel = torch.topk(pooled, Kc, dim=-1, sorted=False)        # [B,KVH,Kc]
            parts.append(mid_rel + mid_lo)

        seg_k: List[torch.Tensor] = []
        seg_v: List[torch.Tensor] = []
        seg_mask: List[torch.Tensor] = []

        # -- selected closed chunks: gather KV-head-granular (cached), expand to query heads --
        if parts:
            sel = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]  # [B,KVH,Ksel]
            Ksel = sel.shape[-1]
            k_sel, v_sel = self.store.gather_chunk_tokens_kvh(layer, sel)    # [B,KVH,Ksel,C,Dh]
            k_keys = k_sel.reshape(B, KVH, Ksel * C, Dh).repeat_interleave(rep, dim=1)  # [B,H,*,Dh]
            v_keys = v_sel.reshape(B, KVH, Ksel * C, Dh).repeat_interleave(rep, dim=1)
            seg_k.append(k_keys); seg_v.append(v_keys)
            seg_mask.append(torch.ones(B, H, L, Ksel * C, dtype=torch.bool, device=device))

        # -- active (partial) chunk: exact tokens, causal within the chunk --
        tok_pos = torch.arange(cur_len, device=device)
        causal = (tok_pos.view(1, 1, 1, cur_len) <= q_local.view(1, 1, L, 1)).expand(B, H, L, cur_len)
        seg_k.append(self._rep_heads(act_krope))
        seg_v.append(self._rep_heads(act_v))
        seg_mask.append(causal)

        routed = RoutedKV(
            token_k=torch.cat(seg_k, dim=2),
            token_v=torch.cat(seg_v, dim=2),
            token_mask=torch.cat(seg_mask, dim=3),
            scale=cfg.scale,
        )

        if cur_len == C:
            self._close_active_chunk(layer, n)
        return routed


# =================================================================================================
# Drop-in attention module
# =================================================================================================
class QwenRoutedAttention(nn.Module):
    """Replacement for ``Qwen3MoeAttention`` that routes through a shared ``ChunkRouter``.

    Adds no parameters: it reuses ``orig``'s projections and norms by reference (FP8-safe).
    """

    def __init__(
        self,
        orig: nn.Module,
        config: Any,
        *,
        mode: str = "exact",
        chunk_size: int = 64,
        group_size: int = 16,
        keep_first: int = 2,
        keep_last: int = 2,
        topk_chunks: int = 8,
        topk_groups: int = 32,
        cache_location: str = "vram",
        vram_cache_chunks: int = 256,
    ) -> None:
        super().__init__()
        assert mode in ("exact", "summary")
        self.orig = orig            # keeps original projections/norms as a child (shared weights)
        self.mode = mode
        self.layer_idx = int(getattr(orig, "layer_idx", 0))

        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.num_heads))
        self.chunk_size = chunk_size
        self.cache_location = cache_location
        self.vram_cache_chunks = vram_cache_chunks

        self._cfg = RouterConfig(
            nhead=self.num_heads, kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups,
            theta=float(getattr(config, "rope_theta", 1_000_000.0)),
        )
        # first_token_level only matters for "summary" mode (whether the first window is resident
        # at token granularity).  Exact mode gathers token KV directly and ignores it.
        self._policy = ChunkPlacementPolicy(
            keep_last=keep_last, keep_first=keep_first, first_token_level=(mode == "exact"),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        cache_position: Optional[torch.Tensor] = None,
        **kw: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        o = self.orig
        B, S, _ = hidden_states.shape
        H, KVH, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        q = o.q_norm(o.q_proj(hidden_states).view(B, S, H, Dh)).transpose(1, 2)    # [B,H,S,Dh]
        k_raw = o.k_norm(o.k_proj(hidden_states).view(B, S, KVH, Dh)).transpose(1, 2)  # pre-rope
        v = o.v_proj(hidden_states).view(B, S, KVH, Dh).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k_raw, cos, sin)   # HF rotate_half convention

        # The decoder forwards ``position_ids`` (not ``cache_position``) to attention; both
        # encode the absolute start of this block, which is what the streaming router needs.
        start_pos = 0
        if cache_position is not None and cache_position.numel() > 0:
            start_pos = int(cache_position.reshape(-1)[0].item())
        else:
            pos_ids = kw.get("position_ids", None)
            if pos_ids is not None and pos_ids.numel() > 0:
                start_pos = int(pos_ids.reshape(-1)[0].item())

        router = self._get_router(past_key_values, B, hidden_states.dtype, hidden_states.device)
        if self.layer_idx == 0 and start_pos == 0:
            router.reset()

        out_heads = self._route(router, q_rope, k_rope, k_raw, v, start_pos)
        out = o.o_proj(out_heads.transpose(1, 2).reshape(B, S, H * Dh))
        return out, None

    # ------------------------------------------------------------------
    def _route(
        self,
        router: ChunkRouter,
        q: torch.Tensor,
        k_rope: torch.Tensor,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        """Stream the query block chunk-by-chunk through the router (prefill and decode alike)."""
        C = self.chunk_size
        S = q.shape[2]
        outs: List[torch.Tensor] = []
        p, done = start_pos, 0
        exact = self.mode == "exact"
        while done < S:
            take = min(C - (p % C), S - done)
            sl = slice(done, done + take)
            if exact:
                routed = router.decode_block_exact(
                    self.layer_idx, q[:, :, sl], k_rope[:, :, sl], k_raw[:, :, sl], v[:, :, sl], p
                )
            else:
                routed = router.decode_block(
                    self.layer_idx, q[:, :, sl], k_rope[:, :, sl], k_raw[:, :, sl], v[:, :, sl], p
                )
            outs.append(self._attend(routed, q[:, :, sl]))
            p += take
            done += take
        return torch.cat(outs, dim=2)

    def _attend(self, routed: RoutedKV, q: torch.Tensor) -> torch.Tensor:
        """Compute attention from the router-provided KV — the router supplies *data only*.

        The router selected what each query may see and fetched it into VRAM; scoring, the
        softmax and the output are this attention module's responsibility.  Accumulate in fp32
        (as torch SDPA does) so a bf16 run tracks the dense baseline across all 48 layers.
        Exact mode uses only the exact token KV; summary mode additionally attends the routed
        group summaries (the approximation the existing class relies on).
        """
        k, v, mask = routed.token_k, routed.token_v, routed.token_mask
        if self.mode == "summary" and routed.summary_k is not None:
            k = torch.cat([k, routed.summary_k], dim=-2)
            v = torch.cat([v, routed.summary_v], dim=-2)
            mask = torch.cat([mask, routed.summary_mask], dim=-1)
        scores = torch.einsum("bhld,bhrd->bhlr", q.float(), k.float()) * routed.scale
        scores = scores.masked_fill(~mask, _NEG)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("bhlr,bhrd->bhld", probs, v.float()).to(v.dtype)

    # ------------------------------------------------------------------
    def _make_store(self, B: int, dtype: torch.dtype, device: torch.device):
        kwargs = dict(
            compute_device=device, policy=self._policy, kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, chunk_size=self.chunk_size,
            groups_per_chunk=self._cfg.groups_per_chunk, batch_size=B, dtype=dtype,
        )
        if self.cache_location == "vram":
            return VramKVCacheStore(**kwargs)
        # RAM tier: cold KV record in host memory, only routed chunks pulled to VRAM each step.
        # pin_memory=False keeps the (multi-GB at 32K) record off the limited pinned pool and
        # makes H2D/D2H copies synchronous (no async-lifetime hazard on the streaming path).
        # A bounded LRU VRAM cache keeps recurring chunks resident so consecutive decode steps
        # only copy newly-required chunks.
        return RamKVCacheStore(pin_memory=False, vram_cache_chunks=self.vram_cache_chunks, **kwargs)

    def _get_router(self, pkv: Any, B: int, dtype: torch.dtype, device: torch.device) -> ChunkRouter:
        """One router/store shared by all layers, attached to the ``past_key_values`` object."""
        holder = pkv if pkv is not None else self  # fall back to per-module if no cache passed
        router = getattr(holder, "_kv_router", None)
        if router is None:
            store = self._make_store(B, dtype, device)
            cls = ExactChunkRouter if self.mode == "exact" else ChunkRouter
            router = cls(self._cfg, store)
            setattr(holder, "_kv_router", router)
        return router


# =================================================================================================
# Model surgery
# =================================================================================================
def _iter_attention_layers(model: nn.Module):
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("model.model.layers not found")
    for layer in layers:
        if hasattr(layer, "self_attn"):
            yield layer


def restore_original_attention(model: nn.Module) -> int:
    """Undo a previous replacement, putting the original ``self_attn`` modules back."""
    n = 0
    for layer in _iter_attention_layers(model):
        a = layer.self_attn
        if isinstance(a, QwenRoutedAttention):
            layer.self_attn = a.orig
            n += 1
    return n


def replace_qwen_attention_with_router(
    model: nn.Module,
    *,
    mode: str = "exact",
    keep_first: int = 2,
    keep_last: int = 2,
    topk_chunks: int = 8,
    topk_groups: int = 32,
    chunk_size: int = 64,
    group_size: int = 16,
    cache_location: str = "vram",
    vram_cache_chunks: int = 256,
) -> int:
    """Replace every ``self_attn`` with a ``QwenRoutedAttention`` (idempotent: unwraps first)."""
    restore_original_attention(model)
    config = model.config
    count = 0
    for layer in _iter_attention_layers(model):
        orig = layer.self_attn
        layer.self_attn = QwenRoutedAttention(
            orig, config, mode=mode, chunk_size=chunk_size, group_size=group_size,
            keep_first=keep_first, keep_last=keep_last, topk_chunks=topk_chunks,
            topk_groups=topk_groups, cache_location=cache_location,
            vram_cache_chunks=vram_cache_chunks,
        )
        count += 1
    return count
