#!/usr/bin/env python3
"""HierarchicalAttention (HA) 40M model with a trainable, routed KV *memory*.

PoC idea
--------
HierarchicalAttention lets every query *select* the chunks of a KV cache that are
relevant to the current token.  Here we abuse that mechanism as a partial
replacement for a Mixture-of-Experts FFN: a fixed pool of KV-cache chunks
("topic memory") sits permanently in VRAM, and the routing picks only the few
chunks that matter for the current text.  Because the selection is sparse, the
gradient of the language-modelling loss only reaches the *selected* memory
chunks — exactly like an expert being activated on demand.

The memory is stored as **raw key / value tensors per layer** (one chunk == 64
tokens == one topic).  It is initialised from a forward pass over the skill text
(see ``generate_skill_kv_chunks_v1.py``) and then *detached* from the source
tokens/embeddings: it carries gradients during training (a leaf parameter with no
history) so updates flow into the memory tensors but no further.

Attention is purely additive: a query attends to

  * its normal causal context inside the training sequence (identical to the
    original dense model), **plus**
  * the top-k routed memory chunks (no causal mask among memory tokens).

With the memory disabled the model is byte-for-byte the original dense model, so
the same class is used for the "original checkpoint" baseline and for the new
KV-memory model — making the validation comparison apples-to-apples.

The module names (``self_attn.q_proj`` …) match ``speed_run_dense_muon_final.pt``
so the dense checkpoint loads strictly.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- KvRouter package import (works from repo root or ExistingModelFineTuning/) -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTMODEL = os.path.dirname(_HERE)
_EFT = os.path.dirname(_TESTMODEL)
_ROOT = os.path.dirname(_EFT)
for _p in (_EFT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramGradKVCacheStore, ChunkPlacementPolicy,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ExistingModelFineTuning.KvRouter import (  # type: ignore
        ChunkRouter, RouterConfig, VramGradKVCacheStore, ChunkPlacementPolicy,
    )


# -----------------------------------------------------------------------------
# Building blocks (identical math to the dense checkpoint)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 / rms) * self.weight.float()
        return out.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def _rotary_tables(positions: torch.Tensor, head_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape ``[len(positions), head_dim]`` in fp32."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [P, half]
    emb = torch.cat((freqs, freqs), dim=-1)                     # [P, head_dim]
    return emb.cos(), emb.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# -----------------------------------------------------------------------------
# HA routed-memory attention
# -----------------------------------------------------------------------------
class HAMemoryAttention(nn.Module):
    """GQA attention; routes over a trainable KV memory via a shared ``ChunkRouter``.

    The module keeps only the projections (names matching the dense checkpoint).  *What* each
    query attends to is decided by the shared :class:`ChunkRouter` owned by :class:`HAMemoryLM`:
    the trainable memory is seeded into the router's gradient-preserving store as the leading
    closed chunks and the sequence is routed right after it, so routing *selects* the few memory
    chunks relevant to the current text (the MoE-like sparse-expert idea) and gradients flow back
    to exactly those memory chunks via the store's gather path.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        kv_heads: int = 2,
        theta: float = 1_000_000.0,
        chunk_size: int = 64,
        topk_chunks: int = 8,
        layer_idx: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert nhead % kv_heads == 0
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = d_model // nhead
        self.theta = float(theta)
        self.chunk_size = chunk_size
        self.topk_chunks = topk_chunks
        self.layer_idx = layer_idx
        self.dropout_p = dropout
        self.num_key_value_groups = nhead // kv_heads

        self.q_proj = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(nhead * self.head_dim, d_model, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        router: Optional[ChunkRouter] = None,
        layer_idx: Optional[int] = None,
        start_pos: int = 0,
        kv_dtype: Optional[torch.dtype] = None,
        capture: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """``x``: [B, S, D].

        * ``router``: shared :class:`ChunkRouter`.  ``None`` → plain dense causal attention
          (used for the baseline and for memory initialisation).
        * ``start_pos``: absolute position of this block's first token (the sequence lives
          *after* the seeded memory, i.e. ``start_pos == mem_len`` for a fresh prefill).
        * ``capture``: if a list is passed (router ``None``), the raw per-token ``(k, v)`` of
          this sequence are appended (used to initialise the memory).
        """
        B, S, _ = x.shape
        H, KVH, Dh = self.nhead, self.kv_heads, self.head_dim
        rep = self.num_key_value_groups

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)     # [B, H, S, Dh]
        k = self.k_proj(x).view(B, S, KVH, Dh).transpose(1, 2)   # [B, KVH, S, Dh]
        v = self.v_proj(x).view(B, S, KVH, Dh).transpose(1, 2)   # [B, KVH, S, Dh]

        pos = torch.arange(start_pos, start_pos + S, device=x.device)
        cos, sin = _rotary_tables(pos, Dh, self.theta)           # [S, Dh]
        cos = cos[None, None]
        sin = sin[None, None]

        # --- dense causal path (baseline / memory initialisation) -----------------
        if router is None:
            if capture is not None:
                # Raw (pre-rope) K/V become the memory initialiser. Detach: the memory
                # is a leaf with no history back to the embeddings.
                capture.append((k.detach().squeeze(0).clone(), v.detach().squeeze(0).clone()))
            q_rope = _apply_rotary(q.float(), cos, sin).to(q.dtype)
            k_rope = _apply_rotary(k.float(), cos, sin).to(k.dtype)
            k_seq = k_rope.repeat_interleave(rep, dim=1)
            v_seq = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(
                q_rope, k_seq, v_seq, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(B, S, H * Dh)
            return self.o_proj(out)

        # --- routed path: the ChunkRouter selects memory + sequence context -------
        if layer_idx is None:
            layer_idx = self.layer_idx
        dt = kv_dtype if kv_dtype is not None else k.dtype
        q_rope = _apply_rotary(q.float(), cos, sin).to(dt)       # [B, H, S, Dh] (head-expanded)
        k_rope = _apply_rotary(k.float(), cos, sin).to(dt)       # [B, KVH, S, Dh]
        k_raw = k.to(dt)
        v = v.to(dt)
        cos_r = cos.to(dt)
        sin_r = sin.to(dt)
        segments = router.route_query_block(
            layer_idx, q_rope, k_rope, k_raw, v, start_pos, cos=cos_r, sin=sin_r,
        )
        out_heads = q_rope.new_empty(B, H, S, Dh)
        for routed, lo, hi in segments:
            # use_summaries=False: attend the real token KV only (the group-summary path is
            # never attended), so gradients reach the seeded memory tokens via the store.
            out_heads[:, :, lo:hi] = routed.attend(q_rope[:, :, lo:hi], use_summaries=False)
        out = out_heads.transpose(1, 2).contiguous().view(B, S, H * Dh)
        return self.o_proj(out)


# -----------------------------------------------------------------------------
# Decoder + LM
# -----------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn: HAMemoryAttention, dff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = attn
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, router=None, layer_idx=None, start_pos=0, kv_dtype=None, capture=None):
        h = self.self_attn(
            self.norm1(x), router=router, layer_idx=layer_idx,
            start_pos=start_pos, kv_dtype=kv_dtype, capture=capture,
        )
        x = x + self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)
        return x


class HAMemoryLM(nn.Module):
    """40M SmallLM whose attention can route over a trainable KV memory."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 384,
        num_heads: int = 6,
        kv_heads: int = 2,
        num_layers: int = 8,
        dff: int = 2048,
        chunk_size: int = 64,
        group_size: int = 16,
        topk_chunks: int = 20,
        topk_groups: int = 32,
        keep_first: int = 2,
        keep_last: int = 8,
        theta: float = 1_000_000.0,
        dropout: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        self.theta = float(theta)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim,
                HAMemoryAttention(
                    hidden_dim, num_heads, kv_heads, theta=theta,
                    chunk_size=chunk_size, topk_chunks=topk_chunks,
                    layer_idx=i, dropout=dropout,
                ),
                dff, dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

        # Shared router config / placement policy. The router is fully memory-agnostic:
        # the trainable memory is seeded into the store as ordinary leading closed chunks
        # (positions [0, M)) through the *same* chunk-closing path the router uses for any
        # streamed sequence, so no memory-specific code lives in the router.
        self._cfg = RouterConfig(
            nhead=num_heads, kv_heads=kv_heads, head_dim=self.head_dim,
            chunk_size=chunk_size, group_size=group_size,
            topk_chunks=topk_chunks, topk_groups=topk_groups, theta=theta,
            current_group_summaries=False,
        )
        self._policy = ChunkPlacementPolicy(
            keep_last=keep_last, keep_first=keep_first, first_token_level=True,
        )
        self._router: Optional[ChunkRouter] = None
        self._mem_len = 0
        self._kv_dtype: Optional[torch.dtype] = None

        # KV memory (one [KVH, M, Dh] tensor per layer). Allocated by
        # ``init_memory_from_tokens``; ``None`` means the model is the plain
        # dense baseline.
        self.memory_k: Optional[nn.ParameterList] = None
        self.memory_v: Optional[nn.ParameterList] = None

    # ------------------------------------------------------------------
    @property
    def has_memory(self) -> bool:
        return self.memory_k is not None

    def _working_dtype(self) -> torch.dtype:
        """The dtype the projections produce (so every store slab shares one dtype)."""
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):
                return torch.get_autocast_gpu_dtype()
        return self.embedding.weight.dtype

    def begin_sequence(self, batch_size: int, device: torch.device, use_memory: bool = True) -> int:
        """Reset the shared router and seed the per-layer trainable memory (positions ``[0, M)``).

        Must be called once before a fresh sequence (prefill, or before the first decode step of an
        incremental generation).  Returns the memory length ``M`` (0 when memory is disabled), which
        is the ``start_pos`` of the sequence that follows.  The memory is streamed into the store as
        ordinary leading closed chunks through the router's normal (gradient-preserving) chunk-closing
        path, so the routed attention's loss gradient reaches exactly the selected memory chunks and
        the router stays completely memory-agnostic.
        """
        use_memory = use_memory and self.has_memory
        dt = self._working_dtype()
        self._kv_dtype = dt
        store = VramGradKVCacheStore(
            compute_device=device, policy=self._policy, kv_heads=self.kv_heads,
            head_dim=self.head_dim, chunk_size=self.chunk_size,
            groups_per_chunk=self._cfg.groups_per_chunk, batch_size=batch_size, dtype=dt,
        )
        self._router = ChunkRouter(self._cfg, store)
        self._mem_len = self.memory_k[0].shape[1] if use_memory else 0
        if use_memory:
            mpos = torch.arange(0, self._mem_len, device=device)
            mcos, msin = _rotary_tables(mpos, self.head_dim, self.theta)
            mcos = mcos[None, None]
            msin = msin[None, None]
            for i in range(self.num_layers):
                mk = self.memory_k[i].unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
                mv = self.memory_v[i].unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
                mk_rope = _apply_rotary(mk.float(), mcos, msin).to(dt)
                self._seed_memory_layer(i, mk_rope, mk.to(dt), mv.to(dt))
        return self._mem_len

    def _seed_memory_layer(self, layer: int, mk_rope: torch.Tensor, mk_raw: torch.Tensor,
                           mv: torch.Tensor) -> None:
        """Stream one layer's memory KV into the store as whole closed chunks.

        Uses the router's *existing* streaming primitives (``_append_active`` + ``_close_active_chunk``)
        — the exact code that closes a chunk during normal decode — so memory chunks get summaries
        identical to streamed chunks and stay gradient-connected to the memory parameters. The router
        needs no memory-specific code.
        """
        C = self.chunk_size
        n_mem = mk_rope.shape[2] // C
        r = self._router
        for j in range(n_mem):
            sl = slice(j * C, (j + 1) * C)
            r._append_active(layer, mk_rope[:, :, sl], mk_raw[:, :, sl], mv[:, :, sl], j * C)
            r._close_active_chunk(layer, j)

    def _block(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run one routed block (the router state must already be set up via ``begin_sequence``)."""
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, router=self._router, layer_idx=i, start_pos=start_pos,
                      kv_dtype=self._kv_dtype)
        x = self.final_norm(x)
        return self.lm_head(x)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                use_memory: bool = True):
        B, S = input_ids.shape
        self.begin_sequence(B, input_ids.device, use_memory=use_memory)
        logits = self._block(input_ids, self._mem_len)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.reshape(-1, self.vocab_size).float(), labels.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate_causality(self, input_ids: torch.Tensor, use_memory: bool = True) -> Dict[str, float]:
        """Confirm the routed attention is causal: incremental decode must match prefill.

        Runs the same ``[1, T]`` sequence two ways and compares the (teacher-forced) LM loss:

        * **prefill** — the whole block routed in one call;
        * **decode**  — the tokens streamed one at a time through the router's KV cache
          (``generate``-style, but feeding the ground-truth tokens so the loss is comparable).

        A future-token leak would make decode (which can only ever see the past) disagree with
        prefill, or make prefill *cheat* and score a lower loss.  Returns both losses, their
        difference and the max per-logit discrepancy.
        """
        was_training = self.training
        self.eval()
        B, T = input_ids.shape
        assert B == 1, "causality check expects batch size 1"
        V = self.vocab_size

        # prefill: the whole sequence routed at once
        self.begin_sequence(B, input_ids.device, use_memory=use_memory)
        mem_len = self._mem_len
        logits_prefill = self._block(input_ids, mem_len)                       # [1, T, V]

        # decode: stream one token at a time, reusing the streaming KV cache (no reset)
        self.begin_sequence(B, input_ids.device, use_memory=use_memory)
        dec = [self._block(input_ids[:, t:t + 1], mem_len + t) for t in range(T)]
        logits_decode = torch.cat(dec, dim=1)                                  # [1, T, V]

        tgt = input_ids[:, 1:].reshape(-1)
        pf = F.cross_entropy(logits_prefill[:, :-1].reshape(-1, V).float(), tgt)
        dc = F.cross_entropy(logits_decode[:, :-1].reshape(-1, V).float(), tgt)
        max_logit_diff = (logits_prefill.float() - logits_decode.float()).abs().max()

        self.train(was_training)
        return {
            "prefill_loss": float(pf),
            "decode_loss": float(dc),
            "delta": float(dc - pf),
            "max_logit_diff": float(max_logit_diff),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_memory_from_tokens(self, token_ids: torch.Tensor) -> None:
        """Initialise the KV memory from a single ``[M]`` sequence of token ids.

        Runs a plain causal forward over the skill text capturing every layer's
        raw ``(k, v)``; those become the per-layer memory tensors. The result is
        detached (no history back to the embeddings) and marked trainable.
        """
        device = self.embedding.weight.device
        ids = token_ids.to(device).long().view(1, -1)
        M = ids.shape[1]
        assert M % self.chunk_size == 0, (
            f"memory length {M} must be a multiple of chunk_size {self.chunk_size}")

        captures: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in self.layers]
        x = self.embedding(ids)
        for i, layer in enumerate(self.layers):
            # capture raw k/v of this layer, with NO router (plain causal)
            x = layer(x, router=None, capture=captures[i])

        mem_k = [cap[0][0].contiguous() for cap in captures]  # each [KVH, M, Dh]
        mem_v = [cap[0][1].contiguous() for cap in captures]

        self.memory_k = nn.ParameterList(
            [nn.Parameter(t.detach().clone(), requires_grad=True) for t in mem_k])
        self.memory_v = nn.ParameterList(
            [nn.Parameter(t.detach().clone(), requires_grad=True) for t in mem_v])

    # ------------------------------------------------------------------
    def memory_parameters(self) -> List[nn.Parameter]:
        if not self.has_memory:
            return []
        return list(self.memory_k.parameters()) + list(self.memory_v.parameters())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size: int, ignore_index: int, chunk_size: int = 64,
                group_size: int = 16, topk_chunks: int = 20, topk_groups: int = 32,
                keep_first: int = 2, keep_last: int = 8) -> HAMemoryLM:
    return HAMemoryLM(
        vocab_size=vocab_size, hidden_dim=384, num_heads=6, kv_heads=2,
        num_layers=8, dff=2048, chunk_size=chunk_size, group_size=group_size,
        topk_chunks=topk_chunks, topk_groups=topk_groups,
        keep_first=keep_first, keep_last=keep_last,
        ignore_index=ignore_index,
    )


# -----------------------------------------------------------------------------
# Out-of-the-box self-test: gradients flow to the memory + causality holds
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = 64
    vocab = 512
    model = build_model(vocab, ignore_index=-100, chunk_size=C, topk_chunks=20).to(dev)

    # Random skill memory: 8 topic chunks x 64 tokens.
    skill = torch.randint(0, vocab, (8 * C,), device=dev)
    model.init_memory_from_tokens(skill)
    n_mem = sum(p.numel() for p in model.memory_parameters())
    print(f"memory params: {n_mem:,}  ({len(model.memory_k)} layers x {tuple(model.memory_k[0].shape)})")

    # --- gradient flow (train mode, memory enabled) ---------------------------
    model.train()
    ids = torch.randint(0, vocab, (2, 4 * C), device=dev)
    tgt = torch.randint(0, vocab, (2, 4 * C), device=dev)
    _, loss = model(ids, labels=tgt, use_memory=True)
    loss.backward()
    g0 = model.memory_k[0].grad
    g_last = model.memory_v[-1].grad
    print(f"loss {float(loss):.4f}")
    print(f"memory_k[0] grad: nonzero={g0 is not None and float(g0.abs().sum()) > 0} "
          f"sum={float(g0.abs().sum()):.4e}")
    print(f"memory_v[-1] grad: nonzero={g_last is not None and float(g_last.abs().sum()) > 0} "
          f"sum={float(g_last.abs().sum()):.4e}")

    # --- causality: decode (512 generated tokens) vs prefill ------------------
    seq = torch.randint(0, vocab, (1, 512), device=dev)
    rep = model.validate_causality(seq, use_memory=True)
    print(f"causality (512 tok): prefill={rep['prefill_loss']:.5f}  "
          f"decode={rep['decode_loss']:.5f}  delta={rep['delta']:+.5f}  "
          f"max|dlogit|={rep['max_logit_diff']:.4e}")
