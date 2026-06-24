"""SGLang attention backend for Hierarchical Global Attention (HGA).

This module adapts the engine-neutral :class:`~FastInference.hga_core.runner.HgaLayerRunner`
to SGLang's :class:`AttentionBackend` contract.  Per the SGLang backend-extension
guide the required steps are, in order:

1. ``init_forward_metadata`` / ``forward_extend`` / ``forward_decode`` (eager),
2. then CUDA-graph capture/replay support.

v0 implements the eager path (batch=1 first, then batched by looping requests)
and stubs the CUDA-graph hooks so they can be filled once eager decode is fast.

Import safety
-------------
``sglang`` is an optional dependency.  When it is not installed the module still
imports (so it can be inspected / unit-tested), but :class:`HgaAttentionBackend`
falls back to a local ``object`` base and ``register_hga_backend`` raises a clear
error.

Known v0 limitation
-------------------
SGLang applies RoPE inside the model's attention module *before* calling
``backend.forward``, so the backend only receives RoPE-applied ``k``.  HGA's
mixed-RoPE summaries want the *pre-RoPE* ``k`` as well.  Until the Qwen3-HGA
model runner exposes a pre-RoPE hook, this adapter uses the roped ``k`` as the
``k_raw`` argument (an approximation flagged here and in the README roadmap).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..hga_core.config import HgaConfig
from ..hga_core.runner import HgaLayerRunner

try:  # pragma: no cover - optional dep
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend as _Base
    _HAS_SGLANG = True
except Exception:  # pragma: no cover
    _Base = object
    _HAS_SGLANG = False


def hga_config_from_model_runner(model_runner, **overrides) -> HgaConfig:
    """Build an :class:`HgaConfig` from a SGLang ``ModelRunner``.

    Reads head/layer geometry from ``model_runner.model_config`` and applies any
    ``overrides`` (chunk/group sizes, top-k budgets, cache budgets, ...).
    """
    mc = model_runner.model_config
    num_kv = getattr(mc, "get_num_kv_heads", None)
    kv_heads = mc.get_num_kv_heads(model_runner.tp_size) if callable(num_kv) else mc.num_key_value_heads
    q_heads = mc.num_attention_heads // getattr(model_runner, "tp_size", 1)
    cfg_kwargs = dict(
        num_layers=mc.num_hidden_layers,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=mc.head_dim,
        rope_theta=float(getattr(mc.hf_config, "rope_theta", 1_000_000.0)),
    )
    cfg_kwargs.update(overrides)
    return HgaConfig(**cfg_kwargs)


class HgaAttentionBackend(_Base):
    """HGA attention backend for SGLang (eager v0)."""

    def __init__(self, model_runner, cfg: Optional[HgaConfig] = None, **cfg_overrides):
        if not _HAS_SGLANG:
            raise RuntimeError(
                "sglang is not installed; HgaAttentionBackend cannot be used as a "
                "live backend. Install sglang or use FastInference.hga_core directly."
            )
        super().__init__()
        self.model_runner = model_runner
        self.device = torch.device(getattr(model_runner, "device", "cuda"))
        self.cfg = cfg or hga_config_from_model_runner(model_runner, **cfg_overrides)
        self.runner = HgaLayerRunner(self.cfg, self.device)
        self.forward_metadata = None

    # ------------------------------------------------------------------
    # metadata (eager)
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch):
        # HGA routing is query-dependent and happens inside forward_*, so the
        # per-iter metadata here is just the lightweight batch view the forward
        # methods read (seq_lens / positions).
        self.forward_metadata = {
            "seq_lens": forward_batch.seq_lens,
            "positions": getattr(forward_batch, "positions", None),
            "batch_size": forward_batch.batch_size,
        }

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        self.init_forward_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    # ------------------------------------------------------------------
    # CUDA graph hooks — v0 stubs (added after eager decode is fast)
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        raise NotImplementedError(
            "HGA CUDA-graph capture is a v1 deliverable. Run with cuda_graph "
            "disabled until eager forward_decode meets latency targets."
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _reshape_qkv(self, q, k, v, layer):
        """[T, H*Dh] (or [T,H,Dh]) -> [H, T, Dh] / [KVH, T, Dh]."""
        Dh = layer.head_dim
        H = layer.tp_q_head_num
        KVH = layer.tp_k_head_num
        T = q.shape[0]
        q = q.view(T, H, Dh).transpose(0, 1).contiguous()       # [H, T, Dh]
        k = k.view(T, KVH, Dh).transpose(0, 1).contiguous()     # [KVH, T, Dh]
        v = v.view(T, KVH, Dh).transpose(0, 1).contiguous()
        return q, k, v

    def _flatten_out(self, out, layer):
        # [H, T, Dh] -> [T, H*Dh]
        H, T, Dh = out.shape
        return out.transpose(0, 1).reshape(T, H * Dh)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache: bool = True, **kwargs):
        """Prefill / chunked prefill. Batch=1 fast path; loops requests otherwise."""
        layer_id = layer.layer_id
        qh, kh, vh = self._reshape_qkv(q, k, v, layer)
        seq_lens = forward_batch.seq_lens
        bs = forward_batch.batch_size
        if bs == 1:
            start = int(seq_lens[0].item()) - qh.shape[1]
            out = self.runner.prefill(layer_id, qh, kh, kh, vh, start_pos=max(0, start))
            return self._flatten_out(out, layer)
        # batch>1: split the packed tokens by per-request extend lengths
        ext = forward_batch.extend_seq_lens
        outs = []
        off = 0
        for i in range(bs):
            n = int(ext[i].item())
            start = int(seq_lens[i].item()) - n
            o = self.runner.prefill(
                layer_id, qh[:, off:off + n], kh[:, off:off + n],
                kh[:, off:off + n], vh[:, off:off + n], start_pos=max(0, start),
            )
            outs.append(o)
            off += n
        return self._flatten_out(torch.cat(outs, dim=1), layer)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache: bool = True, **kwargs):
        """Single-token decode for each request in the batch."""
        layer_id = layer.layer_id
        qh, kh, vh = self._reshape_qkv(q, k, v, layer)   # [H/KVH, bs, Dh] (1 token / req)
        seq_lens = forward_batch.seq_lens
        bs = forward_batch.batch_size
        outs = []
        for i in range(bs):
            start = int(seq_lens[i].item()) - 1
            o = self.runner.decode_step(
                layer_id,
                qh[:, i:i + 1], kh[:, i:i + 1], kh[:, i:i + 1], vh[:, i:i + 1],
                start_pos=max(0, start),
            )
            outs.append(o)
        return self._flatten_out(torch.cat(outs, dim=1), layer)
