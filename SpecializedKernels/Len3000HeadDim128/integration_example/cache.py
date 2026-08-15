"""Fixed-slot expert KV cache for the stock (no-ROS) HGA example.

VLM prefill still writes a growing ``DynamicCache``. The Euler loop only
appends 64 action tokens and then crops them back. ``torch.cat`` +
``crop`` on a 3K prefix, 36 layers × 10 steps, is allocator traffic the
ROS node already avoided with ``StaticExpertCache``.

This is the same idea, local to the example: copy the VLM prefix once
into ``[B, Hkv, n_prompt + 64, D]`` slots. Each expert layer overwrites
the last 64 keys/values. No cat, no crop, prefix stays contiguous.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class _FixedDiffLayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self, n_prompt: int, n_diff: int):
        super().__init__()
        self.n_prompt = int(n_prompt)
        self.n_diff = int(n_diff)

    @property
    def capacity(self) -> int:
        return self.n_prompt + self.n_diff

    def early_init(self, num_heads: int, head_dim: int, dtype, device, batch_size: int = 1):
        shape = (batch_size, num_heads, self.capacity, head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty(shape, dtype=dtype, device=device)
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.early_init(
            key_states.shape[1],
            key_states.shape[-1],
            key_states.dtype,
            key_states.device,
            key_states.shape[0],
        )

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        start = self.n_prompt
        self.keys[:, :, start : start + n].copy_(key_states)
        self.values[:, :, start : start + n].copy_(value_states)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.capacity, 0

    def get_seq_length(self) -> int:
        return self.capacity

    def get_max_length(self) -> int:
        return self.capacity

    get_max_cache_shape = get_max_length

    def crop(self, max_length: int) -> None:
        # Prefix is pinned; diffusion slots are overwritten in-place.
        return None


class DiffusionStaticCache(Cache):
    """VLM prefix + 64 diffusion slots. Drop-in for ``expert(..., past_key_values=)``."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        n_prompt: int,
        n_diff: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int = 1,
    ):
        layers = [_FixedDiffLayer(n_prompt, n_diff) for _ in range(num_layers)]
        super().__init__(layers=layers)
        self.n_prompt = int(n_prompt)
        self.n_diff = int(n_diff)
        for layer in self.layers:
            layer.early_init(num_kv_heads, head_dim, dtype, device, batch_size)

    @classmethod
    def from_vlm_cache(
        cls,
        src: Any,
        *,
        n_prompt: int,
        n_diff: int,
    ) -> "DiffusionStaticCache":
        ref = src.layers[0].keys
        n_prompt = int(n_prompt)
        cache = cls(
            num_layers=len(src.layers),
            num_kv_heads=ref.shape[1],
            head_dim=ref.shape[-1],
            n_prompt=n_prompt,
            n_diff=int(n_diff),
            dtype=ref.dtype,
            device=ref.device,
            batch_size=ref.shape[0],
        )
        cache.load_prompt(src, n_prompt)
        return cache

    def load_prompt(self, src: Any, n_prompt: int) -> None:
        n_prompt = int(n_prompt)
        for dst, layer in zip(self.layers, src.layers):
            dst.keys[:, :, :n_prompt].copy_(layer.keys[:, :, :n_prompt])
            dst.values[:, :, :n_prompt].copy_(layer.values[:, :, :n_prompt])
