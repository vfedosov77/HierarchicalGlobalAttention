"""CUDA-graph the 10-step expert Euler loop (same idea as ROS).

ROS diffusion is ~117 ms because ``GraphedTrajectoryDecoder`` captures
the whole Euler loop and replays it. The official
``model.diffusion.sample`` path is an eager Python loop: 10 expert
forwards × 36 HF layers of host launch. That is ~300 ms for the same
kernels. This file is the stock-repo copy of that graph — it does not
import ``alpamayo_ros``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from .cache import DiffusionStaticCache


class GraphedEulerDecoder:
    """Static-cache 10-step Euler, captured as one CUDA graph."""

    def __init__(
        self,
        model: Any,
        cache: DiffusionStaticCache,
        *,
        position_ids: torch.Tensor,
        n_diff: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.n_diff = int(n_diff)
        self.device = device
        self.x_dims = tuple(model.action_space.get_action_space_dims())
        self.num_steps = int(model.diffusion.num_inference_steps)
        self.position_ids = position_ids.contiguous()
        self.mask = attention_mask
        self.x0 = torch.zeros((1, *self.x_dims), dtype=torch.float32, device=device)
        self.time_steps = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)
        self.forward_kwargs: dict[str, Any] = {}
        if model.config.expert_non_causal_attention:
            self.forward_kwargs["is_causal"] = False
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.out: Optional[torch.Tensor] = None

    def _step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        model = self.model
        embeds = model.action_in_proj(x, t)
        if embeds.dim() == 2:
            embeds = embeds.view(x.shape[0], self.n_diff, -1)
        expert_out = model.expert(
            inputs_embeds=embeds,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            attention_mask=self.mask,
            use_cache=True,
            **self.forward_kwargs,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diff :]
        return model.action_out_proj(last_hidden).view(-1, *self.x_dims)

    def _euler(self) -> torch.Tensor:
        x = self.x0
        n_dim = len(self.x_dims)
        for i in range(self.num_steps):
            dt = self.time_steps[i + 1] - self.time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(x.shape[0], *[1] * n_dim)
            t = self.time_steps[i].view(1, *[1] * n_dim).expand(x.shape[0], *[1] * n_dim)
            x = x + dt * self._step(x, t)
        return x

    def capture(self, *, adapter: Any = None, warmup: int = 3) -> None:
        if adapter is not None:
            # Warmup must not leave _diff_step mid-cycle: capture unrolls
            # steps 0..9 (route on 0,3,6,9) into the graph.
            adapter.reset_diffusion_step()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(max(1, warmup)):
                    if adapter is not None:
                        adapter.reset_diffusion_step()
                    self._euler()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if adapter is not None:
                adapter.reset_diffusion_step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._euler()
        self.graph, self.out = graph, out

    def run(self, noise: torch.Tensor) -> torch.Tensor:
        self.x0.copy_(noise)
        assert self.graph is not None and self.out is not None
        self.graph.replay()
        return self.out


def sample_euler_graphed(
    model: Any,
    vlm_cache: Any,
    *,
    n_prompt: int,
    n_diff: int,
    position_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Replay (or first-time capture) the graphed Euler loop."""
    adapter = getattr(model, "_alpamayo_routed_adapter", None)
    tag = "hga" if adapter is not None else "dense"
    key = (int(n_prompt), int(n_diff), tag)
    store: dict = model.__dict__.setdefault("_example_euler_graphs", {})
    dec: Optional[GraphedEulerDecoder] = store.get(key)
    if dec is None:
        if adapter is not None:
            cache = adapter.wrap_prompt_cache(vlm_cache, n_prompt, n_diff)
        else:
            cache = DiffusionStaticCache.from_vlm_cache(
                vlm_cache, n_prompt=n_prompt, n_diff=n_diff,
            )
        dec = GraphedEulerDecoder(
            model,
            cache,
            position_ids=position_ids,
            n_diff=n_diff,
            device=device,
            attention_mask=None,
        )
        dec.capture(adapter=adapter)
        store[key] = dec
        print(
            f"[stock-hga] CUDA-graphed Euler captured "
            f"({tag}, steps={dec.num_steps}, n_prompt={n_prompt}, Q={n_diff})",
            flush=True,
        )
    else:
        dec.cache.load_prompt(vlm_cache, n_prompt)
        if adapter is not None:
            adapter.refresh_diffusion_from_cache(dec.cache, n_prompt)
        if dec.position_ids.shape == position_ids.shape:
            dec.position_ids.copy_(position_ids)
        else:
            dec.position_ids = position_ids.contiguous()
    noise = torch.randn(1, *dec.x_dims, device=device)
    return dec.run(noise)
