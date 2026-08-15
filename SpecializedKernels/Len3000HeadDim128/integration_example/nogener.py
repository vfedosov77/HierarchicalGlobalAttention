"""Stock Alpamayo 1.5 trajectory sample with VLM prefill only (no CoT generate).

Prefill is the official VLM forward with ``max_new_tokens=0``. Diffusion is
the same 10-step Euler expert as ROS; by default it is CUDA-graphed
(``GraphedEulerDecoder``), which is why ROS reports ~117 ms and the
eager ``model.diffusion.sample`` loop is ~300 ms. Pass ``use_cuda_graph=False``
to force the eager path.
"""
from __future__ import annotations

import time
from typing import Any

import einops
import torch

from alpamayo1_5.models.token_utils import extract_text_tokens, to_special_token


@torch.inference_mode()
def sample_no_generation(
    model,
    data: dict[str, Any],
    *,
    num_traj_samples: int = 1,
    num_traj_sets: int = 1,
    diffusion_kwargs: dict[str, Any] | None = None,
    return_extra: bool = False,
    time_it: bool = True,
    use_cuda_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prefill the VLM (no generated tokens) and run expert diffusion.

    ``data`` matches the official helper layout::

        {
            "tokenized_data": processor output (input_ids + image kwargs),
            "ego_history_xyz": [B, 1, T_hist, 3],
            "ego_history_rot": [B, 1, T_hist, 3, 3],
        }
    """
    n_samples_total = num_traj_samples * num_traj_sets
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    if n_traj_group != 1:
        raise ValueError("only one trajectory group is supported")
    tokenized = dict(data["tokenized_data"])
    input_ids = tokenized.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
    )
    device = input_ids.device
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        tokenized["attention_mask"] = attention_mask

    stats: dict[str, Any] = {"S": int(input_ids.shape[1]), "generated": 0}

    fwd = dict(tokenized)
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with autocast:
        try:
            vlm_out = model.vlm(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
                **fwd,
            )
        except TypeError:
            fwd.pop("mm_token_type_ids", None)
            vlm_out = model.vlm(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
                **fwd,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    stats["vlm_sec"] = time.perf_counter() - t0

    prompt_cache = vlm_out.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    stats["prefill_seq_len"] = int(prefill_seq_len)
    adapter = getattr(model, "_alpamayo_routed_adapter", None)
    rope_deltas = model.vlm.model.rope_deltas
    sequences = input_ids
    b_star = sequences.shape[0] * n_samples_total
    if n_samples_total > 1:
        sequences = sequences.repeat_interleave(n_samples_total, dim=0)

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    # No CoT: there is no <|traj_future_start|>. Offset = last prompt token + 1,
    # which equals the KV cache length, so the expert sees a contiguous prefix.
    offset = model._find_eos_offset(
        sequences=sequences,
        eos_token_id=eos_id,
        device=device,
        warn=False,
    )
    prefix_mask = tokenized.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
    position_ids, attn_4d = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embeds = model.action_in_proj(x, t)
        if embeds.dim() == 2:
            embeds = embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attn_4d,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

    graphed_ok = (
        use_cuda_graph
        and device.type == "cuda"
        and B * n_samples_total == 1
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    sampled = None
    if graphed_ok:
        try:
            from .graphed import sample_euler_graphed

            with autocast:
                sampled = sample_euler_graphed(
                    model,
                    prompt_cache,
                    n_prompt=int(prefill_seq_len),
                    n_diff=int(n_diffusion_tokens),
                    position_ids=position_ids,
                    device=device,
                )
            stats["diffusion_graph"] = True
        except Exception as exc:
            print(
                f"[stock-hga] CUDA graph unavailable ({type(exc).__name__}: {str(exc)[:160]}); "
                "eager Euler",
                flush=True,
            )
            sampled = None
            stats["diffusion_graph"] = False
    if sampled is None:
        if adapter is not None and hasattr(adapter, "wrap_prompt_cache"):
            prompt_cache = adapter.wrap_prompt_cache(
                prompt_cache, int(prefill_seq_len), int(n_diffusion_tokens),
            )
        with autocast:
            sampled = model.diffusion.sample(
                batch_size=B * n_samples_total,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **(diffusion_kwargs or {}),
            )
        stats.setdefault("diffusion_graph", False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    stats["diffusion_sec"] = time.perf_counter() - t1
    stats["predict_sec"] = stats["vlm_sec"] + stats["diffusion_sec"]

    hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
    hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled, hist_xyz, hist_rot)
    pred_xyz = einops.rearrange(
        pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
    )
    pred_rot = einops.rearrange(
        pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
    )

    extra: dict[str, Any] = {"stats": stats}
    if return_extra:
        extra.update(extract_text_tokens(model.tokenizer, sequences))
    if time_it:
        extra_r = ""
        ad = getattr(model, "_alpamayo_routed_adapter", None)
        if ad is not None:
            extra_r = f"  route={getattr(ad, '_n_route', '?')} reuse={getattr(ad, '_n_reuse', '?')}"
        gtag = "  graph" if stats.get("diffusion_graph") else "  eager"
        print(
            f"[nogener] S={stats['S']} generated=0  "
            f"vlm={stats['vlm_sec'] * 1e3:.0f}ms diffusion={stats['diffusion_sec'] * 1e3:.0f}ms "
            f"predict={stats['predict_sec'] * 1e3:.0f}ms{gtag}{extra_r}",
            flush=True,
        )
    return pred_xyz, pred_rot, extra
