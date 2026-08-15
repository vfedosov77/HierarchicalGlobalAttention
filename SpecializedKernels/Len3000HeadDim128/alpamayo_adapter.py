"""Wire Alpamayo to true two-level HGA (128-chunk / 16-group, 192 tokens)."""
from __future__ import annotations

import os
from typing import Any, List, Optional

import torch

from .attention import attach_to_alpamayo
from .kernel import (
    CHUNK,
    GROUP,
    HEAD_DIM,
    MAX_CHUNKS,
    MAX_GROUPS,
    MAX_SEQ,
    TOPK_C,
    TOPK_G,
    fill_hga_means,
)


def _as_model(obj: Any) -> Any:
    if hasattr(obj, "vlm") and hasattr(obj, "expert"):
        return obj
    for attr in ("model", "_model", "alpamayo"):
        inner = getattr(obj, attr, None)
        if inner is not None and hasattr(inner, "vlm") and hasattr(inner, "expert"):
            return inner
    raise TypeError(
        "adapt_alpamayo expected Alpamayo1_5 or a wrapper with .model/.vlm; "
        f"got {type(obj).__name__}"
    )


def _language_model(vlm: Any) -> Any:
    if hasattr(vlm, "model") and hasattr(vlm.model, "language_model"):
        return vlm.model.language_model
    if hasattr(vlm, "language_model"):
        return vlm.language_model
    return vlm


def _attn_modules(root: Any) -> List[Any]:
    found: List[Any] = []
    for mod in root.modules():
        if (
            hasattr(mod, "q_proj")
            and hasattr(mod, "k_proj")
            and hasattr(mod, "v_proj")
            and hasattr(mod, "layer_idx")
        ):
            found.append(mod)
    found.sort(key=lambda m: int(m.layer_idx) if m.layer_idx is not None else -1)
    return found


def _cfg_int(cfg: Any, *names: str, default: int) -> int:
    for name in names:
        if cfg is not None and hasattr(cfg, name):
            val = getattr(cfg, name)
            if val is not None:
                return int(val)
    return default


def _preallocate_attn(
    attn: Any,
    *,
    hq: int,
    kvh: int,
    max_q: int,
    device: torch.device,
    dtype: torch.dtype,
    for_diffusion: bool = False,
) -> None:
    attn._alpamayo_topk = TOPK_C
    attn._alpamayo_group_k = torch.empty(1, kvh, MAX_GROUPS, HEAD_DIM, device=device, dtype=dtype)
    attn._alpamayo_group_k_ready = False
    attn._alpamayo_chunk_k = torch.empty(1, kvh, MAX_CHUNKS, HEAD_DIM, device=device, dtype=dtype)
    attn._alpamayo_chunk_k_ready = False
    attn._alpamayo_out_bshd = torch.empty(1, max_q, hq, HEAD_DIM, device=device, dtype=dtype)
    if for_diffusion:
        attn._alpamayo_route_diff = torch.empty(1, hq, TOPK_C, device=device, dtype=torch.int32)
    else:
        attn._alpamayo_route = torch.empty(1, hq, MAX_CHUNKS, TOPK_C, device=device, dtype=torch.int32)
        attn._alpamayo_q_chunk = torch.empty(1, hq, MAX_CHUNKS, HEAD_DIM, device=device, dtype=dtype)


def _find_predictor(obj: Any) -> Optional[Any]:
    if hasattr(obj, "diffusion_kv_top_fraction") and hasattr(obj, "_graphed"):
        return obj
    for attr in ("_predictor", "predictor"):
        pred = getattr(obj, attr, None)
        if pred is not None and hasattr(pred, "diffusion_kv_top_fraction"):
            return pred
    model = getattr(obj, "model", None)
    if model is not None:
        pred = getattr(model, "_predictor", None)
        if pred is not None:
            return pred
    return None


def _disable_record_attention(obj: Any, model: Any) -> None:
    pred = _find_predictor(obj) or _find_predictor(model)
    if pred is None:
        return
    pred.diffusion_kv_top_fraction = 1.0
    graphed = getattr(pred, "_graphed", None)
    if graphed is not None:
        restore = getattr(graphed, "restore_standard_attention", None)
        if callable(restore):
            restore()
        pred._graphed = None


class AlpamayoRoutedAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.topk = TOPK_C
        self.topk_fine = TOPK_G
        self.hierarchy = (CHUNK, GROUP)
        self.vlm_attns = _attn_modules(_language_model(model.vlm))
        self.expert_attns = _attn_modules(model.expert)
        if not self.vlm_attns:
            raise RuntimeError("no VLM language-model attention modules found")
        if not self.expert_attns:
            raise RuntimeError("no expert attention modules found")

    def refresh_diffusion_from_cache(self, cache: Any, n_prompt: int) -> None:
        layers = getattr(cache, "layers", None)
        if not layers:
            return
        n_prompt = int(n_prompt)
        for attn in self.expert_attns:
            idx = int(attn.layer_idx)
            if idx < 0 or idx >= len(layers):
                continue
            keys = layers[idx].keys
            fill_hga_means(keys, attn._alpamayo_group_k, attn._alpamayo_chunk_k, n_prompt)
            attn._alpamayo_n_prompt = n_prompt
            attn._alpamayo_chunk_k_ready = True
            attn._alpamayo_group_k_ready = True


def _patch_cache_load(adapter: AlpamayoRoutedAdapter) -> None:
    targets: List[Any] = []
    for mod_name in (
        "alpamayo_ros.cuda_graph_diffusion",
        "alpamayo_ros.alpamayo_ros.cuda_graph_diffusion",
    ):
        try:
            import importlib
            targets.append(importlib.import_module(mod_name))
        except ImportError:
            continue
    for pack in targets:
        decoder_cls = getattr(pack, "GraphedTrajectoryDecoder", None)
        if decoder_cls is None:
            continue
        if getattr(decoder_cls.set_frame, "_alpamayo_routed", False):
            continue
        orig_set_frame = decoder_cls.set_frame

        def set_frame(self, prompt_cache, prefill_seq_len, offset, rope_deltas, _orig=orig_set_frame):
            _orig(self, prompt_cache, prefill_seq_len, offset, rope_deltas)
            ad = getattr(self, "_routed_adapter", adapter)
            cache = getattr(self, "cache", None)
            if cache is not None:
                ad.refresh_diffusion_from_cache(cache, prefill_seq_len)

        set_frame._alpamayo_routed = True
        decoder_cls.set_frame = set_frame

        orig_init = decoder_cls.__init__

        def __init__(self, *args, _orig=orig_init, **kwargs):
            _orig(self, *args, **kwargs)
            self._routed_adapter = adapter
            attach_to_alpamayo(expert=adapter.model.expert)

        decoder_cls.__init__ = __init__


def _patch_expert_eager_refresh(adapter: AlpamayoRoutedAdapter) -> None:
    expert = adapter.model.expert
    if getattr(expert.forward, "_alpamayo_routed", False):
        return
    orig = expert.forward
    last_id = {"cache": None, "n": -1}

    def forward(*args, past_key_values=None, **kwargs):
        cache = past_key_values
        if cache is not None and hasattr(cache, "layers") and hasattr(cache, "get_seq_length"):
            try:
                n = int(cache.get_seq_length())
            except TypeError:
                n = int(cache.get_seq_length(0))
            cid = id(cache)
            if last_id["cache"] != cid or last_id["n"] != n:
                if not getattr(cache.layers[0], "selected_keys", None):
                    adapter.refresh_diffusion_from_cache(cache, n)
                last_id["cache"] = cid
                last_id["n"] = n
        return orig(*args, past_key_values=past_key_values, **kwargs)

    forward._alpamayo_routed = True
    expert.forward = forward


def adapt_alpamayo(
    instance: Any,
    *,
    topk: int = TOPK_C,
    max_seq: int = MAX_SEQ,
    fp8_residuals: bool = False,
    fast_vision: bool = True,
    **_ignored: Any,
) -> AlpamayoRoutedAdapter:
    """Patch ``instance`` in place. ``fp8_residuals`` is ignored (not part of 2L).

    ``fast_vision`` swaps the ViT to batched SDPA + bf16 RoPE (not HGA).
    """
    model = _as_model(instance)
    text_cfg = getattr(getattr(model.vlm, "config", None), "text_config", None)
    expert_cfg = getattr(model.expert, "config", None)
    hq = _cfg_int(text_cfg, "num_attention_heads", default=_cfg_int(expert_cfg, "num_attention_heads", default=32))
    kvh = _cfg_int(text_cfg, "num_key_value_heads", default=_cfg_int(expert_cfg, "num_key_value_heads", default=8))
    n_diff = 64
    try:
        n_diff = int(model.action_space.get_action_space_dims()[0])
    except Exception:
        pass

    device = next(model.expert.parameters()).device
    dtype = torch.bfloat16

    attach_to_alpamayo(model.vlm, model.expert)
    if fast_vision:
        _patch_fast_vision_attention(model)
    adapter = AlpamayoRoutedAdapter(model)
    adapter.fp8_residual_layers = 0

    for attn in adapter.vlm_attns:
        _preallocate_attn(attn, hq=hq, kvh=kvh, max_q=max_seq, device=device, dtype=dtype)
    for attn in adapter.expert_attns:
        _preallocate_attn(
            attn, hq=hq, kvh=kvh, max_q=max(n_diff, 64), device=device, dtype=dtype,
            for_diffusion=True,
        )

    _disable_record_attention(instance, model)
    _patch_cache_load(adapter)
    _patch_expert_eager_refresh(adapter)
    _warmup_kernels(device, dtype, hq, kvh, n_diff)
    if fp8_residuals:
        print("[routed] fp8_residuals ignored on the 2-level path", flush=True)
    print(
        f"[routed] true 2-level {CHUNK}/{GROUP} topk_c={TOPK_C} topk_g={TOPK_G} "
        f"vlm_tokens={GROUP + TOPK_G * GROUP}",
        flush=True,
    )
    _maybe_install_vlm_profiler(model)
    try:
        from .profile_e2e_live import install_e2e_profiler

        install_e2e_profiler(instance, model)
    except Exception as exc:
        print(f"[e2e-prof] install failed: {exc}", flush=True)
    model._alpamayo_routed_adapter = adapter
    return adapter


def _maybe_install_vlm_profiler(model: Any) -> None:
    """Hook VLM ``forward`` so the breakdown runs even if ROS imports the image copy."""
    if os.environ.get("HGA_PROFILE_VLM", "") in ("", "0", "false", "False"):
        return
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        print("[vlm-prof] no model.vlm; breakdown off", flush=True)
        return
    try:
        from .profile_vlm_live import attach_vlm_profiler
    except Exception as exc:
        print(f"[vlm-prof] import failed: {exc}", flush=True)
        return
    if getattr(vlm, "_hga_prof_hooks", False):
        return
    prof = attach_vlm_profiler(vlm)

    def _pre(_m, _inp):
        prof.begin()

    def _post(_m, _inp, _out):
        prof.end()

    vlm.register_forward_pre_hook(_pre)
    vlm.register_forward_hook(_post)
    vlm._hga_prof_hooks = True
    print("[vlm-prof] hooked vlm.forward (CUDA-event breakdown)", flush=True)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_vision_bf16(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Same 2D RoPE as Qwen3-VL vision, but stay in the tensor dtype (bf16)."""
    cos = cos.unsqueeze(-2).to(dtype=q.dtype)
    sin = sin.unsqueeze(-2).to(dtype=q.dtype)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _patch_fast_vision_attention(model: Any) -> None:
    """ViT: bf16 RoPE + one batched SDPA when all images share a length.

    Flash-2 at S=720 / D=72 is slower than cuDNN SDPA on this GPU. The stock
    non-FA path issues one SDPA per image. Equal-length cameras (the bag) can
    be B=N SDPA. Stock RoPE casts Q/K to fp32 (~14 ms).
    """
    visual = getattr(getattr(getattr(model, "vlm", None), "model", None), "visual", None)
    if visual is None or not getattr(visual, "blocks", None):
        print("[vision] no visual.blocks; leaving stock attention", flush=True)
        return

    attn0 = visual.blocks[0].attn
    attn_cls = type(attn0)
    if getattr(attn_cls.forward, "_alpamayo_fast_vision", False):
        return
    orig_forward = attn_cls.forward

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        if position_embeddings is None:
            return orig_forward(self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).view(seq_length, 3, self.num_heads, -1)
        query_states, key_states, value_states = qkv.unbind(1)
        cos, sin = position_embeddings
        query_states, key_states = _rope_vision_bf16(query_states, key_states, cos, sin)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        n_img = int(lengths.numel())
        same = n_img > 0 and bool((lengths == lengths[0]).all().item())
        if same:
            s = int(lengths[0].item())
            q = query_states.view(n_img, s, self.num_heads, -1).transpose(1, 2)
            k = key_states.view(n_img, s, self.num_heads, -1).transpose(1, 2)
            v = value_states.view(n_img, s, self.num_heads, -1).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=self.scaling
            )
            attn_output = out.transpose(1, 2).contiguous().view(seq_length, -1)
        else:
            return orig_forward(
                self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs
            )
        return self.proj(attn_output)

    forward._alpamayo_fast_vision = True
    attn_cls.forward = forward

    # If a layer still hits the stock path, keep RoPE in bf16 there too.
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3

        if not getattr(qwen3.apply_rotary_pos_emb_vision, "_alpamayo_bf16", False):
            def apply_rotary_pos_emb_vision(q, k, cos, sin):
                return _rope_vision_bf16(q, k, cos, sin)

            apply_rotary_pos_emb_vision._alpamayo_bf16 = True
            qwen3.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
    except Exception:
        pass

    vcfg = getattr(visual, "config", None)
    if vcfg is not None:
        vcfg._attn_implementation = "sdpa"
    print(
        f"[vision] batched SDPA + bf16 RoPE on {attn_cls.__name__} "
        f"(D={getattr(attn0, 'head_dim', '?')} H={getattr(attn0, 'num_heads', '?')})",
        flush=True,
    )


def _warmup_kernels(device, dtype, hq, kvh, n_diff) -> None:
    from .attention import diffusion_cross_attention, vlm_prefill_attention

    s = 2688
    q = torch.empty(1, hq, s, HEAD_DIM, device=device, dtype=dtype)
    k = torch.empty(1, kvh, s, HEAD_DIM, device=device, dtype=dtype)
    v = torch.empty_like(k)
    vlm_prefill_attention(q, k, v)
    qd = torch.empty(1, hq, n_diff, HEAD_DIM, device=device, dtype=dtype)
    kd = torch.empty(1, kvh, s + n_diff, HEAD_DIM, device=device, dtype=dtype)
    vd = torch.empty_like(kd)
    diffusion_cross_attention(qd, kd, vd, n_prompt=s)
    torch.cuda.synchronize()
    print(
        f"[routed] 2L kernels compiled (S={s} Q={n_diff} hq={hq} kvh={kvh})",
        flush=True,
    )
