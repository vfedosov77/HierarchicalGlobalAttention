"""Wire an Alpamayo instance to the routed VLM + diffusion kernels.

Call from the ROS node after the model is loaded, before the first infer /
CUDA-graph capture::

    from SpecializedKernels.Len3000HeadDim128.alpamayo_adapter import adapt_alpamayo
    adapt_alpamayo(model)          # Alpamayo1_5
    adapt_alpamayo(wrapper)        # AlpamayoOptimizedModelWrapper
    adapt_alpamayo(predictor)      # FP8OptimizedPredictor

What this does
--------------
* VLM **language-model** self-attention → ``vlm_prefill_attention`` (visual +
  extra prompt tokens, causal, GQA 32/8, ``head_dim=128``). Vision encoder is
  left alone (``head_dim=72``).
* Expert diffusion attention → ``diffusion_cross_attention``. Prefix chunk-mean
  keys are computed **once per frame** from the expert cache (fixed averages)
  and reused for all Euler steps. ``record_attention`` is disabled.
* Preallocates ``chunk_k`` and HF-layout ``out`` on every attention layer so
  the hot path does not malloc or transpose Q/K/V. Expert layers also keep
  prefix K/V in preallocated fp8 for native tensor-core attention.
* Forces ``diffusion_kv_top_fraction=1`` and drops any already-captured
  top-k CUDA graph so the next capture uses the new kernels.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import torch

from .attention import (
    DEFAULT_TOPK,
    DIFFUSION_ATTN_NAME,
    VLM_ATTN_NAME,
    attach_to_alpamayo,
    hga2_diff_tokens,
    hga2_vlm_tokens,
    prompt_chunk_keys,
    resolve_dot_kind,
    resolve_hga_from_env,
)
from .kernel import (
    CHUNK,
    DOT_BF16,
    DOT_FP8_NATIVE,
    HEAD_DIM,
    MAX_SEQ,
    N_BLOCK,
    copy_to_fp8,
    fill_chunk_keys,
)

_MAX_SEQ = MAX_SEQ


def _as_model(obj: Any) -> Any:
    """Accept Alpamayo1_5, the ROS wrapper, or the speculative predictor."""
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
    topk: int,
    native_fp8: bool = False,
    max_kv: Optional[int] = None,
    hierarchy: Optional[tuple] = None,
    topk_fine: int = 2,
    dense_current: bool = False,
) -> None:
    attn._alpamayo_topk = int(topk)
    attn._alpamayo_topk_fine = int(topk_fine)
    attn._alpamayo_hierarchy = hierarchy
    attn._alpamayo_dense_current = bool(dense_current)
    attn._alpamayo_route_block = int(
        hierarchy[0] if hierarchy is not None else os.environ.get("HGA_ROUTE_BLOCK", CHUNK)
    )
    explicit = os.environ.get("HGA_DOT_DTYPE")
    if explicit:
        attn._alpamayo_dot_kind = resolve_dot_kind(explicit)
    elif native_fp8 and hasattr(torch, "float8_e4m3fn"):
        attn._alpamayo_dot_kind = DOT_FP8_NATIVE
    else:
        attn._alpamayo_dot_kind = DOT_BF16
    attn._alpamayo_chunk_k = torch.empty(
        1, kvh, N_BLOCK, HEAD_DIM, device=device, dtype=dtype,
    )
    attn._alpamayo_chunk_k_ready = False
    attn._alpamayo_chunk_k_fine = None
    attn._alpamayo_chunk_k_fine_ready = False
    if hierarchy is not None:
        fine = int(hierarchy[1])
        n_fine = max(N_BLOCK, MAX_SEQ // fine)
        attn._alpamayo_chunk_k_fine = torch.empty(
            1, kvh, n_fine, HEAD_DIM, device=device, dtype=dtype,
        )
    # HuggingFace wants [B, S, H, D] back; write it in-place (no transpose alloc).
    attn._alpamayo_out_bshd = torch.empty(
        1, max_q, hq, HEAD_DIM, device=device, dtype=dtype,
    )
    if attn._alpamayo_dot_kind == DOT_FP8_NATIVE and hasattr(torch, "float8_e4m3fn"):
        fp8 = torch.float8_e4m3fn
        kv_len = int(max_kv or max_q)
        attn._alpamayo_q_fp8 = torch.empty(1, hq, max_q, HEAD_DIM, device=device, dtype=fp8)
        attn._alpamayo_k_fp8 = torch.empty(1, kvh, kv_len, HEAD_DIM, device=device, dtype=fp8)
        attn._alpamayo_v_fp8 = torch.empty(1, kvh, kv_len, HEAD_DIM, device=device, dtype=fp8)
        attn._alpamayo_ck_fp8 = torch.empty(1, kvh, N_BLOCK, HEAD_DIM, device=device, dtype=fp8)


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
    """Holds per-layer buffers and refreshes diffusion chunk means once per frame."""

    def __init__(self, model: Any, topk: int = DEFAULT_TOPK) -> None:
        self.model = model
        self.topk = int(topk)
        self.vlm_attns = _attn_modules(_language_model(model.vlm))
        self.expert_attns = _attn_modules(model.expert)
        if not self.vlm_attns:
            raise RuntimeError("no VLM language-model attention modules found")
        if not self.expert_attns:
            raise RuntimeError("no expert attention modules found")

    def refresh_diffusion_from_cache(self, cache: Any, n_prompt: int) -> None:
        """Recompute fixed prefix key averages from each expert-cache layer."""
        layers = getattr(cache, "layers", None)
        if not layers:
            return
        n_prompt = int(n_prompt)
        for attn in self.expert_attns:
            idx = int(attn.layer_idx)
            if idx < 0 or idx >= len(layers):
                continue
            keys = layers[idx].keys
            vals = layers[idx].values
            # keys: [B, KVH, capacity, D] — only the prefix is averaged.
            rb = int(getattr(attn, "_alpamayo_route_block", CHUNK))
            prompt_chunk_keys(keys, n_prompt=n_prompt, out=attn._alpamayo_chunk_k, chunk=rb)
            attn._alpamayo_n_prompt = n_prompt
            attn._alpamayo_chunk_k_ready = True
            ckf = getattr(attn, "_alpamayo_chunk_k_fine", None)
            hier = getattr(attn, "_alpamayo_hierarchy", None)
            if ckf is not None and hier is not None:
                fill_chunk_keys(keys, ckf, n_prompt, chunk=int(hier[1]))
                attn._alpamayo_chunk_k_fine_ready = True
            else:
                attn._alpamayo_chunk_k_fine_ready = False
            ck8 = getattr(attn, "_alpamayo_ck_fp8", None)
            k8 = getattr(attn, "_alpamayo_k_fp8", None)
            v8 = getattr(attn, "_alpamayo_v_fp8", None)
            if ck8 is not None and k8 is not None and v8 is not None:
                copy_to_fp8(attn._alpamayo_chunk_k, ck8)
                copy_to_fp8(keys[:, :, :n_prompt], k8[:, :, :n_prompt])
                copy_to_fp8(vals[:, :, :n_prompt], v8[:, :, :n_prompt])
                attn._alpamayo_kv_fp8_n = n_prompt


def _patch_cache_load(adapter: AlpamayoRoutedAdapter) -> None:
    """After each prompt copy into the expert cache, refresh chunk means."""
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
    if not targets:
        return

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
            # Graph ctor may reset expert attn to sdpa; put ours back.
            attach_to_alpamayo(expert=adapter.model.expert, topk=adapter.topk)

        decoder_cls.__init__ = __init__


def _patch_expert_eager_refresh(adapter: AlpamayoRoutedAdapter) -> None:
    """Eager diffusion passes the VLM cache as past_key_values — refresh once."""
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
            # Graphed StaticExpertCache is refreshed in set_frame. Here we
            # handle the eager path (VLM DynamicCache as prefix).
            if last_id["cache"] != cid or last_id["n"] != n:
                if not getattr(cache.layers[0], "selected_keys", None):
                    adapter.refresh_diffusion_from_cache(cache, n)
                last_id["cache"] = cid
                last_id["n"] = n
        return orig(*args, past_key_values=past_key_values, **kwargs)

    forward._alpamayo_routed = True
    expert.forward = forward


_E4M3_MAX = 448.0
_RESID_MAX = _E4M3_MAX * 2.0 / 3.0


def _decoder_layers(root: Any) -> List[Any]:
    found: List[Any] = []
    for mod in root.modules():
        if (
            hasattr(mod, "input_layernorm")
            and hasattr(mod, "post_attention_layernorm")
            and hasattr(mod, "self_attn")
            and hasattr(mod, "mlp")
        ):
            found.append(mod)
    return found


def _add_fp8_residual(resid: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Clamp branch to leftover e4m3 room, add, store as e4m3 (round-trip)."""
    room = (_E4M3_MAX - resid.abs()).clamp_min(0)
    summed = resid + branch.clamp(-room, room)
    return summed.to(torch.float8_e4m3fn).to(summed.dtype)


def install_fp8_residual_headroom(model: Any) -> int:
    """Keep residual ≤ 2/3 e4m3 max; layer output ≤ max−|resid|. Graph-safe (no .item)."""
    n = 0
    for lyr in _decoder_layers(model):
        if getattr(lyr.forward, "_alpamayo_fp8_resid", False):
            continue

        def make(lyr):
            def forward(
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                **kwargs,
            ):
                residual = hidden_states.clamp(-_RESID_MAX, _RESID_MAX)
                hidden_states = lyr.input_layernorm(residual)
                attn_out = lyr.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
                branch = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                hidden_states = _add_fp8_residual(residual, branch)
                residual = hidden_states.clamp(-_RESID_MAX, _RESID_MAX)
                hidden_states = lyr.post_attention_layernorm(residual)
                hidden_states = _add_fp8_residual(residual, lyr.mlp(hidden_states))
                return hidden_states

            forward._alpamayo_fp8_resid = True
            return forward

        lyr.forward = make(lyr)
        n += 1
    return n


def adapt_alpamayo(
    instance: Any,
    *,
    topk: int = DEFAULT_TOPK,
    max_seq: int = _MAX_SEQ,
    fp8_residuals: bool = False,
) -> AlpamayoRoutedAdapter:
    """Patch ``instance`` in place and return the adapter (also stored on the model).

    ``instance`` may be ``Alpamayo1_5``, ``AlpamayoOptimizedModelWrapper``, or
    ``FP8OptimizedPredictor``. Safe to call once after load.
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

    hier, topk_c, topk_f, dense_cur = resolve_hga_from_env()
    if hier is not None:
        topk = int(topk_c)

    attach_to_alpamayo(model.vlm, model.expert, topk=topk)
    adapter = AlpamayoRoutedAdapter(model, topk=topk)
    adapter.hierarchy = hier
    adapter.topk_fine = int(topk_f)
    adapter.dense_current = bool(dense_cur)

    for attn in adapter.vlm_attns:
        _preallocate_attn(
            attn, hq=hq, kvh=kvh, max_q=max_seq, device=device, dtype=dtype, topk=topk,
            native_fp8=False, hierarchy=hier, topk_fine=topk_f, dense_current=dense_cur,
        )
    for attn in adapter.expert_attns:
        _preallocate_attn(
            attn, hq=hq, kvh=kvh, max_q=max(n_diff, CHUNK), device=device, dtype=dtype, topk=topk,
            native_fp8=True, max_kv=max_seq, hierarchy=hier, topk_fine=topk_f,
            dense_current=dense_cur,
        )

    _disable_record_attention(instance, model)
    _patch_cache_load(adapter)
    _patch_expert_eager_refresh(adapter)
    if fp8_residuals and hasattr(torch, "float8_e4m3fn"):
        n_res = install_fp8_residual_headroom(model)
        adapter.fp8_residual_layers = n_res
        print(
            f"[routed] fp8 residuals on {n_res} decoder layers "
            f"(resid≤{_RESID_MAX:.1f}, branch≤{_E4M3_MAX:.0f}-|resid|)",
            flush=True,
        )
    else:
        adapter.fp8_residual_layers = 0
    _warmup_kernels(
        device, dtype, hq, kvh, n_diff,
        hierarchy=hier, topk=topk, topk_fine=topk_f, dense_current=dense_cur,
    )
    if hier is not None:
        print(
            f"[routed] two-level {hier[0]}/{hier[1]} topk_c={topk} topk_f={topk_f} "
            f"dense_current={dense_cur} "
            f"vlm_tokens={hga2_vlm_tokens(hier[0], hier[1], topk, topk_f, dense_cur)} "
            f"diff_tokens={hga2_diff_tokens(hier[1], topk, topk_f)}",
            flush=True,
        )

    model._alpamayo_routed_adapter = adapter
    return adapter


def _warmup_kernels(
    device: torch.device,
    dtype: torch.dtype,
    hq: int,
    kvh: int,
    n_diff: int,
    hierarchy=None,
    topk: int = DEFAULT_TOPK,
    topk_fine: int = 2,
    dense_current: bool = False,
) -> None:
    """Compile VLM + diffusion kernels (bf16 + native fp8) at load."""
    from .attention import diffusion_cross_attention, prompt_chunk_keys, vlm_prefill_attention
    from .kernel import copy_to_fp8

    s = 2688
    q = torch.empty(1, hq, s, HEAD_DIM, device=device, dtype=dtype)
    k = torch.empty(1, kvh, s, HEAD_DIM, device=device, dtype=dtype)
    v = torch.empty_like(k)
    vlm_kw = dict(dot_kind=DOT_BF16, topk=topk)
    diff_kw = dict(n_prompt=s, dot_kind=DOT_BF16, topk=topk)
    if hierarchy is not None:
        vlm_kw.update(hierarchy=hierarchy, topk_fine=topk_fine, dense_current=dense_current)
        diff_kw.update(hierarchy=hierarchy, topk_fine=topk_fine)
    vlm_prefill_attention(q, k, v, **vlm_kw)
    qd = torch.empty(1, hq, n_diff, HEAD_DIM, device=device, dtype=dtype)
    kd = torch.empty(1, kvh, s + n_diff, HEAD_DIM, device=device, dtype=dtype)
    vd = torch.empty_like(kd)
    diffusion_cross_attention(qd, kd, vd, **diff_kw)
    if hasattr(torch, "float8_e4m3fn"):
        fp8 = torch.float8_e4m3fn
        q8 = torch.empty_like(q, dtype=fp8)
        k8 = torch.empty_like(k, dtype=fp8)
        v8 = torch.empty_like(v, dtype=fp8)
        copy_to_fp8(q, q8)
        copy_to_fp8(k, k8)
        copy_to_fp8(v, v8)
        ck = prompt_chunk_keys(k, n_prompt=s)
        ck8 = torch.empty_like(ck, dtype=fp8)
        copy_to_fp8(ck, ck8)
        out = torch.empty_like(q)
        vlm_prefill_attention(q8, k8, v8, out=out, chunk_k=ck8, dot_kind=DOT_FP8_NATIVE)
        qd8 = torch.empty_like(qd, dtype=fp8)
        kd8 = torch.empty_like(kd, dtype=fp8)
        vd8 = torch.empty_like(vd, dtype=fp8)
        copy_to_fp8(qd, qd8)
        copy_to_fp8(kd, kd8)
        copy_to_fp8(vd, vd8)
        od = torch.empty_like(qd)
        diffusion_cross_attention(
            qd8, kd8, vd8, n_prompt=s, chunk_k=ck8, out=od, dot_kind=DOT_FP8_NATIVE,
        )
    torch.cuda.synchronize()
    print(
        f"[routed] kernels compiled (S={s} Q={n_diff} hq={hq} kvh={kvh} "
        f"native_fp8={hasattr(torch, 'float8_e4m3fn')})",
        flush=True,
    )
