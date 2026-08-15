"""CUDA-event breakdown of one Alpamayo VLM prefill (the ROS ``vlm=`` span).

Attach to ``model.vlm`` and wrap each ``vlm(...)`` call. Enabled from the ROS
node when ``HGA_PROFILE_VLM`` is set.

Leaves are module / kernel spans on the default stream. Nested parents
(``vision``, ``lm_attn``, …) overlap children and are printed only as checks.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Optional


LEAF_ORDER = (
    "vis_patch_embed",
    "vis_pos_embed",
    "vis_rotary_table",
    "vis_norm",
    "vis_qkv",
    "vis_rope",
    "vis_attn_core",
    "vis_o_proj",
    "vis_mlp",
    "vis_merger",
    "vis_deepstack_merge",
    "lm_embed",
    "lm_mrope",
    "lm_rmsnorm",
    "lm_q_proj",
    "lm_k_proj",
    "lm_v_proj",
    "lm_q_norm",
    "lm_k_norm",
    "lm_rope",
    "lm_cache",
    "hga_means",
    "hga_route",
    "hga_attend",
    "lm_o_proj",
    "lm_mlp",
    "lm_deepstack",
    "lm_final_norm",
    "lm_head",
    "rope_index",
)

PARENT_ORDER = (
    "vision",
    "vis_attn",
    "language",
    "lm_attn",
    "lm_layer",
)


class VlmBreakdownProfiler:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self._spans: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self._cpu_ms: dict[str, float] = defaultdict(float)
        self._active = False
        self._depth = 0
        self._handles: list[Any] = []
        self._wrapped = False
        self.frame = 0
        self.meta: dict[str, Any] = {}
        self.history: list[dict[str, float]] = []

    def _ev(self):
        return self.torch.cuda.Event(enable_timing=True)

    def _add(self, name: str, start, end) -> None:
        if self._active:
            self._spans[name].append((start, end))

    def _hook(self, mod, name: str) -> None:
        def pre(_m, _inp, _kw=None):
            if not self._active:
                return
            s = self._ev()
            s.record()
            _m._hga_prof_start = s

        def post(_m, _inp, _out):
            if not self._active:
                return
            s = getattr(_m, "_hga_prof_start", None)
            if s is None:
                return
            e = self._ev()
            e.record()
            self._add(name, s, e)

        self._handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(mod.register_forward_hook(post))

    def _wrap_call(self, fn: Callable, name: str, *, cpu: bool = False) -> Callable:
        torch = self.torch

        def wrapped(*args, **kwargs):
            if not self._active:
                return fn(*args, **kwargs)
            if cpu:
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                self._cpu_ms[name] += (time.perf_counter() - t0) * 1e3
                return out
            s, e = self._ev(), self._ev()
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            self._add(name, s, e)
            return out

        return wrapped

    def attach(self, vlm) -> "VlmBreakdownProfiler":
        if getattr(vlm, "_hga_vlm_profiler", None) is self:
            return self
        visual = getattr(getattr(vlm, "model", None), "visual", None)
        lm = getattr(getattr(vlm, "model", None), "language_model", None)
        if visual is None or lm is None:
            raise RuntimeError("expected vlm.model.visual and vlm.model.language_model")

        self._hook(visual, "vision")
        if hasattr(visual, "patch_embed"):
            self._hook(visual.patch_embed, "vis_patch_embed")
        if hasattr(visual, "pos_embed"):
            self._hook(visual.pos_embed, "vis_pos_embed")
        if hasattr(visual, "rotary_pos_emb"):
            self._hook(visual.rotary_pos_emb, "vis_rotary_table")
        for blk in getattr(visual, "blocks", []):
            if hasattr(blk, "norm1"):
                self._hook(blk.norm1, "vis_norm")
            if hasattr(blk, "norm2"):
                self._hook(blk.norm2, "vis_norm")
            attn = getattr(blk, "attn", None)
            if attn is not None:
                self._hook(attn, "vis_attn")
                if hasattr(attn, "qkv"):
                    self._hook(attn.qkv, "vis_qkv")
                if hasattr(attn, "proj"):
                    self._hook(attn.proj, "vis_o_proj")
            if hasattr(blk, "mlp"):
                self._hook(blk.mlp, "vis_mlp")
        if hasattr(visual, "merger"):
            self._hook(visual.merger, "vis_merger")
        for merger in getattr(visual, "deepstack_merger_list", []):
            self._hook(merger, "vis_deepstack_merge")

        def _vis_pre(_m, args, kwargs):
            if not self._active:
                return
            hs = args[0] if args else kwargs.get("hidden_states")
            grid = kwargs.get("grid_thw")
            if grid is None and len(args) > 1:
                grid = args[1]
            if hs is not None and hasattr(hs, "shape"):
                self.meta["vis_patches"] = int(hs.shape[0])
                self.meta["pixel_shape"] = tuple(hs.shape)
            if grid is not None and hasattr(grid, "detach"):
                g = grid.detach().cpu().tolist()
                self.meta["grid_thw"] = g
                self.meta["n_images"] = len(g)

        self._handles.append(visual.register_forward_pre_hook(_vis_pre, with_kwargs=True))

        self._hook(lm, "language")
        if hasattr(lm, "embed_tokens"):
            self._hook(lm.embed_tokens, "lm_embed")
        if hasattr(lm, "rotary_emb"):
            self._hook(lm.rotary_emb, "lm_mrope")
        if hasattr(lm, "norm"):
            self._hook(lm.norm, "lm_final_norm")
        for layer in getattr(lm, "layers", []):
            self._hook(layer, "lm_layer")
            if hasattr(layer, "input_layernorm"):
                self._hook(layer.input_layernorm, "lm_rmsnorm")
            if hasattr(layer, "post_attention_layernorm"):
                self._hook(layer.post_attention_layernorm, "lm_rmsnorm")
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self._hook(attn, "lm_attn")
                for attr, name in (
                    ("q_proj", "lm_q_proj"),
                    ("k_proj", "lm_k_proj"),
                    ("v_proj", "lm_v_proj"),
                    ("o_proj", "lm_o_proj"),
                    ("q_norm", "lm_q_norm"),
                    ("k_norm", "lm_k_norm"),
                ):
                    if hasattr(attn, attr):
                        self._hook(getattr(attn, attr), name)
            if hasattr(layer, "mlp"):
                self._hook(layer.mlp, "lm_mlp")

        def _lm_pre(_m, args, kwargs):
            if not self._active:
                return
            embeds = kwargs.get("inputs_embeds")
            if embeds is None and args:
                # signature starts with input_ids
                pass
            if embeds is not None and hasattr(embeds, "shape"):
                self.meta["lm_S"] = int(embeds.shape[1])
                self.meta["lm_hidden"] = int(embeds.shape[-1])

        self._handles.append(lm.register_forward_pre_hook(_lm_pre, with_kwargs=True))

        if hasattr(vlm, "lm_head"):
            self._hook(vlm.lm_head, "lm_head")

        self._patch_functions(vlm, visual, lm)
        vlm._hga_vlm_profiler = self
        return self

    def _patch_functions(self, vlm, visual, lm) -> None:
        if self._wrapped:
            return
        self._wrapped = True
        try:
            import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3

            if not getattr(qwen3.apply_rotary_pos_emb, "_hga_prof", False):
                wrapped = self._wrap_call(qwen3.apply_rotary_pos_emb, "lm_rope")
                wrapped._hga_prof = True
                qwen3.apply_rotary_pos_emb = wrapped
            if not getattr(qwen3.apply_rotary_pos_emb_vision, "_hga_prof", False):
                wrapped = self._wrap_call(qwen3.apply_rotary_pos_emb_vision, "vis_rope")
                wrapped._hga_prof = True
                qwen3.apply_rotary_pos_emb_vision = wrapped
        except Exception:
            pass

        inner = getattr(vlm, "model", None)
        if inner is not None and hasattr(inner, "get_rope_index"):
            if not getattr(inner.get_rope_index, "_hga_prof", False):
                wrapped = self._wrap_call(inner.get_rope_index, "rope_index", cpu=True)
                wrapped._hga_prof = True
                inner.get_rope_index = wrapped
        if hasattr(lm, "_deepstack_process") and not getattr(lm._deepstack_process, "_hga_prof", False):
            wrapped = self._wrap_call(lm._deepstack_process, "lm_deepstack")
            wrapped._hga_prof = True
            lm._deepstack_process = wrapped

        try:
            from transformers.cache_utils import DynamicCache

            if not getattr(DynamicCache.update, "_hga_prof", False):
                wrapped = self._wrap_call(DynamicCache.update, "lm_cache")
                wrapped._hga_prof = True
                DynamicCache.update = wrapped
        except Exception:
            pass

        try:
            from SpecializedKernels.Len3000HeadDim128 import kernel as kn
            import SpecializedKernels.Len3000HeadDim128.attention as at

            if not getattr(kn.fill_hga_means, "_hga_prof", False):
                kn.fill_hga_means = self._wrap_call(kn.fill_hga_means, "hga_means")
                kn.fill_hga_means._hga_prof = True
                at.fill_hga_means = kn.fill_hga_means
            if not getattr(kn.chunk_route_vlm_fast, "_hga_prof", False):
                kn.chunk_route_vlm_fast = self._wrap_call(kn.chunk_route_vlm_fast, "hga_route")
                kn.chunk_route_vlm_fast._hga_prof = True
                at.chunk_route_vlm_fast = kn.chunk_route_vlm_fast
            orig_kern = kn._vlm_hga2_kernel
            if not getattr(orig_kern, "_hga_prof", False):
                profiler = self

                class _K:
                    def __getitem__(self, grid):
                        launch = orig_kern[grid]

                        def run(*a, **k):
                            if not profiler._active:
                                return launch(*a, **k)
                            s, e = profiler._ev(), profiler._ev()
                            s.record()
                            out = launch(*a, **k)
                            e.record()
                            profiler._add("hga_attend", s, e)
                            return out

                        return run

                wrapped_k = _K()
                wrapped_k._hga_prof = True
                kn._vlm_hga2_kernel = wrapped_k
                at._vlm_hga2_kernel = wrapped_k
        except Exception:
            pass

        # Vision attention core ≈ vis_attn − qkv − proj − rope (filled in report).
        try:
            from transformers.masking_utils import create_causal_mask

            import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3

            if hasattr(qwen3, "create_causal_mask") and not getattr(
                qwen3.create_causal_mask, "_hga_prof", False
            ):
                wrapped = self._wrap_call(qwen3.create_causal_mask, "causal_mask", cpu=True)
                wrapped._hga_prof = True
                qwen3.create_causal_mask = wrapped
        except Exception:
            pass

    def begin(self) -> None:
        self._depth += 1
        if self._depth > 1:
            return
        self._spans.clear()
        self._cpu_ms.clear()
        self.meta = {}
        self._active = True
        self._wall0 = time.perf_counter()
        self._gpu_s = self._ev()
        self._gpu_s.record()

    def end(self) -> dict[str, float]:
        if self._depth <= 0:
            return {}
        self._depth -= 1
        if self._depth > 0:
            return {}
        gpu_e = self._ev()
        gpu_e.record()
        self.torch.cuda.synchronize()
        self._active = False
        self.frame += 1
        wall_ms = (time.perf_counter() - self._wall0) * 1e3
        gpu_ms = float(self._gpu_s.elapsed_time(gpu_e))

        ms: dict[str, float] = {"wall": wall_ms, "gpu": gpu_ms}
        for name, evs in self._spans.items():
            ms[name] = sum(s.elapsed_time(e) for s, e in evs)
        for name, val in self._cpu_ms.items():
            ms[f"cpu_{name}"] = val

        vis_attn = ms.get("vis_attn", 0.0)
        vis_known = ms.get("vis_qkv", 0.0) + ms.get("vis_o_proj", 0.0) + ms.get("vis_rope", 0.0)
        ms["vis_attn_core"] = max(0.0, vis_attn - vis_known)

        lm_attn = ms.get("lm_attn", 0.0)
        lm_attn_parts = sum(
            ms.get(k, 0.0)
            for k in (
                "lm_q_proj",
                "lm_k_proj",
                "lm_v_proj",
                "lm_q_norm",
                "lm_k_norm",
                "lm_rope",
                "lm_cache",
                "hga_means",
                "hga_route",
                "hga_attend",
                "lm_o_proj",
            )
        )
        ms["lm_attn_rest"] = max(0.0, lm_attn - lm_attn_parts)
        ms["lm_qkv"] = ms.get("lm_q_proj", 0.0) + ms.get("lm_k_proj", 0.0) + ms.get("lm_v_proj", 0.0)
        ms["lm_qk_norm"] = ms.get("lm_q_norm", 0.0) + ms.get("lm_k_norm", 0.0)
        ms["hga_total"] = ms.get("hga_means", 0.0) + ms.get("hga_route", 0.0) + ms.get("hga_attend", 0.0)

        vis_leaves = (
            "vis_patch_embed",
            "vis_pos_embed",
            "vis_rotary_table",
            "vis_norm",
            "vis_qkv",
            "vis_rope",
            "vis_attn_core",
            "vis_o_proj",
            "vis_mlp",
            "vis_merger",
            "vis_deepstack_merge",
        )
        lm_leaves = (
            "lm_embed",
            "lm_mrope",
            "lm_rmsnorm",
            "lm_qkv",
            "lm_qk_norm",
            "lm_rope",
            "lm_cache",
            "hga_total",
            "lm_o_proj",
            "lm_attn_rest",
            "lm_mlp",
            "lm_deepstack",
            "lm_final_norm",
            "lm_head",
        )
        ms["vis_leaves"] = sum(ms.get(k, 0.0) for k in vis_leaves)
        ms["lm_leaves"] = sum(ms.get(k, 0.0) for k in lm_leaves)
        ms["vis_unaccounted"] = max(0.0, ms.get("vision", 0.0) - ms["vis_leaves"])
        # language_model does not include lm_head.
        ms["lm_unaccounted"] = max(
            0.0,
            ms.get("language", 0.0) - (ms["lm_leaves"] - ms.get("lm_head", 0.0)),
        )
        accounted = ms.get("vision", 0.0) + ms.get("language", 0.0) + ms.get("lm_head", 0.0)
        ms["top_unaccounted"] = max(0.0, gpu_ms - accounted)
        self.history.append(ms)
        self._print(ms)
        return ms

    def _emit(self, text: str) -> None:
        print(text, flush=True)
        try:
            print(text, file=sys.stderr, flush=True)
        except Exception:
            pass
        try:
            with open("/tmp/vlm_prof.log", "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            pass

    def _print(self, ms: dict[str, float]) -> None:
        meta = self.meta
        grid = meta.get("grid_thw")
        grid_s = ""
        if grid:
            # compact: unique (t,h,w) × count
            from collections import Counter

            c = Counter(tuple(x) for x in grid)
            grid_s = " ".join(f"{k}×{n}" for k, n in c.items())
        self._emit(
            f"[vlm-prof] frame={self.frame} wall={ms['wall']:.1f}ms gpu={ms['gpu']:.1f}ms "
            f"S={meta.get('lm_S', '?')} vis_patches={meta.get('vis_patches', '?')} "
            f"images={meta.get('n_images', '?')} grid={grid_s or '?'}"
        )

        def line(label: str, key: str, indent: int = 0, extra: str = "") -> None:
            val = ms.get(key, 0.0)
            if val < 0.05 and key not in ("lm_mlp", "vision", "language"):
                return
            pct = 100.0 * val / ms["gpu"] if ms["gpu"] else 0.0
            self._emit(f"[vlm-prof]   {'  ' * indent}{label:<22} {val:7.2f} ms  ({pct:5.1f}%){extra}")

        line("VISION", "vision")
        line("patch_embed", "vis_patch_embed", 1)
        line("pos_embed", "vis_pos_embed", 1)
        line("rotary_table", "vis_rotary_table", 1)
        line("LayerNorm", "vis_norm", 1)
        line("qkv GEMM", "vis_qkv", 1)
        line("RoPE (fp32)", "vis_rope", 1)
        line("attn core (FA/SDPA)", "vis_attn_core", 1)
        line("o_proj", "vis_o_proj", 1)
        line("MLP", "vis_mlp", 1)
        line("merger", "vis_merger", 1)
        line("deepstack merger", "vis_deepstack_merge", 1)
        line("vision unaccounted", "vis_unaccounted", 1)

        line("LANGUAGE", "language")
        line("embed_tokens", "lm_embed", 1)
        line("M-RoPE table", "lm_mrope", 1)
        line("RMSNorm", "lm_rmsnorm", 1)
        line("q/k/v proj", "lm_qkv", 1)
        line("q/k RMSNorm", "lm_qk_norm", 1)
        line("RoPE apply", "lm_rope", 1)
        line("KV cache write", "lm_cache", 1)
        line("HGA means+route+attn", "hga_total", 1)
        extra = ""
        if ms.get("hga_total", 0) > 0.05:
            extra = (
                f"  [means {ms.get('hga_means', 0):.2f}  "
                f"route {ms.get('hga_route', 0):.2f}  "
                f"attend {ms.get('hga_attend', 0):.2f}]"
            )
        if extra:
            self._emit(f"[vlm-prof]     {'HGA parts':<22}{extra}")
        line("o_proj", "lm_o_proj", 1)
        line("attn rest (reshape)", "lm_attn_rest", 1)
        line("MLP (gate/up/down)", "lm_mlp", 1)
        line("deepstack add", "lm_deepstack", 1)
        line("final RMSNorm", "lm_final_norm", 1)
        line("language unaccounted", "lm_unaccounted", 1)
        line("lm_head", "lm_head")
        line("top-level other", "top_unaccounted")
        if ms.get("cpu_rope_index", 0) > 0.05:
            self._emit(
                f"[vlm-prof]   {'get_rope_index (CPU)':<22} {ms['cpu_rope_index']:7.2f} ms  (host, inside wall)"
            )
        if ms.get("cpu_causal_mask", 0) > 0.05:
            self._emit(
                f"[vlm-prof]   {'causal_mask (CPU)':<22} {ms['cpu_causal_mask']:7.2f} ms  (host, inside wall)"
            )

        self._emit(
            f"[vlm-prof] summary  lm_mlp={ms.get('lm_mlp', 0):.1f}  "
            f"vision={ms.get('vision', 0):.1f}  "
            f"lm_qkv+o={ms.get('lm_qkv', 0) + ms.get('lm_o_proj', 0):.1f}  "
            f"hga={ms.get('hga_total', 0):.1f}  "
            f"vis_mlp={ms.get('vis_mlp', 0):.1f}  "
            f"vis_attn={ms.get('vis_attn', 0):.1f}  "
            f"lm_rest={ms.get('language', 0) - ms.get('lm_mlp', 0):.1f}  "
            f"gpu={ms['gpu']:.1f}  wall={ms['wall']:.1f}"
        )


def attach_vlm_profiler(vlm) -> VlmBreakdownProfiler:
    existing = getattr(vlm, "_hga_vlm_profiler", None)
    if isinstance(existing, VlmBreakdownProfiler):
        return existing
    return VlmBreakdownProfiler().attach(vlm)


def profile_enabled() -> bool:
    val = os.environ.get("HGA_PROFILE_VLM", "")
    return val not in ("", "0", "false", "False")
