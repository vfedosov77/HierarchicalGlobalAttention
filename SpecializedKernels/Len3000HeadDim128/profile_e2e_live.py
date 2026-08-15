"""End-to-end Alpamayo frame timing: decode / prep / VLM / diffusion / publish.

Installed from ``adapt_alpamayo`` when ``HGA_PROFILE_VLM`` or ``HGA_PROFILE_E2E``
is set. CUDA-graph replay is timed as a block; one eager expert step is probed
so the 10 Euler steps can be attributed (hooks do not fire inside a graph).
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Optional


def _enabled() -> bool:
    for key in ("HGA_PROFILE_E2E", "HGA_PROFILE_VLM"):
        if os.environ.get(key, "") not in ("", "0", "false", "False"):
            return True
    return False


def _emit(text: str) -> None:
    print(text, flush=True)
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        with open("/tmp/e2e_prof.log", "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


class _SpanBag:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.spans: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self.active = False

    def ev(self):
        return self.torch.cuda.Event(enable_timing=True)

    def hook(self, mod, name: str) -> None:
        def pre(_m, *_a, **_k):
            if not self.active:
                return
            s = self.ev()
            s.record()
            _m._e2e_s = s

        def post(_m, *_a, **_k):
            if not self.active:
                return
            s = getattr(_m, "_e2e_s", None)
            if s is None:
                return
            e = self.ev()
            e.record()
            self.spans[name].append((s, e))

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    def wrap(self, fn: Callable, name: str) -> Callable:
        def wrapped(*a, **k):
            if not self.active:
                return fn(*a, **k)
            s, e = self.ev(), self.ev()
            s.record()
            out = fn(*a, **k)
            e.record()
            self.spans[name].append((s, e))
            return out

        return wrapped

    def ms(self) -> dict[str, float]:
        out = {}
        for name, evs in self.spans.items():
            out[name] = sum(s.elapsed_time(e) for s, e in evs)
        return out


class E2EProfiler:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.frame = 0
        self.last: dict[str, float] = {}
        self.expert_probe: dict[str, float] | None = None
        self.num_steps = 10
        self.graph_used = False
        self._expert_bag = _SpanBag()
        self._expert_hooked = False

    def hook_expert(self, model) -> None:
        if self._expert_hooked:
            return
        bag = self._expert_bag
        if hasattr(model, "action_in_proj"):
            bag.hook(model.action_in_proj, "in_proj")
        if hasattr(model, "action_out_proj"):
            bag.hook(model.action_out_proj, "out_proj")
        expert = model.expert
        bag.hook(expert, "expert")
        for layer in getattr(expert, "layers", []):
            if hasattr(layer, "self_attn"):
                bag.hook(layer.self_attn, "attn")
                for attr, name in (
                    ("q_proj", "q_proj"),
                    ("k_proj", "k_proj"),
                    ("v_proj", "v_proj"),
                    ("o_proj", "o_proj"),
                ):
                    if hasattr(layer.self_attn, attr):
                        bag.hook(getattr(layer.self_attn, attr), name)
            if hasattr(layer, "mlp"):
                bag.hook(layer.mlp, "mlp")
            if hasattr(layer, "input_layernorm"):
                bag.hook(layer.input_layernorm, "rmsnorm")
            if hasattr(layer, "post_attention_layernorm"):
                bag.hook(layer.post_attention_layernorm, "rmsnorm")
        try:
            from SpecializedKernels.Len3000HeadDim128 import kernel as kn
            import SpecializedKernels.Len3000HeadDim128.attention as at

            orig = kn._diff_hga2_vec_kernel
            if not getattr(orig, "_e2e_prof", False):
                profiler = self

                class _K:
                    def __getitem__(self, grid):
                        launch = orig[grid]

                        def run(*a, **k):
                            if not profiler._expert_bag.active:
                                return launch(*a, **k)
                            s, e = profiler._expert_bag.ev(), profiler._expert_bag.ev()
                            s.record()
                            out = launch(*a, **k)
                            e.record()
                            profiler._expert_bag.spans["hga"].append((s, e))
                            return out

                        return run

                wrapped = _K()
                wrapped._e2e_prof = True
                kn._diff_hga2_vec_kernel = wrapped
                at._diff_hga2_vec_kernel = wrapped
        except Exception:
            pass
        self._expert_hooked = True

    def probe_expert_step(self, decoder) -> None:
        """One eager Euler step with hooks. Safe after ``set_frame``."""
        if self.expert_probe is not None:
            return
        self.hook_expert(decoder.model)
        bag = self._expert_bag
        bag.spans.clear()
        bag.active = True
        x = decoder.x0
        t = decoder.time_steps[0].view(1, *([1] * len(decoder.x_dims))).expand_as(x)
        try:
            decoder._step(x, t)
        except Exception as exc:
            _emit(f"[e2e-prof] expert probe failed: {type(exc).__name__}: {exc}")
            bag.active = False
            return
        self.torch.cuda.synchronize()
        bag.active = False
        ms = bag.ms()
        attn_parts = ms.get("q_proj", 0) + ms.get("k_proj", 0) + ms.get("v_proj", 0) + ms.get("o_proj", 0) + ms.get("hga", 0)
        ms["attn_rest"] = max(0.0, ms.get("attn", 0) - attn_parts)
        ms["qkv_o"] = ms.get("q_proj", 0) + ms.get("k_proj", 0) + ms.get("v_proj", 0) + ms.get("o_proj", 0)
        self.expert_probe = ms
        self.num_steps = int(getattr(decoder, "num_steps", 10))

    def print_frame(self, stats: dict[str, Any], decode: dict[str, float] | None = None) -> None:
        self.frame += 1
        def msec(key: str, default: float = 0.0) -> float:
            if key in stats and stats[key] is not None:
                try:
                    v = float(stats[key])
                except (TypeError, ValueError):
                    return default
                return v * 1e3 if v < 10 else v
            return default

        # stats values are seconds for *_sec keys.
        prep = msec("img_sec") + msec("tokenize_sec") + msec("rope_gap_sec") + msec("message_sec") + msec("template_sec") + msec("h2d_sec")
        if prep == 0:
            prep = msec("prep_sec")
        fuse = msec("fuse_sec")
        vlm = msec("vlm_sec")
        diff = msec("diffusion_sec")
        post = msec("post_sec")
        predict = msec("node_predict_sec")
        if predict == 0:
            predict = prep + fuse + vlm + diff + post
        dec = (decode or {})
        jpeg = dec.get("jpeg_ms", 0.0)
        crop = dec.get("crop_ms", 0.0)
        tensor = dec.get("tensor_ms", 0.0)
        echo = dec.get("echo_ms", 0.0)
        decode_ms = dec.get("decode_ms", jpeg + crop + tensor + echo)
        publish = msec("node_publish_sec")
        total = msec("node_total_sec")
        if total == 0:
            total = decode_ms + predict + publish

        set_frame = stats.get("diff_set_frame_ms", 0.0) or 0.0
        replay = stats.get("diff_replay_ms", 0.0) or 0.0
        eager = stats.get("diff_eager_ms", 0.0) or 0.0
        traj = stats.get("diff_traj_ms", 0.0) or 0.0
        mode = stats.get("diff_mode", "?")

        _emit(
            f"[e2e-prof] frame={self.frame} total={total:.1f}ms  "
            f"decode={decode_ms:.1f} predict={predict:.1f} publish={publish:.1f}  "
            f"diff={mode}"
        )
        _emit(f"[e2e-prof]   DECODE                 {decode_ms:7.1f} ms")
        if decode_ms:
            _emit(f"[e2e-prof]     jpeg                 {jpeg:7.1f}")
            _emit(f"[e2e-prof]     crop/remap           {crop:7.1f}")
            _emit(f"[e2e-prof]     HWC→CHW tensor       {tensor:7.1f}")
            if echo:
                _emit(f"[e2e-prof]     model-input echo     {echo:7.1f}")
        _emit(f"[e2e-prof]   PREP                   {prep:7.1f} ms")
        _emit(f"[e2e-prof]     img processor        {msec('img_sec'):7.1f}")
        _emit(f"[e2e-prof]     tokenize             {msec('tokenize_sec'):7.1f}")
        if msec("rope_gap_sec"):
            _emit(f"[e2e-prof]     rope_gap             {msec('rope_gap_sec'):7.1f}")
        if fuse:
            _emit(f"[e2e-prof]   FUSE                   {fuse:7.1f} ms")
        _emit(f"[e2e-prof]   VLM                    {vlm:7.1f} ms")
        _emit(f"[e2e-prof]   DIFFUSION              {diff:7.1f} ms  ({mode}, steps={self.num_steps})")
        if set_frame:
            _emit(f"[e2e-prof]     set_frame (KV copy)  {set_frame:7.1f}")
        if replay:
            _emit(f"[e2e-prof]     CUDA-graph replay    {replay:7.1f}")
        if eager:
            _emit(f"[e2e-prof]     eager euler          {eager:7.1f}")
        if traj:
            _emit(f"[e2e-prof]     action_to_traj       {traj:7.1f}")
        probe = self.expert_probe
        if probe and self.num_steps:
            n = self.num_steps
            _emit(
                f"[e2e-prof]     one expert step      {probe.get('expert', 0) + probe.get('in_proj', 0) + probe.get('out_proj', 0):7.2f}  "
                f"× {n} ≈ {(probe.get('expert', 0) + probe.get('in_proj', 0) + probe.get('out_proj', 0)) * n:.1f} ms"
            )
            _emit(f"[e2e-prof]       in_proj            {probe.get('in_proj', 0):7.2f}  ×{n} = {probe.get('in_proj', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       expert MLP         {probe.get('mlp', 0):7.2f}  ×{n} = {probe.get('mlp', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       expert QKV+O       {probe.get('qkv_o', 0):7.2f}  ×{n} = {probe.get('qkv_o', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       expert HGA attn    {probe.get('hga', 0):7.2f}  ×{n} = {probe.get('hga', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       expert attn rest   {probe.get('attn_rest', 0):7.2f}  ×{n} = {probe.get('attn_rest', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       expert RMSNorm     {probe.get('rmsnorm', 0):7.2f}  ×{n} = {probe.get('rmsnorm', 0) * n:6.1f}")
            _emit(f"[e2e-prof]       out_proj           {probe.get('out_proj', 0):7.2f}  ×{n} = {probe.get('out_proj', 0) * n:6.1f}")
        if post:
            _emit(f"[e2e-prof]   POST                   {post:7.1f} ms")
        _emit(f"[e2e-prof]   PUBLISH                {publish:7.1f} ms")
        _emit(
            f"[e2e-prof] summary  decode={decode_ms:.0f} prep={prep:.0f} vlm={vlm:.0f} "
            f"diff={diff:.0f} publish={publish:.0f} total={total:.0f}"
        )


def install_e2e_profiler(instance: Any, model: Any) -> Optional[E2EProfiler]:
    if not _enabled():
        return None
    if getattr(model, "_hga_e2e_profiler", None) is not None:
        return model._hga_e2e_profiler

    prof = E2EProfiler()
    model._hga_e2e_profiler = prof
    import torch

    # --- GraphedTrajectoryDecoder.set_frame / run / capture ---
    try:
        from alpamayo_ros.cuda_graph_diffusion import GraphedTrajectoryDecoder
    except Exception:
        try:
            from alpamayo_ros.alpamayo_ros.cuda_graph_diffusion import GraphedTrajectoryDecoder
        except Exception as exc:
            _emit(f"[e2e-prof] cannot import GraphedTrajectoryDecoder: {exc}")
            GraphedTrajectoryDecoder = None

    if GraphedTrajectoryDecoder is not None and not getattr(
        GraphedTrajectoryDecoder.set_frame, "_e2e_prof", False
    ):
        orig_set = GraphedTrajectoryDecoder.set_frame
        orig_run = GraphedTrajectoryDecoder.run
        orig_capture = GraphedTrajectoryDecoder.capture

        def set_frame(self, *a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_set(self, *a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._e2e_set_frame_ms = (time.perf_counter() - t0) * 1e3
            try:
                prof.probe_expert_step(self)
            except Exception as exc:
                _emit(f"[e2e-prof] probe after set_frame failed: {exc}")
            return out

        def run(self, *a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_run(self, *a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._e2e_replay_ms = (time.perf_counter() - t0) * 1e3
            prof.graph_used = True
            prof.num_steps = int(getattr(self, "num_steps", 10))
            return out

        def capture(self, *a, **k):
            _emit("[e2e-prof] capturing CUDA graph for diffusion")
            return orig_capture(self, *a, **k)

        set_frame._e2e_prof = True
        GraphedTrajectoryDecoder.set_frame = set_frame
        GraphedTrajectoryDecoder.run = run
        GraphedTrajectoryDecoder.capture = capture

    # --- predictor.predict: stash diffusion sub-times + print ---
    pred = None
    for obj in (instance, model, getattr(instance, "_predictor", None), getattr(model, "_predictor", None)):
        if obj is not None and hasattr(obj, "predict") and hasattr(obj, "_decode_trajectory"):
            pred = obj
            break
    if pred is not None and not getattr(pred.predict, "_e2e_prof", False):
        orig_pred = pred.predict
        orig_decode = pred._decode_trajectory

        def decode_traj(*a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_decode(*a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pred._e2e_decode_traj_ms = (time.perf_counter() - t0) * 1e3
            graphed = getattr(pred, "_graphed", None)
            pred._e2e_diff_bits = {
                "diff_mode": "cuda-graph" if graphed is not None and getattr(graphed, "graph", None) is not None else "eager",
                "diff_set_frame_ms": float(getattr(graphed, "_e2e_set_frame_ms", 0.0) or 0.0) if graphed else 0.0,
                "diff_replay_ms": float(getattr(graphed, "_e2e_replay_ms", 0.0) or 0.0) if graphed else 0.0,
                "diff_traj_ms": 0.0,
            }
            return out

        def predict(*a, **k):
            result = orig_pred(*a, **k)
            stats = result.get("stats") if isinstance(result, dict) else None
            if stats is None:
                return result
            bits = getattr(pred, "_e2e_diff_bits", {})
            stats.update(bits)
            if bits.get("diff_mode") == "eager":
                stats["diff_eager_ms"] = float(getattr(pred, "_e2e_decode_traj_ms", 0.0) or 0.0)
            return result

        decode_traj._e2e_prof = True
        predict._e2e_prof = True
        pred._decode_trajectory = decode_traj
        pred.predict = predict
        _emit("[e2e-prof] hooked predictor.predict + diffusion")

    return prof
