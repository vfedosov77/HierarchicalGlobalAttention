#!/usr/bin/env python3
"""Run stock Alpamayo 1.5 with optional HGA. No ROS. No CoT generation.

    PYTHONPATH=/path/to/Alpamayo-ROS/src:$PWD \
      python -m SpecializedKernels.Len3000HeadDim128.integration_example.run_nogener \
        --model /path/to/Alpamayo-1.5-10B --compare
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_KERNELS = Path(__file__).resolve().parents[1]
_HGA = _KERNELS.parents[1]
if str(_HGA) not in sys.path:
    sys.path.insert(0, str(_HGA))

_DEFAULT_ALPAMAYO_SRC = (
    Path(os.environ["ALPAMAYO_SRC"])
    if os.environ.get("ALPAMAYO_SRC")
    else Path("/home/vladimir/Alpamayo-ROS/src")
)
if _DEFAULT_ALPAMAYO_SRC.is_dir() and str(_DEFAULT_ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_ALPAMAYO_SRC))

_DEFAULT_MODEL = os.environ.get(
    "ALPAMAYO_MODEL",
    "/home/vladimir/Alpamayo-ROS/.local_models/Alpamayo-1.5-10B-official-fp8",
)


def _config_with_local_vlm(model_dir: Path):
    import json

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    config_dict = json.loads((model_dir / "config.json").read_text())
    vlm_ref = config_dict.get("vlm_name_or_path")
    if vlm_ref and not os.path.isabs(str(vlm_ref)):
        local_vlm = model_dir / vlm_ref
        if local_vlm.is_dir():
            config_dict["vlm_name_or_path"] = str(local_vlm.resolve())
    return Alpamayo1_5.config_class.from_dict(config_dict)


def _load_alpamayo(model_path: str, device: torch.device):
    """Load stock ``Alpamayo1_5``. Resolve a sibling ``vlm_processor/`` if needed."""
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    root = Path(model_path).expanduser()
    if root.is_dir() and (root / "fp8_manifest.json").is_file():
        ros_pkg = Path(os.environ.get("ALPAMAYO_SRC", "/home/vladimir/Alpamayo-ROS/src")) / "alpamayo_ros"
        if ros_pkg.is_dir() and str(ros_pkg) not in sys.path:
            sys.path.insert(0, str(ros_pkg))
        from alpamayo_ros.model_optimization_wrapper import _load_persistent_fp8_model

        config = _config_with_local_vlm(root)
        model, _manifest = _load_persistent_fp8_model(root, config, device)
        print("loaded persistent FP8 checkpoint", flush=True)
        return model
    if root.is_dir() and (root / "config.json").is_file():
        config = _config_with_local_vlm(root)
        return Alpamayo1_5.from_pretrained(
            str(root), config=config, dtype=torch.bfloat16, local_files_only=True
        ).to(device)
    return Alpamayo1_5.from_pretrained(model_path, dtype=torch.bfloat16).to(device)


def _add_alpamayo_src(explicit: str | None) -> None:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise SystemExit(f"--alpamayo-src is not a directory: {path}")
        sys.path.insert(0, str(path))


def _synthetic_clip(device: torch.device) -> dict:
    """4 cameras × 4 frames of noise + a short ego history. Enough to prefill."""
    n_cam, n_frm, h, w = 4, 4, 384, 768
    frames = torch.randint(0, 255, (n_cam, n_frm, 3, h, w), dtype=torch.uint8)
    hist_t = 16
    xyz = torch.zeros(1, 1, hist_t, 3, dtype=torch.float32)
    xyz[0, 0, :, 0] = torch.linspace(-1.5, 0.0, hist_t)
    rot = torch.eye(3, dtype=torch.float32).expand(1, 1, hist_t, 3, 3).clone()
    return {
        "image_frames": frames,
        "camera_indices": torch.tensor([0, 1, 2, 6], dtype=torch.long),
        "ego_history_xyz": xyz,
        "ego_history_rot": rot,
    }


def _load_clip(path: str | None, device: torch.device) -> dict:
    if path:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        need = ("image_frames", "ego_history_xyz", "ego_history_rot")
        missing = [k for k in need if k not in blob]
        if missing:
            raise SystemExit(f"{path} missing keys {missing}")
        return blob
    print("No --clip-pt; using synthetic 4×4 camera frames.", flush=True)
    return _synthetic_clip(device)


def _prepare(model, processor, clip: dict, device: torch.device) -> dict:
    from alpamayo1_5 import helper

    frames = clip["image_frames"]
    if isinstance(frames, torch.Tensor) and frames.ndim == 5:
        flat = frames.flatten(0, 1)
    else:
        flat = frames
    messages = helper.create_message(
        flat,
        camera_indices=clip.get("camera_indices"),
        num_frames_per_camera=4,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": dict(inputs),
            "ego_history_xyz": clip["ego_history_xyz"],
            "ego_history_rot": clip["ego_history_rot"],
        },
        device,
    )
    return model_inputs


def _one(model, prepared, label: str, warmup: int, iters: int, *, use_cuda_graph: bool) -> dict:
    from SpecializedKernels.Len3000HeadDim128.integration_example.nogener import (
        sample_no_generation,
    )

    print(
        f"\n=== {label}  warmup={warmup} iters={iters} "
        f"{'CUDA-graph Euler' if use_cuda_graph else 'eager Euler'} ===",
        flush=True,
    )
    for _ in range(max(0, warmup)):
        sample_no_generation(model, prepared, time_it=False, use_cuda_graph=use_cuda_graph)
    stats_list = []
    xyz = None
    for i in range(max(1, iters)):
        xyz, _rot, extra = sample_no_generation(
            model, prepared, return_extra=False, use_cuda_graph=use_cuda_graph,
        )
        stats_list.append(extra["stats"])
        print(
            f"  iter {i + 1}/{iters}  S={stats_list[-1]['S']}  "
            f"vlm={stats_list[-1]['vlm_sec'] * 1e3:.0f}ms  "
            f"diff={stats_list[-1]['diffusion_sec'] * 1e3:.0f}ms  "
            f"predict={stats_list[-1]['predict_sec'] * 1e3:.0f}ms",
            flush=True,
        )
    mid = sorted(stats_list, key=lambda s: s["predict_sec"])[len(stats_list) // 2]
    print(
        f"  median  vlm={mid['vlm_sec'] * 1e3:.0f}ms  "
        f"diff={mid['diffusion_sec'] * 1e3:.0f}ms  "
        f"predict={mid['predict_sec'] * 1e3:.0f}ms  "
        f"traj={tuple(xyz.shape) if xyz is not None else '?'}",
        flush=True,
    )
    return mid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=_DEFAULT_MODEL, help="Alpamayo1_5 checkpoint dir or HF id")
    p.add_argument("--alpamayo-src", default=None, help="Path to Alpamayo-ROS/src (for alpamayo1_5)")
    p.add_argument("--clip-pt", default=None, help="Optional torch.save() clip (image_frames + ego hist)")
    p.add_argument("--mode", choices=("hga", "dense", "compare"), default="hga")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument(
        "--fast-vision",
        action="store_true",
        help="Also patch ViT to batched SDPA + bf16 RoPE (not HGA)",
    )
    p.add_argument(
        "--eager",
        action="store_true",
        help="Skip CUDA-graph Euler (ROS-like ~120 ms). Eager is ~300 ms.",
    )
    args = p.parse_args()
    _add_alpamayo_src(args.alpamayo_src)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from SpecializedKernels.Len3000HeadDim128.integration_example.adapter import adapt_alpamayo

    print(f"GPU    {torch.cuda.get_device_name(0)}", flush=True)
    print(f"model  {args.model}", flush=True)
    print(
        "mode   no-generation (VLM prefill only, max_new_tokens=0, no CoT). "
        "HGA is installed on language prefill + expert diffusion. "
        f"diffusion={'eager Euler' if args.eager else 'CUDA-graph Euler (ROS-like)'}.",
        flush=True,
    )

    model = _load_alpamayo(args.model, device)
    model.eval()
    # Official expert is non-causal + a 4D prefix mask. Flash-2 varlen rejects
    # that shape; stock Alpamayo/ROS dense diffusion uses SDPA here.
    expert_cfg = getattr(getattr(model, "expert", None), "config", None)
    if expert_cfg is not None:
        expert_cfg._attn_implementation = "sdpa"
        if hasattr(expert_cfg, "is_causal"):
            expert_cfg.is_causal = False
    processor = helper.get_processor(model.tokenizer)
    prepared = _prepare(model, processor, _load_clip(args.clip_pt, device), device)
    s = prepared["tokenized_data"]["input_ids"].shape[1]
    print(f"prompt tokens S={s}", flush=True)

    results = {}
    if args.mode in ("dense", "compare"):
        results["dense"] = _one(
            model, prepared, "dense SDPA/FA", args.warmup, args.iters,
            use_cuda_graph=not args.eager,
        )
    if args.mode in ("hga", "compare"):
        adapt_alpamayo(model, fast_vision=args.fast_vision)
        results["hga"] = _one(
            model, prepared, "HGA 2L-192", args.warmup, args.iters,
            use_cuda_graph=not args.eager,
        )

    if "dense" in results and "hga" in results:
        d, h = results["dense"], results["hga"]
        print("\n=== HGA vs dense (no generation, same S) ===", flush=True)
        for key, label in (
            ("vlm_sec", "VLM prefill"),
            ("diffusion_sec", "diffusion"),
            ("predict_sec", "predict"),
        ):
            dv, hv = d[key] * 1e3, h[key] * 1e3
            print(f"  {label:<14}  dense {dv:7.1f} ms   HGA {hv:7.1f} ms   Δ {hv - dv:+7.1f} ms", flush=True)


if __name__ == "__main__":
    main()
