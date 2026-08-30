#!/usr/bin/env python3
"""CPU thread-candidate helpers used by deployment/deploy.py."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_deploy():
    path = ROOT / "deployment" / "deploy.py"
    spec = importlib.util.spec_from_file_location("hga_deploy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ThreadCalibTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.d = load_deploy()

    def test_xeon_18c_36t(self) -> None:
        self.assertEqual(self.d.thread_candidates(18, 36), [6, 12, 18, 24, 30, 36])

    def test_ryzen_6c_12t(self) -> None:
        self.assertEqual(self.d.thread_candidates(6, 12), [6, 12])

    def test_four_core(self) -> None:
        self.assertEqual(self.d.thread_candidates(4, 8), [4, 8])

    def test_twelve_core_smt(self) -> None:
        self.assertEqual(self.d.thread_candidates(12, 24), [6, 12, 18, 24])

    def test_pick_prefers_faster(self) -> None:
        self.assertEqual(self.d.pick_hga_threads([(12, 76.9), (18, 81.6), (24, 45.4)]), 24)

    def test_pick_bias_fewer_within_margin(self) -> None:
        self.assertEqual(self.d.pick_hga_threads([(12, 46.0), (24, 45.4)]), 12)

    def test_omp_places(self) -> None:
        self.assertEqual(self.d.omp_places(12, 18), "cores")
        self.assertEqual(self.d.omp_places(24, 18), "threads")

    def test_pack_fallback_uses_physical_cores(self) -> None:
        self.assertEqual(self.d.default_pack_threads(18), 18)
        self.assertEqual(self.d.default_pack_threads(6), 6)

    def test_pack_candidates_step_four_and_include_endpoint(self) -> None:
        self.assertEqual(
            self.d.pack_thread_candidates(18), [4, 8, 12, 16, 18]
        )
        self.assertEqual(self.d.pack_thread_candidates(6), [4, 6])
        self.assertEqual(self.d.pack_thread_candidates(2), [2])
        self.assertEqual(self.d.pack_thread_candidates(1), [1])

    def test_f16_transport_env_is_persisted(self) -> None:
        args = SimpleNamespace(
            backend_port=8081,
            host_address="127.0.0.1",
            port=8080,
            ctx=131072,
            hga_gpu_prefill=True,
        )
        old_config = self.d.CONFIG_DIR
        try:
            with tempfile.TemporaryDirectory() as td, mock.patch.dict(
                os.environ,
                {
                    "OMP_PLACES": "threads",
                    "HGA_PACK_THREADS": "8",
                    "HGA_F16_TRANSPORT": "1",
                    "HGA_GPU_KV_I8": "1",
                    "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "fp16",
                },
                clear=False,
            ):
                self.d.CONFIG_DIR = Path(td)
                path = self.d.write_api_env(args, "test-key", 24)
                text = path.read_text(encoding="utf-8")
        finally:
            self.d.CONFIG_DIR = old_config
        self.assertIn("HGA_F16_TRANSPORT=1\n", text)
        self.assertIn("HGA_GPU_KV_I8=1\n", text)
        self.assertIn("GGML_CUDA_CUBLAS_COMPUTE_TYPE=fp16\n", text)
        self.assertIn("HGA_PACK_THREADS=8\n", text)


if __name__ == "__main__":
    unittest.main()
