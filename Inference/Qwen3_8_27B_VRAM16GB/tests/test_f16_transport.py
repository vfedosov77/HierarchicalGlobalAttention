#!/usr/bin/env python3
"""Static contract tests for the opt-in HGA F16 activation transport."""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class F16TransportContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.glue = (ROOT / "llama.cpp-hga" / "llama-hga.cpp").read_text(
            encoding="utf-8"
        )
        cls.runner = (ROOT / "deployment" / "run-api.sh").read_text(
            encoding="utf-8"
        )

    def test_transport_is_opt_in(self) -> None:
        self.assertIn('std::getenv("HGA_F16_TRANSPORT")', self.glue)
        self.assertIn('${HGA_F16_TRANSPORT:-0}', self.runner)

    def test_f16_wire_restores_f32_for_quantized_matmul(self) -> None:
        self.assertIn("src->type == GGML_TYPE_F16", self.glue)
        self.assertIn("ggml_cast(gctx->ctx0, cpy, GGML_TYPE_F32)", self.glue)

    def test_routing_accepts_f16_wire_but_stays_f32(self) -> None:
        self.assertIn("hga_stage_as_f32", self.glue)
        self.assertIn("pack_heads_f32", self.glue)
        self.assertIn("ggml_cpu_fp16_to_fp32", self.glue)
        self.assertIn("ggml_cpu_fp32_to_fp16", self.glue)
        self.assertIn("hga_prepare_gpu_prefill_i8_strided", self.glue)

    def test_large_f16_wire_is_prefill_only(self) -> None:
        self.assertIn("phase == HGA_SWAP_PREFILL", self.glue)
        self.assertIn("gctx->cparams.hga_phase == HGA_SWAP_PREFILL", self.glue)


if __name__ == "__main__":
    unittest.main()
