#!/usr/bin/env python3
"""apply_hga.py must drop leftover split-FFN streaming from a patched tree."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_apply():
    path = ROOT / "scripts" / "apply_hga.py"
    spec = importlib.util.spec_from_file_location("hga_apply", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CMAKE = """add_library(llama
            llama-graph.cpp
            llama-hga.cpp
            hga.cpp
            hga-weight-swap.cpp
            hga-split-ffn.cpp
            hga_l2.cpp
            hga-kv-gemv.cpp
)
"""

QWEN = r'''#include "models.h"
#include "llama-hga.h"
#include "hga-split-ffn.h"
#include "llama-memory-recurrent.h"

static ggml_tensor * hga_qwen35_build_layer_ffn_split(
        llm_graph_context * gctx, const llama_model & model, ggml_tensor * cur, const int il) {
    (void) model;
    hga_split_ffn * plan = hga_weight_swap_split((hga_weight_swap *) gctx->cparams.hga_swap);
    return plan ? cur : cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    if (hga_decode_pack(cparams.hga_phase) &&
            hga_weight_swap_split_layer((hga_weight_swap *) cparams.hga_swap, il)) {
        return hga_qwen35_build_layer_ffn_split(this, model, cur, il);
    }

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    return cur;
}
'''


class StripSplitFfnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.apply = load_apply()

    def test_strip_removes_callers_cmake_and_leftover_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src" / "models").mkdir(parents=True)
            (root / "src" / "CMakeLists.txt").write_text(CMAKE, encoding="utf-8")
            (root / "src" / "models" / "qwen35.cpp").write_text(QWEN, encoding="utf-8")
            (root / "src" / "hga-split-ffn.cpp").write_text("int x;\n", encoding="utf-8")
            (root / "src" / "hga-split-ffn.h").write_text("#pragma once\n", encoding="utf-8")

            self.apply.strip_split_ffn(root)
            self.apply.strip_split_ffn(root)

            cmake = (root / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
            qwen = (root / "src" / "models" / "qwen35.cpp").read_text(encoding="utf-8")
            self.assertNotIn("hga-split-ffn.cpp", cmake)
            self.assertIn("hga-weight-swap.cpp", cmake)
            self.assertFalse((root / "src" / "hga-split-ffn.cpp").exists())
            self.assertFalse((root / "src" / "hga-split-ffn.h").exists())
            self.assertNotIn("hga-split-ffn.h", qwen)
            self.assertNotIn("hga_qwen35_build_layer_ffn_split", qwen)
            self.assertNotIn("hga_weight_swap_split", qwen)
            self.assertIn("cur = build_ffn(cur,", qwen)


if __name__ == "__main__":
    unittest.main()
