#!/usr/bin/env python3
"""Unit tests for the 16 GB local full-model speed harness (no GGUF load)."""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "tools" / "bench.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("hga_turing1_bench", BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCH_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BenchParseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = load_bench()

    def test_parse_llama_perf_lines(self) -> None:
        sample = (
            "llama_perf_context_print:        load time =   12345.67 ms\n"
            "llama_perf_context_print: prompt eval time =   10204.12 ms /  2001 tokens"
            " (    5.10 ms per token,   196.10 tokens per second)\n"
            "llama_perf_context_print:        eval time =    5289.26 ms /    64 runs"
            "   (   82.64 ms per token,    12.10 tokens per second)\n"
            "llama_perf_context_print:       total time =   15500.00 ms /  2065 tokens\n"
        )
        perf = self.bench.parse_llama_perf(sample)
        self.assertEqual(perf["load_ms"], 12345.67)
        self.assertEqual(perf["prefill_tokens"], 2001)
        self.assertEqual(perf["prefill_ms"], 10204.12)
        self.assertEqual(perf["prefill_tok_s"], 196.10)
        self.assertEqual(perf["generate_tokens"], 64)
        self.assertEqual(perf["generate_ms"], 5289.26)
        self.assertEqual(perf["generate_tok_s"], 12.10)

    def test_prompt_eval_is_not_generate(self) -> None:
        sample = (
            "llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens"
            " (10.00 ms per token, 100.00 tokens per second)\n"
        )
        perf = self.bench.parse_llama_perf(sample)
        self.assertEqual(perf["prefill_tokens"], 10)
        self.assertIsNone(perf["generate_tok_s"])

    def test_failure_line_detects_oom(self) -> None:
        line = self.bench.failure_line(
            "ggml_gallocr_alloc_graph: failed to allocate CUDA0 buffer of size 123\n"
        )
        self.assertIsNotNone(line)

    def test_prompt_targets_two_thousand_tokens(self) -> None:
        text = self.bench.build_prompt()
        self.assertEqual(text.count(self.bench.PROMPT_SENTENCE), 71)
        self.assertGreater(len(text), 4000)

    def test_optimized_hga_kernel_is_deterministic_default(self) -> None:
        args = self.bench.build_parser().parse_args([])
        self.assertEqual(args.hga_kernel, "tiled")
        self.assertEqual(args.hga_stream_block, 0)
        self.assertEqual(args.hga_verify_streams, 2)
        self.assertFalse(args.hga_stream_sync)
        self.assertEqual(args.hga_stream_chunk_mib, 0)
        self.assertTrue(args.hga_stream_paced)
        self.assertFalse(args.hga_prefill_k_tiles)
        self.assertFalse(args.hga_prefill_stream_async)
        self.assertFalse(args.hga_prefill_stream_paced)
        grouped = self.bench.build_parser().parse_args(["--hga-stream-block", "3"])
        self.assertEqual(grouped.hga_stream_block, 3)
        streamed = self.bench.build_parser().parse_args(["--hga-verify-streams", "3"])
        self.assertEqual(streamed.hga_verify_streams, 3)
        unpaced = self.bench.build_parser().parse_args(["--no-hga-stream-paced"])
        self.assertFalse(unpaced.hga_stream_paced)
        prefill_k = self.bench.build_parser().parse_args(
            ["--no-hga-prefill-k-tiles"]
        )
        self.assertFalse(prefill_k.hga_prefill_k_tiles)
        prefill_async = self.bench.build_parser().parse_args(
            ["--hga-prefill-stream-async"]
        )
        self.assertTrue(prefill_async.hga_prefill_stream_async)
        prefill_paced = self.bench.build_parser().parse_args(
            ["--hga-prefill-stream-paced"]
        )
        self.assertTrue(prefill_paced.hga_prefill_stream_paced)
        env = {
            "HGA_THREADS": "2",
            "HGA_THREADS_BATCH": "80",
            "HGA_VERIFY_BATCH": "0",
            "HGA_VERIFY_TILES": "0",
            "HGA_VERIFY_ROWS": "1",
        }
        self.bench.configure_benchmark_hga_env(env, args.hga_kernel)
        self.assertEqual(env["HGA_THREADS"], os.environ.get("HGA_THREADS", "12"))
        self.assertEqual(env["HGA_THREADS_BATCH"], "1")
        self.assertNotIn("HGA_VERIFY_K_TILES", env)
        self.assertEqual(env["HGA_L2_OFF"], "1")
        self.assertEqual(env["HGA_LOAD_MODE"], "none")
        self.assertEqual(env["HGA_VERIFY_BATCH"], "1")
        self.assertEqual(env["HGA_VERIFY_TILES"], "1")
        self.assertEqual(env["HGA_VERIFY_ROWS"], "0")
        self.assertEqual(env["HGA_STREAM_ASYNC"], "1")

        self.bench.configure_benchmark_hga_env(env, "fused")
        self.assertEqual(env["HGA_VERIFY_BATCH"], "1")
        self.assertEqual(env["HGA_VERIFY_TILES"], "0")
        self.assertEqual(env["HGA_VERIFY_ROWS"], "0")

    def test_pin_log_prefill_then_decode_leftover(self) -> None:
        sample = """
hga-swap: PREFILL  exchange 8 pairs  lm_head CUDA  in 245.3 ms
hga-swap: VERIFY stream 2/8 pairs; 6 leftover pairs deferred until after drop-prefill  free=2494 MiB
hga-swap: PIN-STEP summary  pinned 6/6 leftover pairs; stream fallback=0; no unrecovered OOM
hga-pin: census after leftover VERIFY pins  phase=VERIFY  tensors=851  GPU=850/15345.7MiB  CUDA_Host=1/682.0MiB  CPU=0/0.0MiB  PUSHED=0  MOVE=140 (resident 0)
hga-pin:   resident  626 GPU /   0 host  (n=626)
hga-pin:   exchange  224 GPU /   0 host  (n=224)
hga-pin:   host-ok     0 GPU /   1 host  (n=1)
hga-pin:   lm_head  buf=CUDA0  CUDA  994.6 MiB  staged=1
"""
        pins = self.bench.parse_pin_log(sample)
        self.assertEqual(pins["prefill_pairs"], 8)
        self.assertTrue(pins["lm_head_cuda"])
        self.assertEqual(pins["decode_stream_pairs"], 2)
        self.assertEqual(pins["leftover_pinned"], 6)
        self.assertEqual(pins["leftover_total"], 6)
        self.assertEqual(pins["leftover_pushed"], 0)
        self.assertEqual(pins["leftover_resident_host"], 0)
        self.assertEqual(pins["leftover_exchange_host"], 0)
        baseline = self.bench.load_baseline()
        self.assertEqual(self.bench.evaluate_pin_gates(pins, baseline), [])
        host_head = self.bench.parse_pin_log(
            sample.replace("lm_head CUDA", "lm_head CUDA") +
            "\noutput.weight still on host — should be CUDA-resident\n"
        )
        self.assertTrue(self.bench.evaluate_pin_gates(host_head, baseline))

    def test_one_graph_per_mode_allows_reuse(self) -> None:
        sample = (
            "hga: ubatch=512 graphs_disabled=1\n"
            "hga-swap: STEP 06 ubatch before alloc_graph         "
            "free=1534 used=14610  n_tok=512 n_ubatch=512 n_omax=4096 n_rs=0 phase=1 ctx=0\n"
            "hga-swap: STEP 08 ubatch before graph_compute       "
            "n_tok=512 n_ubatch=512 n_omax=4096 n_rs=0 phase=1 ctx=0\n"
            "hga-swap: STEP 14 ubatch before graph_compute       "
            "n_tok=512 n_ubatch=512 n_omax=4096 n_rs=0 phase=1 ctx=0\n"
            "hga-swap: dropping prefill graphs\n"
            "hga-swap: STEP 29 ubatch before alloc_graph         "
            "free=730 used=15414  n_tok=3 n_ubatch=3 n_omax=4096 n_rs=2 phase=3 ctx=0\n"
            "hga-swap: STEP 31 ubatch before graph_compute       "
            "n_tok=3 n_ubatch=3 n_omax=4096 n_rs=2 phase=3 ctx=0\n"
            "hga-swap: STEP 35 ubatch before graph_compute       "
            "n_tok=3 n_ubatch=3 n_omax=4096 n_rs=2 phase=3 ctx=0\n"
            "llama_perf_context_print:    graphs reused =         22\n"
        )
        graphs = self.bench.parse_graph_log(sample)
        self.assertEqual(graphs["prefill_builds"], 1)
        self.assertEqual(graphs["decode_builds"], 1)
        self.assertEqual(graphs["prefill_extra_rebuilds"], 0)
        self.assertEqual(graphs["decode_extra_rebuilds"], 0)
        self.assertEqual(graphs["graphs_reused"], 22)
        self.assertTrue(graphs["graphs_disabled"])
        self.assertEqual(
            self.bench.evaluate_graph_gates(graphs, self.bench.load_baseline()), []
        )

    def test_extra_prefill_alloc_graph_fails(self) -> None:
        sample = (
            "hga: graphs_disabled=1\n"
            "ubatch before alloc_graph  n_tok=512 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=465 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=464 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=463 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=462 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=461 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=460 n_ubatch=512 n_omax=1 n_rs=0 phase=1 ctx=0\n"
            "ubatch before alloc_graph  n_tok=3 n_ubatch=3 n_omax=1 n_rs=2 phase=3 ctx=0\n"
        )
        graphs = self.bench.parse_graph_log(sample)
        errs = self.bench.evaluate_graph_gates(graphs, self.bench.load_baseline())
        self.assertTrue(any("PREFILL graph built" in e for e in errs), errs)

    def test_rebuild_error_is_extracted(self) -> None:
        stream = (
            "copied hga.{h,cpp}\n"
            "  already patched: CMakeLists.txt\n"
            "error: cuda VMM pool free() not found in /home/vladimir/HGA/"
            "Inference/Qwen3_8_27B/third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu\n"
        )
        errs = self.bench.extract_remote_errors(stream)
        self.assertTrue(any("cuda VMM pool free() not found" in e for e in errs), errs)
        stream2 = "hga-pin-plan: PREFILL\nerror: invalid argument: -no-cnv\n"
        errs2 = self.bench.extract_remote_errors(stream2)
        self.assertTrue(any("invalid argument: -no-cnv" in e for e in errs2), errs2)
        cmake = (
            "CMake Error at CMakeLists.txt:12 (message):\n"
            "  nvcc not found\n"
            "make: *** [all] Error 1\n"
            "HGA_SETUP_FAIL: line 34 exited 1\n"
        )
        cmake_errs = self.bench.extract_remote_errors(cmake)
        self.assertTrue(any("nvcc not found" in e for e in cmake_errs), cmake_errs)
        self.assertTrue(any("Error 1" in e for e in cmake_errs), cmake_errs)
        self.assertTrue(any("HGA_SETUP_FAIL" in e for e in cmake_errs), cmake_errs)
        tb = (
            "Traceback (most recent call last):\n"
            "  File \"scripts/apply_hga.py\", line 16, in <module>\n"
            "    main()\n"
            "RuntimeError: cuda VMM pool free() not found\n"
        )
        tb_errs = self.bench.extract_remote_errors(tb)
        self.assertTrue(any("RuntimeError" in e for e in tb_errs), tb_errs)

    def test_print_remote_failure_includes_error_description(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        stream = (
            "copied hga.{h,cpp}\n"
            "error: cuda VMM pool free() not found in ggml-cuda.cu\n"
            "HGA_SETUP_FAIL: line 34 exited 1\n"
        )
        buf = StringIO()
        with redirect_stdout(buf):
            self.bench.print_remote_failure(1, stream, rebuilt=True)
        out = buf.getvalue()
        self.assertIn("REMOTE BUILD/LAUNCH FAILED", out)
        self.assertIn("cuda VMM pool free() not found", out)
        self.assertIn("HGA_SETUP_FAIL", out)
        self.assertIn("rebuild (setup.sh", out)
        self.assertNotIn("prefill : None tokens", out)

    def test_oom_alloc_dump_last_twenty(self) -> None:
        sample = (
            "hga-alloc-oom: BEGIN why=pair-24-56\n"
            "hga-alloc-oom: last20\n"
            "hga-alloc-oom: 11  module=stream-slot  who=0-32  mib=245.94  "
            "free_before=3222  free_after=2976  ok=1\n"
            "hga-alloc-oom: 12  module=leftover-pin  who=24-56  mib=457.07  "
            "free_before=744  free_after=744  ok=0\n"
            "hga-alloc-oom: live by module\n"
            "hga-alloc-oom:  module=lm_head  n=1  mib=994.6\n"
            "hga-alloc-oom:  module=leftover-pin  n=4  mib=1793.5\n"
            "hga-alloc-oom: END\n"
        )
        dump = self.bench.parse_oom_alloc_dump(sample)
        self.assertEqual(dump["why"], "pair-24-56")
        self.assertEqual(len(dump["events"]), 2)
        self.assertEqual(dump["events"][-1]["who"], "24-56")
        self.assertEqual(dump["events"][-1]["ok"], 0)
        self.assertEqual(dump["live"][0]["module"], "lm_head")
        fallback = self.bench.parse_oom_alloc_dump(
            "hga-swap: PIN-STEP 1/5 leftover pair 4-36  need=457.07 MiB  free=3000 MiB\n"
            "hga-swap: PIN-STEP 1/5 leftover pair 4-36  CUDA alloc 457.07 MiB ok  free_after=2542 MiB\n"
            "hga-swap: PIN-STEP 5/5 leftover pair 24-56  need=457.07 MiB  free=744 MiB\n"
        )
        self.assertGreaterEqual(len(fallback["events"]), 2)
        self.assertEqual(fallback["events"][-1]["module"], "leftover-pin")
        self.assertEqual(fallback["events"][-1]["ok"], 0)

    def test_spec_accept_counters(self) -> None:
        sample = (
            "decoded 67 tokens\n"
            "n_drafted = 33\n"
            "n_accept  = 30\n"
            "accept    = 90.909%\n"
            "hga-prof spec TOTAL steps=13  generate=12090.0 ms (1.08 steps/s)  "
            "draft=1.0 (0%) verify=1.0 (0%) process=1.0 (0%) sample=1.0 (0%) rest=1.0 (0%)\n"
            "hga-prof spec step 1: wall=92.0 ms  n_draft=2 n_acc=2 n_tgt_batch=3\n"
        )
        spec = self.bench.parse_spec_log(sample)
        self.assertTrue(spec["used"])
        self.assertEqual(spec["n_drafted"], 33)
        self.assertEqual(spec["n_accept"], 30)
        self.assertAlmostEqual(spec["accept_pct"], 90.909)
        self.assertEqual(spec["verify_steps"], 13)
        self.assertEqual(spec["decoded"], 67)
        self.assertAlmostEqual(spec["generate_ms"], 12090.0)
        self.assertEqual(self.bench.evaluate_spec_gates(spec, self.bench.load_baseline()), [])
        perf = {"generate_tok_s": None}
        self.bench.fill_perf_from_spec(perf, spec)
        self.assertEqual(perf["generate_tokens"], 67)
        self.assertAlmostEqual(perf["generate_tok_s"], round(1000.0 * 67 / 12090.0, 2))
        missing = self.bench.parse_spec_log("prompt eval time = 1 ms / 10 tokens")
        errs = self.bench.evaluate_spec_gates(missing, self.bench.load_baseline())
        self.assertTrue(any("speculative decoding was not used" in e for e in errs), errs)

    def test_measure_line_round_trip(self) -> None:
        perf = {
            "prefill_tokens": 2001,
            "prefill_ms": 7004.95,
            "prefill_tok_s": 285.66,
            "generate_tokens": 63,
            "generate_ms": 9635.17,
            "generate_tok_s": 6.54,
        }
        line = self.bench.format_measure_line(perf)
        parsed = self.bench.parse_measure_line("ssh noise\n" + line + "\nmore")
        self.assertEqual(parsed["prefill_tokens"], 2001)
        self.assertEqual(parsed["prefill_tok_s"], 285.66)
        self.assertEqual(parsed["generate_tokens"], 63)
        self.assertEqual(parsed["generate_tok_s"], 6.54)

    def test_ssh_run_quotes_compound_commands(self) -> None:
        remote = "cd ~/HGA/Inference/Qwen3_8_27B && chmod +x tools/bench.py"
        cmd = self.bench.ssh_argv("vladimir@host", remote)
        self.assertEqual(cmd[-1].count("cd ~/HGA"), 1)
        self.assertIn("chmod +x tools/bench.py", cmd[-1])
        self.assertTrue(cmd[-1].startswith("bash -lc "))
        self.assertNotEqual(cmd[-1], "bash -lc cd ~/HGA/Inference/Qwen3_8_27B && chmod +x tools/bench.py")

    def test_password_is_read_from_secrets_md(self) -> None:
        path = ROOT / "secrets.md"
        if not path.is_file():
            self.skipTest("secrets.md is not shipped with the 16 GB tree")
        password = self.bench.read_turing1_password()
        self.assertTrue(password)
        self.assertNotIn("\n", password)
        self.assertIn(password, path.read_text(encoding="utf-8"))
        self.assertEqual(self.bench.secrets_path(), path)

    def test_gates_against_checked_in_baseline(self) -> None:
        baseline = self.bench.load_baseline()
        good = {
            "prefill_tokens": 2001,
            "prefill_tok_s": 196.0,
            "generate_tokens": 64,
            "generate_tok_s": 12.1,
        }
        self.assertEqual(self.bench.evaluate_gates(good, baseline), [])
        slow = dict(good, prefill_tok_s=10.0)
        self.assertTrue(self.bench.evaluate_gates(slow, baseline))
        short = dict(good, prefill_tokens=100)
        self.assertTrue(self.bench.evaluate_gates(short, baseline))


if __name__ == "__main__":
    unittest.main(verbosity=2)
