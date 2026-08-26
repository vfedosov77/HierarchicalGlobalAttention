#!/usr/bin/env python3
"""Unit tests for the API profile gateway's request policy."""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_gateway import (
    OUTPUT_TOKEN_LIMIT,
    UPSTREAM_MODEL,
    apply_profile,
    key_matches,
    read_chunked_body,
    request_api_keys,
)
from deploy import (
    ROOT,
    render_backend_unit,
    render_gateway_unit,
    start_systemd,
    systemd_user_env,
    write_access_point,
)


class ApplyProfileTests(unittest.TestCase):
    def test_tool_requests_are_single_call_turns(self) -> None:
        body = {
            "model": "qwen3.8-27b-hga-fast",
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "parallel_tool_calls": True,
        }

        result = apply_profile(body)

        self.assertEqual(result["model"], UPSTREAM_MODEL)
        self.assertIs(result["parallel_tool_calls"], False)
        self.assertIs(body["parallel_tool_calls"], True)

    def test_non_tool_request_preserves_parallel_control(self) -> None:
        result = apply_profile({
            "model": "qwen3.8-27b-hga-fast",
            "parallel_tool_calls": True,
        })

        self.assertIs(result["parallel_tool_calls"], True)

    def test_copilot_x_api_key_is_accepted(self) -> None:
        key = "a" * 64
        from_bearer = request_api_keys({"Authorization": f"Bearer {key}"}, "/v1/models")
        from_x = request_api_keys({"x-api-key": key}, "/v1/models")
        from_api = request_api_keys({"api-key": key}, "/v1/chat/completions")
        from_query = request_api_keys({}, f"/v1/models?api_key={key}")
        self.assertEqual(from_bearer, [key])
        self.assertEqual(from_x, [key])
        self.assertEqual(from_api, [key])
        self.assertEqual(from_query, [key])
        self.assertTrue(key_matches(key, key))
        self.assertFalse(key_matches("b" * 64, key))
        self.assertFalse(key_matches("short", key))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_profile({"model": "gpt-4o"})

    def test_output_limit_matches_recommended_128k_context(self) -> None:
        self.assertEqual(OUTPUT_TOKEN_LIMIT, 131072)

    def test_output_limit_is_independent_of_reasoning_budget(self) -> None:
        normal = apply_profile({"model": "qwen3.8-27b-hga-normal"})
        deep = apply_profile({"model": "qwen3.8-27b-hga-deep", "max_tokens": 999999})

        self.assertEqual(normal["thinking_budget_tokens"], 512)
        self.assertEqual(deep["thinking_budget_tokens"], 4096)
        self.assertEqual(normal["max_tokens"], OUTPUT_TOKEN_LIMIT)
        self.assertEqual(deep["max_tokens"], OUTPUT_TOKEN_LIMIT)

    def test_smaller_explicit_output_limit_is_preserved(self) -> None:
        result = apply_profile({"model": "qwen3.8-27b-hga-normal", "max_tokens": 37})

        self.assertEqual(result["max_tokens"], 37)

    def test_copilot_max_completion_tokens_maps_to_max_tokens(self) -> None:
        result = apply_profile({
            "model": "qwen3.8-27b-hga-normal",
            "max_completion_tokens": 64,
            "stream": True,
        })

        self.assertEqual(result["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", result)

    def test_chunked_body_round_trip(self) -> None:
        from io import BytesIO

        payload = b'{"model":"qwen3.8-27b-hga-fast"}'
        blob = b"10\r\n" + payload[:16] + b"\r\n" + format(len(payload) - 16, "x").encode() + b"\r\n" + payload[16:] + b"\r\n0\r\n\r\n"
        self.assertEqual(read_chunked_body(BytesIO(blob)), payload)

    def test_api_launcher_does_not_ignore_eos(self) -> None:
        launcher = Path(__file__).with_name("run-api.sh").read_text(encoding="utf-8")
        extra = next(line for line in launcher.splitlines() if line.startswith("export HGA_EXTRA="))

        self.assertNotIn("--ignore-eos", extra)
        self.assertIn("--jinja", extra)
        self.assertNotIn("--api-key", extra)

    def test_api_launcher_uses_fixed_capacity_gpu_prefill(self) -> None:
        launcher = Path(__file__).with_name("run-api.sh").read_text(encoding="utf-8")

        self.assertIn("export HGA_UBATCH=768", launcher)
        self.assertIn("export HGA_PREFILL_UBATCH=768", launcher)
        self.assertIn('export HGA_GPU_PREFILL="${HGA_GPU_PREFILL:-1}"', launcher)
        self.assertIn(
            'export HGA_GPU_PREFILL_MAX_KEYS="${HGA_GPU_PREFILL_MAX_KEYS:-3200}"',
            launcher,
        )
        self.assertIn("export HGA_CTX_CHECKPOINTS=0", launcher)
        self.assertIn('export HGA_LAZY_PREFIX_CACHE="${HGA_LAZY_PREFIX_CACHE:-8}"', launcher)

    def test_stop_local_pins_user_systemd_bus(self) -> None:
        script = Path(__file__).with_name("stop-local.sh").read_text(encoding="utf-8")
        self.assertIn('DBUS_SESSION_BUS_ADDRESS="unix:path=${runtime}/bus"', script)
        self.assertIn("systemctl --user stop", script)
        self.assertIn("hga-qwen38.service", script)
        self.assertIn("hga-qwen38-gateway.service", script)

    def test_generated_units_use_this_tree_not_turing1(self) -> None:
        backend = render_backend_unit(ROOT)
        gateway = render_gateway_unit(ROOT)
        self.assertIn(str(ROOT), backend)
        self.assertIn(str(ROOT / "deployment" / "run-api.sh"), backend)
        self.assertNotIn("Inference/Qwen3_8_27B/", backend.replace(str(ROOT), ""))
        self.assertIn(str(ROOT / "deployment" / "api_gateway.py"), gateway)
        self.assertIn("16 GB", backend)

    def test_access_point_docs_keep_opencode_file_key_literal(self) -> None:
        args = argparse.Namespace(host_address="127.0.0.1", port=8080, ctx=131072)
        write_access_point(args, 16376, 12)
        text = (ROOT / "access_point.md").read_text(encoding="utf-8")
        self.assertIn("`{file:~/.config/hga-qwen38/api-key}`", text)
        self.assertIn("hga-local/qwen3.8-27b-hga-fast", text)


class SystemdUserEnvTests(unittest.TestCase):
    def test_pins_session_bus_to_runtime_dir_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            (runtime / "bus").touch()
            fake_env = {
                "XDG_RUNTIME_DIR": str(runtime),
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/stale-session-bus",
                "PATH": "/usr/bin",
            }
            completed = subprocess.CompletedProcess(
                ["systemctl", "--user", "show", "--property=Version"],
                returncode=0,
                stdout="Version=255\n",
                stderr="",
            )
            with (
                patch.dict(os.environ, fake_env, clear=True),
                patch("deploy.shutil.which", return_value="/usr/bin/systemctl"),
                patch("deploy.subprocess.run", return_value=completed) as run,
            ):
                env = systemd_user_env()
            self.assertIsNotNone(env)
            assert env is not None
            bus = f"unix:path={runtime / 'bus'}"
            self.assertEqual(env["DBUS_SESSION_BUS_ADDRESS"], bus)
            self.assertEqual(env["XDG_RUNTIME_DIR"], str(runtime))
            self.assertEqual(run.call_args.kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"], bus)

    def test_missing_bus_socket_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_env = {"XDG_RUNTIME_DIR": tmp, "PATH": "/usr/bin"}
            with (
                patch.dict(os.environ, fake_env, clear=True),
                patch("deploy.shutil.which", return_value="/usr/bin/systemctl"),
                patch("deploy.os.getuid", return_value=999999999),
            ):
                self.assertIsNone(systemd_user_env())

    def test_unreachable_manager_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            (runtime / "bus").touch()
            fake_env = {"XDG_RUNTIME_DIR": str(runtime), "PATH": "/usr/bin"}
            completed = subprocess.CompletedProcess(
                ["systemctl"],
                returncode=1,
                stdout="unknown\n",
                stderr="Process org.freedesktop.systemd1 exited with status 1\n",
            )
            with (
                patch.dict(os.environ, fake_env, clear=True),
                patch("deploy.shutil.which", return_value="/usr/bin/systemctl"),
                patch("deploy.subprocess.run", return_value=completed),
            ):
                self.assertIsNone(systemd_user_env())

    def test_start_systemd_passes_bus_env_to_systemctl(self) -> None:
        env = {
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus",
            "XDG_RUNTIME_DIR": "/run/user/1",
        }
        with patch("deploy.run") as run:
            start_systemd(env)
        systemctl_calls = [call for call in run.call_args_list if call.args[0][0] == "systemctl"]
        self.assertGreaterEqual(len(systemctl_calls), 1)
        for call in systemctl_calls:
            self.assertEqual(call.kwargs.get("env"), env)


if __name__ == "__main__":
    unittest.main()
