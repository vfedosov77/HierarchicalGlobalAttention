#!/usr/bin/env python3
"""Install the HGA OpenAI-compatible AccessPoint on a 16 GB CUDA GPU host.

Default is local install on this machine (OpenCode / GitHub Copilot).
Use --remote only to push to another host over SSH.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path.home() / ".config" / "hga-qwen38"
USER_SYSTEMD = Path.home() / ".config" / "systemd" / "user"
MIN_VRAM_MIB = 15000
HGA_CALIB_REPS = 8
HGA_CALIB_CTX = 8192
HGA_CALIB_UBATCH = 768
HGA_CALIB_MARGIN = 0.05
HGA_ROUTE_MEASURE_RE = re.compile(
    r"HGA_ROUTE_MEASURE prefill_ms_per_layer=([\d.]+)"
)


def run(command: list[str], **kw) -> subprocess.CompletedProcess:
    print("==> " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kw)


def gpu_memory_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    sizes = []
    for line in (out.stdout or "").splitlines():
        try:
            sizes.append(int(float(line.strip())))
        except ValueError:
            continue
    return max(sizes) if sizes else 0


def require_16gb_gpu() -> int:
    mib = gpu_memory_mib()
    if mib < MIN_VRAM_MIB:
        raise SystemExit(
            f"need a CUDA GPU with at least ~16 GiB; nvidia-smi reports {mib} MiB"
        )
    return mib


def _nvidia_smi_query(fields: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]


def gpu_compute_caps() -> list[str]:
    """CMake-style archs: nvidia-smi 8.9 / 12.0 -> 89 / 120."""
    caps: list[str] = []
    seen: set[str] = set()
    for line in _nvidia_smi_query("compute_cap"):
        digits = line.replace(".", "").replace(" ", "")
        if digits.isdigit() and digits not in seen:
            seen.add(digits)
            caps.append(digits)
    return caps


def cuda_arch_hint(arch: str) -> str:
    if arch.startswith(("120", "121", "100", "101", "103")):
        return "CUDA 12.8+ (Blackwell)"
    if arch.startswith("90"):
        return "CUDA 12.0+"
    if arch.startswith("89"):
        return "CUDA 11.8+"
    if arch.startswith(("87", "86")):
        return "CUDA 11.1+"
    if arch.startswith("80"):
        return "CUDA 11.0+"
    if arch.startswith("75"):
        return "CUDA 10.0+"
    return f"a CUDA toolkit that supports sm_{arch}"


def _push_nvcc(found: list[Path], seen: set[str], path: Path | None) -> None:
    if path is None:
        return
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return
    key = str(resolved)
    if key in seen:
        return
    seen.add(key)
    found.append(resolved)


def iter_nvcc_candidates() -> list[Path]:
    """Same places scripts/select_cuda.sh looks (PATH, CUDA_HOME, /usr/local/cuda*)."""
    found: list[Path] = []
    seen: set[str] = set()
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        _push_nvcc(found, seen, Path(home) / "bin" / "nvcc")
    for env_name in ("CUDACXX", "CMAKE_CUDA_COMPILER"):
        value = os.environ.get(env_name)
        if value:
            _push_nvcc(found, seen, Path(value))
    which = shutil.which("nvcc")
    if which:
        _push_nvcc(found, seen, Path(which))
    roots = [
        Path("/usr/local/cuda"),
        Path("/usr/lib/cuda"),
        Path("/usr/lib/nvidia-cuda-toolkit"),
        Path("/opt/cuda"),
        Path.home() / "opt" / "cuda",
    ]
    for parent in (Path("/usr/local"), Path("/opt"), Path.home() / "opt"):
        try:
            roots.extend(sorted(parent.glob("cuda-*")))
        except OSError:
            continue
    for root in roots:
        _push_nvcc(found, seen, root / "bin" / "nvcc")
    return found


def nvcc_version(nvcc: Path) -> str:
    try:
        out = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    marker = "release "
    for line in (out.stdout or "").splitlines():
        idx = line.lower().find(marker)
        if idx < 0:
            continue
        rest = line[idx + len(marker) :]
        return rest.split(",")[0].split()[0].strip()
    return ""


def missing_nvcc_message() -> str:
    lines = [
        "error: no nvcc found. Install the CUDA toolkit (matching this GPU) or set CUDA_HOME.",
        "A CPU-only llama-server cannot load this 27B GGUF on 16 GB (CPU repack abort).",
        "",
        "This machine:",
    ]
    gpus = _nvidia_smi_query("name,memory.total,compute_cap")
    if gpus:
        lines.extend(f"  GPU: {row}" for row in gpus)
    else:
        lines.append("  GPU: nvidia-smi not found or failed")
    for cap in gpu_compute_caps():
        lines.append(f"  sm_{cap} typically needs {cuda_arch_hint(cap)}.")
    lines.extend(
        [
            "",
            "Install a CUDA toolkit that matches this GPU (toolkit only — keep the",
            "NVIDIA driver already providing nvidia-smi), then re-run.",
            "  https://developer.nvidia.com/cuda-downloads",
            "If nvcc is already installed outside PATH, set CUDA_HOME (and optionally",
            "CUDAHOSTCXX) and re-run. Debian/Ubuntu /usr/bin/nvcc is often an older",
            "distro package; a newer toolkit in /usr/local/cuda-* is OK.",
        ]
    )
    return "\n".join(lines)


def ensure_nvcc(skip_build: bool) -> None:
    """Fail fast if nvcc is missing. CUDA toolkit is a host prerequisite, not installed here.

    Skip when llama-server is already built or --skip-build was passed.
    """
    if skip_build or llama_server_bin().is_file():
        return
    print("==> checking for nvcc (CUDA toolkit matching this GPU)", flush=True)
    found = iter_nvcc_candidates()
    if not found:
        raise SystemExit(missing_nvcc_message())
    nvcc = found[0]
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        preferred = Path(home).expanduser() / "bin" / "nvcc"
        if preferred.is_file():
            nvcc = preferred.resolve()
    version = nvcc_version(nvcc)
    extra = f" {version}" if version else ""
    print(f"==> nvcc{extra}: {nvcc}", flush=True)
    toolkit = nvcc.parent.parent
    os.environ.setdefault("CUDA_HOME", str(toolkit))
    os.environ.setdefault("CUDACXX", str(nvcc))
    bin_dir = str(nvcc.parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def logical_cpus() -> int:
    return max(1, os.cpu_count() or 1)


def physical_cores() -> int:
    """Unique physical cores, matching scripts/env.sh _hga_physical_cores."""
    try:
        out = subprocess.run(
            ["lscpu", "-p=CORE"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = None
    if out and out.returncode == 0:
        cores = {
            ln.strip()
            for ln in out.stdout.splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        if cores:
            return max(1, len(cores))
    cpuids: set[tuple[str, str]] = set()
    physical: str | None = None
    core: str | None = None
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return logical_cpus()
    for line in text.splitlines():
        if line.startswith("physical id"):
            physical = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
        elif not line.strip():
            if physical is not None and core is not None:
                cpuids.add((physical, core))
            physical = core = None
    if physical is not None and core is not None:
        cpuids.add((physical, core))
    return max(1, len(cpuids)) if cpuids else logical_cpus()


def cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def cpu_fingerprint(physical: int, logical: int) -> str:
    return f"{cpu_model()} phys={physical} log={logical}"


def omp_places(threads: int, physical: int) -> str:
    return "threads" if threads > physical else "cores"


def thread_candidates(physical: int, logical: int) -> list[int]:
    """Sweep step=min(6, physical cores) up to all logical CPUs (nproc)."""
    physical = max(1, physical)
    logical = max(physical, logical)
    step = min(6, physical)
    found: set[int] = set()
    n = step
    while n <= logical:
        found.add(n)
        n += step
    found.add(physical)
    found.add(logical)
    return sorted(x for x in found if 1 <= x <= logical)


def pick_hga_threads(rows: list[tuple[int, float]]) -> int:
    """Lowest prefill route ms; within 5% prefer fewer workers."""
    if not rows:
        raise ValueError("no thread measurements")
    best_ms = min(ms for _, ms in rows)
    within = [t for t, ms in rows if ms <= best_ms * (1.0 + HGA_CALIB_MARGIN)]
    return min(within)


def detect_threads() -> int:
    env = os.environ.get("HGA_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return physical_cores()


def route_bench_bin() -> Path:
    return ROOT / "build" / "hga-route-bench"


def ensure_route_bench() -> Path:
    binary = route_bench_bin()
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    build = ROOT / "build"
    cpp = ROOT / "cpp"
    print("==> building hga-route-bench (CPU-only HGA routing microbench)", flush=True)
    run(["cmake", "-S", str(cpp), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(build), "-j", str(logical_cpus()), "--target", "hga-route-bench"])
    if not binary.is_file():
        raise SystemExit(f"missing {binary} after cmake build")
    return binary


def run_route_bench(threads: int, physical: int) -> float:
    binary = route_bench_bin()
    env = os.environ.copy()
    env["HGA_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = omp_places(threads, physical)
    env["OMP_PROC_BIND"] = "close"
    proc = subprocess.run(
        [
            str(binary),
            "--threads",
            str(threads),
            "--ctx",
            str(HGA_CALIB_CTX),
            "--ubatch",
            str(HGA_CALIB_UBATCH),
            "--reps",
            str(HGA_CALIB_REPS),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(ROOT),
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    match = HGA_ROUTE_MEASURE_RE.search(text)
    if proc.returncode != 0 or match is None:
        tail = "\n".join(text.splitlines()[-20:])
        raise RuntimeError(
            f"hga-route-bench --threads {threads} failed (exit {proc.returncode})\n{tail}"
        )
    return float(match.group(1))


def write_thread_calibration(payload: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "cpu_threads.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    threads = int(payload["hga_threads"])
    places = payload.get("omp_places") or omp_places(threads, int(payload["physical_cores"]))
    env_text = (
        f"HGA_THREADS={threads}\n"
        f"OMP_NUM_THREADS={threads}\n"
        f"OMP_PLACES={places}\n"
        f"OMP_PROC_BIND=close\n"
    )
    path = CONFIG_DIR / "cpu_threads.env"
    path.write_text(env_text, encoding="utf-8")
    path.chmod(0o644)


def _print_cached_thread_measurements() -> None:
    path = CONFIG_DIR / "cpu_threads.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    rows = payload.get("measurements") or []
    if not rows:
        return
    print("    prior sweep (prefill route ms/layer):", flush=True)
    for row in rows:
        mark = "  <-- selected" if int(row.get("threads", -1)) == int(payload.get("hga_threads", -1)) else ""
        print(
            f"      threads={row.get('threads')}: {row.get('prefill_ms_per_layer')} ms{mark}",
            flush=True,
        )
    print(
        f"    OMP_PLACES={payload.get('omp_places')}  "
        f"(re-run python3 deployment/deploy.py --recalibrate to sweep again)",
        flush=True,
    )


def load_thread_calibration(fingerprint: str) -> int | None:
    path = CONFIG_DIR / "cpu_threads.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("cpu") != fingerprint:
        return None
    try:
        return max(1, int(payload["hga_threads"]))
    except (KeyError, TypeError, ValueError):
        return None


def apply_hga_thread_env(threads: int, physical: int) -> None:
    places = omp_places(threads, physical)
    os.environ["HGA_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PLACES"] = places
    os.environ["OMP_PROC_BIND"] = "close"


def calibrate_hga_threads(*, force: bool = False) -> int:
    """Sweep HGA_THREADS and keep the fastest prefill-route width.

    Skips when the user already exported HGA_THREADS. Caches under
    ~/.config/hga-qwen38/cpu_threads.json keyed by CPU model + core counts.
    """
    physical = physical_cores()
    logical = logical_cpus()
    fingerprint = cpu_fingerprint(physical, logical)
    override = os.environ.get("HGA_THREADS")
    if override and not force:
        try:
            threads = max(1, int(override))
        except ValueError:
            threads = physical
        else:
            print(
                f"==> HGA_THREADS={threads} already set; skipping CPU calibration",
                flush=True,
            )
            apply_hga_thread_env(threads, physical)
            return threads
    if not force:
        cached = load_thread_calibration(fingerprint)
        if cached is not None:
            print(
                f"==> using cached HGA_THREADS={cached} for {fingerprint}",
                flush=True,
            )
            _print_cached_thread_measurements()
            apply_hga_thread_env(cached, physical)
            return cached

    candidates = thread_candidates(physical, logical)
    print(
        f"==> calibrating HGA_THREADS on {fingerprint}: candidates {candidates}",
        flush=True,
    )
    try:
        ensure_route_bench()
    except (subprocess.CalledProcessError, SystemExit) as exc:
        print(f"warning: could not build hga-route-bench ({exc}); using {physical} cores", flush=True)
        apply_hga_thread_env(physical, physical)
        return physical

    rows: list[tuple[int, float]] = []
    for n in candidates:
        try:
            ms = run_route_bench(n, physical)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"  threads={n}: FAIL {exc}", flush=True)
            continue
        rows.append((n, ms))
        print(f"  threads={n}: prefill route {ms:.2f} ms/layer", flush=True)
    if not rows:
        print(f"warning: all calibration runs failed; using {physical} cores", flush=True)
        apply_hga_thread_env(physical, physical)
        return physical
    winner = pick_hga_threads(rows)
    places = omp_places(winner, physical)
    payload = {
        "cpu": fingerprint,
        "physical_cores": physical,
        "logical_cpus": logical,
        "hga_threads": winner,
        "omp_places": places,
        "metric": "prefill_ms_per_layer",
        "margin": HGA_CALIB_MARGIN,
        "measurements": [
            {"threads": t, "prefill_ms_per_layer": round(ms, 3)} for t, ms in rows
        ],
    }
    write_thread_calibration(payload)
    apply_hga_thread_env(winner, physical)
    print(
        f"==> selected HGA_THREADS={winner} OMP_PLACES={places} "
        f"(prefill route {dict(rows)[winner]:.2f} ms/layer)",
        flush=True,
    )
    return winner


def python_bin() -> str:
    return str(Path(sys.executable).resolve())


def llama_cpp_dir() -> Path:
    env = os.environ.get("HGA_LLAMA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "third_party" / "llama.cpp"


def llama_server_bin() -> Path:
    return llama_cpp_dir() / "build" / "bin" / "llama-server"


def gguf_path() -> Path:
    env = os.environ.get("HGA_MODEL")
    if env:
        return Path(env).expanduser()
    stamp = ROOT / "third_party" / "gguf.path"
    if stamp.is_file():
        text = stamp.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return Path.home() / "models" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q4_K_M.gguf"


def ensure_gguf() -> None:
    """Resolve weights via setup.sh (local lookup, download only if missing)."""
    env = os.environ.copy()
    env["HGA_SETUP_ONLY"] = "gguf"
    print("==> resolving GGUF via scripts/setup.sh (skip download if already on this host)", flush=True)
    run(["bash", str(ROOT / "scripts" / "setup.sh")], env=env)
    stamp = ROOT / "third_party" / "gguf.path"
    if stamp.is_file():
        found = stamp.read_text(encoding="utf-8").strip()
        if found:
            os.environ["HGA_MODEL"] = found
    path = gguf_path()
    try:
        ok = path.is_file() and path.stat().st_size > 1_000_000_000
    except OSError:
        ok = False
    if not ok:
        raise SystemExit(
            f"missing GGUF; put Qwen3.8-27B-UD-Q4_K_M.gguf in the current directory, "
            f"{ROOT}, or ~/models/ (or set HGA_MODEL)"
        )


def ensure_llama_server(skip_build: bool) -> None:
    """Run scripts/setup.sh when llama-server is not built yet.

    deploy.py is the user-facing entry point; setup.sh is the clone/patch/build
    helper it invokes. CUDA arch is detected inside setup.sh from nvidia-smi.
    """
    server = llama_server_bin()
    if server.is_file():
        if skip_build:
            return
        source_roots = (ROOT / "llama.cpp-hga", ROOT / "cpp")
        inputs = [ROOT / "scripts" / "apply_hga.py"]
        for source_root in source_roots:
            inputs.extend(
                path
                for path in source_root.rglob("*")
                if path.is_file() and path.suffix in {".c", ".cc", ".cpp", ".cu", ".h", ".hpp"}
            )
        server_mtime = server.stat().st_mtime_ns
        if not any(path.stat().st_mtime_ns > server_mtime for path in inputs):
            return
        print("==> HGA sources changed; rebuilding llama-server", flush=True)
    if skip_build:
        raise SystemExit(f"missing {server}; run scripts/setup.sh")
    reason = "sources changed" if server.is_file() else "not found"
    print(
        f"==> llama-server {reason}; running scripts/setup.sh "
        "(clone llama.cpp, apply HGA, build for this machine's GPU(s))",
        flush=True,
    )
    run(["bash", str(ROOT / "scripts" / "setup.sh")])
    if not server.is_file():
        raise SystemExit(f"scripts/setup.sh finished but {server} is still missing")


def unit_template(name: str) -> Path:
    path = ROOT / "deployment" / name
    if not path.is_file():
        raise SystemExit(f"missing systemd unit template {path}")
    return path


def write_systemd_units() -> None:
    """Install portable unit templates. Paths come from HGA_ROOT in api.env."""
    USER_SYSTEMD.mkdir(parents=True, exist_ok=True)
    for name in ("hga-qwen38.service", "hga-qwen38-gateway.service"):
        shutil.copyfile(unit_template(name), USER_SYSTEMD / name)


def write_api_env(args: argparse.Namespace, key: str, threads: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "api.env"
    backend = f"http://127.0.0.1:{args.backend_port}"
    lines = [
        f"HGA_API_KEY={key}",
        f"HGA_API_HOST={args.host_address}",
        f"HGA_API_PORT={args.port}",
        f"HGA_BACKEND_PORT={args.backend_port}",
        f"HGA_BACKEND_URL={backend}",
        f"HGA_ROOT={ROOT}",
        f"HGA_THREADS={threads}",
        f"OMP_NUM_THREADS={threads}",
        f"OMP_PLACES={os.environ.get('OMP_PLACES', omp_places(threads, physical_cores()))}",
        "OMP_PROC_BIND=close",
        f"HGA_CTX={args.ctx}",
        f"HGA_GPU_PREFILL={1 if args.hga_gpu_prefill else 0}",
        "HGA_GPU_PREFILL_MIN_KEYS=1552",
        "HGA_GPU_PREFILL_MAX_KEYS=2560",
        f"HGA_F16_TRANSPORT={os.environ.get('HGA_F16_TRANSPORT', '0')}",
        f"GGML_CUDA_CUBLAS_COMPUTE_TYPE={os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'auto')}",
        f"HGA_SPEC={os.environ.get('HGA_SPEC', '3')}",
        f"HGA_SPLIT_FFN={os.environ.get('HGA_SPLIT_FFN', '0')}",
        f"HGA_SPLIT_FFN_TILE_CHANNELS={os.environ.get('HGA_SPLIT_FFN_TILE_CHANNELS', '1024')}",
        f"HGA_STREAM_TIMING={os.environ.get('HGA_STREAM_TIMING', '0')}",
        f"HGA_MODEL={os.environ.get('HGA_MODEL', str(Path.home() / 'models' / 'Qwen3.8-27B-GGUF' / 'Qwen3.8-27B-UD-Q4_K_M.gguf'))}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    key_file = CONFIG_DIR / "api-key"
    key_file.write_text(key + "\n", encoding="utf-8")
    key_file.chmod(0o600)
    return path


def _user_runtime_dir() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("XDG_RUNTIME_DIR"):
        candidates.append(Path(os.environ["XDG_RUNTIME_DIR"]))
    candidates.append(Path(f"/run/user/{os.getuid()}"))
    seen: set[str] = set()
    for runtime in candidates:
        key = str(runtime)
        if key in seen:
            continue
        seen.add(key)
        if (runtime / "bus").exists():
            return runtime
    return None


def systemd_user_env() -> dict[str, str] | None:
    """Env that talks to the persistent user systemd, not a stale session bus.

    Desktop and IDE shells often keep DBUS_SESSION_BUS_ADDRESS pointed at a
    GNOME session bus under /tmp. systemctl --user then tries to activate
    org.freedesktop.systemd1 on that bus and fails with:
      Reload daemon failed: Process org.freedesktop.systemd1 exited with status 1
    """
    if shutil.which("systemctl") is None:
        return None
    runtime = _user_runtime_dir()
    if runtime is None:
        return None
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = str(runtime)
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime / 'bus'}"
    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show", "--property=Version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return env


def systemd_user_available() -> bool:
    return systemd_user_env() is not None


def wait_http_ok(url: str, key: str, timeout: int = 300) -> None:
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    last = None
    headers = {"Authorization": f"Bearer {key}"}
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"==> ready {url}", flush=True)
                    return
                last = response.status
        except Exception as exc:  # noqa: BLE001 — poll until llama-server binds
            last = exc
        time.sleep(2)
    log = CONFIG_DIR / "server.log"
    hint = ""
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        abort = [
            ln
            for ln in tail
            if "GGML_ASSERT" in ln
            or "no GPU backend" in ln
            or "failed to load" in ln
            or "missing model" in ln
        ]
        if abort:
            hint = "\n" + "\n".join(abort[-8:])
        hint += f"\nsee {log}"
    raise SystemExit(f"timed out waiting for {url}: {last}{hint}")


def start_systemd(env: dict[str, str]) -> None:
    run(["systemctl", "--user", "daemon-reload"], env=env)
    try:
        run(["loginctl", "enable-linger", os.environ.get("USER", "")], env=env)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("warning: could not enable linger; services may stop at logout", flush=True)
    run(["systemctl", "--user", "enable", "hga-qwen38.service", "hga-qwen38-gateway.service"], env=env)
    run(["systemctl", "--user", "restart", "hga-qwen38.service"], env=env)
    run(["systemctl", "--user", "restart", "hga-qwen38-gateway.service"], env=env)
    run(
        ["systemctl", "--user", "--no-pager", "status", "hga-qwen38.service", "hga-qwen38-gateway.service"],
        env=env,
    )


def restart_local_helper() -> None:
    run(["bash", str(ROOT / "deployment" / "stop-local.sh")])
    run(["bash", str(ROOT / "deployment" / "start-local.sh")])


def public_url(args: argparse.Namespace) -> str:
    host = args.host_address
    if host in {"0.0.0.0", "::"}:
        host = socket.gethostname()
        try:
            host = socket.gethostbyname(host)
        except OSError:
            host = "127.0.0.1"
    return f"http://{host}:{args.port}"


def write_client_examples(base: str) -> None:
    """Write host-specific client snippets under ~/.config only — not the repo."""
    import json

    src = ROOT / "deployment" / "clients"
    dest = CONFIG_DIR / "clients"
    dest.mkdir(parents=True, exist_ok=True)
    opencode = src / "opencode.json"
    if opencode.is_file():
        shutil.copy(opencode, dest / "opencode.json")
    copilot = [
        {
            "name": "HGA Qwen3.8-27B",
            "vendor": "customendpoint",
            "apiType": "chat-completions",
            "apiKey": "${input:hgaApiKey}",
            "models": [
                {
                    "id": "qwen3.8-27b-hga-fast",
                    "name": "HGA Fast",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
                {
                    "id": "qwen3.8-27b-hga-normal",
                    "name": "HGA Normal",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
                {
                    "id": "qwen3.8-27b-hga-deep",
                    "name": "HGA Deep",
                    "url": f"{base}/v1/chat/completions",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 131072,
                    "maxOutputTokens": 131072,
                },
            ],
        }
    ]
    (dest / "chatLanguageModels.json").write_text(
        json.dumps(copilot, indent=2) + "\n", encoding="utf-8"
    )


def write_access_point(args: argparse.Namespace, vram_mib: int, threads: int) -> None:
    """Host-specific notes go under ~/.config only — never into the git tree."""
    base = public_url(args)
    text = f"""# HGA AccessPoint (16 GB CUDA)

OpenAI-compatible API for OpenCode and GitHub Copilot Chat on this machine.

- Base URL: `{base}/v1`
- Chat completions: `{base}/v1/chat/completions`
- Models: `qwen3.8-27b-hga-fast`, `qwen3.8-27b-hga-normal`, `qwen3.8-27b-hga-deep`
- Auth: `Authorization: Bearer $HGA_API_KEY` (file `~/.config/hga-qwen38/api.env`, mode 0600)
- GPU: {vram_mib} MiB  |  HGA threads: {threads}  |  context: {args.ctx}

Do not commit the API key. Source it with:

```bash
set -a; . ~/.config/hga-qwen38/api.env; set +a
```

## Start / stop

From the HGA checkout (the directory that contains `deployment/`):

```bash
# preferred (systemd units and start-local.sh pid files):
./deployment/stop-local.sh
./deployment/start-local.sh

# raw systemctl — pin the persistent user bus. Desktop/IDE shells often keep
# DBUS_SESSION_BUS_ADDRESS pointed at a GNOME session bus under /tmp, which
# makes systemctl --user fail with org.freedesktop.systemd1 exited status 1:
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
systemctl --user start hga-qwen38.service hga-qwen38-gateway.service
systemctl --user stop  hga-qwen38.service hga-qwen38-gateway.service
```

## OpenCode

Merge the `hga-local` provider from `examples/opencode.json` into
`~/.config/opencode/opencode.json`. It talks to `{base}/v1` with
`apiKey` `{{file:~/.config/hga-qwen38/api-key}}` (use an absolute path).
Default model: `hga-local/qwen3.8-27b-hga-fast`. Restart OpenCode after editing.

## GitHub Copilot (VS Code Chat)

1. Command Palette → **Chat: Manage Language Models** → **Add Models** → **Custom Endpoint**.
2. API type: **Chat Completions**.
3. Paste `./deployment/clients/chatLanguageModels.json` (replace the API key prompt).
4. Enable **toolCalling** models for Copilot agent mode.

Copilot CLI:

```bash
export COPILOT_PROVIDER_BASE_URL={base}/v1
export COPILOT_PROVIDER_API_KEY="$HGA_API_KEY"
export COPILOT_MODEL=qwen3.8-27b-hga-fast
```

Inline Copilot completions still use GitHub-hosted models; this endpoint is for Chat / agent.

## Smoke

```bash
HGA_API_KEY="$HGA_API_KEY" python3 ./deployment/smoke.py --url {base}
```
"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "access_point.md").write_text(text, encoding="utf-8")


def install_local(args: argparse.Namespace) -> int:
    key = os.environ.get("HGA_API_KEY")
    if not key:
        raise SystemExit("HGA_API_KEY must be set; refusing to deploy an unauthenticated API")
    if any(c.isspace() for c in key):
        raise SystemExit("HGA_API_KEY may not contain whitespace")
    vram = require_16gb_gpu()
    for script in ROOT.glob("scripts/*.sh"):
        script.chmod(script.stat().st_mode | 0o111)
    for script in ROOT.glob("deployment/*.sh"):
        script.chmod(script.stat().st_mode | 0o111)
    if args.skip_calibrate:
        threads = detect_threads()
        apply_hga_thread_env(threads, physical_cores())
        print(f"==> skip CPU calibration; HGA_THREADS={threads}", flush=True)
    else:
        threads = calibrate_hga_threads(force=args.recalibrate)
    ensure_nvcc(args.skip_build)
    ensure_gguf()
    ensure_llama_server(args.skip_build)
    write_api_env(args, key, threads)
    write_systemd_units()
    write_client_examples(public_url(args) if args.host_address != "0.0.0.0" else f"http://127.0.0.1:{args.port}")
    write_access_point(args, vram, threads)
    print(f"installed AccessPoint files under {CONFIG_DIR}", flush=True)
    print(
        f"HGA_THREADS={threads}  OMP_PLACES={os.environ.get('OMP_PLACES')}  "
        f"(llama-server -t {threads})",
        flush=True,
    )
    if args.no_start:
        print("skipping start (--no-start). Use deployment/start-local.sh or systemctl --user.", flush=True)
        return 0
    systemd_env = None if args.foreground_helper else systemd_user_env()
    if systemd_env is not None:
        try:
            start_systemd(systemd_env)
        except subprocess.CalledProcessError as exc:
            print(
                f"warning: systemd --user failed ({exc}); falling back to deployment/start-local.sh",
                flush=True,
            )
            restart_local_helper()
    else:
        if not args.foreground_helper:
            print(
                "systemd --user is not reachable (stale session bus or no user manager); "
                "starting with deployment/start-local.sh",
                flush=True,
            )
        restart_local_helper()
    if not args.skip_smoke:
        print("==> waiting for llama-server to finish loading (~6-20s)", flush=True)
        wait_http_ok(f"http://127.0.0.1:{args.backend_port}/health", key, timeout=300)
        wait_http_ok(f"http://127.0.0.1:{args.port}/health", key, timeout=60)
        smoke_env = os.environ.copy()
        smoke_env["HGA_API_KEY"] = key
        url = f"http://127.0.0.1:{args.port}"
        run([python_bin(), str(ROOT / "deployment" / "smoke.py"), "--url", url], env=smoke_env)
    print(
        f"AccessPoint ready: {public_url(args)}/v1  (see {CONFIG_DIR / 'access_point.md'})",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", default=True, help="Install on this 16 GB GPU host (default)")
    parser.add_argument("--host-address", default=os.environ.get("HGA_API_HOST", "127.0.0.1"))
    parser.add_argument("--lan", action="store_true", help="Listen on 0.0.0.0 instead of loopback")
    parser.add_argument("--port", type=int, default=int(os.environ.get("HGA_API_PORT", "8080")))
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("HGA_BACKEND_PORT", "8081")))
    parser.add_argument("--ctx", type=int, default=int(os.environ.get("HGA_CTX", "131072")))
    parser.add_argument(
        "--hga-gpu-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU-routing / CUDA-attention prefill (default: enabled)",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Do not sweep HGA_THREADS; use $HGA_THREADS or physical cores",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Ignore the cached CPU thread pick and sweep again",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--no-start", action="store_true", help="Write units and client configs only")
    parser.add_argument("--foreground-helper", action="store_true", help="Use start-local.sh even if systemd exists")
    args = parser.parse_args()
    if args.lan:
        args.host_address = "0.0.0.0"
    return install_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
