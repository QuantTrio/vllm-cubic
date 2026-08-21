#!/usr/bin/env python3
"""Run independent Cubic correctness tests concurrently across visible GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TEST_PATHS = (
    "tests/quantization/test_cubic.py",
    "tests/quantization/test_cubic_dynamic_a8.py",
    "tests/quantization/test_cubic_policy.py",
    "tests/quantization/test_cubic_warmup.py",
    "tests/kernels/test_fused_recurrent_packed_decode.py",
    "tests/kernels/test_fla_layernorm_guard.py",
    "tests/v1/determinism/test_matmul_batch_invariant.py",
    "tests/v1/determinism/test_rms_norm_batch_invariant.py",
)


def _visible_devices() -> list[str]:
    configured = os.getenv("CUDA_VISIBLE_DEVICES")
    if configured:
        devices = [device.strip() for device in configured.split(",")]
        return [device for device in devices if device]
    try:
        output = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader"),
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    devices = _visible_devices()
    requested = int(
        os.getenv("VLLM_CUBIC_CORRECTNESS_GATE_WORKERS", str(len(devices) or 1))
    )
    worker_count = max(1, min(requested, len(devices) or 1))
    cpu_threads = int(os.getenv("VLLM_CUBIC_TEST_CPU_THREADS_PER_WORKER", "2"))
    if cpu_threads < 1:
        raise ValueError("VLLM_CUBIC_TEST_CPU_THREADS_PER_WORKER must be positive")

    processes: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    with tempfile.TemporaryDirectory(prefix="cubic-correctness-") as temporary:
        log_dir = Path(temporary)
        for index in range(worker_count):
            log_path = log_dir / f"shard-{index:02d}.log"
            environment = os.environ.copy()
            if devices:
                environment["CUDA_VISIBLE_DEVICES"] = devices[index]
            environment["OMP_NUM_THREADS"] = str(cpu_threads)
            environment["MKL_NUM_THREADS"] = str(cpu_threads)
            command = (
                str(repo / "venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                "-p",
                "tools.pytest_cubic_shard",
                "--cubic-shard-count",
                str(worker_count),
                "--cubic-shard-index",
                str(index),
                *TEST_PATHS,
            )
            with log_path.open("wb") as output:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            processes.append((index, log_path, process))
            print(
                f"START: Cubic correctness shard {index}/{worker_count} "
                f"on CUDA device {devices[index] if devices else 'none'}",
                flush=True,
            )

        failures = []
        pending = {index for index, _, _ in processes}
        while pending:
            for index, _, process in processes:
                if index in pending and process.poll() is not None:
                    pending.remove(index)
                    print(
                        f"DONE: Cubic correctness shard {index}/{worker_count} "
                        f"returned {process.returncode}",
                        flush=True,
                    )
            if pending:
                print(
                    "WAIT: Cubic correctness shards still running: "
                    + ", ".join(map(str, sorted(pending))),
                    flush=True,
                )
                time.sleep(10)

        for index, log_path, process in processes:
            returncode = process.returncode
            assert returncode is not None
            content = log_path.read_text(errors="replace")
            print(f"===== Cubic correctness shard {index}/{worker_count} =====")
            print(content, end="" if content.endswith("\n") else "\n")
            if returncode:
                failures.append((index, returncode))

    if failures:
        print(f"FAIL: Cubic correctness shards failed: {failures}", file=sys.stderr)
        return 1
    print(f"PASS: {worker_count} Cubic correctness shards completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
