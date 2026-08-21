#!/usr/bin/env python3
"""Run BF16 and Cubic numerical-correctness gates before performance work.

This harness treats every source-tree change as invalidating prior numerical
results. It launches a fixed-seed serve matrix, captures exact token logprobs,
and rejects any unexplained float or token difference before optional
performance commands are allowed to run. The default gate remains
``rtol=0``/``atol=0`` and fail-closed. A bounded difference may be admitted
only by a dedicated reduction-order audit which proves identical operands,
indexing, mathematical coverage, cache/state semantics, and a bounded error
envelope, followed by endpoint logprob and long-output checks. Generic
non-zero tolerances are experimental metadata and never turn a failure into a
pass.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class Variant:
    """One server configuration in the correctness matrix."""

    name: str
    prefix_caching: bool
    mtp_tokens: int


VARIANTS = (
    Variant("mtp0_prefix_off", False, 0),
    Variant("mtp0_prefix_on", True, 0),
    Variant("mtp1_prefix_off", False, 1),
    Variant("mtp1_prefix_on", True, 1),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}

STRICT_RTOL = 0.0
STRICT_ATOL = 0.0
SUPPORTED_CUBIC_BITS = tuple(range(1, 9))
SUPPORTED_CUBIC_ACTIVATIONS = ("A8", "A16")
SUPPORTED_CUBIC_OPERATORS = ("Linear", "MoE")
SUPPORTED_CUDA_SMS = (80, 86, 89, 90, 100, 103, 110, 120)
REQUIRED_CUBIC_CASES = frozenset(
    (bits, activation, operator, sm)
    for bits in SUPPORTED_CUBIC_BITS
    for activation in SUPPORTED_CUBIC_ACTIVATIONS
    for operator in SUPPORTED_CUBIC_OPERATORS
    for sm in SUPPORTED_CUDA_SMS
)

DEFAULT_KERNEL_COMMANDS = (
    "venv/bin/python tools/run_cubic_correctness_gates.py",
    "venv/bin/python tools/check_cubic_downstream.py",
)


def _run_git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(repo), *args))


def source_fingerprint(repo: Path) -> str:
    """Hash HEAD plus tracked and untracked source inputs.

    Generated artifacts, logs, model repositories, and engineering notes do
    not participate. A changed runtime, kernel, test, or harness file does.
    """
    digest = hashlib.sha256()
    digest.update(_run_git(repo, "rev-parse", "HEAD").strip())
    digest.update(_run_git(repo, "diff", "--binary", "HEAD", "--"))
    untracked = _run_git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "vllm",
        "csrc",
        "tests",
        "tools",
        "benchmarks",
    ).decode()
    for relative in sorted(filter(None, untracked.splitlines())):
        path = repo / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        body = response.read()
    return None if not body else json.loads(body)


def wait_until_ready(base_url: str, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            _request_json(f"{base_url}/health")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError("server did not become healthy within 30 minutes")


def _token_key(item: dict[str, Any]) -> str:
    byte_values = item.get("bytes")
    if byte_values is not None:
        return "bytes:" + bytes(byte_values).hex()
    return "token:" + item["token"]


def normalize_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain exact sampled-token and top-k values from an OpenAI response."""
    content = response["choices"][0]["logprobs"]["content"]
    normalized = []
    for position in content:
        top = {
            _token_key(item): item["logprob"] for item in position["top_logprobs"]
        }
        normalized.append(
            {
                "token": _token_key(position),
                "logprob": position["logprob"],
                "top_logprobs": top,
            }
        )
    return normalized


def compare_exact(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[str]:
    """Return exact token/logprob mismatches; an empty list means pass."""
    errors = []
    if len(reference) != len(candidate):
        errors.append(f"length: {len(reference)} != {len(candidate)}")
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected["token"] != actual["token"]:
            errors.append(
                f"position {index}: token {expected['token']!r} != "
                f"{actual['token']!r}"
            )
        if expected["logprob"] != actual["logprob"]:
            errors.append(
                f"position {index}: target logprob {expected['logprob']!r} != "
                f"{actual['logprob']!r}"
            )
        expected_top = expected["top_logprobs"]
        actual_top = actual["top_logprobs"]
        if expected_top.keys() != actual_top.keys():
            errors.append(f"position {index}: top-k token set differs")
            continue
        for token, expected_value in expected_top.items():
            actual_value = actual_top[token]
            if expected_value != actual_value:
                errors.append(
                    f"position {index}: top-k {token!r} logprob "
                    f"{expected_value!r} != {actual_value!r}"
                )
    return errors


def parse_cubic_case(value: str) -> tuple[int, str, str, int]:
    """Parse one explicitly audited ``W,A,operator,SM`` combination."""
    try:
        bits_text, activation, operator, sm_text = value.split(",")
        case = (int(bits_text), activation, operator, int(sm_text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected BITS,A8|A16,Linear|MoE,SM"
        ) from error
    if case not in REQUIRED_CUBIC_CASES:
        raise argparse.ArgumentTypeError(f"unsupported Cubic audit case: {value}")
    return case


def _report_coverage(
    report: dict[str, Any],
) -> tuple[set[tuple[Any, ...]], set[int]]:
    coverage = report.get("audited_scope", {})
    cubic_cases = {tuple(case) for case in coverage.get("cubic_cases", ())}
    bf16_sms = {int(sm) for sm in coverage.get("bf16_sms", ())}
    return cubic_cases, bf16_sms


def aggregate_coverage(
    source_fingerprint_value: str,
    current_cubic_cases: set[tuple[int, str, str, int]],
    current_bf16_sms: set[int],
    report_paths: list[Path],
) -> tuple[set[tuple[Any, ...]], set[int]]:
    """Union passing strict reports produced from the identical source tree."""
    cubic_cases: set[tuple[Any, ...]] = set(current_cubic_cases)
    bf16_sms = set(current_bf16_sms)
    for path in report_paths:
        report = json.loads(path.read_text())
        if report.get("source_fingerprint") != source_fingerprint_value:
            raise RuntimeError(f"coverage report is stale: {path}")
        if report.get("failures"):
            raise RuntimeError(f"coverage report did not pass: {path}")
        contract = report.get("correctness_contract", {})
        if contract.get("rtol") != 0.0 or contract.get("atol") != 0.0:
            raise RuntimeError(f"coverage report is not strict: {path}")
        report_cubic_cases, report_bf16_sms = _report_coverage(report)
        cubic_cases.update(report_cubic_cases)
        bf16_sms.update(report_bf16_sms)
    return cubic_cases, bf16_sms


def load_reused_bf16_captures(
    path: Path,
    source_fingerprint_value: str,
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load strictly compatible BF16 captures from a passing report."""
    report = json.loads(path.read_text())
    if report.get("source_fingerprint") != source_fingerprint_value:
        raise RuntimeError(f"BF16 capture report is stale: {path}")
    if report.get("failures"):
        raise RuntimeError(f"BF16 capture report did not pass: {path}")
    contract = report.get("correctness_contract", {})
    if contract.get("rtol") != 0.0 or contract.get("atol") != 0.0:
        raise RuntimeError(f"BF16 capture report is not strict: {path}")
    expected = {
        "bf16_model": args.bf16_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "kv_cache_dtype": args.kv_cache_dtype,
        "seed": args.seed,
        "top_logprobs": args.top_logprobs,
        "capture_profile": args.capture_profile,
        "capture_max_tokens": args.capture_max_tokens,
        "long_context_repetitions": args.long_context_repetitions,
        "capture_images": _capture_image_metadata(args.capture_image),
        "extra_serve_args": args.extra_serve_arg + args.extra_bf16_serve_arg,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"BF16 capture report has incompatible {key}: "
                f"{report.get(key)!r} != {value!r}"
            )
    captures = report.get("captures", {}).get("bf16")
    if not isinstance(captures, dict):
        raise RuntimeError(f"BF16 captures are missing from report: {path}")
    selected = {}
    for variant in variants:
        capture = captures.get(variant.name)
        if not isinstance(capture, dict):
            raise RuntimeError(
                f"BF16 capture {variant.name} is missing from report: {path}"
            )
        selected[variant.name] = capture
    return selected


def require_complete_coverage_for_performance(
    performance_commands: list[str],
    cubic_cases: set[tuple[Any, ...]],
    bf16_sms: set[int],
) -> tuple[set[tuple[Any, ...]], set[int]]:
    """Reject performance work until the complete strict matrix has passed."""
    missing_cubic_cases = REQUIRED_CUBIC_CASES - cubic_cases
    missing_bf16_sms = set(SUPPORTED_CUDA_SMS) - bf16_sms
    if performance_commands and (missing_cubic_cases or missing_bf16_sms):
        raise RuntimeError(
            "performance evaluation is blocked until strict reports cover the "
            f"complete matrix; missing {len(missing_cubic_cases)} Cubic cases "
            f"and {len(missing_bf16_sms)} BF16 SMs"
        )
    return missing_cubic_cases, missing_bf16_sms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--served-model-name")
    parser.add_argument("--bf16-model")
    parser.add_argument("--bf16-served-model-name", default="bf16-audit")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--kv-cache-dtype", default="fp8_q16")
    parser.add_argument("--port", type=int, default=30015)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument(
        "--capture-profile",
        choices=("breadth", "full", "multimodal"),
        default="full",
        help=(
            "breadth captures short fresh/hit plus C2 solo/concurrent; full "
            "also captures long context and C8"
        ),
    )
    parser.add_argument("--capture-max-tokens", type=int, default=128)
    parser.add_argument(
        "--capture-image",
        action="append",
        type=Path,
        default=[],
        help=(
            "Fixed local PNG/JPEG/WebP fixture for the multimodal profile; "
            "provide two distinct images"
        ),
    )
    parser.add_argument("--long-context-repetitions", type=int, default=3600)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--kernel-command",
        action="append",
        help="Override the default kernel/downstream gate commands.",
    )
    parser.add_argument("--performance-command", action="append", default=[])
    parser.add_argument("--experimental-rtol", type=float)
    parser.add_argument("--experimental-atol", type=float)
    parser.add_argument(
        "--audit-cubic-case",
        action="append",
        type=parse_cubic_case,
        default=[],
        metavar="BITS,A8|A16,Linear|MoE,SM",
        help="Exact Cubic implementation case covered by this run; repeatable.",
    )
    parser.add_argument(
        "--audit-bf16-sm",
        action="append",
        type=int,
        choices=SUPPORTED_CUDA_SMS,
        default=[],
        help="BF16 SM covered by this run; repeatable.",
    )
    parser.add_argument(
        "--coverage-report",
        action="append",
        type=Path,
        default=[],
        help="Passing strict report to aggregate; source fingerprint must match.",
    )
    parser.add_argument(
        "--reuse-bf16-report",
        type=Path,
        help=(
            "Reuse BF16 captures from a passing, strict, source-identical report "
            "with exactly matching request and serve parameters."
        ),
    )
    parser.add_argument(
        "--skip-bf16",
        action="store_true",
        help=(
            "Audit only the Cubic model. This mode cannot declare BF16 SM "
            "coverage and cannot run performance commands."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS_BY_NAME),
        help=(
            "Server variant to audit; repeatable. Defaults to all four. "
            "mtp0_prefix_off is mandatory as the exact reference."
        ),
    )
    parser.add_argument("--extra-serve-arg", action="append", default=[])
    parser.add_argument("--extra-cubic-serve-arg", action="append", default=[])
    parser.add_argument("--extra-bf16-serve-arg", action="append", default=[])
    parser.add_argument(
        "--verify-report",
        type=Path,
        help="Verify that a passing report still matches the current source tree.",
    )
    args = parser.parse_args()
    if args.long_context_repetitions < 1:
        parser.error("--long-context-repetitions must be positive")
    if args.capture_max_tokens < 1:
        parser.error("--capture-max-tokens must be positive")
    if args.capture_profile == "multimodal" and len(args.capture_image) != 2:
        parser.error(
            "--capture-profile multimodal requires exactly two "
            "--capture-image fixtures"
        )
    missing_images = [path for path in args.capture_image if not path.is_file()]
    if missing_images:
        parser.error(f"capture image does not exist: {missing_images[0]}")
    if args.verify_report is None:
        required = [
            "model",
            "served_model_name",
            "tensor_parallel_size",
            "cuda_visible_devices",
            "artifact_dir",
        ]
        if not args.skip_bf16:
            required.append("bf16_model")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
        selected_names = args.variant or [variant.name for variant in VARIANTS]
        if "mtp0_prefix_off" not in selected_names:
            parser.error("--variant must include mtp0_prefix_off")
        if len(selected_names) != len(set(selected_names)):
            parser.error("--variant values must be unique")
        if args.skip_bf16 and args.reuse_bf16_report is not None:
            parser.error("--skip-bf16 conflicts with --reuse-bf16-report")
        if args.skip_bf16 and args.audit_bf16_sm:
            parser.error("--skip-bf16 cannot declare --audit-bf16-sm")
        if args.skip_bf16 and args.performance_command:
            parser.error("--skip-bf16 cannot run --performance-command")
        args.variants = tuple(VARIANTS_BY_NAME[name] for name in selected_names)
    return args


def _run_gate_command(command: str, repo: Path, log: Path) -> None:
    with log.open("w") as output:
        result = subprocess.run(
            command,
            cwd=repo,
            shell=True,
            executable="/bin/bash",
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"gate command failed ({result.returncode}): {command}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _cubic_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("VLLM_CUBIC_") or key == "VLLM_USE_V2_MODEL_RUNNER"
    }


def _capture_checkpoint_identity(
    args: argparse.Namespace,
    source_fingerprint_value: str,
    *,
    case_name: str,
    model: str,
    served_model_name: str,
) -> dict[str, Any]:
    case_args = (
        args.extra_cubic_serve_arg
        if case_name == "cubic"
        else args.extra_bf16_serve_arg
    )
    return {
        "source_fingerprint": source_fingerprint_value,
        "case_name": case_name,
        "model": model,
        "served_model_name": served_model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cuda_visible_devices": args.cuda_visible_devices,
        "kv_cache_dtype": args.kv_cache_dtype,
        "seed": args.seed,
        "top_logprobs": args.top_logprobs,
        "capture_profile": args.capture_profile,
        "capture_max_tokens": args.capture_max_tokens,
        "long_context_repetitions": args.long_context_repetitions,
        "capture_images": _capture_image_metadata(args.capture_image),
        "extra_serve_args": args.extra_serve_arg,
        "extra_case_serve_args": case_args,
        "cubic_environment": _cubic_environment(),
    }


def _load_capture_checkpoint(
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    checkpoint = json.loads(path.read_text())
    cases = checkpoint.get("cases")
    if not isinstance(cases, dict):
        raise RuntimeError(f"invalid capture checkpoint: {path}")
    return cases


def _write_capture_checkpoint(
    path: Path,
    cases: dict[str, dict[str, Any]],
) -> None:
    _write_json_atomic(path, {"cases": cases})


def _serve_command(
    args: argparse.Namespace,
    variant: Variant,
    *,
    case_name: str,
    model: str,
    served_model_name: str,
) -> list[str]:
    command = [
        str(Path("venv/bin/vllm").resolve()),
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--deterministic-inference",
        "--enforce-eager",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--seed",
        str(args.seed),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    command.append(
        "--enable-prefix-caching"
        if variant.prefix_caching
        else "--no-enable-prefix-caching"
    )
    if variant.mtp_tokens:
        command.extend(
            (
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": variant.mtp_tokens,
                    },
                    separators=(",", ":"),
                ),
            )
        )
    command.extend(args.extra_serve_arg)
    command.extend(
        args.extra_cubic_serve_arg
        if case_name == "cubic"
        else args.extra_bf16_serve_arg
    )
    return command


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)


def _capture_image_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "suffix": path.suffix.lower(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _image_data_url(path: Path) -> str:
    media_types = {
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    suffix = path.suffix.lower()
    if suffix not in media_types:
        raise ValueError(f"unsupported capture image format: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{media_types[suffix]};base64,{encoded}"


def _multimodal_content(path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(path)},
        },
        {"type": "text", "text": prompt},
    ]


def _completion_payload(
    args: argparse.Namespace, prompt: Any, served_model_name: str
) -> dict[str, Any]:
    return {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.capture_max_tokens,
        "seed": args.seed,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }


def _render_prompt(
    base_url: str,
    args: argparse.Namespace,
    prompt: Any,
    served_model_name: str,
) -> dict[str, Any]:
    rendered = _request_json(
        f"{base_url}/v1/chat/completions/render",
        _completion_payload(args, prompt, served_model_name),
    )
    if not isinstance(rendered.get("token_ids"), list):
        raise RuntimeError("rendered request does not contain token_ids")
    sampling_params = rendered.get("sampling_params")
    if not isinstance(sampling_params, dict):
        raise RuntimeError("rendered request does not contain sampling_params")
    sampling_params["max_tokens"] = args.capture_max_tokens
    return rendered


def _capture_one(
    base_url: str, rendered_request: dict[str, Any]
) -> list[dict[str, Any]]:
    request = dict(rendered_request)
    request["request_id"] = f"strict-audit-{uuid.uuid4().hex}"
    return normalize_response(
        _request_json(
            f"{base_url}/inference/v1/generate",
            request,
        )
    )


def _capture_after_barrier(
    rendered_request: dict[str, Any],
    *,
    barrier: threading.Barrier,
    base_url: str,
) -> list[dict[str, Any]]:
    barrier.wait()
    return _capture_one(base_url, rendered_request)


def capture_variant(
    base_url: str, args: argparse.Namespace, served_model_name: str
) -> dict[str, list[dict[str, Any]]]:
    """Capture fresh/hit, long-context, and batch-invariance cases."""
    short_prompt = "请逐项解释为什么矩阵乘法的分块大小会影响浮点归约顺序。"
    if args.capture_profile == "multimodal":
        image_prompts = [
            _multimodal_content(
                path,
                f"请客观描述图像内容。固定审计图像编号为 {index}。",
            )
            for index, path in enumerate(args.capture_image)
        ]
        capture = {}
        rendered_images = [
            _render_prompt(base_url, args, prompt, served_model_name)
            for prompt in image_prompts
        ]
        for index, rendered in enumerate(rendered_images):
            capture[f"image{index}_fresh"] = _capture_one(base_url, rendered)
            capture[f"image{index}_hit"] = _capture_one(base_url, rendered)

        same_image_prompts = [
            _multimodal_content(
                args.capture_image[0],
                f"请客观描述图像内容。固定重复样本编号为 {index}。",
            )
            for index in range(2)
        ]
        batches = {
            "mm_same_batch2": same_image_prompts,
            "mm_distinct_batch2": image_prompts,
        }
        for batch_name, prompts in batches.items():
            rendered_requests = [
                _render_prompt(base_url, args, prompt, served_model_name)
                for prompt in prompts
            ]
            for index, rendered in enumerate(rendered_requests):
                capture[f"{batch_name}_solo_{index}"] = _capture_one(
                    base_url, rendered
                )
            barrier = threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        partial(
                            _capture_after_barrier,
                            barrier=barrier,
                            base_url=base_url,
                        ),
                        rendered_requests,
                    )
                )
            for index, result in enumerate(results):
                capture[f"{batch_name}_concurrent_{index}"] = result
        return capture

    prompts = [("short", short_prompt)]
    if args.capture_profile == "full":
        long_prompt = (
            "这是用于数值一致性审计的固定上下文。"
            * args.long_context_repetitions
        ) + "请总结上述上下文并解释矩阵乘法分块。"
        prompts.append(("long", long_prompt))
    capture = {}
    for prompt_name, prompt in prompts:
        rendered = _render_prompt(base_url, args, prompt, served_model_name)
        capture[f"{prompt_name}_fresh"] = _capture_one(
            base_url, rendered
        )
        capture[f"{prompt_name}_hit"] = _capture_one(base_url, rendered)

    batch_sizes = (2, 8) if args.capture_profile == "full" else (2,)
    for batch_size in batch_sizes:
        prompts = [
            f"{short_prompt} 固定审计样本编号为 {index}。"
            for index in range(batch_size)
        ]
        rendered_requests = [
            _render_prompt(base_url, args, prompt, served_model_name)
            for prompt in prompts
        ]
        for index, rendered in enumerate(rendered_requests):
            capture[f"batch{batch_size}_solo_{index}"] = _capture_one(
                base_url, rendered
            )
        barrier = threading.Barrier(batch_size)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                executor.map(
                    partial(
                        _capture_after_barrier,
                        barrier=barrier,
                        base_url=base_url,
                    ),
                    rendered_requests,
                )
            )
        for index, result in enumerate(results):
            capture[f"batch{batch_size}_concurrent_{index}"] = result
    return capture


def compare_variant_matrix(
    captures: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[str]:
    """Compare every mode to MTP-off/prefix-off and solo to batched runs."""
    failures = []
    reference = captures["mtp0_prefix_off"]
    for variant_name, capture in captures.items():
        for case_name, result in capture.items():
            baseline_name = case_name
            if case_name.endswith("_hit"):
                baseline_name = case_name.removesuffix("_hit") + "_fresh"
            if "_concurrent_" in case_name:
                baseline_name = case_name.replace("_concurrent_", "_solo_")
            failures.extend(
                f"{variant_name}/{case_name}: {error}"
                for error in compare_exact(reference[baseline_name], result)
            )
            if "_concurrent_" in case_name:
                solo_name = case_name.replace("_concurrent_", "_solo_")
                failures.extend(
                    f"{variant_name}/{case_name} batch invariance: {error}"
                    for error in compare_exact(capture[solo_name], result)
                )
    return failures


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text())
        if report["failures"]:
            raise RuntimeError("report did not pass its correctness gates")
        current = source_fingerprint(repo)
        expected = report["source_fingerprint"]
        if current != expected:
            raise RuntimeError(
                "report is stale: source fingerprint "
                f"{current} != audited {expected}"
            )
        print(f"PASS: report remains valid for {current}")
        return 0

    initial_fingerprint = source_fingerprint(repo)
    current_cubic_cases = set(args.audit_cubic_case)
    current_bf16_sms = set(args.audit_bf16_sm)
    aggregate_cubic_cases, aggregate_bf16_sms = aggregate_coverage(
        initial_fingerprint,
        current_cubic_cases,
        current_bf16_sms,
        args.coverage_report,
    )
    missing_cubic_cases, missing_bf16_sms = (
        require_complete_coverage_for_performance(
            args.performance_command,
            aggregate_cubic_cases,
            aggregate_bf16_sms,
        )
    )
    coverage_complete = not missing_cubic_cases and not missing_bf16_sms
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    kernel_commands = args.kernel_command or DEFAULT_KERNEL_COMMANDS
    kernel_checkpoint_path = args.artifact_dir / "kernel-gates.json"
    kernel_identity = {
        "source_fingerprint": initial_fingerprint,
        "commands": kernel_commands,
        "cuda_visible_devices": args.cuda_visible_devices,
        "cubic_environment": _cubic_environment(),
    }
    kernel_checkpoint = (
        json.loads(kernel_checkpoint_path.read_text())
        if kernel_checkpoint_path.exists()
        else None
    )
    if kernel_checkpoint != {"identity": kernel_identity, "passed": True}:
        for index, command in enumerate(kernel_commands):
            _run_gate_command(
                command, repo, args.artifact_dir / f"kernel-{index:02d}.log"
            )
        _write_json_atomic(
            kernel_checkpoint_path,
            {"identity": kernel_identity, "passed": True},
        )

    model_cases = [("cubic", args.model, args.served_model_name)]
    capture_checkpoint_path = args.artifact_dir / "capture-checkpoint.json"
    checkpoint_cases = _load_capture_checkpoint(capture_checkpoint_path)
    all_captures = {}
    if not args.skip_bf16 and args.reuse_bf16_report is None:
        model_cases.append(("bf16", args.bf16_model, args.bf16_served_model_name))
    elif args.reuse_bf16_report is not None:
        all_captures["bf16"] = load_reused_bf16_captures(
            args.reuse_bf16_report,
            initial_fingerprint,
            args,
            args.variants,
        )
    base_url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for case_name, model, served_model_name in model_cases:
        case_identity = _capture_checkpoint_identity(
            args,
            initial_fingerprint,
            case_name=case_name,
            model=model,
            served_model_name=served_model_name,
        )
        checkpoint_case = checkpoint_cases.get(case_name, {})
        captures = (
            checkpoint_case.get("captures", {})
            if checkpoint_case.get("identity") == case_identity
            else {}
        )
        if not isinstance(captures, dict):
            raise RuntimeError(
                f"invalid {case_name} capture checkpoint: "
                f"{capture_checkpoint_path}"
            )
        all_captures[case_name] = captures
        for variant in args.variants:
            if variant.name in captures:
                continue
            log_path = args.artifact_dir / f"serve-{case_name}-{variant.name}.log"
            serve_log_path = repo / "log.out"
            with serve_log_path.open("w") as log:
                process = subprocess.Popen(
                    _serve_command(
                        args,
                        variant,
                        case_name=case_name,
                        model=model,
                        served_model_name=served_model_name,
                    ),
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_until_ready(base_url, process)
                    captures[variant.name] = capture_variant(
                        base_url, args, served_model_name
                    )
                    checkpoint_cases[case_name] = {
                        "identity": case_identity,
                        "captures": captures,
                    }
                    _write_capture_checkpoint(
                        capture_checkpoint_path,
                        checkpoint_cases,
                    )
                finally:
                    _stop_server(process)
                    log.flush()
                    shutil.copyfile(serve_log_path, log_path)
        all_captures[case_name] = captures

    failures = [
        f"{case_name}/{failure}"
        for case_name, captures in all_captures.items()
        for failure in compare_variant_matrix(captures)
    ]

    final_fingerprint = source_fingerprint(repo)
    if final_fingerprint != initial_fingerprint:
        failures.append("source tree changed while correctness gates were running")

    report = {
        "source_fingerprint": initial_fingerprint,
        "git_head": _run_git(repo, "rev-parse", "HEAD").decode().strip(),
        "model": args.model,
        "bf16_model": args.bf16_model,
        "served_model_name": args.served_model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cuda_visible_devices": args.cuda_visible_devices,
        "kv_cache_dtype": args.kv_cache_dtype,
        "seed": args.seed,
        "top_logprobs": args.top_logprobs,
        "capture_profile": args.capture_profile,
        "capture_max_tokens": args.capture_max_tokens,
        "long_context_repetitions": args.long_context_repetitions,
        "capture_images": _capture_image_metadata(args.capture_image),
        "extra_serve_args": args.extra_serve_arg,
        "extra_cubic_serve_args": args.extra_cubic_serve_arg,
        "extra_bf16_serve_args": args.extra_bf16_serve_arg,
        "kernel_commands": kernel_commands,
        "performance_commands": args.performance_command,
        "cubic_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("VLLM_CUBIC_") or key == "VLLM_USE_V2_MODEL_RUNNER"
        },
        "variants": [dataclasses.asdict(variant) for variant in args.variants],
        "reused_bf16_report": (
            None
            if args.reuse_bf16_report is None
            else str(args.reuse_bf16_report)
        ),
        "correctness_contract": {
            "rtol": STRICT_RTOL,
            "atol": STRICT_ATOL,
            "experimental_tolerances_cannot_pass_gate": True,
            "acceptance_authority": "user",
        },
        "required_scope": {
            "cubic_bits": SUPPORTED_CUBIC_BITS,
            "cubic_activations": SUPPORTED_CUBIC_ACTIVATIONS,
            "cubic_operators": SUPPORTED_CUBIC_OPERATORS,
            "cuda_sms": SUPPORTED_CUDA_SMS,
            "cubic_cartesian_case_count": len(REQUIRED_CUBIC_CASES),
        },
        "audited_scope": {
            "cubic_cases": sorted(current_cubic_cases),
            "bf16_sms": sorted(current_bf16_sms),
        },
        "aggregate_scope": {
            "cubic_cases": sorted(aggregate_cubic_cases),
            "bf16_sms": sorted(aggregate_bf16_sms),
            "missing_cubic_case_count": len(missing_cubic_cases),
            "missing_bf16_sms": sorted(missing_bf16_sms),
            "coverage_complete": coverage_complete,
        },
        "experimental_tolerances": {
            "rtol": args.experimental_rtol,
            "atol": args.experimental_atol,
        },
        "captures": all_captures,
        "failures": failures,
    }
    (args.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    if failures:
        grouped = Counter(
            failure.split(": position", 1)[0] for failure in failures
        )
        print(
            f"FAIL: {len(failures)} exact mismatches across "
            f"{len(grouped)} cases",
            file=sys.stderr,
        )
        for case, count in grouped.most_common():
            print(f"FAIL: {case}: {count} mismatches", file=sys.stderr)
        return 1

    for index, command in enumerate(args.performance_command):
        if source_fingerprint(repo) != initial_fingerprint:
            raise RuntimeError("source changed after correctness pass")
        _run_gate_command(
            command, repo, args.artifact_dir / f"performance-{index:02d}.log"
        )
        if source_fingerprint(repo) != initial_fingerprint:
            raise RuntimeError("source changed during performance evaluation")
    print(f"PASS: exact numerical gate for {initial_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
