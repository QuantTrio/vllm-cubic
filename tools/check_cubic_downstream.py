#!/usr/bin/env python3
"""Fail if an upstream sync drops a required vLLM-Cubic contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Contract:
    name: str
    path: str
    required: tuple[str, ...]


CONTRACTS = (
    Contract(
        "cubic-format",
        "vllm/model_executor/layers/quantization/cubic.py",
        (
            "def _normalize_group_size(",
            "CUBIC_SUPPORTED_BITS",
            "dynamic_a8=envs.VLLM_CUBIC_DYNAMIC_A8",
            "cubic_w8_precompute_carrier",
        ),
    ),
    Contract(
        "cubic-execution-policy",
        "vllm/model_executor/layers/quantization/cubic_policy.py",
        (
            "CUBIC_SUPPORTED_BITS = tuple(range(1, 9))",
            "CUBIC_ALIGNED_BITS = (1, 2, 4, 8)",
            "CUBIC_TOKEN_BUCKETS =",
            "class CubicExecutionKind(Enum):",
            "class CubicActivationMode(Enum):",
            "def cubic_token_bucket(",
            "def cubic_reconstruction_kind(",
        ),
    ),
    Contract(
        "finite-cubic-token-tactics",
        "vllm/model_executor/layers/quantization/cubic_kernels.py",
        (
            '"M_BUCKET",',
            "M_BUCKET=cubic_token_bucket(x_2d.shape[0])",
            "cubic_token_bucket(num_tokens)",
            "cubic_token_bucket(input_rows)",
        ),
    ),
    Contract(
        "query-protected-fp8-cache",
        "vllm/model_executor/layers/attention/attention.py",
        ('kv_cache_dtype != "fp8_q16"',),
    ),
    Contract(
        "native-kimi-fp8-q16",
        "vllm/v1/attention/backends/mla/flashmla.py",
        (
            'self._fp8_cache_only = self.kv_cache_dtype == "fp8_q16"',
            "self.supports_quant_query_input = False",
            "flash_mla_with_kvcache_fp8_q16(",
            "def _forward_fp8_q16_prefill_context(",
        ),
    ),
    Contract(
        "kimi-fp8-q16-cache-insert",
        "vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py",
        (
            "def fused_mla_key_concat_kv_cache_fp8_q16_insert(",
            "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_q16_insert",
        ),
    ),
    Contract(
        "hybrid-fresh-prefill-classification",
        "vllm/v1/attention/backends/gdn_attn.py",
        ("treat_short_extends_as_decodes=False",),
    ),
    Contract(
        "hybrid-fresh-prefill-cudagraph-guard",
        "vllm/v1/worker/gpu_model_runner.py",
        (
            "has_fresh_prefill: bool = False",
            "and not has_fresh_prefill",
            "has_fresh_prefill=has_fresh_prefill",
        ),
    ),
    Contract(
        "random-default-launch-seed",
        "vllm/config/model.py",
        (
            "def _random_seed() -> int:",
            "default_factory=_random_seed",
        ),
    ),
    Contract(
        "effective-auto-fit-context-log",
        "vllm/v1/core/kv_cache_utils.py",
        ('"effective max_model_len: %d',),
    ),
    Contract(
        "distributed-shutdown-grace",
        "vllm/v1/executor/multiproc_executor.py",
        (
            "cleanup_timeout = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS + 4",
            "timeout=cleanup_timeout",
        ),
    ),
    Contract(
        "multi-architecture-cubic-build",
        "CMakeLists.txt",
        (
            '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;10.7;11.0;12.0"',
            '"csrc/libtorch_stable/quantization/cubic_w4_w8_a8_gemv.cu"',
            "Building Cubic CUDA kernels for:",
        ),
    ),
    Contract(
        "cuda13-release-architecture-matrix",
        ".github/workflows/scripts/build.sh",
        ('"7.5 8.0 8.6 8.9 9.0 10.0 10.3 11.0 12.0"',),
    ),
)


def check_contracts() -> list[str]:
    failures: list[str] = []
    for contract in CONTRACTS:
        path = ROOT / contract.path
        if not path.is_file():
            failures.append(f"{contract.name}: missing {contract.path}")
            continue
        source = path.read_text(encoding="utf-8")
        for required in contract.required:
            if required not in source:
                failures.append(
                    f"{contract.name}: {contract.path} no longer contains {required!r}"
                )
    return failures


def main() -> int:
    failures = check_contracts()
    if failures:
        print("vLLM-Cubic downstream contract FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("Restore or deliberately replace every contract before syncing.")
        return 1
    print(f"vLLM-Cubic downstream contract passed ({len(CONTRACTS)} features).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
