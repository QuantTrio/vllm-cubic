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
        "strict-cubic-correctness-harness",
        "tools/cubic_correctness_harness.py",
        (
            "STRICT_RTOL = 0.0",
            "STRICT_ATOL = 0.0",
            "tools/run_cubic_correctness_gates.py",
            "experimental_tolerances_cannot_pass_gate",
        ),
    ),
    Contract(
        "parallel-cubic-correctness-gates",
        "tools/run_cubic_correctness_gates.py",
        (
            "VLLM_CUBIC_CORRECTNESS_GATE_WORKERS",
            "VLLM_CUBIC_TEST_CPU_THREADS_PER_WORKER",
            "CUDA_VISIBLE_DEVICES",
            "--cubic-shard-count",
            "--cubic-shard-index",
        ),
    ),
    Contract(
        "cubic-format",
        "vllm/model_executor/layers/quantization/cubic.py",
        (
            "def _normalize_group_size(",
            "CUBIC_SUPPORTED_BITS",
            "dynamic_a8=envs.VLLM_CUBIC_DYNAMIC_A8",
            "def materialize_cubic_a8_carrier(",
            "def install_cubic_a8_carrier(",
            'if self.dynamic_a8 and carrier is not None:',
            "if self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:",
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
            "class CubicRuntimeCandidate:",
            "def cubic_runtime_memory(",
            "def select_cubic_runtime_candidate(",
            "def cubic_token_bucket(",
            "def cubic_linear_token_bucket(",
            "def cubic_reconstruction_kind(",
        ),
    ),
    Contract(
        "explicit-cubic-linear-tactics",
        "vllm/model_executor/layers/quantization/cubic_kernels.py",
        (
            "def _cubic_linear_tile(",
            "precomputed_carrier: bool,",
            "_CUBIC_COMPACT_LINEAR_TILE_TACTICS",
            "_CUBIC_LINEAR_RESIDENCY_TACTICS",
            "BLOCK_M=block_m",
            "BLOCK_N=block_n",
            "num_warps=num_warps",
            "cubic_token_bucket(num_tokens)",
            "cubic_token_bucket(input_rows)",
        ),
    ),
    Contract(
        "shape-calibrated-cubic-moe-reduction",
        "vllm/model_executor/layers/quantization/cubic_kernels.py",
        (
            "_CUBIC_MOE_SUM_TACTICS",
            "def _cubic_moe_use_torch_sum(",
            "def calibrate_cubic_moe_sum_backend(",
        ),
    ),
    Contract(
        "speculative-runtime-jit-warmup",
        "vllm/model_executor/warmup/spec_decode_triton_warmup.py",
        (
            "for num_sampled_tokens in {num_spec_tokens, num_spec_tokens + 1}:",
            "for dtype in (torch.float32, torch.int32):",
            "context.initialize_from_forward_context(",
            "context.run_fused_postprocess(",
            "context.run_fused_precopy(",
        ),
    ),
    Contract(
        "bounded-cubic-startup-calibration",
        "vllm/model_executor/warmup/cubic_warmup.py",
        (
            "def _calibration_token_buckets(",
            "representatives = (1, 2, 16, 64, 256, 512)",
            "def _moe_calibration_token_buckets(",
            "moe_token_buckets=moe_token_buckets",
            "def _materialization_token_buckets(",
            "def _materialize_cubic_linear_residency(",
            "_CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS",
            "benefit / extra",
            "key + (backend,)",
            "install_cubic_expanded_metadata(",
            "materialization_buckets = _materialization_token_buckets(max_tokens)",
        ),
    ),
    Contract(
        "parallel-cubic-startup-compilation",
        "vllm/envs.py",
        ("VLLM_CUBIC_COMPILE_WORKERS_PER_GPU",),
    ),
    Contract(
        "low-m-batch-invariant-execution",
        "vllm/model_executor/layers/batch_invariant.py",
        (
            "LOW_M_BATCH_INVARIANT_LIMIT = 8",
            "def use_low_m_batch_invariant(",
            "def linear_batch_invariant(",
        ),
    ),
    Contract(
        "low-m-flash-attention-split-invariance",
        "vllm/v1/attention/backends/flash_attn.py",
        (
            "def _get_max_num_splits(",
            "num_actual_tokens <= LOW_M_BATCH_INVARIANT_LIMIT",
            "_get_max_num_splits(max_num_splits, num_actual_tokens)",
        ),
    ),
    Contract(
        "explicit-batch-invariant-tp-reduction",
        "vllm/model_executor/layers/linear.py",
        (
            "tensor_model_parallel_all_reduce_batch",
            "if envs.VLLM_BATCH_INVARIANT:",
            "output_parallel",
        ),
    ),
    Contract(
        "pynccl-independent-grouped-allreduce",
        "vllm/distributed/device_communicators/cuda_communicator.py",
        (
            "def all_reduce_batch(",
            "pynccl_comm.group_start()",
            "pynccl_comm.group_end()",
        ),
    ),
    Contract(
        "hybrid-speculative-input-partition",
        "vllm/model_executor/layers/mamba/abstract.py",
        (
            "def partition_speculative_token_inputs(",
            "def get_recurrent_state_alignment(",
            "recurrent_state_alignment=self.get_recurrent_state_alignment()",
        ),
    ),
    Contract(
        "qwen-gdn-recurrent-state-alignment",
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        (
            "def get_recurrent_state_alignment(",
            "return FLA_CHUNK_SIZE",
        ),
    ),
    Contract(
        "hybrid-prefix-recurrent-alignment",
        "vllm/v1/core/kv_cache_coordinator.py",
        (
            "spec.recurrent_state_alignment",
            "alignment = lcm(alignment, spec.recurrent_state_alignment)",
        ),
    ),
    Contract(
        "hybrid-prefill-recurrent-alignment",
        "vllm/v1/core/sched/scheduler.py",
        (
            "self.mamba_recurrent_state_alignment = 1",
            "self.mamba_prefill_alignment = lcm(",
            "tail_boundary % state_alignment != 0",
        ),
    ),
    Contract(
        "qwen-hybrid-compile-safety-boundary",
        "vllm/model_executor/models/qwen3_5.py",
        ("@ignore_torch_compile\n@support_torch_compile(",),
    ),
    Contract(
        "qwen-mtp-compile-safety-boundary",
        "vllm/model_executor/models/qwen3_5_mtp.py",
        (
            "@ignore_torch_compile\n@support_torch_compile(",
            "class Qwen3_5MultiTokenPredictor(",
            "class Qwen3_5MTP(",
        ),
    ),
    Contract(
        "gdn-packed-speculative-arithmetic-parity",
        "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py",
        (
            "b_q = b_q / tl.sqrt(",
            "b_k = b_k / tl.sqrt(",
            "b_h = b_h.to(ht.dtype.element_ty).to(tl.float32)",
            "num_warps = 1",
        ),
    ),
    Contract(
        "gemma-prefill-batch-invariant-rmsnorm",
        "vllm/model_executor/layers/layernorm.py",
        (
            "class GemmaRMSNorm(CustomOp):",
            "return rms_norm_batch_invariant(",
            "self.weight.float() + 1.0",
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
