# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
    round_int8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES = 128 * 1024 * 1024
_CUBIC_A8_ROUTE_WORKSPACE_MIN_BYTES = 8 * 1024 * 1024
_CUBIC_A8_ROUTE_WORKSPACE_MAX_BYTES = 128 * 1024 * 1024
_CUBIC_A8_PIPELINE_WORKSPACE_MIN_BYTES = 64 * 1024 * 1024
_CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES = 384 * 1024 * 1024

# (device, N, K, group_size, local_experts, route_ctas) -> tactic. This is
# populated during kernel_warmup, before CUDA graph capture.  Keep the lookup
# graph-safe and free of timing or synchronization in the execution path.
_CUBIC_W2_A8_SITU_TACTICS: dict[
    tuple[int, int, int, int, int, int], tuple[int, int]
] = {}
_CUBIC_A8_MOE_BACKEND_TACTICS: dict[
    tuple[int, int, int, int, int, int, int, int, int], str
] = {}
_CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS: dict[
    tuple[int, int, int, int, int, int, int, int, int], str
] = {}
_CUBIC_A8_MOE_GROUPING_TACTICS: dict[
    tuple[int, int, int, int, int, int, int, int], int
] = {}
_CUBIC_MOE_EXECUTION_TACTICS: dict[
    tuple[int, bool, int, int, int, int, int, int, int], bool
] = {}
_CUBIC_LINEAR_EXECUTION_TACTICS: dict[
    tuple[int, bool, int, int, int, int, int, int], bool
] = {}
_CUBIC_MOE_DENSE_BLOCK_TACTICS: dict[
    tuple[int, bool, int, int, int, int, int, int, int], int
] = {}
_CUBIC_MOE_ROUTE_CTA_TACTICS: dict[
    tuple[int, bool, int, int, int, int, int, int, int, int, int], int
] = {}
_CUBIC8_W2_BLOCK_N_TACTICS: dict[tuple[int, int, int, int, int, int], int] = {}
_CUBIC8_W2_LUT_TACTICS: dict[tuple[int, int, int, int, int, int], int] = {}
_CUBIC_W2_A8_SITU_CANDIDATES = ((8, 2), (16, 2), (32, 2), (32, 4), (64, 4))


class CubicA8Carrier(NamedTuple):
    """Runtime groupwise A8 values and their per-sample FP32 scales."""

    values: torch.Tensor
    scales: torch.Tensor
    group_size: int


class CubicA8Code(NamedTuple):
    """True runtime Cubic8 codes and per-sample/per-group curve metadata.

    Unlike :class:`CubicA8Carrier`, ``codes`` are not linearly reconstructed
    INT8 values.  Consumers must evaluate the persisted Cubic curve from
    ``scales/a/b`` inside their compute tile.  Keeping the two contracts as
    distinct types prevents the linear Dynamic-A8 fallback from silently
    masquerading as a Cubic×Cubic path.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    group_size: int


def _as_groupwise_a8(carrier: CubicA8Code) -> CubicA8Carrier:
    """View the fixed ``a=1,b=0`` online code as linear groupwise A8."""
    return CubicA8Carrier(carrier.codes, carrier.scales, carrier.group_size)


def _cubic8_w2_block_n(
    n: int,
    k: int,
    group_size: int,
    local_experts: int,
    local_routes: int,
) -> int:
    """Read a pre-capture device tactic; use a conservative fallback.

    Device ordinal is intentionally only a process-local lookup component.
    The persisted Cubic tactic cache replaces it with the current ordinal when
    loaded, so equal device fingerprints share measurements across ranks and
    nodes while heterogeneous devices keep independent choices.
    """
    device = torch.accelerator.current_device_index()
    exact = (device, n, k, group_size, local_experts, local_routes)
    if exact in _CUBIC8_W2_BLOCK_N_TACTICS:
        return _CUBIC8_W2_BLOCK_N_TACTICS[exact]
    compatible = [
        (abs(routes - local_routes), block_n)
        for (
            candidate_device,
            candidate_n,
            candidate_k,
            candidate_group,
            candidate_experts,
            routes,
        ), block_n in _CUBIC8_W2_BLOCK_N_TACTICS.items()
        if candidate_device == device
        and candidate_n == n
        and candidate_k == k
        and candidate_group == group_size
        and candidate_experts == local_experts
    ]
    if compatible:
        return min(compatible, key=lambda item: item[0])[1]
    return 64 if group_size == 512 and local_routes >= 16 else 16


def _cubic8_w2_lut_threads(
    n: int,
    k: int,
    group_size: int,
    local_experts: int,
    local_routes: int,
) -> int:
    device = torch.accelerator.current_device_index()
    exact = (device, n, k, group_size, local_experts, local_routes)
    if exact in _CUBIC8_W2_LUT_TACTICS:
        return _CUBIC8_W2_LUT_TACTICS[exact]
    compatible = [
        (abs(routes - local_routes), threads)
        for (
            candidate_device,
            candidate_n,
            candidate_k,
            candidate_group,
            candidate_experts,
            routes,
        ), threads in _CUBIC8_W2_LUT_TACTICS.items()
        if candidate_device == device
        and candidate_n == n
        and candidate_k == k
        and candidate_group == group_size
        and candidate_experts == local_experts
    ]
    if compatible:
        return min(compatible, key=lambda item: item[0])[1]
    return 8 if local_routes <= 16 else 4


_CUBIC_LINEAR_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": 4}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=1),
]

_CUBIC_LINEAR_DENSE_CONFIGS = [
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

_CUBIC_MOE_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": 4}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2),
]

_CUBIC_MOE_GENERIC_GEMV_CONFIGS = [
    *_CUBIC_MOE_GEMV_CONFIGS,
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
]

_CUBIC_MOE_2BIT_A16_CONFIGS = [
    triton.Config({"BLOCK_N": 4}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=1),
]

_CUBIC_MOE_DENSE_N_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
]

def _cubic_a8_moe_backend(
    *,
    num_bits: int,
    n: int,
    k: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    grouped_routes: int,
    route_ctas: int,
) -> str:
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        num_bits,
        n,
        k,
        group_size,
        group_out,
        local_experts,
        grouped_routes,
        route_ctas,
    )
    if exact in _CUBIC_A8_MOE_BACKEND_TACTICS:
        return _CUBIC_A8_MOE_BACKEND_TACTICS[exact]
    matching = [
        (abs(bucket - route_ctas), backend)
        for (
            dev,
            bits,
            nn,
            kk,
            group,
            output_group,
            experts,
            grouped,
            bucket,
        ), backend in _CUBIC_A8_MOE_BACKEND_TACTICS.items()
        if dev == device
        and bits == num_bits
        and nn == n
        and kk == k
        and group == group_size
        and output_group == group_out
        and experts == local_experts
        and grouped == grouped_routes
    ]
    if matching:
        return min(matching, key=lambda item: item[0])[1]
    return "cuda"


def _cubic_online_a8_moe_backend(
    *,
    num_bits: int,
    n: int,
    k: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    grouped_routes: int,
    route_ctas: int,
) -> str:
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        num_bits,
        n,
        k,
        group_size,
        group_out,
        local_experts,
        grouped_routes,
        route_ctas,
    )
    if exact in _CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS:
        return _CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS[exact]
    matching = [
        (abs(bucket - route_ctas), backend)
        for (
            dev,
            bits,
            nn,
            kk,
            group,
            output_group,
            experts,
            grouped,
            bucket,
        ), backend in _CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS.items()
        if dev == device
        and bits == num_bits
        and nn == n
        and kk == k
        and group == group_size
        and output_group == group_out
        and experts == local_experts
        and grouped == grouped_routes
    ]
    if matching:
        return min(matching, key=lambda item: item[0])[1]
    if num_bits == 2 and grouped_routes == 2:
        return "triton"
    if num_bits == 1 or group_size < 128:
        return "triton"
    return "cuda"


def _cubic_a8_moe_grouping(
    *,
    num_bits: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    num_tokens: int,
    fallback: int,
) -> int:
    if group_size < 256:
        return 1
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        local_experts,
        num_tokens,
    )
    if exact in _CUBIC_A8_MOE_GROUPING_TACTICS:
        return _CUBIC_A8_MOE_GROUPING_TACTICS[exact]
    matching = [
        (abs(tokens - num_tokens), grouped)
        for (
            dev,
            bits,
            hidden,
            intermediate,
            group,
            out_group,
            experts,
            tokens,
        ), grouped in _CUBIC_A8_MOE_GROUPING_TACTICS.items()
        if dev == device
        and bits == num_bits
        and hidden == hidden_size
        and intermediate == intermediate_size
        and group == group_size
        and out_group == group_out
        and experts == local_experts
    ]
    return min(matching, key=lambda item: item[0])[1] if matching else fallback


def _cubic_moe_use_gemv(
    *,
    dynamic_a8: bool,
    num_bits: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    num_tokens: int,
    fallback: bool,
) -> bool:
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        dynamic_a8,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        local_experts,
        num_tokens,
    )
    if exact in _CUBIC_MOE_EXECUTION_TACTICS:
        return _CUBIC_MOE_EXECUTION_TACTICS[exact]
    matching = [
        (abs(tokens - num_tokens), use_gemv)
        for (
            dev,
            a8,
            bits,
            hidden,
            intermediate,
            group,
            output_group,
            experts,
            tokens,
        ), use_gemv in _CUBIC_MOE_EXECUTION_TACTICS.items()
        if dev == device
        and a8 == dynamic_a8
        and bits == num_bits
        and hidden == hidden_size
        and intermediate == intermediate_size
        and group == group_size
        and output_group == group_out
        and experts == local_experts
    ]
    return min(matching, key=lambda item: item[0])[1] if matching else fallback


def _cubic_linear_use_gemv(
    *,
    dynamic_a8: bool,
    num_bits: int,
    n: int,
    k: int,
    group_size: int,
    group_out: int,
    m: int,
    fallback: bool,
) -> bool:
    device = torch.accelerator.current_device_index()
    exact = (device, dynamic_a8, num_bits, n, k, group_size, group_out, m)
    if exact in _CUBIC_LINEAR_EXECUTION_TACTICS:
        return _CUBIC_LINEAR_EXECUTION_TACTICS[exact]
    matching = [
        (abs(tokens - m), use_gemv)
        for (
            dev,
            a8,
            bits,
            nn,
            kk,
            group,
            output_group,
            tokens,
        ), use_gemv in _CUBIC_LINEAR_EXECUTION_TACTICS.items()
        if dev == device
        and a8 == dynamic_a8
        and bits == num_bits
        and nn == n
        and kk == k
        and group == group_size
        and output_group == group_out
    ]
    return min(matching, key=lambda item: item[0])[1] if matching else fallback


def _cubic_moe_dense_block_m(
    *,
    dynamic_a8: bool,
    num_bits: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    num_tokens: int,
    fallback: int,
) -> int:
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        dynamic_a8,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        local_experts,
        num_tokens,
    )
    if exact in _CUBIC_MOE_DENSE_BLOCK_TACTICS:
        return _CUBIC_MOE_DENSE_BLOCK_TACTICS[exact]
    matching = [
        (abs(tokens - num_tokens), block_m)
        for (
            dev,
            a8,
            bits,
            hidden,
            intermediate,
            group,
            output_group,
            experts,
            tokens,
        ), block_m in _CUBIC_MOE_DENSE_BLOCK_TACTICS.items()
        if dev == device
        and a8 == dynamic_a8
        and bits == num_bits
        and hidden == hidden_size
        and intermediate == intermediate_size
        and group == group_size
        and output_group == group_out
        and experts == local_experts
    ]
    return min(matching, key=lambda item: item[0])[1] if matching else fallback


def _cubic_moe_route_ctas(
    *,
    dynamic_a8: bool,
    num_bits: int,
    n: int,
    k: int,
    group_size: int,
    group_out: int,
    local_experts: int,
    grouped_routes: int,
    input_rows: int,
    top_k: int,
    fallback: int,
) -> int:
    device = torch.accelerator.current_device_index()
    exact = (
        device,
        dynamic_a8,
        num_bits,
        n,
        k,
        group_size,
        group_out,
        local_experts,
        grouped_routes,
        input_rows,
        top_k,
    )
    if exact in _CUBIC_MOE_ROUTE_CTA_TACTICS:
        return _CUBIC_MOE_ROUTE_CTA_TACTICS[exact]
    matching = [
        (abs(rows - input_rows), route_ctas)
        for (
            dev,
            a8,
            bits,
            nn,
            kk,
            group,
            output_group,
            experts,
            grouped,
            rows,
            routes_per_token,
        ), route_ctas in _CUBIC_MOE_ROUTE_CTA_TACTICS.items()
        if dev == device
        and a8 == dynamic_a8
        and bits == num_bits
        and nn == n
        and kk == k
        and group == group_size
        and output_group == group_out
        and experts == local_experts
        and grouped == grouped_routes
        and routes_per_token == top_k
    ]
    return min(matching, key=lambda item: item[0])[1] if matching else fallback


def _cubic_w2_a8_situ_tactic(
    n: int, k: int, group_size: int, local_experts: int, route_ctas: int
) -> tuple[int, int]:
    device = torch.accelerator.current_device_index()
    exact = (device, n, k, group_size, local_experts, route_ctas)
    if exact in _CUBIC_W2_A8_SITU_TACTICS:
        return _CUBIC_W2_A8_SITU_TACTICS[exact]
    matching = [
        (abs(bucket - route_ctas), config)
        for (
            dev,
            nn,
            kk,
            group,
            experts,
            bucket,
        ), config in _CUBIC_W2_A8_SITU_TACTICS.items()
        if dev == device
        and nn == n
        and kk == k
        and group == group_size
        and experts == local_experts
    ]
    if matching:
        return min(matching, key=lambda item: item[0])[1]
    # Safe pre-warmup/eager fallback. It is not an architecture verdict: the
    # warmup calibration replaces it on every supported CUDA device.
    return (16, 2) if current_platform.is_device_capability((9, 0)) else (32, 4)


@torch.inference_mode()
def calibrate_cubic_w2_a8_situ(
    *,
    n: int,
    k: int,
    group_size: int,
    top_k: int,
    local_experts: int,
    route_ctas_values: tuple[int, ...],
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Measure W2 Dynamic-A8 SITU tactics using controlled route densities."""
    if k % group_size or k % 4:
        return
    device = torch.accelerator.current_device_index()
    max_route_ctas = max(route_ctas_values)
    input_rows = max(max_route_ctas, math.ceil(3 * max_route_ctas / max(top_k, 1)))
    inputs_q = torch.randint(
        -127, 128, (input_rows, k), device="cuda", dtype=torch.int8
    )
    input_words = inputs_q.view(torch.int32)
    input_scale = torch.rand(input_rows, 1, device="cuda", dtype=torch.float32)
    free_bytes, _ = torch.accelerator.get_memory_info()
    bytes_per_expert = 2 * n * (k // 4) + 2 * n * (k // group_size) * 4
    tuning_budget = min(512 * 1024**2, max(32 * 1024**2, free_bytes // 20))
    synthetic_experts = min(
        local_experts, max(1, tuning_budget // max(bytes_per_expert, 1))
    )
    packed = torch.randint(
        0,
        256,
        (synthetic_experts, 2 * n, k // 4),
        device="cuda",
        dtype=torch.uint8,
    )
    packed_words = packed.view(torch.int32)
    scale = torch.rand(
        synthetic_experts,
        2 * n,
        k // group_size,
        device="cuda",
        dtype=torch.float32,
    )

    for route_ctas in route_ctas_values:
        # Include one through three route-block waves per CTA. Sparse EP
        # routing plus per-expert padding commonly crosses the two-wave
        # boundary even when the unpadded expectation does not.
        valid_counts = (2 * route_ctas, 3 * route_ctas)
        max_routes = max(valid_counts)
        route_span = max(route_ctas * top_k, max_routes)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        shuffled_ids = torch.randperm(
            route_span, generator=generator, device="cuda", dtype=torch.int32
        )
        offsets = torch.arange(max_routes, device="cuda", dtype=torch.int32)
        # Group-size two alignment gives each expert block one real route,
        # while its second lane is padding for an odd expert route count.  A
        # deterministic 70% valid-lane ratio represents sparse routing without
        # reading production tokens or router decisions.
        valid_mask = (offsets.remainder(2) == 0) | (
            torch.div(offsets, 2, rounding_mode="floor").remainder(5) < 2
        )
        sorted_ids = torch.full(
            (max_routes,), route_span, device="cuda", dtype=torch.int32
        )
        num_synthetic_valid = int(valid_mask.sum().item())
        sorted_ids[valid_mask] = shuffled_ids[:num_synthetic_valid]
        route_blocks = math.ceil(max_routes / 2)
        active_experts = min(synthetic_experts, math.ceil(route_blocks * 0.9))
        expert_ids = torch.arange(
            max_routes, device="cuda", dtype=torch.int32
        ).remainder_(active_experts)
        topk_weights = torch.ones(route_span, device="cuda", dtype=torch.float32)
        count = torch.empty(1, device="cuda", dtype=torch.int32)
        output = torch.empty(route_span, n, device="cuda", dtype=torch.bfloat16)
        scores: list[tuple[float, tuple[int, int]]] = []
        references: dict[int, torch.Tensor] = {}

        for block_n, num_warps in _CUBIC_W2_A8_SITU_CANDIDATES:
            grid = (triton.cdiv(n, block_n), route_ctas)

            def launch(
                grid=grid,
                output=output,
                topk_weights=topk_weights,
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                count=count,
                block_n=block_n,
                route_ctas=route_ctas,
                num_warps=num_warps,
            ) -> None:
                _cubic_moe_pair_situ_gemv_2bit_a8_dp4a_kernel[grid](
                    input_words,
                    input_scale,
                    packed_words,
                    scale,
                    output,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    n,
                    scale.shape[2],
                    topk_weights.numel(),
                    input_words.stride(0),
                    input_words.stride(1),
                    packed_words.stride(0),
                    packed_words.stride(1),
                    packed_words.stride(2),
                    scale.stride(0),
                    scale.stride(1),
                    scale.stride(2),
                    output.stride(0),
                    output.stride(1),
                    output.stride(0),
                    output.stride(1),
                    GROUP_SIZE=group_size,
                    ACTIVATION_GROUP_SIZE=16,
                    BLOCK_N=block_n,
                    ROUTE_CTAS=route_ctas,
                    MUL_ROUTED_WEIGHT=True,
                    TOP_K=top_k,
                    BETA=4.0,
                    LINEAR_BETA=1.0,
                    HAS_LINEAR_BETA=False,
                    OUTPUT_GROUPWISE_A8=False,
                    OUTPUT_BF16=True,
                    num_warps=num_warps,
                    num_stages=1,
                )

            timings = []
            for valid_count in valid_counts:
                count.fill_(valid_count)
                launch()
                torch.accelerator.synchronize()
                candidate_ids = sorted_ids[:valid_count]
                candidate_ids = candidate_ids[candidate_ids < route_span]
                candidate_output = output.index_select(0, candidate_ids.long())
                reference = references.get(valid_count)
                if reference is None:
                    references[valid_count] = candidate_output
                else:
                    torch.testing.assert_close(candidate_output, reference)
                timings.append(
                    triton.testing.do_bench(
                        launch,
                        warmup=20,
                        rep=60,
                    )
                )
            scores.append((sum(timings) / len(timings), (block_n, num_warps)))

        measured_best_ms = min(score for score, _ in scores)
        # Timing noise can otherwise make identical ranks choose different
        # tactics. Within one percent of the measured winner, prefer the
        # lowest-resource candidate order above; larger wins remain honored.
        near_best = [
            (score, config)
            for score, config in scores
            if score <= measured_best_ms * 1.01
        ]
        candidate_priority = {
            config: index for index, config in enumerate(_CUBIC_W2_A8_SITU_CANDIDATES)
        }
        best_ms, best = min(near_best, key=lambda item: candidate_priority[item[1]])
        _CUBIC_W2_A8_SITU_TACTICS[
            (device, n, k, group_size, local_experts, route_ctas)
        ] = best
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "Cubic W2 A8 SITU tactic: N=%d K=%d G=%d experts=%d/%d "
            "route_ctas=%d "
            "BLOCK_N=%d warps=%d (%.4f ms)",
            n,
            k,
            group_size,
            synthetic_experts,
            local_experts,
            route_ctas,
            best[0],
            best[1],
            best_ms,
        )
        init_logger(__name__).debug("Cubic W2 A8 SITU candidate timings: %s", scores)
        if progress is not None:
            progress(
                route_ctas_values.index(route_ctas) + 1,
                len(route_ctas_values),
                f"route_ctas={route_ctas}",
            )


@triton.jit
def _cubic_compact_local_routes_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    count_ptr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOP_K
    global_ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    id_mask = mask & (global_ids >= 0) & (global_ids < NUM_EXPERTS)
    local_ids = tl.load(expert_map_ptr + global_ids, mask=id_mask, other=-1)
    valid = mask & (local_ids >= 0)
    positions = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    tl.store(sorted_ids_ptr + positions, offsets, mask=valid)
    tl.store(expert_ids_ptr + positions, local_ids, mask=valid)
    tl.store(count_ptr, tl.sum(valid.to(tl.int32), axis=0))


def cubic_compact_local_routes(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = topk_ids.shape[1]
    if topk_ids.shape[0] != 1 or top_k > 32:
        raise ValueError(
            "Cubic local route compaction requires one token and top_k <= 32."
        )
    sorted_ids = torch.empty(top_k, device=topk_ids.device, dtype=torch.int32)
    expert_ids = torch.empty_like(sorted_ids)
    count = torch.empty(1, device=topk_ids.device, dtype=torch.int32)
    _cubic_compact_local_routes_kernel[(1,)](
        topk_ids,
        expert_map,
        sorted_ids,
        expert_ids,
        count,
        NUM_EXPERTS=expert_map.numel(),
        TOP_K=top_k,
        BLOCK_SIZE=triton.next_power_of_2(top_k),
        num_warps=1,
        num_stages=1,
    )
    return sorted_ids, expert_ids, count


def _cubic_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    global_num_experts: int,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return moe_align_block_size(
        topk_ids,
        block_size,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )


@triton.jit
def _decode_cubic_direct(
    packed_low,
    packed_high,
    shifts,
    scale,
    a,
    b,
    NUM_BITS: tl.constexpr,
):
    raw = ((packed_low >> shifts) | (packed_high << (8 - shifts))) & (
        (1 << NUM_BITS) - 1
    )
    if NUM_BITS == 1:
        sign = raw.to(tl.float32) * 2.0 - 1.0
        return sign * scale
    sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
    signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
    signed = tl.where(signed == -sign_bit, 0, signed)
    magnitude_max: tl.constexpr = sign_bit - 1
    signed_f32 = signed.to(tl.float32)
    t = tl.abs(signed_f32) / magnitude_max
    c = 1.0 - a - b
    normalized = t * (a + t * (b + t * c))
    sign = tl.where(signed_f32 < 0, -1.0, tl.where(signed_f32 > 0, 1.0, 0.0))
    return sign * (scale * normalized)


@triton.jit
def _decode_cubic_lut(
    packed_low,
    packed_high,
    shifts,
    scale,
    a,
    b,
    NUM_BITS: tl.constexpr,
):
    raw = ((packed_low >> shifts) | (packed_high << (8 - shifts))) & (
        (1 << NUM_BITS) - 1
    )
    if NUM_BITS == 1:
        sign = raw.to(tl.float32) * 2.0 - 1.0
        return sign * scale

    sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
    signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
    signed = tl.where(signed == -sign_bit, 0, signed)
    magnitude = tl.abs(signed)
    reconstructed = raw.to(tl.float32) * 0.0
    if NUM_BITS == 2:
        reconstructed = tl.where(magnitude == 1, scale, reconstructed)
    else:
        magnitude_max: tl.constexpr = sign_bit - 1
        c = 1.0 - a - b
        for level_index in tl.static_range(1, magnitude_max + 1):
            level = scale * (
                (level_index / magnitude_max)
                * (
                    a
                    + (level_index / magnitude_max)
                    * (b + (level_index / magnitude_max) * c)
                )
            )
            reconstructed = tl.where(
                magnitude == level_index,
                level,
                reconstructed,
            )
    return tl.where(signed < 0, -reconstructed, reconstructed)


@triton.autotune(
    configs=_CUBIC_LINEAR_DENSE_CONFIGS,
    key=[
        "M",
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "GROUP_OUT",
        "BLOCK_K",
        "USE_GROUP_LUT",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_linear_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bp,
    stride_sm,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GROUP_LUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = offs_n_raw % N
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        k_mask = global_k < K
        activation = tl.load(
            a_ptr + offs_m[:, None] * stride_am + global_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        bit_positions = global_k[:, None] * NUM_BITS
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        packed_ptrs = b_ptr + offs_n[None, :] * stride_bn + byte_indices * stride_bp
        weight_mask = k_mask[:, None] & (offs_n_raw[None, :] < N)
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        if 8 % NUM_BITS == 0:
            high = 0
        else:
            high = tl.load(
                packed_ptrs + stride_bp,
                mask=weight_mask & (byte_indices + 1 < PACKED_K),
                other=0,
            ).to(tl.int32)
        if USE_GROUP_LUT:
            group = (k_block * BLOCK_K) // GROUP_SIZE
            metadata_ptrs = (offs_n // GROUP_OUT) * stride_sm + group * stride_sg
            metadata_mask = (offs_n_raw < N) & (group < NUM_GROUPS)
            scale = tl.load(
                scale_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=0.0,
            ).to(tl.float32)[None, :]
            if NUM_BITS > 2:
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_ptrs,
                    mask=metadata_mask,
                    other=1.0,
                ).to(tl.float32)[None, :]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_ptrs,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
            else:
                cubic_a = 1.0
                cubic_b = 0.0
            weight = _decode_cubic_lut(
                low,
                high,
                shifts,
                scale,
                cubic_a,
                cubic_b,
                NUM_BITS,
            ).to(activation.dtype)
        else:
            groups = global_k[:, None] // GROUP_SIZE
            metadata_ptrs = (
                (offs_n[None, :] // GROUP_OUT) * stride_sm + groups * stride_sg
            )
            metadata_mask = weight_mask & (groups < NUM_GROUPS)
            scale = tl.load(
                scale_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=0.0,
            )
            cubic_a = tl.load(
                cubic_a_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=1.0,
            ).to(tl.float32)
            cubic_b = tl.load(
                cubic_b_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=0.0,
            ).to(tl.float32)
            weight = _decode_cubic_direct(
                low,
                high,
                shifts,
                scale.to(tl.float32),
                cubic_a,
                cubic_b,
                NUM_BITS,
            ).to(activation.dtype)
        accumulator = tl.dot(activation, weight, acc=accumulator)

    output_offsets = (
        output_ptr + offs_m[:, None] * stride_om + offs_n_raw[None, :] * stride_on
    )
    tl.store(
        output_offsets,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n_raw[None, :] < N),
    )


@triton.autotune(
    configs=_CUBIC_LINEAR_GEMV_CONFIGS,
    key=["M", "N", "K", "NUM_BITS", "GROUP_SIZE"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_gemv_power2_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_wn,
    stride_ww,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    values_per_word: tl.constexpr = 32 // NUM_BITS
    group_words: tl.constexpr = GROUP_SIZE // values_per_word
    sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
    magnitude_max: tl.constexpr = sign_bit - 1
    offs_word = tl.arange(0, group_words)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group in tl.static_range(0, NUM_GROUPS):
        word_indices = group * group_words + offs_word
        packed = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + word_indices[None, :] * stride_ww,
            mask=n_mask[:, None],
            other=0,
        ).to(tl.int32)
        metadata_offsets = offs_n * stride_sn + group * stride_sg
        scale = tl.load(
            scale_ptr + metadata_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        if NUM_BITS > 2:
            cubic_a = tl.load(
                cubic_a_ptr + metadata_offsets,
                mask=n_mask,
                other=1.0,
            ).to(tl.float32)
            cubic_b = tl.load(
                cubic_b_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            cubic_c = 1.0 - cubic_a - cubic_b
        contribution = tl.zeros((BLOCK_N, group_words), dtype=tl.float32)
        group_k = group * GROUP_SIZE + offs_word * values_per_word
        for lane in tl.static_range(0, values_per_word):
            global_k = group_k + lane
            activation = tl.load(
                input_ptr + row * stride_im + global_k * stride_ik,
                mask=(row < M) & (global_k < K),
                other=0.0,
            ).to(tl.float32)
            code = (packed >> (lane * NUM_BITS)) & ((1 << NUM_BITS) - 1)
            if NUM_BITS == 1:
                weight = code * 2.0 - 1.0
            else:
                signed = tl.where(code >= sign_bit, code - (1 << NUM_BITS), code)
                signed = tl.where(signed == -sign_bit, 0, signed)
                if NUM_BITS == 2:
                    weight = signed.to(tl.float32)
                else:
                    signed_f32 = signed.to(tl.float32)
                    t = tl.abs(signed_f32) / magnitude_max
                    weight = t * (
                        cubic_a[:, None] + t * (cubic_b[:, None] + t * cubic_c[:, None])
                    )
                    weight = tl.where(signed < 0, -weight, weight)
            contribution += weight * activation[None, :]
        accumulator += scale * tl.sum(contribution, axis=1)

    tl.store(
        output_ptr + row * stride_om + offs_n * stride_on,
        accumulator,
        mask=(row < M) & n_mask,
    )


@triton.autotune(
    configs=_CUBIC_LINEAR_GEMV_CONFIGS,
    key=["M", "N", "K", "NUM_BITS", "GROUP_SIZE", "GROUP_OUT", "BLOCK_K"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_gemv_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wp,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        k_mask = global_k < K
        activation = tl.load(
            input_ptr + row * stride_im + global_k * stride_ik,
            mask=(row < M) & k_mask,
            other=0.0,
        ).to(tl.float32)
        bit_positions = global_k[None, :] * NUM_BITS
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        packed_ptrs = (
            weight_ptr + offs_n[:, None] * stride_wn + byte_indices * stride_wp
        )
        weight_mask = n_mask[:, None] & k_mask[None, :]
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        if 8 % NUM_BITS == 0:
            high = 0
        else:
            high = tl.load(
                packed_ptrs + stride_wp,
                mask=weight_mask & (byte_indices + 1 < PACKED_K),
                other=0,
            ).to(tl.int32)
        raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)
        group = (k_block * BLOCK_K) // GROUP_SIZE
        metadata_offsets = (offs_n // GROUP_OUT) * stride_sn + group * stride_sg
        metadata_mask = n_mask & (group < NUM_GROUPS)
        weight_scale = tl.load(
            scale_ptr + metadata_offsets, mask=metadata_mask, other=0.0
        ).to(tl.float32)
        if NUM_BITS == 1:
            weight = raw.to(tl.float32) * 2.0 - 1.0
        else:
            sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
            signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
            signed = tl.where(signed == -sign_bit, 0, signed)
            if NUM_BITS == 2:
                weight = signed.to(tl.float32)
            else:
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=1.0,
                ).to(tl.float32)[:, None]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)[:, None]
                signed_f32 = signed.to(tl.float32)
                magnitude_max: tl.constexpr = sign_bit - 1
                t = tl.abs(signed_f32) / magnitude_max
                weight = t * (cubic_a + t * (cubic_b + t * (1.0 - cubic_a - cubic_b)))
                weight = tl.where(signed < 0, -weight, weight)
        accumulator += weight_scale * tl.sum(weight * activation[None, :], axis=1)

    tl.store(
        output_ptr + row * stride_om + offs_n * stride_on,
        accumulator,
        mask=(row < M) & n_mask,
    )


def cubic_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    input_size: int,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty(x_2d.shape[0], packed.shape[0], device=x.device, dtype=x.dtype)
    gemv_eligible = (
        x_2d.shape[0] <= 8
        and group_size in (32, 64, 128, 256, 512)
        and input_size % group_size == 0
    )
    use_gemv = _cubic_linear_use_gemv(
        dynamic_a8=False,
        num_bits=num_bits,
        n=packed.shape[0],
        k=input_size,
        group_size=group_size,
        group_out=group_out,
        m=x_2d.shape[0],
        fallback=gemv_eligible,
    )
    if (
        use_gemv
        and group_out == 1
        and num_bits in (1, 2, 4, 8)
        and group_size in (128, 256, 512)
    ):
        packed_words = packed.view(torch.int32)
        grid = lambda meta: (
            triton.cdiv(packed.shape[0], meta["BLOCK_N"]),
            x_2d.shape[0],
        )
        _cubic_linear_gemv_power2_kernel[grid](
            x_2d,
            packed_words,
            scale,
            a,
            b,
            output,
            x_2d.shape[0],
            packed.shape[0],
            input_size,
            scale.shape[1],
            x_2d.stride(0),
            x_2d.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            scale.stride(0),
            scale.stride(1),
            output.stride(0),
            output.stride(1),
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
        )
        return output.reshape(*x.shape[:-1], packed.shape[0])
    if use_gemv:
        block_k = 32 if group_size == 1 else min(group_size, 128)
        grid = lambda meta: (
            triton.cdiv(packed.shape[0], meta["BLOCK_N"]),
            x_2d.shape[0],
        )
        _cubic_linear_gemv_kernel[grid](
            x_2d,
            packed,
            scale,
            a,
            b,
            output,
            x_2d.shape[0],
            packed.shape[0],
            input_size,
            packed.shape[1],
            scale.shape[1],
            x_2d.stride(0),
            x_2d.stride(1),
            packed.stride(0),
            packed.stride(1),
            scale.stride(0),
            scale.stride(1),
            output.stride(0),
            output.stride(1),
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            GROUP_OUT=group_out,
            BLOCK_K=block_k,
        )
        return output.reshape(*x.shape[:-1], packed.shape[0])
    block_k = 64 if group_size % 64 == 0 else 32
    use_group_lut = group_size >= block_k and group_size % block_k == 0
    grid = lambda meta: (
        triton.cdiv(x_2d.shape[0], meta["BLOCK_M"]),
        triton.cdiv(packed.shape[0], meta["BLOCK_N"]),
    )
    _cubic_linear_kernel[grid](
        x_2d,
        packed,
        scale,
        a,
        b,
        output,
        x_2d.shape[0],
        packed.shape[0],
        input_size,
        packed.shape[1],
        scale.shape[1],
        x_2d.stride(0),
        x_2d.stride(1),
        packed.stride(0),
        packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        output.stride(-2),
        output.stride(1),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        GROUP_OUT=group_out,
        BLOCK_K=block_k,
        USE_GROUP_LUT=use_group_lut and num_bits <= 4,
    )
    return output.reshape(*x.shape[:-1], packed.shape[0])


@triton.autotune(
    configs=_CUBIC_LINEAR_DENSE_CONFIGS,
    key=["M", "N", "K", "NUM_BITS", "GROUP_SIZE", "BLOCK_K"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_dynamic_a8_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bp,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = offs_n_raw % N
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    activation_scale = tl.load(
        a_scale_ptr + offs_m,
        mask=offs_m < M,
        other=0.0,
    ).to(tl.float32)[:, None]

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        k_mask = global_k < K
        activation = tl.load(
            a_ptr + offs_m[:, None] * stride_am + global_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0,
        )

        bit_positions = global_k[:, None] * NUM_BITS
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        packed_ptrs = b_ptr + offs_n[None, :] * stride_bn + byte_indices * stride_bp
        weight_mask = k_mask[:, None] & (offs_n_raw[None, :] < N)
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        if 8 % NUM_BITS == 0:
            high = 0
        else:
            high = tl.load(
                packed_ptrs + stride_bp,
                mask=weight_mask & (byte_indices + 1 < PACKED_K),
                other=0,
            ).to(tl.int32)
        raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)

        group = (k_block * BLOCK_K) // GROUP_SIZE
        metadata_offsets = offs_n * stride_sn + group * stride_sg
        metadata_mask = (offs_n_raw < N) & (group < NUM_GROUPS)
        weight_scale = tl.load(
            scale_ptr + metadata_offsets,
            mask=metadata_mask,
            other=0.0,
        ).to(tl.float32)[None, :]
        if NUM_BITS == 1:
            carrier = (raw * 254 - 127).to(tl.int8)
        else:
            sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
            signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
            signed = tl.where(signed == -sign_bit, 0, signed)
            if NUM_BITS == 2:
                carrier = (signed * 127).to(tl.int8)
            else:
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=1.0,
                ).to(tl.float32)[None, :]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
                signed_f32 = signed.to(tl.float32)
                magnitude_max: tl.constexpr = sign_bit - 1
                t = tl.abs(signed_f32) / magnitude_max
                normalized = t * (
                    cubic_a + t * (cubic_b + t * (1.0 - cubic_a - cubic_b))
                )
                normalized = tl.where(signed < 0, -normalized, normalized)
                carrier_f32 = tl.extra.cuda.libdevice.rint(normalized * 127.0)
                carrier = tl.maximum(tl.minimum(carrier_f32, 127.0), -127.0).to(tl.int8)

        partial = tl.dot(activation, carrier, out_dtype=tl.int32)
        accumulator += (
            partial.to(tl.float32) * activation_scale * weight_scale * (1.0 / 127.0)
        )

    output_offsets = (
        output_ptr + offs_m[:, None] * stride_om + offs_n_raw[None, :] * stride_on
    )
    tl.store(
        output_offsets,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n_raw[None, :] < N),
    )


@triton.jit
def _cubic_dynamic_a8_carrier(
    raw,
    cubic_a,
    cubic_b,
    NUM_BITS: tl.constexpr,
):
    if NUM_BITS == 1:
        return (raw * 254 - 127).to(tl.int8)
    sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
    signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
    signed = tl.where(signed == -sign_bit, 0, signed)
    if NUM_BITS == 2:
        return (signed * 127).to(tl.int8)
    signed_f32 = signed.to(tl.float32)
    magnitude_max: tl.constexpr = sign_bit - 1
    t = tl.abs(signed_f32) / magnitude_max
    normalized = t * (cubic_a + t * (cubic_b + t * (1.0 - cubic_a - cubic_b)))
    normalized = tl.where(signed < 0, -normalized, normalized)
    carrier_f32 = tl.extra.cuda.libdevice.rint(normalized * 127.0)
    return tl.maximum(tl.minimum(carrier_f32, 127.0), -127.0).to(tl.int8)


@triton.jit
def _cubic_dynamic_a8_carrier_lut(
    raw,
    cubic_a,
    cubic_b,
    NUM_BITS: tl.constexpr,
):
    sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
    magnitude_max: tl.constexpr = sign_bit - 1
    signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
    signed = tl.where(signed == -sign_bit, 0, signed)
    magnitude = tl.abs(signed)
    carrier = raw.to(tl.float32) * 0.0
    cubic_c = 1.0 - cubic_a - cubic_b
    for level_index in tl.static_range(1, magnitude_max + 1):
        level = tl.extra.cuda.libdevice.rint(
            127.0
            * (level_index / magnitude_max)
            * (
                cubic_a
                + (level_index / magnitude_max)
                * (cubic_b + (level_index / magnitude_max) * cubic_c)
            )
        )
        carrier = tl.where(magnitude == level_index, level, carrier)
    carrier = tl.where(signed < 0, -carrier, carrier)
    return tl.maximum(tl.minimum(carrier, 127.0), -127.0).to(tl.int8)


@triton.autotune(
    configs=_CUBIC_LINEAR_GEMV_CONFIGS,
    key=["M", "N", "K", "NUM_BITS", "GROUP_SIZE", "BLOCK_K"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_dynamic_a8_gemv_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wp,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    activation_scale = tl.load(
        input_scale_ptr + row,
        mask=row < M,
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        k_mask = global_k < K
        activation = tl.load(
            input_ptr + row * stride_im + global_k * stride_ik,
            mask=(row < M) & k_mask,
            other=0,
        ).to(tl.int32)
        bit_positions = global_k[None, :] * NUM_BITS
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        packed_ptrs = (
            weight_ptr + offs_n[:, None] * stride_wn + byte_indices * stride_wp
        )
        weight_mask = n_mask[:, None] & k_mask[None, :]
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        if 8 % NUM_BITS == 0:
            high = 0
        else:
            high = tl.load(
                packed_ptrs + stride_wp,
                mask=weight_mask & (byte_indices + 1 < PACKED_K),
                other=0,
            ).to(tl.int32)
        raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)
        group = (k_block * BLOCK_K) // GROUP_SIZE
        metadata_offsets = offs_n * stride_sn + group * stride_sg
        metadata_mask = n_mask & (group < NUM_GROUPS)
        weight_scale = tl.load(
            scale_ptr + metadata_offsets,
            mask=metadata_mask,
            other=0.0,
        ).to(tl.float32)
        if NUM_BITS > 2:
            cubic_a = tl.load(
                cubic_a_ptr + metadata_offsets,
                mask=metadata_mask,
                other=1.0,
            ).to(tl.float32)[:, None]
            cubic_b = tl.load(
                cubic_b_ptr + metadata_offsets,
                mask=metadata_mask,
                other=0.0,
            ).to(tl.float32)[:, None]
        else:
            cubic_a = 1.0
            cubic_b = 0.0
        if NUM_BITS > 2 and NUM_BITS <= 4:
            carrier = _cubic_dynamic_a8_carrier_lut(
                raw,
                cubic_a,
                cubic_b,
                NUM_BITS,
            )
        else:
            carrier = _cubic_dynamic_a8_carrier(raw, cubic_a, cubic_b, NUM_BITS)
        partial = tl.sum(carrier.to(tl.int32) * activation[None, :], axis=1)
        accumulator += (
            partial.to(tl.float32) * activation_scale * weight_scale * (1.0 / 127.0)
        )

    tl.store(
        output_ptr + row * stride_om + offs_n * stride_on,
        accumulator,
        mask=(row < M) & n_mask,
    )


def cubic_linear_dynamic_a8(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    """Apply Cubic Linear with runtime per-token INT8 activations."""
    if num_bits not in range(1, 9):
        raise ValueError(f"Unsupported Cubic bit width: {num_bits}.")
    if group_size not in (32, 64, 128, 256, 512):
        raise ValueError(f"Unsupported Cubic Dynamic A8 group size: {group_size}.")
    if x.shape[-1] != input_size:
        raise ValueError(
            f"Cubic Dynamic A8 expected input width {input_size}, got {x.shape[-1]}."
        )
    if scale.dtype != torch.float32:
        raise ValueError("Cubic Dynamic A8 weight scales must remain FP32.")

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_q, x_scale = per_token_quant_int8(x_2d)
    output = torch.empty(
        x_2d.shape[0],
        packed.shape[0],
        device=x.device,
        dtype=x.dtype,
    )
    use_gemv = _cubic_linear_use_gemv(
        dynamic_a8=True,
        num_bits=num_bits,
        n=packed.shape[0],
        k=input_size,
        group_size=group_size,
        group_out=1,
        m=x_2d.shape[0],
        fallback=packed.shape[0] == 1 or x_2d.shape[0] <= 8,
    )
    if use_gemv:
        block_k = min(group_size, 128)
        grid = lambda meta: (
            triton.cdiv(packed.shape[0], meta["BLOCK_N"]),
            x_2d.shape[0],
        )
        _cubic_linear_dynamic_a8_gemv_kernel[grid](
            x_q,
            x_scale,
            packed,
            scale,
            a,
            b,
            output,
            x_2d.shape[0],
            packed.shape[0],
            input_size,
            packed.shape[1],
            scale.shape[1],
            x_q.stride(0),
            x_q.stride(1),
            packed.stride(0),
            packed.stride(1),
            scale.stride(0),
            scale.stride(1),
            output.stride(0),
            output.stride(1),
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            BLOCK_K=block_k,
        )
        return output.reshape(*x.shape[:-1], packed.shape[0])
    block_k = 32
    grid = lambda meta: (
        triton.cdiv(x_2d.shape[0], meta["BLOCK_M"]),
        triton.cdiv(packed.shape[0], meta["BLOCK_N"]),
    )
    _cubic_linear_dynamic_a8_kernel[grid](
        x_q,
        x_scale,
        packed,
        scale,
        a,
        b,
        output,
        x_2d.shape[0],
        packed.shape[0],
        input_size,
        packed.shape[1],
        scale.shape[1],
        x_q.stride(0),
        x_q.stride(1),
        packed.stride(0),
        packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
    )
    return output.reshape(*x.shape[:-1], packed.shape[0])


@triton.autotune(
    configs=_CUBIC_LINEAR_DENSE_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE", "BLOCK_K"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_precomputed_a8_kernel(
    activation_ptr,
    activation_scale_ptr,
    carrier_ptr,
    weight_scale_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    activation_scale = tl.load(
        activation_scale_ptr + offs_m,
        mask=offs_m < M,
        other=0.0,
    ).to(tl.float32)[:, None]

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        activation = tl.load(
            activation_ptr
            + offs_m[:, None] * stride_am
            + global_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (global_k[None, :] < K),
            other=0,
        )
        weight = tl.load(
            carrier_ptr + offs_n[None, :] * stride_wn + global_k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (global_k[:, None] < K),
            other=0,
        )
        partial = tl.dot(activation, weight, out_dtype=tl.int32)
        group = (k_block * BLOCK_K) // GROUP_SIZE
        weight_scale = tl.load(
            weight_scale_ptr + offs_n * stride_sn + group * stride_sg,
            mask=(offs_n < N) & (group < NUM_GROUPS),
            other=0.0,
        ).to(tl.float32)[None, :]
        accumulator += (
            partial.to(tl.float32) * activation_scale * weight_scale * (1.0 / 127.0)
        )

    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=_CUBIC_LINEAR_GEMV_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE", "BLOCK_K"],
    cache_results=True,
)
@triton.jit
def _cubic_linear_precomputed_a8_gemv_kernel(
    activation_ptr,
    activation_scale_ptr,
    carrier_ptr,
    weight_scale_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    activation_scale = tl.load(
        activation_scale_ptr + row,
        mask=row < M,
        other=0.0,
    ).to(tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        activation = tl.load(
            activation_ptr + row * stride_am + global_k * stride_ak,
            mask=(row < M) & (global_k < K),
            other=0,
        ).to(tl.int32)
        weight = tl.load(
            carrier_ptr + offs_n[:, None] * stride_wn + global_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (global_k[None, :] < K),
            other=0,
        ).to(tl.int32)
        partial = tl.sum(weight * activation[None, :], axis=1)
        group = (k_block * BLOCK_K) // GROUP_SIZE
        weight_scale = tl.load(
            weight_scale_ptr + offs_n * stride_sn + group * stride_sg,
            mask=(offs_n < N) & (group < NUM_GROUPS),
            other=0.0,
        ).to(tl.float32)
        accumulator += (
            partial.to(tl.float32) * activation_scale * weight_scale * (1.0 / 127.0)
        )

    tl.store(
        output_ptr + row * stride_om + offs_n * stride_on,
        accumulator,
        mask=(row < M) & (offs_n < N),
    )


def cubic_w8_precompute_carrier(
    packed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    """Materialize the deterministic W8 INT8 carrier once during loading."""
    if packed.shape[1] != input_size or packed.dtype != torch.uint8:
        raise ValueError("Cubic W8 carrier requires one packed byte per weight.")
    codes = packed.view(torch.int8).to(torch.float32)
    codes = torch.where(codes == -128, 0.0, codes)
    groups = torch.arange(input_size, device=packed.device) // group_size
    coefficient_a = a[:, groups].to(torch.float32)
    coefficient_b = b[:, groups].to(torch.float32)
    t = codes.abs() * (1.0 / 127.0)
    normalized = t * (
        coefficient_a + t * (coefficient_b + t * (1.0 - coefficient_a - coefficient_b))
    )
    return torch.clamp(torch.round(codes.sign() * normalized * 127.0), -127, 127).to(
        torch.int8
    )


def cubic_linear_dynamic_a8_precomputed(
    x: torch.Tensor,
    carrier: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    """Apply Dynamic-A8 using a load-time W8 carrier."""
    del a, b
    if num_bits != 8 or carrier.dtype != torch.int8:
        raise ValueError("Precomputed Cubic Linear requires an INT8 W8 carrier.")
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_q, x_scale = per_token_quant_int8(x_2d)
    output = torch.empty(
        x_2d.shape[0], carrier.shape[0], device=x.device, dtype=x.dtype
    )
    use_gemv = _cubic_linear_use_gemv(
        dynamic_a8=True,
        num_bits=num_bits,
        n=carrier.shape[0],
        k=input_size,
        group_size=group_size,
        group_out=1,
        m=x_2d.shape[0],
        fallback=x_2d.shape[0] <= 8,
    )
    block_k = min(group_size, 128)
    if use_gemv:
        grid = lambda meta: (
            triton.cdiv(carrier.shape[0], meta["BLOCK_N"]),
            x_2d.shape[0],
        )
        _cubic_linear_precomputed_a8_gemv_kernel[grid](
            x_q,
            x_scale,
            carrier,
            scale,
            output,
            x_2d.shape[0],
            carrier.shape[0],
            input_size,
            scale.shape[1],
            x_q.stride(0),
            x_q.stride(1),
            carrier.stride(0),
            carrier.stride(1),
            scale.stride(0),
            scale.stride(1),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=group_size,
            BLOCK_K=block_k,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(x_2d.shape[0], meta["BLOCK_M"]),
            triton.cdiv(carrier.shape[0], meta["BLOCK_N"]),
        )
        _cubic_linear_precomputed_a8_kernel[grid](
            x_q,
            x_scale,
            carrier,
            scale,
            output,
            x_2d.shape[0],
            carrier.shape[0],
            input_size,
            scale.shape[1],
            x_q.stride(0),
            x_q.stride(1),
            carrier.stride(0),
            carrier.stride(1),
            scale.stride(0),
            scale.stride(1),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=group_size,
            BLOCK_K=block_k,
        )
    return output.reshape(*x.shape[:-1], carrier.shape[0])


@torch.inference_mode()
def calibrate_cubic_linear_execution(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    input_size: int,
    dynamic_a8: bool,
    precomputed_carrier: bool = False,
) -> float | None:
    """Measure Cubic Linear GEMV versus dense execution at one real shape."""
    key = (
        torch.accelerator.current_device_index(),
        dynamic_a8,
        num_bits,
        packed.shape[0],
        input_size,
        group_size,
        group_out,
        x.reshape(-1, x.shape[-1]).shape[0],
    )
    if precomputed_carrier:
        if not dynamic_a8:
            raise ValueError("A precomputed carrier is only valid for Dynamic-A8.")
        func = cubic_linear_dynamic_a8_precomputed
    else:
        func = cubic_linear_dynamic_a8 if dynamic_a8 else cubic_linear
    candidates = (
        (True, False) if dynamic_a8 or input_size % group_size == 0 else (False,)
    )
    reference: torch.Tensor | None = None
    scores: list[tuple[float, bool]] = []
    for use_gemv in candidates:
        _CUBIC_LINEAR_EXECUTION_TACTICS[key] = use_gemv

        def launch() -> torch.Tensor:
            kwargs: dict[str, Any] = dict(
                num_bits=num_bits,
                group_size=group_size,
                input_size=input_size,
            )
            if not dynamic_a8:
                kwargs["group_out"] = group_out
            return func(x, packed, scale, a, b, **kwargs)

        try:
            output = launch()
            torch.accelerator.synchronize()
            if reference is None:
                reference = output
            else:
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
            large = x.reshape(-1, x.shape[-1]).shape[0] > 256
            score = triton.testing.do_bench(
                launch,
                warmup=5 if large else 10,
                rep=10 if large else 30,
            )
            scores.append((score, use_gemv))
        except (RuntimeError, AssertionError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic Linear %s W%d %s N=%d K=%d M=%d: %s",
                "A8" if dynamic_a8 else "A16",
                num_bits,
                "GEMV" if use_gemv else "dense",
                packed.shape[0],
                input_size,
                x.reshape(-1, x.shape[-1]).shape[0],
                error,
            )
    if not scores:
        _CUBIC_LINEAR_EXECUTION_TACTICS.pop(key, None)
        return None
    measured_best = min(score for score, _ in scores)
    near_best = [item for item in scores if item[0] <= measured_best * 1.01]
    best_score, best_use_gemv = min(near_best, key=lambda item: not item[1])
    _CUBIC_LINEAR_EXECUTION_TACTICS[key] = best_use_gemv
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Cubic Linear %s: W%d N=%d K=%d G=%d M=%d %s (%.4f ms)",
        "A8-carrier" if precomputed_carrier else ("A8" if dynamic_a8 else "A16"),
        num_bits,
        packed.shape[0],
        input_size,
        group_size,
        x.reshape(-1, x.shape[-1]).shape[0],
        "GEMV" if best_use_gemv else "dense",
        best_score,
    )
    return best_score


def _cubic_linear_custom_op(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    group_out: int,
    input_size: int,
) -> torch.Tensor:
    return cubic_linear(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        input_size=input_size,
    )


def _cubic_linear_custom_op_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    group_out: int,
    input_size: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


direct_register_custom_op(
    op_name="cubic_linear",
    op_func=_cubic_linear_custom_op,
    mutates_args=[],
    fake_impl=_cubic_linear_custom_op_fake,
)


def _cubic_linear_dynamic_a8_custom_op(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    return cubic_linear_dynamic_a8(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=num_bits,
        group_size=group_size,
        input_size=input_size,
    )


def _cubic_linear_dynamic_a8_custom_op_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


direct_register_custom_op(
    op_name="cubic_linear_dynamic_a8",
    op_func=_cubic_linear_dynamic_a8_custom_op,
    mutates_args=[],
    fake_impl=_cubic_linear_dynamic_a8_custom_op_fake,
)


def _cubic_linear_dynamic_a8_precomputed_custom_op(
    x: torch.Tensor,
    carrier: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    return cubic_linear_dynamic_a8_precomputed(
        x,
        carrier,
        scale,
        a,
        b,
        num_bits=num_bits,
        group_size=group_size,
        input_size=input_size,
    )


def _cubic_linear_dynamic_a8_precomputed_custom_op_fake(
    x: torch.Tensor,
    carrier: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_bits: int,
    group_size: int,
    input_size: int,
) -> torch.Tensor:
    del scale, a, b, num_bits, group_size, input_size
    return x.new_empty((*x.shape[:-1], carrier.shape[0]))


direct_register_custom_op(
    op_name="cubic_linear_dynamic_a8_precomputed",
    op_func=_cubic_linear_dynamic_a8_precomputed_custom_op,
    mutates_args=[],
    fake_impl=_cubic_linear_dynamic_a8_precomputed_custom_op_fake,
)


@triton.jit
def _cubic_situ_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    intermediate_size: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    rows = offsets // intermediate_size
    columns = offsets % intermediate_size
    input_offsets = rows * (2 * intermediate_size) + columns
    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + intermediate_size, mask=mask, other=0.0
    ).to(tl.float32)
    gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
    gate = beta * gate_tanh * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
    tl.store(output_ptr + offsets, gate * up, mask=mask)


@triton.jit
def _cubic_situ_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    intermediate_size: tl.constexpr,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < intermediate_size
    input_offsets = row * stride_im + columns * stride_in
    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + intermediate_size * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
    gate = beta * gate_tanh * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
    activated = (gate * up).to(input_ptr.dtype.element_ty).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(activated)), 1e-10)
    activation_scale = absmax / 127.0
    quantized = round_int8(activated * (127.0 / absmax))
    tl.store(
        output_ptr + row * stride_om + columns * stride_on,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row, activation_scale)


def _apply_cubic_situ_quant_int8(
    input: torch.Tensor,
    intermediate_size: int,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = input.numel() // input.shape[-1]
    input_2d = input.reshape(rows, input.shape[-1])
    output = torch.empty(
        rows,
        intermediate_size,
        device=input.device,
        dtype=torch.int8,
    )
    scale = torch.empty(rows, 1, device=input.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(intermediate_size)
    num_warps = min(max(block_size // 256, 1), 8)
    _cubic_situ_quant_int8_kernel[(rows,)](
        input_2d,
        output,
        scale,
        intermediate_size,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        output.stride(1),
        beta,
        linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output, scale


@triton.jit
def _cubic8_online_curve_parameters(candidate: tl.constexpr):
    if candidate == 0:
        return 1.0, 0.0
    if candidate == 1:
        return 0.5, 0.0
    if candidate == 2:
        return 0.5, 0.25
    if candidate == 3:
        return 0.75, -0.25
    return 1.0, 0.0


@triton.jit
def _cubic8_online_value(t, a, b):
    return t * (a + t * (b + t * (1.0 - a - b)))


@triton.jit
def _cubic8_online_nearest_code(normalized, a, b):
    """Invert a monotonic Cubic8 curve without a 128-entry distance tile."""
    t = normalized
    c = 1.0 - a - b
    for _ in tl.static_range(4):
        value = t * (a + t * (b + t * c))
        derivative = a + 2.0 * b * t + 3.0 * c * t * t
        t = tl.maximum(0.0, tl.minimum(1.0, t - (value - normalized) / derivative))
    center = tl.maximum(0, tl.minimum(127, (t * 127.0 + 0.5).to(tl.int32)))
    lower = tl.maximum(center - 1, 0)
    upper = tl.minimum(center + 1, 127)
    center_error = tl.abs(
        _cubic8_online_value(center.to(tl.float32) / 127.0, a, b) - normalized
    )
    lower_error = tl.abs(
        _cubic8_online_value(lower.to(tl.float32) / 127.0, a, b) - normalized
    )
    upper_error = tl.abs(
        _cubic8_online_value(upper.to(tl.float32) / 127.0, a, b) - normalized
    )
    code = tl.where(lower_error < center_error, lower, center)
    error = tl.minimum(lower_error, center_error)
    return tl.where(upper_error < error, upper, code)


@triton.jit
def _cubic8_fit_values(values, valid):
    """Fit and encode one runtime Cubic8 group already resident in a CTA."""
    absolute = tl.abs(values)
    amax = tl.maximum(tl.max(tl.where(valid, absolute, 0.0), axis=0), 1.0e-30)
    best_loss = float("inf")
    best_scale = amax
    best_a = 1.0
    best_b = 0.0
    best_codes = tl.zeros(values.shape, tl.int32)
    for candidate in tl.static_range(4):
        a, b = _cubic8_online_curve_parameters(candidate)
        codes = _cubic8_online_nearest_code(tl.minimum(absolute / amax, 1.0), a, b)
        q = _cubic8_online_value(codes.to(tl.float32) / 127.0, a, b)
        numerator = tl.sum(tl.where(valid, absolute * q, 0.0), axis=0)
        denominator = tl.sum(tl.where(valid, q * q, 0.0), axis=0)
        scale = tl.maximum(numerator / tl.maximum(denominator, 1.0e-30), 1.0e-30)
        codes = _cubic8_online_nearest_code(tl.minimum(absolute / scale, 1.0), a, b)
        q = _cubic8_online_value(codes.to(tl.float32) / 127.0, a, b)
        error = absolute - scale * q
        loss = tl.sum(tl.where(valid, error * error, 0.0), axis=0)
        improved = loss < best_loss
        best_loss = tl.where(improved, loss, best_loss)
        best_scale = tl.where(improved, scale, best_scale)
        best_a = tl.where(improved, a, best_a)
        best_b = tl.where(improved, b, best_b)
        best_codes = tl.where(improved, codes, best_codes)
    signed_codes = tl.where(values < 0.0, -best_codes, best_codes)
    return signed_codes, best_scale, best_a, best_b


@triton.jit
def _cubic8_groupwise_encode_kernel(
    input_ptr,
    codes_ptr,
    scale_ptr,
    a_ptr,
    b_ptr,
    width: tl.constexpr,
    stride_im,
    stride_ik,
    stride_cm,
    stride_ck,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
):
    """Fit one bounded Cubic8 curve for one sample/K-group."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, GROUP_SIZE)
    columns = group * GROUP_SIZE + offsets
    valid = columns < width
    values = tl.load(
        input_ptr + row * stride_im + columns * stride_ik,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    signed_codes, best_scale, best_a, best_b = _cubic8_fit_values(values, valid)
    tl.store(
        codes_ptr + row * stride_cm + columns * stride_ck,
        signed_codes,
        mask=valid,
    )
    metadata = row * stride_sm + group * stride_sg
    tl.store(scale_ptr + metadata, best_scale)
    tl.store(a_ptr + metadata, best_a)
    tl.store(b_ptr + metadata, best_b)


def _quantize_cubic_groupwise_cubic8(
    input: torch.Tensor,
    group_size: int,
) -> CubicA8Code:
    """Encode a float tensor as true per-sample/per-group Cubic8 codes."""
    if group_size not in (16, 32, 64, 128, 256, 512):
        raise ValueError(
            f"Online Cubic8 supports G16/G32/G64/G128/G256/G512, got G{group_size}."
        )
    if input.shape[-1] % group_size:
        raise ValueError(
            f"Online Cubic8 requires width divisible by G={group_size}, "
            f"got {input.shape[-1]}."
        )
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    codes = torch.empty_like(input_2d, dtype=torch.int8)
    metadata_shape = (input_2d.shape[0], input_2d.shape[1] // group_size)
    scales = torch.empty(metadata_shape, device=input.device, dtype=torch.float32)
    a = torch.empty(metadata_shape, device=input.device, dtype=torch.float16)
    b = torch.empty_like(a)
    _cubic8_groupwise_encode_kernel[metadata_shape](
        input_2d,
        codes,
        scales,
        a,
        b,
        input_2d.shape[1],
        input_2d.stride(0),
        input_2d.stride(1),
        codes.stride(0),
        codes.stride(1),
        scales.stride(0),
        scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=8,
        num_stages=1,
    )
    return CubicA8Code(codes, scales, a, b, group_size)


@triton.jit
def _cubic_groupwise_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    width: tl.constexpr,
    stride_im,
    stride_ik,
    stride_om,
    stride_ok,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = columns < width
    values = tl.load(
        input_ptr + row * stride_im + columns * stride_ik,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values)), 1e-10)
    activation_scale = absmax / 127.0
    quantized = round_int8(values * (127.0 / absmax))
    tl.store(
        output_ptr + row * stride_om + columns * stride_ok,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row * stride_sm + group * stride_sg, activation_scale)


def _quantize_cubic_groupwise_a8(
    input: torch.Tensor,
    group_size: int,
) -> CubicA8Carrier:
    """Quantize each sample/K-group independently without changing layout."""
    if input.shape[-1] % group_size:
        raise ValueError(
            f"Cubic groupwise A8 requires width divisible by G={group_size}, "
            f"got {input.shape[-1]}."
        )
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    output = torch.empty_like(input_2d, dtype=torch.int8)
    scales = torch.empty(
        input_2d.shape[0],
        input_2d.shape[1] // group_size,
        device=input.device,
        dtype=torch.float32,
    )
    _cubic_groupwise_quant_int8_kernel[(input_2d.shape[0], scales.shape[1])](
        input_2d,
        output,
        scales,
        input_2d.shape[1],
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=min(max(group_size // 64, 1), 8),
        num_stages=1,
    )
    return CubicA8Carrier(output, scales, group_size)


@triton.jit
def _cubic_situ_groupwise_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    intermediate_size: tl.constexpr,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    stride_sm,
    stride_sg,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = columns < intermediate_size
    input_offsets = row * stride_im + columns * stride_in
    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + intermediate_size * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
    gate = beta * gate_tanh * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
    activated = (gate * up).to(input_ptr.dtype.element_ty).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(activated)), 1e-10)
    activation_scale = absmax / 127.0
    quantized = round_int8(activated * (127.0 / absmax))
    tl.store(
        output_ptr + row * stride_om + columns * stride_on,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row * stride_sm + group * stride_sg, activation_scale)


def _apply_cubic_situ_groupwise_quant_int8(
    input: torch.Tensor,
    intermediate_size: int,
    beta: float,
    linear_beta: float | None,
    group_size: int,
) -> CubicA8Carrier:
    """Fuse SITU with per-sample/per-group A8 carrier production."""
    if intermediate_size % group_size:
        raise ValueError(
            "Cubic SITU groupwise A8 requires intermediate_size divisible "
            f"by G={group_size}, got {intermediate_size}."
        )
    rows = input.numel() // input.shape[-1]
    input_2d = input.reshape(rows, input.shape[-1])
    output = torch.empty(
        rows,
        intermediate_size,
        device=input.device,
        dtype=torch.int8,
    )
    scales = torch.empty(
        rows,
        intermediate_size // group_size,
        device=input.device,
        dtype=torch.float32,
    )
    _cubic_situ_groupwise_quant_int8_kernel[(rows, scales.shape[1])](
        input_2d,
        output,
        scales,
        intermediate_size,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(1),
        beta,
        linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        GROUP_SIZE=group_size,
        num_warps=min(max(group_size // 64, 1), 8),
        num_stages=1,
    )
    return CubicA8Carrier(output, scales, group_size)


@triton.jit
def _cubic_compact_situ_kernel(
    input_ptr,
    output_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    intermediate_size: tl.constexpr,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    route = tl.program_id(1)
    if route < tl.load(num_tokens_post_padded_ptr):
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        columns = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = columns < intermediate_size
        input_offsets = token_id * stride_im + columns * stride_in
        gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(
            input_ptr + input_offsets + intermediate_size * stride_in,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
        gate = beta * gate_tanh * tl.sigmoid(gate)
        if HAS_LINEAR_BETA:
            up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
        tl.store(
            output_ptr + token_id * stride_om + columns * stride_on,
            gate * up,
            mask=mask,
        )


def _apply_cubic_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> None:
    if activation != MoEActivation.SITU:
        apply_moe_activation(activation, output, input)
        return
    if activation_situ_beta is None:
        raise ValueError("Cubic SITU requires activation_situ_beta.")
    block_size = 256
    _cubic_situ_kernel[(triton.cdiv(output.numel(), block_size),)](
        input,
        output,
        output.numel(),
        output.shape[-1],
        activation_situ_beta,
        activation_situ_linear_beta or 1.0,
        HAS_LINEAR_BETA=activation_situ_linear_beta is not None,
        BLOCK_SIZE=block_size,
    )


@triton.autotune(
    configs=_CUBIC_MOE_GEMV_CONFIGS,
    key=[
        "N",
        "K",
        "GROUP_SIZE",
        "ROUTE_CTAS",
        "TOP_K",
        "SUM_ROUTES",
        "DYNAMIC_A8",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_3bit_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
    DYNAMIC_A8: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_packs: tl.constexpr = GROUP_SIZE // 8
    group_bytes: tl.constexpr = GROUP_SIZE * 3 // 8
    offs_pack = tl.arange(0, group_packs)
    t1: tl.constexpr = 1.0 / 3.0
    t2: tl.constexpr = 2.0 / 3.0
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        if DYNAMIC_A8:
            activation_scale = tl.load(input_scale_ptr + token_id // TOP_K).to(
                tl.float32
            )
        else:
            activation_scale = 1.0
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            byte_indices = group * group_bytes + offs_pack * 3
            packed_ptrs = (
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + byte_indices[None, :] * stride_wp
            )
            packed = tl.load(
                packed_ptrs,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            packed |= (
                tl.load(
                    packed_ptrs + stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 8
            )
            packed |= (
                tl.load(
                    packed_ptrs + 2 * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 16
            )
            metadata_ptrs = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            scale = tl.load(
                scale_ptr + metadata_ptrs,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            cubic_a = tl.load(
                cubic_a_ptr + metadata_ptrs,
                mask=n_mask,
                other=1.0,
            ).to(tl.float32)
            cubic_b = tl.load(
                cubic_b_ptr + metadata_ptrs,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            cubic_c = 1.0 - cubic_a - cubic_b
            if DYNAMIC_A8:
                level1 = tl.extra.cuda.libdevice.rint(
                    127.0 * t1 * (cubic_a + t1 * (cubic_b + t1 * cubic_c))
                )
                level2 = tl.extra.cuda.libdevice.rint(
                    127.0 * t2 * (cubic_a + t2 * (cubic_b + t2 * cubic_c))
                )
                level3 = 127.0
            else:
                level1 = scale * t1 * (cubic_a + t1 * (cubic_b + t1 * cubic_c))
                level2 = scale * t2 * (cubic_a + t2 * (cubic_b + t2 * cubic_c))
                level3 = scale[:, None]
            group_k = group * GROUP_SIZE + offs_pack * 8
            contribution = tl.zeros(
                (BLOCK_N, group_packs),
                dtype=tl.float32,
            )
            for lane in tl.static_range(0, 8):
                global_k = group_k + lane
                activation = tl.load(
                    input_ptr + (token_id // TOP_K) * stride_im + global_k * stride_ik,
                    mask=global_k < K,
                    other=0.0,
                ).to(tl.float32)
                code = (packed >> (lane * 3)) & 7
                signed = tl.where(code >= 4, code - 8, code)
                signed = tl.where(signed == -4, 0, signed)
                magnitude = tl.abs(signed)
                weight = tl.where(magnitude == 1, level1[:, None], 0.0)
                weight = tl.where(magnitude == 2, level2[:, None], weight)
                weight = tl.where(magnitude == 3, level3, weight)
                weight = tl.where(signed < 0, -weight, weight)
                contribution += weight * activation[None, :]
            group_sum = tl.sum(contribution, axis=1)
            if DYNAMIC_A8:
                accumulator += group_sum * activation_scale * scale * (1.0 / 127.0)
            else:
                accumulator += group_sum

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_MOE_GEMV_CONFIGS,
    key=["N", "K", "GROUP_SIZE", "ROUTE_CTAS", "TOP_K", "HAS_LINEAR_BETA"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_situ_gemv_3bit_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_packs: tl.constexpr = GROUP_SIZE // 8
    group_bytes: tl.constexpr = GROUP_SIZE * 3 // 8
    offs_pack = tl.arange(0, group_packs)
    t1: tl.constexpr = 1.0 / 3.0
    t2: tl.constexpr = 2.0 / 3.0
    route = tl.program_id(1)
    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        gate_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            byte_indices = group * group_bytes + offs_pack * 3
            weight_offsets = (
                expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + byte_indices[None, :] * stride_wp
            )
            gate_packed = tl.load(
                weight_ptr + weight_offsets,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            gate_packed |= (
                tl.load(
                    weight_ptr + weight_offsets + stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 8
            )
            gate_packed |= (
                tl.load(
                    weight_ptr + weight_offsets + 2 * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 16
            )
            up_weight_offsets = weight_offsets + N * stride_wn
            up_packed = tl.load(
                weight_ptr + up_weight_offsets,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            up_packed |= (
                tl.load(
                    weight_ptr + up_weight_offsets + stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 8
            )
            up_packed |= (
                tl.load(
                    weight_ptr + up_weight_offsets + 2 * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 16
            )
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            gate_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            gate_a = tl.load(
                cubic_a_ptr + metadata_offsets,
                mask=n_mask,
                other=1.0,
            ).to(tl.float32)
            gate_b = tl.load(
                cubic_b_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            gate_c = 1.0 - gate_a - gate_b
            gate_level1 = gate_scale * t1 * (gate_a + t1 * (gate_b + t1 * gate_c))
            gate_level2 = gate_scale * t2 * (gate_a + t2 * (gate_b + t2 * gate_c))
            up_metadata_offsets = metadata_offsets + N * stride_sn
            up_scale = tl.load(
                scale_ptr + up_metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up_a = tl.load(
                cubic_a_ptr + up_metadata_offsets,
                mask=n_mask,
                other=1.0,
            ).to(tl.float32)
            up_b = tl.load(
                cubic_b_ptr + up_metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up_c = 1.0 - up_a - up_b
            up_level1 = up_scale * t1 * (up_a + t1 * (up_b + t1 * up_c))
            up_level2 = up_scale * t2 * (up_a + t2 * (up_b + t2 * up_c))
            group_k = group * GROUP_SIZE + offs_pack * 8
            gate_contribution = tl.zeros(
                (BLOCK_N, group_packs),
                dtype=tl.float32,
            )
            up_contribution = tl.zeros(
                (BLOCK_N, group_packs),
                dtype=tl.float32,
            )
            for lane in tl.static_range(0, 8):
                global_k = group_k + lane
                activation = tl.load(
                    input_ptr + (token_id // TOP_K) * stride_im + global_k * stride_ik,
                    mask=global_k < K,
                    other=0.0,
                ).to(tl.float32)
                gate_code = (gate_packed >> (lane * 3)) & 7
                gate_signed = tl.where(gate_code >= 4, gate_code - 8, gate_code)
                gate_signed = tl.where(gate_signed == -4, 0, gate_signed)
                gate_magnitude = tl.abs(gate_signed)
                gate_weight = tl.where(gate_magnitude == 1, gate_level1[:, None], 0.0)
                gate_weight = tl.where(
                    gate_magnitude == 2, gate_level2[:, None], gate_weight
                )
                gate_weight = tl.where(
                    gate_magnitude == 3, gate_scale[:, None], gate_weight
                )
                gate_weight = tl.where(gate_signed < 0, -gate_weight, gate_weight)
                up_code = (up_packed >> (lane * 3)) & 7
                up_signed = tl.where(up_code >= 4, up_code - 8, up_code)
                up_signed = tl.where(up_signed == -4, 0, up_signed)
                up_magnitude = tl.abs(up_signed)
                up_weight = tl.where(up_magnitude == 1, up_level1[:, None], 0.0)
                up_weight = tl.where(up_magnitude == 2, up_level2[:, None], up_weight)
                up_weight = tl.where(up_magnitude == 3, up_scale[:, None], up_weight)
                up_weight = tl.where(up_signed < 0, -up_weight, up_weight)
                gate_contribution += gate_weight * activation[None, :]
                up_contribution += up_weight * activation[None, :]
            gate_accumulator += tl.sum(gate_contribution, axis=1)
            up_accumulator += tl.sum(up_contribution, axis=1)

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + token_id)
            gate_accumulator *= routed_weight
            up_accumulator *= routed_weight
        gate = gate_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        up = up_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / BETA) - 1.0
        gate = BETA * gate_tanh * tl.sigmoid(gate)
        if HAS_LINEAR_BETA:
            up = LINEAR_BETA * (2.0 * tl.sigmoid(2.0 * up / LINEAR_BETA) - 1.0)
        tl.store(
            output_ptr + token_id * stride_om + offs_n * stride_on,
            gate * up,
            mask=n_mask,
        )
        route += ROUTE_CTAS


@triton.autotune(
    configs=_CUBIC_MOE_2BIT_A16_CONFIGS,
    key=["N", "K", "GROUP_SIZE", "ROUTE_CTAS", "TOP_K", "SUM_ROUTES"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_2bit_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
    DYNAMIC_A8: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        if DYNAMIC_A8:
            activation_scale = tl.load(input_scale_ptr + token_id // TOP_K).to(
                tl.float32
            )
        else:
            activation_scale = 1.0
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            word_indices = group * group_words + offs_word
            packed = tl.load(
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                scale_ptr
                + expert_id * stride_se
                + offs_n * stride_sn
                + group * stride_sg,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            group_k = group * GROUP_SIZE + offs_word * 16
            contribution = tl.zeros(
                (BLOCK_N, group_words),
                dtype=tl.float32,
            )
            for lane in tl.static_range(0, 16):
                global_k = group_k + lane
                activation = tl.load(
                    input_ptr + (token_id // TOP_K) * stride_im + global_k * stride_ik,
                    mask=global_k < K,
                    other=0.0,
                ).to(tl.float32)
                code = (packed >> (lane * 2)) & 3
                ternary = (code & 1) * (1 - (code & 2))
                contribution += ternary * activation[None, :]
            group_sum = tl.sum(contribution, axis=1)
            accumulator += activation_scale * scale * group_sum

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.jit
def _cubic_moe_gemv_2bit_cubic8_kernel(
    input_code_ptr,
    input_scale_ptr,
    input_a_ptr,
    input_b_ptr,
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_wse,
    stride_wsn,
    stride_wsg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    """Exact Cubic8 activation × Cubic-W2 route GEMV.

    The activation remains signed Cubic code in global memory.  Curve decode
    occurs only for the currently consumed K tile; no BF16 activation tensor
    is materialized.
    """
    num_routes = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    while route < num_routes:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for group in tl.static_range(0, NUM_GROUPS):
            metadata = input_row * stride_ism + group * stride_isg
            activation_scale = tl.load(input_scale_ptr + metadata).to(tl.float32)
            activation_a = tl.load(input_a_ptr + metadata).to(tl.float32)
            activation_b = tl.load(input_b_ptr + metadata).to(tl.float32)
            activation_c = 1.0 - activation_a - activation_b
            word_indices = group * group_words + offs_word
            packed = tl.load(
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            weight_scale = tl.load(
                weight_scale_ptr
                + expert_id * stride_wse
                + offs_n * stride_wsn
                + group * stride_wsg,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            group_k = group * GROUP_SIZE + offs_word * 16
            contribution = tl.zeros((BLOCK_N, group_words), dtype=tl.float32)
            for lane in tl.static_range(0, 16):
                global_k = group_k + lane
                raw_code = tl.load(
                    input_code_ptr + input_row * stride_im + global_k * stride_ik,
                    mask=global_k < K,
                    other=0,
                ).to(tl.int32)
                # Triton may represent an INT8 pointer load as its raw byte
                # before widening; normalize explicitly so both signed and
                # raw-byte lowering produce the same signed Cubic code.
                code = tl.where(raw_code < 128, raw_code, raw_code - 256)
                magnitude = tl.abs(code).to(tl.float32)
                t = magnitude / 127.0
                activation = (
                    tl.where(code < 0, -1.0, 1.0)
                    * activation_scale
                    * t
                    * (activation_a + t * (activation_b + t * activation_c))
                )
                weight_code = (packed >> (lane * 2)) & 3
                ternary = (weight_code & 1) * (1 - (weight_code & 2))
                contribution += ternary * activation[None, :]
            accumulator += weight_scale * tl.sum(contribution, axis=1)
        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.jit
def _cubic_dp4a(a, b, c):
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a, b, c],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cubic_3bit_carrier_luts(level1, level2):
    level1 = level1 & 0xFF
    level2 = level2 & 0xFF
    lut0 = (level1 << 8) | (level2 << 16) | 0x7F000000
    lut1 = 0x00008100 | ((-level2 & 0xFF) << 16) | ((-level1 & 0xFF) << 24)
    return lut0, lut1


@triton.jit
def _cubic_3bit_carrier_words(packed, lut0, lut1):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b32 sel, c;
            bfe.u32 sel, $2, 0, 3;
            bfe.u32 c, $2, 3, 3;
            shl.b32 c, c, 4;
            or.b32 sel, sel, c;
            bfe.u32 c, $2, 6, 3;
            shl.b32 c, c, 8;
            or.b32 sel, sel, c;
            bfe.u32 c, $2, 9, 3;
            shl.b32 c, c, 12;
            or.b32 sel, sel, c;
            prmt.b32 $0, $3, $4, sel;
            bfe.u32 sel, $2, 12, 3;
            bfe.u32 c, $2, 15, 3;
            shl.b32 c, c, 4;
            or.b32 sel, sel, c;
            bfe.u32 c, $2, 18, 3;
            shl.b32 c, c, 8;
            or.b32 sel, sel, c;
            bfe.u32 c, $2, 21, 3;
            shl.b32 c, c, 12;
            or.b32 sel, sel, c;
            prmt.b32 $1, $3, $4, sel;
        }
        """,
        constraints="=r,=r,r,r,r",
        args=[packed, lut0, lut1],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cubic_moe_w3_a8_persistent_kernel(
    input_words_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    level1_ptr,
    level2_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_iwm,
    stride_iwk,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Persistent W3/A8 route-tile scheduler for precomputed carriers.

    Each program repeatedly claims a flattened ``(route, output tile)`` task.
    The resident grid is capped by the caller from the device SM count, so the
    same kernel scales down to small GPUs without baking in an H200 topology.
    """
    num_routes = tl.load(num_tokens_post_padded_ptr)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    group_packs: tl.constexpr = GROUP_SIZE // 8
    group_quads: tl.constexpr = GROUP_SIZE // 4
    group_bytes: tl.constexpr = GROUP_SIZE * 3 // 8
    offs_pack = tl.arange(0, group_packs)
    task = tl.program_id(0)
    total_tasks = num_routes * num_n_blocks

    while task < total_tasks:
        route = task // num_n_blocks
        n_block = task - route * num_n_blocks
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            byte_indices = group * group_bytes + offs_pack * 3
            packed_ptrs = (
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + byte_indices[None, :] * stride_wp
            )
            packed = tl.load(
                packed_ptrs,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            packed |= (
                tl.load(
                    packed_ptrs + stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 8
            )
            packed |= (
                tl.load(
                    packed_ptrs + 2 * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 16
            )
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            weight_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            level1 = tl.load(
                level1_ptr + metadata_offsets,
                mask=n_mask,
                other=0,
            ).to(tl.int32)
            level2 = tl.load(
                level2_ptr + metadata_offsets,
                mask=n_mask,
                other=0,
            ).to(tl.int32)
            lut0, lut1 = _cubic_3bit_carrier_luts(level1, level2)
            carrier_lo, carrier_hi = _cubic_3bit_carrier_words(
                packed,
                lut0[:, None],
                lut1[:, None],
            )
            input_word_base = group * group_quads + offs_pack * 2
            activation_lo = tl.load(
                input_words_ptr + input_row * stride_iwm + input_word_base * stride_iwk
            ).to(tl.int32)
            activation_hi = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (input_word_base + 1) * stride_iwk
            ).to(tl.int32)
            dot_lo = _cubic_dp4a(
                carrier_lo,
                activation_lo[None, :],
                tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
            )
            dot_hi = _cubic_dp4a(
                carrier_hi,
                activation_hi[None, :],
                tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
            )
            group_dot = tl.sum(dot_lo + dot_hi, axis=1).to(tl.float32)
            accumulator += group_dot * activation_scale * weight_scale * (1.0 / 127.0)

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        tl.store(
            output_ptr + token_id * stride_om + offs_n * stride_on,
            accumulator,
            mask=n_mask,
        )
        task += NUM_WORKERS


@triton.jit
def _cubic_moe_w3_a8_word_kernel(
    input_words_ptr,
    input_scale_ptr,
    weight_words_ptr,
    scale_ptr,
    level1_ptr,
    level2_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_iwm,
    stride_iwk,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """W3/A8 GEMV using three uint32 loads for each 32 packed weights."""
    num_routes = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    pack_blocks: tl.constexpr = GROUP_SIZE // 32
    group_words: tl.constexpr = GROUP_SIZE * 3 // 32
    group_quads: tl.constexpr = GROUP_SIZE // 4
    offs_block = tl.arange(0, pack_blocks)
    route = tl.program_id(1)

    while route < num_routes:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            word_base = group * group_words + offs_block * 3
            word_ptrs = (
                weight_words_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_base[None, :] * stride_wp
            )
            w0 = tl.load(word_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)
            w1 = tl.load(word_ptrs + stride_wp, mask=n_mask[:, None], other=0).to(
                tl.int32
            )
            w2 = tl.load(word_ptrs + 2 * stride_wp, mask=n_mask[:, None], other=0).to(
                tl.int32
            )
            packed0 = w0 & 0x00FFFFFF
            packed1 = ((w0 >> 24) & 0xFF) | ((w1 & 0x0000FFFF) << 8)
            packed2 = ((w1 >> 16) & 0xFFFF) | ((w2 & 0x000000FF) << 16)
            packed3 = (w2 >> 8) & 0x00FFFFFF

            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            weight_scale = tl.load(
                scale_ptr + metadata_offsets, mask=n_mask, other=0.0
            ).to(tl.float32)
            level1 = tl.load(level1_ptr + metadata_offsets, mask=n_mask, other=0).to(
                tl.int32
            )
            level2 = tl.load(level2_ptr + metadata_offsets, mask=n_mask, other=0).to(
                tl.int32
            )
            lut0, lut1 = _cubic_3bit_carrier_luts(level1, level2)
            activation_base = group * group_quads + offs_block * 8
            zeros = tl.zeros((BLOCK_N, pack_blocks), dtype=tl.int32)
            carrier0, carrier1 = _cubic_3bit_carrier_words(
                packed0, lut0[:, None], lut1[:, None]
            )
            activation0 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 0) * stride_iwk
            ).to(tl.int32)
            activation1 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 1) * stride_iwk
            ).to(tl.int32)
            dots = _cubic_dp4a(carrier0, activation0[None, :], zeros)
            dots += _cubic_dp4a(carrier1, activation1[None, :], zeros)

            carrier0, carrier1 = _cubic_3bit_carrier_words(
                packed1, lut0[:, None], lut1[:, None]
            )
            activation0 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 2) * stride_iwk
            ).to(tl.int32)
            activation1 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 3) * stride_iwk
            ).to(tl.int32)
            dots += _cubic_dp4a(carrier0, activation0[None, :], zeros)
            dots += _cubic_dp4a(carrier1, activation1[None, :], zeros)

            carrier0, carrier1 = _cubic_3bit_carrier_words(
                packed2, lut0[:, None], lut1[:, None]
            )
            activation0 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 4) * stride_iwk
            ).to(tl.int32)
            activation1 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 5) * stride_iwk
            ).to(tl.int32)
            dots += _cubic_dp4a(carrier0, activation0[None, :], zeros)
            dots += _cubic_dp4a(carrier1, activation1[None, :], zeros)

            carrier0, carrier1 = _cubic_3bit_carrier_words(
                packed3, lut0[:, None], lut1[:, None]
            )
            activation0 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 6) * stride_iwk
            ).to(tl.int32)
            activation1 = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (activation_base + 7) * stride_iwk
            ).to(tl.int32)
            dots += _cubic_dp4a(carrier0, activation0[None, :], zeros)
            dots += _cubic_dp4a(carrier1, activation1[None, :], zeros)
            group_dot = tl.sum(dots, axis=1).to(tl.float32)
            accumulator += group_dot * activation_scale * weight_scale * (1.0 / 127.0)

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        tl.store(
            output_ptr + token_id * stride_om + offs_n * stride_on,
            accumulator,
            mask=n_mask,
        )
        route += ROUTE_CTAS


@triton.jit
def _cubic_2bit_carrier_word(packed_byte):
    # Spread four adjacent 2-bit codes into the low two bits of four selector
    # nibbles, then select bytes from [0, +1, 0, -1].  This replaces four
    # multiply/mask carrier expansions with one PRMT.
    selector = (packed_byte | (packed_byte << 4)) & 0x0F0F
    selector = (selector | (selector << 2)) & 0x3333
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[0xFF000100, 0, selector],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cubic_1bit_carrier_words(packed_byte):
    """Expand eight binary codes into two signed INT8 carrier words."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b32 spread, tmp;
            and.b32 spread, $2, 0xff;
            shl.b32 tmp, spread, 12;
            or.b32 spread, spread, tmp;
            and.b32 spread, spread, 0x000f000f;
            shl.b32 tmp, spread, 6;
            or.b32 spread, spread, tmp;
            and.b32 spread, spread, 0x03030303;
            shl.b32 tmp, spread, 3;
            or.b32 spread, spread, tmp;
            and.b32 spread, spread, 0x11111111;
            prmt.b32 $0, 0x00007f81, 0, spread;
            shr.u32 spread, spread, 16;
            prmt.b32 $1, 0x00007f81, 0, spread;
        }
        """,
        constraints="=r,=r,r",
        args=[packed_byte],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )


_CUBIC_1BIT_DP4A_CONFIGS = [
    triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=_CUBIC_1BIT_DP4A_CONFIGS,
    key=[
        "N",
        "GROUP_SIZE",
        "ROUTE_CTAS",
        "MUL_ROUTED_WEIGHT",
        "TOP_K",
        "SUM_ROUTES",
        "GROUPWISE_SCALE",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_1bit_a8_dp4a_kernel(
    input_words_ptr,
    input_scale_ptr,
    weight_words_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_iwm,
    stride_iwk,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
    GROUPWISE_SCALE: tl.constexpr,
):
    num_routes = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 32
    group_quads: tl.constexpr = GROUP_SIZE // 4
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_routes:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        if not GROUPWISE_SCALE:
            activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for group in tl.static_range(0, NUM_GROUPS):
            if GROUPWISE_SCALE:
                group_activation_scale = tl.load(
                    input_scale_ptr + input_row * stride_ism + group * stride_isg
                ).to(tl.float32)
            else:
                group_activation_scale = activation_scale
            packed = tl.load(
                weight_words_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + (group * group_words + offs_word)[None, :] * stride_wp,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            activation_base = group * group_quads + offs_word * 8
            dots = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
            for byte in tl.static_range(0, 4):
                carrier0, carrier1 = _cubic_1bit_carrier_words(packed >> (byte * 8))
                activation0 = tl.load(
                    input_words_ptr
                    + input_row * stride_iwm
                    + (activation_base + byte * 2) * stride_iwk
                ).to(tl.int32)
                activation1 = tl.load(
                    input_words_ptr
                    + input_row * stride_iwm
                    + (activation_base + byte * 2 + 1) * stride_iwk
                ).to(tl.int32)
                dots += _cubic_dp4a(carrier0, activation0[None, :], 0)
                dots += _cubic_dp4a(carrier1, activation1[None, :], 0)
            group_dot = tl.sum(dots, axis=1).to(tl.float32)
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            weight_scale = tl.load(
                scale_ptr + metadata_offsets, mask=n_mask, other=0.0
            ).to(tl.float32)
            accumulator += (
                group_dot * group_activation_scale * weight_scale * (1.0 / 127.0)
            )

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


_CUBIC_2BIT_DP4A_CONFIGS = [
    triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=_CUBIC_2BIT_DP4A_CONFIGS,
    key=[
        "N",
        "NUM_GROUPS",
        "GROUP_SIZE",
        "ROUTE_CTAS",
        "MUL_ROUTED_WEIGHT",
        "TOP_K",
        "SUM_ROUTES",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            word_indices = group * group_words + offs_word
            packed = tl.load(
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
            for quad in tl.static_range(0, 4):
                packed_byte = (packed >> (quad * 8)) & 0xFF
                carriers = _cubic_2bit_carrier_word(packed_byte)
                activation_word = tl.load(
                    input_ptr
                    + input_row * stride_im
                    + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad) * stride_ik
                ).to(tl.int32)
                dot = _cubic_dp4a(carriers, activation_word[None, :], dot)
            weight_scale = tl.load(
                scale_ptr
                + expert_id * stride_se
                + offs_n * stride_sn
                + group * stride_sg,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += activation_scale * weight_scale * tl.sum(dot, axis=1)

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_2BIT_DP4A_CONFIGS,
    key=[
        "N",
        "NUM_GROUPS",
        "GROUP_SIZE",
        "ACTIVATION_GROUP_SIZE",
        "TOP_K",
        "MUL_ROUTED_WEIGHT",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_2bit_groupwise_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION_GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    """W2 carrier dot with one activation scale per input row and K-group."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            activation_group = (
                group * GROUP_SIZE + offs_word * 16
            ) // ACTIVATION_GROUP_SIZE
            activation_scale = tl.load(
                input_scale_ptr + input_row * stride_ism + activation_group * stride_isg
            ).to(tl.float32)
            word_indices = group * group_words + offs_word
            packed = tl.load(
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
            for quad in tl.static_range(0, 4):
                packed_byte = (packed >> (quad * 8)) & 0xFF
                carriers = _cubic_2bit_carrier_word(packed_byte)
                activation_word = tl.load(
                    input_ptr
                    + input_row * stride_im
                    + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad) * stride_ik
                ).to(tl.int32)
                dot = _cubic_dp4a(carriers, activation_word[None, :], dot)
            weight_scale = tl.load(
                scale_ptr
                + expert_id * stride_se
                + offs_n * stride_sn
                + group * stride_sg,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += weight_scale * tl.sum(
                dot.to(tl.float32) * activation_scale[None, :], axis=1
            )

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_2BIT_DP4A_CONFIGS,
    key=["N", "NUM_GROUPS", "GROUP_SIZE", "TOP_K", "MUL_ROUTED_WEIGHT"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_w2_fused_sum_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """Compute and reduce every local W2 route for one token in one CTA."""
    token = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # A runtime loop avoids cloning the complete K/group body TOP_K times at
    # compile time for models with a large route count.
    for route_slot in tl.range(0, TOP_K):
        route = token * TOP_K + route_slot
        global_expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_map_ptr + global_expert).to(tl.int64)
        if expert_id >= 0:
            activation_scale = tl.load(input_scale_ptr + route).to(tl.float32)
            accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for group in tl.static_range(0, NUM_GROUPS):
                word_indices = group * group_words + offs_word
                packed = tl.load(
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + word_indices[None, :] * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                for quad in tl.static_range(0, 4):
                    carriers = _cubic_2bit_carrier_word((packed >> (quad * 8)) & 0xFF)
                    activation_word = tl.load(
                        input_ptr
                        + route * stride_im
                        + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad) * stride_ik
                    ).to(tl.int32)
                    dot = _cubic_dp4a(carriers, activation_word[None, :], dot)
                weight_scale = tl.load(
                    scale_ptr
                    + expert_id * stride_se
                    + offs_n * stride_sn
                    + group * stride_sg,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += activation_scale * weight_scale * tl.sum(dot, axis=1)
            if MUL_ROUTED_WEIGHT:
                accumulator *= tl.load(topk_weights_ptr + route)
            # Match the existing route buffer + moe_sum numerical order.
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)

    tl.store(
        output_ptr + token * stride_om + offs_n * stride_on,
        route_sum,
        mask=n_mask,
    )


@triton.autotune(
    configs=_CUBIC_MOE_GEMV_CONFIGS,
    key=["N", "K", "NUM_BITS", "GROUP_SIZE", "BLOCK_K", "TOP_K"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_w1_w8_fused_sum_a8_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PRECOMPUTED_3BIT_LEVELS: tl.constexpr,
):
    """Compute W1-W8 local routes directly into one row per token.

    This is the bounded-workspace prefill path.  Unlike the expert-sorted
    GEMM path it never materializes [tokens, top_k, hidden] route outputs.
    """
    token = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for route_slot in tl.range(0, TOP_K):
        route = token * TOP_K + route_slot
        global_expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_map_ptr + global_expert).to(tl.int64)
        if expert_id >= 0:
            activation_scale = tl.load(input_scale_ptr + route).to(tl.float32)
            accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                global_k = k_block * BLOCK_K + offs_k
                k_mask = global_k < K
                activation = tl.load(
                    input_ptr + route * stride_im + global_k * stride_ik,
                    mask=k_mask,
                    other=0,
                ).to(tl.int32)
                bit_positions = global_k[None, :] * NUM_BITS
                byte_indices = bit_positions // 8
                shifts = bit_positions % 8
                packed_ptrs = (
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + byte_indices * stride_wp
                )
                weight_mask = n_mask[:, None] & k_mask[None, :]
                low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
                if 8 % NUM_BITS == 0:
                    high = 0
                else:
                    high = tl.load(
                        packed_ptrs + stride_wp,
                        mask=weight_mask & (byte_indices + 1 < PACKED_K),
                        other=0,
                    ).to(tl.int32)
                raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)
                group = (k_block * BLOCK_K) // GROUP_SIZE
                metadata_offsets = (
                    expert_id * stride_se + offs_n * stride_sn + group * stride_sg
                )
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=n_mask & (group < NUM_GROUPS),
                    other=0.0,
                ).to(tl.float32)
                if PRECOMPUTED_3BIT_LEVELS:
                    level1 = tl.load(
                        cubic_a_ptr + metadata_offsets,
                        mask=n_mask & (group < NUM_GROUPS),
                        other=0,
                    ).to(tl.float32)[:, None]
                    level2 = tl.load(
                        cubic_b_ptr + metadata_offsets,
                        mask=n_mask & (group < NUM_GROUPS),
                        other=0,
                    ).to(tl.float32)[:, None]
                    signed = tl.where(raw >= 4, raw - 8, raw)
                    signed = tl.where(signed == -4, 0, signed)
                    magnitude = tl.abs(signed)
                    carrier = tl.where(magnitude == 1, level1, 0.0)
                    carrier = tl.where(magnitude == 2, level2, carrier)
                    carrier = tl.where(magnitude == 3, 127.0, carrier)
                    carrier = tl.where(signed < 0, -carrier, carrier).to(tl.int32)
                else:
                    if NUM_BITS > 2:
                        cubic_a = tl.load(
                            cubic_a_ptr + metadata_offsets,
                            mask=n_mask & (group < NUM_GROUPS),
                            other=1.0,
                        ).to(tl.float32)[:, None]
                        cubic_b = tl.load(
                            cubic_b_ptr + metadata_offsets,
                            mask=n_mask & (group < NUM_GROUPS),
                            other=0.0,
                        ).to(tl.float32)[:, None]
                    else:
                        cubic_a = 1.0
                        cubic_b = 0.0
                    carrier = _cubic_dynamic_a8_carrier(
                        raw, cubic_a, cubic_b, NUM_BITS
                    ).to(tl.int32)
                partial = tl.sum(carrier * activation[None, :], axis=1)
                accumulator += (
                    partial.to(tl.float32)
                    * activation_scale
                    * weight_scale
                    * (1.0 / 127.0)
                )
            if MUL_ROUTED_WEIGHT:
                accumulator *= tl.load(topk_weights_ptr + route)
            # Preserve the old BF16 route-buffer + FP32 moe_sum order.
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)

    tl.store(
        output_ptr + token * stride_om + offs_n * stride_on,
        route_sum,
        mask=n_mask,
    )


@triton.autotune(
    configs=_CUBIC_2BIT_DP4A_CONFIGS,
    key=[
        "N",
        "NUM_GROUPS",
        "GROUP_SIZE",
        "ROUTE_CTAS",
        "MUL_ROUTED_WEIGHT",
        "TOP_K",
        "HAS_LINEAR_BETA",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_situ_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    output_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    stride_osm,
    stride_osg,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION_GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    OUTPUT_GROUPWISE_A8: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        activation_scale = tl.load(input_scale_ptr + token_id // TOP_K).to(tl.float32)
        gate_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            word_indices = group * group_words + offs_word
            weight_offsets = (
                expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp
            )
            gate_packed = tl.load(
                weight_ptr + weight_offsets,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            up_packed = tl.load(
                weight_ptr + weight_offsets + N * stride_wn,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            gate_dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
            up_dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
            for quad in tl.static_range(0, 4):
                gate_byte = (gate_packed >> (quad * 8)) & 0xFF
                up_byte = (up_packed >> (quad * 8)) & 0xFF
                gate_carriers = _cubic_2bit_carrier_word(gate_byte)
                up_carriers = _cubic_2bit_carrier_word(up_byte)
                activation_word = tl.load(
                    input_ptr
                    + (token_id // TOP_K) * stride_im
                    + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad) * stride_ik
                ).to(tl.int32)
                gate_dot = _cubic_dp4a(
                    gate_carriers, activation_word[None, :], gate_dot
                )
                up_dot = _cubic_dp4a(up_carriers, activation_word[None, :], up_dot)
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            gate_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                scale_ptr + metadata_offsets + N * stride_sn,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            gate_accumulator += activation_scale * gate_scale * tl.sum(gate_dot, axis=1)
            up_accumulator += activation_scale * up_scale * tl.sum(up_dot, axis=1)

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + token_id)
            gate_accumulator *= routed_weight
            up_accumulator *= routed_weight
        if OUTPUT_GROUPWISE_A8:
            if OUTPUT_BF16:
                gate = gate_accumulator.to(tl.bfloat16).to(tl.float32)
                up = up_accumulator.to(tl.bfloat16).to(tl.float32)
            else:
                gate = gate_accumulator.to(tl.float16).to(tl.float32)
                up = up_accumulator.to(tl.float16).to(tl.float32)
        else:
            gate = gate_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            up = up_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / BETA) - 1.0
        gate = BETA * gate_tanh * tl.sigmoid(gate)
        if HAS_LINEAR_BETA:
            up = LINEAR_BETA * (2.0 * tl.sigmoid(2.0 * up / LINEAR_BETA) - 1.0)
        activated = gate * up
        if OUTPUT_GROUPWISE_A8:
            if OUTPUT_BF16:
                activated = activated.to(tl.bfloat16).to(tl.float32)
            else:
                activated = activated.to(tl.float16).to(tl.float32)
            quantized = tl.zeros((BLOCK_N,), dtype=tl.int8)
            for subgroup in tl.static_range(0, BLOCK_N // ACTIVATION_GROUP_SIZE):
                subgroup_mask = (
                    tl.arange(0, BLOCK_N) // ACTIVATION_GROUP_SIZE
                ) == subgroup
                absmax = tl.maximum(
                    tl.max(tl.where(subgroup_mask, tl.abs(activated), 0.0)),
                    1e-10,
                )
                q = round_int8(activated * (127.0 / absmax))
                quantized = tl.where(subgroup_mask, q, quantized)
                output_group = (
                    tl.program_id(0) * (BLOCK_N // ACTIVATION_GROUP_SIZE) + subgroup
                )
                tl.store(
                    output_scale_ptr
                    + token_id * stride_osm
                    + output_group * stride_osg,
                    absmax / 127.0,
                )
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                quantized,
                mask=n_mask,
            )
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                activated,
                mask=n_mask,
            )
        route += ROUTE_CTAS


@triton.jit
def _cubic_moe_situ_gemv_2bit_cubic8_producer_kernel(
    input_words_ptr,
    input_scale_ptr,
    weight_words_ptr,
    weight_scale_ptr,
    output_code_ptr,
    output_scale_ptr,
    output_a_ptr,
    output_b_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_WEIGHT_GROUPS: tl.constexpr,
    stride_iwm,
    stride_iwk,
    stride_we,
    stride_wn,
    stride_wp,
    stride_wse,
    stride_wsn,
    stride_wsg,
    stride_ocm,
    stride_ock,
    stride_osm,
    stride_osg,
    WEIGHT_GROUP_SIZE: tl.constexpr,
    OUTPUT_GROUP_SIZE: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Fused W2 projection + SITU + exact online Cubic8 producer.

    One CTA owns one complete output group for one route at a time.  This is
    essential: scale/a/b are per sample and per output group, so independent
    small-N CTAs cannot fit a mathematically equivalent carrier without a
    global floating-point staging tensor.  K is processed in small word tiles
    to bound register pressure while the final group remains CTA-resident.
    """
    output_group = tl.program_id(0)
    offs_n = output_group * OUTPUT_GROUP_SIZE + tl.arange(0, OUTPUT_GROUP_SIZE)
    n_mask = offs_n < N
    group_words: tl.constexpr = WEIGHT_GROUP_SIZE // 16
    word_tiles: tl.constexpr = group_words // WORDS_PER_TILE
    word_lanes = tl.arange(0, WORDS_PER_TILE)
    route = tl.program_id(1)
    num_routes = tl.load(num_tokens_post_padded_ptr)
    while route < num_routes:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        gate_accumulator = tl.zeros((OUTPUT_GROUP_SIZE,), dtype=tl.float32)
        up_accumulator = tl.zeros((OUTPUT_GROUP_SIZE,), dtype=tl.float32)

        # Keep the model-dependent K-group count as a runtime loop.  Statically
        # cloning this body for K=7168/G512 creates a huge compiler graph and
        # turns calibration into minutes of host compilation without exposing
        # any additional parallelism.
        for group in tl.range(0, NUM_WEIGHT_GROUPS):
            metadata = expert_id * stride_wse + offs_n * stride_wsn + group * stride_wsg
            gate_scale = tl.load(
                weight_scale_ptr + metadata,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                weight_scale_ptr + metadata + N * stride_wsn,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            gate_group = tl.zeros((OUTPUT_GROUP_SIZE,), dtype=tl.int32)
            up_group = tl.zeros((OUTPUT_GROUP_SIZE,), dtype=tl.int32)
            for word_tile in tl.static_range(0, word_tiles):
                words = word_tile * WORDS_PER_TILE + word_lanes
                word_indices = group * group_words + words
                offsets = (
                    expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + word_indices[None, :] * stride_wp
                )
                gate_packed = tl.load(
                    weight_words_ptr + offsets,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                up_packed = tl.load(
                    weight_words_ptr + offsets + N * stride_wn,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                gate_dot = tl.zeros((OUTPUT_GROUP_SIZE, WORDS_PER_TILE), dtype=tl.int32)
                up_dot = tl.zeros((OUTPUT_GROUP_SIZE, WORDS_PER_TILE), dtype=tl.int32)
                for quad in tl.static_range(0, 4):
                    activation_word = tl.load(
                        input_words_ptr
                        + input_row * stride_iwm
                        + (group * (WEIGHT_GROUP_SIZE // 4) + words * 4 + quad)
                        * stride_iwk
                    ).to(tl.int32)
                    gate_dot = _cubic_dp4a(
                        _cubic_2bit_carrier_word((gate_packed >> (quad * 8)) & 0xFF),
                        activation_word[None, :],
                        gate_dot,
                    )
                    up_dot = _cubic_dp4a(
                        _cubic_2bit_carrier_word((up_packed >> (quad * 8)) & 0xFF),
                        activation_word[None, :],
                        up_dot,
                    )
                gate_group += tl.sum(gate_dot, axis=1)
                up_group += tl.sum(up_dot, axis=1)
            gate_accumulator += activation_scale * gate_scale * gate_group
            up_accumulator += activation_scale * up_scale * up_group

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + token_id)
            gate_accumulator *= routed_weight
            up_accumulator *= routed_weight
        if OUTPUT_BF16:
            gate = gate_accumulator.to(tl.bfloat16).to(tl.float32)
            up = up_accumulator.to(tl.bfloat16).to(tl.float32)
        else:
            gate = gate_accumulator.to(tl.float16).to(tl.float32)
            up = up_accumulator.to(tl.float16).to(tl.float32)
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / BETA) - 1.0
        gate = BETA * gate_tanh * tl.sigmoid(gate)
        if HAS_LINEAR_BETA:
            up = LINEAR_BETA * (2.0 * tl.sigmoid(2.0 * up / LINEAR_BETA) - 1.0)
        activated = gate * up
        if OUTPUT_BF16:
            activated = activated.to(tl.bfloat16).to(tl.float32)
        else:
            activated = activated.to(tl.float16).to(tl.float32)
        codes, scale, curve_a, curve_b = _cubic8_fit_values(activated, n_mask)
        tl.store(
            output_code_ptr + token_id * stride_ocm + offs_n * stride_ock,
            codes,
            mask=n_mask,
        )
        output_metadata = token_id * stride_osm + output_group * stride_osg
        tl.store(output_scale_ptr + output_metadata, scale)
        tl.store(output_a_ptr + output_metadata, curve_a)
        tl.store(output_b_ptr + output_metadata, curve_b)
        route += ROUTE_CTAS


@triton.jit
def _cubic_moe_grouped2_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    ROUTES_PER_BLOCK: tl.constexpr,
):
    """A small same-expert route group shares each packed-weight load."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route_block = tl.program_id(1)

    while route_block * ROUTES_PER_BLOCK < num_tokens_post_padded:
        token_offsets = route_block * ROUTES_PER_BLOCK + tl.arange(0, ROUTES_PER_BLOCK)
        token_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
        token_mask = (token_ids >= 0) & (token_ids < num_valid_tokens)
        expert_id = tl.load(expert_ids_ptr + route_block).to(tl.int64)
        if expert_id != -1:
            input_rows = token_ids // TOP_K
            activation_scale = tl.load(
                input_scale_ptr + input_rows,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator = tl.zeros((ROUTES_PER_BLOCK, BLOCK_N), dtype=tl.float32)

            for group in tl.range(0, NUM_GROUPS):
                word_indices = group * group_words + offs_word
                packed = tl.load(
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + word_indices[None, :] * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                weight_scale = tl.load(
                    scale_ptr
                    + expert_id * stride_se
                    + offs_n * stride_sn
                    + group * stride_sg,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                # Keep dp4a two-dimensional.  Triton's inline-asm broadcast
                # over a leading route dimension can alias lanes, so compute
                # each route independently while retaining the shared packed
                # weight load above.
                route_lanes = tl.arange(0, ROUTES_PER_BLOCK)
                for r in tl.static_range(0, ROUTES_PER_BLOCK):
                    route_lane = route_lanes == r
                    input_row = tl.sum(tl.where(route_lane, input_rows, 0), axis=0)
                    route_valid = (
                        tl.sum(tl.where(route_lane, token_mask, 0), axis=0) != 0
                    )
                    route_scale = tl.sum(
                        tl.where(route_lane, activation_scale, 0.0), axis=0
                    )
                    dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        packed_byte = (packed >> (quad * 8)) & 0xFF
                        carriers = _cubic_2bit_carrier_word(packed_byte)
                        activation_word = tl.load(
                            input_ptr
                            + input_row * stride_im
                            + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad)
                            * stride_ik,
                            mask=route_valid,
                            other=0,
                        ).to(tl.int32)
                        dot = _cubic_dp4a(carriers, activation_word[None, :], dot)
                    contribution = route_scale * weight_scale * tl.sum(dot, axis=1)
                    accumulator += route_lane[:, None] * contribution[None, :]

            if MUL_ROUTED_WEIGHT:
                routed_weight = tl.load(
                    topk_weights_ptr + token_ids,
                    mask=token_mask,
                    other=0.0,
                )
                accumulator *= routed_weight[:, None]
            tl.store(
                output_ptr
                + token_ids[:, None] * stride_om
                + offs_n[None, :] * stride_on,
                accumulator,
                mask=token_mask[:, None] & n_mask[None, :],
            )
        route_block += ROUTE_CTAS


_CUBIC_2BIT_PAIR_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=_CUBIC_2BIT_PAIR_CONFIGS,
    key=[
        "N",
        "NUM_GROUPS",
        "GROUP_SIZE",
        "ROUTE_CTAS",
        "MUL_ROUTED_WEIGHT",
        "TOP_K",
        "GROUPWISE_SCALE",
    ],
    cache_results=True,
)
@triton.jit(do_not_specialize=["num_valid_tokens"])
def _cubic_moe_pair_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION_GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    GROUPWISE_SCALE: tl.constexpr,
):
    """Process a real route pair or one padded singleton per expert block.

    The paired path expands each packed carrier once and feeds two DP4As.  The
    singleton path is selected once per expert block, outside the group loop,
    so padding does not execute a second projection's arithmetic.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route_block = tl.program_id(1)

    while route_block * 2 < num_tokens_post_padded:
        route_offset = route_block * 2
        token0 = tl.load(sorted_token_ids_ptr + route_offset).to(tl.int64)
        token1 = tl.load(sorted_token_ids_ptr + route_offset + 1).to(tl.int64)
        valid0 = (token0 >= 0) & (token0 < num_valid_tokens)
        valid1 = (token1 >= 0) & (token1 < num_valid_tokens)
        expert_id = tl.load(expert_ids_ptr + route_block).to(tl.int64)
        if (expert_id != -1) & valid0:
            row0 = token0 // TOP_K
            scale0 = (
                0.0
                if GROUPWISE_SCALE
                else tl.load(input_scale_ptr + row0 * stride_ism).to(tl.float32)
            )
            acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
            if valid1:
                row1 = token1 // TOP_K
                scale1 = (
                    0.0
                    if GROUPWISE_SCALE
                    else tl.load(input_scale_ptr + row1 * stride_ism).to(tl.float32)
                )
                acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for group in tl.range(0, NUM_GROUPS):
                    if GROUPWISE_SCALE:
                        activation_group = (
                            group * GROUP_SIZE + offs_word * 16
                        ) // ACTIVATION_GROUP_SIZE
                        group_scale0 = tl.load(
                            input_scale_ptr
                            + row0 * stride_ism
                            + activation_group * stride_isg
                        ).to(tl.float32)
                        group_scale1 = tl.load(
                            input_scale_ptr
                            + row1 * stride_ism
                            + activation_group * stride_isg
                        ).to(tl.float32)
                    else:
                        group_scale0 = scale0
                        group_scale1 = scale1
                    word_indices = group * group_words + offs_word
                    packed = tl.load(
                        weight_ptr
                        + expert_id * stride_we
                        + offs_n[:, None] * stride_wn
                        + word_indices[None, :] * stride_wp,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    weight_scale = tl.load(
                        scale_ptr
                        + expert_id * stride_se
                        + offs_n * stride_sn
                        + group * stride_sg,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    dot1 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        carriers = _cubic_2bit_carrier_word(
                            (packed >> (quad * 8)) & 0xFF
                        )
                        word_offset = (
                            group * (GROUP_SIZE // 4) + offs_word * 4 + quad
                        ) * stride_ik
                        activation0 = tl.load(
                            input_ptr + row0 * stride_im + word_offset
                        ).to(tl.int32)
                        activation1 = tl.load(
                            input_ptr + row1 * stride_im + word_offset
                        ).to(tl.int32)
                        dot0 = _cubic_dp4a(carriers, activation0[None, :], dot0)
                        dot1 = _cubic_dp4a(carriers, activation1[None, :], dot1)
                    if GROUPWISE_SCALE:
                        acc0 += weight_scale * tl.sum(
                            dot0.to(tl.float32) * group_scale0[None, :], axis=1
                        )
                        acc1 += weight_scale * tl.sum(
                            dot1.to(tl.float32) * group_scale1[None, :], axis=1
                        )
                    else:
                        acc0 += group_scale0 * weight_scale * tl.sum(dot0, axis=1)
                        acc1 += group_scale1 * weight_scale * tl.sum(dot1, axis=1)
                if MUL_ROUTED_WEIGHT:
                    acc0 *= tl.load(topk_weights_ptr + token0)
                    acc1 *= tl.load(topk_weights_ptr + token1)
                tl.store(
                    output_ptr + token0 * stride_om + offs_n * stride_on,
                    acc0,
                    mask=n_mask,
                )
                tl.store(
                    output_ptr + token1 * stride_om + offs_n * stride_on,
                    acc1,
                    mask=n_mask,
                )
            else:
                for group in tl.range(0, NUM_GROUPS):
                    if GROUPWISE_SCALE:
                        activation_group = (
                            group * GROUP_SIZE + offs_word * 16
                        ) // ACTIVATION_GROUP_SIZE
                        group_scale0 = tl.load(
                            input_scale_ptr
                            + row0 * stride_ism
                            + activation_group * stride_isg
                        ).to(tl.float32)
                    else:
                        group_scale0 = scale0
                    word_indices = group * group_words + offs_word
                    packed = tl.load(
                        weight_ptr
                        + expert_id * stride_we
                        + offs_n[:, None] * stride_wn
                        + word_indices[None, :] * stride_wp,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    weight_scale = tl.load(
                        scale_ptr
                        + expert_id * stride_se
                        + offs_n * stride_sn
                        + group * stride_sg,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        carriers = _cubic_2bit_carrier_word(
                            (packed >> (quad * 8)) & 0xFF
                        )
                        activation0 = tl.load(
                            input_ptr
                            + row0 * stride_im
                            + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad)
                            * stride_ik
                        ).to(tl.int32)
                        dot0 = _cubic_dp4a(carriers, activation0[None, :], dot0)
                    if GROUPWISE_SCALE:
                        acc0 += weight_scale * tl.sum(
                            dot0.to(tl.float32) * group_scale0[None, :], axis=1
                        )
                    else:
                        acc0 += group_scale0 * weight_scale * tl.sum(dot0, axis=1)
                if MUL_ROUTED_WEIGHT:
                    acc0 *= tl.load(topk_weights_ptr + token0)
                tl.store(
                    output_ptr + token0 * stride_om + offs_n * stride_on,
                    acc0,
                    mask=n_mask,
                )
        route_block += ROUTE_CTAS


@triton.jit
def _cubic_moe_grouped2_situ_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    output_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    stride_osm,
    stride_osg,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION_GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    OUTPUT_GROUPWISE_A8: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
    ROUTES_PER_BLOCK: tl.constexpr,
):
    """A small same-expert route group shares each pair of weight loads."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route_block = tl.program_id(1)

    while route_block * ROUTES_PER_BLOCK < num_tokens_post_padded:
        token_offsets = route_block * ROUTES_PER_BLOCK + tl.arange(0, ROUTES_PER_BLOCK)
        token_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
        token_mask = (token_ids >= 0) & (token_ids < num_valid_tokens)
        expert_id = tl.load(expert_ids_ptr + route_block).to(tl.int64)
        if expert_id != -1:
            input_rows = token_ids // TOP_K
            activation_scale = tl.load(
                input_scale_ptr + input_rows,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            gate_accumulator = tl.zeros((ROUTES_PER_BLOCK, BLOCK_N), dtype=tl.float32)
            up_accumulator = tl.zeros((ROUTES_PER_BLOCK, BLOCK_N), dtype=tl.float32)

            for group in tl.range(0, NUM_GROUPS):
                word_indices = group * group_words + offs_word
                weight_offsets = (
                    expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + word_indices[None, :] * stride_wp
                )
                gate_packed = tl.load(
                    weight_ptr + weight_offsets,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                up_packed = tl.load(
                    weight_ptr + weight_offsets + N * stride_wn,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                metadata_offsets = (
                    expert_id * stride_se + offs_n * stride_sn + group * stride_sg
                )
                gate_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    scale_ptr + metadata_offsets + N * stride_sn,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                route_lanes = tl.arange(0, ROUTES_PER_BLOCK)
                for r in tl.static_range(0, ROUTES_PER_BLOCK):
                    route_lane = route_lanes == r
                    input_row = tl.sum(tl.where(route_lane, input_rows, 0), axis=0)
                    route_valid = (
                        tl.sum(tl.where(route_lane, token_mask, 0), axis=0) != 0
                    )
                    route_scale = tl.sum(
                        tl.where(route_lane, activation_scale, 0.0), axis=0
                    )
                    gate_dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    up_dot = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        gate_carriers = _cubic_2bit_carrier_word(
                            (gate_packed >> (quad * 8)) & 0xFF
                        )
                        up_carriers = _cubic_2bit_carrier_word(
                            (up_packed >> (quad * 8)) & 0xFF
                        )
                        activation_word = tl.load(
                            input_ptr
                            + input_row * stride_im
                            + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad)
                            * stride_ik,
                            mask=route_valid,
                            other=0,
                        ).to(tl.int32)
                        gate_dot = _cubic_dp4a(
                            gate_carriers, activation_word[None, :], gate_dot
                        )
                        up_dot = _cubic_dp4a(
                            up_carriers, activation_word[None, :], up_dot
                        )
                    lane = route_lane[:, None]
                    gate_accumulator += (
                        lane
                        * (route_scale * gate_scale * tl.sum(gate_dot, axis=1))[None, :]
                    )
                    up_accumulator += (
                        lane
                        * (route_scale * up_scale * tl.sum(up_dot, axis=1))[None, :]
                    )

            if MUL_ROUTED_WEIGHT:
                routed_weight = tl.load(
                    topk_weights_ptr + token_ids,
                    mask=token_mask,
                    other=0.0,
                )
                gate_accumulator = gate_accumulator * routed_weight[:, None]
                up_accumulator = up_accumulator * routed_weight[:, None]
            gate = gate_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            up = up_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / BETA) - 1.0
            gate = BETA * gate_tanh * tl.sigmoid(gate)
            if HAS_LINEAR_BETA:
                up = LINEAR_BETA * (2.0 * tl.sigmoid(2.0 * up / LINEAR_BETA) - 1.0)
            tl.store(
                output_ptr
                + token_ids[:, None] * stride_om
                + offs_n[None, :] * stride_on,
                gate * up,
                mask=token_mask[:, None] & n_mask[None, :],
            )
        route_block += ROUTE_CTAS


@triton.jit(do_not_specialize=["num_valid_tokens"])
def _cubic_moe_pair_situ_gemv_2bit_a8_dp4a_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    output_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    stride_osm,
    stride_osg,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION_GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    OUTPUT_GROUPWISE_A8: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Pair-aware fused gate/up projection and SITU activation."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route_block = tl.program_id(1)

    while route_block * 2 < num_tokens_post_padded:
        route_offset = route_block * 2
        token0 = tl.load(sorted_token_ids_ptr + route_offset).to(tl.int64)
        token1 = tl.load(sorted_token_ids_ptr + route_offset + 1).to(tl.int64)
        valid0 = (token0 >= 0) & (token0 < num_valid_tokens)
        valid1 = (token1 >= 0) & (token1 < num_valid_tokens)
        expert_id = tl.load(expert_ids_ptr + route_block).to(tl.int64)
        if (expert_id != -1) & valid0:
            row0 = token0 // TOP_K
            activation_scale0 = tl.load(input_scale_ptr + row0).to(tl.float32)
            gate0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
            up0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
            if valid1:
                row1 = token1 // TOP_K
                activation_scale1 = tl.load(input_scale_ptr + row1).to(tl.float32)
                gate1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
                up1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for group in tl.range(0, NUM_GROUPS):
                    word_indices = group * group_words + offs_word
                    weight_offsets = (
                        expert_id * stride_we
                        + offs_n[:, None] * stride_wn
                        + word_indices[None, :] * stride_wp
                    )
                    gate_packed = tl.load(
                        weight_ptr + weight_offsets,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    up_packed = tl.load(
                        weight_ptr + weight_offsets + N * stride_wn,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    metadata_offsets = (
                        expert_id * stride_se + offs_n * stride_sn + group * stride_sg
                    )
                    gate_scale = tl.load(
                        scale_ptr + metadata_offsets,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    up_scale = tl.load(
                        scale_ptr + metadata_offsets + N * stride_sn,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    gate_dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    up_dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    gate_dot1 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    up_dot1 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        gate_carriers = _cubic_2bit_carrier_word(
                            (gate_packed >> (quad * 8)) & 0xFF
                        )
                        up_carriers = _cubic_2bit_carrier_word(
                            (up_packed >> (quad * 8)) & 0xFF
                        )
                        word_offset = (
                            group * (GROUP_SIZE // 4) + offs_word * 4 + quad
                        ) * stride_ik
                        activation0 = tl.load(
                            input_ptr + row0 * stride_im + word_offset
                        ).to(tl.int32)
                        activation1 = tl.load(
                            input_ptr + row1 * stride_im + word_offset
                        ).to(tl.int32)
                        gate_dot0 = _cubic_dp4a(
                            gate_carriers, activation0[None, :], gate_dot0
                        )
                        up_dot0 = _cubic_dp4a(
                            up_carriers, activation0[None, :], up_dot0
                        )
                        gate_dot1 = _cubic_dp4a(
                            gate_carriers, activation1[None, :], gate_dot1
                        )
                        up_dot1 = _cubic_dp4a(
                            up_carriers, activation1[None, :], up_dot1
                        )
                    gate0 += activation_scale0 * gate_scale * tl.sum(gate_dot0, axis=1)
                    up0 += activation_scale0 * up_scale * tl.sum(up_dot0, axis=1)
                    gate1 += activation_scale1 * gate_scale * tl.sum(gate_dot1, axis=1)
                    up1 += activation_scale1 * up_scale * tl.sum(up_dot1, axis=1)
                if MUL_ROUTED_WEIGHT:
                    routed0 = tl.load(topk_weights_ptr + token0)
                    routed1 = tl.load(topk_weights_ptr + token1)
                    gate0 *= routed0
                    up0 *= routed0
                    gate1 *= routed1
                    up1 *= routed1
                if OUTPUT_GROUPWISE_A8:
                    # Match the established BF16 cache boundary before
                    # producing the online carrier.
                    if OUTPUT_BF16:
                        gate0 = gate0.to(tl.bfloat16).to(tl.float32)
                        up0 = up0.to(tl.bfloat16).to(tl.float32)
                        gate1 = gate1.to(tl.bfloat16).to(tl.float32)
                        up1 = up1.to(tl.bfloat16).to(tl.float32)
                    else:
                        gate0 = gate0.to(tl.float16).to(tl.float32)
                        up0 = up0.to(tl.float16).to(tl.float32)
                        gate1 = gate1.to(tl.float16).to(tl.float32)
                        up1 = up1.to(tl.float16).to(tl.float32)
                else:
                    gate0 = gate0.to(output_ptr.dtype.element_ty).to(tl.float32)
                    up0 = up0.to(output_ptr.dtype.element_ty).to(tl.float32)
                    gate1 = gate1.to(output_ptr.dtype.element_ty).to(tl.float32)
                    up1 = up1.to(output_ptr.dtype.element_ty).to(tl.float32)
                gate0 = (
                    BETA
                    * (2.0 * tl.sigmoid(2.0 * gate0 / BETA) - 1.0)
                    * tl.sigmoid(gate0)
                )
                gate1 = (
                    BETA
                    * (2.0 * tl.sigmoid(2.0 * gate1 / BETA) - 1.0)
                    * tl.sigmoid(gate1)
                )
                if HAS_LINEAR_BETA:
                    up0 = LINEAR_BETA * (
                        2.0 * tl.sigmoid(2.0 * up0 / LINEAR_BETA) - 1.0
                    )
                    up1 = LINEAR_BETA * (
                        2.0 * tl.sigmoid(2.0 * up1 / LINEAR_BETA) - 1.0
                    )
                activated0 = gate0 * up0
                activated1 = gate1 * up1
                if OUTPUT_GROUPWISE_A8:
                    if OUTPUT_BF16:
                        activated0 = activated0.to(tl.bfloat16).to(tl.float32)
                        activated1 = activated1.to(tl.bfloat16).to(tl.float32)
                    else:
                        activated0 = activated0.to(tl.float16).to(tl.float32)
                        activated1 = activated1.to(tl.float16).to(tl.float32)
                    quantized0 = tl.zeros((BLOCK_N,), dtype=tl.int8)
                    quantized1 = tl.zeros((BLOCK_N,), dtype=tl.int8)
                    for subgroup in tl.static_range(
                        0, BLOCK_N // ACTIVATION_GROUP_SIZE
                    ):
                        subgroup_mask = (
                            tl.arange(0, BLOCK_N) // ACTIVATION_GROUP_SIZE
                        ) == subgroup
                        absmax0 = tl.maximum(
                            tl.max(tl.where(subgroup_mask, tl.abs(activated0), 0.0)),
                            1e-10,
                        )
                        absmax1 = tl.maximum(
                            tl.max(tl.where(subgroup_mask, tl.abs(activated1), 0.0)),
                            1e-10,
                        )
                        q0 = round_int8(activated0 * (127.0 / absmax0))
                        q1 = round_int8(activated1 * (127.0 / absmax1))
                        quantized0 = tl.where(subgroup_mask, q0, quantized0)
                        quantized1 = tl.where(subgroup_mask, q1, quantized1)
                        output_group = (
                            tl.program_id(0) * (BLOCK_N // ACTIVATION_GROUP_SIZE)
                            + subgroup
                        )
                        tl.store(
                            output_scale_ptr
                            + token0 * stride_osm
                            + output_group * stride_osg,
                            absmax0 / 127.0,
                        )
                        tl.store(
                            output_scale_ptr
                            + token1 * stride_osm
                            + output_group * stride_osg,
                            absmax1 / 127.0,
                        )
                    tl.store(
                        output_ptr + token0 * stride_om + offs_n * stride_on,
                        quantized0,
                        mask=n_mask,
                    )
                    tl.store(
                        output_ptr + token1 * stride_om + offs_n * stride_on,
                        quantized1,
                        mask=n_mask,
                    )
                else:
                    tl.store(
                        output_ptr + token0 * stride_om + offs_n * stride_on,
                        activated0,
                        mask=n_mask,
                    )
                    tl.store(
                        output_ptr + token1 * stride_om + offs_n * stride_on,
                        activated1,
                        mask=n_mask,
                    )
            else:
                for group in tl.range(0, NUM_GROUPS):
                    word_indices = group * group_words + offs_word
                    weight_offsets = (
                        expert_id * stride_we
                        + offs_n[:, None] * stride_wn
                        + word_indices[None, :] * stride_wp
                    )
                    gate_packed = tl.load(
                        weight_ptr + weight_offsets,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    up_packed = tl.load(
                        weight_ptr + weight_offsets + N * stride_wn,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    metadata_offsets = (
                        expert_id * stride_se + offs_n * stride_sn + group * stride_sg
                    )
                    gate_scale = tl.load(
                        scale_ptr + metadata_offsets,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    up_scale = tl.load(
                        scale_ptr + metadata_offsets + N * stride_sn,
                        mask=n_mask,
                        other=0.0,
                    ).to(tl.float32)
                    gate_dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    up_dot0 = tl.zeros((BLOCK_N, group_words), dtype=tl.int32)
                    for quad in tl.static_range(0, 4):
                        gate_carriers = _cubic_2bit_carrier_word(
                            (gate_packed >> (quad * 8)) & 0xFF
                        )
                        up_carriers = _cubic_2bit_carrier_word(
                            (up_packed >> (quad * 8)) & 0xFF
                        )
                        activation0 = tl.load(
                            input_ptr
                            + row0 * stride_im
                            + (group * (GROUP_SIZE // 4) + offs_word * 4 + quad)
                            * stride_ik
                        ).to(tl.int32)
                        gate_dot0 = _cubic_dp4a(
                            gate_carriers, activation0[None, :], gate_dot0
                        )
                        up_dot0 = _cubic_dp4a(
                            up_carriers, activation0[None, :], up_dot0
                        )
                    gate0 += activation_scale0 * gate_scale * tl.sum(gate_dot0, axis=1)
                    up0 += activation_scale0 * up_scale * tl.sum(up_dot0, axis=1)
                if MUL_ROUTED_WEIGHT:
                    routed0 = tl.load(topk_weights_ptr + token0)
                    gate0 *= routed0
                    up0 *= routed0
                if OUTPUT_GROUPWISE_A8:
                    if OUTPUT_BF16:
                        gate0 = gate0.to(tl.bfloat16).to(tl.float32)
                        up0 = up0.to(tl.bfloat16).to(tl.float32)
                    else:
                        gate0 = gate0.to(tl.float16).to(tl.float32)
                        up0 = up0.to(tl.float16).to(tl.float32)
                else:
                    gate0 = gate0.to(output_ptr.dtype.element_ty).to(tl.float32)
                    up0 = up0.to(output_ptr.dtype.element_ty).to(tl.float32)
                gate0 = (
                    BETA
                    * (2.0 * tl.sigmoid(2.0 * gate0 / BETA) - 1.0)
                    * tl.sigmoid(gate0)
                )
                if HAS_LINEAR_BETA:
                    up0 = LINEAR_BETA * (
                        2.0 * tl.sigmoid(2.0 * up0 / LINEAR_BETA) - 1.0
                    )
                activated0 = gate0 * up0
                if OUTPUT_GROUPWISE_A8:
                    if OUTPUT_BF16:
                        activated0 = activated0.to(tl.bfloat16).to(tl.float32)
                    else:
                        activated0 = activated0.to(tl.float16).to(tl.float32)
                    quantized0 = tl.zeros((BLOCK_N,), dtype=tl.int8)
                    for subgroup in tl.static_range(
                        0, BLOCK_N // ACTIVATION_GROUP_SIZE
                    ):
                        subgroup_mask = (
                            tl.arange(0, BLOCK_N) // ACTIVATION_GROUP_SIZE
                        ) == subgroup
                        absmax0 = tl.maximum(
                            tl.max(tl.where(subgroup_mask, tl.abs(activated0), 0.0)),
                            1e-10,
                        )
                        q0 = round_int8(activated0 * (127.0 / absmax0))
                        quantized0 = tl.where(subgroup_mask, q0, quantized0)
                        output_group = (
                            tl.program_id(0) * (BLOCK_N // ACTIVATION_GROUP_SIZE)
                            + subgroup
                        )
                        tl.store(
                            output_scale_ptr
                            + token0 * stride_osm
                            + output_group * stride_osg,
                            absmax0 / 127.0,
                        )
                    tl.store(
                        output_ptr + token0 * stride_om + offs_n * stride_on,
                        quantized0,
                        mask=n_mask,
                    )
                else:
                    tl.store(
                        output_ptr + token0 * stride_om + offs_n * stride_on,
                        activated0,
                        mask=n_mask,
                    )
        route_block += ROUTE_CTAS


@triton.autotune(
    configs=_CUBIC_MOE_2BIT_A16_CONFIGS,
    key=["N", "K", "GROUP_SIZE", "ROUTE_CTAS", "TOP_K", "HAS_LINEAR_BETA"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_situ_gemv_2bit_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    APPLY_SITU: tl.constexpr,
    DYNAMIC_A8: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_words: tl.constexpr = GROUP_SIZE // 16
    offs_word = tl.arange(0, group_words)
    route = tl.program_id(1)
    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        if DYNAMIC_A8:
            activation_scale = tl.load(input_scale_ptr + token_id // TOP_K).to(
                tl.float32
            )
        else:
            activation_scale = 1.0
        gate_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            word_indices = group * group_words + offs_word
            weight_offsets = (
                expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + word_indices[None, :] * stride_wp
            )
            gate_packed = tl.load(
                weight_ptr + weight_offsets,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            up_packed = tl.load(
                weight_ptr + weight_offsets + N * stride_wn,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            gate_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                scale_ptr + metadata_offsets + N * stride_sn,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            group_k = group * GROUP_SIZE + offs_word * 16
            gate_contribution = tl.zeros(
                (BLOCK_N, group_words),
                dtype=tl.float32,
            )
            up_contribution = tl.zeros(
                (BLOCK_N, group_words),
                dtype=tl.float32,
            )
            for lane in tl.static_range(0, 16):
                global_k = group_k + lane
                activation = tl.load(
                    input_ptr + (token_id // TOP_K) * stride_im + global_k * stride_ik,
                    mask=global_k < K,
                    other=0.0,
                ).to(tl.float32)
                gate_code = (gate_packed >> (lane * 2)) & 3
                gate_value = (gate_code & 1) * (1 - (gate_code & 2))
                up_code = (up_packed >> (lane * 2)) & 3
                up_value = (up_code & 1) * (1 - (up_code & 2))
                gate_contribution += gate_value * activation[None, :]
                up_contribution += up_value * activation[None, :]
            gate_accumulator += (
                activation_scale * gate_scale * tl.sum(gate_contribution, axis=1)
            )
            up_accumulator += (
                activation_scale * up_scale * tl.sum(up_contribution, axis=1)
            )

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + token_id)
            gate_accumulator *= routed_weight
            up_accumulator *= routed_weight
        if APPLY_SITU:
            gate = gate_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            up = up_accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / BETA) - 1.0
            gate = BETA * gate_tanh * tl.sigmoid(gate)
            if HAS_LINEAR_BETA:
                up = LINEAR_BETA * (2.0 * tl.sigmoid(2.0 * up / LINEAR_BETA) - 1.0)
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                gate * up,
                mask=n_mask,
            )
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                gate_accumulator,
                mask=n_mask,
            )
            tl.store(
                output_ptr + token_id * stride_om + (offs_n + N) * stride_on,
                up_accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS


@triton.autotune(
    configs=_CUBIC_MOE_GENERIC_GEMV_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "GROUP_OUT",
        "BLOCK_K",
        "ROUTE_CTAS",
        "TOP_K",
        "SUM_ROUTES",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    while route < num_tokens_post_padded:
        if route < num_tokens_post_padded:
            token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
            expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
            accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                global_k = k_block * BLOCK_K + offs_k
                k_mask = global_k < K
                activation = tl.load(
                    input_ptr + (token_id // TOP_K) * stride_im + global_k * stride_ik,
                    mask=k_mask,
                    other=0.0,
                )
                bit_positions = global_k[None, :] * NUM_BITS
                byte_indices = bit_positions // 8
                shifts = bit_positions % 8
                packed_ptrs = (
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + byte_indices * stride_wp
                )
                weight_mask = n_mask[:, None] & k_mask[None, :]
                low = tl.load(
                    packed_ptrs,
                    mask=weight_mask,
                    other=0,
                ).to(tl.int32)
                if 8 % NUM_BITS == 0:
                    high = 0
                else:
                    high = tl.load(
                        packed_ptrs + stride_wp,
                        mask=weight_mask & (byte_indices + 1 < PACKED_K),
                        other=0,
                    ).to(tl.int32)
                if GROUP_SIZE == 1:
                    if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
                        metadata_ptrs = (
                            expert_id * stride_se
                            + (tl.program_id(0) * BLOCK_N // GROUP_OUT) * stride_sn
                            + global_k * stride_sg
                        )
                        metadata_mask = k_mask
                    else:
                        metadata_ptrs = (
                            expert_id * stride_se
                            + (offs_n[:, None] // GROUP_OUT) * stride_sn
                            + global_k[None, :] * stride_sg
                        )
                        metadata_mask = n_mask[:, None] & k_mask[None, :]
                else:
                    group = (k_block * BLOCK_K) // GROUP_SIZE
                    metadata_ptrs = (
                        expert_id * stride_se
                        + (offs_n // GROUP_OUT) * stride_sn
                        + group * stride_sg
                    )
                    metadata_mask = n_mask & (group < NUM_GROUPS)
                scale = tl.load(
                    scale_ptr + metadata_ptrs,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)
                if (
                    GROUP_SIZE == 1
                    and GROUP_OUT >= BLOCK_N
                    and GROUP_OUT % BLOCK_N == 0
                ):
                    scale = scale[None, :]
                elif GROUP_SIZE != 1:
                    scale = scale[:, None]
                if NUM_BITS > 2:
                    cubic_a = tl.load(
                        cubic_a_ptr + metadata_ptrs,
                        mask=metadata_mask,
                        other=1.0,
                    ).to(tl.float32)
                    cubic_b = tl.load(
                        cubic_b_ptr + metadata_ptrs,
                        mask=metadata_mask,
                        other=0.0,
                    ).to(tl.float32)
                    if (
                        GROUP_SIZE == 1
                        and GROUP_OUT >= BLOCK_N
                        and GROUP_OUT % BLOCK_N == 0
                    ):
                        cubic_a = cubic_a[None, :]
                        cubic_b = cubic_b[None, :]
                    elif GROUP_SIZE != 1:
                        cubic_a = cubic_a[:, None]
                        cubic_b = cubic_b[:, None]
                else:
                    cubic_a = 1.0
                    cubic_b = 0.0
                if NUM_BITS <= 4:
                    weight = _decode_cubic_lut(
                        low,
                        high,
                        shifts,
                        scale,
                        cubic_a,
                        cubic_b,
                        NUM_BITS,
                    )
                else:
                    weight = _decode_cubic_direct(
                        low,
                        high,
                        shifts,
                        scale,
                        cubic_a,
                        cubic_b,
                        NUM_BITS,
                    )
                accumulator += tl.sum(
                    weight * activation[None, :].to(tl.float32),
                    axis=1,
                )

            if MUL_ROUTED_WEIGHT:
                accumulator *= tl.load(topk_weights_ptr + token_id)
            if SUM_ROUTES:
                route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
            else:
                tl.store(
                    output_ptr + token_id * stride_om + offs_n * stride_on,
                    accumulator,
                    mask=n_mask,
                )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_MOE_DENSE_N_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "GROUP_OUT",
        "BLOCK_M",
        "BLOCK_K",
        "TOP_K",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_GROUP_LUT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_group) % group_size_m)
    pid_n = (pid % num_pid_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return
    token_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
    token_mask = token_ids < num_valid_tokens
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_n = offs_n_raw % N
    if expert_id == -1:
        output_ptrs = (
            output_ptr
            + token_ids[:, None] * stride_om
            + offs_n_raw[None, :] * stride_on
        )
        tl.store(
            output_ptrs,
            0.0,
            mask=token_mask[:, None] & (offs_n_raw[None, :] < N),
        )
        return

    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in tl.range(0, tl.cdiv(K, BLOCK_K)):
        global_k = k_block * BLOCK_K + offs_k
        k_mask = global_k < K
        activation = tl.load(
            input_ptr
            + (token_ids[:, None] // TOP_K) * stride_im
            + global_k[None, :] * stride_ik,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        bit_positions = global_k[:, None] * NUM_BITS
        byte_indices = bit_positions // 8
        shifts = bit_positions % 8
        packed_ptrs = (
            weight_ptr
            + expert_id * stride_we
            + offs_n[None, :] * stride_wn
            + byte_indices * stride_wp
        )
        weight_mask = k_mask[:, None] & (offs_n_raw[None, :] < N)
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        if 8 % NUM_BITS == 0:
            high = 0
        else:
            high = tl.load(
                packed_ptrs + stride_wp,
                mask=weight_mask & (byte_indices + 1 < PACKED_K),
                other=0,
            ).to(tl.int32)
        if USE_GROUP_LUT:
            group = (k_block * BLOCK_K) // GROUP_SIZE
            if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
                metadata_ptrs = (
                    expert_id * stride_se
                    + (pid_n * BLOCK_N // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                scale = tl.load(scale_ptr + metadata_ptrs).to(tl.float32)
            else:
                metadata_ptrs = (
                    expert_id * stride_se
                    + (offs_n // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                metadata_mask = (offs_n_raw < N) & (group < NUM_GROUPS)
                scale = tl.load(
                    scale_ptr + metadata_ptrs,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
            if NUM_BITS > 2:
                if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
                    cubic_a = tl.load(cubic_a_ptr + metadata_ptrs).to(tl.float32)
                    cubic_b = tl.load(cubic_b_ptr + metadata_ptrs).to(tl.float32)
                else:
                    cubic_a = tl.load(
                        cubic_a_ptr + metadata_ptrs,
                        mask=metadata_mask,
                        other=1.0,
                    ).to(tl.float32)[None, :]
                    cubic_b = tl.load(
                        cubic_b_ptr + metadata_ptrs,
                        mask=metadata_mask,
                        other=0.0,
                    ).to(tl.float32)[None, :]
            else:
                cubic_a = 1.0
                cubic_b = 0.0
            weight = _decode_cubic_lut(
                low,
                high,
                shifts,
                scale,
                cubic_a,
                cubic_b,
                NUM_BITS,
            ).to(activation.dtype)
        else:
            groups = global_k[:, None] // GROUP_SIZE
            if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
                metadata_ptrs = (
                    expert_id * stride_se
                    + (pid_n * BLOCK_N // GROUP_OUT) * stride_sn
                    + groups * stride_sg
                )
                metadata_mask = k_mask[:, None] & (groups < NUM_GROUPS)
            else:
                metadata_ptrs = (
                    expert_id * stride_se
                    + (offs_n[None, :] // GROUP_OUT) * stride_sn
                    + groups * stride_sg
                )
                metadata_mask = weight_mask & (groups < NUM_GROUPS)
            scale = tl.load(
                scale_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=0.0,
            )
            cubic_a = tl.load(
                cubic_a_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=1.0,
            ).to(tl.float32)
            cubic_b = tl.load(
                cubic_b_ptr + metadata_ptrs,
                mask=metadata_mask,
                other=0.0,
            ).to(tl.float32)
            weight = _decode_cubic_direct(
                low,
                high,
                shifts,
                scale.to(tl.float32),
                cubic_a,
                cubic_b,
                NUM_BITS,
            ).to(activation.dtype)
        accumulator = tl.dot(activation, weight, acc=accumulator)

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + token_ids, mask=token_mask, other=0.0
        )
        accumulator *= routed_weight[:, None]
    output_ptrs = (
        output_ptr + token_ids[:, None] * stride_om + offs_n_raw[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=token_mask[:, None] & (offs_n_raw[None, :] < N),
    )


@triton.autotune(
    configs=_CUBIC_MOE_GEMV_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "BLOCK_K",
        "ROUTE_CTAS",
        "TOP_K",
        "SUM_ROUTES",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_dynamic_a8_gemv_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_block in tl.range(0, tl.cdiv(K, BLOCK_K)):
            global_k = k_block * BLOCK_K + offs_k
            k_mask = global_k < K
            activation = tl.load(
                input_ptr + input_row * stride_im + global_k * stride_ik,
                mask=k_mask,
                other=0,
            ).to(tl.int32)
            bit_positions = global_k[None, :] * NUM_BITS
            byte_indices = bit_positions // 8
            shifts = bit_positions % 8
            packed_ptrs = (
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + byte_indices * stride_wp
            )
            weight_mask = n_mask[:, None] & k_mask[None, :]
            low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
            if 8 % NUM_BITS == 0:
                high = 0
            else:
                high = tl.load(
                    packed_ptrs + stride_wp,
                    mask=weight_mask & (byte_indices + 1 < PACKED_K),
                    other=0,
                ).to(tl.int32)
            raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)
            if GROUP_SIZE == 1:
                metadata_offsets = (
                    expert_id * stride_se
                    + (offs_n[:, None] // GROUP_OUT) * stride_sn
                    + global_k[None, :] * stride_sg
                )
                metadata_mask = n_mask[:, None] & k_mask[None, :]
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=1.0,
                ).to(tl.float32)
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)
                carrier = _cubic_dynamic_a8_carrier(
                    raw, cubic_a, cubic_b, NUM_BITS
                )
                partial = tl.sum(
                    carrier.to(tl.float32)
                    * activation[None, :].to(tl.float32)
                    * weight_scale,
                    axis=1,
                )
                accumulator += partial * activation_scale * (1.0 / 127.0)
            else:
                group = (k_block * BLOCK_K) // GROUP_SIZE
                metadata_offsets = (
                    expert_id * stride_se
                    + (offs_n // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                metadata_mask = n_mask & (group < NUM_GROUPS)
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)
                if NUM_BITS > 2:
                    cubic_a = tl.load(
                        cubic_a_ptr + metadata_offsets,
                        mask=metadata_mask,
                        other=1.0,
                    ).to(tl.float32)[:, None]
                    cubic_b = tl.load(
                        cubic_b_ptr + metadata_offsets,
                        mask=metadata_mask,
                        other=0.0,
                    ).to(tl.float32)[:, None]
                else:
                    cubic_a = 1.0
                    cubic_b = 0.0
                if NUM_BITS > 2 and NUM_BITS <= 4:
                    carrier = _cubic_dynamic_a8_carrier_lut(
                        raw,
                        cubic_a,
                        cubic_b,
                        NUM_BITS,
                    )
                else:
                    carrier = _cubic_dynamic_a8_carrier(
                        raw, cubic_a, cubic_b, NUM_BITS
                    )
                partial = tl.sum(
                    carrier.to(tl.int32) * activation[None, :], axis=1
                )
                accumulator += (
                    partial.to(tl.float32)
                    * activation_scale
                    * weight_scale
                    * (1.0 / 127.0)
                )

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS

    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


def _prune_cubic_a8_gemv_configs(configs, _named_args, **kwargs):
    group_size = kwargs["GROUP_SIZE"]
    num_bits = kwargs["NUM_BITS"]
    if kwargs["PRECOMPUTED_3BIT_LEVELS"]:
        return [config for config in configs if config.kwargs["USE_SPECIAL_3BIT"]]
    if num_bits == 3:
        return [
            config
            for config in configs
            if config.kwargs["USE_SPECIAL_3BIT"]
            or (
                not config.kwargs["USE_LUT"]
                and (
                    config.kwargs["USE_DP4A"]
                    or (
                        config.kwargs["BLOCK_N"] == 8
                        and config.kwargs["BLOCK_K"] == min(group_size, 128)
                    )
                )
            )
        ]
    return [
        config
        for config in configs
        if (
            config.kwargs["USE_SPECIAL_3BIT"]
            or config.kwargs["USE_DP4A"]
            or (
                config.kwargs["BLOCK_K"] <= group_size
                and group_size % config.kwargs["BLOCK_K"] == 0
            )
        )
        and (not config.kwargs["USE_SPECIAL_3BIT"] or num_bits == 3)
        and (not config.kwargs["USE_LUT"] or num_bits <= 6)
    ]


_CUBIC_A8_GEMV_AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": False,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 4,
            "BLOCK_K": 64,
            "USE_DP4A": False,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 128,
            "USE_DP4A": False,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 128,
            "USE_DP4A": False,
            "USE_LUT": True,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 64,
            "USE_DP4A": False,
            "USE_LUT": True,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": False,
            "USE_LUT": True,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": True,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": True,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": True,
            "USE_LUT": True,
            "USE_SPECIAL_3BIT": False,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_N": 4,
            "BLOCK_K": 32,
            "USE_DP4A": False,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": True,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_N": 8,
            "BLOCK_K": 32,
            "USE_DP4A": False,
            "USE_LUT": False,
            "USE_SPECIAL_3BIT": True,
        },
        num_warps=8,
        num_stages=2,
    ),
]


@triton.autotune(
    configs=_CUBIC_A8_GEMV_AUTOTUNE_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "GROUP_OUT",
        "ROUTE_CTAS",
        "MUL_ROUTED_WEIGHT",
        "TOP_K",
        "SUM_ROUTES",
        "PRECOMPUTED_3BIT_LEVELS",
        "GROUPWISE_SCALE",
    ],
    prune_configs_by={"early_config_prune": _prune_cubic_a8_gemv_configs},
    cache_results=True,
)
@triton.jit
def _cubic_moe_dynamic_a8_autotune_gemv_kernel(
    input_ptr,
    input_words_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_im,
    stride_ik,
    stride_iwm,
    stride_iwk,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
    PRECOMPUTED_3BIT_LEVELS: tl.constexpr,
    USE_DP4A: tl.constexpr,
    USE_LUT: tl.constexpr,
    USE_SPECIAL_3BIT: tl.constexpr,
    GROUPWISE_SCALE: tl.constexpr,
):
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_quads: tl.constexpr = GROUP_SIZE // 4
    offs_quad = tl.arange(0, group_quads)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        if not GROUPWISE_SCALE:
            activation_scale = tl.load(input_scale_ptr + input_row).to(tl.float32)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        if USE_SPECIAL_3BIT:
            group_packs: tl.constexpr = GROUP_SIZE // 8
            group_bytes: tl.constexpr = GROUP_SIZE * 3 // 8
            offs_pack = tl.arange(0, group_packs)
            t1: tl.constexpr = 1.0 / 3.0
            t2: tl.constexpr = 2.0 / 3.0
            for group in tl.static_range(0, NUM_GROUPS):
                if GROUPWISE_SCALE:
                    group_activation_scale = tl.load(
                        input_scale_ptr + input_row * stride_ism + group * stride_isg
                    ).to(tl.float32)
                else:
                    group_activation_scale = activation_scale
                byte_indices = group * group_bytes + offs_pack * 3
                packed_ptrs = (
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + byte_indices[None, :] * stride_wp
                )
                packed = tl.load(
                    packed_ptrs,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                packed |= (
                    tl.load(
                        packed_ptrs + stride_wp,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    << 8
                )
                packed |= (
                    tl.load(
                        packed_ptrs + 2 * stride_wp,
                        mask=n_mask[:, None],
                        other=0,
                    ).to(tl.int32)
                    << 16
                )
                metadata_offsets = (
                    expert_id * stride_se
                    + (offs_n // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=n_mask,
                    other=1.0,
                ).to(tl.float32)
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                if PRECOMPUTED_3BIT_LEVELS:
                    level1 = cubic_a
                    level2 = cubic_b
                else:
                    cubic_c = 1.0 - cubic_a - cubic_b
                    level1 = tl.extra.cuda.libdevice.rint(
                        127.0 * t1 * (cubic_a + t1 * (cubic_b + t1 * cubic_c))
                    )
                    level2 = tl.extra.cuda.libdevice.rint(
                        127.0 * t2 * (cubic_a + t2 * (cubic_b + t2 * cubic_c))
                    )
                group_k = group * GROUP_SIZE + offs_pack * 8
                if PRECOMPUTED_3BIT_LEVELS:
                    lut0, lut1 = _cubic_3bit_carrier_luts(
                        level1.to(tl.int32), level2.to(tl.int32)
                    )
                    carrier_lo, carrier_hi = _cubic_3bit_carrier_words(
                        packed,
                        lut0[:, None],
                        lut1[:, None],
                    )
                    input_word_base = group * group_quads + offs_pack * 2
                    activation_lo = tl.load(
                        input_words_ptr
                        + input_row * stride_iwm
                        + input_word_base * stride_iwk
                    ).to(tl.int32)
                    activation_hi = tl.load(
                        input_words_ptr
                        + input_row * stride_iwm
                        + (input_word_base + 1) * stride_iwk
                    ).to(tl.int32)
                    dot_lo = _cubic_dp4a(
                        carrier_lo,
                        activation_lo[None, :],
                        tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
                    )
                    dot_hi = _cubic_dp4a(
                        carrier_hi,
                        activation_hi[None, :],
                        tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
                    )
                    group_dot = tl.sum(dot_lo + dot_hi, axis=1).to(tl.float32)
                else:
                    contribution = tl.zeros((BLOCK_N, group_packs), dtype=tl.float32)
                    for lane in tl.static_range(0, 8):
                        global_k = group_k + lane
                        activation = tl.load(
                            input_ptr + input_row * stride_im + global_k * stride_ik,
                            mask=global_k < K,
                            other=0,
                        ).to(tl.float32)
                        code = (packed >> (lane * 3)) & 7
                        signed = tl.where(code >= 4, code - 8, code)
                        signed = tl.where(signed == -4, 0, signed)
                        magnitude = tl.abs(signed)
                        carrier = tl.where(magnitude == 1, level1[:, None], 0.0)
                        carrier = tl.where(magnitude == 2, level2[:, None], carrier)
                        carrier = tl.where(magnitude == 3, 127.0, carrier)
                        carrier = tl.where(signed < 0, -carrier, carrier)
                        contribution += carrier * activation[None, :]
                    group_dot = tl.sum(contribution, axis=1)
                accumulator += (
                    group_dot * group_activation_scale * weight_scale * (1.0 / 127.0)
                )
        elif USE_DP4A:
            for group in tl.range(0, NUM_GROUPS):
                if GROUPWISE_SCALE:
                    group_activation_scale = tl.load(
                        input_scale_ptr + input_row * stride_ism + group * stride_isg
                    ).to(tl.float32)
                else:
                    group_activation_scale = activation_scale
                metadata_offsets = (
                    expert_id * stride_se
                    + (offs_n // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=n_mask,
                    other=1.0,
                ).to(tl.float32)[:, None]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)[:, None]
                carrier_word = tl.zeros((BLOCK_N, group_quads), dtype=tl.int32)
                for lane in tl.static_range(0, 4):
                    global_k = group * GROUP_SIZE + offs_quad * 4 + lane
                    bit_positions = global_k * NUM_BITS
                    byte_indices = bit_positions // 8
                    shifts = bit_positions % 8
                    packed_ptrs = (
                        weight_ptr
                        + expert_id * stride_we
                        + offs_n[:, None] * stride_wn
                        + byte_indices[None, :] * stride_wp
                    )
                    weight_mask = n_mask[:, None] & (global_k[None, :] < K)
                    low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
                    if 8 % NUM_BITS == 0:
                        high = 0
                    else:
                        high = tl.load(
                            packed_ptrs + stride_wp,
                            mask=weight_mask & (byte_indices[None, :] + 1 < PACKED_K),
                            other=0,
                        ).to(tl.int32)
                    raw = (
                        (low >> shifts[None, :]) | (high << (8 - shifts[None, :]))
                    ) & ((1 << NUM_BITS) - 1)
                    if USE_LUT and NUM_BITS <= 6:
                        carrier = _cubic_dynamic_a8_carrier_lut(
                            raw,
                            cubic_a,
                            cubic_b,
                            NUM_BITS,
                        )
                    else:
                        carrier = _cubic_dynamic_a8_carrier(
                            raw,
                            cubic_a,
                            cubic_b,
                            NUM_BITS,
                        )
                    carrier_word |= (carrier.to(tl.int32) & 0xFF) << (lane * 8)
                activation_word = tl.load(
                    input_words_ptr
                    + input_row * stride_iwm
                    + (group * group_quads + offs_quad) * stride_iwk
                ).to(tl.int32)
                dot = _cubic_dp4a(
                    carrier_word,
                    activation_word[None, :],
                    tl.zeros((BLOCK_N, group_quads), dtype=tl.int32),
                )
                group_dot = tl.sum(dot, axis=1)
                accumulator += (
                    group_dot.to(tl.float32)
                    * group_activation_scale
                    * weight_scale
                    * (1.0 / 127.0)
                )
        else:
            offs_k = tl.arange(0, BLOCK_K)
            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                global_k = k_block * BLOCK_K + offs_k
                k_mask = global_k < K
                activation = tl.load(
                    input_ptr + input_row * stride_im + global_k * stride_ik,
                    mask=k_mask,
                    other=0,
                ).to(tl.int32)
                bit_positions = global_k[None, :] * NUM_BITS
                byte_indices = bit_positions // 8
                shifts = bit_positions % 8
                packed_ptrs = (
                    weight_ptr
                    + expert_id * stride_we
                    + offs_n[:, None] * stride_wn
                    + byte_indices * stride_wp
                )
                weight_mask = n_mask[:, None] & k_mask[None, :]
                low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
                if 8 % NUM_BITS == 0:
                    high = 0
                else:
                    high = tl.load(
                        packed_ptrs + stride_wp,
                        mask=weight_mask & (byte_indices + 1 < PACKED_K),
                        other=0,
                    ).to(tl.int32)
                raw = ((low >> shifts) | (high << (8 - shifts))) & ((1 << NUM_BITS) - 1)
                group = (k_block * BLOCK_K) // GROUP_SIZE
                if GROUPWISE_SCALE:
                    group_activation_scale = tl.load(
                        input_scale_ptr + input_row * stride_ism + group * stride_isg
                    ).to(tl.float32)
                else:
                    group_activation_scale = activation_scale
                metadata_offsets = (
                    expert_id * stride_se
                    + (offs_n // GROUP_OUT) * stride_sn
                    + group * stride_sg
                )
                weight_scale = tl.load(
                    scale_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=n_mask,
                    other=1.0,
                ).to(tl.float32)[:, None]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)[:, None]
                if USE_LUT and NUM_BITS <= 6:
                    carrier = _cubic_dynamic_a8_carrier_lut(
                        raw,
                        cubic_a,
                        cubic_b,
                        NUM_BITS,
                    )
                else:
                    carrier = _cubic_dynamic_a8_carrier(
                        raw,
                        cubic_a,
                        cubic_b,
                        NUM_BITS,
                    )
                partial = tl.sum(carrier.to(tl.int32) * activation[None, :], axis=1)
                accumulator += (
                    partial.to(tl.float32)
                    * group_activation_scale
                    * weight_scale
                    * (1.0 / 127.0)
                )

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_MOE_GEMV_CONFIGS,
    key=["N", "K", "GROUP_SIZE", "ROUTE_CTAS", "TOP_K", "SUM_ROUTES"],
    cache_results=True,
)
@triton.jit
def _cubic_moe_gemv_3bit_groupwise_a8_dp4a_kernel(
    input_words_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    level1_ptr,
    level2_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_iwm,
    stride_iwk,
    stride_ism,
    stride_isg,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUTE_CTAS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    SUM_ROUTES: tl.constexpr,
):
    """Precomputed W3 carrier dot with per-row/per-K-group A8 scales."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    group_packs: tl.constexpr = GROUP_SIZE // 8
    group_quads: tl.constexpr = GROUP_SIZE // 4
    group_bytes: tl.constexpr = GROUP_SIZE * 3 // 8
    offs_pack = tl.arange(0, group_packs)
    route = tl.program_id(1)
    route_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while route < num_tokens_post_padded:
        token_id = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert_id = tl.load(expert_ids_ptr + route).to(tl.int64)
        input_row = token_id // TOP_K
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for group in tl.static_range(0, NUM_GROUPS):
            activation_scale = tl.load(
                input_scale_ptr + input_row * stride_ism + group * stride_isg
            ).to(tl.float32)
            byte_indices = group * group_bytes + offs_pack * 3
            packed_ptrs = (
                weight_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + byte_indices[None, :] * stride_wp
            )
            packed = tl.load(
                packed_ptrs,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            packed |= (
                tl.load(
                    packed_ptrs + stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 8
            )
            packed |= (
                tl.load(
                    packed_ptrs + 2 * stride_wp,
                    mask=n_mask[:, None],
                    other=0,
                ).to(tl.int32)
                << 16
            )
            metadata_offsets = (
                expert_id * stride_se + offs_n * stride_sn + group * stride_sg
            )
            weight_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            level1 = tl.load(
                level1_ptr + metadata_offsets,
                mask=n_mask,
                other=0,
            ).to(tl.int32)
            level2 = tl.load(
                level2_ptr + metadata_offsets,
                mask=n_mask,
                other=0,
            ).to(tl.int32)
            lut0, lut1 = _cubic_3bit_carrier_luts(level1, level2)
            carrier_lo, carrier_hi = _cubic_3bit_carrier_words(
                packed,
                lut0[:, None],
                lut1[:, None],
            )
            input_word_base = group * group_quads + offs_pack * 2
            activation_lo = tl.load(
                input_words_ptr + input_row * stride_iwm + input_word_base * stride_iwk
            ).to(tl.int32)
            activation_hi = tl.load(
                input_words_ptr
                + input_row * stride_iwm
                + (input_word_base + 1) * stride_iwk
            ).to(tl.int32)
            dot_lo = _cubic_dp4a(
                carrier_lo,
                activation_lo[None, :],
                tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
            )
            dot_hi = _cubic_dp4a(
                carrier_hi,
                activation_hi[None, :],
                tl.zeros((BLOCK_N, group_packs), dtype=tl.int32),
            )
            group_dot = tl.sum(dot_lo + dot_hi, axis=1).to(tl.float32)
            accumulator += group_dot * activation_scale * weight_scale * (1.0 / 127.0)

        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + token_id)
        if SUM_ROUTES:
            route_sum += accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
        else:
            tl.store(
                output_ptr + token_id * stride_om + offs_n * stride_on,
                accumulator,
                mask=n_mask,
            )
        route += ROUTE_CTAS
    if SUM_ROUTES:
        tl.store(
            output_ptr + tl.program_id(1) * stride_om + offs_n * stride_on,
            route_sum,
            mask=n_mask,
        )


@triton.autotune(
    configs=_CUBIC_MOE_DENSE_N_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_BITS",
        "GROUP_SIZE",
        "GROUP_OUT",
        "BLOCK_M",
        "BLOCK_K",
        "TOP_K",
        "PRECOMPUTED_3BIT_LEVELS",
        "PRECOMPUTED_CARRIER_LUT",
    ],
    cache_results=True,
)
@triton.jit
def _cubic_moe_dynamic_a8_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    scale_ptr,
    cubic_a_ptr,
    cubic_b_ptr,
    carrier_lut_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wp,
    stride_se,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    PRECOMPUTED_3BIT_LEVELS: tl.constexpr,
    PRECOMPUTED_CARRIER_LUT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_group) % group_size_m)
    pid_n = (pid % num_pid_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return
    token_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
    token_mask = token_ids < num_valid_tokens
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_n = offs_n_raw % N
    if expert_id == -1:
        output_ptrs = (
            output_ptr
            + token_ids[:, None] * stride_om
            + offs_n_raw[None, :] * stride_on
        )
        tl.store(
            output_ptrs,
            0.0,
            mask=token_mask[:, None] & (offs_n_raw[None, :] < N),
        )
        return

    input_rows = token_ids // TOP_K
    activation_scale = tl.load(
        input_scale_ptr + input_rows,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)[:, None]
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for group in range(0, NUM_GROUPS):
        if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
            metadata_offsets = (
                expert_id * stride_se
                + (pid_n * BLOCK_N // GROUP_OUT) * stride_sn
                + group * stride_sg
            )
            weight_scale = tl.load(scale_ptr + metadata_offsets).to(tl.float32)
        else:
            metadata_offsets = (
                expert_id * stride_se
                + (offs_n // GROUP_OUT) * stride_sn
                + group * stride_sg
            )
            metadata_mask = offs_n_raw < N
            weight_scale = tl.load(
                scale_ptr + metadata_offsets,
                mask=metadata_mask,
                other=0.0,
            ).to(tl.float32)[None, :]
        if NUM_BITS > 2:
            if GROUP_OUT >= BLOCK_N and GROUP_OUT % BLOCK_N == 0:
                cubic_a = tl.load(cubic_a_ptr + metadata_offsets).to(tl.float32)
                cubic_b = tl.load(cubic_b_ptr + metadata_offsets).to(tl.float32)
            else:
                cubic_a = tl.load(
                    cubic_a_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=1.0,
                ).to(tl.float32)[None, :]
                cubic_b = tl.load(
                    cubic_b_ptr + metadata_offsets,
                    mask=metadata_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
        for group_k_block in range(0, GROUP_SIZE // BLOCK_K):
            global_k = group * GROUP_SIZE + group_k_block * BLOCK_K + offs_k
            k_mask = global_k < K
            activation = tl.load(
                input_ptr
                + input_rows[:, None] * stride_im
                + global_k[None, :] * stride_ik,
                mask=token_mask[:, None] & k_mask[None, :],
                other=0,
            )
            bit_positions = global_k[:, None] * NUM_BITS
            byte_indices = bit_positions // 8
            shifts = bit_positions % 8
            packed_ptrs = (
                weight_ptr
                + expert_id * stride_we
                + offs_n[None, :] * stride_wn
                + byte_indices * stride_wp
            )
            weight_mask = k_mask[:, None] & (offs_n_raw[None, :] < N)
            low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
            if 8 % NUM_BITS == 0:
                high = 0
            else:
                high = tl.load(
                    packed_ptrs + stride_wp,
                    mask=weight_mask & (byte_indices + 1 < PACKED_K),
                    other=0,
                ).to(tl.int32)
            raw = ((low >> shifts) | (high << (8 - shifts))) & (
                (1 << NUM_BITS) - 1
            )

            if NUM_BITS == 1:
                carrier = (raw * 254 - 127).to(tl.int8)
            else:
                sign_bit: tl.constexpr = 1 << (NUM_BITS - 1)
                signed = tl.where(raw >= sign_bit, raw - (1 << NUM_BITS), raw)
                signed = tl.where(signed == -sign_bit, 0, signed)
                if NUM_BITS == 2:
                    carrier = (signed * 127).to(tl.int8)
                elif PRECOMPUTED_CARRIER_LUT:
                    magnitude = tl.abs(signed)
                    carrier = tl.load(
                        carrier_lut_ptr
                        + metadata_offsets[None, :] * sign_bit
                        + magnitude,
                        mask=weight_mask,
                        other=0,
                    )
                    carrier = tl.where(signed < 0, -carrier, carrier).to(tl.int8)
                elif PRECOMPUTED_3BIT_LEVELS:
                    magnitude = tl.abs(signed)
                    carrier = tl.where(magnitude == 1, cubic_a, 0.0)
                    carrier = tl.where(magnitude == 2, cubic_b, carrier)
                    carrier = tl.where(magnitude == 3, 127.0, carrier)
                    carrier = tl.where(signed < 0, -carrier, carrier).to(tl.int8)
                else:
                    signed_f32 = signed.to(tl.float32)
                    magnitude_max: tl.constexpr = sign_bit - 1
                    t = tl.abs(signed_f32) / magnitude_max
                    normalized = t * (
                        cubic_a + t * (cubic_b + t * (1.0 - cubic_a - cubic_b))
                    )
                    normalized = tl.where(signed < 0, -normalized, normalized)
                    carrier_f32 = tl.extra.cuda.libdevice.rint(normalized * 127.0)
                    carrier = tl.maximum(tl.minimum(carrier_f32, 127.0), -127.0).to(
                        tl.int8
                    )

            partial = tl.dot(activation, carrier, out_dtype=tl.int32)
            accumulator += (
                partial.to(tl.float32)
                * activation_scale
                * weight_scale
                * (1.0 / 127.0)
            )

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + token_ids,
            mask=token_mask,
            other=0.0,
        )
        accumulator *= routed_weight[:, None]
    output_ptrs = (
        output_ptr + token_ids[:, None] * stride_om + offs_n_raw[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=token_mask[:, None] & (offs_n_raw[None, :] < N),
    )


def _launch_cubic_moe_gemv(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    top_k: int,
    multiply_routed_weight: bool,
    sum_routes: bool,
    route_ctas: int | None = None,
    dense_block_m: int = 16,
) -> None:
    del dense_block_m
    route_ctas = (
        1
        if sum_routes
        else min(
            sorted_token_ids.numel(),
            sorted_token_ids.numel() if route_ctas is None else route_ctas,
        )
    )
    if (
        group_out == 1
        and num_bits == 2
        and group_size in (128, 256, 512)
        and logical_k % group_size == 0
    ):
        packed_words = packed.view(torch.int32)
        grid = lambda meta: (
            triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
            route_ctas,
        )
        _cubic_moe_gemv_2bit_kernel[grid](
            inputs,
            inputs,
            packed_words,
            scale,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            logical_k,
            scale.shape[2],
            inputs.stride(0),
            inputs.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            packed_words.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
            DYNAMIC_A8=False,
        )
        return

    if (
        group_out == 1
        and num_bits == 3
        and group_size in (128, 256)
        and logical_k % group_size == 0
    ):
        grid = lambda meta: (
            triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
            route_ctas,
        )
        _cubic_moe_gemv_3bit_kernel[grid](
            inputs,
            inputs,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            logical_k,
            scale.shape[2],
            inputs.stride(0),
            inputs.stride(1),
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
            DYNAMIC_A8=False,
        )
        return

    block_k = 128 if group_size == 1 else min(group_size, 64 if num_bits >= 7 else 256)
    grid = lambda meta: (
        triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
        route_ctas,
    )
    _cubic_moe_gemv_kernel[grid](
        inputs,
        packed,
        scale,
        a,
        b,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        packed.shape[1],
        logical_k,
        packed.shape[2],
        scale.shape[2],
        inputs.stride(0),
        inputs.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        GROUP_OUT=group_out,
        BLOCK_K=block_k,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        SUM_ROUTES=sum_routes,
    )


def _launch_cubic_moe_situ_cubic8_2bit(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    group_size: int,
    output_group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    beta: float,
    linear_beta: float | None,
    route_ctas: int | None = None,
    words_per_tile: int = 8,
    use_cuda: bool | None = None,
    threads_per_output: int | None = None,
) -> CubicA8Code:
    """Fuse W2+SITU with per-sample/group A8 production, without BF16 staging.

    The online contract is the one validated in
    ``cubic_online_groupwise_a8_rmse.md``: absmax/127 scale, rounded INT8
    carrier, and implicit ``a=1,b=0``.  The legacy return wrapper is retained
    temporarily so callers of the experimental helper remain source-compatible.
    """
    if group_size not in (256, 512) or output_group_size not in (
        16,
        32,
        64,
        128,
        256,
        512,
    ):
        raise ValueError(
            "Fused Cubic8 W2 producer supports weight G256/G512 and "
            "activation G16/G32/G64/G128/G256/G512."
        )
    output_n = packed.shape[1] // 2
    if packed.shape[1] % 2 or output_n % output_group_size:
        raise ValueError(
            f"Gated output N={output_n} must be divisible by G{output_group_size}."
        )
    if group_size // 16 % words_per_tile:
        raise ValueError(
            f"G{group_size} is not divisible by {words_per_tile} packed words."
        )
    kernel_inputs, input_scale = per_token_quant_int8(inputs.contiguous())
    input_words = kernel_inputs.view(torch.int32)
    packed_words = packed.view(torch.int32)
    rows = topk_weights.numel()
    groups = output_n // output_group_size
    codes = torch.empty(rows, output_n, device=inputs.device, dtype=torch.int8)
    output_scale = torch.empty(rows, groups, device=inputs.device, dtype=torch.float32)
    output_a = torch.empty(rows, groups, device=inputs.device, dtype=torch.float16)
    output_b = torch.empty_like(output_a)
    route_ctas = min(
        sorted_token_ids.numel(),
        128 if route_ctas is None else route_ctas,
    )
    if use_cuda is None:
        use_cuda = current_platform.is_cuda() and hasattr(
            torch.ops._C, "cubic_w2_situ_cubic8_producer"
        )
    if use_cuda:
        if threads_per_output is None:
            threads_per_output = 4 if output_group_size == 256 else 8
        torch.ops._C.cubic_w2_situ_cubic8_producer(
            kernel_inputs,
            input_scale,
            packed,
            scale,
            codes,
            output_scale,
            output_a,
            output_b,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            group_size,
            output_group_size,
            top_k,
            multiply_routed_weight,
            route_ctas,
            beta,
            linear_beta or 1.0,
            linear_beta is not None,
            threads_per_output,
        )
        return CubicA8Code(codes, output_scale, output_a, output_b, output_group_size)
    # Triton reference uses the same production W2+SITU DP4A body and changes
    # only its epilogue to emit one linear A8 group per program.  Keeping this
    # path makes the CUDA producer choice calibratable rather than assumed.
    grid = (groups, route_ctas)
    _cubic_moe_situ_gemv_2bit_a8_dp4a_kernel.fn[grid](
        input_words,
        input_scale,
        packed_words,
        scale,
        codes,
        output_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output_n,
        scale.shape[2],
        input_words.stride(0),
        input_words.stride(1),
        packed_words.stride(0),
        packed_words.stride(1),
        packed_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        codes.stride(0),
        codes.stride(1),
        output_scale.stride(0),
        output_scale.stride(1),
        GROUP_SIZE=group_size,
        ACTIVATION_GROUP_SIZE=output_group_size,
        BLOCK_N=output_group_size,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        BETA=beta,
        LINEAR_BETA=linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        OUTPUT_GROUPWISE_A8=True,
        OUTPUT_BF16=inputs.dtype == torch.bfloat16,
        num_warps=8,
        num_stages=1,
    )
    return CubicA8Code(codes, output_scale, output_a, output_b, output_group_size)


def _launch_cubic_moe_situ_gemv_2bit(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    beta: float,
    linear_beta: float | None,
    dynamic_a8: bool = False,
    route_ctas: int | None = None,
    grouped_routes: int = 1,
) -> None:
    if dynamic_a8:
        kernel_inputs, input_scale = per_token_quant_int8(inputs.contiguous())
    else:
        kernel_inputs, input_scale = inputs, inputs
    packed_words = packed.view(torch.int32)
    # Route kernels stride over the valid route count.  Capping the resident
    # route dimension avoids launching one mostly-idle CTA for every global
    # top-k slot under expert parallelism, while retaining enough parallelism
    # for large prefills.
    route_ctas = min(
        sorted_token_ids.numel(),
        128 if route_ctas is None else route_ctas,
    )
    if dynamic_a8:
        input_words = kernel_inputs.view(torch.int32)
        if grouped_routes in (2, 4):
            if grouped_routes == 2:
                # Calibrated during vLLM kernel warmup using controlled route
                # densities, before CUDA graph capture.  This prevents dummy
                # graph routes from training the production tactic.
                block_n, num_warps = _cubic_w2_a8_situ_tactic(
                    output.shape[-1],
                    inputs.shape[-1],
                    group_size,
                    packed.shape[0],
                    route_ctas,
                )
                grid = (
                    triton.cdiv(output.shape[-1], block_n),
                    route_ctas,
                )
                grouped_kernel = _cubic_moe_pair_situ_gemv_2bit_a8_dp4a_kernel
                launch_config = {
                    "BLOCK_N": block_n,
                    "num_warps": num_warps,
                    "num_stages": 1,
                }
            else:
                block_n = 16
                grid = (
                    triton.cdiv(output.shape[-1], block_n),
                    route_ctas,
                )
                grouped_kernel = _cubic_moe_grouped2_situ_gemv_2bit_a8_dp4a_kernel
                launch_config = {
                    "BLOCK_N": block_n,
                    "ROUTES_PER_BLOCK": grouped_routes,
                    "num_warps": 4,
                    "num_stages": 1,
                }
            grouped_kernel[grid](
                input_words,
                input_scale,
                packed_words,
                scale,
                output,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                output.shape[-1],
                scale.shape[2],
                topk_weights.numel(),
                input_words.stride(0),
                input_words.stride(1),
                packed_words.stride(0),
                packed_words.stride(1),
                packed_words.stride(2),
                scale.stride(0),
                scale.stride(1),
                scale.stride(2),
                output.stride(-2),
                output.stride(-1),
                output.stride(-2),
                output.stride(-1),
                GROUP_SIZE=group_size,
                ACTIVATION_GROUP_SIZE=16,
                ROUTE_CTAS=route_ctas,
                MUL_ROUTED_WEIGHT=multiply_routed_weight,
                TOP_K=top_k,
                BETA=beta,
                LINEAR_BETA=linear_beta or 1.0,
                HAS_LINEAR_BETA=linear_beta is not None,
                OUTPUT_GROUPWISE_A8=False,
                OUTPUT_BF16=inputs.dtype == torch.bfloat16,
                **launch_config,
            )
            return
        a8_grid = lambda meta: (
            triton.cdiv(output.shape[-1], meta["BLOCK_N"]),
            route_ctas,
        )
        _cubic_moe_situ_gemv_2bit_a8_dp4a_kernel[a8_grid](
            input_words,
            input_scale,
            packed_words,
            scale,
            output,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            output.shape[-1],
            scale.shape[2],
            input_words.stride(0),
            input_words.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            packed_words.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ACTIVATION_GROUP_SIZE=16,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            BETA=beta,
            LINEAR_BETA=linear_beta or 1.0,
            HAS_LINEAR_BETA=linear_beta is not None,
            OUTPUT_GROUPWISE_A8=False,
            OUTPUT_BF16=inputs.dtype == torch.bfloat16,
        )
        return
    a16_grid = lambda meta: (
        triton.cdiv(output.shape[-1], meta["BLOCK_N"]),
        route_ctas,
    )
    _cubic_moe_situ_gemv_2bit_kernel[a16_grid](
        kernel_inputs,
        input_scale,
        packed_words,
        scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output.shape[-1],
        logical_k,
        scale.shape[2],
        kernel_inputs.stride(0),
        kernel_inputs.stride(1),
        packed_words.stride(0),
        packed_words.stride(1),
        packed_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        GROUP_SIZE=group_size,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        BETA=beta,
        LINEAR_BETA=linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        APPLY_SITU=True,
        DYNAMIC_A8=dynamic_a8,
    )


def _launch_cubic_moe_gate_up_gemv_2bit(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
) -> None:
    packed_words = packed.view(torch.int32)
    route_ctas = sorted_token_ids.numel()
    intermediate_size = output.shape[-1] // 2
    grid = lambda meta: (
        triton.cdiv(intermediate_size, meta["BLOCK_N"]),
        route_ctas,
    )
    _cubic_moe_situ_gemv_2bit_kernel[grid](
        inputs,
        inputs,
        packed_words,
        scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        intermediate_size,
        logical_k,
        scale.shape[2],
        inputs.stride(0),
        inputs.stride(1),
        packed_words.stride(0),
        packed_words.stride(1),
        packed_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        GROUP_SIZE=group_size,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        BETA=1.0,
        LINEAR_BETA=1.0,
        HAS_LINEAR_BETA=False,
        APPLY_SITU=False,
        DYNAMIC_A8=False,
    )


def _apply_cubic_compact_situ(
    output: torch.Tensor,
    input: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> None:
    intermediate_size = output.shape[-1]
    block_size = 256
    grid = (
        triton.cdiv(intermediate_size, block_size),
        sorted_token_ids.numel(),
    )
    _cubic_compact_situ_kernel[grid](
        input,
        output,
        sorted_token_ids,
        num_tokens_post_padded,
        intermediate_size,
        input.stride(-2),
        input.stride(-1),
        output.stride(-2),
        output.stride(-1),
        beta,
        linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )


def _launch_cubic_moe_situ_gemv_3bit(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    beta: float,
    linear_beta: float | None,
) -> None:
    route_ctas = sorted_token_ids.numel()
    grid = lambda meta: (
        triton.cdiv(output.shape[-1], meta["BLOCK_N"]),
        route_ctas,
    )
    _cubic_moe_situ_gemv_3bit_kernel[grid](
        inputs,
        packed,
        scale,
        a,
        b,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output.shape[-1],
        logical_k,
        scale.shape[2],
        inputs.stride(0),
        inputs.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        GROUP_SIZE=group_size,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        BETA=beta,
        LINEAR_BETA=linear_beta or 1.0,
        HAS_LINEAR_BETA=linear_beta is not None,
    )


def _launch_cubic_moe_gemm(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    num_bits: int,
    group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    sum_routes: bool,
    dense_block_m: int = 16,
    route_ctas: int | None = None,
    group_out: int = 1,
) -> None:
    del route_ctas
    if sum_routes:
        raise ValueError("Cubic MoE GEMM does not support route-sum fusion.")
    if dense_block_m not in (1, 16, 32):
        raise ValueError(f"Unsupported Cubic A16 BLOCK_M: {dense_block_m}.")
    block_m, block_k, group_m = dense_block_m, 32, 8
    block_k = 64 if group_size % 64 == 0 else 32
    use_group_lut = group_size >= block_k and group_size % block_k == 0
    n = packed.shape[1]
    em = sorted_token_ids.shape[0]
    grid = lambda meta: (triton.cdiv(em, block_m) * triton.cdiv(n, meta["BLOCK_N"]),)
    _cubic_moe_kernel[grid](
        inputs,
        packed,
        scale,
        a,
        b,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        logical_k,
        packed.shape[2],
        scale.shape[2],
        em,
        topk_weights.numel(),
        inputs.stride(0),
        inputs.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(2),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        GROUP_OUT=group_out,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        USE_GROUP_LUT=use_group_lut and num_bits <= 4,
    )


def _launch_cubic_moe_dynamic_a8(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    num_bits: int,
    group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    use_gemv: bool,
    sum_routes: bool,
    quantized_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    route_ctas: int | None = None,
    grouped_routes: int = 1,
    dense_block_m: int = 16,
    carrier_lut: torch.Tensor | None = None,
    group_out: int = 1,
) -> None:
    if not use_gemv and group_size == 1:
        _launch_cubic_moe_gemm(
            inputs,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            logical_k=logical_k,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=top_k,
            multiply_routed_weight=multiply_routed_weight,
            sum_routes=sum_routes,
            dense_block_m=dense_block_m,
        )
        return
    if quantized_inputs is None:
        inputs_q, input_scale = per_token_quant_int8(inputs.contiguous())
    else:
        inputs_q, input_scale = quantized_inputs
    precomputed_3bit_levels = (
        num_bits == 3 and a.dtype == torch.int8 and b.dtype == torch.int8
    )
    if use_gemv:
        route_ctas = (
            1
            if sum_routes
            else min(
                sorted_token_ids.numel(),
                128 if route_ctas is None else route_ctas,
            )
        )
        if group_out == 1 and num_bits == 1 and logical_k % group_size == 0:
            packed_words = packed.view(torch.int32)
            input_words = inputs_q.view(torch.int32)
            grid = lambda meta: (
                triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
                route_ctas,
            )
            _cubic_moe_gemv_1bit_a8_dp4a_kernel[grid](
                input_words,
                input_scale,
                packed_words,
                scale,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                packed.shape[1],
                scale.shape[2],
                input_words.stride(0),
                input_words.stride(1),
                input_scale.stride(0),
                input_scale.stride(1),
                packed_words.stride(0),
                packed_words.stride(1),
                packed_words.stride(2),
                scale.stride(0),
                scale.stride(1),
                scale.stride(2),
                output.stride(-2),
                output.stride(-1),
                GROUP_SIZE=group_size,
                ROUTE_CTAS=route_ctas,
                MUL_ROUTED_WEIGHT=multiply_routed_weight,
                TOP_K=top_k,
                SUM_ROUTES=sum_routes,
                GROUPWISE_SCALE=False,
            )
            return
        if (
            group_out == 1
            and
            num_bits == 2
            and group_size == 512
            and logical_k % group_size == 0
            and grouped_routes == 1
            and not sum_routes
            and _cubic_a8_moe_backend(
                num_bits=num_bits,
                n=packed.shape[1],
                k=logical_k,
                group_size=group_size,
                group_out=group_out,
                local_experts=packed.shape[0],
                grouped_routes=grouped_routes,
                route_ctas=route_ctas,
            )
            == "cuda"
        ):
            torch.ops._C.cubic_w2_a8_gemv(
                inputs_q,
                input_scale,
                packed,
                scale,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                group_size,
                top_k,
                multiply_routed_weight,
                route_ctas,
            )
            return
        if (
            group_out == 1
            and
            num_bits == 2
            and group_size in (128, 256, 512)
            and logical_k % group_size == 0
        ):
            packed_words = packed.view(torch.int32)
            input_words = inputs_q.view(torch.int32)
            if grouped_routes in (2, 4):
                if grouped_routes == 2:
                    grid = lambda meta: (
                        triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
                        route_ctas,
                    )
                    _cubic_moe_pair_gemv_2bit_a8_dp4a_kernel[grid](
                        input_words,
                        input_scale,
                        packed_words,
                        scale,
                        output,
                        topk_weights,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        packed.shape[1],
                        scale.shape[2],
                        topk_weights.numel(),
                        input_words.stride(0),
                        input_words.stride(1),
                        input_scale.stride(0),
                        input_scale.stride(1),
                        packed_words.stride(0),
                        packed_words.stride(1),
                        packed_words.stride(2),
                        scale.stride(0),
                        scale.stride(1),
                        scale.stride(2),
                        output.stride(-2),
                        output.stride(-1),
                        GROUP_SIZE=group_size,
                        ACTIVATION_GROUP_SIZE=group_size,
                        ROUTE_CTAS=route_ctas,
                        MUL_ROUTED_WEIGHT=multiply_routed_weight,
                        TOP_K=top_k,
                        GROUPWISE_SCALE=False,
                    )
                    return
                block_n = 16
                grouped_grid = (
                    triton.cdiv(packed.shape[1], block_n),
                    route_ctas,
                )
                _cubic_moe_grouped2_gemv_2bit_a8_dp4a_kernel[grouped_grid](
                    input_words,
                    input_scale,
                    packed_words,
                    scale,
                    output,
                    topk_weights,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    packed.shape[1],
                    scale.shape[2],
                    topk_weights.numel(),
                    input_words.stride(0),
                    input_words.stride(1),
                    packed_words.stride(0),
                    packed_words.stride(1),
                    packed_words.stride(2),
                    scale.stride(0),
                    scale.stride(1),
                    scale.stride(2),
                    output.stride(-2),
                    output.stride(-1),
                    GROUP_SIZE=group_size,
                    BLOCK_N=block_n,
                    ROUTE_CTAS=route_ctas,
                    MUL_ROUTED_WEIGHT=multiply_routed_weight,
                    TOP_K=top_k,
                    ROUTES_PER_BLOCK=grouped_routes,
                    num_warps=4,
                    num_stages=1,
                )
                return
            grid = lambda meta: (
                triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
                route_ctas,
            )
            _cubic_moe_gemv_2bit_a8_dp4a_kernel[grid](
                input_words,
                input_scale,
                packed_words,
                scale,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                packed.shape[1],
                scale.shape[2],
                input_words.stride(0),
                input_words.stride(1),
                packed_words.stride(0),
                packed_words.stride(1),
                packed_words.stride(2),
                scale.stride(0),
                scale.stride(1),
                scale.stride(2),
                output.stride(-2),
                output.stride(-1),
                GROUP_SIZE=group_size,
                ROUTE_CTAS=route_ctas,
                MUL_ROUTED_WEIGHT=multiply_routed_weight,
                TOP_K=top_k,
                SUM_ROUTES=sum_routes,
            )
            return
        if (
            group_out == 1
            and
            num_bits == 3
            and precomputed_3bit_levels
            and group_size in (128, 256, 512)
            and logical_k % group_size == 0
            and not sum_routes
            and _cubic_a8_moe_backend(
                num_bits=num_bits,
                n=packed.shape[1],
                k=logical_k,
                group_size=group_size,
                group_out=group_out,
                local_experts=packed.shape[0],
                grouped_routes=grouped_routes,
                route_ctas=route_ctas,
            )
            == "cuda"
        ):
            torch.ops._C.cubic_w3_a8_gemv(
                inputs_q,
                input_scale,
                packed,
                scale,
                a,
                b,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                group_size,
                top_k,
                multiply_routed_weight,
                route_ctas,
                topk_weights.numel(),
                grouped_routes,
            )
            return
        # Building the carrier table once per (output, group) removes repeated
        # cubic Horner evaluations.  Require at least two weight-code uses per
        # positive level; below that crossover (W8/G128), the direct Triton
        # kernel avoids shared-memory traffic and is faster.
        carrier_level_reuse = group_size / (1 << (num_bits - 1))
        if (
            4 <= num_bits <= 8
            and group_size in (128, 256, 512)
            and carrier_level_reuse >= 2
            and logical_k % group_size == 0
            and a.dtype == torch.float16
            and b.dtype == torch.float16
            and not sum_routes
            and _cubic_a8_moe_backend(
                num_bits=num_bits,
                n=packed.shape[1],
                k=logical_k,
                group_size=group_size,
                group_out=group_out,
                local_experts=packed.shape[0],
                grouped_routes=grouped_routes,
                route_ctas=route_ctas,
            )
            == "cuda"
        ):
            torch.ops._C.cubic_w4_w8_a8_gemv(
                inputs_q,
                input_scale,
                packed,
                scale,
                a,
                b,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                num_bits,
                group_size,
                group_out,
                top_k,
                multiply_routed_weight,
                route_ctas,
                topk_weights.numel(),
                grouped_routes,
            )
            return
        if (
            group_size >= 4
            and 3 <= num_bits <= 8
            and logical_k % group_size == 0
            and logical_k % 4 == 0
        ):
            input_words = inputs_q.view(torch.int32)
            grid = lambda meta: (
                triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
                route_ctas,
            )
            _cubic_moe_dynamic_a8_autotune_gemv_kernel[grid](
                inputs_q,
                input_words,
                input_scale,
                packed,
                scale,
                a,
                b,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                packed.shape[1],
                logical_k,
                packed.shape[2],
                scale.shape[2],
                inputs_q.stride(0),
                inputs_q.stride(1),
                input_words.stride(0),
                input_words.stride(1),
                input_scale.stride(0),
                input_scale.stride(1),
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                scale.stride(0),
                scale.stride(1),
                scale.stride(2),
                output.stride(-2),
                output.stride(-1),
                NUM_BITS=num_bits,
                GROUP_SIZE=group_size,
                GROUP_OUT=group_out,
                ROUTE_CTAS=route_ctas,
                MUL_ROUTED_WEIGHT=multiply_routed_weight,
                TOP_K=top_k,
                SUM_ROUTES=sum_routes,
                PRECOMPUTED_3BIT_LEVELS=precomputed_3bit_levels,
                GROUPWISE_SCALE=False,
            )
            return
        if (
            group_out == 1
            and num_bits == 3
            and group_size in (128, 256)
            and logical_k % group_size == 0
        ):
            grid = lambda meta: (
                triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
                route_ctas,
            )
            _cubic_moe_gemv_3bit_kernel[grid](
                inputs_q,
                input_scale,
                packed,
                scale,
                a,
                b,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                packed.shape[1],
                logical_k,
                scale.shape[2],
                inputs_q.stride(0),
                inputs_q.stride(1),
                packed.stride(0),
                packed.stride(1),
                packed.stride(2),
                scale.stride(0),
                scale.stride(1),
                scale.stride(2),
                output.stride(-2),
                output.stride(-1),
                GROUP_SIZE=group_size,
                ROUTE_CTAS=route_ctas,
                MUL_ROUTED_WEIGHT=multiply_routed_weight,
                TOP_K=top_k,
                SUM_ROUTES=sum_routes,
                DYNAMIC_A8=True,
            )
            return
        block_k = min(group_size, 128)
        grid = lambda meta: (
            triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
            route_ctas,
        )
        _cubic_moe_dynamic_a8_gemv_kernel[grid](
            inputs_q,
            input_scale,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            logical_k,
            packed.shape[2],
            scale.shape[2],
            inputs_q.stride(0),
            inputs_q.stride(1),
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            GROUP_OUT=group_out,
            BLOCK_K=block_k,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
        )
        return
    if sum_routes:
        raise ValueError("Cubic Dynamic A8 GEMM does not support route-sum fusion.")
    if dense_block_m not in (16, 32):
        raise ValueError(f"Unsupported Cubic Dynamic A8 BLOCK_M: {dense_block_m}.")
    block_m, block_k, group_m = dense_block_m, 32, 8
    n = packed.shape[1]
    em = sorted_token_ids.shape[0]
    dense_grid = lambda meta: (
        triton.cdiv(em, block_m) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _cubic_moe_dynamic_a8_kernel[dense_grid](
        inputs_q,
        input_scale,
        packed,
        scale,
        a,
        b,
        a if carrier_lut is None else carrier_lut,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        logical_k,
        packed.shape[2],
        scale.shape[2],
        em,
        topk_weights.numel(),
        inputs_q.stride(0),
        inputs_q.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        GROUP_OUT=group_out,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        PRECOMPUTED_3BIT_LEVELS=precomputed_3bit_levels,
        PRECOMPUTED_CARRIER_LUT=carrier_lut is not None,
    )


def _launch_cubic_moe_cubic8_w2(
    carrier: CubicA8Code,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    top_k: int,
    multiply_routed_weight: bool,
    sum_routes: bool = False,
    route_ctas: int | None = None,
    block_n: int = 16,
) -> None:
    """Launch the first exact Cubic×Cubic consumer (offline W2)."""
    if carrier.group_size not in (256, 512):
        raise ValueError("Cubic8×W2 currently supports G256/G512.")
    if carrier.codes.dtype != torch.int8:
        raise ValueError("Cubic8 activation codes must use signed INT8 storage.")
    if carrier.scales.dtype != torch.float32:
        raise ValueError("Cubic8 activation scales must use FP32.")
    if carrier.a.dtype != torch.float16 or carrier.b.dtype != torch.float16:
        raise ValueError("Cubic8 activation a/b metadata must use FP16.")
    if carrier.codes.shape[-1] != logical_k or logical_k % carrier.group_size:
        raise ValueError(
            f"Cubic8×W2 expected K={logical_k} divisible by G{carrier.group_size}."
        )
    metadata_shape = (carrier.codes.shape[0], logical_k // carrier.group_size)
    if (
        tuple(carrier.scales.shape) != metadata_shape
        or tuple(carrier.a.shape) != metadata_shape
        or tuple(carrier.b.shape) != metadata_shape
    ):
        raise ValueError(f"Cubic8 metadata must have shape {metadata_shape}.")
    route_ctas = (
        1
        if sum_routes
        else min(sorted_token_ids.numel(), 128 if route_ctas is None else route_ctas)
    )
    if block_n not in (8, 16, 32, 64, 128):
        raise ValueError(f"Unsupported Cubic8×W2 BLOCK_N={block_n}.")
    packed_words = packed.view(torch.int32)
    grid = (triton.cdiv(packed.shape[1], block_n), route_ctas)
    _cubic_moe_gemv_2bit_cubic8_kernel[grid](
        carrier.codes,
        carrier.scales,
        carrier.a,
        carrier.b,
        packed_words,
        weight_scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        packed.shape[1],
        logical_k,
        logical_k // carrier.group_size,
        carrier.codes.stride(0),
        carrier.codes.stride(1),
        carrier.scales.stride(0),
        carrier.scales.stride(1),
        packed_words.stride(0),
        packed_words.stride(1),
        packed_words.stride(2),
        weight_scale.stride(0),
        weight_scale.stride(1),
        weight_scale.stride(2),
        output.stride(-2),
        output.stride(-1),
        GROUP_SIZE=carrier.group_size,
        ROUTE_CTAS=route_ctas,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        TOP_K=top_k,
        SUM_ROUTES=sum_routes,
        BLOCK_N=block_n,
        num_warps=2 if block_n == 8 else 4 if block_n <= 32 else 8,
        num_stages=1,
    )


def _launch_cubic_moe_cubic8_w2_moment(
    carrier: CubicA8Code,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    weight_group_size: int,
    top_k: int,
    multiply_routed_weight: bool,
    route_ctas: int | None = None,
) -> None:
    """Consume Cubic8 with three DP4A polynomial moments (W2 weights)."""
    if not current_platform.is_cuda() or not hasattr(
        torch.ops._C, "cubic_w2_cubic8_moment_gemv"
    ):
        raise NotImplementedError("Cubic8 moment GEMV requires the CUDA extension.")
    if carrier.group_size not in (16, 32, 64, 128, 256, 512):
        raise ValueError(f"Unsupported activation G{carrier.group_size}.")
    if weight_group_size not in (256, 512):
        raise ValueError(f"Unsupported weight G{weight_group_size}.")
    metadata_shape = (carrier.codes.shape[0], logical_k // carrier.group_size)
    if (
        carrier.codes.dtype != torch.int8
        or carrier.scales.dtype != torch.float32
        or carrier.a.dtype != torch.float16
        or carrier.b.dtype != torch.float16
        or tuple(carrier.scales.shape) != metadata_shape
        or tuple(carrier.a.shape) != metadata_shape
        or tuple(carrier.b.shape) != metadata_shape
    ):
        raise ValueError("Invalid Cubic8 carrier layout for moment GEMV.")
    route_ctas = min(
        sorted_token_ids.numel(),
        128 if route_ctas is None else route_ctas,
    )
    torch.ops._C.cubic_w2_cubic8_moment_gemv(
        carrier.codes,
        carrier.scales,
        carrier.a,
        carrier.b,
        packed,
        weight_scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        weight_group_size,
        carrier.group_size,
        top_k,
        multiply_routed_weight,
        route_ctas,
    )


def _launch_cubic_moe_cubic8_w2_lut(
    carrier: CubicA8Code,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    top_k: int,
    multiply_routed_weight: bool,
    route_ctas: int | None = None,
    threads_per_output: int = 8,
) -> None:
    """Exact W2 consumer using one CTA-shared Cubic decode LUT per group."""
    if not current_platform.is_cuda() or not hasattr(
        torch.ops._C, "cubic_w2_cubic8_lut_gemv"
    ):
        raise NotImplementedError("Cubic8 LUT GEMV requires the CUDA extension.")
    if carrier.group_size not in (128, 256, 512):
        raise ValueError("Cubic8 LUT GEMV requires activation G128/G256/G512.")
    weight_group_size = logical_k // weight_scale.shape[2]
    if weight_group_size not in (256, 512):
        raise ValueError(f"Unsupported W2 weight G{weight_group_size}.")
    metadata_shape = (carrier.codes.shape[0], logical_k // carrier.group_size)
    if (
        tuple(carrier.scales.shape) != metadata_shape
        or tuple(carrier.a.shape) != metadata_shape
        or tuple(carrier.b.shape) != metadata_shape
    ):
        raise ValueError("Invalid Cubic8 carrier metadata shape.")
    route_ctas = min(
        sorted_token_ids.numel(),
        128 if route_ctas is None else route_ctas,
    )
    torch.ops._C.cubic_w2_cubic8_lut_gemv(
        carrier.codes,
        carrier.scales,
        carrier.a,
        carrier.b,
        packed,
        weight_scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        weight_group_size,
        carrier.group_size,
        top_k,
        multiply_routed_weight,
        route_ctas,
        threads_per_output,
    )


def _launch_cubic_moe_groupwise_a8(
    carrier: CubicA8Carrier,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    logical_k: int,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    top_k: int,
    multiply_routed_weight: bool,
    sum_routes: bool,
    route_ctas: int | None = None,
    grouped_routes: int = 1,
) -> None:
    """Consume per-sample/per-K-group A8 without touching legacy kernels."""
    if (
        carrier.group_size < 16
        or carrier.group_size > group_size
        or carrier.group_size % 16
        or group_size % carrier.group_size
    ):
        raise ValueError(
            "Cubic A8 carrier group must be a >=16 multiple of 16 that "
            f"divides weight G={group_size}; got G={carrier.group_size}."
        )
    if carrier.values.dtype != torch.int8 or carrier.scales.dtype != torch.float32:
        raise ValueError("Cubic groupwise A8 requires INT8 values and FP32 scales.")
    if carrier.values.shape[-1] != logical_k:
        raise ValueError(
            f"Cubic groupwise A8 expected K={logical_k}, "
            f"got {carrier.values.shape[-1]}."
        )
    expected_groups = logical_k // group_size
    expected_activation_groups = logical_k // carrier.group_size
    if carrier.scales.shape != (
        carrier.values.shape[0],
        expected_activation_groups,
    ):
        raise ValueError(
            "Cubic groupwise A8 scale shape must be "
            f"({carrier.values.shape[0]}, {expected_activation_groups}), got "
            f"{tuple(carrier.scales.shape)}."
        )
    route_ctas = (
        1
        if sum_routes
        else min(
            sorted_token_ids.numel(),
            128 if route_ctas is None else route_ctas,
        )
    )
    backend = _cubic_online_a8_moe_backend(
        num_bits=num_bits,
        n=packed.shape[1],
        k=logical_k,
        group_size=group_size,
        group_out=group_out,
        local_experts=packed.shape[0],
        grouped_routes=grouped_routes,
        route_ctas=route_ctas,
    )
    cuda_subgroup_w2 = num_bits == 2 and group_size == 512 and carrier.group_size == 256
    if carrier.group_size != group_size and not cuda_subgroup_w2:
        backend = "triton"
    input_words = carrier.values.view(torch.int32)
    grid = lambda meta: (
        triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
        route_ctas,
    )
    if num_bits == 1:
        if carrier.group_size != group_size:
            raise ValueError("W1 groupwise A8 currently requires matching groups.")
        packed_words = packed.view(torch.int32)
        _cubic_moe_gemv_1bit_a8_dp4a_kernel[grid](
            input_words,
            carrier.scales,
            packed_words,
            scale,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            expected_groups,
            input_words.stride(0),
            input_words.stride(1),
            carrier.scales.stride(0),
            carrier.scales.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            packed_words.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
            GROUPWISE_SCALE=True,
        )
        return
    if num_bits == 2 and grouped_routes == 2 and backend == "triton":
        packed_words = packed.view(torch.int32)
        pair_grid = lambda meta: (
            triton.cdiv(packed.shape[1], meta["BLOCK_N"]),
            route_ctas,
        )
        _cubic_moe_pair_gemv_2bit_a8_dp4a_kernel[pair_grid](
            input_words,
            carrier.scales,
            packed_words,
            scale,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            expected_groups,
            topk_weights.numel(),
            input_words.stride(0),
            input_words.stride(1),
            carrier.scales.stride(0),
            carrier.scales.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            packed_words.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ACTIVATION_GROUP_SIZE=carrier.group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            GROUPWISE_SCALE=True,
        )
        return
    if (
        num_bits == 2
        and group_size in (256, 512)
        and current_platform.is_cuda()
        and backend == "cuda"
        and (carrier.group_size == group_size or cuda_subgroup_w2)
        and output.dtype == torch.bfloat16
    ):
        torch.ops._C.cubic_w2_groupwise_a8_gemv(
            carrier.values,
            carrier.scales,
            packed,
            scale,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            group_size,
            top_k,
            multiply_routed_weight,
            route_ctas,
            topk_weights.numel(),
            grouped_routes,
        )
        return
    if num_bits == 2 and group_size in (128, 256, 512):
        packed_words = packed.view(torch.int32)
        _cubic_moe_gemv_2bit_groupwise_a8_dp4a_kernel[grid](
            input_words,
            carrier.scales,
            packed_words,
            scale,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            expected_groups,
            input_words.stride(0),
            input_words.stride(1),
            carrier.scales.stride(0),
            carrier.scales.stride(1),
            packed_words.stride(0),
            packed_words.stride(1),
            packed_words.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ACTIVATION_GROUP_SIZE=carrier.group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
        )
        return
    if (
        num_bits == 3
        and group_size in (128, 256, 512)
        and a.dtype == torch.int8
        and b.dtype == torch.int8
    ):
        if (
            current_platform.is_cuda()
            and backend == "cuda"
            and carrier.group_size == group_size
            and output.dtype == torch.bfloat16
        ):
            torch.ops._C.cubic_w3_groupwise_a8_gemv(
                carrier.values,
                carrier.scales,
                packed,
                scale,
                a,
                b,
                output,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                group_size,
                top_k,
                multiply_routed_weight,
                route_ctas,
                topk_weights.numel(),
                grouped_routes,
            )
            return
        _cubic_moe_gemv_3bit_groupwise_a8_dp4a_kernel[grid](
            input_words,
            carrier.scales,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            logical_k,
            expected_groups,
            input_words.stride(0),
            input_words.stride(1),
            carrier.scales.stride(0),
            carrier.scales.stride(1),
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            GROUP_SIZE=group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
        )
        return
    if (
        4 <= num_bits <= 8
        and group_size in (128, 256, 512)
        and a.dtype == torch.float16
        and b.dtype == torch.float16
        and current_platform.is_cuda()
        and backend == "cuda"
        and carrier.group_size == group_size
        and output.dtype == torch.bfloat16
    ):
        torch.ops._C.cubic_w4_w8_groupwise_a8_gemv(
            carrier.values,
            carrier.scales,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            num_bits,
            group_size,
            1,
            top_k,
            multiply_routed_weight,
            route_ctas,
            topk_weights.numel(),
            grouped_routes,
        )
        return
    if 2 <= num_bits <= 8 and grouped_routes == 1:
        precomputed_3bit_levels = (
            num_bits == 3 and a.dtype == torch.int8 and b.dtype == torch.int8
        )
        _cubic_moe_dynamic_a8_autotune_gemv_kernel[grid](
            carrier.values,
            input_words,
            carrier.scales,
            packed,
            scale,
            a,
            b,
            output,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            packed.shape[1],
            logical_k,
            packed.shape[2],
            expected_groups,
            carrier.values.stride(0),
            carrier.values.stride(1),
            input_words.stride(0),
            input_words.stride(1),
            carrier.scales.stride(0),
            carrier.scales.stride(1),
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            output.stride(-2),
            output.stride(-1),
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            ROUTE_CTAS=route_ctas,
            MUL_ROUTED_WEIGHT=multiply_routed_weight,
            TOP_K=top_k,
            SUM_ROUTES=sum_routes,
            PRECOMPUTED_3BIT_LEVELS=precomputed_3bit_levels,
            GROUPWISE_SCALE=True,
        )
        return
    raise ValueError(
        f"Cubic groupwise A8 consumer is not implemented for W{num_bits}/G{group_size}."
    )


@torch.inference_mode()
def calibrate_cubic_moe_route_ctas(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    *,
    dynamic_a8: bool,
    global_num_experts: int,
    logical_k: int,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    top_k: int,
    multiply_routed_weight: bool,
    grouped_routes: int,
    groupwise_a8: bool = False,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
) -> None:
    """Choose route parallelism for one actual Cubic MoE projection shape."""
    sorted_ids, expert_ids, count = _cubic_align_block_size(
        topk_ids,
        grouped_routes,
        global_num_experts,
        packed.shape[0],
        expert_map,
    )
    key = (
        torch.accelerator.current_device_index(),
        dynamic_a8,
        num_bits,
        packed.shape[1],
        logical_k,
        group_size,
        group_out,
        packed.shape[0],
        grouped_routes,
        topk_ids.shape[0],
        top_k,
    )
    candidates = tuple(
        dict.fromkeys(min(sorted_ids.numel(), value) for value in (8, 16, 32, 64, 128))
    )
    groupwise_carrier = (
        _quantize_cubic_groupwise_a8(inputs.contiguous(), group_size)
        if dynamic_a8 and groupwise_a8
        else None
    )
    quantized_inputs = (
        per_token_quant_int8(inputs.contiguous())
        if dynamic_a8 and not groupwise_a8
        else None
    )
    reference: torch.Tensor | None = None
    scores: list[tuple[float, int]] = []
    for route_ctas in candidates:
        _CUBIC_MOE_ROUTE_CTA_TACTICS[key] = route_ctas
        output_size = packed.shape[1] // 2 if situ_beta is not None else packed.shape[1]
        output = torch.zeros(
            topk_ids.shape[0],
            top_k,
            output_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        def launch(output=output, route_ctas=route_ctas) -> None:
            if situ_beta is not None:
                _launch_cubic_moe_situ_gemv_2bit(
                    inputs,
                    packed,
                    scale,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                    dynamic_a8=dynamic_a8,
                    route_ctas=route_ctas,
                    grouped_routes=grouped_routes,
                )
            elif groupwise_carrier is not None:
                _launch_cubic_moe_groupwise_a8(
                    groupwise_carrier,
                    packed,
                    scale,
                    a,
                    b,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    num_bits=num_bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    sum_routes=False,
                    route_ctas=route_ctas,
                    grouped_routes=grouped_routes,
                )
            elif dynamic_a8:
                _launch_cubic_moe_dynamic_a8(
                    inputs,
                    packed,
                    scale,
                    a,
                    b,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    num_bits=num_bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    use_gemv=True,
                    sum_routes=False,
                    quantized_inputs=quantized_inputs,
                    route_ctas=route_ctas,
                    grouped_routes=grouped_routes,
                )
            else:
                _launch_cubic_moe_gemv(
                    inputs,
                    packed,
                    scale,
                    a,
                    b,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    num_bits=num_bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    sum_routes=False,
                    route_ctas=route_ctas,
                )

        try:
            launch()
            torch.accelerator.synchronize()
            if reference is None:
                reference = output.clone()
            else:
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
            score = triton.testing.do_bench(launch, warmup=10, rep=30)
            scores.append((score, route_ctas))
        except (RuntimeError, AssertionError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic %s W%d route_ctas=%d for N=%d K=%d M=%d: %s",
                "Online A8" if groupwise_a8 else "A8" if dynamic_a8 else "A16",
                num_bits,
                route_ctas,
                packed.shape[1],
                logical_k,
                topk_ids.shape[0],
                error,
            )
    if not scores:
        _CUBIC_MOE_ROUTE_CTA_TACTICS.pop(key, None)
        return
    measured_best = min(score for score, _ in scores)
    near_best = [item for item in scores if item[0] <= measured_best * 1.01]
    best_score, best_route_ctas = min(near_best, key=lambda item: item[1])
    _CUBIC_MOE_ROUTE_CTA_TACTICS[key] = best_route_ctas
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Cubic %s route CTA: W%d N=%d K=%d G=%d M=%d top_k=%d grouped=%d "
        "ctas=%d (%.4f ms)",
        "Online A8" if groupwise_a8 else "A8" if dynamic_a8 else "A16",
        num_bits,
        packed.shape[1],
        logical_k,
        group_size,
        topk_ids.shape[0],
        top_k,
        grouped_routes,
        best_route_ctas,
        best_score,
    )


@torch.inference_mode()
def calibrate_cubic_a8_moe_backend(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    *,
    global_num_experts: int,
    logical_k: int,
    num_bits: int,
    group_size: int,
    group_out: int = 1,
    top_k: int,
    multiply_routed_weight: bool,
    grouped_routes: int,
    groupwise_a8: bool = False,
) -> None:
    """Choose the CUDA or Triton Cubic A8 route-GEMV implementation."""
    sorted_ids, expert_ids, count = _cubic_align_block_size(
        topk_ids,
        grouped_routes,
        global_num_experts,
        packed.shape[0],
        expert_map,
    )
    expected_local_routes = (
        topk_ids.shape[0] * top_k * packed.shape[0] + global_num_experts - 1
    ) // global_num_experts
    expected_blocks = (expected_local_routes + grouped_routes - 1) // grouped_routes
    route_ctas = min(max(1 << (max(expected_blocks, 1) - 1).bit_length(), 16), 128)
    route_ctas = _cubic_moe_route_ctas(
        dynamic_a8=True,
        num_bits=num_bits,
        n=packed.shape[1],
        k=logical_k,
        group_size=group_size,
        group_out=group_out,
        local_experts=packed.shape[0],
        grouped_routes=grouped_routes,
        input_rows=topk_ids.shape[0],
        top_k=top_k,
        fallback=route_ctas,
    )
    route_ctas = min(sorted_ids.numel(), route_ctas)
    key = (
        torch.accelerator.current_device_index(),
        num_bits,
        packed.shape[1],
        logical_k,
        group_size,
        group_out,
        packed.shape[0],
        grouped_routes,
        route_ctas,
    )
    registry = (
        _CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS
        if groupwise_a8
        else _CUBIC_A8_MOE_BACKEND_TACTICS
    )
    groupwise_carrier = (
        _quantize_cubic_groupwise_a8(inputs.contiguous(), group_size)
        if groupwise_a8
        else None
    )
    quantized_inputs = (
        None if groupwise_a8 else per_token_quant_int8(inputs.contiguous())
    )
    reference: torch.Tensor | None = None
    scores: list[tuple[float, str]] = []

    # The generic Triton consumer is singleton-route only.  W2 has a dedicated
    # paired Triton kernel; W3-W8 paired routes are CUDA-only.  Do not treat an
    # intentionally unavailable candidate as a calibration failure.
    backends = (
        ("cuda",)
        if groupwise_a8 and grouped_routes != 1 and num_bits != 2
        else ("cuda", "triton")
    )
    for backend in backends:
        registry[key] = backend
        output = torch.zeros(
            topk_ids.shape[0],
            top_k,
            packed.shape[1],
            device=inputs.device,
            dtype=inputs.dtype,
        )

        def launch(output=output) -> None:
            if groupwise_carrier is not None:
                _launch_cubic_moe_groupwise_a8(
                    groupwise_carrier,
                    packed,
                    scale,
                    a,
                    b,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    num_bits=num_bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    sum_routes=False,
                    route_ctas=route_ctas,
                    grouped_routes=grouped_routes,
                )
            else:
                _launch_cubic_moe_dynamic_a8(
                    inputs,
                    packed,
                    scale,
                    a,
                    b,
                    output,
                    topk_weights,
                    sorted_ids,
                    expert_ids,
                    count,
                    logical_k=logical_k,
                    num_bits=num_bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=multiply_routed_weight,
                    use_gemv=True,
                    sum_routes=False,
                    quantized_inputs=quantized_inputs,
                    route_ctas=route_ctas,
                    grouped_routes=grouped_routes,
                )

        try:
            launch()
            torch.accelerator.synchronize()
            if reference is None:
                reference = output.clone()
            else:
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
            score = triton.testing.do_bench(launch, warmup=20, rep=60)
            scores.append((score, backend))
        except (RuntimeError, AssertionError, ValueError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic %s %s backend for W%d N=%d K=%d: %s",
                "Online A8" if groupwise_a8 else "A8",
                backend,
                num_bits,
                packed.shape[1],
                logical_k,
                error,
            )

    if not scores:
        registry.pop(key, None)
        return
    measured_best = min(score for score, _ in scores)
    near_best = [item for item in scores if item[0] <= measured_best * 1.01]
    best_score, best_backend = min(
        near_best, key=lambda item: 0 if item[1] == "cuda" else 1
    )
    registry[key] = best_backend
    from vllm.logger import init_logger

    candidate_scores = ", ".join(
        f"{backend}={score:.4f}ms" for score, backend in scores
    )
    init_logger(__name__).info(
        "Cubic %s backend: W%d N=%d K=%d G=%d grouped=%d route_ctas=%d "
        "candidates=[%s], selected=%s (%.4f ms)",
        "Online A8" if groupwise_a8 else "A8",
        num_bits,
        packed.shape[1],
        logical_k,
        group_size,
        grouped_routes,
        route_ctas,
        candidate_scores,
        best_backend,
        best_score,
    )


@torch.inference_mode()
def calibrate_cubic_a8_moe_layer_backends(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> None:
    """Select A8 projection backends using complete MoE layer latency."""
    device = torch.accelerator.current_device_index()

    def matching_keys(n: int, k: int) -> list[tuple[int, ...]]:
        return [
            key
            for key in _CUBIC_A8_MOE_BACKEND_TACTICS
            if key[0] == device
            and key[1] == num_bits
            and key[2] == n
            and key[3] == k
            and key[4] == group_size
            and key[5] == w1.shape[0]
        ]

    gate_keys = matching_keys(w1.shape[1], hidden_size)
    down_keys = matching_keys(w2.shape[1], intermediate_size)
    if not gate_keys or not down_keys:
        return

    original = {
        key: _CUBIC_A8_MOE_BACKEND_TACTICS[key] for key in gate_keys + down_keys
    }
    args = (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
    )
    kwargs: dict[str, Any] = dict(
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        num_bits=num_bits,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )
    candidates = (
        ("cuda", "cuda"),
        ("cuda", "triton"),
        ("triton", "cuda"),
        ("triton", "triton"),
    )
    route_scenarios = [("observed", topk_ids)]
    if expert_map is not None:
        local_global_ids = torch.nonzero(expert_map >= 0).flatten()
        remote_global_ids = torch.nonzero(expert_map < 0).flatten()
        if local_global_ids.numel() and remote_global_ids.numel():
            top_k = topk_ids.shape[1]
            local_fraction = local_global_ids.numel() / global_num_experts
            mean = top_k * local_fraction
            stddev = math.sqrt(top_k * local_fraction * (1.0 - local_fraction))
            high_local_routes = min(top_k, max(1, math.ceil(mean + 3.0 * stddev)))
            positions = torch.arange(
                hidden_states.shape[0], device=topk_ids.device, dtype=torch.int64
            )[:, None]
            local_offsets = torch.arange(
                high_local_routes, device=topk_ids.device, dtype=torch.int64
            )[None, :]
            remote_offsets = torch.arange(
                top_k - high_local_routes,
                device=topk_ids.device,
                dtype=torch.int64,
            )[None, :]
            dense_local_ids = local_global_ids[
                (positions * high_local_routes + local_offsets)
                % local_global_ids.numel()
            ]
            dense_remote_ids = remote_global_ids[
                (positions * (top_k - high_local_routes) + remote_offsets)
                % remote_global_ids.numel()
            ]
            high_ids = torch.cat((dense_local_ids, dense_remote_ids), dim=1).to(
                topk_ids.dtype
            )
            route_scenarios.append((f"high_local={high_local_routes}", high_ids))

    references: dict[str, torch.Tensor] = {}
    scores: list[tuple[float, tuple[str, str]]] = []
    for gate_backend, down_backend in candidates:
        for key in gate_keys:
            _CUBIC_A8_MOE_BACKEND_TACTICS[key] = gate_backend
        for key in down_keys:
            _CUBIC_A8_MOE_BACKEND_TACTICS[key] = down_backend

        try:
            scenario_scores = []
            for scenario, scenario_topk_ids in route_scenarios:

                def launch(scenario_topk_ids=scenario_topk_ids) -> torch.Tensor:
                    return cubic_fused_moe_dynamic_a8(
                        *args, scenario_topk_ids, **kwargs
                    )

                output = launch()
                torch.accelerator.synchronize()
                if scenario not in references:
                    references[scenario] = output
                else:
                    torch.testing.assert_close(
                        output,
                        references[scenario],
                        rtol=0.02,
                        atol=0.02,
                    )
                scenario_scores.append(
                    triton.testing.do_bench(launch, warmup=10, rep=30)
                )
            scores.append((max(scenario_scores), (gate_backend, down_backend)))
        except (RuntimeError, AssertionError, ValueError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic A8 layer backends W13=%s W2=%s for W%d H=%d I=%d: %s",
                gate_backend,
                down_backend,
                num_bits,
                hidden_size,
                intermediate_size,
                error,
            )

    if not scores:
        _CUBIC_A8_MOE_BACKEND_TACTICS.update(original)
        return
    measured_best = min(score for score, _ in scores)
    # Avoid switching projection families for differences small enough to be
    # overturned by graph replay and distributed launch overhead.
    near_best = [item for item in scores if item[0] <= measured_best * 1.03]
    best_score, (best_gate, best_down) = min(
        near_best,
        key=lambda item: candidates.index(item[1]),
    )
    for key in gate_keys:
        _CUBIC_A8_MOE_BACKEND_TACTICS[key] = best_gate
    for key in down_keys:
        _CUBIC_A8_MOE_BACKEND_TACTICS[key] = best_down
    from vllm.logger import init_logger

    candidate_scores = ", ".join(
        f"{gate}/{down}={score:.4f}ms" for score, (gate, down) in scores
    )
    init_logger(__name__).info(
        "Cubic A8 layer backends: W%d H=%d I=%d G=%d M=%d "
        "routes=%s candidates=[%s], selected=%s/%s (%.4f ms)",
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        hidden_states.shape[0],
        [scenario for scenario, _ in route_scenarios],
        candidate_scores,
        best_gate,
        best_down,
        best_score,
    )


@torch.inference_mode()
def calibrate_cubic_a8_moe_grouping(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    group_out: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> None:
    """Measure route grouping for a complete route-kernel MoE layer."""
    device = torch.accelerator.current_device_index()
    key = (
        device,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        w1.shape[0],
        hidden_states.shape[0],
    )
    execution_key = (
        device,
        True,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        w1.shape[0],
        hidden_states.shape[0],
    )
    had_execution_tactic = execution_key in _CUBIC_MOE_EXECUTION_TACTICS
    previous_execution_tactic = _CUBIC_MOE_EXECUTION_TACTICS.get(execution_key)
    _CUBIC_MOE_EXECUTION_TACTICS[execution_key] = True
    reference: torch.Tensor | None = None
    scores: list[tuple[float, int]] = []
    args = (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
    )
    kwargs: dict[str, Any] = dict(
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )
    candidates = (
        (1,)
        if group_size < 256
        else (1, 2, 4, 8)
        if num_bits >= 4
        else (1, 2)
    )
    for grouped_routes in candidates:
        _CUBIC_A8_MOE_GROUPING_TACTICS[key] = grouped_routes

        def launch() -> torch.Tensor:
            return cubic_fused_moe_dynamic_a8(*args, **kwargs)

        try:
            output = launch()
            torch.accelerator.synchronize()
            if reference is None:
                reference = output
            else:
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
            score = triton.testing.do_bench(launch, warmup=10, rep=30)
            scores.append((score, grouped_routes))
        except (RuntimeError, AssertionError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic A8 route grouping=%d for W%d H=%d I=%d: %s",
                grouped_routes,
                num_bits,
                hidden_size,
                intermediate_size,
                error,
            )
    if not scores:
        _CUBIC_A8_MOE_GROUPING_TACTICS.pop(key, None)
        if had_execution_tactic:
            _CUBIC_MOE_EXECUTION_TACTICS[execution_key] = bool(
                previous_execution_tactic
            )
        else:
            _CUBIC_MOE_EXECUTION_TACTICS.pop(execution_key, None)
        return
    measured_best = min(score for score, _ in scores)
    near_best = [item for item in scores if item[0] <= measured_best * 1.01]
    best_score, best_grouping = min(near_best, key=lambda item: item[1])
    _CUBIC_A8_MOE_GROUPING_TACTICS[key] = best_grouping
    if had_execution_tactic:
        _CUBIC_MOE_EXECUTION_TACTICS[execution_key] = bool(previous_execution_tactic)
    else:
        _CUBIC_MOE_EXECUTION_TACTICS.pop(execution_key, None)
    from vllm.logger import init_logger

    candidate_scores = ", ".join(
        f"grouped-{grouping}={score:.4f}ms" for score, grouping in scores
    )
    init_logger(__name__).info(
        "Cubic A8 route grouping: W%d H=%d I=%d G=%d M=%d "
        "candidates=[%s], selected=%d (%.4f ms)",
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        hidden_states.shape[0],
        candidate_scores,
        best_grouping,
        best_score,
    )


@torch.inference_mode()
def calibrate_cubic_moe_execution(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    dynamic_a8: bool,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    group_out: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> None:
    """Measure route-GEMV versus expert-sorted GEMM for a complete MoE layer."""
    key = (
        torch.accelerator.current_device_index(),
        dynamic_a8,
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        group_out,
        w1.shape[0],
        hidden_states.shape[0],
    )
    func = cubic_fused_moe_dynamic_a8 if dynamic_a8 else cubic_fused_moe
    args = (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
    )
    kwargs: dict[str, Any] = dict(
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )
    reference: torch.Tensor | None = None
    scores: list[tuple[float, bool, int]] = []
    candidates = ((True, 16), (False, 16), (False, 32))
    for use_gemv, dense_block_m in candidates:
        _CUBIC_MOE_EXECUTION_TACTICS[key] = use_gemv
        _CUBIC_MOE_DENSE_BLOCK_TACTICS[key] = dense_block_m

        def launch() -> torch.Tensor:
            return func(*args, **kwargs)

        try:
            output = launch()
            torch.accelerator.synchronize()
            if reference is None:
                reference = output
            else:
                torch.testing.assert_close(
                    output,
                    reference,
                    rtol=0.02,
                    atol=0.02,
                )
            large = hidden_states.shape[0] > 256
            score = triton.testing.do_bench(
                launch,
                warmup=5 if large else 10,
                rep=10 if large else 30,
            )
            scores.append((score, use_gemv, dense_block_m))
        except (RuntimeError, AssertionError) as error:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping Cubic %s W%d %s for H=%d I=%d M=%d: %s",
                "A8" if dynamic_a8 else "A16",
                num_bits,
                "GEMV" if use_gemv else f"GEMM/B{dense_block_m}",
                hidden_size,
                intermediate_size,
                hidden_states.shape[0],
                error,
            )
    if not scores:
        _CUBIC_MOE_EXECUTION_TACTICS.pop(key, None)
        _CUBIC_MOE_DENSE_BLOCK_TACTICS.pop(key, None)
        return
    measured_best = min(score for score, _, _ in scores)
    near_best = [item for item in scores if item[0] <= measured_best * 1.01]
    best_score, best_use_gemv, best_block_m = min(
        near_best,
        key=lambda item: (not item[1], item[2]),
    )
    _CUBIC_MOE_EXECUTION_TACTICS[key] = best_use_gemv
    _CUBIC_MOE_DENSE_BLOCK_TACTICS[key] = best_block_m
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Cubic %s execution: W%d H=%d I=%d G=%d M=%d %s (%.4f ms)",
        "A8" if dynamic_a8 else "A16",
        num_bits,
        hidden_size,
        intermediate_size,
        group_size,
        hidden_states.shape[0],
        "GEMV" if best_use_gemv else f"GEMM/B{best_block_m}",
        best_score,
    )


def _launch_cubic_moe_w2_fused_sum_2bit_a8(
    quantized_inputs: tuple[torch.Tensor, torch.Tensor],
    packed: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    group_size: int,
    multiply_routed_weight: bool,
) -> None:
    inputs_q, input_scale = quantized_inputs
    input_words = inputs_q.view(torch.int32)
    packed_words = packed.view(torch.int32)
    top_k = topk_ids.shape[1]
    grid = lambda meta: (
        triton.cdiv(output.shape[1], meta["BLOCK_N"]),
        output.shape[0],
    )
    _cubic_moe_w2_fused_sum_2bit_a8_dp4a_kernel[grid](
        input_words,
        input_scale,
        packed_words,
        scale,
        output,
        topk_weights,
        topk_ids,
        expert_map,
        output.shape[1],
        scale.shape[2],
        input_words.stride(0),
        input_words.stride(1),
        packed_words.stride(0),
        packed_words.stride(1),
        packed_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(0),
        output.stride(1),
        GROUP_SIZE=group_size,
        TOP_K=top_k,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
    )


def _launch_cubic_moe_fused_sum_dynamic_a8(
    quantized_inputs: tuple[torch.Tensor, torch.Tensor],
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    logical_k: int,
    num_bits: int,
    group_size: int,
    multiply_routed_weight: bool,
) -> None:
    inputs_q, input_scale = quantized_inputs
    top_k = topk_ids.shape[1]
    block_k = min(group_size, 128)
    grid = lambda meta: (
        triton.cdiv(output.shape[1], meta["BLOCK_N"]),
        output.shape[0],
    )
    _cubic_moe_w1_w8_fused_sum_a8_kernel[grid](
        inputs_q,
        input_scale,
        packed,
        scale,
        a,
        b,
        output,
        topk_weights,
        topk_ids,
        expert_map,
        output.shape[1],
        logical_k,
        packed.shape[2],
        scale.shape[2],
        inputs_q.stride(0),
        inputs_q.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(0),
        output.stride(1),
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
        TOP_K=top_k,
        MUL_ROUTED_WEIGHT=multiply_routed_weight,
        PRECOMPUTED_3BIT_LEVELS=(
            num_bits == 3 and a.dtype == torch.int8 and b.dtype == torch.int8
        ),
    )


def cubic_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    group_out: int = 1,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    use_sm120_fallbacks = torch.cuda.get_device_capability(hidden_states.device) == (
        12,
        0,
    )
    use_gemv = (
        num_tokens <= 8
        and num_bits <= 8
        and group_size <= 512
        and group_size & (group_size - 1) == 0
    )
    use_gemv = _cubic_moe_use_gemv(
        dynamic_a8=False,
        num_bits=num_bits,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        group_out=group_out,
        local_experts=w1.shape[0],
        num_tokens=num_tokens,
        fallback=use_gemv,
    )
    dense_block_m = _cubic_moe_dense_block_m(
        dynamic_a8=False,
        num_bits=num_bits,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        group_out=group_out,
        local_experts=w1.shape[0],
        num_tokens=num_tokens,
        fallback=16,
    )
    block_m = 1 if use_gemv else dense_block_m
    if use_gemv and num_tokens == 1 and expert_map is not None and top_k <= 32:
        sorted_ids, expert_ids, padded_count = cubic_compact_local_routes(
            topk_ids,
            expert_map,
        )
    else:
        sorted_ids, expert_ids, padded_count = _cubic_align_block_size(
            topk_ids,
            block_m,
            global_num_experts,
            w1.shape[0],
            expert_map,
        )
    route_ctas = None
    if use_gemv:
        max_route_ctas = 4 if use_sm120_fallbacks else 128
        fallback_route_ctas = min(sorted_ids.numel(), max_route_ctas)
        route_ctas = _cubic_moe_route_ctas(
            dynamic_a8=False,
            num_bits=num_bits,
            n=w1.shape[1],
            k=hidden_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w1.shape[0],
            grouped_routes=1,
            input_rows=num_tokens,
            top_k=top_k,
            fallback=fallback_route_ctas,
        )
    intermediate = w1.shape[1]
    activation_dim = intermediate // 2 if activation.is_gated else intermediate
    cache2 = torch.empty(
        num_tokens * top_k,
        activation_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    launch = _launch_cubic_moe_gemv if use_gemv else _launch_cubic_moe_gemm
    fused_2bit_situ = (
        use_gemv
        and num_bits == 2
        and activation == MoEActivation.SITU
        and group_size in (128, 256, 512)
        and group_out == 1
        and hidden_size % group_size == 0
    )
    if fused_2bit_situ:
        assert activation_situ_beta is not None
        _launch_cubic_moe_situ_gemv_2bit(
            hidden_states,
            w1,
            w1_scale,
            cache2,
            topk_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            logical_k=hidden_size,
            group_size=group_size,
            top_k=top_k,
            multiply_routed_weight=apply_router_weight_on_input,
            beta=activation_situ_beta,
            linear_beta=activation_situ_linear_beta,
            route_ctas=route_ctas,
        )
    else:
        cache1 = torch.empty(
            num_tokens,
            top_k,
            intermediate,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        launch(
            hidden_states,
            w1,
            w1_scale,
            w1_a,
            w1_b,
            cache1,
            topk_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            logical_k=hidden_size,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=top_k,
            multiply_routed_weight=apply_router_weight_on_input,
            sum_routes=False,
            route_ctas=route_ctas,
            dense_block_m=dense_block_m,
        )
    if not fused_2bit_situ:
        _apply_cubic_moe_activation(
            activation,
            cache2,
            cache1.view(-1, intermediate),
            activation_situ_beta,
            activation_situ_linear_beta,
        )
    output = torch.empty_like(hidden_states)
    fuse_route_sum = use_gemv and num_tokens == 1 and num_bits in (2, 3)
    use_torch_moe_sum = use_sm120_fallbacks
    if fuse_route_sum:
        w2_output = output
    elif expert_map is not None and use_torch_moe_sum:
        w2_output = torch.zeros(
            num_tokens,
            top_k,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        w2_output = torch.empty(
            num_tokens,
            top_k,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    down_route_ctas = route_ctas
    if use_gemv and not fuse_route_sum:
        down_route_ctas = _cubic_moe_route_ctas(
            dynamic_a8=False,
            num_bits=num_bits,
            n=w2.shape[1],
            k=intermediate_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w2.shape[0],
            grouped_routes=1,
            input_rows=num_tokens * top_k,
            top_k=1,
            fallback=route_ctas or min(sorted_ids.numel(), 128),
        )
    launch(
        cache2,
        w2,
        w2_scale,
        w2_a,
        w2_b,
        w2_output,
        topk_weights,
        sorted_ids,
        expert_ids,
        padded_count,
        logical_k=intermediate_size,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        top_k=1,
        multiply_routed_weight=not apply_router_weight_on_input,
        sum_routes=fuse_route_sum,
        route_ctas=down_route_ctas,
        dense_block_m=dense_block_m,
    )
    if not fuse_route_sum:
        if use_torch_moe_sum:
            torch.sum(w2_output, dim=1, out=output)
        else:
            ops.moe_sum(w2_output, output, topk_ids, expert_map)
    return output


def cubic_fused_moe_dynamic_a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    group_out: int = 1,
    _pipeline_chunked: bool = False,
) -> torch.Tensor:
    """Apply Cubic fused MoE with dynamic per-token INT8 activations."""
    if num_bits not in range(1, 9):
        raise ValueError(f"Unsupported Cubic bit width: {num_bits}.")
    if group_size not in (1, 32, 64, 128, 256, 512):
        raise ValueError(f"Unsupported Cubic Dynamic A8 group size: {group_size}.")
    if group_out <= 0:
        raise ValueError(f"Unsupported Cubic output group size: {group_out}.")

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    # Bound the W1 -> activation intermediate before selecting individual
    # kernels.  Large scheduler batches otherwise materialize
    # [tokens, top_k, 2 * intermediate] in one allocation (1.5 GiB for an
    # 8192-token production prefill).  Token slices are mathematically
    # independent: activation quantization is per token and route reduction is
    # confined to each token.  Keeping the scheduler batch intact avoids
    # repeated full-model prefill boundaries while this local pipeline remains
    # portable to devices with much smaller memory headroom.
    cache1_bytes_per_token = top_k * w1.shape[1] * hidden_states.element_size()
    cache1_nbytes = num_tokens * cache1_bytes_per_token
    if (
        hidden_states.is_cuda
        and num_tokens > 1
        and not _pipeline_chunked
        and cache1_nbytes > _CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES
    ):
        output = torch.empty_like(hidden_states)
        free_bytes, _ = torch.accelerator.get_memory_info(hidden_states.device)
        pipeline_budget = min(
            _CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES,
            max(
                _CUBIC_A8_PIPELINE_WORKSPACE_MIN_BYTES,
                free_bytes // 4,
            ),
        )
        token_chunk = max(1, pipeline_budget // cache1_bytes_per_token)
        for token_start in range(0, num_tokens, token_chunk):
            token_end = min(token_start + token_chunk, num_tokens)
            output[token_start:token_end] = cubic_fused_moe_dynamic_a8(
                hidden_states[token_start:token_end],
                w1,
                w2,
                w1_scale,
                w2_scale,
                w1_a,
                w1_b,
                w2_a,
                w2_b,
                topk_weights[token_start:token_end],
                topk_ids[token_start:token_end],
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                num_bits=num_bits,
                group_size=group_size,
                group_out=group_out,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
                _pipeline_chunked=True,
            )
        return output
    # Expert-parallel prefills remain route-sparse far beyond eight input rows.
    # These crossover limits cover every Cubic bit width and avoid padding each
    # active expert to BLOCK_M=16 while the route kernels are still faster.
    # W2/G128+ has a byte-parallel DP4A implementation and remains preferable
    # for dense prefills as well.  Small-group W3 stays conservative because
    # statically expanding its many groups has excessive compile cost.
    route_limit: int | None
    if num_bits == 1:
        # The W1 DP4A route kernel is still 7-10x faster than the dense
        # fallback at the ~117 local routes produced by a 64-token EP batch.
        # In CUDA graphs the dense grid is sized from the static EM buffer,
        # not the live route count, so switching at 32 rows launches more than
        # 100k mostly-empty CTAs.  Keep route execution through this measured
        # crossover for every group size; larger prefills can still select the
        # dense kernel.
        route_limit = 128
    elif num_bits == 2:
        route_limit = (
            None
            if group_size in (128, 256, 512)
            and hidden_size % group_size == 0
            and intermediate_size % group_size == 0
            else 128
        )
    elif num_bits == 3:
        route_limit = 256 if group_size == 256 else 128 if group_size == 512 else 8
    elif num_bits == 4 or num_bits == 5:
        route_limit = 128
    elif num_bits == 6:
        route_limit = 256 if group_size >= 64 else 128
    else:
        route_limit = 128
    use_gemv = (
        (group_out != 1 and group_size != 1)
        or route_limit is None
        or num_tokens <= route_limit
    )
    if group_size == 1:
        use_gemv = False
    use_gemv = _cubic_moe_use_gemv(
        dynamic_a8=True,
        num_bits=num_bits,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        group_out=group_out,
        local_experts=w1.shape[0],
        num_tokens=num_tokens,
        fallback=use_gemv,
    )
    # G256/G512 have enough group work for two same-expert routes to amortize
    # each packed-weight load.  W4-W8 needs a modest batch before expert
    # collisions offset pair padding; below that crossover, retain the
    # singleton path.  These are format/shape rules, independent of model
    # expert numbering or placement.
    grouped_routes = (
        2
        if use_gemv
        and (
            (num_bits == 2 and group_size in (256, 512))
            or (
                num_bits == 3
                and group_size in (256, 512)
                and w1_a.dtype == torch.int8
                and w1_b.dtype == torch.int8
                and w2_a.dtype == torch.int8
                and w2_b.dtype == torch.int8
            )
            or (
                4 <= num_bits <= 8
                and group_size in (256, 512)
                and num_tokens >= 16
                and group_size / (1 << (num_bits - 1)) >= 2
                and hidden_size % group_size == 0
                and intermediate_size % group_size == 0
                and w1_a.dtype == torch.float16
                and w1_b.dtype == torch.float16
                and w2_a.dtype == torch.float16
                and w2_b.dtype == torch.float16
            )
        )
        else 1
    )
    if use_gemv and num_bits >= 2:
        grouped_routes = _cubic_a8_moe_grouping(
            num_bits=num_bits,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w1.shape[0],
            num_tokens=num_tokens,
            fallback=grouped_routes,
        )
    if (
        use_gemv
        and group_out > 1
        and group_size >= 128
        and 4 <= num_bits <= 8
    ):
        if num_tokens >= 256:
            grouped_routes = 8
        elif num_tokens >= 16:
            grouped_routes = 4
        else:
            grouped_routes = 1
    # Dense W3 with precomputed carriers is weight-decode limited.  A 32-row
    # expert tile reuses every decoded carrier across twice as many routes and
    # reaches a substantially better INT8 tensor-core tile.  Keep alignment
    # and both projections on the same tile size; route-GEMV paths retain
    # their independently selected route grouping.
    dense_block_m = 1 if group_size == 1 and num_tokens <= 8 else 16
    if (
        not use_gemv
        and num_bits == 3
        and group_size == 256
        and w1_a.dtype == torch.int8
        and w1_b.dtype == torch.int8
        and w2_a.dtype == torch.int8
        and w2_b.dtype == torch.int8
    ):
        dense_block_m = 32
    dense_block_m = _cubic_moe_dense_block_m(
        dynamic_a8=True,
        num_bits=num_bits,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        group_out=group_out,
        local_experts=w1.shape[0],
        num_tokens=num_tokens,
        fallback=dense_block_m,
    )
    use_sm120_fallbacks = torch.cuda.get_device_capability(hidden_states.device) == (
        12,
        0,
    )
    if (
        use_gemv
        and grouped_routes == 1
        and num_tokens == 1
        and expert_map is not None
        and top_k <= 32
    ):
        sorted_ids, expert_ids, padded_count = cubic_compact_local_routes(
            topk_ids,
            expert_map,
        )
    else:
        sorted_ids, expert_ids, padded_count = _cubic_align_block_size(
            topk_ids,
            grouped_routes if use_gemv else dense_block_m,
            global_num_experts,
            w1.shape[0],
            expert_map,
        )
    route_ctas = None
    if use_gemv:
        expected_local_routes = (
            num_tokens * top_k * w1.shape[0] + global_num_experts - 1
        ) // global_num_experts
        expected_local_blocks = (
            expected_local_routes + grouped_routes - 1
        ) // grouped_routes
        route_ctas = min(
            max(1 << (max(expected_local_blocks, 1) - 1).bit_length(), 16),
            128,
        )
        route_ctas = _cubic_moe_route_ctas(
            dynamic_a8=True,
            num_bits=num_bits,
            n=w1.shape[1],
            k=hidden_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w1.shape[0],
            grouped_routes=grouped_routes,
            input_rows=num_tokens,
            top_k=top_k,
            fallback=route_ctas,
        )
    intermediate = w1.shape[1]
    activation_dim = intermediate // 2 if activation.is_gated else intermediate
    quantized_cache2: tuple[torch.Tensor, torch.Tensor] | None = None
    groupwise_cache2: CubicA8Carrier | None = None
    cubic8_cache2: CubicA8Code | None = None
    # Keep the established per-token Dynamic A8 path as the unconditional
    # fallback.  The first integrated groupwise consumer is route-GEMV; dense
    # prefill remains on the current path until its independent kernel is
    # implemented and calibrated.
    # Online Cubic means that the persistent producer/consumer boundary holds
    # true Cubic codes and curve metadata.  The old groupwise linear INT8
    # carrier is deliberately not selected by this switch: it remains useful
    # as an implementation primitive, but must not masquerade as Cubic8.
    #
    # The first calibrated exact consumer is W2/G512.  Small route batches are
    # compute-bound by curve reconstruction and retain established Dynamic A8;
    # larger batches amortize that arithmetic over the output tile and benefit
    # from the narrower persistent activation.  Other shapes fall back until
    # their exact consumers have independently demonstrated a gain.
    use_cubic8_cache2 = False
    use_groupwise_cache2 = False
    fused_2bit_situ = (
        use_gemv
        and num_bits == 2
        and activation == MoEActivation.SITU
        and group_size in (128, 256, 512)
        and hidden_size % group_size == 0
    )
    if fused_2bit_situ:
        assert activation_situ_beta is not None
        if use_cubic8_cache2:
            # A true Online Cubic boundary cannot first materialize cache2 as
            # BF16.  Re-align singleton routes for the fused producer because
            # its expert-id contract is one entry per route, then write only
            # Cubic code + metadata to global memory.
            producer_sorted_ids, producer_expert_ids, producer_padded_count = (
                _cubic_align_block_size(
                    topk_ids,
                    1,
                    global_num_experts,
                    w1.shape[0],
                    expert_map,
                )
            )
            cubic8_cache2 = _launch_cubic_moe_situ_cubic8_2bit(
                hidden_states,
                w1,
                w1_scale,
                topk_weights,
                producer_sorted_ids,
                producer_expert_ids,
                producer_padded_count,
                logical_k=hidden_size,
                group_size=group_size,
                output_group_size=256,
                top_k=top_k,
                multiply_routed_weight=apply_router_weight_on_input,
                beta=activation_situ_beta,
                linear_beta=activation_situ_linear_beta,
                route_ctas=route_ctas,
            )
        else:
            cache2 = torch.empty(
                num_tokens * top_k,
                activation_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _launch_cubic_moe_situ_gemv_2bit(
                hidden_states,
                w1,
                w1_scale,
                cache2,
                topk_weights,
                sorted_ids,
                expert_ids,
                padded_count,
                logical_k=hidden_size,
                group_size=group_size,
                top_k=top_k,
                multiply_routed_weight=apply_router_weight_on_input,
                beta=activation_situ_beta,
                linear_beta=activation_situ_linear_beta,
                dynamic_a8=True,
                route_ctas=route_ctas,
                grouped_routes=grouped_routes,
            )
            if use_groupwise_cache2:
                groupwise_cache2 = _quantize_cubic_groupwise_a8(cache2, group_size)
            else:
                quantized_cache2 = per_token_quant_int8(cache2.contiguous())
    else:
        cache1 = torch.empty(
            num_tokens,
            top_k,
            intermediate,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        _launch_cubic_moe_dynamic_a8(
            hidden_states,
            w1,
            w1_scale,
            w1_a,
            w1_b,
            cache1,
            topk_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            logical_k=hidden_size,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=top_k,
            multiply_routed_weight=apply_router_weight_on_input,
            use_gemv=use_gemv,
            sum_routes=False,
            route_ctas=route_ctas,
            grouped_routes=grouped_routes,
            dense_block_m=dense_block_m,
        )
    if not fused_2bit_situ and activation == MoEActivation.SITU:
        if activation_situ_beta is None:
            raise ValueError("Cubic SITU requires activation_situ_beta.")
        if use_cubic8_cache2:
            cache2 = torch.empty(
                num_tokens * top_k,
                activation_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _apply_cubic_moe_activation(
                activation,
                cache2,
                cache1.view(-1, intermediate),
                activation_situ_beta,
                activation_situ_linear_beta,
            )
            cubic8_cache2 = _quantize_cubic_groupwise_cubic8(cache2, group_size)
        elif use_groupwise_cache2:
            groupwise_cache2 = _apply_cubic_situ_groupwise_quant_int8(
                cache1.view(-1, intermediate),
                activation_dim,
                activation_situ_beta,
                activation_situ_linear_beta,
                group_size,
            )
            cache2 = groupwise_cache2.values
        else:
            quantized_cache2 = _apply_cubic_situ_quant_int8(
                cache1.view(-1, intermediate),
                activation_dim,
                activation_situ_beta,
                activation_situ_linear_beta,
            )
            cache2 = quantized_cache2[0]
    elif not fused_2bit_situ:
        cache2 = torch.empty(
            num_tokens * top_k,
            activation_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        _apply_cubic_moe_activation(
            activation,
            cache2,
            cache1.view(-1, intermediate),
            activation_situ_beta,
            activation_situ_linear_beta,
        )
    # The route-major down-projection buffer is num_tokens * top_k * hidden and
    # reaches hundreds of MiB for a normal prefill.  The bounded-workspace
    # Keep the route-major workspace bounded for large batches.  A token-major
    # W2 CTA serializes TOP_K routes, so it is only beneficial when TOP_K is
    # one; with several routes, parallel route CTAs plus moe_sum are faster.
    route_output_nbytes = (
        num_tokens * top_k * hidden_size * hidden_states.element_size()
    )
    if (
        (num_bits == 2 and num_tokens >= 16)
        or route_output_nbytes >= _CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES
    ) and (
        group_size in (128, 256, 512)
        and intermediate_size % group_size == 0
        and (
            expert_map is not None
            or groupwise_cache2 is not None
            or cubic8_cache2 is not None
        )
        and (
            quantized_cache2 is not None
            or groupwise_cache2 is not None
            or cubic8_cache2 is not None
        )
    ):
        output = torch.empty_like(hidden_states)
        if (
            num_bits == 2
            and top_k == 1
            and groupwise_cache2 is None
            and cubic8_cache2 is None
        ):
            assert quantized_cache2 is not None
            _launch_cubic_moe_w2_fused_sum_2bit_a8(
                quantized_cache2,
                w2,
                w2_scale,
                output,
                topk_weights,
                topk_ids,
                expert_map,
                group_size=group_size,
                multiply_routed_weight=not apply_router_weight_on_input,
            )
        else:
            # Keep the measured production route kernels, but bound their
            # temporary route-major output.  This avoids the 100s-of-MiB
            # allocation without replacing a dense prefill with the fallback
            # token-major prototype.  The budget adapts to genuinely small
            # devices and to memory left by CUDA graph private pools.
            free_bytes, _ = torch.accelerator.get_memory_info(hidden_states.device)
            reusable_bytes = max(
                0,
                torch.accelerator.memory_reserved(hidden_states.device)
                - torch.accelerator.memory_allocated(hidden_states.device),
            )
            workspace_budget = min(
                _CUBIC_A8_ROUTE_WORKSPACE_MAX_BYTES,
                max(
                    _CUBIC_A8_ROUTE_WORKSPACE_MIN_BYTES,
                    (free_bytes + reusable_bytes) // 2,
                ),
            )
            bytes_per_token = top_k * hidden_size * hidden_states.element_size()
            token_chunk = max(1, workspace_budget // bytes_per_token)
            if cubic8_cache2 is not None:
                q_cache2 = cubic8_cache2.codes
                q_cache2_scale = cubic8_cache2.scales
            elif quantized_cache2 is not None:
                q_cache2, q_cache2_scale = quantized_cache2
            else:
                assert groupwise_cache2 is not None
                q_cache2 = groupwise_cache2.values
                q_cache2_scale = groupwise_cache2.scales
            token_start = 0
            while token_start < num_tokens:
                token_end = min(token_start + token_chunk, num_tokens)
                chunk_tokens = token_end - token_start
                route_start = token_start * top_k
                route_end = token_end * top_k
                chunk_topk_ids = topk_ids[token_start:token_end]
                chunk_topk_weights = topk_weights[token_start:token_end]
                try:
                    route_output_factory = (
                        torch.zeros if use_sm120_fallbacks else torch.empty
                    )
                    chunk_route_output = route_output_factory(
                        chunk_tokens,
                        top_k,
                        hidden_size,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                except torch.cuda.OutOfMemoryError:
                    if token_chunk == 1:
                        raise
                    token_chunk = max(1, token_chunk // 2)
                    torch.accelerator.empty_cache()
                    continue
                chunk_use_gemv = route_limit is None or chunk_tokens <= route_limit
                chunk_use_gemv = _cubic_moe_use_gemv(
                    dynamic_a8=True,
                    num_bits=num_bits,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    group_size=group_size,
                    group_out=group_out,
                    local_experts=w2.shape[0],
                    num_tokens=chunk_tokens,
                    fallback=chunk_use_gemv,
                )
                # An online carrier is only produced after the enclosing
                # shape selected its route consumer.  A smaller workspace
                # slice cannot fall through to the dense per-token consumer:
                # the original BF16 activation was intentionally not kept.
                if groupwise_cache2 is not None or cubic8_cache2 is not None:
                    chunk_use_gemv = True
                chunk_grouped_routes = (
                    2
                    if chunk_use_gemv
                    and group_size in (256, 512)
                    and (
                        (num_bits == 2)
                        or (
                            num_bits == 3
                            and w2_a.dtype == torch.int8
                            and w2_b.dtype == torch.int8
                        )
                        or (
                            4 <= num_bits <= 8
                            and chunk_tokens >= 16
                            and group_size / (1 << (num_bits - 1)) >= 2
                            and w2_a.dtype == torch.float16
                            and w2_b.dtype == torch.float16
                        )
                    )
                    else 1
                )
                if chunk_use_gemv and num_bits >= 2:
                    chunk_grouped_routes = _cubic_a8_moe_grouping(
                        num_bits=num_bits,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        group_size=group_size,
                        group_out=group_out,
                        local_experts=w2.shape[0],
                        num_tokens=chunk_tokens,
                        fallback=chunk_grouped_routes,
                    )
                if (
                    chunk_use_gemv
                    and group_out > 1
                    and group_size >= 128
                    and 4 <= num_bits <= 8
                ):
                    if chunk_tokens >= 256:
                        chunk_grouped_routes = 8
                    elif chunk_tokens >= 16:
                        chunk_grouped_routes = 4
                    else:
                        chunk_grouped_routes = 1
                # The exact Cubic8 kernel maps one expert id per valid route;
                # unlike the pair/quad DP4A kernels it does not interpret an
                # expert id as covering several aligned routes.
                if cubic8_cache2 is not None:
                    chunk_grouped_routes = 1
                chunk_sorted_ids, chunk_expert_ids, chunk_padded_count = (
                    _cubic_align_block_size(
                        chunk_topk_ids,
                        chunk_grouped_routes if chunk_use_gemv else dense_block_m,
                        global_num_experts,
                        w2.shape[0],
                        expert_map,
                    )
                )
                chunk_route_ctas = None
                if chunk_use_gemv:
                    expected_local_routes = (
                        chunk_tokens * top_k * w2.shape[0] + global_num_experts - 1
                    ) // global_num_experts
                    expected_local_blocks = (
                        expected_local_routes + chunk_grouped_routes - 1
                    ) // chunk_grouped_routes
                    chunk_route_ctas = min(
                        max(
                            1 << (max(expected_local_blocks, 1) - 1).bit_length(),
                            16,
                        ),
                        128,
                    )
                    chunk_route_ctas = _cubic_moe_route_ctas(
                        dynamic_a8=True,
                        num_bits=num_bits,
                        n=w2.shape[1],
                        k=intermediate_size,
                        group_size=group_size,
                        group_out=group_out,
                        local_experts=w2.shape[0],
                        grouped_routes=chunk_grouped_routes,
                        input_rows=chunk_tokens * top_k,
                        top_k=1,
                        fallback=chunk_route_ctas,
                    )
                if cubic8_cache2 is not None:
                    chunk_carrier = CubicA8Code(
                        cubic8_cache2.codes[route_start:route_end],
                        cubic8_cache2.scales[route_start:route_end],
                        cubic8_cache2.a[route_start:route_end],
                        cubic8_cache2.b[route_start:route_end],
                        cubic8_cache2.group_size,
                    )
                    _launch_cubic_moe_groupwise_a8(
                        _as_groupwise_a8(chunk_carrier),
                        w2,
                        w2_scale,
                        w2_a,
                        w2_b,
                        chunk_route_output,
                        chunk_topk_weights,
                        chunk_sorted_ids,
                        chunk_expert_ids,
                        chunk_padded_count,
                        logical_k=intermediate_size,
                        num_bits=num_bits,
                        group_size=group_size,
                        group_out=group_out,
                        top_k=1,
                        multiply_routed_weight=not apply_router_weight_on_input,
                        sum_routes=False,
                        route_ctas=chunk_route_ctas,
                        grouped_routes=1,
                    )
                elif groupwise_cache2 is not None and chunk_use_gemv:
                    chunk_groupwise_carrier = CubicA8Carrier(
                        q_cache2[route_start:route_end],
                        q_cache2_scale[route_start:route_end],
                        groupwise_cache2.group_size,
                    )
                    _launch_cubic_moe_groupwise_a8(
                        chunk_groupwise_carrier,
                        w2,
                        w2_scale,
                        w2_a,
                        w2_b,
                        chunk_route_output,
                        chunk_topk_weights,
                        chunk_sorted_ids,
                        chunk_expert_ids,
                        chunk_padded_count,
                        logical_k=intermediate_size,
                        num_bits=num_bits,
                        group_size=group_size,
                        group_out=group_out,
                        top_k=1,
                        multiply_routed_weight=not apply_router_weight_on_input,
                        sum_routes=False,
                        route_ctas=chunk_route_ctas,
                        grouped_routes=chunk_grouped_routes,
                    )
                else:
                    # The established per-token consumer is also the automatic
                    # fallback if calibration selects a dense execution mode.
                    assert quantized_cache2 is not None
                    _launch_cubic_moe_dynamic_a8(
                        cache2[route_start:route_end],
                        w2,
                        w2_scale,
                        w2_a,
                        w2_b,
                        chunk_route_output,
                        chunk_topk_weights,
                        chunk_sorted_ids,
                        chunk_expert_ids,
                        chunk_padded_count,
                        logical_k=intermediate_size,
                        num_bits=num_bits,
                        group_size=group_size,
                        group_out=group_out,
                        top_k=1,
                        multiply_routed_weight=not apply_router_weight_on_input,
                        use_gemv=chunk_use_gemv,
                        sum_routes=False,
                        quantized_inputs=(
                            q_cache2[route_start:route_end],
                            q_cache2_scale[route_start:route_end],
                        ),
                        route_ctas=chunk_route_ctas,
                        grouped_routes=chunk_grouped_routes,
                        dense_block_m=dense_block_m,
                    )
                if use_sm120_fallbacks:
                    torch.sum(
                        chunk_route_output,
                        dim=1,
                        out=output[token_start:token_end],
                    )
                else:
                    ops.moe_sum(
                        chunk_route_output,
                        output[token_start:token_end],
                        chunk_topk_ids,
                        expert_map,
                    )
                token_start = token_end
        return output
    # Grouped-route kernels write one output row per route.  They cannot target
    # the single-row fused-sum buffer without an explicit cross-route
    # reduction; doing so writes routes 1..top_k-1 out of bounds.
    use_native_w4_w8_down = (
        use_gemv
        and 4 <= num_bits <= 8
        and group_size in (128, 256, 512)
        and group_size / (1 << (num_bits - 1)) >= 2
        and intermediate_size % group_size == 0
        and w2_a.dtype == torch.float16
        and w2_b.dtype == torch.float16
        and _cubic_a8_moe_backend(
            num_bits=num_bits,
            n=w2.shape[1],
            k=intermediate_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w2.shape[0],
            grouped_routes=grouped_routes,
            route_ctas=route_ctas or 1,
        )
        == "cuda"
    )
    fuse_route_sum = (
        use_gemv
        and num_tokens == 1
        and grouped_routes == 1
        and groupwise_cache2 is None
        and cubic8_cache2 is None
        and not use_native_w4_w8_down
    )
    if (
        not fuse_route_sum
        and route_output_nbytes >= _CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES
        and hidden_states.is_cuda
    ):
        free_bytes, _ = torch.accelerator.get_memory_info(hidden_states.device)
        if free_bytes < route_output_nbytes + 64 * 1024 * 1024:
            # Large CUDA-graph private pools can leave enough aggregate cached
            # space but no contiguous segment for this eager prefill buffer.
            # Releasing only unused cache here is a rare, size-gated fallback;
            # subsequent MoE layers reuse the newly contiguous route buffer.
            torch.accelerator.empty_cache()
    use_torch_moe_sum = use_sm120_fallbacks
    if fuse_route_sum:
        w2_output = torch.empty_like(hidden_states)
    elif expert_map is not None and use_torch_moe_sum:
        w2_output = torch.zeros(
            num_tokens,
            top_k,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        w2_output = torch.empty(
            num_tokens,
            top_k,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    down_route_ctas = route_ctas
    if use_gemv and not fuse_route_sum:
        down_route_ctas = _cubic_moe_route_ctas(
            dynamic_a8=True,
            num_bits=num_bits,
            n=w2.shape[1],
            k=intermediate_size,
            group_size=group_size,
            group_out=group_out,
            local_experts=w2.shape[0],
            grouped_routes=grouped_routes,
            input_rows=num_tokens * top_k,
            top_k=1,
            fallback=route_ctas or min(sorted_ids.numel(), 128),
        )
    if cubic8_cache2 is not None:
        cubic8_sorted_ids, cubic8_expert_ids, cubic8_padded_count = (
            _cubic_align_block_size(
                topk_ids,
                1,
                global_num_experts,
                w2.shape[0],
                expert_map,
            )
        )
        _launch_cubic_moe_groupwise_a8(
            _as_groupwise_a8(cubic8_cache2),
            w2,
            w2_scale,
            w2_a,
            w2_b,
            w2_output,
            topk_weights,
            cubic8_sorted_ids,
            cubic8_expert_ids,
            cubic8_padded_count,
            logical_k=intermediate_size,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=1,
            multiply_routed_weight=not apply_router_weight_on_input,
            sum_routes=False,
            route_ctas=down_route_ctas,
            grouped_routes=1,
        )
    elif groupwise_cache2 is not None:
        _launch_cubic_moe_groupwise_a8(
            groupwise_cache2,
            w2,
            w2_scale,
            w2_a,
            w2_b,
            w2_output,
            topk_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            logical_k=intermediate_size,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=1,
            multiply_routed_weight=not apply_router_weight_on_input,
            sum_routes=fuse_route_sum,
            route_ctas=down_route_ctas,
            grouped_routes=grouped_routes,
        )
    else:
        _launch_cubic_moe_dynamic_a8(
            cache2,
            w2,
            w2_scale,
            w2_a,
            w2_b,
            w2_output,
            topk_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            logical_k=intermediate_size,
            num_bits=num_bits,
            group_size=group_size,
            group_out=group_out,
            top_k=1,
            multiply_routed_weight=not apply_router_weight_on_input,
            use_gemv=use_gemv,
            sum_routes=fuse_route_sum,
            quantized_inputs=quantized_cache2,
            route_ctas=down_route_ctas,
            grouped_routes=grouped_routes,
            dense_block_m=dense_block_m,
        )
    if fuse_route_sum:
        return w2_output
    output = torch.empty_like(hidden_states)
    if use_torch_moe_sum:
        torch.sum(w2_output, dim=1, out=output)
    else:
        ops.moe_sum(w2_output, output, topk_ids, expert_map)
    return output


def _cubic_fused_moe_custom_op(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    num_bits: int,
    group_size: int,
    group_out: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> torch.Tensor:
    return cubic_fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
        activation=MoEActivation.from_str(activation),
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


def _cubic_fused_moe_custom_op_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    num_bits: int,
    group_size: int,
    group_out: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="cubic_fused_moe",
    op_func=_cubic_fused_moe_custom_op,
    mutates_args=[],
    fake_impl=_cubic_fused_moe_custom_op_fake,
)


def _cubic_fused_moe_dynamic_a8_custom_op(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    num_bits: int,
    group_size: int,
    group_out: int,
    hidden_size: int,
    intermediate_size: int,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
) -> torch.Tensor:
    return cubic_fused_moe_dynamic_a8(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
        activation=MoEActivation.from_str(activation),
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        num_bits=num_bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


direct_register_custom_op(
    op_name="cubic_fused_moe_dynamic_a8",
    op_func=_cubic_fused_moe_dynamic_a8_custom_op,
    mutates_args=[],
    fake_impl=_cubic_fused_moe_custom_op_fake,
)
