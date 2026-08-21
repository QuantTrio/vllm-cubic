# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import regex as re
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.cubic_policy import (
    CUBIC_SUPPORTED_BITS,
    cubic_dynamic_a8_group_size,
)
from vllm.model_executor.parameter import (
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs

CUBIC_FORMAT = "cubic-pack-quantized"
CUBIC_LEGACY_METADATA_FORMAT = "float32-scale-float16-ab"
CUBIC_COMPACT_METADATA_FORMAT = "int8-scale-int4-ab-global-fp32"
CUBIC_E5M9_CURVE2_METADATA_FORMAT = "e5m9-scale-curve2-tensor-fp16"


class CubicLogicalScaleParameter(PerTensorScaleParameter):
    """Load metadata for each logical partition of a fused weight."""

    def _load_into_shard_id(
        self,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        **kwargs: Any,
    ) -> None:
        shard = self._shard_id_as_int(shard_id)
        if loaded_weight.ndim == 0:
            value = loaded_weight
        elif loaded_weight.shape[0] == 1:
            value = loaded_weight[0]
        elif shard < loaded_weight.shape[0]:
            value = loaded_weight[shard]
        else:
            raise ValueError(
                "Cubic logical metadata does not contain shard "
                f"{shard}: shape={tuple(loaded_weight.shape)}."
            )
        target = self.data[shard]
        if target.shape != value.shape:
            raise ValueError(
                "Cubic logical metadata shape mismatch: "
                f"expected {tuple(target.shape)}, got {tuple(value.shape)}."
            )
        target.copy_(value)


def _normalize_group_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return 1, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(size, int) for size in value)
    ):
        return value[0], value[1]
    raise ValueError("Cubic group_size must be an integer or [group_out, group_in].")


@dataclass(frozen=True)
class CubicScheme:
    num_bits: int
    group_size: int
    group_out: int
    param_dtype: torch.dtype = torch.float16
    metadata_format: str = CUBIC_LEGACY_METADATA_FORMAT
    reserved_code: str = "zero"

    def __post_init__(self) -> None:
        if self.num_bits not in CUBIC_SUPPORTED_BITS:
            raise ValueError(
                f"Cubic num_bits must be in {CUBIC_SUPPORTED_BITS}, "
                f"got {self.num_bits}."
            )
        if self.group_size <= 0 or self.group_out <= 0:
            raise ValueError("Cubic group dimensions must be positive.")
        if self.metadata_format not in (
            CUBIC_LEGACY_METADATA_FORMAT,
            CUBIC_COMPACT_METADATA_FORMAT,
            CUBIC_E5M9_CURVE2_METADATA_FORMAT,
        ):
            raise ValueError(
                f"Unsupported Cubic metadata format {self.metadata_format!r}."
            )
        if self.param_dtype != torch.float16:
            raise ValueError(
                "Cubic runtime a/b metadata must use FP16 after checkpoint decoding."
            )
        expected_code = "binary" if self.num_bits == 1 else "zero"
        if self.reserved_code != expected_code:
            raise ValueError(
                f"Cubic {self.num_bits}-bit weights require "
                f"reserved_code={expected_code!r}."
            )

    @property
    def effective_bits(self) -> float:
        metadata_bits = (
            64 if self.metadata_format == CUBIC_LEGACY_METADATA_FORMAT else 16
        )
        return self.num_bits + metadata_bits / (self.group_out * self.group_size)

    @property
    def group_shape(self) -> tuple[int, int]:
        return self.group_out, self.group_size


@dataclass(frozen=True)
class CubicQuantizedTensor:
    packed: torch.Tensor
    scale: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    shape: tuple[int, ...]
    num_bits: int
    group_size: int
    group_out: int
    loss: torch.Tensor


class CubicLinearMetadataParameter(PackedvLLMParameter):
    """Cubic metadata sharded in weight-group coordinates."""

    def __init__(
        self,
        *,
        input_size_per_partition: int,
        input_group_size: int,
        **kwargs: Any,
    ) -> None:
        self.input_size_per_partition = input_size_per_partition
        self.input_group_size = input_group_size
        super().__init__(**kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        shard_size = self.data.shape[self.input_dim]
        input_offset = self.tp_rank * self.input_size_per_partition
        group_offset = input_offset // self.input_group_size
        loaded_weight = loaded_weight.narrow(self.input_dim, group_offset, shard_size)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


def decode_cubic_compact_metadata(
    scale_code: torch.Tensor,
    packed_ab: torch.Tensor,
    scale_global: torch.Tensor,
    a_global: torch.Tensor,
    b_global: torch.Tensor,
    output_partition_sizes: list[int],
    group_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand compact checkpoint metadata into the existing runtime ABI."""
    output_group_counts = torch.tensor(
        output_partition_sizes,
        device=scale_code.device,
        dtype=torch.int64,
    ).div(group_out, rounding_mode="floor")

    def expand_global(value: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(value.float(), output_group_counts)[:, None]

    signed_a = (packed_ab.to(torch.int8) << 4) >> 4
    signed_b = packed_ab.to(torch.int8) >> 4
    scale = scale_code.float() * expand_global(scale_global)
    a = 1.0 + signed_a.float() * expand_global(a_global)
    b = signed_b.float() * expand_global(b_global)
    return scale, a.half(), b.half()


def decode_cubic_e5m9_curve2_metadata(
    metadata: torch.Tensor,
    curve_a: torch.Tensor,
    curve_b: torch.Tensor,
    output_partition_sizes: list[int],
    group_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the portable 14-bit E5M9 scale and 2-bit curve ABI."""
    if metadata.dtype != torch.uint16:
        raise ValueError("Cubic E5M9 metadata must remain uint16.")
    if curve_a.dtype != torch.float16 or curve_b.dtype != torch.float16:
        raise ValueError("Cubic E5M9 curve tables must remain FP16.")
    if curve_a.shape != curve_b.shape or curve_a.shape[-1] != 4:
        raise ValueError("Cubic E5M9 curve tables must have shape [..., 4].")
    metadata_i32 = metadata.int()
    curve_id = torch.bitwise_right_shift(metadata_i32, 14).long()
    scale_bits = torch.bitwise_left_shift(metadata_i32 & 0x3FFF, 1).to(
        torch.int16
    )
    scale = scale_bits.view(torch.float16).float()
    if curve_a.shape[0] == 1:
        a = curve_a[0, curve_id]
        b = curve_b[0, curve_id]
    else:
        output_group_counts = torch.tensor(
            output_partition_sizes,
            device=metadata.device,
            dtype=torch.int64,
        ).div(group_out, rounding_mode="floor")
        partition = torch.repeat_interleave(
            torch.arange(
                len(output_group_counts),
                device=metadata.device,
                dtype=torch.int64,
            ),
            output_group_counts,
        )[:, None]
        a = curve_a[partition, curve_id]
        b = curve_b[partition, curve_id]
    return scale, a, b


def _decode_cubic_moe_e5m9_curve2_metadata(
    metadata: torch.Tensor,
    curve_a: torch.Tensor,
    curve_b: torch.Tensor,
    output_partition_sizes: tuple[int, ...],
    group_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if metadata.dtype != torch.uint16:
        raise ValueError("Cubic MoE E5M9 metadata must remain uint16.")
    output_groups = tuple(size // group_out for size in output_partition_sizes)
    partition = torch.repeat_interleave(
        torch.arange(len(output_groups), device=metadata.device),
        torch.tensor(output_groups, device=metadata.device),
    )
    metadata_i32 = metadata.int()
    curve_id = torch.bitwise_right_shift(metadata_i32, 14).long()
    scale_bits = torch.bitwise_left_shift(metadata_i32 & 0x3FFF, 1).to(
        torch.int16
    )
    scale = scale_bits.view(torch.float16).float()
    selected_a = curve_a[:, partition]
    selected_b = curve_b[:, partition]
    a = selected_a.gather(2, curve_id)
    b = selected_b.gather(2, curve_id)
    return scale, a, b


def expanded_cubic_metadata(
    layer: torch.nn.Module,
    scheme: CubicScheme,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the logical FP32/FP16 metadata for either checkpoint ABI."""
    if scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
        return decode_cubic_compact_metadata(
            layer.weight_scale,
            layer.weight_ab,
            layer.weight_scale_global,
            layer.weight_a_global,
            layer.weight_b_global,
            layer.output_partition_sizes,
            scheme.group_out,
        )
    if scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
        return decode_cubic_e5m9_curve2_metadata(
            layer.weight_metadata,
            layer.weight_curve_a,
            layer.weight_curve_b,
            layer.output_partition_sizes,
            scheme.group_out,
        )
    return layer.weight_scale, layer.weight_a, layer.weight_b


def install_cubic_expanded_metadata(
    layer: torch.nn.Module,
    scheme: CubicScheme,
    *,
    token_buckets: tuple[int, ...] | None = None,
) -> None:
    """Cache decoded metadata while retaining the compact checkpoint ABI."""
    if getattr(layer, "cubic_metadata_is_expanded", False):
        return
    scale, coefficient_a, coefficient_b = expanded_cubic_metadata(layer, scheme)
    layer.register_buffer("cubic_weight_scale_expanded", scale)
    layer.register_buffer("cubic_weight_a_expanded", coefficient_a)
    layer.register_buffer("cubic_weight_b_expanded", coefficient_b)
    layer.cubic_metadata_is_expanded = True
    layer.cubic_expanded_metadata_token_buckets = token_buckets


def cubic_levels(
    total_bits: int,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if total_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Unsupported Cubic bit width: {total_bits}.")
    a_tensor = torch.as_tensor(a, device=device, dtype=dtype)
    if total_bits == 1:
        return torch.ones((*a_tensor.shape, 1), device=a_tensor.device, dtype=dtype)
    magnitude_max = (1 << (total_bits - 1)) - 1
    b_tensor = torch.as_tensor(b, device=device, dtype=dtype)
    t = (
        torch.arange(magnitude_max + 1, device=a_tensor.device, dtype=dtype)
        / magnitude_max
    )
    c = 1 - a_tensor - b_tensor
    return t * (a_tensor[..., None] + t * (b_tensor[..., None] + t * c[..., None]))


def cubic_carrier_levels(
    total_bits: int,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return nonnegative Cubic levels rounded to the INT8 carrier grid."""
    levels = cubic_levels(
        total_bits,
        a,
        b,
        device=device,
        dtype=torch.float32,
    )
    return torch.clamp(torch.round(levels * 127), 0, 127).to(torch.int8)


def cubic_w5_curve2_pair_lut(
    curve_a: torch.Tensor,
    curve_b: torch.Tensor,
) -> torch.Tensor:
    """Pack two W5 Curve2 INT8 carriers into each portable LUT entry."""
    levels = cubic_carrier_levels(5, curve_a, curve_b)
    raw = torch.arange(32, device=levels.device, dtype=torch.int64)
    signed = torch.where(raw >= 16, raw - 32, raw)
    signed = torch.where(signed == -16, 0, signed)
    carrier = levels[..., signed.abs()]
    carrier = torch.where(signed < 0, -carrier, carrier).to(torch.int32)
    pair = torch.arange(1024, device=levels.device, dtype=torch.int64)
    low = carrier[..., pair & 31] & 0xFF
    high = carrier[..., pair >> 5] & 0xFF
    return (low | (high << 8)).to(torch.int16).contiguous()


def _prepare_3bit_carrier_metadata(
    a: torch.nn.Parameter,
    b: torch.nn.Parameter,
) -> None:
    if a.dtype == torch.int8 and b.dtype == torch.int8:
        return
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Cubic 3-bit a/b metadata must be FP16 before conversion.")
    levels = cubic_carrier_levels(3, a, b)
    a.data = levels[..., 1].contiguous()
    b.data = levels[..., 2].contiguous()


def cubic_is_strictly_monotonic(a: float, b: float) -> bool:
    c = 1 - a - b
    points = [0.0, 1.0]
    if c != 0:
        vertex = -b / (3 * c)
        if 0 < vertex < 1:
            points.append(vertex)
    return min(a + 2 * b * t + 3 * c * t * t for t in points) > 0


def pack_cubic_codes(codes: torch.Tensor, total_bits: int) -> torch.Tensor:
    if total_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Unsupported Cubic bit width: {total_bits}.")
    if total_bits == 1:
        if torch.any((codes != -1) & (codes != 1)):
            raise ValueError("1-bit Cubic codes must be binary -1 or +1.")
    else:
        magnitude_max = (1 << (total_bits - 1)) - 1
        if torch.any((codes < -magnitude_max) | (codes > magnitude_max)):
            raise ValueError("Cubic codes contain an out-of-range or reserved value.")

    shape = codes.shape
    num_values = shape[-1]
    flat = codes.reshape(-1, num_values).to(torch.int64)
    raw = (
        (flat > 0).to(torch.int64)
        if total_bits == 1
        else flat & ((1 << total_bits) - 1)
    )
    num_bytes = math.ceil(num_values * total_bits / 8)
    packed = torch.zeros(
        flat.shape[0], num_bytes, dtype=torch.int64, device=codes.device
    )
    base = torch.arange(num_values, device=codes.device) * total_bits
    for bit in range(total_bits):
        positions = base + bit
        byte_indices = positions // 8
        shifts = positions % 8
        values = ((raw >> bit) & 1) << shifts
        packed.scatter_add_(1, byte_indices.expand(flat.shape[0], -1), values)
    return packed.to(torch.uint8).reshape(*shape[:-1], num_bytes)


def unpack_cubic_codes(
    packed: torch.Tensor,
    total_bits: int,
    num_values: int,
) -> torch.Tensor:
    if total_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Unsupported Cubic bit width: {total_bits}.")
    required_bytes = math.ceil(num_values * total_bits / 8)
    if packed.shape[-1] < required_bytes:
        raise ValueError(
            f"Packed Cubic tensor needs {required_bytes} bytes, got {packed.shape[-1]}."
        )

    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int64)
    positions = torch.arange(num_values, device=packed.device) * total_bits
    byte_indices = positions // 8
    shifts = positions % 8
    low = flat[:, byte_indices] >> shifts
    next_indices = torch.clamp(byte_indices + 1, max=flat.shape[1] - 1)
    high = flat[:, next_indices] << (8 - shifts)
    crosses_byte = shifts + total_bits > 8
    raw = torch.where(crosses_byte, low | high, low) & ((1 << total_bits) - 1)

    if total_bits == 1:
        codes = raw * 2 - 1
    else:
        sign_bit = 1 << (total_bits - 1)
        codes = torch.where(raw >= sign_bit, raw - (1 << total_bits), raw)
        codes = torch.where(codes == -sign_bit, 0, codes)
    return codes.to(torch.int8).reshape(*packed.shape[:-1], num_values)


def materialize_cubic_a8_carrier(
    packed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    input_size: int,
    group_out: int,
) -> torch.Tensor:
    """Materialize the exact INT8 carrier used by Dynamic-A8 Linear."""
    if packed.dtype != torch.uint8:
        raise ValueError("Cubic A8 carrier requires packed uint8 weights.")
    expected_bytes = math.ceil(input_size * num_bits / 8)
    if packed.shape[1] != expected_bytes:
        raise ValueError(
            "Cubic A8 carrier packed width does not match the logical input size."
        )
    if packed.shape[0] % group_out or a.shape[0] != packed.shape[0] // group_out:
        raise ValueError("Cubic A8 carrier metadata has an invalid output shape.")

    codes = unpack_cubic_codes(packed, num_bits, input_size).to(torch.float32)
    if num_bits == 1:
        carrier = (codes * 127.0).to(torch.int8)
        return carrier.T.contiguous().T

    input_groups = torch.arange(input_size, device=packed.device) // group_size
    output_groups = torch.arange(packed.shape[0], device=packed.device) // group_out
    coefficient_a = a[output_groups[:, None], input_groups[None, :]].to(torch.float32)
    coefficient_b = b[output_groups[:, None], input_groups[None, :]].to(torch.float32)
    magnitude_max = (1 << (num_bits - 1)) - 1
    t = codes.abs() / magnitude_max
    normalized = t * (
        coefficient_a + t * (coefficient_b + t * (1.0 - coefficient_a - coefficient_b))
    )
    carrier = torch.clamp(torch.round(codes.sign() * normalized * 127.0), -127, 127).to(
        torch.int8
    )
    return carrier.T.contiguous().T


def install_cubic_a8_carrier(
    layer: torch.nn.Module,
    scheme: CubicScheme,
) -> None:
    """Replace a legacy packed Linear weight with its runtime A8 carrier."""
    install_cubic_carrier(layer, scheme, dynamic_a8=True)


@dataclass(frozen=True)
class CubicMarlinWeight:
    weight: torch.Tensor
    scales: torch.Tensor
    input_global_scale: torch.Tensor | None
    workspace: torch.Tensor
    empty_int: torch.Tensor
    input_group_size: int | None = None

    @property
    def persistent_bytes(self) -> int:
        tensors = (
            self.scales,
            self.input_global_scale,
            self.workspace,
            self.empty_int,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )


@dataclass(frozen=True)
class CubicExactMarlinWeight:
    """Packed W4 codes and exact per-group levels for Cubic Marlin."""

    weight: torch.Tensor
    levels: torch.Tensor
    workspace: torch.Tensor

    @property
    def persistent_bytes(self) -> int:
        return self.weight.nbytes + self.levels.nbytes + self.workspace.nbytes


def prepare_cubic_exact_marlin_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    coefficient_a: torch.Tensor,
    coefficient_b: torch.Tensor,
    *,
    params_dtype: torch.dtype,
    num_bits: int,
    group_size: int,
    group_out: int,
    input_size: int,
) -> CubicExactMarlinWeight | None:
    """Prepare exact W4/G32 Cubic codes for the Marlin path."""
    if (
        num_bits != 4
        or group_size != 32
        or group_out != 1
        or params_dtype != torch.bfloat16
        or packed.dtype != torch.uint8
        or input_size % 128
        or packed.shape[0] % 128
    ):
        return None
    if packed.shape[1] != input_size // 2:
        return None

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )

    output_size = packed.shape[0]
    biased_packed = torch.bitwise_xor(packed, 0x88).contiguous()
    qweight = biased_packed.view(torch.int32).T.contiguous()
    empty_perm = torch.empty(0, dtype=torch.int32, device=packed.device)
    weight = ops.gptq_marlin_repack(
        qweight,
        empty_perm,
        input_size,
        output_size,
        4,
        is_a_8bit=False,
    )

    normalized = cubic_levels(
        4,
        coefficient_a,
        coefficient_b,
        dtype=torch.float32,
    )[..., 1:]
    logical_levels = (normalized * scale.float().unsqueeze(-1)).to(params_dtype)
    levels = torch.stack(
        [
            marlin_permute_scales(
                logical_levels[..., index].T.contiguous(),
                input_size,
                output_size,
                group_size,
            )
            for index in range(7)
        ]
    ).contiguous()
    return CubicExactMarlinWeight(
        weight=weight,
        levels=levels,
        workspace=marlin_make_workspace_new(packed.device),
    )


def apply_cubic_exact_marlin_weight(
    x: torch.Tensor,
    prepared: CubicExactMarlinWeight,
    *,
    output_size: int,
    input_size: int,
) -> torch.Tensor:
    """Execute one exact Cubic Marlin candidate."""
    import vllm._custom_ops as ops

    flattened = x.reshape(-1, input_size).contiguous()
    output = ops.cubic_marlin_gemm(
        flattened,
        prepared.weight,
        prepared.levels,
        prepared.workspace,
        flattened.shape[0],
        output_size,
        input_size,
    )
    return output.reshape(*x.shape[:-1], output_size)


def prepare_cubic_marlin_weight(
    carrier: torch.Tensor,
    scale: torch.Tensor,
    *,
    params_dtype: torch.dtype,
    group_size: int,
    group_out: int,
    dynamic_a8: bool,
    input_group_size: int | None = None,
) -> CubicMarlinWeight | None:
    """Prepare a tensor-core candidate without mutating the owning layer."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
        check_marlin_supports_shape,
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    if carrier.dtype != torch.int8 or carrier.ndim != 2:
        raise ValueError("Cubic Marlin preparation requires a 2-D INT8 carrier.")
    n, k = carrier.shape
    expected_scale_shape = (n // group_out, math.ceil(k / group_size))
    if scale.shape != expected_scale_shape:
        raise ValueError("Cubic Marlin scale shape does not match the carrier.")
    # Marlin requires every local K partition to contain complete input
    # groups. Compact Cubic remains the canonical path when a logical group
    # is wider than, or straddles, a TP-local partition.
    if group_size > k or k % group_size:
        return None
    execution_group_size = group_size
    execution_scale = scale
    if dynamic_a8 and input_group_size is not None:
        if k % input_group_size or input_group_size < 32:
            return None
        if input_group_size % group_size != 0 and group_size % input_group_size != 0:
            return None
        if input_group_size < group_size:
            execution_group_size = input_group_size
            execution_scale = scale.repeat_interleave(
                group_size // input_group_size, dim=1
            ).contiguous()
    shape_supported, _ = check_marlin_supports_shape(n, k, k, execution_group_size)
    if not (
        check_marlin_supported(scalar_types.uint8b128, execution_group_size)
        and shape_supported
    ):
        return None

    if dynamic_a8:
        qweight = carrier.contiguous().view(torch.int32).T.contiguous()
    else:
        biased = (carrier.to(torch.int16) + 128).to(torch.uint8).contiguous()
        qweight = biased.view(torch.int32).T.contiguous()
    empty_int = torch.empty(0, dtype=torch.int32, device=carrier.device)
    weight = ops.gptq_marlin_repack(
        qweight,
        empty_int,
        k,
        n,
        8,
        is_a_8bit=dynamic_a8,
    )
    scale_dtype = torch.float16 if dynamic_a8 else params_dtype
    scales = execution_scale.repeat_interleave(group_out, dim=0).T.contiguous().to(
        scale_dtype
    ) * (1.0 / 127.0)
    scales = marlin_permute_scales(
        scales,
        k,
        n,
        execution_group_size,
        is_a_8bit=dynamic_a8,
    )
    return CubicMarlinWeight(
        weight=weight,
        scales=scales,
        input_global_scale=None,
        workspace=marlin_make_workspace_new(carrier.device),
        empty_int=empty_int,
        input_group_size=(
            input_group_size if dynamic_a8 and input_group_size else None
        ),
    )


def apply_cubic_marlin_weight(
    x: torch.Tensor,
    prepared: CubicMarlinWeight,
    *,
    output_size: int,
    input_size: int,
    dynamic_a8: bool,
) -> torch.Tensor:
    """Execute one prepared Cubic Marlin candidate."""
    if dynamic_a8:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.cubic_kernels import (
            _quantize_cubic_groupwise_a8,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_pad_dim,
            marlin_repacked_nk,
            marlin_unpad_output,
            should_use_atomic_add_reduce,
        )
        from vllm.scalar_type import scalar_types

        flattened = x.reshape(-1, input_size)
        padded_n, padded_k = marlin_repacked_nk(prepared.weight, 8)
        flattened = marlin_pad_dim(flattened, input_size, padded_k)
        activation_group_size = prepared.input_group_size or padded_k
        activation = _quantize_cubic_groupwise_a8(
            flattened.contiguous(), activation_group_size
        )
        activation_scales = activation.scales
        if prepared.input_global_scale is not None:
            activation_scales = (
                activation_scales * prepared.input_global_scale
            )
        output_buffer = torch.empty(
            (flattened.shape[0], padded_n),
            dtype=x.dtype,
            device=x.device,
        )
        output = ops.marlin_gemm(
            activation.values,
            output_buffer,
            prepared.weight,
            None,
            prepared.scales,
            activation_scales,
            None,
            prepared.empty_int,
            prepared.empty_int,
            prepared.empty_int,
            prepared.workspace,
            scalar_types.int8,
            size_m=flattened.shape[0],
            size_n=padded_n,
            size_k=padded_k,
            is_k_full=True,
            use_atomic_add=should_use_atomic_add_reduce(
                m=flattened.shape[0],
                n=padded_n,
                k=padded_k,
                device=x.device,
                dtype=x.dtype,
            ),
            use_fp32_reduce=False,
            is_zp_float=False,
        )
        output = marlin_unpad_output(output, output_size, padded_n)
        return output.reshape(*x.shape[:-1], output_size)

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    return apply_gptq_marlin_linear(
        x,
        prepared.weight,
        prepared.scales,
        prepared.empty_int,
        prepared.empty_int,
        prepared.empty_int,
        prepared.workspace,
        scalar_types.int8 if dynamic_a8 else scalar_types.uint8b128,
        output_size,
        input_size,
        True,
        input_global_scale=prepared.input_global_scale,
        input_dtype=torch.int8 if dynamic_a8 else None,
        input_group_size=prepared.input_group_size,
    )


def install_cubic_carrier(
    layer: torch.nn.Module,
    scheme: CubicScheme,
    *,
    dynamic_a8: bool,
    backend: str = "triton",
) -> None:
    """Replace packed codes with an exact carrier and its best resident ABI."""
    representation = getattr(layer, "cubic_runtime_representation", "packed")
    if representation == "a8-carrier":
        return
    if representation != "packed":
        raise RuntimeError(
            "Cubic carrier installation requires canonical packed weights, got "
            f"{representation!r}."
        )
    scale, coefficient_a, coefficient_b = expanded_cubic_metadata(layer, scheme)
    if scheme.metadata_format in (
        CUBIC_COMPACT_METADATA_FORMAT,
        CUBIC_E5M9_CURVE2_METADATA_FORMAT,
    ):
        from vllm.model_executor.layers.quantization.cubic_kernels import (
            materialize_cubic_compact_a8_carrier,
        )

        curve2 = scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
        carrier = materialize_cubic_compact_a8_carrier(
            layer.weight_packed,
            layer.weight_metadata if curve2 else layer.weight_scale,
            layer.weight_curve_a if curve2 else layer.weight_ab,
            layer.weight_curve_b if curve2 else layer.weight_scale_global,
            layer.weight_curve_a if curve2 else layer.weight_a_global,
            layer.weight_curve_b if curve2 else layer.weight_b_global,
            layer.weight_global_index,
            num_bits=scheme.num_bits,
            group_size=scheme.group_size,
            input_size=layer.input_size_per_partition,
            group_out=scheme.group_out,
            e5m9_curve2_metadata=curve2,
        )
    else:
        carrier = materialize_cubic_a8_carrier(
            layer.weight_packed,
            coefficient_a,
            coefficient_b,
            num_bits=scheme.num_bits,
            group_size=scheme.group_size,
            input_size=layer.input_size_per_partition,
            group_out=scheme.group_out,
        )
    layer.weight_packed.data = carrier
    if hasattr(layer, "weight_scale"):
        layer.weight_scale.data = scale
    else:
        layer.register_buffer("weight_scale", scale)
    layer.cubic_weight_packed_is_carrier = True
    layer.cubic_runtime_representation = "a8-carrier"
    if backend == "marlin":
        _install_cubic_marlin_weight(layer, scheme, dynamic_a8=dynamic_a8)
    elif backend != "triton":
        raise ValueError(f"Unsupported Cubic carrier backend: {backend}.")


def materialize_cubic_a16_weight(
    layer: torch.nn.Module,
    scheme: CubicScheme,
) -> torch.Tensor:
    """Materialize the model-dtype weight represented by packed Cubic codes."""
    scale, coefficient_a, coefficient_b = expanded_cubic_metadata(layer, scheme)
    return dequantize_cubic(
        layer.weight_packed,
        scale,
        coefficient_a,
        coefficient_b,
        total_bits=scheme.num_bits,
        group_size=scheme.group_size,
        group_out=scheme.group_out,
        num_values=layer.input_size_per_partition,
        output_dtype=layer.params_dtype,
    )


def install_cubic_a16_weight(
    layer: torch.nn.Module,
    scheme: CubicScheme,
    *,
    retain_packed: bool = False,
    online_buckets: tuple[int, ...] = (),
) -> None:
    """Replace packed codes with the exact model-dtype A16 resident weight."""
    representation = getattr(layer, "cubic_runtime_representation", "packed")
    if representation == "a16-dense":
        return
    if representation != "packed":
        raise RuntimeError(
            "Cubic A16 installation requires canonical packed weights, got "
            f"{representation!r}."
        )
    expanded = materialize_cubic_a16_weight(layer, scheme)
    if retain_packed:
        layer.register_buffer(
            "cubic_a16_packed_weight",
            layer.weight_packed.detach().clone(),
            persistent=False,
        )
        layer.cubic_a16_online_buckets = online_buckets
    layer.weight_packed.data = expanded
    layer.cubic_weight_is_expanded_a16 = True
    layer.cubic_runtime_representation = "a16-dense"


def install_cubic_exact_marlin_weight(
    layer: torch.nn.Module,
    scheme: CubicScheme,
    *,
    token_buckets: tuple[int, ...],
) -> None:
    """Install an exact Cubic Marlin candidate beside packed weights."""
    scale, coefficient_a, coefficient_b = expanded_cubic_metadata(layer, scheme)
    prepared = prepare_cubic_exact_marlin_weight(
        layer.weight_packed,
        scale,
        coefficient_a,
        coefficient_b,
        params_dtype=layer.params_dtype,
        num_bits=scheme.num_bits,
        group_size=scheme.group_size,
        group_out=scheme.group_out,
        input_size=layer.input_size_per_partition,
    )
    if prepared is None:
        raise RuntimeError("Selected exact Cubic Marlin path is unsupported.")
    layer.register_buffer("cubic_exact_marlin_weight", prepared.weight)
    layer.register_buffer("cubic_exact_marlin_levels", prepared.levels)
    layer.register_buffer("cubic_exact_marlin_workspace", prepared.workspace)
    layer.cubic_exact_marlin_token_buckets = token_buckets


def _install_cubic_marlin_weight(
    layer: torch.nn.Module,
    scheme: CubicScheme,
    *,
    dynamic_a8: bool,
) -> None:
    """Repack a resident Cubic INT8 carrier for Marlin tensor cores."""
    representation = getattr(layer, "cubic_runtime_representation", "packed")
    if representation != "a8-carrier":
        raise RuntimeError(
            "Cubic Marlin installation requires an admitted A8 carrier, got "
            f"{representation!r}."
        )
    carrier = layer.weight_packed
    if carrier.dtype != torch.int8:
        raise ValueError("Cubic Marlin preparation requires an INT8 carrier.")
    prepared = prepare_cubic_marlin_weight(
        carrier,
        layer.weight_scale,
        params_dtype=layer.params_dtype,
        group_size=scheme.group_size,
        group_out=scheme.group_out,
        dynamic_a8=dynamic_a8,
        input_group_size=(
            cubic_dynamic_a8_group_size(
                input_size=layer.input_size_per_partition,
                weight_group_size=scheme.group_size,
            )
            if dynamic_a8
            else None
        ),
    )
    if prepared is None:
        return

    layer.weight_packed.data = prepared.weight
    layer.register_buffer("cubic_marlin_scales", prepared.scales)
    layer.register_buffer(
        "cubic_marlin_input_global_scale", prepared.input_global_scale
    )
    layer.register_buffer("cubic_marlin_workspace", prepared.workspace)
    layer.register_buffer("cubic_marlin_empty_int", prepared.empty_int)
    layer.cubic_weight_packed_is_carrier = False
    layer.cubic_weight_packed_is_marlin = True
    layer.cubic_runtime_representation = "marlin"
    layer.cubic_marlin_dynamic_a8 = dynamic_a8
    layer.cubic_marlin_input_group_size = prepared.input_group_size


def dequantize_cubic(
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    total_bits: int,
    group_size: int,
    group_out: int,
    num_values: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if scale.dtype != torch.float32:
        raise ValueError(f"Cubic scale must be FP32, got {scale.dtype}.")
    expected_groups = math.ceil(num_values / group_size)
    codes = unpack_cubic_codes(packed, total_bits, num_values)
    expected_shape = (
        (expected_groups,)
        if codes.ndim == 1
        else (
            *codes.shape[:-2],
            math.ceil(codes.shape[-2] / group_out),
            expected_groups,
        )
    )
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"Expected Cubic metadata shape {expected_shape}, got {tuple(scale.shape)}."
        )
    if a.shape != scale.shape or b.shape != scale.shape:
        raise ValueError("Cubic scale, a and b tensors must have identical shapes.")

    input_groups = torch.arange(num_values, device=packed.device) // group_size
    if codes.ndim == 1:
        scale_values = scale[..., input_groups]
        a_values = a[..., input_groups].to(torch.float32)
        b_values = b[..., input_groups].to(torch.float32)
    else:
        output_groups = torch.arange(codes.shape[-2], device=packed.device) // group_out
        metadata_indices = (output_groups[:, None], input_groups[None, :])
        scale_values = scale[..., metadata_indices[0], metadata_indices[1]]
        a_values = a[..., metadata_indices[0], metadata_indices[1]].to(torch.float32)
        b_values = b[..., metadata_indices[0], metadata_indices[1]].to(torch.float32)
    code_f32 = codes.to(torch.float32)
    if total_bits == 1:
        result = code_f32 * scale_values
    else:
        magnitude_max = (1 << (total_bits - 1)) - 1
        t = code_f32.abs() / magnitude_max
        c_values = 1 - a_values - b_values
        normalized = t * (a_values + t * (b_values + t * c_values))
        result = torch.sign(code_f32) * (scale_values * normalized)
    return result.to(output_dtype)


def dequantize_cubic_carrier(
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    total_bits: int,
    group_size: int,
    group_out: int,
    num_values: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode Cubic weights through the signed INT8 carrier representation."""
    if scale.dtype != torch.float32:
        raise ValueError(f"Cubic scale must be FP32, got {scale.dtype}.")
    expected_groups = math.ceil(num_values / group_size)
    codes = unpack_cubic_codes(packed, total_bits, num_values)
    expected_shape = (
        (expected_groups,)
        if codes.ndim == 1
        else (
            *codes.shape[:-2],
            math.ceil(codes.shape[-2] / group_out),
            expected_groups,
        )
    )
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"Expected Cubic metadata shape {expected_shape}, got {tuple(scale.shape)}."
        )
    if a.shape != scale.shape or b.shape != scale.shape:
        raise ValueError("Cubic scale, a and b tensors must have identical shapes.")

    input_groups = torch.arange(num_values, device=packed.device) // group_size
    if codes.ndim == 1:
        scale_values = scale[..., input_groups]
        a_values = a[..., input_groups].to(torch.float32)
        b_values = b[..., input_groups].to(torch.float32)
    else:
        output_groups = torch.arange(codes.shape[-2], device=packed.device) // group_out
        metadata_indices = (output_groups[:, None], input_groups[None, :])
        scale_values = scale[..., metadata_indices[0], metadata_indices[1]]
        a_values = a[..., metadata_indices[0], metadata_indices[1]].to(torch.float32)
        b_values = b[..., metadata_indices[0], metadata_indices[1]].to(torch.float32)
    code_f32 = codes.to(torch.float32)
    if total_bits == 1:
        carrier = code_f32 * 127
    else:
        magnitude_max = (1 << (total_bits - 1)) - 1
        t = code_f32.abs() / magnitude_max
        normalized = t * (a_values + t * (b_values + t * (1 - a_values - b_values)))
        carrier = torch.clamp(
            torch.sign(code_f32) * torch.round(normalized * 127),
            -127,
            127,
        )
    return (scale_values * carrier / 127).to(output_dtype)


def _candidate_pairs() -> list[tuple[float, float]]:
    pairs = [(1.0, 0.0)]
    for a in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        for b in (-0.75, -0.25, 0.0, 0.25, 0.75):
            if (a, b) not in pairs and cubic_is_strictly_monotonic(a, b):
                pairs.append((a, b))
    return pairs


def _fit_scale(
    values: torch.Tensor,
    valid: torch.Tensor,
    levels: torch.Tensor,
    start_scale: torch.Tensor,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_values = values.abs()
    scale = start_scale.clamp_min(torch.finfo(torch.float32).tiny)
    indices = torch.zeros_like(values, dtype=torch.long)
    for _ in range(iterations):
        candidates = scale[:, None, None] * levels[None, None, :]
        distances = (abs_values[..., None] - candidates).abs()
        indices = distances.argmin(dim=-1)
        q = levels[indices] * valid
        numerator = (abs_values * q).sum(dim=-1)
        denominator = q.square().sum(dim=-1)
        updated = torch.where(denominator > 0, numerator / denominator, scale)
        scale = updated.clamp_min(torch.finfo(torch.float32).tiny)
    q = levels[indices] * valid
    reconstructed = values.sign() * scale[:, None] * q
    loss = ((values - reconstructed).square() * valid).sum(dim=-1)
    return scale, indices, loss


def quantize_cubic(
    weight: torch.Tensor,
    *,
    total_bits: int = 4,
    group_size: int = 128,
) -> CubicQuantizedTensor:
    scheme = CubicScheme(
        num_bits=total_bits,
        group_size=group_size,
        group_out=1,
        reserved_code="binary" if total_bits == 1 else "zero",
    )
    original_shape = tuple(weight.shape)
    num_values = original_shape[-1]
    num_groups = math.ceil(num_values / group_size)
    padded_values = num_groups * group_size
    values = F.pad(weight.to(torch.float32), (0, padded_values - num_values))
    values = values.reshape(-1, group_size)
    valid = torch.ones_like(values)
    if padded_values != num_values:
        valid.reshape(*original_shape[:-1], num_groups, group_size)[
            ..., -1, -(padded_values - num_values) :
        ] = 0

    group_amax = (
        (values.abs() * valid).amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    )
    best_loss = torch.full_like(group_amax, torch.inf)
    best_scale = group_amax.clone()
    best_a = torch.ones_like(group_amax)
    best_b = torch.zeros_like(group_amax)

    if total_bits == 1:
        denominator = valid.sum(dim=-1).clamp_min(1)
        best_scale = ((values.abs() * valid).sum(dim=-1) / denominator).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        best_loss = (
            (values - values.sign() * best_scale[:, None]).square() * valid
        ).sum(dim=-1)
    else:
        for a_value, b_value in _candidate_pairs():
            levels = cubic_levels(
                total_bits,
                a_value,
                b_value,
                device=weight.device,
                dtype=torch.float32,
            )
            for multiplier in (0.65, 0.8, 1.0, 1.15):
                scale, _, loss = _fit_scale(
                    values, valid, levels, group_amax * multiplier
                )
                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                best_scale = torch.where(improved, scale, best_scale)
                best_a = torch.where(improved, a_value, best_a)
                best_b = torch.where(improved, b_value, best_b)

    stored_scale = best_scale.to(torch.float32)
    stored_a = best_a.to(scheme.param_dtype)
    stored_b = best_b.to(scheme.param_dtype)
    if total_bits == 1:
        codes = torch.where(values < 0, -1, 1).to(torch.int64)
    else:
        levels = cubic_levels(
            total_bits,
            stored_a,
            stored_b,
            device=weight.device,
            dtype=torch.float32,
        )
        reconstructed_levels = stored_scale[:, None] * levels
        distances = (values.abs()[..., None] - reconstructed_levels[:, None, :]).abs()
        magnitudes = distances.argmin(dim=-1)
        codes = values.sign().to(torch.int64) * magnitudes
    codes = (codes * valid.to(torch.int64)).reshape(
        *original_shape[:-1], padded_values
    )[..., :num_values]

    packed = pack_cubic_codes(codes, total_bits)
    metadata_shape = (*original_shape[:-1], num_groups)
    scale = stored_scale.reshape(metadata_shape)
    a = stored_a.reshape(metadata_shape)
    b = stored_b.reshape(metadata_shape)
    reference = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=total_bits,
        group_size=group_size,
        group_out=1,
        num_values=num_values,
    )
    loss = (weight.to(torch.float32) - reference).square().sum(dim=-1)
    return CubicQuantizedTensor(
        packed=packed,
        scale=scale,
        a=a,
        b=b,
        shape=original_shape,
        num_bits=total_bits,
        group_size=group_size,
        group_out=1,
        loss=loss,
    )


def _match_score(
    prefix: str, layer: torch.nn.Module, targets: list[str]
) -> tuple[int, int, int, int] | None:
    score = None
    for target in targets:
        if target == prefix:
            candidate = (3, 0, 0, 0)
        elif target.startswith("re:") and _regex_matches_prefix(target[3:], prefix):
            pattern = target[3:]
            candidate = (
                2,
                int(pattern.startswith("^")),
                -pattern.count(".*"),
                len(pattern),
            )
        elif (
            target == layer.__class__.__name__
            or (target == "Linear" and isinstance(layer, LinearBase))
            or (
                target in ("FusedMoE", "RoutedExperts")
                and isinstance(layer, RoutedExperts)
            )
        ):
            candidate = (1, 0, 0, 0)
        else:
            continue
        score = candidate if score is None else max(score, candidate)
    return score


def _regex_matches_prefix(
    pattern: str, prefix: str, *, match_subtree: bool = False
) -> bool:
    """Match module prefixes both at and below the model root."""
    matcher = re.match if match_subtree else re.fullmatch
    return (
        matcher(pattern, prefix) is not None
        or matcher(pattern, f".{prefix}") is not None
    )


def _is_ignored(prefix: str, ignore: list[str]) -> bool:
    return any(
        item == prefix
        or (
            item.startswith("re:")
            and _regex_matches_prefix(item[3:], prefix, match_subtree=True)
        )
        for item in ignore
    )


class CubicConfig(QuantizationConfig):
    def __init__(
        self,
        schemes: list[tuple[list[str], CubicScheme]],
        ignore: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.schemes = schemes
        self.ignore = ignore or []

    def get_name(self) -> QuantizationMethods:
        return "cubic"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CubicConfig":
        if config.get("quant_method") != "cubic":
            raise ValueError("Cubic config requires quant_method='cubic'.")
        if config.get("format") != CUBIC_FORMAT:
            raise ValueError(f"Cubic config requires format={CUBIC_FORMAT!r}.")
        groups = config.get("config_groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Cubic config requires at least one config group.")

        schemes = []
        for group in groups.values():
            weights = group.get("weights") or {}
            metadata_format = weights.get(
                "metadata_format", CUBIC_LEGACY_METADATA_FORMAT
            )
            scale_dtype = weights.get("scale_dtype", "float32")
            expected_scale_dtypes = (
                ("float32", "torch.float32")
                if metadata_format == CUBIC_LEGACY_METADATA_FORMAT
                else (
                    ("packed_e5m9", "uint16", "torch.uint16")
                    if metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                    else ("int8", "torch.int8")
                )
            )
            if scale_dtype not in expected_scale_dtypes:
                raise ValueError(
                    f"Cubic {metadata_format} requires scale_dtype in "
                    f"{expected_scale_dtypes}."
                )
            param_dtype = weights.get("param_dtype", "float16")
            expected_param_dtypes = (
                ("float16", "torch.float16")
                if metadata_format == CUBIC_LEGACY_METADATA_FORMAT
                else (
                    ("float16", "torch.float16", "curve2_tensor_fp16")
                    if metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                    else ("packed_int4",)
                )
            )
            if param_dtype not in expected_param_dtypes:
                raise ValueError(
                    f"Cubic {metadata_format} requires param_dtype in "
                    f"{expected_param_dtypes}."
                )
            if (
                metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                and weights.get("curve_count", 4) != 4
            ):
                raise ValueError("Cubic E5M9/curve2 metadata requires 4 curves.")
            num_bits = int(weights["num_bits"])
            group_out, group_in = _normalize_group_size(weights["group_size"])
            scheme = CubicScheme(
                num_bits=num_bits,
                group_size=group_in,
                group_out=group_out,
                param_dtype=torch.float16,
                metadata_format=metadata_format,
                reserved_code=weights.get(
                    "reserved_code", "binary" if num_bits == 1 else "zero"
                ),
            )
            targets = group.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError("Each Cubic config group requires targets.")
            schemes.append((targets, scheme))
        return cls(schemes, list(config.get("ignore", [])))

    def _scheme_for(self, layer: torch.nn.Module, prefix: str) -> CubicScheme | None:
        if _is_ignored(prefix, self.ignore):
            return None
        matched = [
            (score, scheme)
            for targets, scheme in self.schemes
            if (score := _match_score(prefix, layer, targets))
        ]
        if not matched:
            return None
        best_score = max(score for score, _ in matched)
        best = [scheme for score, scheme in matched if score == best_score]
        if len(best) > 1:
            raise ValueError(f"Layer {prefix!r} matches multiple Cubic config groups.")
        return best[0]

    def has_explicit_scheme(self, prefix: str) -> bool:
        """Return whether an exact or regex target explicitly selects a prefix."""
        if _is_ignored(prefix, self.ignore):
            return False
        return any(
            target == prefix
            or (
                target.startswith("re:")
                and _regex_matches_prefix(target[3:], prefix)
            )
            for targets, _ in self.schemes
            for target in targets
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | FusedMoEMethodBase | None:
        scheme = self._scheme_for(layer, prefix)
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            return CubicEmbeddingMethod(
                scheme,
                dynamic_a8=envs.VLLM_CUBIC_DYNAMIC_A8,
            ) if scheme else None
        if isinstance(layer, LinearBase):
            return (
                CubicLinearMethod(
                    scheme,
                    dynamic_a8=envs.VLLM_CUBIC_DYNAMIC_A8,
                )
                if scheme
                else UnquantizedLinearMethod()
            )
        if isinstance(layer, RoutedExperts):
            if scheme and scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
                raise ValueError("INT8/INT4 compact metadata is unsupported by MoE.")
            return (
                CubicMoEMethod(
                    scheme,
                    layer.moe_config,
                    dynamic_a8=envs.VLLM_CUBIC_DYNAMIC_A8,
                )
                if scheme
                else UnquantizedFusedMoEMethod(layer.moe_config)
            )
        return None


@register_weight_loader_v2_supported_method
class CubicLinearMethod(LinearMethodBase):
    def __init__(self, scheme: CubicScheme, *, dynamic_a8: bool = False) -> None:
        self.scheme = scheme
        self.dynamic_a8 = dynamic_a8

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight_loader = extra_weight_attrs["weight_loader"]
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.scheme.group_out:
            raise ValueError(
                "Cubic linear output size must be divisible by group_out="
                f"{self.scheme.group_out}."
            )
        tp_size = getattr(layer, "tp_size", 1)
        if (
            tp_size > 1
            and input_size_per_partition % self.scheme.group_size
            and self.scheme.group_size % input_size_per_partition
        ):
            raise ValueError(
                "Cubic linear input groups must align with TP shard boundaries; "
                f"local input size {input_size_per_partition} and group_in "
                f"{self.scheme.group_size} do not divide each other."
            )
        packed_input = math.ceil(input_size_per_partition * self.scheme.num_bits / 8)
        num_groups = math.ceil(input_size_per_partition / self.scheme.group_size)
        output_groups = output_size_per_partition // self.scheme.group_out
        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            packed_factor=Fraction(8, self.scheme.num_bits),
            packed_dim=1,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition, packed_input, dtype=torch.uint8
            ),
        )
        metadata_args = {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
            "packed_dim": 0,
            "packed_factor": self.scheme.group_out,
            "input_size_per_partition": input_size_per_partition,
            "input_group_size": self.scheme.group_size,
        }
        layer.register_parameter("weight_packed", weight)
        if self.scheme.metadata_format == CUBIC_LEGACY_METADATA_FORMAT:
            scale = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.float32),
                **metadata_args,
            )
            a = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.float16),
                **metadata_args,
            )
            b = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.float16),
                **metadata_args,
            )
            layer.register_parameter("weight_scale", scale)
            layer.register_parameter("weight_a", a)
            layer.register_parameter("weight_b", b)
        elif self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
            scale = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.int8),
                **metadata_args,
            )
            ab = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.uint8),
                **metadata_args,
            )
            layer.register_parameter("weight_scale", scale)
            layer.register_parameter("weight_ab", ab)
            for name in ("scale", "a", "b"):
                global_parameter = CubicLogicalScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                layer.register_parameter(f"weight_{name}_global", global_parameter)
        else:
            metadata = CubicLinearMetadataParameter(
                data=torch.empty(output_groups, num_groups, dtype=torch.uint16),
                **metadata_args,
            )
            layer.register_parameter("weight_metadata", metadata)
            for name in ("a", "b"):
                curve = CubicLogicalScaleParameter(
                    data=torch.empty(
                        len(output_partition_sizes), 4, dtype=torch.float16
                    ),
                    weight_loader=weight_loader,
                )
                layer.register_parameter(f"weight_curve_{name}", curve)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        layer.cubic_runtime_representation = "packed"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expected_bytes = math.ceil(
            layer.input_size_per_partition * self.scheme.num_bits / 8
        )
        if layer.weight_packed.dtype != torch.uint8:
            raise ValueError("Cubic packed linear weight must remain uint8.")
        if layer.weight_packed.shape[-1] != expected_bytes:
            raise ValueError("Cubic packed linear weight has an invalid width.")
        expected_groups = math.ceil(
            layer.input_size_per_partition / self.scheme.group_size
        )
        expected_shape = (
            layer.output_size_per_partition // self.scheme.group_out,
            expected_groups,
        )
        metadata = (
            layer.weight_metadata
            if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
            else layer.weight_scale
        )
        if tuple(metadata.shape) != expected_shape:
            raise ValueError("Cubic linear metadata has an invalid group count.")
        if self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
            if layer.weight_scale.dtype != torch.int8:
                raise ValueError("Compact Cubic scale codes must remain INT8.")
            if layer.weight_ab.dtype != torch.uint8:
                raise ValueError("Compact Cubic packed a/b must remain uint8.")
        elif self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
            if layer.weight_metadata.dtype != torch.uint16:
                raise ValueError("Cubic E5M9 metadata must remain uint16.")
            expected_curves = (len(layer.output_partition_sizes), 4)
            for curves in (layer.weight_curve_a, layer.weight_curve_b):
                if curves.dtype != torch.float16:
                    raise ValueError("Cubic curve tables must remain FP16.")
                if tuple(curves.shape) != expected_curves:
                    raise ValueError("Cubic curve tables have an invalid shape.")
        elif layer.weight_scale.dtype != torch.float32:
            raise ValueError("Cubic linear scale must remain FP32 at runtime.")
        if self.scheme.metadata_format != CUBIC_LEGACY_METADATA_FORMAT:
            group_counts = torch.tensor(
                layer.output_partition_sizes,
                device=metadata.device,
                dtype=torch.int64,
            ).div(self.scheme.group_out, rounding_mode="floor")
            global_index = torch.repeat_interleave(
                torch.arange(
                    len(group_counts), device=metadata.device, dtype=torch.uint8
                ),
                group_counts,
            )
            layer.register_buffer("weight_global_index", global_index)
            if (
                self.dynamic_a8
                and self.scheme.num_bits == 5
                and self.scheme.metadata_format
                == CUBIC_E5M9_CURVE2_METADATA_FORMAT
            ):
                layer.register_buffer(
                    "weight_curve_pair_lut",
                    cubic_w5_curve2_pair_lut(
                        layer.weight_curve_a,
                        layer.weight_curve_b,
                    ),
                )
            return
        if (
            layer.weight_a.dtype != torch.float16
            or layer.weight_b.dtype != torch.float16
        ):
            raise ValueError("Cubic linear a/b must remain FP16 at runtime.")

    def dequantize(self, layer: torch.nn.Module) -> torch.Tensor:
        """Materialize the Linear weight for an operator requiring a vector."""
        cached_weight = getattr(layer, "_cubic_operator_weight", None)
        if cached_weight is not None:
            return cached_weight
        if getattr(layer, "cubic_weight_is_expanded_a16", False):
            return layer.weight_packed
        if getattr(layer, "cubic_weight_packed_is_carrier", False):
            scale = layer.weight_scale.repeat_interleave(
                self.scheme.group_out, dim=0
            ).repeat_interleave(self.scheme.group_size, dim=1)
            weight = (
                layer.weight_packed.float()
                * scale[:, : layer.input_size_per_partition]
                * (1.0 / 127.0)
            ).to(layer.params_dtype)
            layer._cubic_operator_weight = weight
            return weight
        scale, a, b = expanded_cubic_metadata(layer, self.scheme)
        weight = dequantize_cubic(
            layer.weight_packed,
            scale,
            a,
            b,
            total_bits=self.scheme.num_bits,
            group_size=self.scheme.group_size,
            group_out=self.scheme.group_out,
            num_values=layer.input_size_per_partition,
            output_dtype=layer.params_dtype,
        )
        layer._cubic_operator_weight = weight
        return weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.cubic_kernels import (
            cubic_linear,
            cubic_linear_compact,
            cubic_linear_curve2,
            cubic_linear_dynamic_a8,
            cubic_linear_dynamic_a8_compact,
            cubic_linear_dynamic_a8_curve2,
            cubic_linear_dynamic_a8_precomputed,
            cubic_linear_dynamic_a8_w5_curve2_pair_lut,
        )

        representation = getattr(layer, "cubic_runtime_representation", "packed")
        state_flags = {
            "a16-dense": getattr(layer, "cubic_weight_is_expanded_a16", False),
            "a8-carrier": getattr(layer, "cubic_weight_packed_is_carrier", False),
            "marlin": getattr(layer, "cubic_weight_packed_is_marlin", False),
        }
        if representation != "packed" and not state_flags.get(representation, False):
            raise RuntimeError(
                "Cubic runtime representation and compatibility flags disagree: "
                f"{representation!r}."
            )

        if (
            not envs.VLLM_BATCH_INVARIANT
            and hasattr(layer, "cubic_exact_marlin_weight")
            and not self.dynamic_a8
        ):
            from vllm.model_executor.layers.quantization.cubic_policy import (
                cubic_linear_token_bucket,
            )

            tokens = x.numel() // x.shape[-1]
            bucket = cubic_linear_token_bucket(tokens)
            if bucket in layer.cubic_exact_marlin_token_buckets:
                prepared = CubicExactMarlinWeight(
                    weight=layer.cubic_exact_marlin_weight,
                    levels=layer.cubic_exact_marlin_levels,
                    workspace=layer.cubic_exact_marlin_workspace,
                )
                output = apply_cubic_exact_marlin_weight(
                    x,
                    prepared,
                    output_size=layer.output_size_per_partition,
                    input_size=layer.input_size_per_partition,
                )
                return output if bias is None else output + bias

        if representation == "a16-dense":
            packed = getattr(layer, "cubic_a16_packed_weight", None)
            if packed is not None:
                from vllm.model_executor.layers.quantization.cubic_kernels import (
                    cubic_linear_token_bucket,
                )

                tokens = x.numel() // x.shape[-1]
                bucket = cubic_linear_token_bucket(tokens)
                if bucket in layer.cubic_a16_online_buckets:
                    if getattr(layer, "cubic_metadata_is_expanded", False):
                        output = cubic_linear(
                            x,
                            packed,
                            layer.cubic_weight_scale_expanded,
                            layer.cubic_weight_a_expanded,
                            layer.cubic_weight_b_expanded,
                            num_bits=self.scheme.num_bits,
                            group_size=self.scheme.group_size,
                            group_out=self.scheme.group_out,
                            input_size=layer.input_size_per_partition,
                        )
                    elif (
                        self.scheme.metadata_format
                        == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                    ):
                        output = cubic_linear_curve2(
                            x,
                            packed,
                            layer.weight_metadata,
                            layer.weight_curve_a,
                            layer.weight_curve_b,
                            layer.weight_global_index,
                            num_bits=self.scheme.num_bits,
                            group_size=self.scheme.group_size,
                            group_out=self.scheme.group_out,
                            input_size=layer.input_size_per_partition,
                        )
                    elif self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
                        output = cubic_linear_compact(
                            x,
                            packed,
                            layer.weight_scale,
                            layer.weight_ab,
                            layer.weight_scale_global,
                            layer.weight_a_global,
                            layer.weight_b_global,
                            layer.weight_global_index,
                            num_bits=self.scheme.num_bits,
                            group_size=self.scheme.group_size,
                            group_out=self.scheme.group_out,
                            input_size=layer.input_size_per_partition,
                        )
                    else:
                        output = cubic_linear(
                            x,
                            packed,
                            layer.weight_scale,
                            layer.weight_a,
                            layer.weight_b,
                            num_bits=self.scheme.num_bits,
                            group_size=self.scheme.group_size,
                            group_out=self.scheme.group_out,
                            input_size=layer.input_size_per_partition,
                        )
                    return output if bias is None else output + bias
            if envs.VLLM_BATCH_INVARIANT:
                from vllm.model_executor.layers.batch_invariant import (
                    linear_batch_invariant,
                )

                return linear_batch_invariant(x, layer.weight_packed, bias)
            return torch.nn.functional.linear(x, layer.weight_packed, bias)

        if representation == "marlin":
            if layer.cubic_marlin_dynamic_a8 != self.dynamic_a8:
                raise RuntimeError(
                    "Cubic Marlin activation mode differs from the loaded method."
                )
            prepared = CubicMarlinWeight(
                weight=layer.weight_packed,
                scales=layer.cubic_marlin_scales,
                input_global_scale=layer.cubic_marlin_input_global_scale,
                workspace=layer.cubic_marlin_workspace,
                empty_int=layer.cubic_marlin_empty_int,
                input_group_size=layer.cubic_marlin_input_group_size,
            )
            output = apply_cubic_marlin_weight(
                x,
                prepared,
                output_size=layer.output_size_per_partition,
                input_size=layer.input_size_per_partition,
                dynamic_a8=self.dynamic_a8,
            )
            return output if bias is None else output + bias

        carrier = (
            layer.weight_packed
            if representation == "a8-carrier"
            else getattr(layer, "weight_carrier", None)
        )
        if self.dynamic_a8 and carrier is not None:
            unused_curve = (
                layer.weight_ab
                if self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT
                else (
                    layer.weight_metadata
                    if self.scheme.metadata_format
                    == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                    else layer.weight_a
                )
            )
            if torch.compiler.is_compiling():
                output = torch.ops.vllm.cubic_linear_dynamic_a8_precomputed(
                    x,
                    carrier,
                    layer.weight_scale,
                    unused_curve,
                    unused_curve,
                    self.scheme.num_bits,
                    self.scheme.group_size,
                    self.scheme.group_out,
                    layer.input_size_per_partition,
                )
            else:
                output = cubic_linear_dynamic_a8_precomputed(
                    x,
                    carrier,
                    layer.weight_scale,
                    unused_curve,
                    unused_curve,
                    num_bits=self.scheme.num_bits,
                    group_size=self.scheme.group_size,
                    group_out=self.scheme.group_out,
                    input_size=layer.input_size_per_partition,
                )
            return output if bias is None else output + bias

        if (
            not self.dynamic_a8
            and getattr(layer, "cubic_metadata_is_expanded", False)
        ):
            token_buckets = layer.cubic_expanded_metadata_token_buckets
            use_expanded_metadata = token_buckets is None
            if not use_expanded_metadata:
                from vllm.model_executor.layers.quantization.cubic_policy import (
                    cubic_linear_token_bucket,
                )

                tokens = x.numel() // x.shape[-1]
                use_expanded_metadata = (
                    cubic_linear_token_bucket(tokens) in token_buckets
                )
            if use_expanded_metadata:
                expanded_args = (
                    x,
                    layer.weight_packed,
                    layer.cubic_weight_scale_expanded,
                    layer.cubic_weight_a_expanded,
                    layer.cubic_weight_b_expanded,
                    self.scheme.num_bits,
                    self.scheme.group_size,
                    self.scheme.group_out,
                    layer.input_size_per_partition,
                )
                if torch.compiler.is_compiling():
                    output = torch.ops.vllm.cubic_linear_expanded_metadata(
                        *expanded_args
                    )
                else:
                    output = cubic_linear(
                        *expanded_args[:5],
                        num_bits=expanded_args[5],
                        group_size=expanded_args[6],
                        group_out=expanded_args[7],
                        input_size=expanded_args[8],
                    )
                return output if bias is None else output + bias

        if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
            curve_args = (
                x,
                layer.weight_packed,
                layer.weight_metadata,
                layer.weight_curve_a,
                layer.weight_curve_b,
                layer.weight_global_index,
                self.scheme.num_bits,
                self.scheme.group_size,
                self.scheme.group_out,
                layer.input_size_per_partition,
            )
            if self.dynamic_a8:
                pair_lut = getattr(layer, "weight_curve_pair_lut", None)
                if pair_lut is not None and x.numel() // x.shape[-1] <= 8:
                    pair_args = (
                        x,
                        layer.weight_packed,
                        layer.weight_metadata,
                        layer.weight_global_index,
                        pair_lut,
                        self.scheme.group_size,
                        self.scheme.group_out,
                        layer.input_size_per_partition,
                    )
                    if torch.compiler.is_compiling():
                        cubic_ops = torch.ops.vllm
                        pair_op = (
                            cubic_ops.cubic_linear_dynamic_a8_w5_curve2_pair_lut
                        )
                        output = pair_op(*pair_args)
                    else:
                        output = cubic_linear_dynamic_a8_w5_curve2_pair_lut(
                            *pair_args[:5],
                            group_size=pair_args[5],
                            group_out=pair_args[6],
                            input_size=pair_args[7],
                        )
                    return output if bias is None else output + bias
                if torch.compiler.is_compiling():
                    output = torch.ops.vllm.cubic_linear_dynamic_a8_curve2(
                        *curve_args
                    )
                else:
                    output = cubic_linear_dynamic_a8_curve2(
                        *curve_args[:6],
                        num_bits=curve_args[6],
                        group_size=curve_args[7],
                        group_out=curve_args[8],
                        input_size=curve_args[9],
                    )
            elif torch.compiler.is_compiling():
                output = torch.ops.vllm.cubic_linear_curve2(*curve_args)
            else:
                output = cubic_linear_curve2(
                    *curve_args[:6],
                    num_bits=curve_args[6],
                    group_size=curve_args[7],
                    group_out=curve_args[8],
                    input_size=curve_args[9],
                )
            return output if bias is None else output + bias

        if self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
            if getattr(layer, "cubic_metadata_is_expanded", False):
                expanded_args = (
                    x,
                    layer.weight_packed,
                    layer.cubic_weight_scale_expanded,
                    layer.cubic_weight_a_expanded,
                    layer.cubic_weight_b_expanded,
                    self.scheme.num_bits,
                    self.scheme.group_size,
                    self.scheme.group_out,
                    layer.input_size_per_partition,
                )
                if self.dynamic_a8:
                    if torch.compiler.is_compiling():
                        expanded_op = (
                            torch.ops.vllm.cubic_linear_dynamic_a8_expanded_metadata
                        )
                        output = expanded_op(*expanded_args)
                    else:
                        output = cubic_linear_dynamic_a8_compact(
                            *expanded_args[:2],
                            *expanded_args[2:5],
                            layer.weight_a_global,
                            layer.weight_b_global,
                            layer.weight_global_index,
                            num_bits=expanded_args[5],
                            group_size=expanded_args[6],
                            group_out=expanded_args[7],
                            input_size=expanded_args[8],
                            _expanded_metadata=True,
                        )
                elif torch.compiler.is_compiling():
                    output = torch.ops.vllm.cubic_linear_expanded_metadata(
                        *expanded_args
                    )
                else:
                    output = cubic_linear_compact(
                        *expanded_args[:2],
                        *expanded_args[2:5],
                        layer.weight_a_global,
                        layer.weight_b_global,
                        layer.weight_global_index,
                        num_bits=expanded_args[5],
                        group_size=expanded_args[6],
                        group_out=expanded_args[7],
                        input_size=expanded_args[8],
                        _expanded_metadata=True,
                    )
                return output if bias is None else output + bias
            compact_args = (
                x,
                layer.weight_packed,
                layer.weight_scale,
                layer.weight_ab,
                layer.weight_scale_global,
                layer.weight_a_global,
                layer.weight_b_global,
                layer.weight_global_index,
                self.scheme.num_bits,
                self.scheme.group_size,
                self.scheme.group_out,
                layer.input_size_per_partition,
            )
            if self.dynamic_a8:
                if torch.compiler.is_compiling():
                    output = torch.ops.vllm.cubic_linear_dynamic_a8_compact(
                        *compact_args
                    )
                else:
                    output = cubic_linear_dynamic_a8_compact(
                        *compact_args[:8],
                        num_bits=compact_args[8],
                        group_size=compact_args[9],
                        group_out=compact_args[10],
                        input_size=compact_args[11],
                    )
            elif torch.compiler.is_compiling():
                output = torch.ops.vllm.cubic_linear_compact(*compact_args)
            else:
                output = cubic_linear_compact(
                    *compact_args[:8],
                    num_bits=compact_args[8],
                    group_size=compact_args[9],
                    group_out=compact_args[10],
                    input_size=compact_args[11],
                )
            return output if bias is None else output + bias

        if self.dynamic_a8:
            if torch.compiler.is_compiling():
                output = torch.ops.vllm.cubic_linear_dynamic_a8(
                    x,
                    layer.weight_packed,
                    layer.weight_scale,
                    layer.weight_a,
                    layer.weight_b,
                    self.scheme.num_bits,
                    self.scheme.group_size,
                    self.scheme.group_out,
                    layer.input_size_per_partition,
                )
            else:
                output = cubic_linear_dynamic_a8(
                    x,
                    layer.weight_packed,
                    layer.weight_scale,
                    layer.weight_a,
                    layer.weight_b,
                    num_bits=self.scheme.num_bits,
                    group_size=self.scheme.group_size,
                    group_out=self.scheme.group_out,
                    input_size=layer.input_size_per_partition,
                )
            return output if bias is None else output + bias

        if torch.compiler.is_compiling():
            output = torch.ops.vllm.cubic_linear(
                x,
                layer.weight_packed,
                layer.weight_scale,
                layer.weight_a,
                layer.weight_b,
                self.scheme.num_bits,
                self.scheme.group_size,
                self.scheme.group_out,
                layer.input_size_per_partition,
            )
        else:
            output = cubic_linear(
                x,
                layer.weight_packed,
                layer.weight_scale,
                layer.weight_a,
                layer.weight_b,
                num_bits=self.scheme.num_bits,
                group_size=self.scheme.group_size,
                group_out=self.scheme.group_out,
                input_size=layer.input_size_per_partition,
            )
        return output if bias is None else output + bias


class CubicEmbeddingMethod(CubicLinearMethod):
    """Cubic weight method for embedding lookup and parallel LM heads."""

    def embedding(
        self, layer: torch.nn.Module, input_: torch.Tensor
    ) -> torch.Tensor:
        if self.scheme.group_out != 1:
            raise ValueError("Cubic embedding lookup requires group_out=1.")
        if getattr(layer, "cubic_weight_is_expanded_a16", False):
            return torch.nn.functional.embedding(input_, layer.weight_packed)
        ids = input_.reshape(-1).long().contiguous()
        if self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
            from vllm.model_executor.layers.quantization.cubic_kernels import (
                cubic_embedding_compact,
            )

            return cubic_embedding_compact(
                ids,
                layer.weight_packed,
                layer.weight_scale,
                layer.weight_ab,
                layer.weight_scale_global,
                layer.weight_a_global,
                layer.weight_b_global,
                num_bits=self.scheme.num_bits,
                group_size=self.scheme.group_size,
                input_size=layer.input_size_per_partition,
                output_dtype=layer.params_dtype,
            ).reshape(*input_.shape, layer.input_size_per_partition)
        if (
            self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT
            and layer.weight_curve_a.shape == (1, 4)
        ):
            from vllm.model_executor.layers.quantization.cubic_kernels import (
                cubic_embedding_e5m9_curve2,
            )

            return cubic_embedding_e5m9_curve2(
                ids,
                layer.weight_packed,
                layer.weight_metadata,
                layer.weight_curve_a,
                layer.weight_curve_b,
                num_bits=self.scheme.num_bits,
                group_size=self.scheme.group_size,
                input_size=layer.input_size_per_partition,
                output_dtype=layer.params_dtype,
            ).reshape(*input_.shape, layer.input_size_per_partition)
        valid_ids = (ids >= 0) & (ids < layer.weight_packed.shape[0])
        safe_ids = torch.where(valid_ids, ids, 0)
        packed = layer.weight_packed.index_select(0, safe_ids)
        if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
            metadata = layer.weight_metadata.index_select(0, safe_ids)
            scale, a, b = decode_cubic_e5m9_curve2_metadata(
                metadata,
                layer.weight_curve_a,
                layer.weight_curve_b,
                [ids.numel()],
                self.scheme.group_out,
            )
        elif self.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
            scale, a, b = decode_cubic_compact_metadata(
                layer.weight_scale.index_select(0, safe_ids),
                layer.weight_ab.index_select(0, safe_ids),
                layer.weight_scale_global,
                layer.weight_a_global,
                layer.weight_b_global,
                [ids.numel()],
                self.scheme.group_out,
            )
        else:
            scale = layer.weight_scale.index_select(0, safe_ids)
            a = layer.weight_a.index_select(0, safe_ids)
            b = layer.weight_b.index_select(0, safe_ids)
        output = dequantize_cubic(
            packed,
            scale,
            a,
            b,
            total_bits=self.scheme.num_bits,
            group_size=self.scheme.group_size,
            group_out=self.scheme.group_out,
            num_values=layer.input_size_per_partition,
            output_dtype=layer.params_dtype,
        )
        output.masked_fill_(~valid_ids[:, None], 0)
        return output.reshape(*input_.shape, layer.input_size_per_partition)

    def tie_weights(
        self, layer: torch.nn.Module, embed_tokens: torch.nn.Module
    ) -> torch.nn.Module:
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_a",
            "weight_b",
            "weight_ab",
            "weight_scale_global",
            "weight_a_global",
            "weight_b_global",
            "weight_metadata",
            "weight_curve_a",
            "weight_curve_b",
        ):
            if hasattr(embed_tokens, name):
                setattr(layer, name, getattr(embed_tokens, name))
        return layer


class CubicMoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        scheme: CubicScheme,
        moe: Any,
        *,
        dynamic_a8: bool = False,
    ) -> None:
        super().__init__(moe)
        self.scheme = scheme
        self.dynamic_a8 = dynamic_a8

    @property
    def supports_eplb(self) -> bool:
        return True

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        group_size = self.scheme.group_size
        group_out = self.scheme.group_out
        if hidden_size % group_size or intermediate_size_per_partition % group_size:
            raise ValueError(
                "Cubic fused MoE requires hidden and per-partition intermediate "
                f"sizes to be divisible by group_size={group_size}."
            )
        w13_output = (
            2 * intermediate_size_per_partition
            if self.moe.is_act_and_mul
            else intermediate_size_per_partition
        )
        if w13_output % group_out or hidden_size % group_out:
            raise ValueError(
                "Cubic fused MoE output sizes must be divisible by "
                f"group_out={group_out}."
            )
        attrs = dict(extra_weight_attrs)
        attrs.update({"is_transposed": False, "quant_method": "group"})
        weight_loader = attrs["weight_loader"]
        loaded_shards: dict[str, torch.Tensor] = {}

        def register(
            name: str,
            data: torch.Tensor,
            expected_shards: tuple[str, ...],
        ) -> None:
            marker = torch.zeros(
                (num_experts, len(expected_shards)),
                dtype=torch.bool,
                device="cpu",
            )
            loaded_shards[name] = marker

            def tracked_weight_loader(
                param: torch.nn.Parameter,
                loaded_weight: torch.Tensor,
                weight_name: str,
                shard_id: str,
                expert_id: int,
                return_success: bool = False,
            ) -> bool | None:
                success = weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    local_expert_id = layer._map_global_expert_id_to_local_expert_id(
                        expert_id
                    )
                    marker[local_expert_id, expected_shards.index(shard_id)] = True
                return success if return_success else None

            param = torch.nn.Parameter(data, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(
                param,
                {**attrs, "weight_loader": tracked_weight_loader},
            )

        def register_curves(
            name: str,
            logical_shards: tuple[str, ...],
        ) -> None:
            marker = torch.zeros(
                (num_experts, len(logical_shards)), dtype=torch.bool, device="cpu"
            )
            loaded_shards[name] = marker

            def curve_weight_loader(
                param: torch.nn.Parameter,
                loaded_weight: torch.Tensor,
                weight_name: str,
                shard_id: str,
                expert_id: int,
                return_success: bool = False,
            ) -> bool | None:
                del weight_name
                local_expert_id = layer._map_global_expert_id_to_local_expert_id(
                    expert_id
                )
                if local_expert_id == -1:
                    return False if return_success else None
                shard = logical_shards.index(shard_id)
                loaded_weight = loaded_weight.reshape(-1)
                if loaded_weight.shape != (4,):
                    raise ValueError("Cubic MoE curve tables must contain 4 values.")
                param.data[local_expert_id, shard].copy_(loaded_weight)
                marker[local_expert_id, shard] = True
                return True if return_success else None

            param = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    len(logical_shards),
                    4,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(
                param,
                {**attrs, "weight_loader": curve_weight_loader},
            )

        register(
            "w13_weight_packed",
            torch.empty(
                num_experts,
                w13_output,
                math.ceil(hidden_size * self.scheme.num_bits / 8),
                dtype=torch.uint8,
            ),
            ("w1", "w3"),
        )
        register(
            "w2_weight_packed",
            torch.empty(
                num_experts,
                hidden_size,
                math.ceil(intermediate_size_per_partition * self.scheme.num_bits / 8),
                dtype=torch.uint8,
            ),
            ("w2",),
        )
        for prefix, output_size, input_size in (
            ("w13", w13_output, hidden_size),
            ("w2", hidden_size, intermediate_size_per_partition),
        ):
            metadata_shape = (
                num_experts,
                output_size // group_out,
                input_size // group_size,
            )
            logical_shards = ("w1", "w3") if prefix == "w13" else ("w2",)
            if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
                register(
                    f"{prefix}_weight_metadata",
                    torch.empty(metadata_shape, dtype=torch.uint16),
                    logical_shards,
                )
                register_curves(f"{prefix}_weight_curve_a", logical_shards)
                register_curves(f"{prefix}_weight_curve_b", logical_shards)
            else:
                register(
                    f"{prefix}_weight_scale",
                    torch.empty(metadata_shape, dtype=torch.float32),
                    logical_shards,
                )
                register(
                    f"{prefix}_weight_a",
                    torch.empty(metadata_shape, dtype=torch.float16),
                    logical_shards,
                )
                register(
                    f"{prefix}_weight_b",
                    torch.empty(metadata_shape, dtype=torch.float16),
                    logical_shards,
                )
        layer.cubic_hidden_size = hidden_size
        layer.cubic_intermediate_size = intermediate_size_per_partition
        layer.cubic_weight_loader = weight_loader
        layer.cubic_loaded_shards = loaded_shards
        layer.cubic_fused_checkpoint_layout = True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name, marker in layer.cubic_loaded_shards.items():
            if not marker.all():
                missing = marker.logical_not().nonzero().tolist()
                raise ValueError(f"Cubic {name} has unloaded expert shards {missing}.")
        if layer.w13_weight_packed.dtype != torch.uint8:
            raise ValueError("Cubic packed w13 weight must remain uint8.")
        if layer.w2_weight_packed.dtype != torch.uint8:
            raise ValueError("Cubic packed w2 weight must remain uint8.")
        expected_w13_bytes = math.ceil(
            layer.cubic_hidden_size * self.scheme.num_bits / 8
        )
        expected_w2_bytes = math.ceil(
            layer.cubic_intermediate_size * self.scheme.num_bits / 8
        )
        if layer.w13_weight_packed.shape[-1] != expected_w13_bytes:
            raise ValueError("Cubic packed w13 weight has an invalid width.")
        if layer.w2_weight_packed.shape[-1] != expected_w2_bytes:
            raise ValueError("Cubic packed w2 weight has an invalid width.")
        expected_w13_metadata = (
            layer.w13_weight_packed.shape[0],
            layer.w13_weight_packed.shape[1] // self.scheme.group_out,
            layer.cubic_hidden_size // self.scheme.group_size,
        )
        expected_w2_metadata = (
            layer.w2_weight_packed.shape[0],
            layer.w2_weight_packed.shape[1] // self.scheme.group_out,
            layer.cubic_intermediate_size // self.scheme.group_size,
        )
        if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
            w13_metadata = layer.w13_weight_metadata
            w2_metadata = layer.w2_weight_metadata
        else:
            w13_metadata = layer.w13_weight_scale
            w2_metadata = layer.w2_weight_scale
        if (
            tuple(w13_metadata.shape) != expected_w13_metadata
            or tuple(w2_metadata.shape) != expected_w2_metadata
        ):
            raise ValueError("Cubic fused MoE metadata has an invalid group count.")
        if self.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
            w13_scale, w13_a, w13_b = _decode_cubic_moe_e5m9_curve2_metadata(
                layer.w13_weight_metadata,
                layer.w13_weight_curve_a,
                layer.w13_weight_curve_b,
                (
                    layer.cubic_intermediate_size,
                    layer.cubic_intermediate_size,
                ),
                self.scheme.group_out,
            )
            w2_scale, w2_a, w2_b = _decode_cubic_moe_e5m9_curve2_metadata(
                layer.w2_weight_metadata,
                layer.w2_weight_curve_a,
                layer.w2_weight_curve_b,
                (layer.cubic_hidden_size,),
                self.scheme.group_out,
            )
            for name, value in (
                ("w13_weight_scale", w13_scale),
                ("w13_weight_a", w13_a),
                ("w13_weight_b", w13_b),
                ("w2_weight_scale", w2_scale),
                ("w2_weight_a", w2_a),
                ("w2_weight_b", w2_b),
            ):
                layer.register_buffer(name, value)
        if layer.w13_weight_scale.dtype != torch.float32:
            raise ValueError("Cubic fused MoE scale must remain FP32 at runtime.")
        if layer.w2_weight_scale.dtype != torch.float32:
            raise ValueError("Cubic fused MoE scale must remain FP32 at runtime.")
        metadata_pairs = (
            (layer.w13_weight_a, layer.w13_weight_b),
            (layer.w2_weight_a, layer.w2_weight_b),
        )
        if self.dynamic_a8 and self.scheme.num_bits == 3:
            for a, b in metadata_pairs:
                _prepare_3bit_carrier_metadata(a, b)
        else:
            for a, b in metadata_pairs:
                if a.dtype != torch.float16 or b.dtype != torch.float16:
                    raise ValueError("Cubic fused MoE a/b must remain FP16.")

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> None:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.cubic_kernels import (
            cubic_fused_moe,
            cubic_fused_moe_dynamic_a8,
        )

        args = (
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_weight_a,
            layer.w13_weight_b,
            layer.w2_weight_a,
            layer.w2_weight_b,
            topk_weights,
            topk_ids,
        )
        if torch.compiler.is_compiling():
            op = (
                torch.ops.vllm.cubic_fused_moe_dynamic_a8
                if self.dynamic_a8
                else torch.ops.vllm.cubic_fused_moe
            )
            return op(
                *args,
                layer.expert_map,
                layer.activation.value,
                layer.apply_router_weight_on_input,
                layer.global_num_experts,
                self.scheme.num_bits,
                self.scheme.group_size,
                self.scheme.group_out,
                layer.cubic_hidden_size,
                layer.cubic_intermediate_size,
                self.moe.activation_situ_beta,
                self.moe.activation_situ_linear_beta,
            )
        func = cubic_fused_moe_dynamic_a8 if self.dynamic_a8 else cubic_fused_moe
        return func(
            *args,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            num_bits=self.scheme.num_bits,
            group_size=self.scheme.group_size,
            group_out=self.scheme.group_out,
            hidden_size=layer.cubic_hidden_size,
            intermediate_size=layer.cubic_intermediate_size,
            activation_situ_beta=self.moe.activation_situ_beta,
            activation_situ_linear_beta=self.moe.activation_situ_linear_beta,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cubic fused MoE uses external routing.")
