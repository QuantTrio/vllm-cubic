# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

CUBIC_SUPPORTED_BITS = tuple(range(1, 9))
CUBIC_TOKEN_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)
CUBIC_ALIGNED_BITS = (1, 2, 4, 8)
CUBIC_DYNAMIC_A8_GROUP_CANDIDATES = (32, 64, 128)
_CUBIC_LINEAR_RESIDENCY_TOTAL_MEMORY_DIVISOR = 4
_CUBIC_LINEAR_RESIDENCY_FREE_MEMORY_DIVISOR = 2


class CubicExecutionKind(Enum):
    LINEAR = "linear"
    ROUTED_MOE = "routed_moe"


class CubicActivationMode(Enum):
    A16 = "a16"
    A8 = "a8"


class CubicReconstructionKind(Enum):
    ALIGNED = "aligned"
    STREAM = "stream"


class CubicMetadataResidency(Enum):
    COMPACT = "compact"
    EXPANDED = "expanded"


class CubicCarrierResidency(Enum):
    ONLINE = "online"
    PRECOMPUTED = "precomputed"
    EXPANDED = "expanded"


@dataclass(frozen=True)
class CubicRuntimeCandidate:
    """One measured execution path and its persistent memory requirement."""

    name: str
    latency_ms: float
    extra_memory_bytes: int
    metadata: CubicMetadataResidency
    carrier: CubicCarrierResidency

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Cubic runtime candidate requires a name.")
        if self.latency_ms <= 0:
            raise ValueError("Cubic runtime latency must be positive.")
        if self.extra_memory_bytes < 0:
            raise ValueError("Cubic runtime extra memory cannot be negative.")


@dataclass(frozen=True)
class CubicRuntimeMemory:
    packed_weight_bytes: int
    compact_metadata_bytes: int
    expanded_metadata_bytes: int
    carrier_bytes: int
    carrier_replacement_extra_bytes: int


def cubic_runtime_memory(
    *,
    num_values: int,
    num_groups: int,
    num_bits: int,
    logical_tensors: int = 1,
    output_groups: int = 0,
) -> CubicRuntimeMemory:
    """Account for persistent Cubic representations before selecting a path."""
    if num_values < 1 or num_groups < 1 or logical_tensors < 1:
        raise ValueError("Cubic runtime memory dimensions must be positive.")
    if num_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Cubic bit width must be in [1, 8], got {num_bits}.")
    if output_groups < 0:
        raise ValueError("Cubic output group count cannot be negative.")
    packed_weight_bytes = (num_values * num_bits + 7) // 8
    carrier_bytes = num_values
    return CubicRuntimeMemory(
        packed_weight_bytes=packed_weight_bytes,
        compact_metadata_bytes=(2 * num_groups + 12 * logical_tensors + output_groups),
        expanded_metadata_bytes=8 * num_groups,
        carrier_bytes=carrier_bytes,
        carrier_replacement_extra_bytes=max(0, carrier_bytes - packed_weight_bytes),
    )


def select_cubic_runtime_candidate(
    candidates: tuple[CubicRuntimeCandidate, ...],
    *,
    extra_memory_budget_bytes: int,
    compact_preference_margin: float = 0.03,
) -> CubicRuntimeCandidate:
    """Select a measured path without hiding persistent memory expansion."""
    if extra_memory_budget_bytes < 0:
        raise ValueError("Cubic runtime memory budget cannot be negative.")
    if not 0 <= compact_preference_margin < 1:
        raise ValueError("Cubic compact preference margin must be in [0, 1).")
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.extra_memory_bytes <= extra_memory_budget_bytes
    )
    if not eligible:
        raise ValueError("No Cubic runtime candidate fits the memory budget.")
    fastest_latency = min(candidate.latency_ms for candidate in eligible)
    near_fastest = tuple(
        candidate
        for candidate in eligible
        if candidate.latency_ms <= fastest_latency * (1.0 + compact_preference_margin)
    )
    return min(
        near_fastest,
        key=lambda candidate: (
            candidate.extra_memory_bytes,
            candidate.latency_ms,
            candidate.name,
        ),
    )


def cubic_linear_residency_budget(
    *, total_memory_bytes: int, free_memory_bytes: int
) -> int:
    """Bound persistent Linear acceleration by capacity and free headroom."""
    if total_memory_bytes < 0 or free_memory_bytes < 0:
        raise ValueError("Cubic runtime memory sizes cannot be negative.")
    if free_memory_bytes > total_memory_bytes:
        raise ValueError("Free device memory cannot exceed total device memory.")
    return min(
        total_memory_bytes // _CUBIC_LINEAR_RESIDENCY_TOTAL_MEMORY_DIVISOR,
        free_memory_bytes // _CUBIC_LINEAR_RESIDENCY_FREE_MEMORY_DIVISOR,
    )


def cubic_token_bucket(num_tokens: int) -> int:
    """Return the policy bucket without changing the real tensor shape."""
    if num_tokens < 1:
        raise ValueError(f"Cubic token count must be positive, got {num_tokens}.")
    return next(
        (bucket for bucket in CUBIC_TOKEN_BUCKETS if num_tokens <= bucket),
        CUBIC_TOKEN_BUCKETS[-1],
    )


def cubic_linear_token_bucket(num_tokens: int) -> int:
    """Return the measured tactic bucket for the real Linear row count."""
    return cubic_token_bucket(num_tokens)


def cubic_dynamic_a8_group_size(*, input_size: int, weight_group_size: int) -> int:
    """Choose the finest efficiently aligned activation group.

    Activation and weight groups are compatible when either partition refines
    the other.  Their product is then segmented by the smaller group without
    introducing a partial or shifted boundary.
    """
    if input_size < 1 or weight_group_size < 1:
        raise ValueError("Cubic input and weight group sizes must be positive.")
    for candidate in CUBIC_DYNAMIC_A8_GROUP_CANDIDATES:
        if input_size % candidate:
            continue
        if candidate % weight_group_size == 0 or weight_group_size % candidate == 0:
            return candidate
    return input_size


def cubic_reconstruction_kind(num_bits: int) -> CubicReconstructionKind:
    """Classify packed supply independently of execution and activation mode."""
    if num_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Cubic bit width must be in [1, 8], got {num_bits}.")
    if num_bits in CUBIC_ALIGNED_BITS:
        return CubicReconstructionKind.ALIGNED
    return CubicReconstructionKind.STREAM
