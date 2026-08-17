# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.cubic_policy import (
    CUBIC_SUPPORTED_BITS,
    CUBIC_TOKEN_BUCKETS,
    CubicActivationMode,
    CubicCarrierResidency,
    CubicExecutionKind,
    CubicMetadataResidency,
    CubicReconstructionKind,
    CubicRuntimeCandidate,
    cubic_linear_residency_budget,
    cubic_linear_token_bucket,
    cubic_reconstruction_kind,
    cubic_runtime_memory,
    cubic_token_bucket,
    select_cubic_runtime_candidate,
)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(1, 1), (3, 4), (8, 8), (9, 16), (257, 512), (8192, 8192), (65536, 8192)],
)
def test_cubic_token_bucket_only_selects_policy(num_tokens: int, expected: int) -> None:
    assert cubic_token_bucket(num_tokens) == expected


def test_cubic_token_buckets_are_finite_and_ordered() -> None:
    assert tuple(sorted(set(CUBIC_TOKEN_BUCKETS))) == CUBIC_TOKEN_BUCKETS
    assert {cubic_token_bucket(num_tokens) for num_tokens in range(1, 65537)} == set(
        CUBIC_TOKEN_BUCKETS
    )


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(1, 1), (2, 1), (4, 1), (8, 1), (9, 16), (17, 32)],
)
def test_cubic_linear_low_m_uses_one_tactic_identity(
    num_tokens: int, expected: int
) -> None:
    assert cubic_linear_token_bucket(num_tokens) == expected


def test_cubic_supported_bits_cover_the_complete_format() -> None:
    assert tuple(range(1, 9)) == CUBIC_SUPPORTED_BITS


@pytest.mark.parametrize("execution", list(CubicExecutionKind))
@pytest.mark.parametrize("activation", list(CubicActivationMode))
@pytest.mark.parametrize("num_bits", range(1, 9))
def test_reconstruction_policy_covers_all_execution_modes(
    execution: CubicExecutionKind,
    activation: CubicActivationMode,
    num_bits: int,
) -> None:
    del execution, activation
    expected = (
        CubicReconstructionKind.ALIGNED
        if 8 % num_bits == 0
        else CubicReconstructionKind.STREAM
    )
    assert cubic_reconstruction_kind(num_bits) is expected


@pytest.mark.parametrize("num_tokens", [0, -1])
def test_cubic_token_bucket_rejects_invalid_shapes(num_tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        cubic_token_bucket(num_tokens)


@pytest.mark.parametrize("num_bits", [0, 9])
def test_cubic_reconstruction_policy_rejects_invalid_bits(num_bits: int) -> None:
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        cubic_reconstruction_kind(num_bits)


def _candidate(
    name: str,
    latency_ms: float,
    extra_memory_bytes: int,
    *,
    expanded: bool = False,
    carrier: bool = False,
) -> CubicRuntimeCandidate:
    return CubicRuntimeCandidate(
        name=name,
        latency_ms=latency_ms,
        extra_memory_bytes=extra_memory_bytes,
        metadata=(
            CubicMetadataResidency.EXPANDED
            if expanded
            else CubicMetadataResidency.COMPACT
        ),
        carrier=(
            CubicCarrierResidency.PRECOMPUTED
            if carrier
            else CubicCarrierResidency.ONLINE
        ),
    )


def test_runtime_policy_prefers_compact_path_when_performance_is_near_equal() -> None:
    compact = _candidate("compact", 1.02, 0)
    expanded = _candidate("expanded", 1.0, 1024, expanded=True)

    selected = select_cubic_runtime_candidate(
        (compact, expanded), extra_memory_budget_bytes=2048
    )

    assert selected is compact


def test_runtime_policy_spends_budget_for_material_speedup() -> None:
    compact = _candidate("compact", 1.2, 0)
    expanded = _candidate("expanded", 1.0, 1024, expanded=True)

    selected = select_cubic_runtime_candidate(
        (compact, expanded), extra_memory_budget_bytes=2048
    )

    assert selected is expanded


def test_runtime_policy_never_selects_an_unbudgeted_carrier() -> None:
    compact = _candidate("compact", 1.2, 0)
    carrier = _candidate("carrier", 0.5, 4096, carrier=True)

    selected = select_cubic_runtime_candidate(
        (compact, carrier), extra_memory_budget_bytes=1024
    )

    assert selected is compact


@pytest.mark.parametrize("num_bits", range(1, 9))
def test_runtime_memory_accounts_for_every_cubic_bit_width(num_bits: int) -> None:
    memory = cubic_runtime_memory(
        num_values=257,
        num_groups=17,
        num_bits=num_bits,
        logical_tensors=2,
        output_groups=5,
    )

    assert memory.packed_weight_bytes == (257 * num_bits + 7) // 8
    assert memory.compact_metadata_bytes == 2 * 17 + 2 * 12 + 5
    assert memory.expanded_metadata_bytes == 8 * 17
    assert memory.carrier_bytes == 257
    assert memory.carrier_replacement_extra_bytes == (
        257 - (257 * num_bits + 7) // 8
    )


def test_w8_carrier_can_replace_packed_storage_without_growth() -> None:
    memory = cubic_runtime_memory(
        num_values=4096,
        num_groups=32,
        num_bits=8,
    )

    assert memory.packed_weight_bytes == memory.carrier_bytes
    assert memory.carrier_replacement_extra_bytes == 0


@pytest.mark.parametrize(
    ("total_memory_bytes", "free_memory_bytes", "expected"),
    [
        (80 << 30, 64 << 30, 16 << 30),
        (24 << 30, 12 << 30, 3 << 30),
        (24 << 30, 24 << 30, 6 << 30),
        (0, 0, 0),
    ],
)
def test_linear_residency_budget_preserves_capacity_and_headroom(
    total_memory_bytes: int,
    free_memory_bytes: int,
    expected: int,
) -> None:
    assert (
        cubic_linear_residency_budget(
            total_memory_bytes=total_memory_bytes,
            free_memory_bytes=free_memory_bytes,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("total_memory_bytes", "free_memory_bytes"),
    [(-1, 0), (1, -1), (1, 2)],
)
def test_linear_residency_budget_rejects_invalid_memory(
    total_memory_bytes: int,
    free_memory_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        cubic_linear_residency_budget(
            total_memory_bytes=total_memory_bytes,
            free_memory_bytes=free_memory_bytes,
        )
