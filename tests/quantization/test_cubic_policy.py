# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.cubic_policy import (
    CUBIC_SUPPORTED_BITS,
    CUBIC_TOKEN_BUCKETS,
    CubicActivationMode,
    CubicExecutionKind,
    CubicReconstructionKind,
    cubic_linear_token_bucket,
    cubic_reconstruction_kind,
    cubic_token_bucket,
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
