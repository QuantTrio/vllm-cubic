# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


class CubicExecutionKind(Enum):
    LINEAR = "linear"
    ROUTED_MOE = "routed_moe"


class CubicActivationMode(Enum):
    A16 = "a16"
    A8 = "a8"


class CubicReconstructionKind(Enum):
    ALIGNED = "aligned"
    STREAM = "stream"


def cubic_token_bucket(num_tokens: int) -> int:
    """Return the policy bucket without changing the real tensor shape."""
    if num_tokens < 1:
        raise ValueError(f"Cubic token count must be positive, got {num_tokens}.")
    return next(
        (bucket for bucket in CUBIC_TOKEN_BUCKETS if num_tokens <= bucket),
        CUBIC_TOKEN_BUCKETS[-1],
    )


def cubic_reconstruction_kind(num_bits: int) -> CubicReconstructionKind:
    """Classify packed supply independently of execution and activation mode."""
    if num_bits not in CUBIC_SUPPORTED_BITS:
        raise ValueError(f"Cubic bit width must be in [1, 8], got {num_bits}.")
    if num_bits in CUBIC_ALIGNED_BITS:
        return CubicReconstructionKind.ALIGNED
    return CubicReconstructionKind.STREAM
