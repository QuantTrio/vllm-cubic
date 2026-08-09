# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import secrets

import pytest

from vllm.engine.arg_utils import EngineArgs


def test_engine_seed_is_random_by_default(monkeypatch: pytest.MonkeyPatch):
    """Keep the fork's launch-random seed policy across upstream syncs."""
    values = iter([0, 41])
    monkeypatch.setattr(secrets, "randbelow", lambda _: next(values))

    assert EngineArgs().seed == 1
    assert EngineArgs().seed == 42


@pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
def test_engine_seed_preserves_explicit_value(seed: int):
    assert EngineArgs(seed=seed).seed == seed
