# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
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


def test_deterministic_inference_configures_inductor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)

    args = EngineArgs(deterministic_inference=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
    assert args.compilation_config.inductor_compile_config["combo_kernels"] is False
    assert (
        args.compilation_config.inductor_compile_config["benchmark_combo_kernel"]
        is False
    )
