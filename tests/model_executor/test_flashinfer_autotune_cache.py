# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.model_executor.warmup import flashinfer_autotune_cache


def test_resolve_flashinfer_autotune_file_separates_ranks(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "flashinfer_autotune_cache_hash",
        lambda runner: "config-hash",
    )

    shared = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(object())
    rank_3 = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(
        object(), rank=3
    )

    assert shared == tmp_path / "config-hash" / "autotune_configs.json"
    assert rank_3 == tmp_path / "config-hash" / "autotune_configs_rank_3.json"
