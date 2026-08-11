# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.warmup import cubic_warmup
from vllm.model_executor.warmup.cubic_warmup import _assign_cubic_tasks


def test_cubic_w2_situ_specs_keep_the_normalized_output_group():
    from vllm.model_executor.layers.quantization.cubic import (
        CubicMoEMethod,
        CubicScheme,
    )

    layer = torch.nn.Module()
    method = object.__new__(CubicMoEMethod)
    method.scheme = CubicScheme(num_bits=2, group_size=512, group_out=128)
    layer.quant_method = method
    layer.cubic_intermediate_size = 3072
    layer.cubic_hidden_size = 3584
    layer.top_k = 16
    layer.w13_weight_packed = torch.empty(112, 0, 0, device="meta", dtype=torch.uint8)

    assert cubic_warmup._cubic_w2_a8_situ_specs(layer) == (
        (3072, 3584, 128, 512, 16, 112),
    )


def _linear_task(bits: int) -> tuple[object, ...]:
    return (
        "linear",
        bits,
        256,
        1,
        3072,
        7168,
        "torch.float16",
        "torch.float16",
        16,
    )


def _metadata(
    rank: int,
    *,
    fingerprint: tuple[object, ...] = ("cuda", "sm90"),
    cache_domain: tuple[object, ...] = ("cache", "/shared", 0, 1),
    tasks: tuple[tuple[object, ...], ...] = (),
    cache_hit: bool = False,
) -> dict[str, object]:
    return {
        "rank": rank,
        "fingerprint": fingerprint,
        "cache_domain": cache_domain,
        "tasks": tasks,
        "cache_hit": cache_hit,
    }


def test_cubic_calibration_is_distributed_once_per_device_and_cache_domain():
    tasks = (_linear_task(2), _linear_task(3))
    assignments, pending = _assign_cubic_tasks(
        [_metadata(0, tasks=tasks), _metadata(1, tasks=tasks)]
    )

    assert pending == len(tasks)
    assert assignments[0].isdisjoint(assignments[1])
    assert assignments[0] | assignments[1] == set(tasks)


def test_cubic_calibration_repeats_for_heterogeneous_devices():
    task = _linear_task(2)
    assignments, pending = _assign_cubic_tasks(
        [
            _metadata(0, fingerprint=("cuda", "sm90"), tasks=(task,)),
            _metadata(1, fingerprint=("cuda", "sm80"), tasks=(task,)),
        ]
    )

    assert pending == 2
    assert assignments == {0: {task}, 1: {task}}


def test_cubic_calibration_repeats_for_node_local_cache_domains():
    task = _linear_task(2)
    assignments, pending = _assign_cubic_tasks(
        [
            _metadata(0, cache_domain=("host", "node-a"), tasks=(task,)),
            _metadata(1, cache_domain=("host", "node-b"), tasks=(task,)),
        ]
    )

    assert pending == 2
    assert assignments == {0: {task}, 1: {task}}


def test_cubic_shared_cache_hit_skips_duplicate_calibration():
    task = _linear_task(2)
    assignments, pending = _assign_cubic_tasks(
        [
            _metadata(0, tasks=(task,), cache_hit=True),
            _metadata(1, tasks=(task,)),
        ]
    )

    assert pending == 0
    assert assignments == {0: set(), 1: set()}


def test_cubic_warmup_does_not_initialize_distributed_state_without_cubic_layers(
    monkeypatch,
):
    monkeypatch.setattr(cubic_warmup.envs, "VLLM_CUBIC_AUTOTUNE", True)
    monkeypatch.setattr(cubic_warmup, "_cubic_calibration_tasks", lambda *_: ())

    def unexpected_world_initialization():
        raise AssertionError("non-Cubic models must not initialize Cubic warmup")

    monkeypatch.setattr(cubic_warmup, "_cubic_world", unexpected_world_initialization)

    cubic_warmup.cubic_kernel_warmup(
        torch.nn.Identity(), max_tokens=16, capture_sizes=(1, 8)
    )


def test_cubic_warmup_materializes_merged_tactics_on_every_rank(monkeypatch):
    task = _linear_task(2)
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(cubic_warmup.envs, "VLLM_CUBIC_AUTOTUNE", True)
    monkeypatch.setattr(cubic_warmup, "_cubic_calibration_tasks", lambda *_: (task,))
    monkeypatch.setattr(cubic_warmup, "_cubic_world", lambda: (0, 1, None))
    monkeypatch.setattr(
        cubic_warmup, "_cubic_device_fingerprint", lambda: ("cuda", "sm120")
    )
    monkeypatch.setattr(cubic_warmup, "_cubic_tactic_cache_key", lambda *_: "key")
    monkeypatch.setattr(cubic_warmup, "_load_cubic_tactic_cache", lambda *_: True)
    monkeypatch.setattr(
        cubic_warmup, "_cubic_cache_domain", lambda *_: ("cache", "/shared")
    )
    monkeypatch.setattr(
        cubic_warmup, "_cubic_all_gather", lambda payload, *_: [payload]
    )
    monkeypatch.setattr(cubic_warmup, "_export_cubic_tactics", lambda: {})
    monkeypatch.setattr(cubic_warmup, "_merge_cubic_tactics", lambda *_: None)
    monkeypatch.setattr(cubic_warmup, "_cubic_w2_a8_situ_specs", lambda *_: ())
    monkeypatch.setattr(cubic_warmup, "_cubic_barrier", lambda *_: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    def record(family):
        def warmup(*_args, **kwargs):
            calls.append((family, kwargs.get("calibrate", True)))

        return warmup

    monkeypatch.setattr(cubic_warmup, "_warmup_cubic_linear_families", record("linear"))
    monkeypatch.setattr(cubic_warmup, "_warmup_cubic_moe_families", record("moe"))

    cubic_warmup.cubic_kernel_warmup(
        torch.nn.Identity(), max_tokens=16, capture_sizes=(1, 8)
    )

    assert calls == [
        ("linear", True),
        ("moe", True),
        ("linear", False),
        ("moe", False),
    ]


def test_cubic_calibration_keeps_large_linear_buckets_beyond_capture_sizes():
    buckets = cubic_warmup._calibration_token_buckets(
        8192,
        (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )

    assert 32 in buckets
    assert 128 in buckets
    assert 512 in buckets
    assert 1024 in buckets
    assert 8192 in buckets
