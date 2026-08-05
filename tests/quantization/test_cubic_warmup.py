# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.warmup.cubic_warmup import _assign_cubic_tasks


def _linear_task(bits: int) -> tuple[object, ...]:
    return ("linear", bits, 256, 3072, 7168, "torch.float16", "torch.float16", 16)


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
