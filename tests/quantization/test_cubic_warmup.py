# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import cubic_warmup
from vllm.model_executor.warmup.cubic_warmup import _assign_cubic_tasks


def test_cubic_tactic_cache_key_allows_unbundled_native_sources(monkeypatch):
    original_read_bytes = Path.read_bytes

    def read_installed_source(path: Path) -> bytes:
        if path.suffix == ".cu":
            raise FileNotFoundError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_installed_source)
    monkeypatch.setattr(
        cubic_warmup, "_cubic_device_fingerprint", lambda: ("cuda", "sm90")
    )
    monkeypatch.setattr(cubic_warmup, "_cubic_model_signature", lambda *_: [])

    cache_key = cubic_warmup._cubic_tactic_cache_key(torch.nn.Identity(), (1,))

    assert len(cache_key) == 64


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


def test_cubic_linear_calibrates_low_m_once() -> None:
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    layer = torch.nn.Module()
    method = object.__new__(CubicLinearMethod)
    method.scheme = CubicScheme(num_bits=5, group_size=512, group_out=128)
    layer.quant_method = method
    layer.input_size_per_partition = 5120
    layer.output_size_per_partition = 4352
    layer.weight_a = torch.empty(0, device="meta", dtype=torch.float16)
    layer.weight_b = torch.empty(0, device="meta", dtype=torch.float16)

    tasks = cubic_warmup._cubic_calibration_tasks(layer, (2, 4, 8, 16, 32))

    assert sorted(task[-1] for task in tasks if task[0] == "linear") == [
        2,
        4,
        8,
        16,
        32,
    ]


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
        False,
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


def test_cubic_moe_sum_tactic_is_remapped_to_every_local_device(monkeypatch):
    registry: dict[tuple[object, ...], object] = {}
    monkeypatch.setattr(
        cubic_warmup,
        "_cubic_tactic_registries",
        lambda: {"_CUBIC_MOE_SUM_TACTICS": registry},
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 7)
    fingerprint = ("cuda", "sm120")
    cache_domain = ("cache", "/shared")
    payload = {
        "fingerprint": fingerprint,
        "cache_domain": cache_domain,
        "registries": {
            "_CUBIC_MOE_SUM_TACTICS": [
                [[2, 5120, 16, 64, True], False],
            ]
        },
    }

    cubic_warmup._merge_cubic_tactics([payload], fingerprint, cache_domain)

    assert registry == {(7, 5120, 16, 64, True): False}


def test_cubic_warmup_does_not_initialize_distributed_state_without_cubic_layers(
    monkeypatch,
):
    monkeypatch.setattr(cubic_warmup.envs, "VLLM_CUBIC_AUTOTUNE", True)
    monkeypatch.setattr(
        cubic_warmup, "_cubic_calibration_tasks", lambda *_, **__: ()
    )

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
    monkeypatch.setattr(
        cubic_warmup, "_cubic_calibration_tasks", lambda *_, **__: (task,)
    )
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


def test_cubic_calibration_uses_bounded_representative_buckets():
    buckets = cubic_warmup._calibration_token_buckets(
        8192,
        (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )

    assert buckets == (1, 2, 16, 64, 256, 512)


def test_cubic_calibration_keeps_the_largest_small_bucket():
    assert cubic_warmup._calibration_token_buckets(32, ()) == (1, 2, 16, 32)


def test_cubic_moe_calibration_adds_only_the_largest_runtime_bucket():
    base = cubic_warmup._calibration_token_buckets(8192, ())

    assert cubic_warmup._moe_calibration_token_buckets(8192, base) == (
        1,
        2,
        16,
        64,
        256,
        512,
        8192,
    )
    assert cubic_warmup._moe_calibration_token_buckets(256, base[:5]) == base[:5]


def test_cubic_calibration_routes_large_bucket_only_to_moe() -> None:
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicMoEMethod,
        CubicScheme,
    )

    model = torch.nn.Module()
    linear = torch.nn.Module()
    linear_method = object.__new__(CubicLinearMethod)
    linear_method.scheme = CubicScheme(num_bits=4, group_size=512, group_out=1)
    linear_method.dynamic_a8 = True
    linear.quant_method = linear_method
    linear.input_size_per_partition = 4096
    linear.output_size_per_partition = 4096
    linear.weight_a = torch.empty(0, device="meta", dtype=torch.float16)
    linear.weight_b = torch.empty(0, device="meta", dtype=torch.float16)

    moe = torch.nn.Module()
    moe_method = object.__new__(CubicMoEMethod)
    moe_method.scheme = CubicScheme(num_bits=4, group_size=512, group_out=128)
    moe_method.moe = SimpleNamespace(
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    moe.quant_method = moe_method
    moe.cubic_hidden_size = 4096
    moe.cubic_intermediate_size = 2048
    moe.top_k = 6
    moe.global_num_experts = 256
    moe.w13_weight_packed = torch.empty(
        32, 0, 0, device="meta", dtype=torch.uint8
    )
    moe.w13_weight_a = torch.empty(0, device="meta", dtype=torch.float16)
    moe.w2_weight_a = torch.empty(0, device="meta", dtype=torch.float16)
    moe.activation = "silu"
    moe.apply_router_weight_on_input = False
    model.add_module("linear", linear)
    model.add_module("moe", moe)

    tasks = cubic_warmup._cubic_calibration_tasks(
        model,
        (1, 16, 64, 256, 512),
        moe_token_buckets=(1, 16, 64, 256, 512, 8192),
    )

    assert 8192 not in {task[-1] for task in tasks if task[0] == "linear"}
    assert 8192 in {task[-1] for task in tasks if task[0] == "moe"}


def test_cubic_materialization_still_covers_every_runtime_bucket():
    assert cubic_warmup._materialization_token_buckets(8192) == (
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


def test_cubic_resident_validation_rejects_localized_semantic_error() -> None:
    reference = torch.full((1024,), 10.0)
    resident = reference.clone()
    resident[0] += 2.0

    with pytest.raises(AssertionError, match="accumulation-order bounds"):
        cubic_warmup._validate_cubic_resident_output(resident, reference)


def test_cubic_resident_validation_rejects_systemic_error() -> None:
    reference = torch.ones(1024)
    resident = reference * 1.02

    with pytest.raises(AssertionError, match="accumulation-order bounds"):
        cubic_warmup._validate_cubic_resident_output(resident, reference)


def test_cubic_batch_validation_allows_one_ulp_reduction_drift() -> None:
    inputs = torch.ones(64, 8)

    def batch_dependent_operation(x: torch.Tensor) -> torch.Tensor:
        output = x.clone()
        if x.shape[0] > 1:
            output[0, 0] = torch.nextafter(
                output[0, 0], torch.tensor(float("inf"))
            )
        return output

    cubic_warmup._validate_cubic_batch_invariance(batch_dependent_operation, inputs)


def test_cubic_batch_validation_rejects_wrong_row_mapping() -> None:
    inputs = torch.ones(64, 8)

    def selectively_batch_dependent_operation(x: torch.Tensor) -> torch.Tensor:
        output = x.clone()
        if x.shape[0] > 2:
            output[2] = output[2].roll(1)
        return output

    with pytest.raises(AssertionError):
        cubic_warmup._validate_cubic_batch_invariance(
            selectively_batch_dependent_operation, inputs
        )


def test_cubic_column_mapping_rejects_silent_permutation() -> None:
    input_size = 32
    indices = cubic_warmup._cubic_linear_probe_indices(input_size, 8)
    weight = torch.arange(64, dtype=torch.float32).reshape(2, input_size)
    expected = weight[:, indices].T

    cubic_warmup._validate_cubic_linear_column_mapping(
        lambda x: x @ weight.T,
        expected,
        indices,
        input_size=input_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(AssertionError, match="column mapping"):
        cubic_warmup._validate_cubic_linear_column_mapping(
            lambda x: x @ weight.roll(1, dims=1).T,
            expected,
            indices,
            input_size=input_size,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_a16_marlin_representation_requires_exact_stored_weight() -> None:
    carrier = torch.tensor([[127, 64]], dtype=torch.int8)
    scale = torch.tensor([[1.0]], dtype=torch.float32)
    marlin_scale = scale.to(torch.bfloat16) * (1.0 / 127.0)
    reconstructed = (carrier.float() * marlin_scale.float()).to(torch.bfloat16)

    assert cubic_warmup._a16_marlin_representation_matches_expanded_weight(
        carrier,
        scale,
        reconstructed,
        group_size=2,
        group_out=1,
    )

    changed = reconstructed.clone()
    changed[0, 1] = torch.nextafter(
        changed[0, 1], torch.tensor(float("inf"), dtype=torch.bfloat16)
    )
    assert not cubic_warmup._a16_marlin_representation_matches_expanded_weight(
        carrier,
        scale,
        changed,
        group_size=2,
        group_out=1,
    )


def test_a16_marlin_representation_accounts_for_scale_storage_dtype() -> None:
    carrier = torch.tensor([[3]], dtype=torch.int8)
    scale = torch.tensor([[1e-6]], dtype=torch.float32)
    fp32_scale_reconstruction = (
        carrier.float() * scale * (1.0 / 127.0)
    ).to(torch.bfloat16)

    assert not cubic_warmup._a16_marlin_representation_matches_expanded_weight(
        carrier,
        scale,
        fp32_scale_reconstruction,
        group_size=1,
        group_out=1,
    )


def test_cubic_residency_rejection_is_independent_of_metadata_storage() -> None:
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = True
    method.scheme = CubicScheme(num_bits=4, group_size=32, group_out=1)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 5120
    layer.output_size_per_partition = 17408
    key = cubic_warmup._linear_residency_backend_key(layer, method, "marlin")
    compact_key = (*key[:2], not key[2], *key[3:])

    assert cubic_warmup._linear_residency_backend_rejected(
        {compact_key: True}, layer, method, "marlin"
    )


def test_cubic_linear_residency_counts_every_repeated_layer(monkeypatch) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    def make_group(bits: int, members: int):
        method = object.__new__(CubicLinearMethod)
        method.dynamic_a8 = False
        method.scheme = CubicScheme(num_bits=bits, group_size=128, group_out=1)
        layers = []
        for _ in range(members):
            layer = torch.nn.Module()
            layer.input_size_per_partition = 128
            layer.output_size_per_partition = 128
            layers.append((layer, method))
        return layers

    repeated = make_group(4, 8)
    singleton = make_group(8, 1)
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    for layers, online_ms, extra in (
        (repeated, 2.0, 100),
        (singleton, 3.0, 300),
    ):
        layer, method = layers[0]
        key = cubic_warmup._linear_residency_key(layer, method, 1)
        cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
            online_ms,
            1.0,
            "dense",
            extra,
        )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 8000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (1600, 8000))
    installed: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda layer, _scheme: installed.append(id(layer)),
    )

    cubic_warmup._materialize_cubic_linear_residency(
        {("repeated",): repeated, ("singleton",): singleton}
    )

    assert installed == [id(layer) for layer, _ in repeated]


def test_cubic_linear_residency_uses_partial_group_when_memory_is_tight(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(num_bits=4, group_size=128, group_out=1)
    members = []
    for _ in range(8):
        layer = torch.nn.Module()
        layer.input_size_per_partition = 128
        layer.output_size_per_partition = 128
        members.append((layer, method))

    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    key = cubic_warmup._linear_residency_key(members[0][0], method, 1)
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
        2.0,
        1.0,
        "dense",
        200,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 8000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (1600, 8000))
    installed: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda layer, _scheme: installed.append(id(layer)),
    )

    cubic_warmup._materialize_cubic_linear_residency({("repeated",): members})

    assert installed == [id(layer) for layer, _ in members[:4]]
    assert all(
        layer.cubic_runtime_residency == "packed-online"
        for layer, _ in members[4:]
    )


def test_cubic_linear_residency_allocates_incremental_candidates(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CUBIC_COMPACT_METADATA_FORMAT,
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(
        num_bits=4,
        group_size=128,
        group_out=1,
        metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
    )
    members = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.input_size_per_partition = 128
        layer.output_size_per_partition = 128
        members.append((layer, method))

    key = cubic_warmup._linear_residency_key(members[0][0], method, 1)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS[key] = (
        10.0,
        6.0,
        100,
    )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
        10.0,
        2.0,
        "dense",
        300,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 2000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (1600, 2000))
    expanded: list[int] = []
    carriers: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_expanded_metadata",
        lambda layer, _scheme: expanded.append(id(layer)),
    )
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda layer, _scheme: carriers.append(id(layer)),
    )

    cubic_warmup._materialize_cubic_linear_residency({("repeated",): members})

    assert len(expanded) == 1
    assert len(carriers) == 1
    assert {
        layer.cubic_runtime_residency for layer, _ in members
    } == {"expanded-metadata", "dense-expanded-replaces-packed"}


def test_cubic_linear_residency_compares_multiple_backends(monkeypatch) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(num_bits=4, group_size=128, group_out=1)
    members = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.input_size_per_partition = 128
        layer.output_size_per_partition = 128
        members.append((layer, method))

    key = cubic_warmup._linear_residency_key(members[0][0], method, 1)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("marlin",)] = (
        10.0,
        4.0,
        "marlin",
        100,
    )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
        10.0,
        2.0,
        "dense",
        500,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 3000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (2400, 3000))
    marlin: list[int] = []
    dense: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_carrier",
        lambda layer, _scheme, *, dynamic_a8, backend: marlin.append(id(layer)),
    )
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda layer, _scheme: dense.append(id(layer)),
    )

    cubic_warmup._materialize_cubic_linear_residency({("repeated",): members})

    assert len(marlin) == 1
    assert len(dense) == 1
    assert {layer.cubic_runtime_residency for layer, _ in members} == {
        "marlin-carrier-replaces-packed",
        "dense-expanded-replaces-packed",
    }


def test_cubic_linear_residency_rejects_incomplete_backend(monkeypatch) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(num_bits=4, group_size=128, group_out=1)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 128
    layer.output_size_per_partition = 128
    key_m1 = cubic_warmup._linear_residency_key(layer, method, 1)
    key_m16 = cubic_warmup._linear_residency_key(layer, method, 16)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    for key in (key_m1, key_m16):
        cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
            10.0,
            8.0,
            "dense",
            100,
        )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key_m1 + ("marlin",)] = (
        10.0,
        1.0,
        "marlin",
        100,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 1000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (800, 1000))
    dense: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda selected, _scheme: dense.append(id(selected)),
    )

    cubic_warmup._materialize_cubic_linear_residency(
        {("layer",): [(layer, method)]}
    )

    assert dense == [id(layer)]
    assert layer.cubic_runtime_residency == "dense-expanded-replaces-packed"


def test_cubic_linear_residency_rejects_incomplete_exact_marlin(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(num_bits=4, group_size=32, group_out=1)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 128
    layer.output_size_per_partition = 128
    key_m1 = cubic_warmup._linear_residency_key(layer, method, 1)
    key_m16 = cubic_warmup._linear_residency_key(layer, method, 16)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    for key in (key_m1, key_m16):
        cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key + ("dense",)] = (
            10.0,
            8.0,
            "dense",
            100,
        )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[
        key_m1 + ("exact-marlin",)
    ] = (10.0, 1.0, "exact-marlin", 100)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 1000})(),
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (800, 1000))
    dense: list[int] = []
    exact: list[int] = []
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_a16_weight",
        lambda selected, _scheme: dense.append(id(selected)),
    )
    monkeypatch.setattr(
        cubic_warmup,
        "install_cubic_exact_marlin_weight",
        lambda selected, _scheme, *, token_buckets: exact.append(id(selected)),
    )

    cubic_warmup._materialize_cubic_linear_residency(
        {("layer",): [(layer, method)]}
    )

    assert exact == []
    assert dense == [id(layer)]
    assert layer.cubic_runtime_residency == "dense-expanded-replaces-packed"


def test_cubic_linear_residency_keeps_packed_for_mixed_dense_tactics(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(num_bits=4, group_size=32, group_out=1)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 128
    layer.output_size_per_partition = 128
    layer.weight_packed = torch.empty(128, 64, dtype=torch.uint8)
    key_m1 = cubic_warmup._linear_residency_key(layer, method, 1)
    key_m16 = cubic_warmup._linear_residency_key(layer, method, 16)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key_m1 + ("dense",)] = (
        1.0,
        2.0,
        "dense",
        100,
    )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key_m16 + ("dense",)] = (
        4.0,
        1.0,
        "dense",
        100,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 100_000})(),
    )
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda *_: (80_000, 100_000)
    )
    installed: list[tuple[bool, tuple[int, ...]]] = []

    def install(_layer, _scheme, *, retain_packed=False, online_buckets=()):
        installed.append((retain_packed, online_buckets))

    monkeypatch.setattr(cubic_warmup, "install_cubic_a16_weight", install)

    cubic_warmup._materialize_cubic_linear_residency(
        {("layer",): [(layer, method)]}
    )

    assert installed == [(True, (1,))]
    assert layer.cubic_runtime_residency == "dense-expanded-with-packed-dispatch"


def test_cubic_linear_residency_combines_expanded_metadata_and_dense(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CUBIC_COMPACT_METADATA_FORMAT,
        CubicLinearMethod,
        CubicScheme,
    )

    method = object.__new__(CubicLinearMethod)
    method.dynamic_a8 = False
    method.scheme = CubicScheme(
        num_bits=4,
        group_size=32,
        group_out=1,
        metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
    )
    layer = torch.nn.Module()
    layer.input_size_per_partition = 128
    layer.output_size_per_partition = 128
    layer.weight_packed = torch.empty(128, 64, dtype=torch.uint8)
    key_m1 = cubic_warmup._linear_residency_key(layer, method, 1)
    key_m16 = cubic_warmup._linear_residency_key(layer, method, 16)
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS[key_m1] = (
        2.0,
        4.1,
        100,
    )
    cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS[key_m16] = (
        10.0,
        5.0,
        100,
    )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS.clear()
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key_m1 + ("dense",)] = (
        2.0,
        4.0,
        "dense",
        300,
    )
    cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[key_m16 + ("dense",)] = (
        10.0,
        1.0,
        "dense",
        300,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: type("Properties", (), {"total_memory": 100_000})(),
    )
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda *_: (80_000, 100_000)
    )
    dense_installs = []
    metadata_installs = []

    def install_dense(
        _layer, _scheme, *, retain_packed=False, online_buckets=()
    ):
        dense_installs.append((retain_packed, online_buckets))

    def install_metadata(_layer, _scheme, *, token_buckets=None):
        metadata_installs.append(token_buckets)

    monkeypatch.setattr(cubic_warmup, "install_cubic_a16_weight", install_dense)
    monkeypatch.setattr(
        cubic_warmup, "install_cubic_expanded_metadata", install_metadata
    )

    cubic_warmup._materialize_cubic_linear_residency(
        {("layer",): [(layer, method)]}
    )

    assert dense_installs == [(True, (1,))]
    assert metadata_installs == []
    assert (
        layer.cubic_runtime_residency
        == "dense-expanded-with-packed-dispatch"
    )
