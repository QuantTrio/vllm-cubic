# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-aware distributed tactic calibration for Cubic kernels.

The calibration deliberately uses synthetic routes and activations.  It is
run by vLLM's kernel-warmup phase, before CUDA graph capture, so graph-capture
dummy routing cannot train a production tactic.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_COMPACT_METADATA_FORMAT,
    CUBIC_E5M9_CURVE2_METADATA_FORMAT,
    CubicLinearMethod,
    CubicMoEMethod,
    apply_cubic_exact_marlin_weight,
    apply_cubic_marlin_weight,
    cubic_carrier_levels,
    expanded_cubic_metadata,
    install_cubic_a16_weight,
    install_cubic_carrier,
    install_cubic_exact_marlin_weight,
    install_cubic_expanded_metadata,
    materialize_cubic_a8_carrier,
    materialize_cubic_a16_weight,
    prepare_cubic_exact_marlin_weight,
    prepare_cubic_marlin_weight,
)
from vllm.model_executor.layers.quantization.cubic_policy import (
    CUBIC_TOKEN_BUCKETS,
    CubicCarrierResidency,
    CubicMetadataResidency,
    CubicRuntimeCandidate,
    cubic_dynamic_a8_group_size,
    cubic_linear_residency_budget,
    cubic_linear_token_bucket,
    cubic_runtime_memory,
    cubic_token_bucket,
)

logger = init_logger(__name__)

_CUBIC_TACTIC_CACHE_SCHEMA = 48
_CUBIC_TACTIC_CACHE_FILENAME = "cubic_tactics.json"
_CUBIC_TACTIC_REGISTRY_NAMES = (
    "_CUBIC_W2_A8_SITU_TACTICS",
    "_CUBIC_A8_MOE_BACKEND_TACTICS",
    "_CUBIC_ONLINE_A8_MOE_BACKEND_TACTICS",
    "_CUBIC_A8_MOE_GROUPING_TACTICS",
    "_CUBIC_MOE_EXECUTION_TACTICS",
    "_CUBIC_LINEAR_EXECUTION_TACTICS",
    "_CUBIC_LINEAR_BLOCK_K_TACTICS",
    "_CUBIC_LINEAR_TILE_TACTICS",
    "_CUBIC_COMPACT_LINEAR_TILE_TACTICS",
    "_CUBIC_LINEAR_STREAM_TACTICS",
    "_CUBIC_LINEAR_RESIDENCY_TACTICS",
    "_CUBIC_LINEAR_REJECTED_RESIDENCIES",
    "_CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS",
    "_CUBIC_MOE_DENSE_BLOCK_TACTICS",
    "_CUBIC_MOE_DENSE_BLOCK_K_TACTICS",
    "_CUBIC_MOE_ROUTE_CTA_TACTICS",
    "_CUBIC_MOE_SUM_TACTICS",
    "_CUBIC8_W2_BLOCK_N_TACTICS",
    "_CUBIC8_W2_LUT_TACTICS",
)

CalibrationTask = tuple[Any, ...]


def _linear_metadata_signature(
    layer: torch.nn.Module, method: CubicLinearMethod
) -> tuple[str, str]:
    if method.scheme.metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT:
        return str(layer.weight_metadata.dtype), str(layer.weight_curve_a.dtype)
    if method.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT:
        return str(layer.weight_scale.dtype), str(layer.weight_ab.dtype)
    return str(layer.weight_a.dtype), str(layer.weight_b.dtype)


def _cubic_tactic_registries() -> dict[str, dict[tuple[Any, ...], Any]]:
    from vllm.model_executor.layers.quantization import cubic_kernels

    return {name: getattr(cubic_kernels, name) for name in _CUBIC_TACTIC_REGISTRY_NAMES}


def _cubic_model_signature(
    model: torch.nn.Module, token_buckets: tuple[int, ...]
) -> list[Any]:
    signatures: list[Any] = [["tokens", *token_buckets]]
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, CubicLinearMethod):
            metadata_signature = _linear_metadata_signature(module, method)
            signatures.append(
                [
                    "linear",
                    method.scheme.num_bits,
                    method.scheme.group_size,
                    method.scheme.group_out,
                    int(module.input_size_per_partition),
                    int(module.output_size_per_partition),
                    list(module.weight_packed.shape),
                    *metadata_signature,
                    method.scheme.metadata_format,
                    bool(method.dynamic_a8),
                ]
            )
        elif isinstance(method, CubicMoEMethod):
            signatures.append(
                [
                    "moe",
                    method.scheme.num_bits,
                    method.scheme.group_size,
                    method.scheme.group_out,
                    int(module.cubic_hidden_size),
                    int(module.cubic_intermediate_size),
                    int(module.top_k),
                    int(module.global_num_experts),
                    list(module.w13_weight_packed.shape),
                    list(module.w2_weight_packed.shape),
                    str(module.w13_weight_a.dtype),
                    str(module.w2_weight_a.dtype),
                    str(module.activation),
                    bool(module.apply_router_weight_on_input),
                    method.moe.activation_situ_beta,
                    method.moe.activation_situ_linear_beta,
                    bool(method.dynamic_a8),
                ]
            )
    serialized = {json.dumps(value, sort_keys=True) for value in signatures}
    return [json.loads(value) for value in sorted(serialized)]


def _cubic_tactic_cache_key(
    model: torch.nn.Module, token_buckets: tuple[int, ...]
) -> str:
    import triton

    source_paths = (
        Path(__file__),
        Path(__file__).resolve().parents[1]
        / "layers"
        / "quantization"
        / "cubic.py",
        Path(__file__).resolve().parents[1]
        / "layers"
        / "quantization"
        / "cubic_kernels.py",
        Path(__file__).resolve().parents[1]
        / "layers"
        / "quantization"
        / "cubic_policy.py",
        Path(__file__).resolve().parents[1]
        / "layers"
        / "quantization"
        / "utils"
        / "marlin_utils.py",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "cubic_w2_a8_gemv.cu",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "cubic_w3_a8_gemv.cu",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "cubic_w4_w8_a8_gemv.cu",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "marlin"
        / "marlin.cu",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "marlin"
        / "marlin_template.h",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "quantization"
        / "marlin"
        / "kernel.h",
        Path(__file__).resolve().parents[3]
        / "csrc"
        / "libtorch_stable"
        / "torch_bindings.cpp",
    )
    source_hash = hashlib.sha256()
    for path in source_paths:
        source_hash.update(path.name.encode())
        try:
            source_hash.update(path.read_bytes())
        except FileNotFoundError:
            source_hash.update(b"<source-not-packaged>")
    extension_identity: list[Any] = []
    try:
        import vllm._C_stable_libtorch as vllm_extension

        extension_path = Path(vllm_extension.__file__)
        extension_stat = extension_path.stat()
        extension_identity = [
            str(extension_path),
            extension_stat.st_size,
            extension_stat.st_mtime_ns,
        ]
    except (ImportError, OSError, TypeError):
        pass
    identity = {
        "schema": _CUBIC_TACTIC_CACHE_SCHEMA,
        "source": source_hash.hexdigest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "extension": extension_identity,
        "device": _cubic_device_fingerprint(),
        "model": _cubic_model_signature(model, token_buckets),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest


def _cubic_device_fingerprint() -> tuple[Any, ...]:
    import triton
    from triton.runtime.driver import driver

    device = torch.accelerator.current_device_index()
    properties = torch.cuda.get_device_properties(device)
    target = driver.active.get_current_target()
    return (
        target.backend,
        target.arch,
        target.warp_size,
        properties.name,
        properties.major,
        properties.minor,
        properties.total_memory,
        properties.multi_processor_count,
        torch.__version__,
        torch.version.cuda,
        triton.__version__,
    )


def _linear_task_id(
    layer: torch.nn.Module,
    method: CubicLinearMethod,
    tokens: int,
) -> CalibrationTask:
    metadata_signature = _linear_metadata_signature(layer, method)
    return (
        "linear",
        method.scheme.num_bits,
        method.scheme.group_size,
        method.scheme.group_out,
        int(layer.input_size_per_partition),
        int(layer.output_size_per_partition),
        *metadata_signature,
        getattr(method, "dynamic_a8", False),
        tokens,
    )


def _moe_task_id(
    layer: torch.nn.Module,
    method: CubicMoEMethod,
    tokens: int,
) -> CalibrationTask:
    return (
        "moe",
        method.scheme.num_bits,
        method.scheme.group_size,
        method.scheme.group_out,
        int(layer.cubic_hidden_size),
        int(layer.cubic_intermediate_size),
        int(layer.top_k),
        int(layer.w13_weight_packed.shape[0]),
        int(layer.global_num_experts),
        str(layer.w13_weight_a.dtype),
        str(layer.w2_weight_a.dtype),
        str(layer.activation),
        bool(layer.apply_router_weight_on_input),
        method.moe.activation_situ_beta,
        method.moe.activation_situ_linear_beta,
        tokens,
    )


def _cubic_calibration_tasks(
    model: torch.nn.Module,
    token_buckets: tuple[int, ...],
    *,
    moe_token_buckets: tuple[int, ...] | None = None,
) -> tuple[CalibrationTask, ...]:
    tasks: set[CalibrationTask] = set()
    if moe_token_buckets is None:
        moe_token_buckets = token_buckets
    linear_token_buckets = tuple(
        sorted({cubic_linear_token_bucket(m) for m in token_buckets})
    )
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, CubicLinearMethod):
            tasks.update(
                _linear_task_id(module, method, m) for m in linear_token_buckets
            )
        elif isinstance(method, CubicMoEMethod):
            tasks.update(_moe_task_id(module, method, m) for m in moe_token_buckets)
    tasks.update(("w2_situ", *spec) for spec in _cubic_w2_a8_situ_specs(model))
    return tuple(sorted(tasks, key=repr))


def _cubic_task_weight(task: CalibrationTask) -> int:
    if task[0] == "linear":
        k, n, m = task[4], task[5], task[-1]
        return int(k) * int(n) * (32 + int(m))
    if task[0] == "w2_situ":
        _, n, k, _, _, _, local_experts = task
        return int(n) * int(k) * max(int(local_experts), 1) * 256
    _, _, _, _, hidden, intermediate, top_k, local_experts, _, *tail = task
    tokens = int(tail[-1])
    return (
        int(hidden)
        * int(intermediate)
        * max(int(local_experts), 1)
        * (32 + tokens * int(top_k))
    )


class _CalibrationProgress:
    def __init__(self, rank: int, tasks: set[CalibrationTask]) -> None:
        self.rank = rank
        self.total = len(tasks)
        self.completed = 0
        self.partial: dict[CalibrationTask, float] = {}
        self.started = time.monotonic()

    def _report(self, task: CalibrationTask, detail: str) -> None:
        elapsed = time.monotonic() - self.started
        # Calibration is dominated by Triton compilation and the number of
        # candidate tactics, not by the tensor FLOP count used to balance work
        # across ranks.  Extrapolating wall time from that weight can therefore
        # produce absurd multi-year ETAs while tasks are completing normally.
        # Use observed wall time per completed task-equivalent instead. Internal
        # phase milestones keep the first large task from remaining at 0% for
        # minutes. The ETA is approximate and converges as calibration proceeds.
        effective_completed = self.completed + sum(self.partial.values())
        remaining = max(self.total - effective_completed, 0.0)
        eta = elapsed * remaining / max(effective_completed, 1e-3)
        logger.info(
            "Cubic calibration progress rank=%d %d/%d tasks (%.1f%%), "
            "elapsed=%.1fs, ETA~%.1fs, %s=%s",
            self.rank,
            self.completed,
            self.total,
            100.0 * effective_completed / max(self.total, 1),
            elapsed,
            eta,
            detail,
            task,
        )

    def phase(
        self,
        task: CalibrationTask,
        completed_phases: int,
        total_phases: int,
        phase: str,
    ) -> None:
        if total_phases <= 0:
            return
        self.partial[task] = min(max(completed_phases / total_phases, 0.0), 0.999)
        self._report(task, f"phase={completed_phases}/{total_phases} {phase}")

    def __call__(self, task: CalibrationTask) -> None:
        self.partial.pop(task, None)
        self.completed += 1
        self._report(task, "completed")


def _cubic_world() -> tuple[int, int, Any | None]:
    try:
        from vllm.distributed.parallel_state import get_world_group

        world = get_world_group()
        return world.rank, world.world_size, world.cpu_group
    except (AssertionError, RuntimeError):
        return 0, 1, None


def _cubic_all_gather(value: Any, world_size: int, cpu_group: Any | None) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, value, group=cpu_group)
    return gathered


def _cubic_barrier(world_size: int, cpu_group: Any | None) -> None:
    if world_size > 1:
        torch.distributed.barrier(group=cpu_group)


def _cubic_cache_domain(
    cache_key: str | None,
    rank: int,
    world_size: int,
    cpu_group: Any | None,
) -> tuple[Any, ...]:
    """Follow Triton's configured cache manager and filesystem visibility."""
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    host = socket.gethostname()
    manager_class = knobs.cache.manager_class
    managed_cache = manager_class is not None and manager_class is not FileCacheManager
    manager_signature = None
    if managed_cache:
        manager_signature = (
            manager_class.__module__,
            manager_class.__qualname__,
            tuple(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if key.startswith("TRITON_")
                )
            ),
        )
    cache_root = (
        Path(knobs.cache.dir) if cache_key is not None and not managed_cache else None
    )
    path_value = str(cache_root) if cache_root is not None else None
    probe_metadata = _cubic_all_gather(
        {
            "rank": rank,
            "host": host,
            "path": path_value,
            "manager": manager_signature,
            "token": uuid.uuid4().hex,
        },
        world_size,
        cpu_group,
    )
    leaders: dict[tuple[str, str], dict[str, Any]] = {}
    for item in probe_metadata:
        if item["path"] is None:
            continue
        key = (item["host"], item["path"])
        if key not in leaders or int(item["rank"]) < int(leaders[key]["rank"]):
            leaders[key] = item
    local_leader = leaders.get((host, path_value)) if path_value is not None else None
    local_probe: Path | None = None
    if local_leader is not None and int(local_leader["rank"]) == rank:
        try:
            assert cache_root is not None
            cache_root.mkdir(parents=True, exist_ok=True)
            local_probe = cache_root / f".visibility.{local_leader['token']}"
            local_probe.write_text(str(local_leader["token"]))
        except OSError as error:
            logger.warning("Unable to create Cubic cache visibility probe: %s", error)
            local_probe = None
    _cubic_barrier(world_size, cpu_group)
    visible: list[int] = []
    if cache_root is not None:
        for item in leaders.values():
            if item["path"] != path_value:
                continue
            probe = cache_root / f".visibility.{item['token']}"
            try:
                if probe.read_text() == str(item["token"]):
                    visible.append(int(item["rank"]))
            except OSError:
                pass
    # Every rank must finish observing the probes before a leader removes its
    # file.  Without this barrier, ranks on the same host/cache can race and
    # report different cache domains, causing duplicate calibration work.
    _cubic_barrier(world_size, cpu_group)
    if local_probe is not None:
        try:
            local_probe.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("Unable to remove Cubic cache visibility probe: %s", error)
    _cubic_barrier(world_size, cpu_group)
    if manager_signature is not None:
        return ("manager", *manager_signature)
    if visible:
        return ("cache", path_value, *sorted(visible))
    return ("host", host, path_value)


def _assign_cubic_tasks(metadata: list[dict[str, Any]]) -> tuple[dict[int, set], int]:
    assignments: dict[int, set[CalibrationTask]] = {
        int(item["rank"]): set() for item in metadata
    }
    loads = {rank: 0 for rank in assignments}
    work: dict[tuple[tuple[Any, ...], tuple[Any, ...], CalibrationTask], list[int]] = {}
    cached: set[tuple[tuple[Any, ...], tuple[Any, ...], CalibrationTask]] = set()
    for item in metadata:
        rank = int(item["rank"])
        fingerprint = tuple(item["fingerprint"])
        cache_domain = tuple(item["cache_domain"])
        for task_value in item["tasks"]:
            task = tuple(task_value)
            key = (cache_domain, fingerprint, task)
            work.setdefault(key, []).append(rank)
            if item["cache_hit"]:
                cached.add(key)
    pending = [(key, ranks) for key, ranks in work.items() if key not in cached]
    pending.sort(key=lambda item: (-_cubic_task_weight(item[0][2]), repr(item[0])))
    for (_, _, task), eligible_ranks in pending:
        owner = min(eligible_ranks, key=lambda rank: (loads[rank], rank))
        assignments[owner].add(task)
        loads[owner] += _cubic_task_weight(task)
    return assignments, len(pending)


def _export_cubic_tactics() -> dict[str, list[list[Any]]]:
    return {
        name: [[list(key), value] for key, value in registry.items()]
        for name, registry in _cubic_tactic_registries().items()
    }


def _merge_cubic_tactics(
    payloads: list[dict[str, Any]],
    fingerprint: tuple[Any, ...],
    cache_domain: tuple[Any, ...],
) -> None:
    registries = _cubic_tactic_registries()
    for registry in registries.values():
        registry.clear()
    device = torch.accelerator.current_device_index()
    for payload in payloads:
        if (
            tuple(payload["fingerprint"]) != fingerprint
            or tuple(payload["cache_domain"]) != cache_domain
        ):
            continue
        for name, entries in payload["registries"].items():
            registry = registries[name]
            for key, value in entries:
                local_key = (device, *key[1:])
                if name in (
                    "_CUBIC_W2_A8_SITU_TACTICS",
                    "_CUBIC_LINEAR_TILE_TACTICS",
                    "_CUBIC_LINEAR_STREAM_TACTICS",
                    "_CUBIC_LINEAR_RESIDENCY_TACTICS",
                    "_CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS",
                ):
                    value = tuple(value)
                registry[local_key] = value


def _load_cubic_tactic_cache(cache_key: str) -> bool:
    from triton.runtime.cache import get_cache_manager

    path_value = get_cache_manager(cache_key).get_file(_CUBIC_TACTIC_CACHE_FILENAME)
    if path_value is None:
        return False
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text())
        if payload.get("schema") != _CUBIC_TACTIC_CACHE_SCHEMA:
            return False
        cached = payload["registries"]
        registries = _cubic_tactic_registries()
        decoded: dict[str, dict[tuple[Any, ...], Any]] = {}
        for name, registry in registries.items():
            entries = cached[name]
            values: dict[tuple[Any, ...], Any] = {}
            for key, value in entries:
                if not key:
                    raise ValueError("Cubic tactic cache has an empty tactic key.")
                key_tuple = (torch.accelerator.current_device_index(), *key[1:])
                if name in (
                    "_CUBIC_W2_A8_SITU_TACTICS",
                    "_CUBIC_LINEAR_TILE_TACTICS",
                    "_CUBIC_LINEAR_STREAM_TACTICS",
                ):
                    value = tuple(value)
                values[key_tuple] = value
            decoded[name] = values
        for name, registry in registries.items():
            registry.clear()
            registry.update(decoded[name])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if path.exists():
            logger.warning("Ignoring invalid Cubic tactic cache %s: %s", path, error)
        return False
    logger.info("Loaded Cubic tactic cache: %s", path)
    return True


def _save_cubic_tactic_cache(cache_key: str) -> None:
    from triton.runtime.cache import get_cache_manager

    registries = _cubic_tactic_registries()
    payload = {
        "schema": _CUBIC_TACTIC_CACHE_SCHEMA,
        "registries": {
            name: [[[0, *key[1:]], value] for key, value in registry.items()]
            for name, registry in registries.items()
        },
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path = get_cache_manager(cache_key).put(
        data,
        _CUBIC_TACTIC_CACHE_FILENAME,
        binary=False,
    )
    logger.info("Saved Cubic tactic cache: %s", path)


def _cubic_w2_a8_situ_specs(
    model: torch.nn.Module,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return unique (N, K, group_out, group_in, top_k, experts) shapes."""
    specs: set[tuple[int, int, int, int, int, int]] = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, CubicMoEMethod):
            continue
        if method.scheme.num_bits != 2:
            continue
        n = getattr(module, "cubic_intermediate_size", None)
        k = getattr(module, "cubic_hidden_size", None)
        top_k = getattr(module, "top_k", None)
        packed = getattr(module, "w13_weight_packed", None)
        if n is None or k is None or top_k is None or packed is None:
            continue
        specs.add(
            (
                int(n),
                int(k),
                method.scheme.group_out,
                method.scheme.group_size,
                int(top_k),
                packed.shape[0],
            )
        )
    return tuple(sorted(specs))


def _calibration_token_buckets(
    max_tokens: int, capture_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    del capture_sizes
    largest_bucket = cubic_token_bucket(max_tokens)
    representatives = (1, 2, 16, 64, 256, 512)
    selected = tuple(bucket for bucket in representatives if bucket <= largest_bucket)
    if largest_bucket not in selected and largest_bucket < representatives[-1]:
        selected = (*selected, largest_bucket)
    return tuple(sorted(set(selected)))


def _moe_calibration_token_buckets(
    max_tokens: int, base_buckets: tuple[int, ...]
) -> tuple[int, ...]:
    """Add one bounded large-M point for routed-weight reuse calibration."""
    largest_bucket = cubic_token_bucket(max_tokens)
    if largest_bucket <= base_buckets[-1]:
        return base_buckets
    return (*base_buckets, largest_bucket)


def _materialization_token_buckets(max_tokens: int) -> tuple[int, ...]:
    largest_bucket = cubic_token_bucket(max_tokens)
    return tuple(bucket for bucket in CUBIC_TOKEN_BUCKETS if bucket <= largest_bucket)


def _validate_cubic_resident_output(
    resident: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_nrmse: float = 5e-3,
) -> None:
    """Reject a candidate outside bounded accumulation-order error."""
    resident_fp32 = resident.float()
    reference_fp32 = reference.float()
    if not torch.isfinite(resident_fp32).all():
        raise AssertionError("Cubic resident candidate produced non-finite output.")
    error = resident_fp32 - reference_fp32
    error_energy = error.square().mean()
    reference_energy = (
        reference_fp32.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
    )
    nrmse = torch.sqrt(error_energy / reference_energy).item()
    resident_rows = resident_fp32.reshape(-1, resident_fp32.shape[-1])
    reference_rows = reference_fp32.reshape(-1, reference_fp32.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(
        resident_rows, reference_rows, dim=1, eps=1e-12
    )
    reference_norm = torch.linalg.vector_norm(reference_rows, dim=1).clamp_min(1e-12)
    norm_ratio_error = (
        (torch.linalg.vector_norm(resident_rows, dim=1) / reference_norm - 1)
        .abs()
        .max()
        .item()
    )
    min_cosine = cosine.min().item()
    if (
        nrmse > max_nrmse
        or min_cosine < 0.9999
        or norm_ratio_error > 0.01
    ):
        raise AssertionError(
            "Cubic resident candidate exceeds accumulation-order bounds: "
            f"nrmse={nrmse:.6g}, min_cosine={min_cosine:.8g}, "
            f"max_norm_ratio_error={norm_ratio_error:.6g}."
        )


def _cubic_linear_probe_indices(input_size: int, group_size: int) -> torch.Tensor:
    """Select boundary and hashed columns that expose layout mistakes."""
    indices = {
        0,
        min(input_size - 1, 1),
        max(0, group_size - 1),
        min(input_size - 1, group_size),
        input_size // 2,
        input_size - 1,
    }
    for value in (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F):
        indices.add(value % input_size)
    return torch.tensor(sorted(indices), dtype=torch.int64)


def _validate_cubic_linear_column_mapping(
    operation: Callable[[torch.Tensor], torch.Tensor],
    expected_columns: torch.Tensor,
    indices: torch.Tensor,
    *,
    input_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Reject layout/group errors using reduction-free one-hot probes."""
    indices = indices.to(device=device)
    probes = torch.zeros(
        (indices.numel(), input_size), device=device, dtype=dtype
    )
    probes[torch.arange(indices.numel(), device=device), indices] = 1
    actual = operation(probes)
    expected = expected_columns.to(device=device, dtype=actual.dtype)
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise AssertionError("Cubic column probe produced an invalid output.")
    negative_inf = torch.full_like(expected, float("-inf"))
    positive_inf = torch.full_like(expected, float("inf"))
    lower = torch.nextafter(expected, negative_inf)
    upper = torch.nextafter(expected, positive_inf)
    mismatch = (actual < lower) | (actual > upper)
    if mismatch.any():
        count = int(mismatch.sum().item())
        max_error = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(
            "Cubic column mapping differs by more than one output ULP: "
            f"mismatches={count}, max_abs_error={max_error:.6g}."
        )


def _cubic_a8_expected_columns(
    carrier: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    group_size: int,
    group_out: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return canonical A8 outputs for one-hot input columns."""
    indices = indices.to(device=carrier.device)
    output_groups = torch.arange(carrier.shape[0], device=carrier.device) // group_out
    input_groups = indices // group_size
    selected_scale = scale[output_groups[:, None], input_groups[None, :]].float()
    selected_carrier = carrier[:, indices].float()
    return (selected_carrier * selected_scale * (1.0 / 127.0)).T.to(dtype)


def _invoke_cubic_probe(
    probe: torch.Tensor,
    *,
    operation: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    return operation(probe, *args, **kwargs)


def _validate_cubic_batch_invariance(
    operation: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> None:
    """Reject semantic row drift while allowing bounded reduction rounding."""
    row_count = min(inputs.reshape(-1, inputs.shape[-1]).shape[0], 512)
    if row_count == 1:
        return
    ramp = torch.linspace(
        -1,
        1,
        inputs.shape[-1],
        device=inputs.device,
        dtype=inputs.dtype,
    )
    indices = torch.arange(inputs.shape[-1], device=inputs.device, dtype=torch.int64)
    hashed = ((indices * 1103515245 + 12345) % 65536).to(torch.float32)
    hashed = ((hashed - 32768) / 32768).to(inputs.dtype)
    probes = torch.stack(
        (
            ramp,
            ramp.flip(0) * 0.125,
            hashed,
            hashed.roll(inputs.shape[-1] // 3) * 2,
        )
    )
    solo = torch.cat([operation(row.reshape(1, -1)) for row in probes])
    repeats = (row_count + probes.shape[0] - 1) // probes.shape[0]
    batched_inputs = probes.repeat(repeats, 1)[:row_count]
    batched = operation(batched_inputs)
    expected = solo.repeat(repeats, 1)[:row_count]
    if batched.shape != expected.shape:
        raise AssertionError(
            f"resident output shape {tuple(batched.shape)} does not match "
            f"independent-row shape {tuple(expected.shape)}"
        )
    batched_fp32 = batched.float()
    expected_fp32 = expected.float()
    if not torch.isfinite(batched_fp32).all():
        raise AssertionError("resident batched output contains non-finite values")
    error = batched_fp32 - expected_fp32
    reference_energy = (
        expected_fp32.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
    )
    nrmse = torch.sqrt(error.square().mean() / reference_energy).item()
    expected_rows = expected_fp32.reshape(row_count, -1)
    batched_rows = batched_fp32.reshape(row_count, -1)
    cosine = torch.nn.functional.cosine_similarity(
        expected_rows, batched_rows, dim=1, eps=1e-12
    )
    expected_norm = torch.linalg.vector_norm(expected_rows, dim=1).clamp_min(1e-12)
    norm_ratio_error = (
        ((torch.linalg.vector_norm(batched_rows, dim=1) / expected_norm) - 1)
        .abs()
        .max()
        .item()
    )
    min_cosine = cosine.min().item()
    if nrmse > 5e-3 or min_cosine < 0.9999 or norm_ratio_error > 0.01:
        raise AssertionError(
            "resident batched rows exceed reduction-rounding bounds: "
            f"nrmse={nrmse:.6g}, min_cosine={min_cosine:.6g}, "
            f"max_norm_ratio_error={norm_ratio_error:.6g}"
        )


def _a16_marlin_representation_matches_expanded_weight(
    carrier: torch.Tensor,
    scale: torch.Tensor,
    expanded_weight: torch.Tensor,
    *,
    group_size: int,
    group_out: int,
) -> bool:
    """Return whether Marlin carrier and scales preserve the A16 weight."""
    if carrier.shape != expanded_weight.shape:
        return False
    input_groups = torch.arange(carrier.shape[1], device=carrier.device) // group_size
    row_chunk = max(group_out, 256 - (256 % group_out))
    for row_start in range(0, carrier.shape[0], row_chunk):
        row_end = min(row_start + row_chunk, carrier.shape[0])
        output_groups = (
            torch.arange(row_start, row_end, device=carrier.device) // group_out
        )
        scale_values = (scale.to(expanded_weight.dtype) * (1.0 / 127.0))[
            output_groups[:, None], input_groups[None, :]
        ]
        reconstructed = (carrier[row_start:row_end].float() * scale_values.float()).to(
            expanded_weight.dtype
        )
        if not torch.equal(reconstructed, expanded_weight[row_start:row_end]):
            return False
    return True


def _linear_residency_key(
    layer: torch.nn.Module,
    method: CubicLinearMethod,
    tokens: int,
) -> tuple[int, bool, bool, int, int, int, int, int, int]:
    return (
        torch.accelerator.current_device_index(),
        method.dynamic_a8,
        method.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT,
        method.scheme.num_bits,
        int(layer.output_size_per_partition),
        int(layer.input_size_per_partition),
        method.scheme.group_out,
        method.scheme.group_size,
        cubic_linear_token_bucket(tokens),
    )


def _linear_residency_backend_key(
    layer: torch.nn.Module,
    method: CubicLinearMethod,
    backend: str,
) -> tuple[int, bool, bool, int, int, int, int, int, str]:
    return _linear_residency_key(layer, method, 1)[:-1] + (backend,)


def _linear_residency_backend_rejected(
    rejected: dict[tuple[Any, ...], bool],
    layer: torch.nn.Module,
    method: CubicLinearMethod,
    backend: str,
) -> bool:
    expected = _linear_residency_backend_key(layer, method, backend)
    # Metadata storage does not affect a resident carrier backend's arithmetic.
    # Reject the backend for the entire execution signature if either compact
    # or legacy metadata exposed batch-dependent output.
    return any(
        key[0:2] == expected[0:2] and key[3:] == expected[3:] and value
        for key, value in rejected.items()
    )


def _linear_residency_latency_score(
    latency_by_bucket: dict[int, float],
) -> float:
    """Score residency across representative serving workloads.

    Every decode-sized bucket recurs for the full generation, not only M=1.
    Larger buckets represent one-shot prefill work.
    """
    return sum(
        latency * (128 if bucket <= 128 else 1)
        for bucket, latency in latency_by_bucket.items()
    )


def _materialize_cubic_linear_residency(
    layer_groups: dict[
        CalibrationTask, list[tuple[torch.nn.Module, CubicLinearMethod]]
    ],
) -> None:
    if os.getenv("VLLM_CUBIC_DEBUG_DISABLE_LINEAR_RESIDENCY"):
        return
    from vllm.model_executor.layers.quantization import cubic_kernels

    carrier_registry = cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS
    rejected_registry = cubic_kernels._CUBIC_LINEAR_REJECTED_RESIDENCIES
    metadata_registry = cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS
    device = torch.accelerator.current_device_index()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory, _ = torch.accelerator.get_memory_info(device)
    budget = cubic_linear_residency_budget(
        total_memory_bytes=total_memory,
        free_memory_bytes=free_memory,
    )
    remaining = budget
    hybrid_online_buckets: dict[tuple[Any, ...], tuple[int, ...]] = {}
    hybrid_expanded_metadata_buckets: dict[
        tuple[Any, ...], tuple[int, ...]
    ] = {}
    hybrid_expanded_metadata_dense_buckets: dict[
        tuple[Any, ...], tuple[int, ...]
    ] = {}
    exact_marlin_buckets: dict[tuple[Any, ...], tuple[int, ...]] = {}
    plans: list[
        tuple[
            list[tuple[torch.nn.Module, CubicLinearMethod]],
            tuple[CubicRuntimeCandidate, ...],
            dict[str, str],
        ]
    ] = []
    for members in layer_groups.values():
        layer, method = members[0]
        prefix = _linear_residency_key(layer, method, 1)[:-1]
        carrier_measurements: dict[str, dict[int, tuple[float, float, str, int]]] = {}
        for key, value in carrier_registry.items():
            if key[:-2] == prefix:
                carrier_measurements.setdefault(key[-1], {})[key[-2]] = value
        metadata_measurements = {
            key[-1]: value
            for key, value in metadata_registry.items()
            if key[:-1] == prefix
        }
        if not carrier_measurements and not metadata_measurements:
            continue
        required_buckets = (
            set(metadata_measurements)
            if metadata_measurements
            else set().union(*(set(values) for values in carrier_measurements.values()))
        )
        partial_exact_measurements = carrier_measurements.get("exact-marlin", {})
        carrier_measurements = {
            backend: values
            for backend, values in carrier_measurements.items()
            if set(values) == required_buckets
            and not _linear_residency_backend_rejected(
                rejected_registry, layer, method, backend
            )
        }
        source = (
            metadata_measurements
            if metadata_measurements
            else next(iter(carrier_measurements.values()), None)
        )
        if source is None:
            continue
        online_ms = _linear_residency_latency_score(
            {bucket: source[bucket][0] for bucket in required_buckets}
        )
        candidates = [
            CubicRuntimeCandidate(
                name="packed-online",
                latency_ms=online_ms,
                extra_memory_bytes=0,
                metadata=(
                    CubicMetadataResidency.COMPACT
                    if method.scheme.metadata_format != "float32-scale-float16-ab"
                    else CubicMetadataResidency.EXPANDED
                ),
                carrier=CubicCarrierResidency.ONLINE,
            )
        ]
        installers: dict[str, str] = {}
        metadata_extra_bytes = 0
        if metadata_measurements:
            metadata_extras = {value[2] for value in metadata_measurements.values()}
            if len(metadata_extras) != 1:
                raise RuntimeError(
                    "Cubic metadata residency measurements disagree by token bucket."
                )
            candidate = CubicRuntimeCandidate(
                name="expanded-metadata",
                latency_ms=_linear_residency_latency_score(
                    {
                        bucket: metadata_measurements[bucket][1]
                        for bucket in required_buckets
                    }
                ),
                extra_memory_bytes=metadata_extras.pop(),
                metadata=CubicMetadataResidency.EXPANDED,
                carrier=CubicCarrierResidency.ONLINE,
            )
            candidates.append(candidate)
            installers[candidate.name] = "metadata"
            metadata_extra_bytes = candidate.extra_memory_bytes
            expanded_buckets = tuple(
                bucket
                for bucket in sorted(required_buckets)
                if metadata_measurements[bucket][1] * 1.03
                < metadata_measurements[bucket][0]
            )
            compact_buckets = tuple(
                bucket
                for bucket in sorted(required_buckets)
                if bucket not in expanded_buckets
            )
            if expanded_buckets and compact_buckets:
                hybrid = CubicRuntimeCandidate(
                    name="expanded-metadata-with-packed-dispatch",
                    latency_ms=_linear_residency_latency_score(
                        {
                            bucket: (
                                metadata_measurements[bucket][1]
                                if bucket in expanded_buckets
                                else metadata_measurements[bucket][0]
                            )
                            for bucket in required_buckets
                        }
                    ),
                    extra_memory_bytes=candidate.extra_memory_bytes,
                    metadata=CubicMetadataResidency.EXPANDED,
                    carrier=CubicCarrierResidency.ONLINE,
                )
                candidates.append(hybrid)
                installers[hybrid.name] = "metadata-hybrid"
                hybrid_expanded_metadata_buckets[prefix] = expanded_buckets
        for measured_backend, measurements in carrier_measurements.items():
            if measured_backend == "exact-marlin":
                continue
            backends = {value[2] for value in measurements.values()}
            extras = {value[3] for value in measurements.values()}
            if backends != {measured_backend} or len(extras) != 1:
                raise RuntimeError(
                    "Cubic carrier residency measurements disagree by token bucket."
                )
            backend = measured_backend
            candidate = CubicRuntimeCandidate(
                name=(
                    "dense-expanded-replaces-packed"
                    if backend == "dense"
                    else f"{backend}-carrier-replaces-packed"
                ),
                latency_ms=_linear_residency_latency_score(
                    {bucket: measurements[bucket][1] for bucket in required_buckets}
                ),
                extra_memory_bytes=extras.pop(),
                metadata=CubicMetadataResidency.EXPANDED,
                carrier=(
                    CubicCarrierResidency.EXPANDED
                    if backend == "dense"
                    else CubicCarrierResidency.PRECOMPUTED
                ),
            )
            metadata_resident_buckets = (
                {
                    bucket
                    for bucket in required_buckets
                    if metadata_measurements[bucket][1] * 1.03
                    < measurements[bucket][1]
                }
                if metadata_measurements
                else set()
            )
            metadata_beats_resident = bool(metadata_resident_buckets)
            if not metadata_beats_resident:
                candidates.append(candidate)
                installers[candidate.name] = backend
            if backend == "dense":
                online_buckets = tuple(
                    bucket
                    for bucket in sorted(required_buckets)
                    if measurements[bucket][0] * 1.03 < measurements[bucket][1]
                )
                dense_buckets = tuple(
                    bucket
                    for bucket in sorted(required_buckets)
                    if bucket not in online_buckets
                )
                if (
                    not metadata_beats_resident
                    and online_buckets
                    and dense_buckets
                    and hasattr(layer, "weight_packed")
                ):
                    packed_bytes = (
                        layer.weight_packed.numel() * layer.weight_packed.element_size()
                    )
                    hybrid = CubicRuntimeCandidate(
                        name="dense-expanded-with-packed-dispatch",
                        latency_ms=_linear_residency_latency_score(
                            {
                                bucket: (
                                    measurements[bucket][0]
                                    if bucket in online_buckets
                                    else measurements[bucket][1]
                                )
                                for bucket in required_buckets
                            }
                        ),
                        extra_memory_bytes=(
                            candidate.extra_memory_bytes + packed_bytes
                        ),
                        metadata=candidate.metadata,
                        carrier=candidate.carrier,
                    )
                    candidates.append(hybrid)
                    installers[hybrid.name] = "dense-hybrid"
                    hybrid_online_buckets[prefix] = online_buckets
                if metadata_measurements:
                    expanded_buckets = tuple(
                        bucket
                        for bucket in sorted(required_buckets)
                        if bucket in metadata_resident_buckets
                    )
                    dense_buckets = tuple(
                        bucket
                        for bucket in sorted(required_buckets)
                        if bucket not in expanded_buckets
                    )
                    if (
                        expanded_buckets
                        and dense_buckets
                        and hasattr(layer, "weight_packed")
                    ):
                        packed_bytes = (
                            layer.weight_packed.numel()
                            * layer.weight_packed.element_size()
                        )
                        hybrid = CubicRuntimeCandidate(
                            name="dense-with-expanded-metadata-dispatch",
                            latency_ms=_linear_residency_latency_score(
                                {
                                    bucket: (
                                        metadata_measurements[bucket][1]
                                        if bucket in expanded_buckets
                                        else measurements[bucket][1]
                                    )
                                    for bucket in required_buckets
                                }
                            ),
                            extra_memory_bytes=(
                                candidate.extra_memory_bytes
                                + packed_bytes
                                + metadata_extra_bytes
                            ),
                            metadata=CubicMetadataResidency.EXPANDED,
                            carrier=CubicCarrierResidency.EXPANDED,
                        )
                        candidates.append(hybrid)
                        installers[hybrid.name] = "dense-metadata-hybrid"
                        hybrid_expanded_metadata_dense_buckets[prefix] = (
                            expanded_buckets
                        )
        exact_measurements = (
            partial_exact_measurements
            if set(partial_exact_measurements) == required_buckets
            else {}
        )
        if exact_measurements and not _linear_residency_backend_rejected(
            rejected_registry, layer, method, "exact-marlin"
        ):
            extras = {value[3] for value in exact_measurements.values()}
            if len(extras) != 1:
                raise RuntimeError(
                    "Exact Cubic Marlin measurements disagree on memory use."
                )
            exact_buckets = CUBIC_TOKEN_BUCKETS
            candidate = CubicRuntimeCandidate(
                name="exact-marlin-with-packed-dispatch",
                latency_ms=_linear_residency_latency_score(
                    {
                        bucket: (
                            exact_measurements[bucket][1]
                            if bucket in exact_measurements
                            else source[bucket][0]
                        )
                        for bucket in required_buckets
                    }
                ),
                extra_memory_bytes=extras.pop(),
                metadata=(
                    CubicMetadataResidency.COMPACT
                    if method.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT
                    else CubicMetadataResidency.EXPANDED
                ),
                carrier=CubicCarrierResidency.PRECOMPUTED,
            )
            candidates.append(candidate)
            installers[candidate.name] = "exact-marlin-hybrid"
            exact_marlin_buckets[prefix] = exact_buckets
        plans.append((members, tuple(candidates), installers))

    states: list[
        tuple[
            torch.nn.Module,
            CubicLinearMethod,
            tuple[CubicRuntimeCandidate, ...],
            dict[str, str],
            CubicRuntimeCandidate,
        ]
    ] = []
    for members, plan_candidates, installers in plans:
        states.extend(
            (layer, method, plan_candidates, installers, plan_candidates[0])
            for layer, method in members
        )
    while True:
        best: tuple[float, float, int, str, int, CubicRuntimeCandidate] | None = None
        for index, (_, _, state_candidates, _, current) in enumerate(states):
            for target in state_candidates:
                extra = target.extra_memory_bytes - current.extra_memory_bytes
                benefit = current.latency_ms - target.latency_ms
                if (
                    extra < 0
                    or extra > remaining
                    or target.latency_ms * 1.01 >= current.latency_ms
                ):
                    continue
                priority = float("inf") if extra == 0 else benefit / extra
                choice = (priority, benefit, -extra, target.name, index, target)
                if best is None or choice[:4] > best[:4]:
                    best = choice
        if best is None:
            break
        _, _, neg_extra, _, index, target = best
        layer, method, state_candidates, installers, _ = states[index]
        states[index] = (layer, method, state_candidates, installers, target)
        remaining += neg_extra

    for layer, method, _, installers, selected in states:
        selected_backend = installers.get(selected.name)
        if selected_backend == "metadata":
            install_cubic_expanded_metadata(layer, method.scheme)
        elif selected_backend == "metadata-hybrid":
            prefix = _linear_residency_key(layer, method, 1)[:-1]
            install_cubic_expanded_metadata(
                layer,
                method.scheme,
                token_buckets=hybrid_expanded_metadata_buckets[prefix],
            )
        elif selected_backend == "dense":
            install_cubic_a16_weight(layer, method.scheme)
        elif selected_backend == "dense-hybrid":
            prefix = _linear_residency_key(layer, method, 1)[:-1]
            install_cubic_a16_weight(
                layer,
                method.scheme,
                retain_packed=True,
                online_buckets=hybrid_online_buckets[prefix],
            )
        elif selected_backend == "dense-metadata-hybrid":
            prefix = _linear_residency_key(layer, method, 1)[:-1]
            expanded_buckets = hybrid_expanded_metadata_dense_buckets[prefix]
            install_cubic_a16_weight(
                layer,
                method.scheme,
                retain_packed=True,
                online_buckets=expanded_buckets,
            )
            install_cubic_expanded_metadata(layer, method.scheme)
        elif selected_backend == "exact-marlin-hybrid":
            prefix = _linear_residency_key(layer, method, 1)[:-1]
            install_cubic_exact_marlin_weight(
                layer,
                method.scheme,
                token_buckets=exact_marlin_buckets[prefix],
            )
        elif selected_backend is not None:
            install_cubic_carrier(
                layer,
                method.scheme,
                dynamic_a8=method.dynamic_a8,
                backend=selected_backend,
            )
        layer.cubic_runtime_residency = selected.name

    for members, plan_candidates, _ in plans:
        layer, method = members[0]
        counts = {
            candidate.name: sum(
                selected.name == candidate.name
                for state_layer, _, _, _, selected in states
                if state_layer in {item[0] for item in members}
            )
            for candidate in plan_candidates
        }
        logger.info(
            "Cubic Linear residency W%d G=%dx%d N=%d K=%d: %s, budget_left=%.2f MiB",
            method.scheme.num_bits,
            method.scheme.group_out,
            method.scheme.group_size,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            counts,
            remaining / (1024**2),
        )


@torch.inference_mode()
def _warmup_cubic_linear_families(
    model: torch.nn.Module,
    token_buckets: tuple[int, ...],
    owned_tasks: set[CalibrationTask] | None = None,
    progress: _CalibrationProgress | None = None,
    calibrate: bool = True,
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        calibrate_cubic_compact_linear_execution,
        calibrate_cubic_linear_execution,
        cubic_linear,
        cubic_linear_compact,
        cubic_linear_dynamic_a8,
        cubic_linear_dynamic_a8_compact,
        cubic_linear_dynamic_a8_precomputed,
        cubic_linear_dynamic_a8_w5_curve2_pair_lut,
        materialize_cubic_compact_a8_carrier,
    )

    token_buckets = tuple(sorted({cubic_linear_token_bucket(m) for m in token_buckets}))
    layer_groups: dict[
        CalibrationTask, list[tuple[torch.nn.Module, CubicLinearMethod]]
    ] = {}
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, CubicLinearMethod):
            continue
        if isinstance(module, VocabParallelEmbedding) and not isinstance(
            module, ParallelLMHead
        ):
            continue
        key = _linear_task_id(module, method, 0)[:-1]
        layer_groups.setdefault(key, []).append((module, method))
    if not calibrate:
        _materialize_cubic_linear_residency(layer_groups)
    layers = {key: members[0] for key, members in layer_groups.items()}

    for layer, method in layers.values():
        bits = method.scheme.num_bits
        group_size = method.scheme.group_size
        group_out = method.scheme.group_out
        k = int(layer.input_size_per_partition)
        n = int(layer.output_size_per_partition)
        assigned_tokens = tuple(
            m
            for m in token_buckets
            if owned_tasks is None or _linear_task_id(layer, method, m) in owned_tasks
        )
        if not assigned_tokens:
            continue
        if getattr(layer, "cubic_weight_packed_is_marlin", False) or getattr(
            layer, "cubic_weight_is_expanded_a16", False
        ):
            if progress is not None:
                for m in assigned_tokens:
                    progress(_linear_task_id(layer, method, m))
            continue
        if getattr(layer, "cubic_weight_packed_is_carrier", False):
            if not method.dynamic_a8:
                raise RuntimeError("A resident Cubic INT8 carrier requires Dynamic-A8.")
            unused_curve = (
                layer.weight_ab
                if method.scheme.metadata_format == CUBIC_COMPACT_METADATA_FORMAT
                else (
                    layer.weight_metadata
                    if method.scheme.metadata_format
                    == CUBIC_E5M9_CURVE2_METADATA_FORMAT
                    else layer.weight_a
                )
            )
            for m in assigned_tokens:
                x = torch.randn(m, k, device="cuda", dtype=layer.params_dtype)
                cubic_linear_dynamic_a8_precomputed(
                    x,
                    layer.weight_packed,
                    layer.weight_scale,
                    unused_curve,
                    unused_curve,
                    num_bits=bits,
                    group_size=group_size,
                    group_out=group_out,
                    input_size=k,
                    _compile_only=True,
                )
                if progress is not None:
                    progress(_linear_task_id(layer, method, m))
            continue
        logger.info(
            "%s Cubic Linear %s: W%d G=%dx%d N=%d K=%d M=%s",
            "Calibrating" if calibrate else "Materializing",
            "A8" if method.dynamic_a8 else "A16",
            bits,
            group_out,
            group_size,
            n,
            k,
            assigned_tokens,
        )
        if method.scheme.metadata_format != "float32-scale-float16-ab":
            compact_op = (
                cubic_linear_dynamic_a8_compact
                if method.dynamic_a8
                else cubic_linear_compact
            )
            scale, coefficient_a, coefficient_b = expanded_cubic_metadata(
                layer, method.scheme
            )
            curve2_metadata = (
                method.scheme.metadata_format
                == CUBIC_E5M9_CURVE2_METADATA_FORMAT
            )
            if curve2_metadata:
                compact_scale = layer.weight_metadata
                compact_a = layer.weight_curve_a
                compact_b = layer.weight_curve_b
                compact_a_global = layer.weight_curve_a
                compact_b_global = layer.weight_curve_b
                compact_metadata_bytes = (
                    layer.weight_metadata.nbytes
                    + layer.weight_curve_a.nbytes
                    + layer.weight_curve_b.nbytes
                )
            else:
                compact_scale = layer.weight_scale
                compact_a = layer.weight_ab
                compact_b = layer.weight_scale_global
                compact_a_global = layer.weight_a_global
                compact_b_global = layer.weight_b_global
                compact_metadata_bytes = (
                    layer.weight_scale.nbytes
                    + layer.weight_ab.nbytes
                    + layer.weight_scale_global.nbytes
                    + layer.weight_a_global.nbytes
                    + layer.weight_b_global.nbytes
                )
            residency_carrier = None
            canonical_carrier = None
            residency_marlin = None
            residency_exact_marlin = None
            residency_expanded = None
            residency_marlin_extra_bytes = 0
            residency_expanded_extra_bytes = 0
            if calibrate:
                driver_free_memory, total_memory = torch.accelerator.get_memory_info(
                    layer.weight_packed.device
                )
                device = layer.weight_packed.device
                reclaimable_cache = max(
                    0,
                    torch.cuda.memory_reserved(device)
                    - torch.cuda.memory_allocated(device),
                )
                free_memory = min(
                    total_memory,
                    driver_free_memory + reclaimable_cache,
                )
                packed_bytes = layer.weight_packed.numel()
                if not method.dynamic_a8 and n * k * 2 <= free_memory:
                    residency_exact_marlin = prepare_cubic_exact_marlin_weight(
                        layer.weight_packed,
                        scale,
                        coefficient_a,
                        coefficient_b,
                        params_dtype=layer.params_dtype,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                    )
                if n * k * 5 <= free_memory:
                    residency_carrier = materialize_cubic_compact_a8_carrier(
                        layer.weight_packed,
                        compact_scale,
                        compact_a,
                        compact_b,
                        compact_a_global,
                        compact_b_global,
                        layer.weight_global_index,
                        num_bits=bits,
                        group_size=group_size,
                        input_size=k,
                        group_out=group_out,
                        e5m9_curve2_metadata=curve2_metadata,
                    )
                    canonical_carrier = materialize_cubic_a8_carrier(
                        layer.weight_packed,
                        coefficient_a,
                        coefficient_b,
                        num_bits=bits,
                        group_size=group_size,
                        input_size=k,
                        group_out=group_out,
                    )
                    if not torch.equal(residency_carrier, canonical_carrier):
                        mismatch = residency_carrier != canonical_carrier
                        mismatch_count = int(mismatch.sum().item())
                        first = mismatch.nonzero()[0]
                        first_n, first_k = (int(first[0]), int(first[1]))
                        first_og = first_n // group_out
                        first_ig = first_k // group_size
                        metadata_value = int(compact_scale[first_og, first_ig])
                        metadata_detail = ""
                        if curve2_metadata:
                            curve_id = metadata_value >> 14
                            global_index = int(
                                layer.weight_global_index[first_og].item()
                            )
                            curve_a_table = compact_a.reshape(-1, 4)
                            curve_b_table = compact_b.reshape(-1, 4)
                            metadata_detail = (
                                f", global_index={global_index}, "
                                f"curve={curve_id}, compact_ab=("
                                f"{float(curve_a_table[global_index, curve_id])}, "
                                f"{float(curve_b_table[global_index, curve_id])})"
                            )
                        raise RuntimeError(
                            "Cubic compact carrier materialization differs from "
                            "the canonical packed-code decoder: "
                            f"W{bits} G={group_out}x{group_size} N={n} K={k}, "
                            f"mismatches={mismatch_count}, first=({first_n}, "
                            f"{first_k}), compact="
                            f"{int(residency_carrier[first_n, first_k])}, "
                            f"canonical="
                            f"{int(canonical_carrier[first_n, first_k])}, "
                            f"metadata=0x{metadata_value:04x}{metadata_detail}, "
                            f"canonical_ab=("
                            f"{float(coefficient_a[first_og, first_ig])}, "
                            f"{float(coefficient_b[first_og, first_ig])})."
                        )
                    activation_group_size = (
                        cubic_dynamic_a8_group_size(
                            input_size=k,
                            weight_group_size=group_size,
                        )
                        if method.dynamic_a8
                        else None
                    )
                    residency_marlin = prepare_cubic_marlin_weight(
                        residency_carrier,
                        scale,
                        params_dtype=layer.params_dtype,
                        group_size=group_size,
                        group_out=group_out,
                        dynamic_a8=method.dynamic_a8,
                        input_group_size=activation_group_size,
                    )
                    residency_marlin_extra_bytes = max(0, n * k - packed_bytes)
                    residency_marlin_extra_bytes += max(
                        0,
                        scale.numel() * scale.element_size()
                        - compact_metadata_bytes,
                    )
                    if residency_marlin is not None:
                        residency_marlin_extra_bytes += (
                            residency_marlin.persistent_bytes
                        )
                if not method.dynamic_a8:
                    expanded_bytes = (
                        n * k * torch.empty((), dtype=layer.params_dtype).element_size()
                    )
                    if expanded_bytes * 4 <= free_memory:
                        residency_expanded = materialize_cubic_a16_weight(
                            layer, method.scheme
                        )
                        residency_expanded_extra_bytes = max(
                            0, expanded_bytes - packed_bytes
                        )
                if (
                    residency_marlin is not None
                    and not method.dynamic_a8
                    and residency_expanded is not None
                    and not _a16_marlin_representation_matches_expanded_weight(
                        residency_carrier,
                        scale,
                        residency_expanded,
                        group_size=group_size,
                        group_out=group_out,
                    )
                ):
                    residency_marlin = None
                    residency_marlin_extra_bytes = 0
            for m in assigned_tokens:
                x = torch.randn(m, k, device="cuda", dtype=layer.params_dtype)
                if calibrate:
                    online_ms = calibrate_cubic_compact_linear_execution(
                        x,
                        layer.weight_packed,
                        compact_scale,
                        compact_a,
                        compact_b,
                        compact_a_global,
                        compact_b_global,
                        layer.weight_global_index,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                        dynamic_a8=method.dynamic_a8,
                        e5m9_curve2_metadata=curve2_metadata,
                        curve_pair_lut=getattr(
                            layer, "weight_curve_pair_lut", None
                        ),
                    )
                    expanded_metadata_ms = calibrate_cubic_compact_linear_execution(
                        x,
                        layer.weight_packed,
                        scale,
                        coefficient_a,
                        coefficient_b,
                        compact_a_global,
                        compact_b_global,
                        layer.weight_global_index,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                        dynamic_a8=method.dynamic_a8,
                        expanded_metadata=True,
                    )
                    expanded_metadata = partial(
                        compact_op,
                        x,
                        layer.weight_packed,
                        scale,
                        coefficient_a,
                        coefficient_b,
                        compact_a_global,
                        compact_b_global,
                        layer.weight_global_index,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                        _expanded_metadata=True,
                    )
                    has_canonical = (
                        method.dynamic_a8 and canonical_carrier is not None
                    ) or (not method.dynamic_a8 and residency_expanded is not None)
                    if (
                        online_ms is not None
                        and expanded_metadata_ms is not None
                        and has_canonical
                    ):
                        pair_lut = getattr(layer, "weight_curve_pair_lut", None)
                        if method.dynamic_a8 and pair_lut is not None and m <= 8:
                            online = partial(
                                cubic_linear_dynamic_a8_w5_curve2_pair_lut,
                                x,
                                layer.weight_packed,
                                layer.weight_metadata,
                                layer.weight_global_index,
                                pair_lut,
                                group_size=group_size,
                                group_out=group_out,
                                input_size=k,
                            )
                        else:
                            online = partial(
                                compact_op,
                                x,
                                layer.weight_packed,
                                compact_scale,
                                compact_a,
                                compact_b,
                                compact_a_global,
                                compact_b_global,
                                layer.weight_global_index,
                                num_bits=bits,
                                group_size=group_size,
                                group_out=group_out,
                                input_size=k,
                                _e5m9_curve2_metadata=curve2_metadata,
                            )
                        canonical = (
                            partial(
                                cubic_linear_dynamic_a8_precomputed,
                                x,
                                canonical_carrier,
                                scale,
                                coefficient_a,
                                coefficient_b,
                                num_bits=bits,
                                group_size=group_size,
                                group_out=group_out,
                                input_size=k,
                            )
                            if method.dynamic_a8
                            else partial(
                                torch.nn.functional.linear,
                                x,
                                residency_expanded,
                            )
                        )
                        marlin_canonical = (
                            partial(
                                cubic_linear_dynamic_a8_precomputed,
                                x,
                                canonical_carrier,
                                scale.to(torch.float16).to(scale.dtype),
                                coefficient_a,
                                coefficient_b,
                                num_bits=bits,
                                group_size=group_size,
                                group_out=group_out,
                                input_size=k,
                            )
                            if method.dynamic_a8
                            else canonical
                        )
                        probe_indices = _cubic_linear_probe_indices(k, group_size)
                        if method.dynamic_a8:
                            canonical_columns = _cubic_a8_expected_columns(
                                canonical_carrier,
                                scale,
                                probe_indices,
                                group_size=group_size,
                                group_out=group_out,
                                dtype=layer.params_dtype,
                            )
                            marlin_columns = _cubic_a8_expected_columns(
                                canonical_carrier,
                                scale.to(torch.float16).to(scale.dtype),
                                probe_indices,
                                group_size=group_size,
                                group_out=group_out,
                                dtype=layer.params_dtype,
                            )
                        else:
                            canonical_columns = residency_expanded[
                                :, probe_indices.to(residency_expanded.device)
                            ].T
                            marlin_columns = canonical_columns

                        pair_lut = getattr(layer, "weight_curve_pair_lut", None)
                        if method.dynamic_a8 and pair_lut is not None:
                            online_probe = partial(
                                _invoke_cubic_probe,
                                operation=cubic_linear_dynamic_a8_w5_curve2_pair_lut,
                                args=(
                                    layer.weight_packed,
                                    layer.weight_metadata,
                                    layer.weight_global_index,
                                    pair_lut,
                                ),
                                kwargs={
                                    "group_size": group_size,
                                    "group_out": group_out,
                                    "input_size": k,
                                },
                            )
                        else:
                            online_probe = partial(
                                _invoke_cubic_probe,
                                operation=compact_op,
                                args=(
                                    layer.weight_packed,
                                    compact_scale,
                                    compact_a,
                                    compact_b,
                                    compact_a_global,
                                    compact_b_global,
                                    layer.weight_global_index,
                                ),
                                kwargs={
                                    "num_bits": bits,
                                    "group_size": group_size,
                                    "group_out": group_out,
                                    "input_size": k,
                                    "_e5m9_curve2_metadata": curve2_metadata,
                                },
                            )

                        _validate_cubic_linear_column_mapping(
                            online_probe,
                            canonical_columns,
                            probe_indices,
                            input_size=k,
                            dtype=layer.params_dtype,
                            device=layer.weight_packed.device,
                        )
                        try:
                            _validate_cubic_resident_output(
                                online(), canonical()
                            )
                        except AssertionError as error:
                            logger.warning(
                                "Rejecting Cubic Linear online implementation "
                                "for W%d G=%dx%d N=%d K=%d M=%d: %s",
                                bits,
                                group_out,
                                group_size,
                                n,
                                k,
                                m,
                                error,
                            )
                        else:
                            _validate_cubic_resident_output(
                                expanded_metadata(), canonical()
                            )
                            from vllm.model_executor.layers.quantization import (
                                cubic_kernels,
                            )

                            cubic_kernels._CUBIC_LINEAR_METADATA_RESIDENCY_TACTICS[
                                _linear_residency_key(layer, method, m)
                            ] = (
                                online_ms,
                                expanded_metadata_ms,
                                scale.nbytes
                                + coefficient_a.nbytes
                                + coefficient_b.nbytes,
                            )
                    residents: list[
                        tuple[str, Callable[[torch.Tensor], torch.Tensor], int]
                    ] = []
                    if residency_exact_marlin is not None:
                        residents.append(
                            (
                                "exact-marlin",
                                partial(
                                    apply_cubic_exact_marlin_weight,
                                    prepared=residency_exact_marlin,
                                    output_size=n,
                                    input_size=k,
                                ),
                                residency_exact_marlin.persistent_bytes,
                            )
                        )
                    if residency_marlin is not None:
                        residents.append(
                            (
                                "marlin",
                                partial(
                                    apply_cubic_marlin_weight,
                                    prepared=residency_marlin,
                                    output_size=n,
                                    input_size=k,
                                    dynamic_a8=method.dynamic_a8,
                                ),
                                residency_marlin_extra_bytes,
                            )
                        )
                    if not method.dynamic_a8 and residency_expanded is not None:
                        residents.append(
                            (
                                "dense",
                                partial(
                                    torch.nn.functional.linear,
                                    weight=residency_expanded,
                                ),
                                residency_expanded_extra_bytes,
                            )
                        )
                    if method.dynamic_a8 and residency_carrier is not None:
                        calibrate_cubic_linear_execution(
                            x,
                            residency_carrier,
                            scale,
                            coefficient_a,
                            coefficient_b,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                            dynamic_a8=True,
                            precomputed_carrier=True,
                        )
                        residents.append(
                            (
                                "triton",
                                partial(
                                    cubic_linear_dynamic_a8_precomputed,
                                    carrier=residency_carrier,
                                    scale=scale,
                                    a=coefficient_a,
                                    b=coefficient_b,
                                    num_bits=bits,
                                    group_size=group_size,
                                    group_out=group_out,
                                    input_size=k,
                                ),
                                residency_marlin_extra_bytes,
                            )
                        )
                    if online_ms is not None and residents:
                        import triton

                        online = partial(
                            compact_op,
                            x,
                            layer.weight_packed,
                            compact_scale,
                            compact_a,
                            compact_b,
                            compact_a_global,
                            compact_b_global,
                            layer.weight_global_index,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                            _e5m9_curve2_metadata=curve2_metadata,
                        )
                        from vllm.model_executor.layers.quantization import (
                            cubic_kernels,
                        )

                        for backend, operation, extra_bytes in residents:
                            resident = partial(operation, x)
                            _validate_cubic_linear_column_mapping(
                                operation,
                                marlin_columns
                                if backend == "marlin"
                                else canonical_columns,
                                probe_indices,
                                input_size=k,
                                dtype=layer.params_dtype,
                                device=layer.weight_packed.device,
                            )
                            try:
                                _validate_cubic_batch_invariance(operation, x)
                            except AssertionError as error:
                                cubic_kernels._CUBIC_LINEAR_REJECTED_RESIDENCIES[
                                    _linear_residency_backend_key(
                                        layer, method, backend
                                    )
                                ] = True
                                logger.warning(
                                    "Rejecting Cubic Linear %s residency for "
                                    "W%d G=%dx%d N=%d K=%d M=%d: shared-row "
                                    "output is batch-dependent: %s",
                                    backend,
                                    bits,
                                    group_out,
                                    group_size,
                                    n,
                                    k,
                                    m,
                                    error,
                                )
                                continue
                            try:
                                expected = (
                                    marlin_canonical
                                    if backend == "marlin"
                                    else canonical
                                )
                                _validate_cubic_resident_output(
                                    resident(),
                                    expected(),
                                )
                            except AssertionError as error:
                                cubic_kernels._CUBIC_LINEAR_REJECTED_RESIDENCIES[
                                    _linear_residency_backend_key(
                                        layer, method, backend
                                    )
                                ] = True
                                logger.warning(
                                    "Rejecting Cubic Linear %s residency for "
                                    "W%d G=%dx%d N=%d K=%d M=%d: %s",
                                    backend,
                                    bits,
                                    group_out,
                                    group_size,
                                    n,
                                    k,
                                    m,
                                    error,
                                )
                                continue
                            key = _linear_residency_key(layer, method, m)
                            cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[
                                key + (backend,)
                            ] = (
                                online_ms,
                                triton.testing.do_bench(resident, warmup=10, rep=30),
                                backend,
                                extra_bytes,
                            )
                else:
                    compact_op(
                        x,
                        layer.weight_packed,
                        compact_scale,
                        compact_a,
                        compact_b,
                        compact_a_global,
                        compact_b_global,
                        layer.weight_global_index,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                        _compile_only=True,
                        _e5m9_curve2_metadata=curve2_metadata,
                    )
                if progress is not None:
                    progress(_linear_task_id(layer, method, m))
            continue
        a16_a, a16_b = layer.weight_a, layer.weight_b
        a8_a, a8_b = layer.weight_a, layer.weight_b
        if bits == 3 and layer.weight_a.dtype == torch.int8:
            a16_a = torch.full_like(layer.weight_a, 0.5, dtype=torch.float16)
            a16_b = torch.full_like(layer.weight_b, 0.25, dtype=torch.float16)
        elif bits == 3:
            levels = cubic_carrier_levels(3, layer.weight_a, layer.weight_b)
            a8_a, a8_b = levels[..., 1].contiguous(), levels[..., 2].contiguous()
        residency_carrier = None
        residency_marlin = None
        residency_expanded = None
        residency_marlin_extra_bytes = 0
        residency_expanded_extra_bytes = 0
        if calibrate:
            free_memory, _ = torch.accelerator.get_memory_info(
                layer.weight_packed.device
            )
            carrier_bytes = n * k
            memory = cubic_runtime_memory(
                num_values=n * k,
                num_groups=layer.weight_scale.numel(),
                num_bits=bits,
            )
            if carrier_bytes * 5 <= free_memory:
                residency_carrier = materialize_cubic_a8_carrier(
                    layer.weight_packed,
                    layer.weight_a,
                    layer.weight_b,
                    num_bits=bits,
                    group_size=group_size,
                    input_size=k,
                    group_out=group_out,
                )
                activation_group_size = (
                    cubic_dynamic_a8_group_size(
                        input_size=k,
                        weight_group_size=group_size,
                    )
                    if method.dynamic_a8
                    else None
                )
                residency_marlin = prepare_cubic_marlin_weight(
                    residency_carrier,
                    layer.weight_scale,
                    params_dtype=layer.params_dtype,
                    group_size=group_size,
                    group_out=group_out,
                    dynamic_a8=method.dynamic_a8,
                    input_group_size=activation_group_size,
                )
                residency_marlin_extra_bytes = memory.carrier_replacement_extra_bytes
                if residency_marlin is not None:
                    residency_marlin_extra_bytes += residency_marlin.persistent_bytes
            if not method.dynamic_a8:
                expanded_bytes = (
                    n * k * torch.empty((), dtype=layer.params_dtype).element_size()
                )
                if expanded_bytes * 4 <= free_memory:
                    residency_expanded = materialize_cubic_a16_weight(
                        layer, method.scheme
                    )
                    residency_expanded_extra_bytes = max(
                        0, expanded_bytes - memory.packed_weight_bytes
                    )
            if (
                residency_marlin is not None
                and not method.dynamic_a8
                and residency_expanded is not None
                and not _a16_marlin_representation_matches_expanded_weight(
                    residency_carrier,
                    layer.weight_scale,
                    residency_expanded,
                    group_size=group_size,
                    group_out=group_out,
                )
            ):
                residency_marlin = None
                residency_marlin_extra_bytes = 0
        modes: list[
            tuple[
                bool,
                Callable[..., torch.Tensor],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                bool,
            ]
        ]
        if method.dynamic_a8:
            carrier = (
                layer.weight_packed
                if getattr(layer, "cubic_weight_packed_is_carrier", False)
                else getattr(layer, "weight_carrier", None)
            )
            if carrier is None:
                modes = [
                    (
                        True,
                        cubic_linear_dynamic_a8,
                        layer.weight_packed,
                        a8_a,
                        a8_b,
                        False,
                    )
                ]
            else:
                modes = [
                    (
                        True,
                        cubic_linear_dynamic_a8_precomputed,
                        carrier,
                        a8_a,
                        a8_b,
                        True,
                    )
                ]
        else:
            modes = [(False, cubic_linear, layer.weight_packed, a16_a, a16_b, False)]
        for m in token_buckets:
            task = _linear_task_id(layer, method, m)
            if owned_tasks is not None and task not in owned_tasks:
                continue
            x = torch.randn(m, k, device="cuda", dtype=layer.params_dtype)
            phase_index = 0
            phase_total = (2 if calibrate else 1) * len(modes)
            if calibrate:
                has_resident_candidate = (
                    residency_marlin is not None
                    or (method.dynamic_a8 and residency_carrier is not None)
                    or residency_expanded is not None
                )
                if not has_resident_candidate:
                    for (
                        dynamic_a8,
                        _,
                        weight,
                        coefficient_a,
                        coefficient_b,
                        precomputed_carrier,
                    ) in modes:
                        calibrate_cubic_linear_execution(
                            x,
                            weight,
                            layer.weight_scale,
                            coefficient_a,
                            coefficient_b,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                            dynamic_a8=dynamic_a8,
                            precomputed_carrier=precomputed_carrier,
                        )
                        phase_index += 1
                        if progress is not None:
                            progress.phase(
                                task,
                                phase_index,
                                phase_total,
                                f"{'A8' if dynamic_a8 else 'A16'} execution tactics",
                            )
                has_canonical = (
                    method.dynamic_a8 and residency_carrier is not None
                ) or (not method.dynamic_a8 and residency_expanded is not None)
                if has_canonical:
                    import triton

                    coefficient_a = a8_a if method.dynamic_a8 else a16_a
                    coefficient_b = a8_b if method.dynamic_a8 else a16_b
                    online_func = (
                        cubic_linear_dynamic_a8 if method.dynamic_a8 else cubic_linear
                    )
                    online = partial(
                        online_func,
                        x,
                        layer.weight_packed,
                        layer.weight_scale,
                        coefficient_a,
                        coefficient_b,
                        num_bits=bits,
                        group_size=group_size,
                        group_out=group_out,
                        input_size=k,
                    )
                    canonical = (
                        partial(
                            cubic_linear_dynamic_a8_precomputed,
                            x,
                            residency_carrier,
                            layer.weight_scale,
                            a8_a,
                            a8_b,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                        )
                        if method.dynamic_a8
                        else partial(
                            torch.nn.functional.linear,
                            x,
                            residency_expanded,
                        )
                    )
                    marlin_canonical = (
                        partial(
                            cubic_linear_dynamic_a8_precomputed,
                            x,
                            residency_carrier,
                            layer.weight_scale.to(torch.float16).to(
                                layer.weight_scale.dtype
                            ),
                            a8_a,
                            a8_b,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                        )
                        if method.dynamic_a8
                        else canonical
                    )
                    probe_indices = _cubic_linear_probe_indices(k, group_size)
                    if method.dynamic_a8:
                        canonical_columns = _cubic_a8_expected_columns(
                            residency_carrier,
                            layer.weight_scale,
                            probe_indices,
                            group_size=group_size,
                            group_out=group_out,
                            dtype=layer.params_dtype,
                        )
                        marlin_columns = _cubic_a8_expected_columns(
                            residency_carrier,
                            layer.weight_scale.to(torch.float16).to(
                                layer.weight_scale.dtype
                            ),
                            probe_indices,
                            group_size=group_size,
                            group_out=group_out,
                            dtype=layer.params_dtype,
                        )
                    else:
                        canonical_columns = residency_expanded[
                            :, probe_indices.to(residency_expanded.device)
                        ].T
                        marlin_columns = canonical_columns
                    online_probe = partial(
                        _invoke_cubic_probe,
                        operation=online_func,
                        args=(
                            layer.weight_packed,
                            layer.weight_scale,
                            coefficient_a,
                            coefficient_b,
                        ),
                        kwargs={
                            "num_bits": bits,
                            "group_size": group_size,
                            "group_out": group_out,
                            "input_size": k,
                        },
                    )
                    _validate_cubic_linear_column_mapping(
                        online_probe,
                        canonical_columns,
                        probe_indices,
                        input_size=k,
                        dtype=layer.params_dtype,
                        device=layer.weight_packed.device,
                    )
                    _validate_cubic_resident_output(online(), canonical())
                    mode_residents: list[
                        tuple[str, Callable[[torch.Tensor], torch.Tensor], int]
                    ] = []
                    if residency_marlin is not None:
                        mode_residents.append(
                            (
                                "marlin",
                                partial(
                                    apply_cubic_marlin_weight,
                                    prepared=residency_marlin,
                                    output_size=n,
                                    input_size=k,
                                    dynamic_a8=method.dynamic_a8,
                                ),
                                residency_marlin_extra_bytes,
                            )
                        )
                    if residency_expanded is not None:
                        mode_residents.append(
                            (
                                "dense",
                                partial(
                                    torch.nn.functional.linear,
                                    weight=residency_expanded,
                                ),
                                residency_expanded_extra_bytes,
                            )
                        )
                    if method.dynamic_a8 and residency_carrier is not None:
                        calibrate_cubic_linear_execution(
                            x,
                            residency_carrier,
                            layer.weight_scale,
                            a8_a,
                            a8_b,
                            num_bits=bits,
                            group_size=group_size,
                            group_out=group_out,
                            input_size=k,
                            dynamic_a8=True,
                            precomputed_carrier=True,
                        )
                        mode_residents.append(
                            (
                                "triton",
                                partial(
                                    cubic_linear_dynamic_a8_precomputed,
                                    carrier=residency_carrier,
                                    scale=layer.weight_scale,
                                    a=a8_a,
                                    b=a8_b,
                                    num_bits=bits,
                                    group_size=group_size,
                                    group_out=group_out,
                                    input_size=k,
                                ),
                                residency_marlin_extra_bytes,
                            )
                        )
                    if mode_residents:
                        from vllm.model_executor.layers.quantization import (
                            cubic_kernels,
                        )

                        online_ms = triton.testing.do_bench(online, warmup=10, rep=30)
                        for backend, operation, extra_bytes in mode_residents:
                            resident = partial(operation, x)
                            _validate_cubic_linear_column_mapping(
                                operation,
                                marlin_columns
                                if backend == "marlin"
                                else canonical_columns,
                                probe_indices,
                                input_size=k,
                                dtype=layer.params_dtype,
                                device=layer.weight_packed.device,
                            )
                            try:
                                _validate_cubic_batch_invariance(operation, x)
                            except AssertionError as error:
                                cubic_kernels._CUBIC_LINEAR_REJECTED_RESIDENCIES[
                                    _linear_residency_backend_key(
                                        layer, method, backend
                                    )
                                ] = True
                                logger.warning(
                                    "Rejecting Cubic Linear %s residency for "
                                    "W%d G=%dx%d N=%d K=%d M=%d: shared-row "
                                    "output is batch-dependent: %s",
                                    backend,
                                    bits,
                                    group_out,
                                    group_size,
                                    n,
                                    k,
                                    m,
                                    error,
                                )
                                continue
                            try:
                                expected = (
                                    marlin_canonical
                                    if backend == "marlin"
                                    else canonical
                                )
                                _validate_cubic_resident_output(
                                    resident(), expected()
                                )
                            except AssertionError as error:
                                cubic_kernels._CUBIC_LINEAR_REJECTED_RESIDENCIES[
                                    _linear_residency_backend_key(
                                        layer, method, backend
                                    )
                                ] = True
                                logger.warning(
                                    "Rejecting Cubic Linear %s residency for "
                                    "W%d G=%dx%d N=%d K=%d M=%d: %s",
                                    backend,
                                    bits,
                                    group_out,
                                    group_size,
                                    n,
                                    k,
                                    m,
                                    error,
                                )
                                continue
                            key = _linear_residency_key(layer, method, m)
                            cubic_kernels._CUBIC_LINEAR_RESIDENCY_TACTICS[
                                key + (backend,)
                            ] = (
                                online_ms,
                                triton.testing.do_bench(resident, warmup=10, rep=30),
                                backend,
                                extra_bytes,
                            )
            for (
                dynamic_a8,
                func,
                weight,
                coefficient_a,
                coefficient_b,
                _,
            ) in modes:
                kwargs = dict(
                    num_bits=bits,
                    group_size=group_size,
                    group_out=group_out,
                    input_size=k,
                )
                output = func(
                    x,
                    weight,
                    layer.weight_scale,
                    coefficient_a,
                    coefficient_b,
                    **kwargs,
                )
                if not torch.isfinite(output).all():
                    raise AssertionError(
                        f"Non-finite Cubic Linear calibration output for W{bits}."
                    )
                phase_index += 1
                if progress is not None:
                    progress.phase(
                        task,
                        phase_index,
                        phase_total,
                        f"{'A8' if dynamic_a8 else 'A16'} correctness",
                    )
            if progress is not None:
                progress(task)


def _synthetic_routes(
    layer: torch.nn.Module, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    top_k = int(layer.top_k)
    expert_map = getattr(layer, "expert_map", None)
    if expert_map is None:
        topk_ids = torch.arange(
            tokens * top_k, device="cuda", dtype=torch.int32
        ).remainder_(layer.global_num_experts)
    else:
        local = (expert_map >= 0).nonzero().flatten().to(torch.int32)
        remote = (expert_map < 0).nonzero().flatten().to(torch.int32)
        if local.numel() == 0:
            raise ValueError("Cubic calibration found no local experts.")
        total_routes = tokens * top_k
        expected_local = max(
            1,
            round(total_routes * local.numel() / layer.global_num_experts),
        )
        if remote.numel() == 0:
            topk_ids = local[
                torch.arange(total_routes, device="cuda").remainder(local.numel())
            ]
        else:
            topk_ids = remote[
                torch.arange(total_routes, device="cuda").remainder(remote.numel())
            ]
            local_positions = (
                torch.linspace(
                    0,
                    total_routes - 1,
                    expected_local,
                    device="cuda",
                )
                .round()
                .long()
            )
            topk_ids[local_positions] = local[
                torch.arange(expected_local, device="cuda").remainder(local.numel())
            ]
    topk_ids = topk_ids.view(tokens, top_k)
    topk_weights = torch.full(
        (tokens, top_k), 1.0 / top_k, device="cuda", dtype=torch.float32
    )
    return topk_weights, topk_ids


@torch.inference_mode()
def _warmup_cubic_moe_families(
    model: torch.nn.Module,
    token_buckets: tuple[int, ...],
    owned_tasks: set[CalibrationTask] | None = None,
    progress: _CalibrationProgress | None = None,
    calibrate: bool = True,
    graph_capture_sizes: tuple[int, ...] = (),
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_a8_moe_grouping,
        calibrate_cubic_a8_moe_backend,
        calibrate_cubic_a8_moe_grouping,
        calibrate_cubic_a8_moe_layer_backends,
        calibrate_cubic_moe_execution,
        calibrate_cubic_moe_route_ctas,
        calibrate_cubic_moe_sum_backend,
        cubic_fused_moe,
        cubic_fused_moe_dynamic_a8,
    )

    layers: dict[CalibrationTask, tuple[torch.nn.Module, CubicMoEMethod]] = {}
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, CubicMoEMethod):
            continue
        key = _moe_task_id(module, method, 0)[:-1]
        layers.setdefault(key, (module, method))

    for layer, method in layers.values():
        bits = method.scheme.num_bits
        group_size = method.scheme.group_size
        group_out = method.scheme.group_out
        hidden = int(layer.cubic_hidden_size)
        intermediate = int(layer.cubic_intermediate_size)
        top_k = int(layer.top_k)
        experts = int(layer.w13_weight_packed.shape[0])
        assigned_tokens = tuple(
            m
            for m in token_buckets
            if owned_tasks is None or _moe_task_id(layer, method, m) in owned_tasks
        )
        if not assigned_tokens:
            continue
        logger.info(
            "%s Cubic MoE %s: W%d G=%dx%d H=%d I=%d top_k=%d experts=%d M=%s",
            "Calibrating" if calibrate else "Materializing",
            "A8" if method.dynamic_a8 else "A16",
            bits,
            group_out,
            group_size,
            hidden,
            intermediate,
            top_k,
            experts,
            assigned_tokens,
        )
        a16_coefficients: tuple[torch.Tensor, ...] | None = None
        a8_coefficients: tuple[torch.Tensor, ...] | None = None
        if bits == 3 and layer.w13_weight_a.dtype == torch.int8:
            # Dynamic-A8 stores precomputed carrier levels in a/b. A16 needs
            # FP16 polynomial coefficients; synthetic values preserve the
            # exact production shapes without mutating the loaded checkpoint.
            a16_coefficients = tuple(
                torch.full_like(value, 0.5, dtype=torch.float16)
                for value in (
                    layer.w13_weight_a,
                    layer.w13_weight_b,
                    layer.w2_weight_a,
                    layer.w2_weight_b,
                )
            )
        elif bits == 3:
            carrier_metadata: list[torch.Tensor] = []
            for coefficient_a, coefficient_b in (
                (layer.w13_weight_a, layer.w13_weight_b),
                (layer.w2_weight_a, layer.w2_weight_b),
            ):
                levels = cubic_carrier_levels(3, coefficient_a, coefficient_b)
                carrier_metadata.extend(
                    (levels[..., 1].contiguous(), levels[..., 2].contiguous())
                )
            a8_coefficients = tuple(carrier_metadata)
        for tokens in token_buckets:
            task = _moe_task_id(layer, method, tokens)
            if owned_tasks is not None and task not in owned_tasks:
                continue
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            topk_weights, topk_ids = _synthetic_routes(layer, tokens)
            if calibrate:
                calibrate_cubic_moe_sum_backend(
                    topk_ids,
                    layer.expert_map,
                    hidden,
                    x.dtype,
                )
            carrier_reuse = group_size / (1 << (bits - 1)) if bits > 1 else 0
            a8_a, a8_b, a8_w2_a, a8_w2_b = (
                a8_coefficients
                if a8_coefficients is not None
                else (
                    layer.w13_weight_a,
                    layer.w13_weight_b,
                    layer.w2_weight_a,
                    layer.w2_weight_b,
                )
            )
            a16_a, a16_b, a16_w2_a, a16_w2_b = (
                a16_coefficients
                if a16_coefficients is not None
                else (
                    layer.w13_weight_a,
                    layer.w13_weight_b,
                    layer.w2_weight_a,
                    layer.w2_weight_b,
                )
            )
            if not calibrate:
                dynamic_a8 = method.dynamic_a8
                gate_a, gate_b, down_a, down_b = (
                    (a8_a, a8_b, a8_w2_a, a8_w2_b)
                    if dynamic_a8
                    else (a16_a, a16_b, a16_w2_a, a16_w2_b)
                )
                func = cubic_fused_moe_dynamic_a8 if dynamic_a8 else cubic_fused_moe
                output = func(
                    x,
                    layer.w13_weight_packed,
                    layer.w2_weight_packed,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    gate_a,
                    gate_b,
                    down_a,
                    down_b,
                    topk_weights,
                    topk_ids,
                    activation=layer.activation,
                    apply_router_weight_on_input=(layer.apply_router_weight_on_input),
                    global_num_experts=layer.global_num_experts,
                    expert_map=layer.expert_map,
                    num_bits=bits,
                    group_out=group_out,
                    group_size=group_size,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    activation_situ_beta=method.moe.activation_situ_beta,
                    activation_situ_linear_beta=(
                        method.moe.activation_situ_linear_beta
                    ),
                )
                if not torch.isfinite(output).all():
                    raise AssertionError(
                        f"Non-finite Cubic MoE materialization output for W{bits}."
                    )
                continue
            has_cuda_candidate = (
                (bits == 2 and group_size in (256, 512))
                or (
                    bits == 3
                    and a8_a.dtype == torch.int8
                    and group_size in (128, 256, 512)
                )
            ) or (4 <= bits <= 8 and group_size in (128, 256, 512))
            grouped_routes = (
                2
                if group_size in (256, 512)
                and (
                    bits == 3
                    or (4 <= bits <= 8 and tokens >= 16 and carrier_reuse >= 2)
                )
                else 1
            )
            can_calibrate_grouping = (
                method.dynamic_a8
                and tokens <= 1024
                and group_size in (128, 256, 512)
                and (bits == 2 or has_cuda_candidate)
            )
            phase_index = 0
            if can_calibrate_grouping:
                calibrate_cubic_a8_moe_grouping(
                    x,
                    layer.w13_weight_packed,
                    layer.w2_weight_packed,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    a8_a,
                    a8_b,
                    a8_w2_a,
                    a8_w2_b,
                    topk_weights,
                    topk_ids,
                    activation=layer.activation,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    global_num_experts=layer.global_num_experts,
                    expert_map=layer.expert_map,
                    num_bits=bits,
                    group_size=group_size,
                    group_out=group_out,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    activation_situ_beta=method.moe.activation_situ_beta,
                    activation_situ_linear_beta=(
                        method.moe.activation_situ_linear_beta
                    ),
                    cuda_graph_replay=tokens in graph_capture_sizes,
                )
                grouped_routes = _cubic_a8_moe_grouping(
                    num_bits=bits,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    group_size=group_size,
                    group_out=group_out,
                    local_experts=experts,
                    num_tokens=tokens,
                    fallback=grouped_routes,
                    precomputed_3bit_levels=(
                        bits == 3
                        and a8_a.dtype == torch.int8
                        and a8_b.dtype == torch.int8
                        and a8_w2_a.dtype == torch.int8
                        and a8_w2_b.dtype == torch.int8
                    ),
                    fp16_curve=(
                        a8_a.dtype == torch.float16
                        and a8_b.dtype == torch.float16
                        and a8_w2_a.dtype == torch.float16
                        and a8_w2_b.dtype == torch.float16
                    ),
                )
            # Singleton routes have a generic Triton competitor for W2-W8;
            # paired W2 additionally has its dedicated Triton pair kernel.
            # Other paired widths are CUDA-only and need no backend contest.
            calibrate_backend = (
                method.dynamic_a8
                and has_cuda_candidate
                and tokens <= 128
                and (grouped_routes == 1 or bits == 2)
            )
            # Two route-CTA projections, execution selection, correctness,
            # plus Dynamic-A8-only grouping/backend comparisons.
            phase_total = 4 + int(can_calibrate_grouping) + 3 * int(calibrate_backend)
            if can_calibrate_grouping:
                phase_index += 1
                if progress is not None:
                    progress.phase(task, phase_index, phase_total, "A8 route grouping")
            down_inputs = torch.randn(
                tokens * top_k,
                intermediate,
                device="cuda",
                dtype=torch.bfloat16,
            )
            down_topk_weights = topk_weights.reshape(-1, 1)
            down_topk_ids = topk_ids.reshape(-1, 1)
            dynamic_a8 = method.dynamic_a8
            coefficients = (
                (a8_a, a8_b, a8_w2_a, a8_w2_b)
                if dynamic_a8
                else (a16_a, a16_b, a16_w2_a, a16_w2_b)
            )
            gate_a, gate_b, down_a, down_b = coefficients
            route_grouping = grouped_routes if dynamic_a8 else 1
            calibrate_cubic_moe_route_ctas(
                x,
                layer.w13_weight_packed,
                layer.w13_weight_scale,
                gate_a,
                gate_b,
                topk_weights,
                topk_ids,
                layer.expert_map,
                dynamic_a8=dynamic_a8,
                global_num_experts=layer.global_num_experts,
                logical_k=hidden,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                top_k=top_k,
                multiply_routed_weight=layer.apply_router_weight_on_input,
                grouped_routes=route_grouping,
                situ_beta=(
                    method.moe.activation_situ_beta
                    if bits == 2 and layer.activation == MoEActivation.SITU
                    else None
                ),
                situ_linear_beta=(
                    method.moe.activation_situ_linear_beta
                    if bits == 2 and layer.activation == MoEActivation.SITU
                    else None
                ),
            )
            phase_index += 1
            if progress is not None:
                progress.phase(
                    task,
                    phase_index,
                    phase_total,
                    f"{'A8' if dynamic_a8 else 'A16'} W13 route CTA",
                )
            calibrate_cubic_moe_route_ctas(
                down_inputs,
                layer.w2_weight_packed,
                layer.w2_weight_scale,
                down_a,
                down_b,
                down_topk_weights,
                down_topk_ids,
                layer.expert_map,
                dynamic_a8=dynamic_a8,
                global_num_experts=layer.global_num_experts,
                logical_k=intermediate,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                top_k=1,
                multiply_routed_weight=(not layer.apply_router_weight_on_input),
                grouped_routes=route_grouping,
            )
            phase_index += 1
            if progress is not None:
                progress.phase(
                    task,
                    phase_index,
                    phase_total,
                    f"{'A8' if dynamic_a8 else 'A16'} W2 route CTA",
                )
            if calibrate_backend:
                calibrate_cubic_a8_moe_backend(
                    x,
                    layer.w13_weight_packed,
                    layer.w13_weight_scale,
                    a8_a,
                    a8_b,
                    topk_weights,
                    topk_ids,
                    layer.expert_map,
                    global_num_experts=layer.global_num_experts,
                    logical_k=hidden,
                    num_bits=bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=top_k,
                    multiply_routed_weight=layer.apply_router_weight_on_input,
                    grouped_routes=grouped_routes,
                )
                phase_index += 1
                if progress is not None:
                    progress.phase(task, phase_index, phase_total, "A8 W13 backend")
                calibrate_cubic_a8_moe_backend(
                    down_inputs,
                    layer.w2_weight_packed,
                    layer.w2_weight_scale,
                    a8_w2_a,
                    a8_w2_b,
                    down_topk_weights,
                    down_topk_ids,
                    layer.expert_map,
                    global_num_experts=layer.global_num_experts,
                    logical_k=intermediate,
                    num_bits=bits,
                    group_size=group_size,
                    group_out=group_out,
                    top_k=1,
                    multiply_routed_weight=not layer.apply_router_weight_on_input,
                    grouped_routes=grouped_routes,
                )
                phase_index += 1
                if progress is not None:
                    progress.phase(task, phase_index, phase_total, "A8 W2 backend")
                calibrate_cubic_a8_moe_layer_backends(
                    x,
                    layer.w13_weight_packed,
                    layer.w2_weight_packed,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    a8_a,
                    a8_b,
                    a8_w2_a,
                    a8_w2_b,
                    topk_weights,
                    topk_ids,
                    activation=layer.activation,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    global_num_experts=layer.global_num_experts,
                    expert_map=layer.expert_map,
                    num_bits=bits,
                    group_out=group_out,
                    group_size=group_size,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    activation_situ_beta=method.moe.activation_situ_beta,
                    activation_situ_linear_beta=(
                        method.moe.activation_situ_linear_beta
                    ),
                )
                phase_index += 1
                if progress is not None:
                    progress.phase(task, phase_index, phase_total, "A8 layer backend")
            common = dict(
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation_situ_beta=method.moe.activation_situ_beta,
                activation_situ_linear_beta=method.moe.activation_situ_linear_beta,
            )
            execution_common = dict(
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation_situ_beta=method.moe.activation_situ_beta,
                activation_situ_linear_beta=(method.moe.activation_situ_linear_beta),
            )
            calibrate_cubic_moe_execution(
                x,
                layer.w13_weight_packed,
                layer.w2_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                gate_a,
                gate_b,
                down_a,
                down_b,
                topk_weights,
                topk_ids,
                dynamic_a8=dynamic_a8,
                **execution_common,
            )
            phase_index += 1
            if progress is not None:
                progress.phase(
                    task,
                    phase_index,
                    phase_total,
                    f"{'A8' if dynamic_a8 else 'A16'} execution",
                )
            func = cubic_fused_moe_dynamic_a8 if dynamic_a8 else cubic_fused_moe
            output = func(
                x,
                layer.w13_weight_packed,
                layer.w2_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                gate_a,
                gate_b,
                down_a,
                down_b,
                topk_weights,
                topk_ids,
                **common,
            )
            phase_index += 1
            if progress is not None:
                progress.phase(
                    task,
                    phase_index,
                    phase_total,
                    f"{'A8' if dynamic_a8 else 'A16'} correctness",
                )
            if not torch.isfinite(output).all():
                raise AssertionError(
                    f"Non-finite Cubic MoE calibration output for W{bits}."
                )
            if progress is not None:
                progress(task)


@torch.inference_mode()
def cubic_kernel_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    capture_sizes: tuple[int, ...],
) -> None:
    """Calibrate the Cubic tactics used by ``model`` on the current device."""
    if not envs.VLLM_CUBIC_AUTOTUNE:
        return
    token_buckets = _calibration_token_buckets(max_tokens, capture_sizes)
    moe_token_buckets = _moe_calibration_token_buckets(max_tokens, token_buckets)
    materialization_buckets = _materialization_token_buckets(max_tokens)
    local_tasks = _cubic_calibration_tasks(
        model,
        token_buckets,
        moe_token_buckets=moe_token_buckets,
    )
    if not local_tasks:
        return
    rank, world_size, cpu_group = _cubic_world()
    fingerprint = _cubic_device_fingerprint()
    cache_key: str | None = None
    cache_hit = False
    try:
        cache_key = _cubic_tactic_cache_key(model, moe_token_buckets)
        cache_hit = _load_cubic_tactic_cache(cache_key)
    except Exception as error:  # noqa: BLE001 - cache failure must not block startup
        logger.warning("Cubic tactic cache is unavailable; recalibrating: %s", error)
    if not cache_hit:
        for registry in _cubic_tactic_registries().values():
            registry.clear()
    cache_domain = _cubic_cache_domain(cache_key, rank, world_size, cpu_group)
    metadata = _cubic_all_gather(
        {
            "rank": rank,
            "fingerprint": fingerprint,
            "cache_domain": cache_domain,
            "tasks": local_tasks,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
        },
        world_size,
        cpu_group,
    )
    assignments, global_pending = _assign_cubic_tasks(metadata)
    owned_tasks = assignments[rank]
    progress = _CalibrationProgress(rank, owned_tasks)
    logger.info(
        "Cubic calibration plan rank=%d: assigned=%d local=%d, "
        "global unique pending=%d, cache_hit=%s",
        rank,
        len(owned_tasks),
        len(local_tasks),
        global_pending,
        cache_hit,
    )
    calibration_complete = True
    for family, warmup in (
        ("Linear", _warmup_cubic_linear_families),
        ("MoE", _warmup_cubic_moe_families),
    ):
        try:
            family_buckets = token_buckets if family == "Linear" else moe_token_buckets
            warmup_kwargs = (
                {"graph_capture_sizes": capture_sizes} if family == "MoE" else {}
            )
            warmup(model, family_buckets, owned_tasks, progress, **warmup_kwargs)
        except (RuntimeError, AssertionError, ValueError) as error:
            calibration_complete = False
            logger.warning(
                "Cubic %s calibration failed; retaining safe fallback "
                "tactics for unfinished signatures: %s",
                family,
                error,
            )
            torch.accelerator.empty_cache()
    specs = _cubic_w2_a8_situ_specs(model)
    owned_specs = [spec for spec in specs if ("w2_situ", *spec) in owned_tasks]
    if owned_specs:
        from vllm.model_executor.layers.quantization.cubic_kernels import (
            calibrate_cubic_w2_a8_situ,
        )

        route_ctas = (16, 32, 64, 128)
        logger.info(
            "Calibrating assigned Cubic W2 A8 SITU tactics: %s",
            owned_specs,
        )
        for n, k, group_out, group_size, top_k, local_experts in owned_specs:
            task = (
                "w2_situ",
                n,
                k,
                group_out,
                group_size,
                top_k,
                local_experts,
            )
            if n % group_out or k % group_size:
                progress(task)
                continue
            try:
                calibrate_cubic_w2_a8_situ(
                    n=n,
                    k=k,
                    group_out=group_out,
                    group_size=group_size,
                    top_k=top_k,
                    local_experts=local_experts,
                    route_ctas_values=route_ctas,
                    progress=(
                        lambda done, total, phase, task=task: progress.phase(
                            task, done, total, phase
                        )
                    ),
                )
                progress(task)
            except (RuntimeError, AssertionError) as error:
                calibration_complete = False
                logger.warning(
                    "Cubic tactic calibration failed for N=%d K=%d G=%dx%d; "
                    "using the safe fallback: %s",
                    n,
                    k,
                    group_out,
                    group_size,
                    error,
                )
                torch.accelerator.empty_cache()
    results = _cubic_all_gather(
        {
            "rank": rank,
            "fingerprint": fingerprint,
            "cache_domain": cache_domain,
            "complete": calibration_complete,
            "registries": _export_cubic_tactics(),
        },
        world_size,
        cpu_group,
    )
    _merge_cubic_tactics(results, fingerprint, cache_domain)
    logger.info("Materializing merged Cubic tactics on rank=%d", rank)
    _warmup_cubic_linear_families(
        model,
        materialization_buckets,
        calibrate=False,
    )
    _warmup_cubic_moe_families(
        model,
        materialization_buckets,
        calibrate=False,
        graph_capture_sizes=capture_sizes,
    )
    torch.accelerator.synchronize()
    compatible_complete = all(
        item["complete"]
        for item in results
        if tuple(item["fingerprint"]) == fingerprint
        and tuple(item["cache_domain"]) == cache_domain
    )
    cache_writers = {
        (domain, key): min(
            int(item["rank"])
            for item in metadata
            if tuple(item["cache_domain"]) == domain and item["cache_key"] == key
        )
        for domain, key in {
            (tuple(item["cache_domain"]), item["cache_key"])
            for item in metadata
            if item["cache_key"] is not None
        }
    }
    if (
        compatible_complete
        and not cache_hit
        and cache_key is not None
        and cache_writers[(cache_domain, cache_key)] == rank
    ):
        try:
            _save_cubic_tactic_cache(cache_key)
        except Exception as error:  # noqa: BLE001 - cache failure is non-fatal
            logger.warning("Unable to save Cubic tactic cache: %s", error)
    _cubic_barrier(world_size, cpu_group)
    logger.info(
        "Cubic calibration complete rank=%d: %d/%d assigned tasks, "
        "elapsed=%.1fs, cache_key=%s",
        rank,
        progress.completed,
        progress.total,
        time.monotonic() - progress.started,
        cache_key,
    )
    # Release synthetic tensors before graph-pool sizing/capture.
    torch.accelerator.empty_cache()
