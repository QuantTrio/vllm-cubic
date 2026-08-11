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
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.cubic import (
    CubicLinearMethod,
    CubicMoEMethod,
    cubic_carrier_levels,
)

logger = init_logger(__name__)

_CUBIC_TACTIC_CACHE_SCHEMA = 15
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
    "_CUBIC_MOE_DENSE_BLOCK_TACTICS",
    "_CUBIC_MOE_ROUTE_CTA_TACTICS",
    "_CUBIC8_W2_BLOCK_N_TACTICS",
    "_CUBIC8_W2_LUT_TACTICS",
)

CalibrationTask = tuple[Any, ...]


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
            signatures.append(
                [
                    "linear",
                    method.scheme.num_bits,
                    method.scheme.group_size,
                    method.scheme.group_out,
                    int(module.input_size_per_partition),
                    int(module.output_size_per_partition),
                    list(module.weight_packed.shape),
                    str(module.weight_a.dtype),
                    str(module.weight_b.dtype),
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
        / "cubic_kernels.py",
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
        / "torch_bindings.cpp",
    )
    source_hash = hashlib.sha256()
    for path in source_paths:
        source_hash.update(path.read_bytes())
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
    return (
        "linear",
        method.scheme.num_bits,
        method.scheme.group_size,
        method.scheme.group_out,
        int(layer.input_size_per_partition),
        int(layer.output_size_per_partition),
        str(layer.weight_a.dtype),
        str(layer.weight_b.dtype),
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
    model: torch.nn.Module, token_buckets: tuple[int, ...]
) -> tuple[CalibrationTask, ...]:
    tasks: set[CalibrationTask] = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, CubicLinearMethod):
            tasks.update(_linear_task_id(module, method, m) for m in token_buckets)
        elif isinstance(method, CubicMoEMethod):
            tasks.update(_moe_task_id(module, method, m) for m in token_buckets)
    tasks.update(("w2_situ", *spec) for spec in _cubic_w2_a8_situ_specs(model))
    return tuple(sorted(tasks, key=repr))


def _cubic_task_weight(task: CalibrationTask) -> int:
    if task[0] == "linear":
        _, _, _, _, k, n, _, _, m = task
        return int(k) * int(n) * (32 + int(m))
    if task[0] == "w2_situ":
        _, n, k, _, _, local_experts = task
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
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return unique (N, K, group_size, top_k, local_experts) shapes."""
    specs: set[tuple[int, int, int, int, int]] = set()
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
            (int(n), int(k), method.scheme.group_size, int(top_k), packed.shape[0])
        )
    return tuple(sorted(specs))


def _calibration_token_buckets(
    max_tokens: int, capture_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    targets = (1, 8, 16, 32, 64, 128, 256, 512, 1024, max_tokens)
    available = tuple(
        sorted({value for value in capture_sizes if 0 < value <= max_tokens})
    )
    if not available:
        return tuple(sorted({min(max(value, 1), max_tokens) for value in targets}))
    selected = {
        min(available, key=lambda value: abs(value - target)) for target in targets[:-1]
    }
    selected.add(min(1024, max_tokens))
    selected.add(max_tokens)
    return tuple(sorted(selected))


@torch.inference_mode()
def _warmup_cubic_linear_families(
    model: torch.nn.Module,
    token_buckets: tuple[int, ...],
    owned_tasks: set[CalibrationTask] | None = None,
    progress: _CalibrationProgress | None = None,
    calibrate: bool = True,
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        calibrate_cubic_linear_execution,
        cubic_linear,
        cubic_linear_dynamic_a8,
        cubic_linear_dynamic_a8_precomputed,
    )

    layers: dict[CalibrationTask, tuple[torch.nn.Module, CubicLinearMethod]] = {}
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, CubicLinearMethod):
            continue
        key = _linear_task_id(module, method, 0)[:-1]
        layers.setdefault(key, (module, method))

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
        a16_a, a16_b = layer.weight_a, layer.weight_b
        a8_a, a8_b = layer.weight_a, layer.weight_b
        if bits == 3 and layer.weight_a.dtype == torch.int8:
            a16_a = torch.full_like(layer.weight_a, 0.5, dtype=torch.float16)
            a16_b = torch.full_like(layer.weight_b, 0.25, dtype=torch.float16)
        elif bits == 3:
            levels = cubic_carrier_levels(3, layer.weight_a, layer.weight_b)
            a8_a, a8_b = levels[..., 1].contiguous(), levels[..., 2].contiguous()
        if method.dynamic_a8:
            carrier = getattr(layer, "weight_carrier", None)
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
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_a8_moe_grouping,
        calibrate_cubic_a8_moe_backend,
        calibrate_cubic_a8_moe_grouping,
        calibrate_cubic_a8_moe_layer_backends,
        calibrate_cubic_moe_execution,
        calibrate_cubic_moe_route_ctas,
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
                    group_size=group_size,
                    group_out=group_out,
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
    local_tasks = _cubic_calibration_tasks(model, token_buckets)
    if not local_tasks:
        return
    rank, world_size, cpu_group = _cubic_world()
    fingerprint = _cubic_device_fingerprint()
    cache_key: str | None = None
    cache_hit = False
    try:
        cache_key = _cubic_tactic_cache_key(model, token_buckets)
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
            warmup(model, token_buckets, owned_tasks, progress)
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
        for n, k, group_size, top_k, local_experts in owned_specs:
            task = ("w2_situ", n, k, group_size, top_k, local_experts)
            if k % group_size:
                progress(task)
                continue
            try:
                calibrate_cubic_w2_a8_situ(
                    n=n,
                    k=k,
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
                    "Cubic tactic calibration failed for N=%d K=%d G=%d; "
                    "using the safe fallback: %s",
                    n,
                    k,
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
        token_buckets,
        calibrate=False,
    )
    _warmup_cubic_moe_families(
        model,
        token_buckets,
        calibrate=False,
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
