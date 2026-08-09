# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quantize Kimi-K3 into data-free MSE-fitted Cubic 2.5-bit mixed precision.

[Minimum Requirements]
 - One or more CUDA GPUs; 32 GiB per worker is the recommended minimum.

[Python Dependencies]
 - Install a CUDA-enabled PyTorch build compatible with the local driver.
 - venv/bin/python3 -m pip install torch safetensors

[Processing Time]
 - Approximately 30 minutes on 8xH200; other devices scale with throughput.

[Example Command]
python -u examples/quantization/quantize_k3.py \
  --source /path/to/Kimi-K3 \
  --output /path/to/Kimi-K3-Cubic-2.5Bit \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

Group-local Cubic parameters are selected by a least-squares objective.  The
final report presents scale-free NRMSE so loss values are comparable across
bit widths; the square root is reporting-only and does not change candidate
selection.

Dynamic-A8 carrier correction (round(127*q)/127) is enabled by default.  Pass
--disable-a8-correction to fit only the continuous Cubic reconstruction.

"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import statistics
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CUBIC_FORMAT = "cubic-pack-quantized"
DEFAULT_SOURCE = Path("__YOUR_PATH__/moonshotai/Kimi-K3")
DEFAULT_OUTPUT = Path("__YOUR_PATH__/Kimi-K3-Cubic-2.5Bit")

# START-END:BITS@GROUP_SIZE. Kimi-K3 has dense layer 0 and MoE layers 1--92.
# This schedule has a 2.498641304-bit effective width: layers 1--3 use W3
# G256, layers 4--32 use W3 G512, layers 33--91 use W2 G512, and layer 92
# uses W4 G512.
MOE_SCHEDULE = "1-3:3@256,4-32:3@512,33-91:2@512,92:4@512"

# Residual-attention scoring projections are copied in source dtype because a
# 7168-to-1 scorer has negligible storage/compute cost and directly affects the
# softmax that selects residual streams.
LINEAR_SCHEDULE = ""

DEFAULT_DEVICES = "cuda:0"
DEFAULT_SHARD_SIZE_GIB = 3.0
DEFAULT_ROW_CHUNK_SIZE = -1
DEFAULT_TENSOR_BATCH_SIZE = 32
A8_CARRIER_AWARE = True
QUANT_LOSS_STATS: dict[str, dict[str, Any]] = {}

NUM_MOE_LAYERS = 92
NUM_EXPERTS = 896
HIDDEN_SIZE = 7168
EXPERT_INPUT_SIZE = 3584
MOE_INTERMEDIATE_SIZE = 3072
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class MoERule:
    start_layer: int
    end_layer: int
    bits: int
    group_size: int


@dataclass(frozen=True)
class LinearRule:
    layer: int
    bits: int
    group_size: int

    @property
    def prefix(self) -> str:
        return f"language_model.model.layers.{self.layer}.mlp_res_proj"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scheme(bits: int, group_size: int) -> None:
    if bits not in range(1, 9):
        raise ValueError(f"Cubic bits must be in 1--8, got {bits}.")
    if group_size <= 0 or group_size * bits % 8:
        raise ValueError(f"{bits}-bit group size {group_size} is not byte aligned.")


def _parse_moe_schedule(value: str) -> list[MoERule]:
    rules = []
    covered = set()
    for item in value.split(","):
        match = re.fullmatch(
            r"(\d+)(?:-(\d+))?:(\d+)@(\d+)",
            item.strip(),
        )
        if match is None:
            raise ValueError("MoE schedule entries must use START-END:BITS@GROUP_SIZE.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        bits = int(match.group(3))
        group_size = int(match.group(4))
        _validate_scheme(bits, group_size)
        if start > end:
            raise ValueError(f"Invalid MoE layer range: {item!r}.")
        layers = set(range(start, end + 1))
        if layers & covered:
            raise ValueError(f"Overlapping MoE schedule entry: {item!r}.")
        covered |= layers
        rules.append(MoERule(start, end, bits, group_size))
    expected = set(range(1, NUM_MOE_LAYERS + 1))
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise ValueError(
            f"MoE schedule must cover layers 1--92; missing={missing}, extra={extra}."
        )
    return rules


def _parse_linear_schedule(value: str) -> list[LinearRule]:
    if not value.strip():
        return []
    rules = []
    covered = set()
    for item in value.split(","):
        match = re.fullmatch(r"(\d+):(\d+)@(\d+)", item.strip())
        if match is None:
            raise ValueError("Linear schedule entries must use LAYER:BITS@GROUP_SIZE.")
        layer = int(match.group(1))
        bits = int(match.group(2))
        group_size = int(match.group(3))
        _validate_scheme(bits, group_size)
        if layer in covered:
            raise ValueError(f"Duplicate Linear layer: {layer}.")
        covered.add(layer)
        rules.append(LinearRule(layer, bits, group_size))
    return rules


def _moe_rule_for_key(key: str, rules: list[MoERule]) -> MoERule:
    match = re.search(r"\.layers\.(\d+)\.", key)
    if match is None:
        raise ValueError(f"Cannot determine layer for expert tensor {key}.")
    layer = int(match.group(1))
    for rule in rules:
        if rule.start_layer <= layer <= rule.end_layer:
            return rule
    raise ValueError(f"No MoE quantization rule for layer {layer}.")


def _cubic_levels(
    bits: int,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    *,
    device: torch.device,
) -> torch.Tensor:
    a_tensor = torch.as_tensor(a, device=device, dtype=torch.float32)
    if bits == 1:
        return torch.ones(
            (*a_tensor.shape, 1),
            device=device,
            dtype=torch.float32,
        )
    magnitude_max = (1 << (bits - 1)) - 1
    b_tensor = torch.as_tensor(b, device=device, dtype=torch.float32)
    t = (
        torch.arange(
            magnitude_max + 1,
            device=device,
            dtype=torch.float32,
        )
        / magnitude_max
    )
    c = 1 - a_tensor - b_tensor
    return t * (a_tensor[..., None] + t * (b_tensor[..., None] + t * c[..., None]))


def _carrier_levels(levels: torch.Tensor) -> torch.Tensor:
    return torch.round(127 * levels) / 127


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 1:
        if torch.any((codes != -1) & (codes != 1)):
            raise ValueError("1-bit Cubic codes must be -1 or +1.")
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        if torch.any((codes < -magnitude_max) | (codes > magnitude_max)):
            raise ValueError("Cubic codes contain a reserved value.")

    shape = codes.shape
    num_values = shape[-1]
    flat = codes.reshape(-1, num_values).to(torch.int64)
    raw = (flat > 0).to(torch.int64) if bits == 1 else flat & ((1 << bits) - 1)
    num_bytes = math.ceil(num_values * bits / 8)
    packed = torch.zeros(
        flat.shape[0],
        num_bytes,
        dtype=torch.int64,
        device=codes.device,
    )
    base = torch.arange(num_values, device=codes.device) * bits
    for bit in range(bits):
        positions = base + bit
        byte_indices = positions // 8
        shifts = positions % 8
        values = ((raw >> bit) & 1) << shifts
        packed.scatter_add_(
            1,
            byte_indices.expand(flat.shape[0], -1),
            values,
        )
    return packed.to(torch.uint8).reshape(*shape[:-1], num_bytes)


def _decode_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    packed = packed.to(device=device, non_blocking=True)
    scale = scale.to(device=device, non_blocking=True)
    lookup = torch.tensor(
        _E2M1_LEVELS + tuple(-level for level in _E2M1_LEVELS),
        device=device,
        dtype=torch.float32,
    )
    low = lookup[(packed & 0x0F).to(torch.long)]
    high = lookup[(packed >> 4).to(torch.long)]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    scale_f32 = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32),
        scale.to(torch.int32) - 127,
    )
    return values * scale_f32.repeat_interleave(32, dim=-1)


def _reset_quant_loss_stats() -> None:
    QUANT_LOSS_STATS.clear()


def _record_quant_loss_stats(
    bits: int,
    group_size: int,
    groups: torch.Tensor,
    scale: torch.Tensor,
    q: torch.Tensor,
) -> None:
    """Accumulate final-format errors without synchronizing the GPU."""

    key = f"{bits}@{group_size}"
    bucket = QUANT_LOSS_STATS.get(key)
    if bucket is None:
        zero = torch.zeros((), device=groups.device, dtype=torch.float32)
        bucket = {
            "bits": bits,
            "group_size": group_size,
            "values": 0,
            "signal_sse": zero,
            "continuous_sse": zero.clone(),
            "carrier_sse": zero.clone(),
            "clipped_values": torch.zeros((), device=groups.device, dtype=torch.int64),
        }
        QUANT_LOSS_STATS[key] = bucket
    group_count = int(groups.shape[0])
    value_count = int(groups.numel())
    bucket["values"] += value_count
    # Bound diagnostic temporaries independently of the conversion row chunk.
    # Eight million FP32 values keep each temporary around 32 MiB and avoid a
    # second full expert-batch materialization on small GPUs.
    stats_group_chunk = max(1, (8 * 1024 * 1024) // group_size)
    for start in range(0, group_count, stats_group_chunk):
        stop = min(group_count, start + stats_group_chunk)
        group_chunk = groups[start:stop]
        scale_chunk = scale[start:stop]
        q_chunk = q[start:stop]
        reconstructed = group_chunk.sign() * scale_chunk[:, None] * q_chunk
        continuous_sse = (group_chunk - reconstructed).square().sum()
        if bits <= 2:
            # Symmetric W1/W2 levels are already exact on the A8 carrier grid.
            carrier_sse = continuous_sse
        else:
            carrier_q = _carrier_levels(q_chunk)
            carrier_reconstructed = (
                group_chunk.sign() * scale_chunk[:, None] * carrier_q
            )
            carrier_sse = (group_chunk - carrier_reconstructed).square().sum()
        bucket["signal_sse"] = bucket["signal_sse"] + group_chunk.square().sum()
        bucket["continuous_sse"] = bucket["continuous_sse"] + continuous_sse
        bucket["carrier_sse"] = bucket["carrier_sse"] + carrier_sse
        bucket["clipped_values"] = (
            bucket["clipped_values"] + (group_chunk.abs() > scale_chunk[:, None]).sum()
        )


def _quant_loss_stats_snapshot() -> dict[str, dict[str, int | float]]:
    """Synchronize once per completed source shard and return plain scalars."""

    result: dict[str, dict[str, int | float]] = {}
    for key, bucket in QUANT_LOSS_STATS.items():
        result[key] = {
            name: (
                int(value)
                if name in ("bits", "group_size", "values")
                else float(value.item())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            for name, value in bucket.items()
        }
    return result


def _merge_quant_loss_stats(
    destination: dict[str, dict[str, int | float]],
    source: dict[str, dict[str, int | float]],
) -> None:
    for key, incoming in source.items():
        bucket = destination.get(key)
        if bucket is None:
            destination[key] = dict(incoming)
            continue
        for name, value in incoming.items():
            if name in ("bits", "group_size"):
                if bucket[name] != value:
                    raise ValueError(f"Inconsistent loss statistic {key} {name}.")
            else:
                bucket[name] += value


def _summarize_quant_loss_bucket(
    bucket: dict[str, int | float],
    a8_carrier_aware: bool,
) -> dict[str, int | float]:
    values = int(bucket["values"])
    signal_sse = max(float(bucket["signal_sse"]), 1.0e-30)
    continuous_sse = float(bucket["continuous_sse"])
    carrier_sse = float(bucket["carrier_sse"])
    objective_sse = (
        0.5 * (continuous_sse + carrier_sse) if a8_carrier_aware else continuous_sse
    )
    result: dict[str, int | float] = {
        "bits": int(bucket["bits"]),
        "loss": math.sqrt(objective_sse / signal_sse),
        "clipped_percent": 100.0 * int(bucket["clipped_values"]) / max(values, 1),
    }
    if a8_carrier_aware:
        result["a8_correction_loss"] = math.sqrt(carrier_sse / signal_sse)
    if int(bucket["group_size"]) > 0:
        result["group_size"] = int(bucket["group_size"])
    return result


def _quant_loss_report(
    raw_by_scheme: dict[str, dict[str, int | float]],
    a8_carrier_aware: bool,
) -> dict[str, Any]:
    by_bit_raw: dict[str, dict[str, int | float]] = {}
    for bucket in raw_by_scheme.values():
        bit_key = str(int(bucket["bits"]))
        aggregate = by_bit_raw.get(bit_key)
        if aggregate is None:
            aggregate = dict(bucket)
            aggregate["group_size"] = -1
            by_bit_raw[bit_key] = aggregate
        else:
            for name, value in bucket.items():
                if name in ("bits", "group_size"):
                    continue
                aggregate[name] += value
    return {
        "objective": (
            "mean(continuous MSE, rounded-A8-carrier MSE)"
            if a8_carrier_aware
            else "continuous MSE"
        ),
        "normalization": (
            "loss is sqrt(joint SSE / source weight SSE), i.e. NRMSE; "
            "the fitting objective itself remains least squares."
        ),
        "loss_metadata_precision": "FP32 scale and FP16 a/b",
        "by_bit": {
            key: _summarize_quant_loss_bucket(value, a8_carrier_aware)
            for key, value in sorted(by_bit_raw.items(), key=lambda item: int(item[0]))
        },
        "by_bit_and_group_size": {
            key: _summarize_quant_loss_bucket(value, a8_carrier_aware)
            for key, value in sorted(
                raw_by_scheme.items(),
                key=lambda item: (
                    int(item[1]["bits"]),
                    int(item[1]["group_size"]),
                ),
            )
        },
    }


def _fit_scale(
    values: torch.Tensor,
    levels: torch.Tensor,
    start_scale: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = values.abs()
    carrier_levels = _carrier_levels(levels) if A8_CARRIER_AWARE else None
    scale = start_scale.clamp_min(torch.finfo(torch.float32).tiny)
    for _ in range(iterations):
        candidates = scale[:, None, None] * levels[None, None, :]
        distances = (absolute[..., None] - candidates).abs()
        if carrier_levels is not None:
            carrier_candidates = scale[:, None, None] * carrier_levels[None, None, :]
            distances = distances.square()
            distances += (absolute[..., None] - carrier_candidates).square()
        indices = distances.argmin(dim=-1)
        q = levels[indices]
        if carrier_levels is None:
            numerator = (absolute * q).sum(dim=-1)
            denominator = q.square().sum(dim=-1)
        else:
            carrier_q = carrier_levels[indices]
            numerator = (absolute * (q + carrier_q)).sum(dim=-1)
            denominator = (q.square() + carrier_q.square()).sum(dim=-1)
        updated = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.where(denominator > 0, updated, scale)
    q = levels[indices]
    reconstructed = values.sign() * scale[:, None] * q
    loss = (values - reconstructed).square().sum(dim=-1)
    if carrier_levels is not None:
        carrier_q = carrier_levels[indices]
        carrier_reconstructed = values.sign() * scale[:, None] * carrier_q
        loss += (values - carrier_reconstructed).square().sum(dim=-1)
    return scale, loss


def _quantize_symmetric(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = weight.reshape(-1, group_size)
    absolute = groups.abs()
    if bits == 1:
        scale = absolute.mean(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        codes = torch.where(groups < 0, -1, 1).to(torch.int8)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        scale = absolute.amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        for _ in range(8):
            magnitudes = torch.round(absolute / scale[:, None] * magnitude_max).clamp_(
                0, magnitude_max
            )
            levels = magnitudes / magnitude_max
            denominator = levels.square().sum(dim=-1)
            updated = (absolute * levels).sum(dim=-1) / denominator.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            scale = torch.where(denominator > 0, updated, scale)
        codes = (groups.sign() * magnitudes).to(torch.int8)
    final_q = (
        torch.ones_like(groups)
        if bits == 1
        else magnitudes.to(torch.float32) / magnitude_max
    )
    _record_quant_loss_stats(bits, group_size, groups, scale, final_q)
    codes = codes.reshape(weight.shape)
    metadata_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // group_size,
    )
    scale = scale.reshape(metadata_shape).to(torch.float32)
    a = torch.ones(
        metadata_shape,
        device=weight.device,
        dtype=torch.float16,
    )
    b = torch.zeros_like(a)
    return _pack_codes(codes, bits), scale, a, b


def _quantize_curved(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = weight.to(torch.float32).reshape(-1, group_size)
    calibration_stride = {
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 8,
        8: 8,
    }[bits]
    calibration = groups[:, ::calibration_stride]
    group_amax = groups.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    best_loss = torch.full_like(group_amax, torch.inf)
    best_scale = group_amax.clone()
    best_a = torch.ones_like(group_amax)
    best_b = torch.zeros_like(group_amax)
    candidate_pairs = {
        3: (
            (1.0, 0.0),
            (0.75, -0.25),
            (1.0, -0.75),
            (0.25, 0.25),
            (0.5, 0.25),
        ),
        4: (
            (0.75, -0.75),
            (0.5, -0.25),
            (1.0, -0.25),
            (0.5, 0.0),
            (1.0, 0.0),
        ),
        5: (
            (1.0, -0.25),
            (1.0, 0.25),
            (1.0, 0.0),
            (0.75, -0.25),
            (0.25, 0.25),
            (0.5, -0.25),
            (0.75, -0.75),
            (0.25, 0.0),
        ),
        6: (
            (0.5, -0.75),
            (1.25, 0.25),
            (1.0, 0.0),
            (0.75, -0.25),
            (1.25, 0.0),
        ),
        7: (
            (0.75, 0.75),
            (0.5, -0.75),
            (1.25, 0.25),
            (1.0, 0.0),
            (0.5, -0.25),
        ),
        8: (
            (0.5, 0.0),
            (0.25, 0.25),
            (0.5, -0.75),
            (0.75, 0.25),
            (1.0, 0.0),
        ),
    }[bits]
    multipliers = {
        3: (0.65, 1.0),
        4: (0.65, 1.0),
        5: (0.65, 0.8, 1.0, 1.15),
        6: (0.65, 0.8, 1.0, 1.15),
        7: (1.0,),
        8: (0.65, 0.8, 1.0, 1.15),
    }[bits]
    iterations = 8 if bits == 3 else 2
    for a_value, b_value in candidate_pairs:
        levels = _cubic_levels(
            bits,
            a_value,
            b_value,
            device=weight.device,
        )
        for multiplier in multipliers:
            scale, loss = _fit_scale(
                calibration,
                levels,
                group_amax * multiplier,
                iterations,
            )
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_scale = torch.where(improved, scale, best_scale)
            best_a = torch.where(improved, a_value, best_a)
            best_b = torch.where(improved, b_value, best_b)

    stored_scale = best_scale.to(torch.float32)
    stored_a = best_a.to(torch.float16)
    stored_b = best_b.to(torch.float16)
    levels = _cubic_levels(
        bits,
        stored_a,
        stored_b,
        device=weight.device,
    )
    distances = (
        groups.abs()[..., None] - stored_scale[:, None, None] * levels[:, None, :]
    ).abs()
    if A8_CARRIER_AWARE:
        carrier_levels = _carrier_levels(levels)
        distances = distances.square()
        distances += (
            groups.abs()[..., None]
            - stored_scale[:, None, None] * carrier_levels[:, None, :]
        ).square()
    magnitudes = distances.argmin(dim=-1)
    final_q = torch.gather(levels, 1, magnitudes)
    _record_quant_loss_stats(
        bits,
        group_size,
        groups,
        stored_scale,
        final_q,
    )
    codes = (groups.sign().to(torch.int64) * magnitudes).reshape(weight.shape)
    metadata_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // group_size,
    )
    return (
        _pack_codes(codes, bits),
        stored_scale.reshape(metadata_shape),
        stored_a.reshape(metadata_shape),
        stored_b.reshape(metadata_shape),
    )


def _quantize_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    row_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"Weight K={weight.shape[-1]} is not divisible by group size {group_size}."
        )
    if bits <= 2:
        return _quantize_symmetric(weight, bits, group_size)
    if bits == 3:
        return _quantize_curved(weight, bits, group_size)

    if row_chunk_size == -1:
        row_chunk_size = _automatic_row_chunk_size(weight, bits, group_size)

    packed_chunks = []
    scale_chunks = []
    a_chunks = []
    b_chunks = []
    rows = weight.reshape(-1, weight.shape[-1])
    for chunk in rows.split(row_chunk_size):
        packed, scale, a, b = _quantize_curved(
            chunk,
            bits,
            group_size,
        )
        packed_chunks.append(packed)
        scale_chunks.append(scale)
        a_chunks.append(a)
        b_chunks.append(b)
    prefix = weight.shape[:-1]
    return (
        torch.cat(packed_chunks).reshape(*prefix, -1),
        torch.cat(scale_chunks).reshape(*prefix, -1),
        torch.cat(a_chunks).reshape(*prefix, -1),
        torch.cat(b_chunks).reshape(*prefix, -1),
    )


def _automatic_row_chunk_size(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> int:
    rows = weight.numel() // weight.shape[-1]
    if weight.device.type != "cuda":
        return min(rows, 32)
    free_bytes, _ = torch.accelerator.get_memory_info(weight.device)
    level_count = 1 << (bits - 1)
    groups_per_row = math.ceil(weight.shape[-1] / group_size)
    distance_bytes_per_row = groups_per_row * group_size * level_count * 4
    temporary_factor = 7 if A8_CARRIER_AWARE else 4
    budget = min(int(free_bytes * 0.1), 12 * 1024**3)
    estimated = max(1, budget // (distance_bytes_per_row * temporary_factor))
    estimated = min(rows, estimated, 2048)
    return estimated // 32 * 32 if estimated >= 32 else estimated


def _is_strictly_monotonic(a: float, b: float) -> bool:
    c = 1 - a - b
    points = [0.0, 1.0]
    if c:
        vertex = -b / (3 * c)
        if 0 < vertex < 1:
            points.append(vertex)
    return min(a + 2 * b * point + 3 * c * point * point for point in points) > 0


def _linear_candidate_pairs() -> list[tuple[float, float]]:
    pairs = [(1.0, 0.0)]
    for a in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        for b in (-0.75, -0.25, 0.0, 0.25, 0.75):
            if (a, b) not in pairs and _is_strictly_monotonic(a, b):
                pairs.append((a, b))
    return pairs


def _quantize_linear_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"Linear K={weight.shape[-1]} is not divisible by group size {group_size}."
        )
    groups = weight.to(torch.float32).reshape(-1, group_size)
    group_amax = groups.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    if bits == 1:
        scale = groups.abs().mean(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        a = torch.ones_like(scale, dtype=torch.float16)
        b = torch.zeros_like(a)
        codes = torch.where(groups < 0, -1, 1).to(torch.int64)
    else:
        best_loss = torch.full_like(group_amax, torch.inf)
        scale = group_amax.clone()
        best_a = torch.ones_like(group_amax)
        best_b = torch.zeros_like(group_amax)
        for a_value, b_value in _linear_candidate_pairs():
            levels = _cubic_levels(
                bits,
                a_value,
                b_value,
                device=weight.device,
            )
            for multiplier in (0.65, 0.8, 1.0, 1.15):
                candidate_scale, loss = _fit_scale(
                    groups,
                    levels,
                    group_amax * multiplier,
                    8,
                )
                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                scale = torch.where(improved, candidate_scale, scale)
                best_a = torch.where(improved, a_value, best_a)
                best_b = torch.where(improved, b_value, best_b)
        scale = scale.to(torch.float32)
        a = best_a.to(torch.float16)
        b = best_b.to(torch.float16)
        levels = _cubic_levels(bits, a, b, device=weight.device)
        distances = (
            groups.abs()[..., None] - scale[:, None, None] * levels[:, None, :]
        ).abs()
        if A8_CARRIER_AWARE:
            carrier_levels = _carrier_levels(levels)
            distances = distances.square()
            distances += (
                groups.abs()[..., None]
                - scale[:, None, None] * carrier_levels[:, None, :]
            ).square()
        magnitudes = distances.argmin(dim=-1)
        final_q = torch.gather(levels, 1, magnitudes)
        _record_quant_loss_stats(
            bits,
            group_size,
            groups,
            scale,
            final_q,
        )
        codes = groups.sign().to(torch.int64) * magnitudes
    if bits == 1:
        _record_quant_loss_stats(
            bits,
            group_size,
            groups,
            scale,
            torch.ones_like(groups),
        )
    codes = codes.reshape(weight.shape)
    metadata_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // group_size,
    )
    return (
        _pack_codes(codes, bits),
        scale.reshape(metadata_shape).to(torch.float32),
        a.reshape(metadata_shape),
        b.reshape(metadata_shape),
    )


def _quantize_linear(
    weight: torch.Tensor,
    rule: LinearRule,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    weight = weight.to(device=device, dtype=torch.float32)
    packed, scale, a, b = _quantize_linear_weight(
        weight,
        rule.bits,
        rule.group_size,
    )
    return {
        f"{rule.prefix}.weight_packed": packed.cpu(),
        f"{rule.prefix}.weight_scale": scale.cpu(),
        f"{rule.prefix}.weight_a": a.cpu(),
        f"{rule.prefix}.weight_b": b.cpu(),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _timed_print(
    message: str,
    started_at: float,
    eta_seconds: float | None,
) -> None:
    elapsed = _format_duration(time.monotonic() - started_at)
    eta = _format_duration(eta_seconds) if eta_seconds is not None else "calculating"
    print(f"[elapsed {elapsed} | ETA {eta}] {message}", flush=True)


def _estimate_eta(
    started_at: float,
    completed: int,
    total: int,
) -> float | None:
    if completed <= 0:
        return None
    elapsed = time.monotonic() - started_at
    return elapsed * (total - completed) / completed


def _estimate_parallel_eta(
    completed: int,
    total: int,
    worker_count: int,
    task_durations: list[float],
    active_started_at: dict[int, float],
) -> float | None:
    if completed < min(worker_count, total):
        return None
    typical = statistics.median(task_durations[-2 * worker_count :])
    now = time.monotonic()
    if any(
        now - task_started_at > 2 * typical
        for task_started_at in active_started_at.values()
    ):
        return None
    active_progress = [
        min(now - task_started_at, typical)
        for task_started_at in active_started_at.values()
    ]
    remaining_work = max(
        0.0,
        (total - completed) * typical - sum(active_progress),
    )
    active_tail = max(
        (typical - progress for progress in active_progress),
        default=0.0,
    )
    return max(remaining_work / worker_count, active_tail)


class MaterializedShardWriter:
    def __init__(
        self,
        directory: Path,
        task_index: int,
        max_bytes: int,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.directory = directory
        self.task_index = task_index
        self.max_bytes = max_bytes
        self.progress_callback = progress_callback
        header_reserve = min(64 * 1024**2, max_bytes // 16)
        self.payload_limit = max_bytes - header_reserve
        self.part = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self.payload_bytes = 0
        self.names: list[str] = []

    def add(self, tensors: dict[str, torch.Tensor]) -> None:
        size = sum(_tensor_bytes(tensor) for tensor in tensors.values())
        if size > self.payload_limit:
            names = ", ".join(tensors)
            raise ValueError(
                f"Tensor group {names} needs {size} bytes, above the "
                f"configured shard payload limit {self.payload_limit}."
            )
        if self.tensors and self.payload_bytes + size > self.payload_limit:
            self.flush()
        duplicate = self.tensors.keys() & tensors.keys()
        if duplicate:
            raise KeyError(f"Duplicate output tensors: {sorted(duplicate)}.")
        self.tensors.update(tensors)
        self.payload_bytes += size

    def flush(self) -> None:
        if not self.tensors:
            return
        self.part += 1
        name = f"part-task-{self.task_index:06d}-{self.part:06d}.safetensors"
        destination = self.directory / name
        temporary = destination.with_suffix(".safetensors.tmp")
        save_file(self.tensors, temporary)
        os.replace(temporary, destination)
        actual_size = destination.stat().st_size
        if actual_size > self.max_bytes:
            raise ValueError(
                f"{name} is {actual_size} bytes, above max {self.max_bytes}."
            )
        self.names.append(name)
        if self.progress_callback is not None:
            self.progress_callback(f"wrote {name} ({actual_size / 1024**3:.3f} GiB)")
        self.tensors = {}
        self.payload_bytes = 0

    def close(self) -> list[str]:
        self.flush()
        return self.names


def _convert_source_shard(
    source_shard: Path,
    writer: MaterializedShardWriter,
    device: torch.device,
    moe_rules: list[MoERule],
    linear_rules: dict[str, LinearRule],
    row_chunk_size: int,
    tensor_batch_size: int,
) -> None:
    with safe_open(source_shard, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        key_set = set(keys)
        packed_keys = [
            key
            for key in keys
            if key.endswith(".weight_packed")
            and key.removesuffix(".weight_packed") + ".weight_scale" in key_set
        ]
        source_scale_keys = {
            key.removesuffix(".weight_packed") + ".weight_scale" for key in packed_keys
        }

        for key in keys:
            if key in packed_keys or key in source_scale_keys:
                continue
            linear_rule = linear_rules.get(key)
            if linear_rule is None:
                writer.add({key: reader.get_tensor(key)})
            else:
                writer.add(
                    _quantize_linear(
                        reader.get_tensor(key),
                        linear_rule,
                        device,
                    )
                )

        shape_groups: dict[
            tuple[tuple[int, ...], tuple[int, ...], int, int],
            list[str],
        ] = defaultdict(list)
        for key in packed_keys:
            base = key.removesuffix(".weight_packed")
            rule = _moe_rule_for_key(key, moe_rules)
            shape_groups[
                (
                    tuple(reader.get_slice(key).get_shape()),
                    tuple(reader.get_slice(base + ".weight_scale").get_shape()),
                    rule.bits,
                    rule.group_size,
                )
            ].append(key)

        for (_, _, bits, group_size), grouped_keys in shape_groups.items():
            for start in range(0, len(grouped_keys), tensor_batch_size):
                batch_keys = grouped_keys[start : start + tensor_batch_size]
                source_packed = torch.stack(
                    [reader.get_tensor(key) for key in batch_keys]
                )
                source_scale = torch.stack(
                    [
                        reader.get_tensor(
                            key.removesuffix(".weight_packed") + ".weight_scale"
                        )
                        for key in batch_keys
                    ]
                )
                weights = _decode_mxfp4(
                    source_packed,
                    source_scale,
                    device,
                )
                packed, scale, a, b = _quantize_weight(
                    weights,
                    bits,
                    group_size,
                    row_chunk_size,
                )
                packed = packed.cpu()
                scale = scale.cpu()
                a = a.cpu()
                b = b.cpu()
                for index, key in enumerate(batch_keys):
                    base = key.removesuffix(".weight_packed")
                    writer.add(
                        {
                            key: packed[index].clone(),
                            base + ".weight_scale": scale[index].clone(),
                            base + ".weight_a": a[index].clone(),
                            base + ".weight_b": b[index].clone(),
                        }
                    )
                del (
                    source_packed,
                    source_scale,
                    weights,
                    packed,
                    scale,
                    a,
                    b,
                )


def _worker(
    worker_index: int,
    device_name: str,
    task_queue: Any,
    progress_queue: Any,
    source: str,
    staging: str,
    moe_rules: list[MoERule],
    linear_rules: list[LinearRule],
    max_shard_bytes: int,
    row_chunk_size: int,
    tensor_batch_size: int,
    a8_carrier_aware: bool,
) -> None:
    global A8_CARRIER_AWARE
    A8_CARRIER_AWARE = a8_carrier_aware
    task_index = -1
    shard_name = ""
    try:
        torch.set_num_threads(1)
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.accelerator.set_device_index(device.index or 0)
        source_path = Path(source)
        staging_path = Path(staging)
        linear_by_weight = {f"{rule.prefix}.weight": rule for rule in linear_rules}
        while True:
            task = task_queue.get()
            if task is None:
                return
            task_index, shard_name = task
            task_started_at = time.monotonic()
            _reset_quant_loss_stats()
            progress_queue.put(
                ("started", worker_index, device_name, task_index, shard_name)
            )

            def report(
                message: str,
                current_task_index: int = task_index,
                current_shard_name: str = shard_name,
            ) -> None:
                progress_queue.put(
                    (
                        "log",
                        worker_index,
                        device_name,
                        current_task_index,
                        current_shard_name,
                        message,
                    )
                )

            writer = MaterializedShardWriter(
                staging_path,
                task_index,
                max_shard_bytes,
                report,
            )
            _convert_source_shard(
                source_path / shard_name,
                writer,
                device,
                moe_rules,
                linear_by_weight,
                row_chunk_size,
                tensor_batch_size,
            )
            if device.type == "cuda":
                torch.accelerator.empty_cache()
            names = writer.close()
            progress_queue.put(
                (
                    "completed",
                    worker_index,
                    device_name,
                    task_index,
                    shard_name,
                    time.monotonic() - task_started_at,
                    len(names),
                    _quant_loss_stats_snapshot(),
                )
            )
    except BaseException:
        progress_queue.put(
            (
                "error",
                worker_index,
                device_name,
                task_index,
                shard_name,
                traceback.format_exc(),
            )
        )


def _run_workers(
    devices: list[str],
    source_shards: list[str],
    source: Path,
    staging: Path,
    moe_rules: list[MoERule],
    linear_rules: list[LinearRule],
    max_shard_bytes: int,
    row_chunk_size: int,
    tensor_batch_size: int,
    a8_carrier_aware: bool,
    started_at: float,
) -> dict[str, dict[str, int | float]]:
    # Workers pull the next source shard only after finishing their current
    # shard. Output part names use the source-shard index, so scheduling does
    # not change tensor-to-part assignment.
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    progress_queue = context.Queue()
    for task in enumerate(source_shards):
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)

    arguments = [
        (
            index,
            device,
            task_queue,
            progress_queue,
            str(source),
            str(staging),
            moe_rules,
            linear_rules,
            max_shard_bytes,
            row_chunk_size,
            tensor_batch_size,
            a8_carrier_aware,
        )
        for index, device in enumerate(devices)
    ]
    processes = [
        context.Process(target=_worker, args=worker_args) for worker_args in arguments
    ]
    for process in processes:
        process.start()

    total = len(source_shards)
    completed = 0
    task_durations: list[float] = []
    active_started_at: dict[int, float] = {}
    pending = set(range(len(processes)))
    empty_after_exit = 0
    quant_loss_stats: dict[str, dict[str, int | float]] = {}
    try:
        while pending or completed < total:
            try:
                event = progress_queue.get(timeout=1)
            except queue.Empty:
                event = None
            if event is None and not pending:
                empty_after_exit += 1
                if empty_after_exit >= 5:
                    raise RuntimeError(
                        f"Workers completed only {completed}/{total} source shards."
                    )
            else:
                empty_after_exit = 0
            if event is not None:
                kind = event[0]
                if kind == "started":
                    _, worker, device, task_index, shard_name = event
                    active_started_at[worker] = time.monotonic()
                    eta = _estimate_parallel_eta(
                        completed,
                        total,
                        len(devices),
                        task_durations,
                        active_started_at,
                    )
                    _timed_print(
                        f"[worker {worker} {device}] started "
                        f"task {task_index + 1}/{total}: {shard_name}",
                        started_at,
                        eta,
                    )
                elif kind == "log":
                    _, worker, device, task_index, shard_name, message = event
                    eta = _estimate_parallel_eta(
                        completed,
                        total,
                        len(devices),
                        task_durations,
                        active_started_at,
                    )
                    _timed_print(
                        f"[worker {worker} {device}] task "
                        f"{task_index + 1}/{total} {shard_name}: {message}",
                        started_at,
                        eta,
                    )
                elif kind == "completed":
                    (
                        _,
                        worker,
                        device,
                        task_index,
                        shard_name,
                        task_seconds,
                        part_count,
                        task_loss_stats,
                    ) = event
                    _merge_quant_loss_stats(quant_loss_stats, task_loss_stats)
                    completed += 1
                    active_started_at.pop(worker, None)
                    task_durations.append(task_seconds)
                    eta = _estimate_parallel_eta(
                        completed,
                        total,
                        len(devices),
                        task_durations,
                        active_started_at,
                    )
                    _timed_print(
                        f"[worker {worker} {device}] completed "
                        f"{completed}/{total}: {shard_name}; "
                        f"task time {_format_duration(task_seconds)}, "
                        f"{part_count} output parts",
                        started_at,
                        eta,
                    )
                elif kind == "error":
                    _, worker, device, task_index, shard_name, details = event
                    raise RuntimeError(
                        f"Worker {worker} ({device}) failed on task "
                        f"{task_index + 1} ({shard_name}):\n{details}"
                    )
                else:
                    raise RuntimeError(f"Unknown worker progress event: {kind!r}.")

            for index in tuple(pending):
                returncode = processes[index].exitcode
                if returncode is None:
                    continue
                pending.remove(index)
                if returncode:
                    raise RuntimeError(
                        f"Quantization worker {index} exited with code {returncode}."
                    )
    except BaseException:
        for index in pending:
            processes[index].terminate()
        raise
    finally:
        for process in processes:
            process.join()
        task_queue.close()
        progress_queue.close()
    return quant_loss_stats


def _weights_config(bits: int, group_size: int) -> dict:
    return {
        "num_bits": bits,
        "group_size": group_size,
        "strategy": "group",
        "symmetric": True,
        "dynamic": False,
        "scale_dtype": "torch.float32",
        "param_dtype": "torch.float16",
        "reserved_code": "binary" if bits == 1 else "zero",
        "packing": "little-endian-bitstream",
    }


def _layer_target(start: int, end: int) -> str:
    pattern = "|".join(str(layer) for layer in range(start, end + 1))
    return (
        rf"re:.*\.layers\.(?:{pattern})\."
        r"block_sparse_moe\.experts"
    )


def _effective_bits(
    moe_rules: list[MoERule],
    linear_rules: list[LinearRule],
) -> tuple[float, float, float]:
    rule_by_layer = {
        layer: rule
        for rule in moe_rules
        for layer in range(rule.start_layer, rule.end_layer + 1)
    }
    expert_bit_sum = sum(
        rule_by_layer[layer].bits + 64 / rule_by_layer[layer].group_size
        for layer in range(1, NUM_MOE_LAYERS + 1)
    )
    payload = (
        sum(rule_by_layer[layer].bits for layer in range(1, NUM_MOE_LAYERS + 1))
        / NUM_MOE_LAYERS
    )
    expert_effective = expert_bit_sum / NUM_MOE_LAYERS
    expert_values_per_layer = (
        NUM_EXPERTS * 3 * EXPERT_INPUT_SIZE * MOE_INTERMEDIATE_SIZE
    )
    linear_bit_sum = sum(rule.bits + 64 / rule.group_size for rule in linear_rules)
    converted_effective = (
        expert_values_per_layer * expert_bit_sum + HIDDEN_SIZE * linear_bit_sum
    ) / (expert_values_per_layer * NUM_MOE_LAYERS + HIDDEN_SIZE * len(linear_rules))
    return payload, expert_effective, converted_effective


def _quantization_config(
    source_config: dict,
    moe_rules: list[MoERule],
    linear_rules: list[LinearRule],
) -> dict:
    config_groups = {}
    for rule in moe_rules:
        name = f"moe_layers_{rule.start_layer}_{rule.end_layer}"
        config_groups[name] = {
            "targets": [_layer_target(rule.start_layer, rule.end_layer)],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(rule.bits, rule.group_size),
        }
    for rule in linear_rules:
        config_groups[f"linear_layer_{rule.layer}_{rule.bits}bit"] = {
            "targets": [rule.prefix],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(rule.bits, rule.group_size),
        }
    payload, expert_effective, converted_effective = _effective_bits(
        moe_rules,
        linear_rules,
    )
    group_overrides = {
        str(layer): rule.group_size
        for rule in moe_rules
        for layer in range(rule.start_layer, rule.end_layer + 1)
    }
    return {
        "quant_method": "cubic",
        "format": CUBIC_FORMAT,
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": source_config.get("ignore", []),
        "runtime_weight_storage": "native_packed_bitstream",
        "layer_bit_schedule": [
            {
                "start_layer": rule.start_layer,
                "end_layer": rule.end_layer,
                "num_bits": rule.bits,
                "group_size": rule.group_size,
            }
            for rule in moe_rules
        ],
        "layer_group_size_overrides": group_overrides,
        "tensor_bit_overrides": [
            {
                "target": rule.prefix,
                "num_bits": rule.bits,
                "group_size": rule.group_size,
            }
            for rule in linear_rules
        ],
        "expert_payload_bits": payload,
        "expert_effective_bits": expert_effective,
        "converted_tensor_effective_bits": converted_effective,
    }


def _copy_model_assets(source: Path, staging: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "cubic_quantization_manifest.json",
        "cubic_quantization_audit.json",
        "cubic_quantization_report.json",
    }
    for path in source.iterdir():
        if (
            path.name in excluded
            or path.name.startswith("model-")
            and path.suffix == ".safetensors"
        ):
            continue
        destination = staging / path.name
        if path.is_dir():
            shutil.copytree(path, destination, symlinks=False)
        else:
            shutil.copy2(path, destination, follow_symlinks=True)


def _finalize_shards(
    staging: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, str], int, list[str]]:
    parts = sorted(staging.glob("part-task-*.safetensors"))
    if not parts:
        raise RuntimeError("No output safetensors were produced.")
    total = len(parts)
    width = max(5, len(str(total)))
    final_names = [
        f"model-{index:0{width}d}-of-{total:0{width}d}.safetensors"
        for index in range(1, total + 1)
    ]
    for part, name in zip(parts, final_names, strict=True):
        os.replace(part, staging / name)

    weight_map = {}
    total_size = 0
    for position, name in enumerate(final_names, start=1):
        path = staging / name
        total_size += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            for key in keys:
                if key in weight_map:
                    raise KeyError(f"Duplicate tensor in output: {key}.")
                weight_map[key] = name
        if progress_callback is not None:
            progress_callback(position, total, name)
    return weight_map, total_size, final_names


def _expected_output_keys(
    source_weight_map: dict[str, str],
    linear_rules: list[LinearRule],
) -> set[str]:
    linear_weights = {f"{rule.prefix}.weight": rule for rule in linear_rules}
    packed_keys = {
        key
        for key in source_weight_map
        if key.endswith(".weight_packed")
        and key.removesuffix(".weight_packed") + ".weight_scale" in source_weight_map
    }
    source_scales = {
        key.removesuffix(".weight_packed") + ".weight_scale" for key in packed_keys
    }
    expected = set()
    for key in source_weight_map:
        if key in source_scales:
            continue
        if key in packed_keys:
            base = key.removesuffix(".weight_packed")
            expected.update(
                {
                    key,
                    base + ".weight_scale",
                    base + ".weight_a",
                    base + ".weight_b",
                }
            )
        elif key in linear_weights:
            base = key.removesuffix(".weight")
            expected.update(
                {
                    base + ".weight_packed",
                    base + ".weight_scale",
                    base + ".weight_a",
                    base + ".weight_b",
                }
            )
        else:
            expected.add(key)
    return expected


def _audit(
    model: Path,
    max_shard_bytes: int,
    expected_keys: set[str],
    expected_widths: set[int],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    if any(path.is_symlink() for path in model.rglob("*")):
        raise ValueError("Output contains a symbolic link.")
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    quantization = text_config["quantization_config"]
    if quantization.get("expert_placement") is not None:
        raise ValueError("Output unexpectedly contains expert_placement.")
    if (
        quantization.get("quant_method") != "cubic"
        or quantization.get("format") != CUBIC_FORMAT
    ):
        raise ValueError("Output does not declare the Cubic format.")
    if quantization["converted_tensor_effective_bits"] > 2.5:
        raise ValueError("Converted effective width exceeds 2.5 bits.")
    widths = {
        group["weights"]["num_bits"] for group in quantization["config_groups"].values()
    }
    if widths != expected_widths:
        raise ValueError(
            f"Expected widths {sorted(expected_widths)}, got {sorted(widths)}."
        )

    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if set(weight_map) != expected_keys:
        missing = expected_keys - weight_map.keys()
        extra = weight_map.keys() - expected_keys
        raise ValueError(
            f"Output key mismatch: missing={len(missing)}, extra={len(extra)}."
        )
    shard_names = sorted(set(weight_map.values()))
    actual_map = {}
    dtype_counts = Counter()
    total_size = 0
    for position, shard_name in enumerate(shard_names, start=1):
        shard = model / shard_name
        if shard.is_symlink() or not shard.is_file():
            raise FileNotFoundError(f"Invalid shard: {shard}.")
        size = shard.stat().st_size
        if size > max_shard_bytes:
            raise ValueError(f"{shard_name} exceeds configured shard size.")
        total_size += size
        with safe_open(shard, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            for key in keys:
                if key in actual_map:
                    raise KeyError(f"Duplicate tensor: {key}.")
                actual_map[key] = shard_name
                dtype = reader.get_slice(key).get_dtype()
                dtype_counts[str(dtype)] += 1
                if key.endswith(".weight_scale") and dtype != "F32":
                    raise ValueError(f"Invalid Cubic scale {key}.")
                if key.endswith((".weight_a", ".weight_b")) and dtype != "F16":
                    raise ValueError(f"Invalid Cubic parameter {key}.")
        if progress_callback is not None:
            progress_callback(position, len(shard_names), shard_name)
    if actual_map != weight_map:
        raise ValueError("Safetensors headers do not match weight_map.")
    if total_size != index["metadata"]["total_size"]:
        raise ValueError("Index total_size does not match materialized files.")
    return {
        "checkpoint": str(model),
        "shards": len(shard_names),
        "tensors": len(weight_map),
        "total_size": total_size,
        "max_shard_bytes": max_shard_bytes,
        "widths_present": sorted(widths),
        "converted_tensor_effective_bits": quantization[
            "converted_tensor_effective_bits"
        ],
        "dtype_counts": dict(dtype_counts),
    }


def _preflight(
    source: Path,
    output: Path,
    staging: Path,
    devices: list[str],
    moe_rules: list[MoERule],
    linear_rules: list[LinearRule],
) -> tuple[dict, dict]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if staging.exists():
        raise FileExistsError(
            f"Incomplete output exists: {staging}. "
            "Inspect and remove it before retrying."
        )
    for name in ("config.json", "model.safetensors.index.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing source file: {source / name}")
    if not devices:
        raise ValueError("At least one conversion device is required.")
    if len(set(devices)) != len(devices):
        raise ValueError("Conversion devices must not contain duplicates.")

    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    source_shards = set(index["weight_map"].values())
    missing_shards = [
        shard for shard in sorted(source_shards) if not (source / shard).is_file()
    ]
    if missing_shards:
        raise FileNotFoundError(f"Missing {len(missing_shards)} source shards.")
    linear_weights = {f"{rule.prefix}.weight" for rule in linear_rules}
    missing_linear = linear_weights - index["weight_map"].keys()
    if missing_linear:
        raise KeyError(
            f"Linear schedule targets are missing: {sorted(missing_linear)}."
        )
    payload, expert_effective, converted_effective = _effective_bits(
        moe_rules,
        linear_rules,
    )
    if converted_effective > 2.5:
        raise ValueError(
            f"Schedule effective width is {converted_effective}, above 2.5."
        )
    return config, index


def quantize(args: argparse.Namespace) -> None:
    started_at = time.monotonic()
    source = args.source.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    moe_rules = _parse_moe_schedule(args.moe_schedule)
    linear_rules = _parse_linear_schedule(args.linear_schedule)
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    max_shard_bytes = int(args.shard_size_gib * 1024**3)
    if max_shard_bytes <= 0:
        raise ValueError("--shard-size-gib must be positive.")
    config, source_index = _preflight(
        source,
        output,
        staging,
        devices,
        moe_rules,
        linear_rules,
    )
    payload, expert_effective, converted_effective = _effective_bits(
        moe_rules,
        linear_rules,
    )
    plan = {
        "source": str(source),
        "output": str(output),
        "temporary_output": str(staging),
        "moe_schedule": args.moe_schedule,
        "linear_schedule": args.linear_schedule,
        "devices": devices,
        "shard_size_gib": args.shard_size_gib,
        "a8_carrier_aware": args.a8_carrier_aware,
        "a8_correction_default": "enabled",
        "fitting_objective": "groupwise-least-squares",
        "reported_loss": "NRMSE = sqrt(joint SSE / source weight SSE)",
        "row_chunk_size": args.row_chunk_size,
        "source_shards": len(set(source_index["weight_map"].values())),
        "expert_payload_bits": payload,
        "expert_effective_bits": expert_effective,
        "converted_tensor_effective_bits": converted_effective,
        "worker_scheduling": "dynamic_source_shard_queue",
        "output_partitioning": "deterministic_source_shard_index",
    }
    _timed_print(
        json.dumps(plan, ensure_ascii=False, indent=2),
        started_at,
        None,
    )
    if args.plan:
        total_elapsed = time.monotonic() - started_at
        _timed_print(
            f"Plan validation completed. Total elapsed: "
            f"{_format_duration(total_elapsed)}.",
            started_at,
            0,
        )
        return

    staging.mkdir(parents=True)
    source_shards = sorted(set(source_index["weight_map"].values()))
    raw_quant_loss_stats = _run_workers(
        devices,
        source_shards,
        source,
        staging,
        moe_rules,
        linear_rules,
        max_shard_bytes,
        args.row_chunk_size,
        args.tensor_batch_size,
        args.a8_carrier_aware,
        started_at,
    )
    loss_statistics = _quant_loss_report(
        raw_quant_loss_stats,
        args.a8_carrier_aware,
    )
    _timed_print(
        "All source shards converted; finalizing output shards.",
        started_at,
        None,
    )
    finalize_started_at = time.monotonic()

    def report_finalize(position: int, total: int, name: str) -> None:
        eta = _estimate_eta(finalize_started_at, position, total)
        _timed_print(
            f"[finalize] indexed {position}/{total}: {name}",
            started_at,
            eta,
        )

    weight_map, total_size, shard_names = _finalize_shards(
        staging,
        report_finalize,
    )
    expected_keys = _expected_output_keys(
        source_index["weight_map"],
        linear_rules,
    )
    if set(weight_map) != expected_keys:
        raise ValueError("Converted keys do not match the source model.")

    text_config = config.get("text_config", config)
    source_quantization = text_config.get("quantization_config", {})
    text_config["quantization_config"] = _quantization_config(
        source_quantization,
        moe_rules,
        linear_rules,
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (staging / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _copy_model_assets(source, staging)

    manifest = {
        **plan,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "source_config_sha256": _sha256(source / "config.json"),
        "source_index_sha256": _sha256(source / "model.safetensors.index.json"),
        "output_shards": len(shard_names),
        "output_total_bytes": total_size,
        "combined_report": "cubic_quantization_report.json",
    }
    _timed_print(
        "Output metadata written; auditing materialized checkpoint.",
        started_at,
        None,
    )
    audit_started_at = time.monotonic()

    def report_audit(position: int, total: int, name: str) -> None:
        eta = _estimate_eta(audit_started_at, position, total)
        _timed_print(
            f"[audit] checked {position}/{total}: {name}",
            started_at,
            eta,
        )

    audit = _audit(
        staging,
        max_shard_bytes,
        expected_keys,
        {
            *(rule.bits for rule in moe_rules),
            *(rule.bits for rule in linear_rules),
        },
        report_audit,
    )
    combined_report = {
        "manifest": manifest,
        "loss_statistics": loss_statistics,
        "audit": audit,
    }
    (staging / "cubic_quantization_report.json").write_text(
        json.dumps(combined_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    _timed_print(
        json.dumps(
            {
                "loss_by_bit": loss_statistics["by_bit"],
                "loss_by_bit_and_group_size": loss_statistics["by_bit_and_group_size"],
                "audit": audit,
                "combined_report": str(output / "cubic_quantization_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        started_at,
        0,
    )
    _timed_print(
        f"Quantized checkpoint written to {output}.",
        started_at,
        0,
    )
    total_elapsed = time.monotonic() - started_at
    _timed_print(
        f"Total elapsed: {_format_duration(total_elapsed)}.",
        started_at,
        0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Source Kimi-K3 MXFP4 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New materialized Cubic checkpoint; must not exist.",
    )
    parser.add_argument(
        "--moe-schedule",
        default=MOE_SCHEDULE,
        help="MoE rules formatted START-END:BITS@GROUP_SIZE.",
    )
    parser.add_argument(
        "--linear-schedule",
        default=LINEAR_SCHEDULE,
        help=(
            "Optional mlp_res_proj rules formatted LAYER:BITS@GROUP_SIZE; "
            "disabled by default because these tensors score residual streams."
        ),
    )
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help=(
            "Comma-separated conversion-only devices, one worker per device; "
            "independent of the inference topology."
        ),
    )
    parser.add_argument(
        "--shard-size-gib",
        type=float,
        default=DEFAULT_SHARD_SIZE_GIB,
        help="Maximum size of each materialized safetensors shard.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_ROW_CHUNK_SIZE,
        help=(
            "Rows per high-bit calibration chunk; -1 selects a memory-aware "
            "value, while a positive value forces an exact chunk size."
        ),
    )
    parser.add_argument(
        "--tensor-batch-size",
        type=int,
        default=DEFAULT_TENSOR_BATCH_SIZE,
        help="MXFP4 expert tensors converted together on each GPU.",
    )
    a8_group = parser.add_mutually_exclusive_group()
    a8_group.add_argument(
        "--a8-carrier-aware",
        dest="a8_carrier_aware",
        action="store_true",
        default=True,
        help=(
            "Jointly fit continuous Cubic and round(127*q)/127 for Dynamic A8; "
            "enabled by default."
        ),
    )
    a8_group.add_argument(
        "--disable-a8-correction",
        dest="a8_carrier_aware",
        action="store_false",
        help=(
            "Disable round(127*q)/127 carrier correction and fit only the "
            "continuous Cubic reconstruction."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate and print the conversion plan without writing.",
    )
    args = parser.parse_args()
    if args.row_chunk_size == 0 or args.row_chunk_size < -1:
        parser.error("--row-chunk-size must be -1 or a positive integer.")
    if args.tensor_batch_size <= 0:
        parser.error("--tensor-batch-size must be positive.")
    return args


if __name__ == "__main__":
    quantize(parse_args())
