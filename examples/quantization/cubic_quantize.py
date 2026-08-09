# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert a Hugging Face checkpoint to the Cubic packed format.

The MXFP4 input path is intended for the public Kimi-K3 checkpoint. It decodes
the existing E2M1/E8M0 expert weights before fitting Cubic groups. Uniform and
per-layer 1--8-bit schedules are supported without widening packed weights at
runtime.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_FORMAT,
    _fit_scale,
    cubic_levels,
    pack_cubic_codes,
)

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, device: torch.device
) -> torch.Tensor:
    packed = packed.to(device=device, non_blocking=True)
    scale = scale.to(device=device, non_blocking=True)
    lookup = torch.tensor(
        _E2M1_LEVELS + tuple(-x for x in _E2M1_LEVELS),
        device=device,
        dtype=torch.float32,
    )
    low = lookup[(packed & 0x0F).to(torch.long)]
    high = lookup[(packed >> 4).to(torch.long)]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    scale_f32 = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127
    )
    return values * scale_f32.repeat_interleave(32, dim=-1)


def _quantize_symmetric_cubic(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"Kimi-K3 expert K={weight.shape[-1]} is not divisible by "
            f"group_size={group_size}."
        )
    groups = weight.reshape(-1, group_size)
    absolute = groups.abs()
    if bits == 1:
        scale = absolute.mean(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        codes = torch.where(groups < 0, -1, 1).to(torch.int8)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        scale = absolute.amax(dim=-1)
        scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
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
    codes = codes.reshape(weight.shape)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (*weight.shape[:-1], weight.shape[-1] // group_size)
    scale = scale.reshape(metadata_shape).to(torch.float32)
    a = torch.ones(metadata_shape, device=weight.device, dtype=torch.float16)
    b = torch.zeros(metadata_shape, device=weight.device, dtype=torch.float16)
    return packed, scale, a, b


def _quantize_kimi_cubic(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"Kimi-K3 expert K={weight.shape[-1]} is not divisible by "
            f"group_size={group_size}."
        )
    groups = weight.to(torch.float32).reshape(-1, group_size)
    calibration_stride = {
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 8,
        8: 8,
    }[bits]
    calibration_groups = groups[:, ::calibration_stride]
    valid = torch.ones_like(calibration_groups)
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
        levels = cubic_levels(
            bits,
            a_value,
            b_value,
            device=weight.device,
            dtype=torch.float32,
        )
        for multiplier in multipliers:
            scale, _, loss = _fit_scale(
                calibration_groups,
                valid,
                levels,
                group_amax * multiplier,
                iterations=iterations,
            )
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_scale = torch.where(improved, scale, best_scale)
            best_a = torch.where(improved, a_value, best_a)
            best_b = torch.where(improved, b_value, best_b)

    stored_scale = best_scale.to(torch.float32)
    stored_a = best_a.to(torch.float16)
    stored_b = best_b.to(torch.float16)
    levels = cubic_levels(
        bits,
        stored_a,
        stored_b,
        device=weight.device,
        dtype=torch.float32,
    )
    distances = (
        groups.abs()[..., None] - stored_scale[:, None, None] * levels[:, None, :]
    ).abs()
    magnitudes = distances.argmin(dim=-1)
    codes = (groups.sign().to(torch.int64) * magnitudes).reshape(weight.shape)
    metadata_shape = (*weight.shape[:-1], weight.shape[-1] // group_size)
    return (
        pack_cubic_codes(codes, bits),
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
    if bits <= 2:
        return _quantize_symmetric_cubic(weight, bits, group_size)
    if bits == 3:
        return _quantize_kimi_cubic(weight, bits, group_size)
    packed_chunks = []
    scale_chunks = []
    a_chunks = []
    b_chunks = []
    for chunk in weight.reshape(-1, weight.shape[-1]).split(row_chunk_size):
        packed, scale, a, b = _quantize_kimi_cubic(chunk, bits, group_size)
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


LayerBitRule = tuple[int, int, int]


def _parse_layer_bit_schedule(value: str | None) -> list[LayerBitRule]:
    if not value:
        return []
    rules: list[LayerBitRule] = []
    covered: set[int] = set()
    for item in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?:(\d+)", item.strip())
        if match is None:
            raise ValueError(
                "Layer schedule entries must use START-END:BITS or LAYER:BITS."
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        bits = int(match.group(3))
        if start > end or bits not in range(1, 9):
            raise ValueError(f"Invalid layer schedule entry: {item!r}.")
        layers = set(range(start, end + 1))
        if covered & layers:
            raise ValueError("Cubic layer schedule ranges must not overlap.")
        covered |= layers
        rules.append((start, end, bits))
    return rules


def _bits_for_key(key: str, default_bits: int, rules: list[LayerBitRule]) -> int:
    match = re.search(r"\.layers\.(\d+)\.", key)
    if match is None:
        return default_bits
    layer = int(match.group(1))
    return next(
        (bits for start, end, bits in rules if start <= layer <= end),
        default_bits,
    )


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


def _cubic_config(
    bits: int,
    group_size: int,
    old_config: dict,
    layer_rules: list[LayerBitRule],
    expert_placement: dict | None = None,
) -> dict:
    config_groups = {
        "fallback": {
            "targets": ["Linear", "FusedMoE"],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(bits, group_size),
        }
    }
    for start, end, layer_bits in layer_rules:
        layer_pattern = "|".join(str(layer) for layer in range(start, end + 1))
        config_groups[f"layers_{start}_{end}"] = {
            "targets": [
                (
                    rf"re:.*\.layers\.(?:{layer_pattern})\."
                    r"block_sparse_moe\.experts"
                )
            ],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(layer_bits, group_size),
        }
    config = {
        "quant_method": "cubic",
        "format": CUBIC_FORMAT,
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": old_config.get("ignore", []),
        "runtime_weight_storage": "native_packed_bitstream",
    }
    if layer_rules:
        layer_count = sum(end - start + 1 for start, end, _ in layer_rules)
        payload_bits = (
            sum(
                (end - start + 1) * layer_bits for start, end, layer_bits in layer_rules
            )
            / layer_count
        )
        config["layer_bit_schedule"] = [
            {"start_layer": start, "end_layer": end, "num_bits": layer_bits}
            for start, end, layer_bits in layer_rules
        ]
        config["converted_tensor_effective_bits"] = payload_bits + 64 / group_size
    if expert_placement is not None:
        config["expert_placement"] = expert_placement
    return config


def _convert_shard(
    shard_path: Path,
    device: torch.device,
    bits: int,
    layer_rules: list[LayerBitRule],
    group_size: int,
    row_chunk_size: int,
    tensor_batch_size: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    tensors: dict[str, torch.Tensor] = {}
    added_keys: list[str] = []
    with safe_open(shard_path, framework="pt", device="cpu") as reader:
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
            if key not in source_scale_keys and key not in packed_keys:
                tensors[key] = reader.get_tensor(key)

        shape_groups: dict[tuple[tuple[int, ...], tuple[int, ...], int], list[str]] = (
            defaultdict(list)
        )
        for key in packed_keys:
            base = key.removesuffix(".weight_packed")
            shape_groups[
                (
                    tuple(reader.get_slice(key).get_shape()),
                    tuple(reader.get_slice(base + ".weight_scale").get_shape()),
                    _bits_for_key(key, bits, layer_rules),
                )
            ].append(key)

        for grouped_keys in shape_groups.values():
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
                weight = _decode_mxfp4(source_packed, source_scale, device)
                tensor_bits = _bits_for_key(batch_keys[0], bits, layer_rules)
                packed, scale, a, b = _quantize_weight(
                    weight, tensor_bits, group_size, row_chunk_size
                )
                packed_cpu = packed.cpu()
                scale_cpu = scale.cpu()
                a_cpu = a.cpu()
                b_cpu = b.cpu()
                for index, key in enumerate(batch_keys):
                    base = key.removesuffix(".weight_packed")
                    scale_key = base + ".weight_scale"
                    a_key = base + ".weight_a"
                    b_key = base + ".weight_b"
                    tensors[key] = packed_cpu[index].clone()
                    tensors[scale_key] = scale_cpu[index].clone()
                    tensors[a_key] = a_cpu[index].clone()
                    tensors[b_key] = b_cpu[index].clone()
                    added_keys.extend((a_key, b_key))
                del (
                    source_packed,
                    source_scale,
                    weight,
                    packed,
                    scale,
                    a,
                    b,
                    packed_cpu,
                    scale_cpu,
                    a_cpu,
                    b_cpu,
                )
    return tensors, added_keys


def convert_checkpoint(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.resume or args.overwrite)
    device = torch.device(args.device)

    with (source / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    text_config = config.get("text_config", config)
    old_quant_config = text_config.get("quantization_config", {})
    layer_rules = _parse_layer_bit_schedule(args.layer_bit_schedule)
    expert_placement = None
    if args.expert_placement_from is not None:
        placement_path = args.expert_placement_from
        if placement_path.is_dir():
            placement_path = placement_path / "config.json"
        with placement_path.open(encoding="utf-8") as file:
            placement_config = json.load(file)
        if placement_config.get("strategy") == "explicit":
            expert_placement = placement_config
        else:
            placement_text_config = placement_config.get(
                "text_config", placement_config
            )
            expert_placement = placement_text_config.get("quantization_config", {}).get(
                "expert_placement"
            )
        if expert_placement is None:
            raise ValueError(
                "--expert-placement-from does not contain expert_placement."
            )
    text_config["quantization_config"] = _cubic_config(
        args.bits,
        args.group_size,
        old_quant_config,
        layer_rules,
        expert_placement,
    )

    index_path = source / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    weight_map = dict(index["weight_map"])
    for key, shard_name in tuple(weight_map.items()):
        if not key.endswith(".weight_packed"):
            continue
        base = key.removesuffix(".weight_packed")
        if base + ".weight_scale" in weight_map:
            weight_map[base + ".weight_a"] = shard_name
            weight_map[base + ".weight_b"] = shard_name
    total_size = 0

    all_shards = list(sorted(source.glob("model-*.safetensors")))
    selected_shards = [
        (shard_number, shard_path)
        for shard_number, shard_path in enumerate(all_shards, start=1)
        if (shard_number - 1) % args.shard_worker_count == args.shard_worker_index
    ]
    for shard_number, shard_path in selected_shards:
        destination = output / shard_path.name
        if args.resume and not args.overwrite and destination.exists():
            with safe_open(destination, framework="pt", device="cpu") as reader:
                len(reader.keys())
            total_size += destination.stat().st_size
            print(
                f"[{shard_number:03d}] {shard_path.name}: already complete",
                flush=True,
            )
            continue
        tensors, added_keys = _convert_shard(
            shard_path,
            device,
            args.bits,
            layer_rules,
            args.group_size,
            args.row_chunk_size,
            args.tensor_batch_size,
        )
        for key in added_keys:
            weight_map[key] = shard_path.name

        temporary = destination.with_name(destination.name + ".tmp")
        save_file(tensors, temporary)
        temporary.replace(destination)
        total_size += destination.stat().st_size
        print(
            f"[{shard_number:03d}] {shard_path.name}: "
            f"{destination.stat().st_size / (1024**3):.2f} GiB",
            flush=True,
        )
        del tensors
        if device.type == "cuda":
            torch.accelerator.empty_cache()

    if args.shard_worker_count != 1:
        print(
            f"Cubic shard worker {args.shard_worker_index}/"
            f"{args.shard_worker_count} completed.",
            flush=True,
        )
        return

    index["weight_map"] = weight_map
    index.setdefault("metadata", {})["total_size"] = total_size
    with (output / index_path.name).open("w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (output / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")

    for path in source.iterdir():
        if (
            path.name in {"config.json", index_path.name}
            or path.suffix == ".safetensors"
        ):
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(
                path,
                destination,
                dirs_exist_ok=args.resume or args.overwrite,
            )
        else:
            shutil.copy2(path, destination)

    if layer_rules:
        layer_count = sum(end - start + 1 for start, end, _ in layer_rules)
        payload_bits = (
            sum((end - start + 1) * bits for start, end, bits in layer_rules)
            / layer_count
        )
    else:
        payload_bits = args.bits
    effective_bits = payload_bits + 64 / args.group_size
    print(f"Cubic checkpoint written to {output}")
    print(f"Converted-tensor effective width: {effective_bits:.4f} bits/weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--layer-bit-schedule",
        help="Per-layer overrides such as '1-23:3,24-92:2'.",
    )
    parser.add_argument(
        "--expert-placement-from",
        type=Path,
        help=(
            "Copy an explicit expert_placement from a JSON file or another "
            "model config."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk-size", type=int, default=32)
    parser.add_argument("--tensor-batch-size", type=int, default=32)
    parser.add_argument("--shard-worker-index", type=int, default=0)
    parser.add_argument("--shard-worker-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_worker_index < args.shard_worker_count:
        parser.error("shard worker index must be in [0, shard worker count).")
    try:
        layer_rules = _parse_layer_bit_schedule(args.layer_bit_schedule)
    except ValueError as error:
        parser.error(str(error))
    if layer_rules:
        count = sum(end - start + 1 for start, end, _ in layer_rules)
        payload_bits = (
            sum((end - start + 1) * bits for start, end, bits in layer_rules) / count
        )
    else:
        payload_bits = args.bits
    if payload_bits + 64 / args.group_size > 2.5:
        print(
            "Warning: selected Cubic metadata layout exceeds 2.5 bits/weight "
            "for converted tensors."
        )
    return args


if __name__ == "__main__":
    convert_checkpoint(parse_args())
