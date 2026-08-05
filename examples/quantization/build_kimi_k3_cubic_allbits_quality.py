# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the quality-and-speed Kimi-K3 Cubic 1--8-bit checkpoint view.

The builder reuses completed Cubic checkpoints through symlinks. FusedMoE
layers use the proven 2/3-bit paths, the final layer returns to 4-bit, and
small residual Linear tensors carry the remaining 1/5/6/7/8-bit widths. No
FusedMoE layer mixes widths across its experts or projections.
"""

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_FORMAT,
    CubicConfig,
    quantize_cubic,
)

NUM_MOE_LAYERS = 92
NUM_EXPERTS = 896
HIDDEN_SIZE = 7168
EXPERT_INPUT_SIZE = 3584
MOE_INTERMEDIATE_SIZE = 3072
DEFAULT_GROUP_SIZE = 512
QUALITY_GROUP_SIZE = 256
# Preserve the quality-critical leading experts at 3-bit/G256. Using G512
# for 17 layers saves the metadata needed to keep the all-in effective
# average below 2.5 bits without reducing any leading layer to 2-bit.
QUALITY_LAYERS = (*range(1, 24), *range(40, 92))
LINEAR_BITS = {
    0: 8,
    1: 7,
    2: 6,
    3: 5,
    48: 1,
}
LAYER_BITS = {
    **{layer: 3 for layer in range(1, 24)},
    **{layer: 2 for layer in range(24, 92)},
    92: 4,
}


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


def _effective_bits() -> tuple[float, float, float]:
    expert_values_per_layer = (
        NUM_EXPERTS * 3 * EXPERT_INPUT_SIZE * MOE_INTERMEDIATE_SIZE
    )
    expert_bit_sum = sum(
        LAYER_BITS[layer]
        + 64 / (QUALITY_GROUP_SIZE if layer in QUALITY_LAYERS else DEFAULT_GROUP_SIZE)
        for layer in range(1, NUM_MOE_LAYERS + 1)
    )
    expert_effective = expert_bit_sum / NUM_MOE_LAYERS
    linear_bit_sum = sum(
        bits + 64 / DEFAULT_GROUP_SIZE for bits in LINEAR_BITS.values()
    )
    converted_effective = (
        expert_values_per_layer * expert_bit_sum + HIDDEN_SIZE * linear_bit_sum
    ) / (expert_values_per_layer * NUM_MOE_LAYERS + HIDDEN_SIZE * len(LINEAR_BITS))
    payload = sum(LAYER_BITS.values()) / NUM_MOE_LAYERS
    return payload, expert_effective, converted_effective


def _quantization_config(old_config: dict) -> dict:
    payload, expert_effective, converted_effective = _effective_bits()
    config_groups = {
        "fused_moe_fallback": {
            "targets": ["FusedMoE"],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(2, DEFAULT_GROUP_SIZE),
        },
        "layers_1_23": {
            "targets": [_layer_target(1, 23)],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(3, QUALITY_GROUP_SIZE),
        },
        "layers_24_39": {
            "targets": [_layer_target(24, 39)],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(2, DEFAULT_GROUP_SIZE),
        },
        "layers_40_91": {
            "targets": [_layer_target(40, 91)],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(2, QUALITY_GROUP_SIZE),
        },
        "layer_92": {
            "targets": [_layer_target(92, 92)],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(4, DEFAULT_GROUP_SIZE),
        },
    }
    for layer, bits in LINEAR_BITS.items():
        prefix = f"language_model.model.layers.{layer}.mlp_res_proj"
        config_groups[f"linear_layer_{layer}_{bits}bit"] = {
            "targets": [prefix],
            "input_activations": None,
            "output_activations": None,
            "weights": _weights_config(bits, DEFAULT_GROUP_SIZE),
        }
    schedule = (
        (1, 23, 3),
        (24, 91, 2),
        (92, 92, 4),
    )
    result = {
        "quant_method": "cubic",
        "format": CUBIC_FORMAT,
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": old_config.get("ignore", []),
        "runtime_weight_storage": "native_packed_bitstream",
        "layer_bit_schedule": [
            {
                "start_layer": start,
                "end_layer": end,
                "num_bits": bits,
            }
            for start, end, bits in schedule
        ],
        "layer_group_size_overrides": {
            str(layer): QUALITY_GROUP_SIZE for layer in QUALITY_LAYERS
        },
        "tensor_bit_overrides": [
            {
                "target": (f"language_model.model.layers.{layer}.mlp_res_proj"),
                "num_bits": bits,
                "group_size": DEFAULT_GROUP_SIZE,
            }
            for layer, bits in LINEAR_BITS.items()
        ],
        "expert_payload_bits": payload,
        "expert_effective_bits": expert_effective,
        "converted_tensor_effective_bits": converted_effective,
    }
    if "expert_placement" in old_config:
        result["expert_placement"] = old_config["expert_placement"]
    CubicConfig.from_config(result)
    return result


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _materialize_linear_shard(
    source: Path,
    destination: Path,
    linear_bits: dict[str, int],
) -> tuple[str, ...]:
    linear_weights = {f"{prefix}.weight": bits for prefix, bits in linear_bits.items()}
    with safe_open(source, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        missing = linear_weights.keys() - keys
        if missing:
            raise KeyError(f"{source} does not contain {sorted(missing)}.")
        tensors = {
            key: reader.get_tensor(key) for key in keys if key not in linear_weights
        }
        new_keys: list[str] = []
        for prefix, bits in linear_bits.items():
            quantized = quantize_cubic(
                reader.get_tensor(f"{prefix}.weight"),
                total_bits=bits,
                group_size=DEFAULT_GROUP_SIZE,
            )
            names = (
                f"{prefix}.weight_packed",
                f"{prefix}.weight_scale",
                f"{prefix}.weight_a",
                f"{prefix}.weight_b",
            )
            tensors[names[0]] = quantized.packed
            tensors[names[1]] = quantized.scale
            tensors[names[2]] = quantized.a
            tensors[names[3]] = quantized.b
            new_keys.extend(names)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, temporary)
    os.replace(temporary, destination)
    return tuple(new_keys)


def build(args: argparse.Namespace) -> None:
    allbits = args.allbits.resolve()
    curve3 = args.curve3.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    allbits_config = json.loads((allbits / "config.json").read_text())
    text_config = allbits_config.get("text_config", allbits_config)
    old_quantization = text_config["quantization_config"]
    text_config["quantization_config"] = _quantization_config(old_quantization)

    index = json.loads((allbits / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    linear_specs_by_shard: dict[str, dict[str, int]] = {}
    for layer, bits in LINEAR_BITS.items():
        prefix = f"language_model.model.layers.{layer}.mlp_res_proj"
        weight_name = f"{prefix}.weight"
        shard_name = weight_map.pop(weight_name)
        linear_specs_by_shard.setdefault(shard_name, {})[prefix] = bits
    quality_shards = {
        weight_map[
            f"language_model.model.layers.{layer}."
            "block_sparse_moe.experts.0.w1.weight_packed"
        ]
        for layer in QUALITY_LAYERS
    }
    for shard in sorted(allbits.glob("model-*.safetensors")):
        destination = output / shard.name
        source = curve3 / shard.name if shard.name in quality_shards else shard
        linear_bits = linear_specs_by_shard.get(shard.name)
        if linear_bits:
            new_keys = _materialize_linear_shard(
                source,
                destination,
                linear_bits,
            )
            for key in new_keys:
                weight_map[key] = shard.name
        else:
            _link(source, destination)

    for path in allbits.iterdir():
        if path.name in {
            "config.json",
            "model.safetensors.index.json",
        } or path.name.startswith("model-"):
            continue
        _link(path, output / path.name)

    total_size = sum(path.stat().st_size for path in output.glob("model-*.safetensors"))
    index.setdefault("metadata", {})["total_size"] = total_size
    (output / "config.json").write_text(
        json.dumps(allbits_config, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    )
    payload, expert_effective, converted_effective = _effective_bits()
    manifest = {
        "format": CUBIC_FORMAT,
        "base_checkpoint": str(allbits),
        "quality_checkpoint": str(curve3),
        "materialized_shards": sorted(linear_specs_by_shard),
        "quality_layers": list(QUALITY_LAYERS),
        "linear_bit_targets": [
            {
                "target": (f"language_model.model.layers.{layer}.mlp_res_proj"),
                "num_bits": bits,
            }
            for layer, bits in LINEAR_BITS.items()
        ],
        "widths_present": list(range(1, 9)),
        "layer_0": "BF16 except 8-bit mlp_res_proj",
        "layer_92_bits": LAYER_BITS[92],
        "expert_payload_bits": payload,
        "expert_effective_bits": expert_effective,
        "converted_tensor_effective_bits": converted_effective,
        "total_size": total_size,
    }
    (output / "cubic_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("allbits", type=Path)
    parser.add_argument("curve3", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
