# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit a Kimi-K3 Cubic checkpoint without loading model weights to a GPU."""

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open

from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_FORMAT,
    CubicConfig,
)

NUM_EXPERTS = 896
EXPERT_INPUT_SIZE = 3584
MOE_INTERMEDIATE_SIZE = 3072
LINEAR_INPUT_SIZE = 7168


def _layer_layout(quantization: dict) -> dict[int, tuple[int, int]]:
    schedule = quantization.get("layer_bit_schedule", [])
    group_overrides = {
        int(layer): int(size)
        for layer, size in quantization.get("layer_group_size_overrides", {}).items()
    }
    fallback_group = int(
        quantization["config_groups"]["fused_moe_fallback"]["weights"]["group_size"]
    )
    layout: dict[int, tuple[int, int]] = {}
    for rule in schedule:
        for layer in range(
            int(rule["start_layer"]),
            int(rule["end_layer"]) + 1,
        ):
            if layer in layout:
                raise ValueError(f"Layer {layer} has overlapping bit rules.")
            layout[layer] = (
                int(rule["num_bits"]),
                group_overrides.get(layer, fallback_group),
            )
    if set(layout) != set(range(1, 93)):
        raise ValueError("Cubic Kimi-K3 schedule must cover layers 1--92.")
    return layout


def _expected_shape(
    projection: str,
    suffix: str,
    bits: int,
    group_size: int,
) -> tuple[int, int]:
    if projection in ("w1", "w3"):
        output_size = MOE_INTERMEDIATE_SIZE
        input_size = EXPERT_INPUT_SIZE
    else:
        output_size = EXPERT_INPUT_SIZE
        input_size = MOE_INTERMEDIATE_SIZE
    width = (
        math.ceil(input_size * bits / 8)
        if suffix == "weight_packed"
        else math.ceil(input_size / group_size)
    )
    return output_size, width


def audit(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    quantization = text_config["quantization_config"]
    if (
        quantization.get("quant_method") != "cubic"
        or quantization.get("format") != CUBIC_FORMAT
        or quantization.get("runtime_weight_storage") != "native_packed_bitstream"
    ):
        raise ValueError("Checkpoint does not declare native Cubic storage.")
    CubicConfig.from_config(quantization)
    layout = _layer_layout(quantization)
    tensor_overrides = quantization.get("tensor_bit_overrides", [])
    widths = {bits for bits, _ in layout.values()} | {
        int(rule["num_bits"]) for rule in tensor_overrides
    }
    if widths != set(range(1, 9)):
        raise ValueError(f"Expected Cubic widths 1--8, got {sorted(widths)}.")
    if layout[92][0] != 4:
        raise ValueError("The final Kimi-K3 layer must be 4-bit.")
    effective_bits = float(quantization["converted_tensor_effective_bits"])
    if effective_bits > 2.5:
        raise ValueError(f"Effective width exceeds 2.5: {effective_bits}.")

    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    headers = {}
    for shard_name in shard_names:
        shard = model / shard_name
        if not shard.exists():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard}.")
        with safe_open(shard, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            for key in keys:
                tensor = reader.get_slice(key)
                headers[key] = (
                    tuple(tensor.get_shape()),
                    tensor.get_dtype(),
                )

    dtypes = {
        "weight_packed": "U8",
        "weight_scale": "F32",
        "weight_a": "F16",
        "weight_b": "F16",
    }
    expert_tensor_count = 0
    for layer, (bits, group_size) in layout.items():
        for expert in range(NUM_EXPERTS):
            for projection in ("w1", "w2", "w3"):
                base = (
                    f"language_model.model.layers.{layer}."
                    f"block_sparse_moe.experts.{expert}.{projection}"
                )
                for suffix, dtype in dtypes.items():
                    key = f"{base}.{suffix}"
                    if key not in weight_map or key not in headers:
                        raise KeyError(f"Missing Cubic expert tensor: {key}.")
                    shape, actual_dtype = headers[key]
                    expected_shape = _expected_shape(
                        projection, suffix, bits, group_size
                    )
                    if shape != expected_shape or actual_dtype != dtype:
                        raise ValueError(
                            f"{key}: got {shape}/{actual_dtype}, expected "
                            f"{expected_shape}/{dtype}."
                        )
                    expert_tensor_count += 1

    override_widths = {int(rule["num_bits"]) for rule in tensor_overrides}
    if override_widths != {1, 5, 6, 7, 8}:
        raise ValueError(
            "Kimi-K3 AllBits Linear overrides must provide 1/5/6/7/8-bit "
            f"widths, got {sorted(override_widths)}."
        )
    linear_targets = []
    for linear in tensor_overrides:
        linear_prefix = str(linear["target"])
        linear_bits = int(linear["num_bits"])
        linear_group_size = int(linear["group_size"])
        if f"{linear_prefix}.weight" in weight_map:
            raise ValueError(f"{linear_prefix} still contains an unpacked weight.")
        linear_expected = {
            "weight_packed": (
                (1, math.ceil(LINEAR_INPUT_SIZE * linear_bits / 8)),
                "U8",
            ),
            "weight_scale": (
                (1, math.ceil(LINEAR_INPUT_SIZE / linear_group_size)),
                "F32",
            ),
            "weight_a": (
                (1, math.ceil(LINEAR_INPUT_SIZE / linear_group_size)),
                "F16",
            ),
            "weight_b": (
                (1, math.ceil(LINEAR_INPUT_SIZE / linear_group_size)),
                "F16",
            ),
        }
        for suffix, expected in linear_expected.items():
            key = f"{linear_prefix}.{suffix}"
            if key not in weight_map or headers.get(key) != expected:
                raise ValueError(f"{key}: got {headers.get(key)}, expected {expected}.")
        linear_targets.append({"target": linear_prefix, "num_bits": linear_bits})

    total_size = sum((model / shard).stat().st_size for shard in shard_names)
    indexed_size = int(index.get("metadata", {}).get("total_size", -1))
    if total_size != indexed_size:
        raise ValueError(f"Index total_size={indexed_size}, actual={total_size}.")
    return {
        "checkpoint": str(model),
        "shards": len(shard_names),
        "expert_tensors": expert_tensor_count,
        "experts_per_layer": NUM_EXPERTS,
        "layers": len(layout),
        "widths_present": sorted(widths),
        "layer_0": "BF16 except 8-bit mlp_res_proj",
        "layer_92_bits": layout[92][0],
        "linear_bit_targets": linear_targets,
        "expert_payload_bits": quantization["expert_payload_bits"],
        "expert_effective_bits": quantization["expert_effective_bits"],
        "converted_tensor_effective_bits": effective_bits,
        "index_total_bytes": total_size,
        "all_headers_readable": True,
        "uniform_width_within_each_fused_moe_layer": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.model.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
