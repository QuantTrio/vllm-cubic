# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_COMPACT_METADATA_FORMAT,
    CUBIC_E5M9_CURVE2_METADATA_FORMAT,
    CUBIC_FORMAT,
    CubicConfig,
    CubicEmbeddingMethod,
    CubicLinearMetadataParameter,
    CubicLinearMethod,
    CubicScheme,
    cubic_is_strictly_monotonic,
    cubic_levels,
    decode_cubic_compact_metadata,
    decode_cubic_e5m9_curve2_metadata,
    dequantize_cubic,
    pack_cubic_codes,
    quantize_cubic,
    unpack_cubic_codes,
)


def test_cubic_selects_embedding_method_for_vocab_parallel_layer(monkeypatch):
    import vllm.model_executor.layers.vocab_parallel_embedding as vocab_module

    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "embedding": {
                    "targets": ["embed_tokens"],
                    "weights": {
                        "num_bits": 8,
                        "group_size": [1, 32],
                        "metadata_format": CUBIC_E5M9_CURVE2_METADATA_FORMAT,
                        "scale_dtype": "packed_e5m9",
                        "param_dtype": "torch.float16",
                        "curve_count": 4,
                    },
                }
            },
        }
    )
    layer = vocab_module.VocabParallelEmbedding(
        128,
        32,
        quant_config=config,
        prefix="embed_tokens",
    )

    assert isinstance(layer.quant_method, CubicEmbeddingMethod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_curve2_embedding_gathers_without_full_weight_expansion(bits):
    device = torch.device("cuda")
    outputs, inputs, group_size = 128, 128, 32
    if bits == 1:
        codes = torch.randint(0, 2, (outputs, inputs), device=device).mul(2).sub(1)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            device=device,
        )
    codes = codes.to(torch.int16)
    layer = torch.nn.Module()
    tensors = {
        "weight_packed": pack_cubic_codes(codes, bits),
        "weight_metadata": _encode_e5m9_curve2(
            torch.full((outputs, inputs // group_size), 0.03125, device=device),
            torch.randint(
                0,
                4,
                (outputs, inputs // group_size),
                device=device,
            ),
        ),
        "weight_curve_a": torch.tensor(
            [[1.0, 0.75, 0.5, 0.25]], device=device, dtype=torch.float16
        ),
        "weight_curve_b": torch.tensor(
            [[0.0, 0.125, 0.25, 0.375]], device=device, dtype=torch.float16
        ),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, False))
    layer.input_size_per_partition = inputs
    layer.output_size_per_partition = outputs
    layer.output_partition_sizes = [outputs]
    layer.params_dtype = torch.bfloat16
    method = CubicEmbeddingMethod(
        CubicScheme(
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            metadata_format=CUBIC_E5M9_CURVE2_METADATA_FORMAT,
            reserved_code="binary" if bits == 1 else "zero",
        )
    )
    method.process_weights_after_loading(layer)
    ids = torch.tensor([[7, 3, 7], [1, 127, 0]], device=device)

    actual = method.embedding(layer, ids)
    expected = method.dequantize(layer)[ids]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_compact_embedding_gathers_without_full_weight_expansion(bits):
    device = torch.device("cuda")
    outputs, inputs, group_size = 128, 128, 32
    if bits == 1:
        codes = torch.randint(0, 2, (outputs, inputs), device=device).mul(2).sub(1)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            device=device,
        )
    groups = (outputs, inputs // group_size)
    layer = torch.nn.Module()
    tensors = {
        "weight_packed": pack_cubic_codes(codes.to(torch.int16), bits),
        "weight_scale": torch.randint(1, 64, groups, device=device).to(torch.int8),
        "weight_ab": torch.randint(0, 256, groups, device=device).to(torch.uint8),
        "weight_scale_global": torch.tensor([0.03125], device=device),
        "weight_a_global": torch.tensor([0.0625], device=device),
        "weight_b_global": torch.tensor([0.03125], device=device),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, False))
    layer.input_size_per_partition = inputs
    layer.output_size_per_partition = outputs
    layer.output_partition_sizes = [outputs]
    layer.params_dtype = torch.bfloat16
    method = CubicEmbeddingMethod(
        CubicScheme(
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
            reserved_code="binary" if bits == 1 else "zero",
        )
    )
    method.process_weights_after_loading(layer)
    ids = torch.tensor([[-1, 7, 3, 7], [1, 127, 0, outputs]], device=device)

    actual = method.embedding(layer, ids)
    expected = torch.zeros_like(actual)
    dequantized = method.dequantize(layer)
    valid = (ids >= 0) & (ids < outputs)
    expected[valid] = dequantized[ids[valid]]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cubic_embedding_uses_resident_a16_weight_without_redecoding():
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    layer = SimpleNamespace(
        weight_packed=weight,
        cubic_weight_is_expanded_a16=True,
    )
    method = CubicEmbeddingMethod(
        CubicScheme(num_bits=8, group_size=4, group_out=1)
    )
    ids = torch.tensor([[5, 1, 3], [0, 2, 4]])

    actual = method.embedding(layer, ids)

    torch.testing.assert_close(actual, weight[ids], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_curve2_embedding_zeros_ids_not_owned_by_tp_rank():
    device = torch.device("cuda")
    outputs, inputs, group_size = 8, 32, 32
    codes = torch.arange(outputs * inputs, device=device).reshape(outputs, inputs)
    codes = codes.remainder(255).sub(127).to(torch.int16)
    layer = torch.nn.Module()
    tensors = {
        "weight_packed": pack_cubic_codes(codes, 8),
        "weight_metadata": _encode_e5m9_curve2(
            torch.ones((outputs, 1), device=device),
            torch.zeros((outputs, 1), device=device, dtype=torch.int64),
        ),
        "weight_curve_a": torch.ones(
            (1, 4), device=device, dtype=torch.float16
        ),
        "weight_curve_b": torch.zeros(
            (1, 4), device=device, dtype=torch.float16
        ),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, False))
    layer.input_size_per_partition = inputs
    layer.output_size_per_partition = outputs
    layer.output_partition_sizes = [outputs]
    layer.params_dtype = torch.bfloat16
    method = CubicEmbeddingMethod(
        CubicScheme(
            num_bits=8,
            group_size=group_size,
            group_out=1,
            metadata_format=CUBIC_E5M9_CURVE2_METADATA_FORMAT,
        )
    )
    method.process_weights_after_loading(layer)
    ids = torch.tensor([[-1, 0, outputs - 1, outputs]], device=device)

    actual = method.embedding(layer, ids)
    expected = torch.zeros_like(actual)
    dequantized = method.dequantize(layer)
    expected[0, 1] = dequantized[0]
    expected[0, 2] = dequantized[-1]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _encode_e5m9_curve2(
    scale: torch.Tensor, curve_id: torch.Tensor
) -> torch.Tensor:
    scale_bits = scale.half().contiguous().view(torch.int16).int()
    scale_code = torch.bitwise_right_shift(scale_bits, 1)
    return (
        scale_code | torch.bitwise_left_shift(curve_id.int(), 14)
    ).to(torch.uint16)


def test_cubic_e5m9_curve2_metadata_decodes_logical_partitions():
    scale = torch.tensor([[0.5, 1.0], [2.0, 4.0], [8.0, 16.0]])
    curve_id = torch.tensor([[0, 1], [2, 3], [1, 0]])
    metadata = _encode_e5m9_curve2(scale, curve_id)
    curve_a = torch.tensor(
        [[1.0, 0.75, 0.5, 0.25], [1.0, 1.25, 1.5, 1.75]],
        dtype=torch.float16,
    )
    curve_b = torch.tensor(
        [[0.0, 0.25, 0.5, 0.75], [0.0, -0.25, -0.5, -0.75]],
        dtype=torch.float16,
    )

    actual_scale, actual_a, actual_b = decode_cubic_e5m9_curve2_metadata(
        metadata,
        curve_a,
        curve_b,
        output_partition_sizes=[2, 1],
        group_out=1,
    )

    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_a,
        torch.tensor(
            [[1.0, 0.75], [0.5, 0.25], [1.25, 1.0]], dtype=torch.float16
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_b,
        torch.tensor(
            [[0.0, 0.25], [0.5, 0.75], [-0.25, 0.0]], dtype=torch.float16
        ),
        rtol=0,
        atol=0,
    )


def test_cubic_config_accepts_e5m9_curve2_metadata():
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "curve2": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 6,
                        "group_size": [1, 128],
                        "metadata_format": CUBIC_E5M9_CURVE2_METADATA_FORMAT,
                        "scale_dtype": "packed_e5m9",
                        "param_dtype": "torch.float16",
                        "curve_count": 4,
                    },
                }
            },
        }
    )

    assert config.schemes[0][1].metadata_format == CUBIC_E5M9_CURVE2_METADATA_FORMAT


def test_cubic_e5m9_curve_tables_load_fused_logical_partitions(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    layer = torch.nn.Module()
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=6,
            group_size=32,
            group_out=1,
            metadata_format=CUBIC_E5M9_CURVE2_METADATA_FORMAT,
        )
    )
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[16, 16, 16],
        input_size=128,
        output_size=48,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args, **_kwargs: None,
    )

    assert layer.weight_metadata.shape == (48, 4)
    assert layer.weight_curve_a.shape == (3, 4)
    values = torch.tensor([[1.0, 0.75, 0.5, 0.25]], dtype=torch.float16)
    layer.weight_curve_a.load_qkv_weight(values, shard_id="v")

    torch.testing.assert_close(layer.weight_curve_a[2], values[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("dynamic_a8", (False, True))
@pytest.mark.parametrize("tokens", (1, 9))
def test_cubic_e5m9_curve2_kernel_matches_expanded_metadata(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    dynamic_a8: bool,
    tokens: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear,
        cubic_linear_curve2,
        cubic_linear_dynamic_a8,
        cubic_linear_dynamic_a8_curve2,
    )

    if tokens == 1:
        monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    device = torch.device("cuda")
    outputs = inputs = 128
    group_size = 32
    generator = torch.Generator(device=device).manual_seed(bits)
    if bits == 1:
        codes = torch.randint(
            0, 2, (outputs, inputs), generator=generator, device=device
        ) * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            generator=generator,
            device=device,
        )
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs, inputs // group_size)
    curve_id = torch.randint(
        0, 4, metadata_shape, generator=generator, device=device
    )
    scale = torch.full(metadata_shape, 0.03125, device=device)
    metadata = _encode_e5m9_curve2(scale, curve_id)
    curve_a = torch.tensor(
        [[1.0, 0.75, 0.5, 0.25]], dtype=torch.float16, device=device
    )
    curve_b = torch.tensor(
        [[0.0, 0.125, 0.25, 0.375]], dtype=torch.float16, device=device
    )
    _, a, b = decode_cubic_e5m9_curve2_metadata(
        metadata, curve_a, curve_b, [outputs], 1
    )
    global_index = torch.zeros(outputs, dtype=torch.uint8, device=device)
    x = torch.randn(tokens, inputs, generator=generator, device=device).bfloat16()

    if dynamic_a8:
        expected = cubic_linear_dynamic_a8(
            x,
            packed,
            scale,
            a,
            b,
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            input_size=inputs,
        )
        actual = cubic_linear_dynamic_a8_curve2(
            x,
            packed,
            metadata,
            curve_a,
            curve_b,
            global_index,
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            input_size=inputs,
        )
    else:
        expected = cubic_linear(
            x,
            packed,
            scale,
            a,
            b,
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            input_size=inputs,
        )
        actual = cubic_linear_curve2(
            x,
            packed,
            metadata,
            curve_a,
            curve_b,
            global_index,
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            input_size=inputs,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cubic_metadata_row_loader_uses_global_group_coordinates(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    parameter = CubicLinearMetadataParameter(
        data=torch.zeros(2, 1),
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=128,
        input_size_per_partition=256,
        input_group_size=512,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    parameter.tp_rank = 5
    loaded = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    parameter.load_row_parallel_weight(loaded)

    torch.testing.assert_close(parameter, loaded[:, 2:3])


@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_code_round_trip_with_tail(bits: int):
    if bits == 1:
        codes = torch.tensor([-1, 1], dtype=torch.int16)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.arange(-magnitude_max, magnitude_max + 1, dtype=torch.int16)
    codes = codes.repeat(math.ceil(137 / codes.numel()))[:137]

    packed = pack_cubic_codes(codes, bits)
    actual = unpack_cubic_codes(packed, bits, codes.numel())

    torch.testing.assert_close(actual, codes.to(torch.int8), rtol=0, atol=0)
    assert packed.shape == (math.ceil(137 * bits / 8),)


@pytest.mark.parametrize("bits", range(2, 9))
def test_cubic_reserved_code_maps_to_zero(bits: int):
    reserved_raw = 1 << (bits - 1)
    packed = torch.tensor([reserved_raw], dtype=torch.uint8)

    actual = unpack_cubic_codes(packed, bits, 1)

    assert actual.item() == 0


def test_cubic_one_bit_uses_native_binary_codes():
    codes = torch.tensor([-1, 1, -1, -1, 1, 1, -1, 1], dtype=torch.int8)

    packed = pack_cubic_codes(codes, 1)
    actual = unpack_cubic_codes(packed, 1, codes.numel())

    assert packed.tolist() == [0b10110010]
    torch.testing.assert_close(actual, codes, rtol=0, atol=0)
    with pytest.raises(ValueError, match="binary"):
        pack_cubic_codes(torch.tensor([0], dtype=torch.int8), 1)


def test_cubic_linear_can_materialize_operator_weight():
    weight = torch.linspace(-2, 2, 137).reshape(1, -1)
    quantized = quantize_cubic(weight, total_bits=1, group_size=128)
    layer = SimpleNamespace(
        weight_packed=quantized.packed,
        weight_scale=quantized.scale,
        weight_a=quantized.a,
        weight_b=quantized.b,
        input_size_per_partition=weight.shape[-1],
        params_dtype=torch.bfloat16,
    )
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=1,
            group_size=128,
            group_out=1,
            reserved_code="binary",
        )
    )

    actual = method.dequantize(layer)
    cached = method.dequantize(layer)
    expected = dequantize_cubic(
        quantized.packed,
        quantized.scale,
        quantized.a,
        quantized.b,
        total_bits=1,
        group_size=128,
        group_out=1,
        num_values=weight.shape[-1],
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cached is actual


def test_cubic_resident_a16_uses_batch_invariant_linear(monkeypatch):
    from vllm import envs

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    calls = []

    def batch_invariant_linear(x, weight, bias=None):
        calls.append((x, weight, bias))
        return x @ weight.T if bias is None else x @ weight.T + bias

    monkeypatch.setattr(
        "vllm.model_executor.layers.batch_invariant.linear_batch_invariant",
        batch_invariant_linear,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.cubic.apply_cubic_exact_marlin_weight",
        lambda *_args, **_kwargs: pytest.fail(
            "exact Marlin must not run in batch-invariant mode"
        ),
    )
    weight = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    layer = SimpleNamespace(
        weight_packed=weight,
        cubic_a16_packed_weight=None,
        cubic_weight_is_expanded_a16=True,
        cubic_runtime_representation="a16-dense",
        cubic_exact_marlin_weight=torch.empty(0),
        cubic_exact_marlin_levels=torch.empty(0),
        cubic_exact_marlin_workspace=torch.empty(0),
        cubic_exact_marlin_token_buckets=(1,),
        input_size_per_partition=3,
        output_size_per_partition=2,
    )
    method = CubicLinearMethod(CubicScheme(num_bits=4, group_size=32, group_out=1))
    x = torch.tensor([[1, 2, 3]], dtype=torch.bfloat16)

    actual = method.apply(layer, x)

    torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is weight
    assert calls[0][2] is None


def test_cubic_runtime_representation_fails_closed_on_flag_mismatch():
    layer = SimpleNamespace(
        cubic_runtime_representation="a16-dense",
        cubic_weight_is_expanded_a16=False,
    )
    method = CubicLinearMethod(CubicScheme(num_bits=4, group_size=32, group_out=1))

    with pytest.raises(RuntimeError, match="representation"):
        method.apply(layer, torch.ones(1, 32, dtype=torch.bfloat16))


def test_cubic_expanded_metadata_dispatches_only_selected_token_buckets(
    monkeypatch: pytest.MonkeyPatch,
):
    expanded_calls = []
    compact_calls = []

    def expanded(x, *_args, **_kwargs):
        expanded_calls.append(x.shape[0])
        return torch.ones(x.shape[0], 2, dtype=x.dtype)

    def compact(x, *_args, **_kwargs):
        compact_calls.append(x.shape[0])
        return torch.full((x.shape[0], 2), 2, dtype=x.dtype)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.cubic_kernels.cubic_linear",
        expanded,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.cubic_kernels.cubic_linear_compact",
        compact,
    )
    layer = SimpleNamespace(
        weight_packed=torch.zeros(2, 1, dtype=torch.uint8),
        weight_scale=torch.ones(2, 2, dtype=torch.int8),
        weight_ab=torch.zeros(2, 2, dtype=torch.uint8),
        weight_scale_global=torch.ones(1),
        weight_a_global=torch.ones(1),
        weight_b_global=torch.ones(1),
        weight_global_index=torch.zeros(2, dtype=torch.uint8),
        cubic_metadata_is_expanded=True,
        cubic_expanded_metadata_token_buckets=(1,),
        cubic_weight_scale_expanded=torch.ones(2, 2),
        cubic_weight_a_expanded=torch.ones(2, 2, dtype=torch.float16),
        cubic_weight_b_expanded=torch.zeros(2, 2, dtype=torch.float16),
        input_size_per_partition=4,
    )
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=4,
            group_size=2,
            group_out=1,
            metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
        )
    )

    small = method.apply(layer, torch.ones(1, 4, dtype=torch.bfloat16))
    large = method.apply(layer, torch.ones(16, 4, dtype=torch.bfloat16))

    assert expanded_calls == [1]
    assert compact_calls == [16]
    torch.testing.assert_close(small, torch.ones_like(small))
    torch.testing.assert_close(large, torch.full_like(large, 2))


def test_cubic_resident_a16_dispatches_expanded_metadata_before_dense(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    expanded_calls = []

    def expanded(x, *_args, **_kwargs):
        expanded_calls.append(x.shape[0])
        return torch.full((x.shape[0], 2), 3, dtype=x.dtype)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.cubic_kernels.cubic_linear",
        expanded,
    )
    dense = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    layer = SimpleNamespace(
        weight_packed=dense,
        cubic_a16_packed_weight=torch.zeros(2, 2, dtype=torch.uint8),
        cubic_a16_online_buckets=(1,),
        cubic_weight_is_expanded_a16=True,
        cubic_runtime_representation="a16-dense",
        cubic_metadata_is_expanded=True,
        cubic_weight_scale_expanded=torch.ones(2, 2),
        cubic_weight_a_expanded=torch.ones(2, 2, dtype=torch.float16),
        cubic_weight_b_expanded=torch.zeros(2, 2, dtype=torch.float16),
        input_size_per_partition=4,
    )
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=4,
            group_size=2,
            group_out=1,
            metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
        )
    )

    small = method.apply(layer, torch.ones(1, 4, dtype=torch.bfloat16))
    large_input = torch.ones(16, 4, dtype=torch.bfloat16)
    large = method.apply(layer, large_input)

    assert expanded_calls == [1]
    torch.testing.assert_close(small, torch.full_like(small, 3))
    torch.testing.assert_close(large, large_input @ dense.T, rtol=0, atol=0)


@pytest.mark.parametrize("bits", range(2, 9))
def test_cubic_exact_zero_endpoints_and_int_special_case(bits: int):
    magnitude_max = (1 << (bits - 1)) - 1
    codes = torch.tensor([-magnitude_max, 0, magnitude_max], dtype=torch.int8)
    packed = pack_cubic_codes(codes, bits)
    scale = torch.tensor([3.25], dtype=torch.float32)
    a = torch.tensor([1.0], dtype=torch.float16)
    b = torch.tensor([0.0], dtype=torch.float16)

    actual = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=bits,
        group_size=128,
        group_out=1,
        num_values=3,
    )

    expected = codes.float() * scale.item() / magnitude_max
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.tolist() == [-3.25, 0.0, 3.25]


def test_cubic_one_bit_dequantizes_without_widening():
    codes = torch.tensor([-1, 1, 1, -1], dtype=torch.int8)
    packed = pack_cubic_codes(codes, 1)
    scale = torch.tensor([3.25], dtype=torch.float32)
    a = torch.tensor([1.0], dtype=torch.float16)
    b = torch.tensor([0.0], dtype=torch.float16)

    actual = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=1,
        group_size=128,
        group_out=1,
        num_values=4,
    )

    torch.testing.assert_close(actual, codes.float() * 3.25, rtol=0, atol=0)


def test_cubic_two_dimensional_group_metadata_is_expanded_on_both_axes():
    codes = torch.tensor(
        [
            [-7, -3, 0, 7, -7, -3, 0, 7],
            [-7, -3, 0, 7, -7, -3, 0, 7],
            [-7, -3, 0, 7, -7, -3, 0, 7],
            [-7, -3, 0, 7, -7, -3, 0, 7],
        ],
        dtype=torch.int8,
    )
    packed = pack_cubic_codes(codes, 4)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    a = torch.ones_like(scale, dtype=torch.float16)
    b = torch.zeros_like(scale, dtype=torch.float16)

    actual = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=4,
        group_size=4,
        group_out=2,
        num_values=8,
    )

    expanded_scale = scale.repeat_interleave(2, 0).repeat_interleave(4, 1)
    expected = codes.float() * expanded_scale / 7
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cubic_config_accepts_legacy_and_two_dimensional_group_shapes():
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "legacy": {
                    "targets": ["re:.*legacy"],
                    "weights": {"num_bits": 5, "group_size": 512},
                },
                "two_dimensional": {
                    "targets": ["re:.*two_dimensional"],
                    "weights": {"num_bits": 5, "group_size": [128, 512]},
                },
                "output_only": {
                    "targets": ["re:.*output_only"],
                    "weights": {"num_bits": 5, "group_size": [512, 1]},
                },
            },
        }
    )

    assert config.schemes[0][1].group_shape == (1, 512)
    assert config.schemes[1][1].group_shape == (128, 512)
    assert config.schemes[2][1].group_shape == (512, 1)


def test_cubic_compact_metadata_decodes_per_logical_partition():
    scale_code = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.int8)
    a_code = torch.tensor([[1, -2], [3, -4], [5, -6]], dtype=torch.int8)
    b_code = torch.tensor([[-1, 2], [-3, 4], [-5, 6]], dtype=torch.int8)
    packed_ab = ((a_code & 0xF) | ((b_code & 0xF) << 4)).to(torch.uint8)

    scale, a, b = decode_cubic_compact_metadata(
        scale_code,
        packed_ab,
        torch.tensor([0.5, 0.25]),
        torch.tensor([0.125, 0.25]),
        torch.tensor([0.5, 0.75]),
        [2, 1],
        1,
    )

    torch.testing.assert_close(
        scale,
        torch.tensor([[1.0, 1.5], [2.0, 2.5], [1.5, 1.75]]),
    )
    torch.testing.assert_close(
        a,
        torch.tensor(
            [[1.125, 0.75], [1.375, 0.5], [2.25, -0.5]],
            dtype=torch.float16,
        ),
    )
    torch.testing.assert_close(
        b,
        torch.tensor(
            [[-0.5, 1.0], [-1.5, 2.0], [-3.75, 4.5]],
            dtype=torch.float16,
        ),
    )


def test_cubic_config_accepts_compact_metadata():
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "compact": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": [1, 32],
                        "scale_dtype": "torch.int8",
                        "param_dtype": "packed_int4",
                        "metadata_format": ("int8-scale-int4-ab-global-fp32"),
                    },
                }
            },
        }
    )

    scheme = config.schemes[0][1]
    assert scheme.metadata_format == "int8-scale-int4-ab-global-fp32"
    assert scheme.effective_bits == 4.5


def test_cubic_linear_keeps_compact_metadata_resident_after_loading():
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.uint8), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.tensor(
                [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]],
                dtype=torch.int8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_ab",
        torch.nn.Parameter(
            torch.full((3, 4), 0x1F, dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    for name, value in (
        ("scale", [0.5, 0.25]),
        ("a", [0.125, 0.25]),
        ("b", [0.5, 0.75]),
    ):
        layer.register_parameter(
            f"weight_{name}_global",
            torch.nn.Parameter(torch.tensor(value), requires_grad=False),
        )
    layer.input_size_per_partition = 8
    layer.output_size_per_partition = 3
    layer.output_partition_sizes = [2, 1]
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=4,
            group_size=2,
            group_out=1,
            metadata_format="int8-scale-int4-ab-global-fp32",
        )
    )

    method.process_weights_after_loading(layer)

    assert layer.weight_scale.dtype == torch.int8
    assert layer.weight_ab.dtype == torch.uint8
    assert not hasattr(layer, "weight_a")
    assert not hasattr(layer, "weight_b")
    assert layer.weight_global_index.tolist() == [0, 0, 1]
    scale, a, b = decode_cubic_compact_metadata(
        layer.weight_scale,
        layer.weight_ab,
        layer.weight_scale_global,
        layer.weight_a_global,
        layer.weight_b_global,
        layer.output_partition_sizes,
        method.scheme.group_out,
    )
    torch.testing.assert_close(scale[:, 0], torch.tensor([1.0, 3.0, 2.5]))
    torch.testing.assert_close(
        a[:, 0],
        torch.tensor([0.875, 0.875, 0.75], dtype=torch.float16),
    )
    torch.testing.assert_close(
        b[:, 0],
        torch.tensor([0.5, 0.5, 0.75], dtype=torch.float16),
    )


def test_cubic_dynamic_a8_keeps_compact_metadata_resident_after_loading():
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.uint8), False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(3, 4, dtype=torch.int8), False),
    )
    layer.register_parameter(
        "weight_ab",
        torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.uint8), False),
    )
    for name in ("scale", "a", "b"):
        layer.register_parameter(
            f"weight_{name}_global",
            torch.nn.Parameter(torch.ones(2), False),
        )
    layer.input_size_per_partition = 8
    layer.output_size_per_partition = 3
    layer.output_partition_sizes = [2, 1]
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=4,
            group_size=2,
            group_out=1,
            metadata_format="int8-scale-int4-ab-global-fp32",
        ),
        dynamic_a8=True,
    )

    method.process_weights_after_loading(layer)

    assert layer.weight_scale.dtype == torch.int8
    assert layer.weight_ab.dtype == torch.uint8
    assert not hasattr(layer, "weight_carrier")
    assert not hasattr(layer, "weight_a")
    assert not hasattr(layer, "weight_b")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_a8_replaces_packed_codes_with_runtime_carrier(bits: int):
    from vllm.model_executor.layers.quantization.cubic import (
        install_cubic_a8_carrier,
        materialize_cubic_a8_carrier,
    )

    device = torch.device("cuda")
    if bits == 1:
        codes = (
            torch.randint(0, 2, (4, 8), device=device, dtype=torch.int16)
            .mul_(2)
            .sub_(1)
        )
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (4, 8),
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    scale = torch.ones(4, 1, device=device, dtype=torch.float32)
    a = torch.full((4, 1), 0.5, device=device, dtype=torch.float16)
    b = torch.full((4, 1), 0.25, device=device, dtype=torch.float16)
    expected = materialize_cubic_a8_carrier(
        packed,
        a,
        b,
        num_bits=bits,
        group_size=8,
        input_size=8,
        group_out=1,
    )
    layer = torch.nn.Module()
    for name, value in (
        ("weight_packed", packed),
        ("weight_scale", scale),
        ("weight_a", a),
        ("weight_b", b),
    ):
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.input_size_per_partition = 8
    layer.output_size_per_partition = 4
    layer.output_partition_sizes = [4]
    layer.params_dtype = torch.bfloat16
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=bits,
            group_size=8,
            group_out=1,
            reserved_code="binary" if bits == 1 else "zero",
        ),
        dynamic_a8=True,
    )

    method.process_weights_after_loading(layer)
    install_cubic_a8_carrier(layer, method.scheme)

    assert layer.cubic_weight_packed_is_carrier
    assert layer.weight_packed.dtype == torch.int8
    assert layer.weight_packed.numel() == codes.numel()
    assert not hasattr(layer, "weight_carrier")
    torch.testing.assert_close(layer.weight_packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        method.dequantize(layer),
        expected.to(torch.float32).mul(1.0 / 127.0).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("group_out", [1, 32])
@pytest.mark.parametrize("dynamic_a8", [False, True])
def test_cubic_marlin_carrier_supports_all_bits_and_output_groups(
    bits: int,
    group_out: int,
    dynamic_a8: bool,
):
    from vllm.model_executor.layers.quantization.cubic import (
        apply_cubic_marlin_weight,
        materialize_cubic_a8_carrier,
        prepare_cubic_marlin_weight,
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    n, k, group_size = 64, 128, 32
    magnitude_max = 1 if bits == 1 else (1 << (bits - 1)) - 1
    codes = torch.randint(
        -magnitude_max,
        magnitude_max + 1,
        (n, k),
        device="cuda",
        dtype=torch.int16,
    )
    if bits == 1:
        codes = torch.where(codes < 0, -1, 1)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (n // group_out, k // group_size)
    scale = torch.rand(metadata_shape, device="cuda", dtype=torch.float32)
    a = torch.rand(metadata_shape, device="cuda", dtype=torch.float16)
    b = torch.rand(metadata_shape, device="cuda", dtype=torch.float16)
    carrier = materialize_cubic_a8_carrier(
        packed,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        input_size=k,
        group_out=group_out,
    )
    prepared = prepare_cubic_marlin_weight(
        carrier,
        scale,
        params_dtype=torch.float16,
        group_size=group_size,
        group_out=group_out,
        dynamic_a8=dynamic_a8,
    )
    assert prepared is not None

    x = torch.randn(3, k, device="cuda", dtype=torch.float16)
    actual = apply_cubic_marlin_weight(
        x,
        prepared,
        output_size=n,
        input_size=k,
        dynamic_a8=dynamic_a8,
    )
    expanded_scale = scale.repeat_interleave(group_out, dim=0).repeat_interleave(
        group_size, dim=1
    )
    weight = carrier.float() * expanded_scale[:, :k] / 127
    if dynamic_a8:
        quantized_x, x_scale = per_token_quant_int8(x)
        reference_x = quantized_x.float() * x_scale
    else:
        reference_x = x.float()
    reference = torch.nn.functional.linear(reference_x, weight).to(x.dtype)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_out", [1, 8])
def test_cubic_dynamic_a8_marlin_preserves_group_mapping_exactly(
    group_out: int,
):
    from vllm.model_executor.layers.quantization.cubic import (
        apply_cubic_marlin_weight,
        prepare_cubic_marlin_weight,
    )

    n, k, group_size = 64, 128, 32
    carrier = (
        torch.arange(n * k, device="cuda", dtype=torch.int32)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
        .reshape(n, k)
    )
    output_groups = n // group_out
    group_scales = torch.tensor(
        [127.0, 63.5, 31.75, 15.875], device="cuda", dtype=torch.float32
    )
    scale = group_scales.repeat(output_groups, 1)
    prepared = prepare_cubic_marlin_weight(
        carrier,
        scale,
        params_dtype=torch.float16,
        group_size=group_size,
        group_out=group_out,
        dynamic_a8=True,
    )
    assert prepared is not None

    output_scale = scale.repeat_interleave(group_out, dim=0)
    for input_index in (0, 31, 32, 63, 64, 95, 96, 127):
        x = torch.zeros((1, k), device="cuda", dtype=torch.float16)
        x[0, input_index] = 127
        actual = apply_cubic_marlin_weight(
            x,
            prepared,
            output_size=n,
            input_size=k,
            dynamic_a8=True,
        )
        expected = (
            carrier[:, input_index].float() * output_scale[:, input_index // group_size]
        ).to(actual.dtype)
        torch.testing.assert_close(actual.squeeze(0), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 19, 64, 65, 128, 256, 512])
@pytest.mark.parametrize("activation_group_size", [32, 128])
def test_cubic_dynamic_a8_marlin_supports_groupwise_activation_scales(
    tokens: int,
    activation_group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic import (
        apply_cubic_marlin_weight,
        prepare_cubic_marlin_weight,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _quantize_cubic_groupwise_a8,
    )

    torch.manual_seed(31)
    n, k, group_size = 128, 256, 32
    carrier = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand((n, k // group_size), device="cuda") + 0.25
    prepared = prepare_cubic_marlin_weight(
        carrier,
        scale,
        params_dtype=torch.bfloat16,
        group_size=group_size,
        group_out=1,
        dynamic_a8=True,
        input_group_size=activation_group_size,
    )
    assert prepared is not None
    assert prepared.scales.dtype == torch.float16

    x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
    actual = apply_cubic_marlin_weight(
        x,
        prepared,
        output_size=n,
        input_size=k,
        dynamic_a8=True,
    )
    assert actual.dtype == x.dtype
    activation = _quantize_cubic_groupwise_a8(x, activation_group_size)
    x_q, x_scale = activation.values, activation.scales
    expected = torch.zeros((tokens, n), device="cuda", dtype=torch.float32)
    for group in range(k // group_size):
        start = group * group_size
        stop = start + group_size
        activation_group = start // activation_group_size
        partial = x_q[:, start:stop].float() @ carrier[:, start:stop].float().T
        expected += (
            partial
            * x_scale[:, activation_group : activation_group + 1]
            * scale[:, group]
            / 127
        )
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0.02, atol=0.25
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 19, 65])
def test_cubic_dynamic_a8_marlin_refines_weight_scales_for_finer_activations(
    tokens: int,
):
    from vllm.model_executor.layers.quantization.cubic import (
        apply_cubic_marlin_weight,
        prepare_cubic_marlin_weight,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _quantize_cubic_groupwise_a8,
    )

    torch.manual_seed(37)
    n, k = 128, 256
    weight_group_size, activation_group_size = 128, 32
    carrier = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand((n, k // weight_group_size), device="cuda") + 0.25
    prepared = prepare_cubic_marlin_weight(
        carrier,
        scale,
        params_dtype=torch.bfloat16,
        group_size=weight_group_size,
        group_out=1,
        dynamic_a8=True,
        input_group_size=activation_group_size,
    )
    assert prepared is not None
    assert prepared.input_group_size == activation_group_size

    x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
    actual = apply_cubic_marlin_weight(
        x,
        prepared,
        output_size=n,
        input_size=k,
        dynamic_a8=True,
    )
    activation = _quantize_cubic_groupwise_a8(x, activation_group_size)
    x_q, x_scale = activation.values, activation.scales
    expected = torch.zeros((tokens, n), device="cuda", dtype=torch.float32)
    for start in range(0, k, activation_group_size):
        stop = start + activation_group_size
        activation_group = start // activation_group_size
        weight_group = start // weight_group_size
        partial = x_q[:, start:stop].float() @ carrier[:, start:stop].float().T
        expected += (
            partial
            * x_scale[:, activation_group : activation_group + 1]
            * scale[:, weight_group]
            / 127
        )
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0.02, atol=0.25
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_marlin_defers_groups_wider_than_tp_local_input():
    from vllm.model_executor.layers.quantization.cubic import (
        prepare_cubic_marlin_weight,
    )

    carrier = torch.zeros((128, 256), device="cuda", dtype=torch.int8)
    scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    prepared = prepare_cubic_marlin_weight(
        carrier,
        scale,
        params_dtype=torch.bfloat16,
        group_size=512,
        group_out=128,
        dynamic_a8=True,
    )

    assert prepared is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [2, 64])
def test_cubic_exact_marlin_preserves_w4_g32_codes_and_batch_rows(tokens: int):
    from vllm.model_executor.layers.quantization.cubic import (
        apply_cubic_exact_marlin_weight,
        dequantize_cubic,
        prepare_cubic_exact_marlin_weight,
    )

    torch.manual_seed(4)
    n = k = 128
    codes = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int8)
    packed = pack_cubic_codes(codes, 4)
    scale = torch.rand(n, k // 32, device="cuda", dtype=torch.float32) * 0.1
    coefficient_a = torch.rand(n, k // 32, device="cuda", dtype=torch.float16)
    coefficient_b = torch.rand(n, k // 32, device="cuda", dtype=torch.float16)
    prepared = prepare_cubic_exact_marlin_weight(
        packed,
        scale,
        coefficient_a,
        coefficient_b,
        params_dtype=torch.bfloat16,
        num_bits=4,
        group_size=32,
        group_out=1,
        input_size=k,
    )
    assert prepared is not None

    inputs = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    actual = apply_cubic_exact_marlin_weight(
        inputs,
        prepared,
        output_size=n,
        input_size=k,
    )
    weight = dequantize_cubic(
        packed,
        scale,
        coefficient_a,
        coefficient_b,
        total_bits=4,
        group_size=32,
        group_out=1,
        num_values=k,
        output_dtype=torch.bfloat16,
    )
    reference = torch.nn.functional.linear(inputs, weight)
    if tokens == 2:
        independent = torch.cat(
            [
                apply_cubic_exact_marlin_weight(
                    row[None],
                    prepared,
                    output_size=n,
                    input_size=k,
                )
                for row in inputs
            ]
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual, independent, rtol=0, atol=0)
    else:
        error = actual.float() - reference.float()
        reference_rms = reference.float().square().mean().sqrt()
        nrmse = error.square().mean().sqrt() / reference_rms
        assert nrmse.item() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("group_out", (1, 8))
def test_compact_dynamic_a8_uses_resident_carrier_without_marlin(
    bits: int, group_out: int
):
    from vllm.model_executor.layers.quantization.cubic import (
        install_cubic_a8_carrier,
    )

    device = torch.device("cuda")
    n, k, group_size = 16, 64, 128
    magnitude_max = 1 if bits == 1 else (1 << (bits - 1)) - 1
    codes = torch.randint(
        -magnitude_max,
        magnitude_max + 1,
        (n, k),
        device=device,
        dtype=torch.int16,
    )
    if bits == 1:
        codes = torch.where(codes < 0, -1, 1)
    metadata_shape = (n // group_out, 1)
    layer = torch.nn.Module()
    for name, value in (
        ("weight_packed", pack_cubic_codes(codes, bits)),
        (
            "weight_scale",
            torch.full(metadata_shape, 64, device=device, dtype=torch.int8),
        ),
        ("weight_ab", torch.zeros(metadata_shape, device=device, dtype=torch.uint8)),
        ("weight_scale_global", torch.tensor([1 / 64], device=device)),
        ("weight_a_global", torch.tensor([1 / 8], device=device)),
        ("weight_b_global", torch.tensor([1 / 8], device=device)),
    ):
        layer.register_parameter(name, torch.nn.Parameter(value, False))
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.output_partition_sizes = [n]
    layer.params_dtype = torch.float16
    method = CubicLinearMethod(
        CubicScheme(
            num_bits=bits,
            group_size=group_size,
            group_out=group_out,
            metadata_format=CUBIC_COMPACT_METADATA_FORMAT,
            reserved_code="binary" if bits == 1 else "zero",
        ),
        dynamic_a8=True,
    )
    method.process_weights_after_loading(layer)
    x = torch.randn(3, k, device=device, dtype=torch.float16)
    reference = method.apply(layer, x)

    install_cubic_a8_carrier(layer, method.scheme)

    assert layer.cubic_weight_packed_is_carrier
    assert not getattr(layer, "cubic_weight_packed_is_marlin", False)
    actual = method.apply(layer, x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_cubic_config_ignore_prefix_regex_excludes_subtree():
    from vllm.model_executor.layers.linear import LinearBase

    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "layer_zero": {
                    "targets": ["re:.*\\.layers\\.0\\.mlp\\.down_proj"],
                    "weights": {"num_bits": 8, "group_size": [128, 128]},
                },
            },
            "ignore": [r"re:^mtp\."],
        }
    )
    layer = object.__new__(LinearBase)

    assert config._scheme_for(layer, "mtp.layers.0.mlp.down_proj") is None
    assert config._scheme_for(layer, "model.layers.0.mlp.down_proj").num_bits == 8


def test_cubic_internal_group_interfaces_require_both_dimensions():
    import vllm.model_executor.layers.quantization.cubic_kernels as cubic_kernels

    source_path = Path(cubic_kernels.__file__)
    tree = ast.parse(source_path.read_text())
    definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    for definition in definitions.values():
        keyword_defaults = dict(
            zip(definition.args.kwonlyargs, definition.args.kw_defaults)
        )
        for argument, default in keyword_defaults.items():
            if argument.arg == "group_out":
                assert default is None, (
                    f"{definition.name} must require the normalized output-group "
                    "dimension"
                )

    grouped_kernels = {
        name
        for name, definition in definitions.items()
        if any(argument.arg == "GROUP_OUT" for argument in definition.args.args)
    }
    missing: list[tuple[int, str]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        target = call.func
        if isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Attribute) and target.attr == "fn":
            target = target.value
        if not isinstance(target, ast.Name) or target.id not in grouped_kernels:
            continue
        if not any(keyword.arg == "GROUP_OUT" for keyword in call.keywords):
            missing.append((call.lineno, target.id))

    assert missing == []


def test_cubic_levels_are_monotonic_and_normalized():
    for a, b in ((1.0, 0.0), (0.5, 0.25), (1.25, -0.25)):
        assert cubic_is_strictly_monotonic(a, b)
        levels = cubic_levels(8, a, b)
        assert levels[0].item() == 0
        assert levels[-1].item() == 1
        assert torch.all(levels[1:] > levels[:-1])


def test_cubic_quantization_persists_metadata_and_handles_tail():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(3, 259, generator=generator)

    quantized = quantize_cubic(weight, total_bits=4, group_size=128)
    actual = dequantize_cubic(
        quantized.packed,
        quantized.scale,
        quantized.a,
        quantized.b,
        total_bits=4,
        group_size=128,
        group_out=1,
        num_values=259,
    )

    assert quantized.packed.shape == (3, math.ceil(259 * 4 / 8))
    assert quantized.scale.shape == quantized.a.shape == quantized.b.shape == (3, 3)
    assert quantized.scale.dtype == torch.float32
    assert quantized.a.dtype == quantized.b.dtype == torch.float16
    assert actual.shape == weight.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        quantized.loss, (weight - actual).square().sum(dim=-1), rtol=0, atol=0
    )


def test_cubic_config_supports_custom_linear_and_moe_widths():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase

    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "linear": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 2, "group_size": 128},
                },
                "moe": {
                    "targets": ["FusedMoE"],
                    "weights": {"num_bits": 5, "group_size": 128},
                },
                "special": {
                    "targets": ["re:.*special_proj"],
                    "weights": {"num_bits": 8, "group_size": 128},
                },
            },
        }
    )

    assert config.schemes[0][1].effective_bits == 2.5
    assert (
        config._scheme_for(  # noqa: SLF001
            object.__new__(LinearBase), "model.regular_proj"
        ).num_bits
        == 2
    )
    assert (
        config._scheme_for(  # noqa: SLF001
            object.__new__(RoutedExperts), "model.experts"
        ).num_bits
        == 5
    )
    assert (
        config._scheme_for(  # noqa: SLF001
            object.__new__(LinearBase), "model.special_proj"
        ).num_bits
        == 8
    )
    assert not config.has_explicit_scheme("model.regular_proj")
    assert config.has_explicit_scheme("model.special_proj")
    binary = CubicScheme(
        num_bits=1,
        group_size=128,
        group_out=1,
        reserved_code="binary",
    )
    assert binary.effective_bits == 1.5
    with pytest.raises(ValueError, match="reserved_code"):
        CubicScheme(num_bits=1, group_size=128, group_out=1)


def test_cubic_regex_target_matches_top_level_submodel_prefix():
    from vllm.model_executor.layers.linear import LinearBase

    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "vision": {
                    "targets": [r"re:.*\.visual\.blocks\.\d+\.attn\.proj"],
                    "weights": {"num_bits": 8, "group_size": [1, 128]},
                },
            },
            "ignore": [r"re:.*\.visual\.blocks\.1\."],
        }
    )
    layer = object.__new__(LinearBase)

    for prefix in (
        "visual.blocks.0.attn.proj",
        "model.visual.blocks.0.attn.proj",
    ):
        assert config._scheme_for(layer, prefix).num_bits == 8  # noqa: SLF001
        assert config.has_explicit_scheme(prefix)

    assert config._scheme_for(  # noqa: SLF001
        layer, "audiovisual.blocks.0.attn.proj"
    ) is None
    assert config._scheme_for(  # noqa: SLF001
        layer, "visual.blocks.1.attn.proj"
    ) is None


def test_cubic_anchored_regex_overrides_broad_base_model_regex():
    from vllm.model_executor.layers.linear import LinearBase

    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "base": {
                    "targets": [r"re:.*\.layers\.\d+\.self_attn\.qkv_proj"],
                    "weights": {"num_bits": 6, "group_size": [1, 128]},
                },
                "mtp": {
                    "targets": [r"re:^mtp\.layers\.\d+\.self_attn\.qkv_proj"],
                    "weights": {"num_bits": 8, "group_size": [1, 128]},
                },
            },
        }
    )
    layer = object.__new__(LinearBase)

    assert (
        config._scheme_for(  # noqa: SLF001
            layer, "language_model.model.layers.0.self_attn.qkv_proj"
        ).num_bits
        == 6
    )
    assert (
        config._scheme_for(  # noqa: SLF001
            layer, "mtp.layers.0.self_attn.qkv_proj"
        ).num_bits
        == 8
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_decode_compacts_only_local_expert_routes():
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_compact_local_routes,
    )

    topk_ids = torch.tensor([[3, 32, 7, -1, 11, 2]], device="cuda", dtype=torch.int32)
    expert_map = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    expert_map[3] = 0
    expert_map[7] = 5
    expert_map[11] = 111

    sorted_ids, expert_ids, count = cubic_compact_local_routes(topk_ids, expert_map)

    assert count.item() == 3
    torch.testing.assert_close(
        sorted_ids[:3],
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_ids[:3],
        torch.tensor([0, 5, 111], device="cuda", dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_linear_kernel_reads_native_packed_width(bits: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import cubic_linear

    torch.manual_seed(7)
    device = torch.device("cuda")
    rows, outputs, inputs, group_size = 5, 70, 259, 128
    if bits == 1:
        codes = torch.randint(
            0,
            2,
            (outputs, inputs),
            device=device,
            dtype=torch.int16,
        )
        codes = codes * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    groups = math.ceil(inputs / group_size)
    scale = torch.rand(outputs, groups, device=device, dtype=torch.float32) + 0.2
    a = torch.ones(outputs, groups, device=device, dtype=torch.float16)
    b = torch.zeros_like(a)
    x = torch.randn(rows, inputs, device=device, dtype=torch.bfloat16)
    weight = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=bits,
        group_size=group_size,
        group_out=1,
        num_values=inputs,
        output_dtype=torch.bfloat16,
    )

    actual = cubic_linear(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        input_size=inputs,
    )

    assert packed.numel() == outputs * math.ceil(inputs * bits / 8)
    torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 128), (128, 1), (32, 64)),
)
@pytest.mark.parametrize("rows", (1, 16))
def test_cubic_a16_linear_supports_two_dimensional_groups(
    bits: int,
    group_out: int,
    group_size: int,
    rows: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import cubic_linear

    generator = torch.Generator(device="cuda").manual_seed(
        bits * 1000 + group_out * 10 + group_size + rows
    )
    device = torch.device("cuda")
    outputs = inputs = 128
    if bits == 1:
        codes = torch.randint(
            0,
            2,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
        codes = codes * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, inputs // group_size)
    scale = torch.rand(
        metadata_shape, generator=generator, device=device, dtype=torch.float32
    )
    scale = scale * 0.02 + 0.01
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(
        rows, inputs, generator=generator, device=device, dtype=torch.bfloat16
    )
    weight = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=bits,
        group_size=group_size,
        group_out=group_out,
        num_values=inputs,
        output_dtype=torch.bfloat16,
    )

    actual = cubic_linear(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
    )

    torch.testing.assert_close(actual, x @ weight.T, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 32), (1, 128), (128, 1), (32, 64)),
)
def test_cubic_compact_a16_supports_all_bits_and_group_layouts(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    group_out: int,
    group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_compact,
    )

    device = torch.device("cuda")
    outputs = inputs = 128
    generator = torch.Generator(device=device).manual_seed(
        bits * 1000 + group_out * 10 + group_size
    )
    if bits == 1:
        codes = torch.randint(
            0,
            2,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
        codes = codes * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, inputs // group_size)
    scale_code = torch.randint(
        1,
        32,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    a_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    b_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    packed_ab = ((b_code & 0xF) << 4 | (a_code & 0xF)).to(torch.uint8)
    scale_global = torch.tensor([0.001], device=device)
    a_global = torch.tensor([0.03125], device=device)
    b_global = torch.tensor([0.03125], device=device)
    global_index = torch.zeros(metadata_shape[0], device=device, dtype=torch.uint8)
    scale = scale_code.float() * scale_global
    a = (1.0 + a_code.float() * a_global).half()
    b = (b_code.float() * b_global).half()
    weight = dequantize_cubic(
        packed,
        scale,
        a,
        b,
        total_bits=bits,
        group_size=group_size,
        group_out=group_out,
        num_values=inputs,
        output_dtype=torch.bfloat16,
    )
    x = torch.randn(9, inputs, generator=generator, device=device, dtype=torch.bfloat16)

    low_m = cubic_linear_compact(
        x[:1],
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
    )
    torch.testing.assert_close(low_m, x[:1] @ weight.T, rtol=0, atol=0)

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    actual = cubic_linear_compact(
        x,
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
    )
    expanded = cubic_linear_compact(
        x,
        packed,
        scale,
        a,
        b,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
        _expanded_metadata=True,
    )
    tokenwise = torch.cat(
        [
            cubic_linear_compact(
                row,
                packed,
                scale_code,
                packed_ab,
                scale_global,
                a_global,
                b_global,
                global_index,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                input_size=inputs,
            )
            for row in x.split(1)
        ]
    )
    expanded_tokenwise = torch.cat(
        [
            cubic_linear_compact(
                row,
                packed,
                scale,
                a,
                b,
                a_global,
                b_global,
                global_index,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                input_size=inputs,
                _expanded_metadata=True,
            )
            for row in x.split(1)
        ]
    )

    torch.testing.assert_close(actual, x @ weight.T, rtol=0.03, atol=0.03)
    torch.testing.assert_close(expanded, x @ weight.T, rtol=0.03, atol=0.03)
    torch.testing.assert_close(actual, tokenwise, rtol=0, atol=0)
    torch.testing.assert_close(expanded, expanded_tokenwise, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 8), (1, 16), (1, 128), (128, 1), (32, 64)),
)
def test_cubic_compact_a8_matches_expanded_metadata_path(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    group_out: int,
    group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8,
        cubic_linear_dynamic_a8_compact,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    device = torch.device("cuda")
    outputs = inputs = 128
    generator = torch.Generator(device=device).manual_seed(
        bits * 1000 + group_out * 10 + group_size
    )
    if bits == 1:
        codes = torch.randint(
            0,
            2,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
        codes = codes * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, inputs // group_size)
    scale_code = torch.randint(
        1,
        32,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    a_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    b_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    packed_ab = ((b_code & 0xF) << 4 | (a_code & 0xF)).to(torch.uint8)
    scale_global = torch.tensor([0.001], device=device)
    a_global = torch.tensor([0.03125], device=device)
    b_global = torch.tensor([0.03125], device=device)
    global_index = torch.zeros(metadata_shape[0], device=device, dtype=torch.uint8)
    scale = scale_code.float() * scale_global
    a = (1.0 + a_code.float() * a_global).half()
    b = (b_code.float() * b_global).half()
    x = torch.randn(9, inputs, generator=generator, device=device, dtype=torch.bfloat16)

    expected = cubic_linear_dynamic_a8(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
    )
    actual = cubic_linear_dynamic_a8_compact(
        x,
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
    )
    expanded = cubic_linear_dynamic_a8_compact(
        x,
        packed,
        scale,
        a,
        b,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=inputs,
        _expanded_metadata=True,
    )
    tokenwise = torch.cat(
        [
            cubic_linear_dynamic_a8_compact(
                row,
                packed,
                scale_code,
                packed_ab,
                scale_global,
                a_global,
                b_global,
                global_index,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                input_size=inputs,
            )
            for row in x.split(1)
        ]
    )
    expanded_tokenwise = torch.cat(
        [
            cubic_linear_dynamic_a8_compact(
                row,
                packed,
                scale,
                a,
                b,
                a_global,
                b_global,
                global_index,
                num_bits=bits,
                group_size=group_size,
                group_out=group_out,
                input_size=inputs,
                _expanded_metadata=True,
            )
            for row in x.split(1)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(expanded, expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(actual, tokenwise, rtol=0, atol=0)
    torch.testing.assert_close(expanded, expanded_tokenwise, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", (32, 65, 128))
@pytest.mark.parametrize("group_size", (32, 64, 128))
def test_cubic_groupwise_a8_batched_groups_matches_reference(
    tokens: int, group_size: int
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_groupwise_quant_int8_kernel,
        _quantize_cubic_groupwise_a8,
    )

    generator = torch.Generator(device="cuda").manual_seed(
        20260820 + tokens + group_size
    )
    x = torch.randn(
        tokens,
        16 * group_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = _quantize_cubic_groupwise_a8(x, group_size)
    expected_values = torch.empty_like(x, dtype=torch.int8)
    expected_scales = torch.empty(
        tokens,
        x.shape[1] // group_size,
        device=x.device,
        dtype=torch.float32,
    )
    _cubic_groupwise_quant_int8_kernel[(tokens, expected_scales.shape[1])](
        x,
        expected_values,
        expected_scales,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        expected_values.stride(0),
        expected_values.stride(1),
        expected_scales.stride(0),
        expected_scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=min(max(group_size // 64, 1), 8),
        num_stages=1,
    )

    torch.testing.assert_close(actual.scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(actual.values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_compact_carrier_preserves_online_a8_codes(bits: int):
    from vllm.model_executor.layers.quantization.cubic import (
        decode_cubic_compact_metadata,
        materialize_cubic_a8_carrier,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8_compact,
        cubic_linear_dynamic_a8_precomputed,
        materialize_cubic_compact_a8_carrier,
    )

    device = torch.device("cuda")
    outputs, inputs, group_size = 96, 128, 32
    generator = torch.Generator(device=device).manual_seed(9100 + bits)
    if bits == 1:
        codes = torch.randint(
            0,
            2,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
        codes = codes * 2 - 1
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        codes = torch.randint(
            -magnitude_max,
            magnitude_max + 1,
            (outputs, inputs),
            generator=generator,
            device=device,
            dtype=torch.int16,
        )
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs, inputs // group_size)
    scale_code = torch.randint(
        1,
        32,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    packed_ab = torch.randint(
        0,
        256,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.uint8,
    )
    scale_global = torch.tensor([0.001], device=device)
    a_global = torch.tensor([0.0313], device=device)
    b_global = torch.tensor([0.0271], device=device)
    global_index = torch.zeros(outputs, device=device, dtype=torch.uint8)
    x = torch.randn(
        1,
        inputs,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    carrier = materialize_cubic_compact_a8_carrier(
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
        e5m9_curve2_metadata=False,
    )
    _, decoded_a, decoded_b = decode_cubic_compact_metadata(
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        [outputs],
        1,
    )
    canonical_carrier = materialize_cubic_a8_carrier(
        packed,
        decoded_a,
        decoded_b,
        num_bits=bits,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )
    torch.testing.assert_close(carrier, canonical_carrier, rtol=0, atol=0)
    expected = cubic_linear_dynamic_a8_compact(
        x,
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        input_size=inputs,
    )
    actual = cubic_linear_dynamic_a8_precomputed(
        x,
        carrier.T.contiguous().T,
        scale_code.float() * scale_global,
        a_global,
        b_global,
        num_bits=bits,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_e5m9_carrier_preserves_online_a8_codes(
    monkeypatch: pytest.MonkeyPatch, bits: int
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8_curve2,
        cubic_linear_dynamic_a8_precomputed,
        materialize_cubic_compact_a8_carrier,
    )

    monkeypatch.setenv("VLLM_CUBIC_DEBUG_DISABLE_PRECOMPUTED_DP4A", "1")
    device = torch.device("cuda")
    outputs, inputs, group_size = 96, 128, 32
    generator = torch.Generator(device=device).manual_seed(9200 + bits)
    magnitude_max = (1 << (bits - 1)) - 1
    codes = torch.randint(
        -max(1, magnitude_max),
        max(1, magnitude_max) + 1,
        (outputs, inputs),
        generator=generator,
        device=device,
        dtype=torch.int16,
    )
    if bits == 1:
        codes = torch.where(codes >= 0, 1, -1)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs, inputs // group_size)
    curve_id = torch.randint(
        0, 4, metadata_shape, generator=generator, device=device
    )
    scale = torch.full(metadata_shape, 0.03125, device=device)
    metadata = _encode_e5m9_curve2(scale, curve_id)
    curve_a = torch.tensor(
        [[1.0, 0.75, 0.5, 0.25]], dtype=torch.float16, device=device
    )
    curve_b = torch.tensor(
        [[0.0, 0.125, 0.25, 0.375]], dtype=torch.float16, device=device
    )
    global_index = torch.zeros(outputs, device=device, dtype=torch.uint8)
    x = torch.randn(
        1, inputs, generator=generator, device=device, dtype=torch.bfloat16
    )
    carrier = materialize_cubic_compact_a8_carrier(
        packed,
        metadata,
        curve_a,
        curve_b,
        curve_a,
        curve_b,
        global_index,
        num_bits=bits,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
        e5m9_curve2_metadata=True,
    )
    expected = cubic_linear_dynamic_a8_curve2(
        x,
        packed,
        metadata,
        curve_a,
        curve_b,
        global_index,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        input_size=inputs,
    )
    actual = cubic_linear_dynamic_a8_precomputed(
        x,
        carrier.T.contiguous().T,
        scale,
        curve_a,
        curve_b,
        num_bits=bits,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", (1, 7))
def test_cubic_w5_curve2_pair_lut_matches_scalar_decode(tokens: int):
    from vllm.model_executor.layers.quantization.cubic import (
        cubic_w5_curve2_pair_lut,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8_curve2,
        cubic_linear_dynamic_a8_w5_curve2_pair_lut,
    )

    device = torch.device("cuda")
    outputs, inputs, group_size = 256, 512, 128
    generator = torch.Generator(device=device).manual_seed(20260818 + tokens)
    codes = torch.randint(
        -15,
        16,
        (outputs, inputs),
        generator=generator,
        device=device,
        dtype=torch.int16,
    )
    packed = pack_cubic_codes(codes, 5)
    scale = (
        torch.rand(
            outputs,
            inputs // group_size,
            generator=generator,
            device=device,
        )
        * 0.02
        + 0.001
    ).half()
    curve_id = torch.randint(
        0,
        4,
        scale.shape,
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    metadata = _encode_e5m9_curve2(scale, curve_id)
    curve_a = torch.tensor(
        [[1.0, 0.8, 1.2, 0.6]], device=device, dtype=torch.float16
    )
    curve_b = torch.tensor(
        [[0.0, 0.1, -0.1, 0.3]], device=device, dtype=torch.float16
    )
    global_index = torch.zeros(outputs, device=device, dtype=torch.uint8)
    pair_lut = cubic_w5_curve2_pair_lut(curve_a, curve_b)
    x = torch.randn(
        tokens, inputs, generator=generator, device=device, dtype=torch.bfloat16
    )

    expected = cubic_linear_dynamic_a8_curve2(
        x,
        packed,
        metadata,
        curve_a,
        curve_b,
        global_index,
        num_bits=5,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )
    actual = cubic_linear_dynamic_a8_w5_curve2_pair_lut(
        x,
        packed,
        metadata,
        global_index,
        pair_lut,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", (1, 7))
def test_cubic_w5_compact_dp4a_matches_scalar_decode(tokens: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8_compact,
        cubic_linear_dynamic_a8_w5_compact_dp4a,
    )

    device = torch.device("cuda")
    outputs, inputs, group_size = 256, 512, 128
    generator = torch.Generator(device=device).manual_seed(20260819 + tokens)
    codes = torch.randint(
        -15,
        16,
        (outputs, inputs),
        generator=generator,
        device=device,
        dtype=torch.int16,
    )
    packed = pack_cubic_codes(codes, 5)
    metadata_shape = (outputs, inputs // group_size)
    scale_code = torch.randint(
        1,
        64,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    a_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    b_code = torch.randint(
        -7,
        8,
        metadata_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    packed_ab = ((b_code & 0xF) << 4 | (a_code & 0xF)).to(torch.uint8)
    scale_global = torch.tensor([0.001], device=device)
    a_global = torch.tensor([0.03125], device=device)
    b_global = torch.tensor([0.03125], device=device)
    global_index = torch.zeros(outputs, device=device, dtype=torch.uint8)
    x = torch.randn(
        tokens, inputs, generator=generator, device=device, dtype=torch.bfloat16
    )

    expected = cubic_linear_dynamic_a8_compact(
        x,
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        num_bits=5,
        group_size=group_size,
        group_out=1,
        input_size=inputs,
    )
    actual = cubic_linear_dynamic_a8_w5_compact_dp4a(
        x,
        packed,
        scale_code,
        packed_ab,
        scale_global,
        a_global,
        b_global,
        global_index,
        group_size=group_size,
        input_size=inputs,
        group_out=1,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("tokens", (1, 7, 8))
def test_cubic_fused_moe_kernel_supports_kimi_situ(bits: int, tokens: int):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_fused_moe,
    )

    torch.manual_seed(11)
    device = torch.device("cuda")
    experts, top_k = 4, 2
    if bits <= 4:
        hidden, intermediate, group_size = 512, 512, 512
    else:
        hidden, intermediate, group_size = 130, 70, 32

    def make_weight(
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        if bits == 1:
            codes = torch.randint(0, 2, shape, device=device, dtype=torch.int16)
            codes = codes * 2 - 1
        else:
            magnitude_max = (1 << (bits - 1)) - 1
            codes = torch.randint(
                -magnitude_max,
                magnitude_max + 1,
                shape,
                device=device,
                dtype=torch.int16,
            )
        packed = pack_cubic_codes(codes, bits)
        groups = math.ceil(shape[-1] / group_size)
        scale = (
            torch.rand(*shape[:-1], groups, device=device, dtype=torch.float32) * 0.02
            + 0.01
        )
        a_value = 0.5 if bits >= 3 else 1.0
        b_value = 0.25 if bits >= 3 else 0.0
        a = torch.full(
            (*shape[:-1], groups),
            a_value,
            device=device,
            dtype=torch.float16,
        )
        b = torch.full_like(a, b_value)
        reference = dequantize_cubic(
            packed,
            scale,
            a,
            b,
            total_bits=bits,
            group_size=group_size,
            group_out=1,
            num_values=shape[-1],
            output_dtype=torch.bfloat16,
        )
        return packed, scale, a, b, reference

    p1, s1, a1, b1, w1 = make_weight((experts, 2 * intermediate, hidden))
    p2, s2, a2, b2, w2 = make_weight((experts, hidden, intermediate))
    expert_map = None
    if tokens == 1:
        local_experts = torch.tensor([0, 2], device=device)
        expert_map = torch.tensor([0, -1, 1, -1], device=device, dtype=torch.int32)
        p1, s1, a1, b1 = (
            tensor.index_select(0, local_experts) for tensor in (p1, s1, a1, b1)
        )
        p2, s2, a2, b2 = (
            tensor.index_select(0, local_experts) for tensor in (p2, s2, a2, b2)
        )
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    token_indices = torch.arange(tokens, device=device)
    topk_ids = torch.stack(
        (token_indices % experts, (token_indices + 1) % experts), dim=1
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(tokens, top_k, device=device), dim=1)

    actual = cubic_fused_moe(
        x,
        p1,
        p2,
        s1,
        s2,
        a1,
        b1,
        a2,
        b2,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SITU,
        apply_router_weight_on_input=False,
        global_num_experts=experts,
        expert_map=expert_map,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    expected_parts = torch.zeros(
        tokens, top_k, hidden, device=device, dtype=torch.float32
    )
    for token in range(tokens):
        for route in range(top_k):
            expert = topk_ids[token, route]
            if expert_map is not None and expert_map[expert] < 0:
                continue
            gate_up = x[token : token + 1] @ w1[expert].T
            gate = gate_up[:, :intermediate].float()
            up = gate_up[:, intermediate:].float()
            activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            activated *= 25.0 * torch.tanh(up / 25.0)
            expert_output = activated.to(torch.bfloat16) @ w2[expert].T
            expected_parts[token, route] = (
                expert_output.float() * topk_weights[token, route]
            )
    expected = expected_parts.sum(dim=1)
    relative_rmse = (actual.float() - expected).square().mean().sqrt()
    relative_rmse /= expected.square().mean().sqrt()

    assert torch.isfinite(actual).all()
    assert relative_rmse.item() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 128), (128, 1), (32, 64), (32, 128)),
)
@pytest.mark.parametrize("tokens", (1, 16))
def test_cubic_a16_moe_supports_two_dimensional_groups(
    bits: int,
    group_out: int,
    group_size: int,
    tokens: int,
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import cubic_fused_moe

    device = torch.device("cuda")
    experts, top_k = 2, 2
    hidden = intermediate = 128
    generator = torch.Generator(device=device).manual_seed(
        bits * 1000 + group_out * 10 + group_size + tokens
    )

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        if bits == 1:
            codes = torch.randint(
                0,
                2,
                shape,
                generator=generator,
                device=device,
                dtype=torch.int16,
            )
            codes = codes * 2 - 1
        else:
            magnitude_max = (1 << (bits - 1)) - 1
            codes = torch.randint(
                -magnitude_max,
                magnitude_max + 1,
                shape,
                generator=generator,
                device=device,
                dtype=torch.int16,
            )
        packed = pack_cubic_codes(codes, bits)
        metadata_shape = (
            shape[0],
            shape[1] // group_out,
            shape[2] // group_size,
        )
        scale = torch.rand(
            metadata_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        scale = scale * 0.02 + 0.01
        a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
        b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
        reference = dequantize_cubic(
            packed,
            scale,
            a,
            b,
            total_bits=bits,
            group_size=group_size,
            group_out=group_out,
            num_values=shape[-1],
            output_dtype=torch.bfloat16,
        )
        return packed, scale, a, b, reference

    p1, s1, a1, b1, w1 = make_weight((experts, 2 * intermediate, hidden))
    p2, s2, a2, b2, w2 = make_weight((experts, hidden, intermediate))
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.tensor([[0, 1]], device=device, dtype=torch.int32).repeat(
        tokens, 1
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=1
    )

    actual = cubic_fused_moe(
        x,
        p1,
        p2,
        s1,
        s2,
        a1,
        b1,
        a2,
        b2,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SITU,
        apply_router_weight_on_input=False,
        global_num_experts=experts,
        expert_map=None,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    expected = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for token in range(tokens):
        for route in range(top_k):
            expert = topk_ids[token, route]
            gate_up = x[token : token + 1] @ w1[expert].T
            gate = gate_up[:, :intermediate].float()
            up = gate_up[:, intermediate:].float()
            activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            activated *= 25.0 * torch.tanh(up / 25.0)
            expert_output = activated.to(torch.bfloat16) @ w2[expert].T
            expected[token] += expert_output[0].float() * topk_weights[token, route]

    relative_rmse = (actual.float() - expected).square().mean().sqrt()
    relative_rmse /= expected.square().mean().sqrt()
    assert torch.isfinite(actual).all()
    assert relative_rmse.item() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", (2, 3))
def test_cubic_decode_w2_route_sum_fusion_is_exact(bits: int):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _launch_cubic_moe_gemv,
    )

    torch.manual_seed(29)
    device = torch.device("cuda")
    local_experts = 2
    hidden = intermediate = group_size = 256
    top_k = 4
    packed = torch.randint(
        0,
        256,
        (
            local_experts,
            hidden,
            math.ceil(intermediate * bits / 8),
        ),
        device=device,
        dtype=torch.uint8,
    )
    scale = torch.rand(
        local_experts,
        hidden,
        intermediate // group_size,
        device=device,
        dtype=torch.float32,
    )
    a = torch.full_like(scale, 0.75, dtype=torch.float16)
    b = torch.full_like(scale, 0.125, dtype=torch.float16)
    inputs = torch.randn(
        top_k,
        intermediate,
        device=device,
        dtype=torch.bfloat16,
    )
    topk_weights = torch.softmax(
        torch.randn(1, top_k, device=device),
        dim=-1,
    )
    topk_ids = torch.tensor(
        [[0, 2, 1, 3]],
        device=device,
        dtype=torch.int32,
    )
    expert_map = torch.tensor(
        [0, 1, -1, -1],
        device=device,
        dtype=torch.int32,
    )
    sorted_ids = torch.tensor(
        [0, 2, 0, 0],
        device=device,
        dtype=torch.int32,
    )
    expert_ids = torch.tensor(
        [0, 1, 0, 0],
        device=device,
        dtype=torch.int32,
    )
    route_count = torch.tensor([2], device=device, dtype=torch.int32)
    route_outputs = torch.zeros(
        1,
        top_k,
        hidden,
        device=device,
        dtype=torch.bfloat16,
    )
    expected = torch.empty(
        1,
        hidden,
        device=device,
        dtype=torch.bfloat16,
    )
    actual = torch.empty_like(expected)
    launch_args = (
        inputs,
        packed,
        scale,
        a,
        b,
    )
    launch_routing = (
        topk_weights,
        sorted_ids,
        expert_ids,
        route_count,
    )

    _launch_cubic_moe_gemv(
        *launch_args,
        route_outputs,
        *launch_routing,
        logical_k=intermediate,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        top_k=1,
        multiply_routed_weight=True,
        sum_routes=False,
    )
    ops.moe_sum(route_outputs, expected, topk_ids, expert_map)
    _launch_cubic_moe_gemv(
        *launch_args,
        actual,
        *launch_routing,
        logical_k=intermediate,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        top_k=1,
        multiply_routed_weight=True,
        sum_routes=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cubic_moe_sum_policy_uses_nearest_calibrated_token_bucket(monkeypatch):
    from vllm.model_executor.layers.quantization import cubic_kernels

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_MOE_SUM_TACTICS",
        {
            (3, 5120, 16, 32, True): True,
            (3, 5120, 16, 128, True): False,
        },
    )
    expert_map = torch.empty(0, dtype=torch.int32)

    assert cubic_kernels._cubic_moe_use_torch_sum(5120, 16, 40, expert_map)
    assert not cubic_kernels._cubic_moe_use_torch_sum(5120, 16, 96, expert_map)


def test_cubic_moe_sum_policy_falls_back_to_safe_zeroed_routes(monkeypatch):
    from vllm.model_executor.layers.quantization import cubic_kernels

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(cubic_kernels, "_CUBIC_MOE_SUM_TACTICS", {})

    assert cubic_kernels._cubic_moe_use_torch_sum(3584, 16, 1, None)
