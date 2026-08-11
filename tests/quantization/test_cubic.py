# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.cubic import (
    CUBIC_FORMAT,
    CubicConfig,
    CubicLinearMetadataParameter,
    CubicLinearMethod,
    CubicScheme,
    cubic_is_strictly_monotonic,
    cubic_levels,
    dequantize_cubic,
    pack_cubic_codes,
    quantize_cubic,
    unpack_cubic_codes,
)


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
    method = CubicLinearMethod(CubicScheme(1, 128, reserved_code="binary"))

    actual = method.dequantize(layer)
    cached = method.dequantize(layer)
    expected = dequantize_cubic(
        quantized.packed,
        quantized.scale,
        quantized.a,
        quantized.b,
        total_bits=1,
        group_size=128,
        num_values=weight.shape[-1],
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cached is actual


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
    binary = CubicScheme(1, 128, reserved_code="binary")
    assert binary.effective_bits == 1.5
    with pytest.raises(ValueError, match="reserved_code"):
        CubicScheme(1, 128)


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
        input_size=inputs,
    )

    assert packed.numel() == outputs * math.ceil(inputs * bits / 8)
    torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", (4, 5, 6, 8))
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
@pytest.mark.parametrize("bits", (4, 5, 6, 8))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 128), (128, 1), (32, 64)),
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
        top_k=1,
        multiply_routed_weight=True,
        sum_routes=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
