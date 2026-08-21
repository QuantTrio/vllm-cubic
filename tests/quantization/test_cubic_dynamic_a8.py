# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.cubic import (
    cubic_carrier_levels,
    dequantize_cubic_carrier,
    pack_cubic_codes,
    unpack_cubic_codes,
)
from vllm.model_executor.layers.quantization.cubic_policy import (
    cubic_dynamic_a8_group_size,
)

GROUP_SIZES = (32, 64, 128, 256, 512)
ONLINE_MLP_REMOVED = pytest.mark.skip(
    reason="Online MLP A8 was rolled back after failing performance gates"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_out", (1, 32))
def test_cubic_w2_situ_calibration_uses_normalized_group_shape(group_out: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _CUBIC_W2_A8_SITU_TACTICS,
        calibrate_cubic_w2_a8_situ,
    )

    calibrate_cubic_w2_a8_situ(
        n=64,
        k=128,
        group_out=group_out,
        group_size=128,
        top_k=1,
        local_experts=1,
        route_ctas_values=(1,),
    )

    device = torch.accelerator.current_device_index()
    assert (device, 64, 128, group_out, 128, 1, 1) in (_CUBIC_W2_A8_SITU_TACTICS)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(None, False), ("0", False), ("1", True)],
)
@ONLINE_MLP_REMOVED
def test_cubic_dynamic_a8_online_environment_contract(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    expected: bool,
) -> None:
    from vllm import envs

    name = "VLLM_CUBIC_DYNAMIC_A8_ONLINE"
    if raw_value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw_value)

    assert envs.environment_variables[name]() is expected


def _make_codes(shape: tuple[int, ...], bits: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1000 + bits)
    if bits == 1:
        codes = torch.randint(0, 2, shape, generator=generator, dtype=torch.int16)
        return codes * 2 - 1
    magnitude_max = (1 << (bits - 1)) - 1
    return torch.randint(
        -magnitude_max,
        magnitude_max + 1,
        shape,
        generator=generator,
        dtype=torch.int16,
    )


def test_cubic_a8_moe_grouping_keeps_small_input_groups_singleton() -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_a8_moe_grouping,
    )

    assert (
        _cubic_a8_moe_grouping(
            num_bits=8,
            hidden_size=3072,
            intermediate_size=1024,
            group_size=128,
            group_out=1,
            local_experts=32,
            num_tokens=1,
            fallback=8,
        )
        == 1
    )


@pytest.mark.parametrize(
    (
        "bits",
        "group_out",
        "group_size",
        "precomputed_3bit_levels",
        "fp16_curve",
        "expected",
    ),
    (
        (1, 1, 512, False, False, (1,)),
        (2, 1, 512, False, False, (1, 2, 4)),
        (2, 128, 512, False, False, (1,)),
        (3, 1, 256, True, False, (1, 2)),
        (3, 128, 256, True, False, (1,)),
        (3, 1, 256, False, False, (1,)),
        (4, 128, 512, False, True, (1, 2, 4, 8)),
        (8, 128, 128, False, True, (1,)),
        (5, 512, 1, False, True, (1,)),
    ),
)
def test_cubic_a8_moe_grouping_candidates_match_exact_consumers(
    bits: int,
    group_out: int,
    group_size: int,
    precomputed_3bit_levels: bool,
    fp16_curve: bool,
    expected: tuple[int, ...],
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_a8_moe_grouping_candidates,
    )

    assert (
        _cubic_a8_moe_grouping_candidates(
            num_bits=bits,
            group_size=group_size,
            group_out=group_out,
            precomputed_3bit_levels=precomputed_3bit_levels,
            fp16_curve=fp16_curve,
        )
        == expected
    )


@pytest.mark.parametrize(("tokens", "tail_routes"), ((1, 5), (64, 150)))
def test_cubic_ep_route_scenarios_include_cross_rank_tail(
    tokens: int,
    tail_routes: int,
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_ep_route_scenarios,
    )

    global_experts = 896
    local_experts = 112
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32)
    expert_map[:local_experts] = torch.arange(local_experts, dtype=torch.int32)
    topk_ids = torch.arange(tokens * 16, dtype=torch.int32).view(tokens, 16)
    topk_ids.remainder_(global_experts - local_experts).add_(local_experts)

    scenarios = _cubic_ep_route_scenarios(
        topk_ids,
        expert_map=expert_map,
        global_num_experts=global_experts,
    )

    assert [name for name, _ in scenarios] == [
        "nominal",
        f"ep_tail_local={tail_routes}",
    ]
    assert torch.equal(scenarios[0][1], topk_ids)
    assert int((expert_map[scenarios[1][1].long()] >= 0).sum()) == tail_routes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_candidate_benchmark_supports_cuda_graph_replay() -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _benchmark_cubic_candidate,
    )

    value = torch.ones(256, device="cuda")

    def launch() -> torch.Tensor:
        return value + 1

    score = _benchmark_cubic_candidate(
        launch,
        cuda_graph_replay=True,
        warmup=1,
        rep=1,
    )

    assert score > 0


def test_cubic_a8_moe_grouping_rejects_incompatible_cached_tactic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic_policy import (
        cubic_token_bucket,
    )

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    key = (0, 3, 3072, 1024, 512, 128, 32, cubic_token_bucket(64))
    monkeypatch.setitem(cubic_kernels._CUBIC_A8_MOE_GROUPING_TACTICS, key, 2)

    assert (
        cubic_kernels._cubic_a8_moe_grouping(
            num_bits=3,
            hidden_size=3072,
            intermediate_size=1024,
            group_size=512,
            group_out=128,
            local_experts=32,
            num_tokens=64,
            fallback=2,
            precomputed_3bit_levels=True,
        )
        == 1
    )


def test_expected_cubic_moe_route_blocks_accounts_for_expert_padding() -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _expected_cubic_moe_route_blocks,
    )

    assert _expected_cubic_moe_route_blocks(48, 32, 1) == 48
    assert _expected_cubic_moe_route_blocks(48, 32, 2) == pytest.approx(
        31.6017, abs=1e-4
    )
    assert _expected_cubic_moe_route_blocks(61.2, 32, 2) == pytest.approx(
        38.4255, abs=1e-4
    )


def test_cubic_a8_moe_grouping_prefers_calibrated_tactic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 7)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_A8_MOE_GROUPING_TACTICS",
        {(7, 4, 4096, 2048, 512, 128, 32, 64): 1},
    )

    assert (
        cubic_kernels._cubic_a8_moe_grouping(
            num_bits=4,
            hidden_size=4096,
            intermediate_size=2048,
            group_size=512,
            group_out=128,
            local_experts=32,
            num_tokens=64,
            fallback=8,
        )
        == 1
    )


def test_cubic_linear_tile_reuses_nearest_calibrated_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 7)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_LINEAR_TILE_TACTICS",
        {
            (7, True, False, 8, 4096, 1024, 512, 128, 64): (16, 64, 4, 3),
            (7, True, False, 8, 4096, 1024, 512, 128, 256): (32, 64, 8, 3),
        },
    )

    assert cubic_kernels._cubic_linear_tile(
        dynamic_a8=True,
        precomputed_carrier=False,
        num_bits=8,
        n=4096,
        k=1024,
        group_size=512,
        group_out=128,
        m=96,
        fallback=(8, 32, 4, 2),
    ) == (16, 64, 4, 3)


def test_cubic_linear_tactics_distinguish_packed_and_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 7)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_LINEAR_TILE_TACTICS",
        {
            (7, True, False, 4, 4096, 4096, 32, 1, 1): (1, 8, 4, 1),
            (7, True, True, 4, 4096, 4096, 32, 1, 1): (1, 64, 8, 3),
        },
    )
    common = {
        "dynamic_a8": True,
        "num_bits": 4,
        "n": 4096,
        "k": 4096,
        "group_size": 32,
        "group_out": 1,
        "m": 1,
        "fallback": (1, 16, 4, 1),
    }

    assert cubic_kernels._cubic_linear_tile(**common, precomputed_carrier=False) == (
        1,
        8,
        4,
        1,
    )
    assert cubic_kernels._cubic_linear_tile(**common, precomputed_carrier=True) == (
        1,
        64,
        8,
        3,
    )


def _reference_carriers(
    codes: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    if bits == 1:
        return codes.to(torch.int8) * 127
    group_indices = torch.arange(codes.shape[-1], device=codes.device) // group_size
    a_values = a[..., group_indices].to(torch.float32)
    b_values = b[..., group_indices].to(torch.float32)
    magnitude_max = (1 << (bits - 1)) - 1
    code_f32 = codes.to(torch.float32)
    t = code_f32.abs() / magnitude_max
    q = t * (a_values + t * (b_values + t * (1 - a_values - b_values)))
    return (code_f32.sign() * torch.round(127 * q)).to(torch.int8)


def _reference_dynamic_a8_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bits: int,
    group_size: int,
    activation_group_size: int | None = None,
) -> torch.Tensor:
    input_size = x.shape[-1]
    x_f32 = x.reshape(-1, input_size).to(torch.float32)
    activation_group_size = activation_group_size or input_size
    activation_groups = input_size // activation_group_size
    grouped_x = x_f32.reshape(-1, activation_groups, activation_group_size)
    amax = grouped_x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    activation_scale = amax / 127
    x_int8 = (
        torch.round(grouped_x * (127 / amax))
        .clamp(-127, 127)
        .to(torch.int8)
        .reshape(-1, input_size)
    )

    codes = unpack_cubic_codes(packed, bits, input_size)
    carriers = _reference_carriers(codes, a, b, bits, group_size)
    output = torch.zeros(
        x_int8.shape[0], packed.shape[0], device=x.device, dtype=torch.float32
    )
    start = 0
    while start < input_size:
        weight_group = start // group_size
        activation_group = start // activation_group_size
        end = min(
            (weight_group + 1) * group_size,
            (activation_group + 1) * activation_group_size,
            input_size,
        )
        partial = (
            x_int8[:, start:end].to(torch.float32)
            @ carriers[:, start:end].to(torch.float32).T
        )
        output += partial * (
            activation_scale[:, activation_group]
            * scale[:, weight_group][None, :]
            / 127
        )
        start = end
    return output.reshape(*x.shape[:-1], packed.shape[0]).to(x.dtype)


@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_carrier_levels_use_round_to_even(bits: int):
    a = torch.tensor([0.5, 1.0], dtype=torch.float16)
    b = torch.tensor([0.25, 0.0], dtype=torch.float16)

    actual = cubic_carrier_levels(bits, a, b)
    if bits == 1:
        expected = torch.full((2, 1), 127, dtype=torch.int8)
    else:
        magnitude_max = (1 << (bits - 1)) - 1
        t = torch.arange(magnitude_max + 1, dtype=torch.float32) / magnitude_max
        a_f32 = a.float()[:, None]
        b_f32 = b.float()[:, None]
        q = t * (a_f32 + t * (b_f32 + t * (1 - a_f32 - b_f32)))
        expected = torch.round(127 * q).to(torch.int8)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.int8
    assert torch.all(actual[..., 1:] >= actual[..., :-1])
    assert torch.all(actual[..., -1] == 127)
    if bits > 1:
        assert torch.all(actual[..., 0] == 0)


@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_carrier_dequant_preserves_checkpoint_packing(bits: int, group_size: int):
    outputs, input_size = 3, group_size + 13
    codes = _make_codes((outputs, input_size), bits)
    packed = pack_cubic_codes(codes, bits)
    packed_before = packed.clone()
    groups = math.ceil(input_size / group_size)
    scale = torch.linspace(0.2, 1.1, outputs * groups, dtype=torch.float32).reshape(
        outputs, groups
    )
    a = torch.full((outputs, groups), 0.5, dtype=torch.float16)
    b = torch.full((outputs, groups), 0.25, dtype=torch.float16)

    actual = dequantize_cubic_carrier(
        packed,
        scale,
        a,
        b,
        total_bits=bits,
        group_size=group_size,
        group_out=1,
        num_values=input_size,
    )
    carriers = _reference_carriers(codes, a, b, bits, group_size).float()
    group_indices = torch.arange(input_size) // group_size
    expected = carriers * scale[:, group_indices] / 127

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(packed, packed_before, rtol=0, atol=0)
    assert packed.dtype == torch.uint8
    assert packed.shape == (outputs, math.ceil(input_size * bits / 8))


def test_cubic_carrier_dequant_requires_fp32_weight_scale():
    packed = pack_cubic_codes(torch.tensor([[0, 1]], dtype=torch.int8), 2)
    metadata = torch.ones((1, 1), dtype=torch.float16)

    with pytest.raises(ValueError, match="scale must be FP32"):
        dequantize_cubic_carrier(
            packed,
            metadata,
            metadata,
            torch.zeros_like(metadata),
            total_bits=2,
            group_size=32,
            group_out=1,
            num_values=2,
        )


@pytest.mark.parametrize("enabled", (False, True))
def test_cubic_dynamic_a8_is_runtime_only(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
):
    from vllm import envs
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.cubic import (
        CUBIC_FORMAT,
        CubicConfig,
        CubicLinearMethod,
    )

    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8", enabled)
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "linear": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "group_size": 128},
                }
            },
        }
    )

    method = config.get_quant_method(object.__new__(LinearBase), "model.proj")

    assert isinstance(method, CubicLinearMethod)
    assert method.dynamic_a8 is enabled


@pytest.mark.parametrize("tokens", (1, 8, 16, 64, 256, 2048))
@pytest.mark.parametrize("dynamic_a8", (False, True))
def test_cubic_linear_activation_mode_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    dynamic_a8: bool,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        CubicLinearMethod,
        CubicScheme,
    )

    calls = {"a16": 0, "a8": 0}

    def a16(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        calls["a16"] += 1
        return x.new_zeros((*x.shape[:-1], 7))

    def a8(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        calls["a8"] += 1
        return x.new_zeros((*x.shape[:-1], 7))

    monkeypatch.setattr(cubic_kernels, "cubic_linear", a16)
    monkeypatch.setattr(cubic_kernels, "cubic_linear_dynamic_a8", a8)
    monkeypatch.setattr(cubic_kernels, "cubic_linear_dynamic_a8_precomputed", a8)

    layer = SimpleNamespace(
        weight_packed=torch.empty(7, 64, dtype=torch.uint8),
        weight_carrier=torch.empty(7, 64, dtype=torch.int8),
        weight_scale=torch.empty(7, 2, dtype=torch.float32),
        weight_a=torch.empty(7, 2, dtype=torch.float16),
        weight_b=torch.empty(7, 2, dtype=torch.float16),
        input_size_per_partition=64,
    )
    method = CubicLinearMethod(
        CubicScheme(num_bits=8, group_size=32, group_out=1),
        dynamic_a8=dynamic_a8,
    )

    output = method.apply(layer, torch.empty(tokens, 64, dtype=torch.bfloat16))

    assert output.shape == (tokens, 7)
    assert calls == ({"a16": 0, "a8": 1} if dynamic_a8 else {"a16": 1, "a8": 0})


@ONLINE_MLP_REMOVED
def test_cubic_dynamic_a8_online_is_a_single_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm import envs
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.cubic import (
        CUBIC_FORMAT,
        CubicConfig,
        CubicLinearMethod,
    )

    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8", False)
    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8_ONLINE", True)
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "linear": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 2, "group_size": 512},
                }
            },
        }
    )

    method = config.get_quant_method(object.__new__(LinearBase), "model.proj")

    assert isinstance(method, CubicLinearMethod)
    assert method.dynamic_a8 is True


@ONLINE_MLP_REMOVED
def test_cubic_online_a8_and_query_protected_cubic_kv_are_orthogonal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online W/A carrier selection must not quantize MLA queries."""
    from vllm import envs
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.cubic import (
        CUBIC_FORMAT,
        CubicConfig,
        CubicLinearMethod,
    )

    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8", False)
    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8_ONLINE", True)
    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "linear": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 3, "group_size": 256},
                }
            },
        }
    )
    method = config.get_quant_method(object.__new__(LinearBase), "model.proj")
    cache_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="cubic8"),
        attention_config=SimpleNamespace(use_prefill_query_quantization=True),
    )

    assert isinstance(method, CubicLinearMethod)
    assert method.dynamic_a8 is True
    assert (
        MLACommonMetadataBuilder.determine_prefill_query_data_type(
            cache_config, torch.bfloat16, "cubic8"
        )
        == torch.bfloat16
    )


@pytest.mark.parametrize(
    ("bits", "group_size"),
    tuple((bits, group) for bits in range(1, 9) for group in GROUP_SIZES),
)
@ONLINE_MLP_REMOVED
def test_cubic_online_groupwise_a8_is_not_decode_batch_limited(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    group_size: int,
):
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _cubic_online_groupwise_a8_eligible,
    )

    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8_ONLINE", True)
    metadata_dtype = torch.int8 if bits == 3 else torch.float16
    metadata = torch.empty(1, dtype=metadata_dtype)

    assert _cubic_online_groupwise_a8_eligible(
        num_bits=bits,
        group_size=group_size,
        intermediate_size=2 * group_size,
        num_tokens=64,
        use_gemv=True,
        grouped_routes=2,
        activation=MoEActivation.SITU,
        w2_a=metadata,
        w2_b=metadata,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_dynamic_a8_quantization_handles_zero_activation_row():
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    x = torch.zeros((2, 129), device="cuda", dtype=torch.bfloat16)
    quantized, scale = per_token_quant_int8(x)

    assert torch.count_nonzero(quantized).item() == 0
    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", (256, 512))
@ONLINE_MLP_REMOVED
def test_cubic_online_groupwise_a8_producer_is_per_sample_per_group(
    group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _quantize_cubic_groupwise_a8,
    )

    x = torch.randn(3, 2 * group_size, device="cuda", dtype=torch.bfloat16)
    x[0, :group_size] *= 0.01
    x[0, group_size:] *= 10
    carrier = _quantize_cubic_groupwise_a8(x, group_size)
    expected_scales = (
        x.float().reshape(3, 2, group_size).abs().amax(dim=-1).clamp_min(1e-10) / 127
    )

    torch.testing.assert_close(carrier.scales, expected_scales)
    assert carrier.values.shape == x.shape
    assert carrier.scales.shape == (3, 2)
    assert carrier.scales.dtype == torch.float32
    assert carrier.values.dtype == torch.int8
    assert not torch.equal(carrier.scales[0, 0], carrier.scales[0, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", (256, 512))
@ONLINE_MLP_REMOVED
def test_cubic_online_cubic8_producer_persists_true_curve_metadata(
    group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _quantize_cubic_groupwise_cubic8,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260804)
    x = torch.randn(
        4,
        2 * group_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    encoded = _quantize_cubic_groupwise_cubic8(x, group_size)
    codes = encoded.codes.reshape(4, 2, group_size)
    t = codes.float().abs() / 127.0
    a = encoded.a.float()[..., None]
    b = encoded.b.float()[..., None]
    decoded = (
        codes.sign().float()
        * encoded.scales[..., None]
        * t
        * (a + t * (b + t * (1.0 - a - b)))
    ).reshape_as(x)

    assert encoded.codes.dtype == torch.int8
    assert encoded.scales.dtype == torch.float32
    assert encoded.a.dtype == torch.float16
    assert encoded.b.dtype == torch.float16
    assert encoded.scales.shape == encoded.a.shape == encoded.b.shape == (4, 2)
    assert torch.any((encoded.a != 1) | (encoded.b != 0))
    nrmse = (
        decoded - x.float()
    ).square().mean().sqrt() / x.float().square().mean().sqrt()
    assert nrmse < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", (256, 512))
@ONLINE_MLP_REMOVED
def test_cubic_online_cubic8_w2_consumer_matches_exact_curve(
    group_size: int,
):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        CubicA8Code,
        _launch_cubic_moe_cubic8_w2,
        _launch_cubic_moe_cubic8_w2_moment,
    )

    device = torch.device("cuda")
    rows, outputs, groups = 3, 37, 2
    input_size = groups * group_size
    activation_codes = torch.randint(
        -127, 128, (rows, input_size), device=device, dtype=torch.int8
    )
    activation_scale = torch.tensor(
        [[0.01, 0.03], [0.07, 0.02], [0.04, 0.09]],
        device=device,
        dtype=torch.float32,
    )
    activation_a = torch.tensor(
        [[0.5, 0.75], [1.0, 0.5], [0.5, 1.0]],
        device=device,
        dtype=torch.float16,
    )
    activation_b = torch.tensor(
        [[0.25, -0.25], [0.0, 0.0], [0.25, 0.0]],
        device=device,
        dtype=torch.float16,
    )
    carrier = CubicA8Code(
        activation_codes,
        activation_scale,
        activation_a,
        activation_b,
        group_size,
    )
    weight_codes = _make_codes((1, outputs, input_size), 2).to(device)
    packed = pack_cubic_codes(weight_codes, 2)
    weight_scale = torch.linspace(
        0.003,
        0.031,
        outputs * groups,
        device=device,
        dtype=torch.float32,
    ).reshape(1, outputs, groups)
    actual = torch.empty(rows, outputs, device=device, dtype=torch.bfloat16)
    topk_weights = torch.ones(rows, device=device)
    sorted_ids = torch.arange(rows, device=device, dtype=torch.int32)
    expert_ids = torch.zeros(rows, device=device, dtype=torch.int32)
    count = torch.tensor(rows, device=device, dtype=torch.int32)
    _launch_cubic_moe_cubic8_w2(
        carrier,
        packed,
        weight_scale,
        actual,
        topk_weights,
        sorted_ids,
        expert_ids,
        count,
        logical_k=input_size,
        top_k=1,
        multiply_routed_weight=False,
        route_ctas=rows,
    )
    moment = torch.empty_like(actual)
    _launch_cubic_moe_cubic8_w2_moment(
        carrier,
        packed,
        weight_scale,
        moment,
        topk_weights,
        sorted_ids,
        expert_ids,
        count,
        logical_k=input_size,
        weight_group_size=group_size,
        top_k=1,
        multiply_routed_weight=False,
        route_ctas=rows,
    )
    lut = torch.empty_like(actual)
    torch.ops._C.cubic_w2_cubic8_lut_gemv(
        carrier.codes,
        carrier.scales,
        carrier.a,
        carrier.b,
        packed,
        weight_scale,
        lut,
        topk_weights,
        sorted_ids,
        expert_ids,
        count,
        group_size,
        group_size,
        1,
        False,
        rows,
        8,
    )

    code_groups = activation_codes.reshape(rows, groups, group_size)
    t = code_groups.float().abs() / 127.0
    aa = activation_a.float()[..., None]
    bb = activation_b.float()[..., None]
    activation = (
        code_groups.sign().float()
        * activation_scale[..., None]
        * t
        * (aa + t * (bb + t * (1.0 - aa - bb)))
    )
    expected = torch.zeros(rows, outputs, device=device, dtype=torch.float32)
    for group in range(groups):
        expected += (
            activation[:, group]
            @ weight_codes[0, :, group * group_size : (group + 1) * group_size]
            .float()
            .T
        ) * weight_scale[0, :, group]
    error_nrmse = (
        actual.float() - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert error_nrmse < 0.005
    moment_error_nrmse = (
        moment.float() - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert moment_error_nrmse < 0.01
    lut_error_nrmse = (
        lut.float() - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert lut_error_nrmse < 0.005
    assert torch.isfinite(actual).all()
    assert torch.isfinite(moment).all()
    assert torch.isfinite(lut).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", (256, 512))
@ONLINE_MLP_REMOVED
def test_cubic_online_cubic8_fused_w2_producer_matches_staged_reference(
    group_size: int,
):
    """W2+SITU producer avoids BF16 storage without changing its contract."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _launch_cubic_moe_situ_cubic8_2bit,
        _launch_cubic_moe_situ_gemv_2bit,
        _quantize_cubic_groupwise_a8,
    )

    device = torch.device("cuda")
    experts, rows = 2, 2
    hidden = output_n = group_size
    generator = torch.Generator(device=device).manual_seed(91000 + group_size)
    x = torch.randn(
        rows, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    codes = _make_codes((experts, 2 * output_n, hidden), 2).to(device)
    packed = pack_cubic_codes(codes, 2)
    weight_scale = (
        torch.rand(
            experts,
            2 * output_n,
            hidden // group_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    topk_ids = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    topk_weights = torch.ones(rows, 1, device=device, dtype=torch.float32)
    expert_map = torch.arange(experts, device=device, dtype=torch.int32)
    sorted_ids, expert_ids, padded_count = moe_align_block_size(
        topk_ids,
        1,
        experts,
        expert_map,
        ignore_invalid_experts=True,
    )
    staged = torch.empty(rows, output_n, device=device, dtype=torch.bfloat16)
    _launch_cubic_moe_situ_gemv_2bit(
        x,
        packed,
        weight_scale,
        staged,
        topk_weights,
        sorted_ids,
        expert_ids,
        padded_count,
        logical_k=hidden,
        group_size=group_size,
        group_out=1,
        top_k=1,
        multiply_routed_weight=False,
        beta=4.0,
        linear_beta=25.0,
        dynamic_a8=True,
        grouped_routes=1,
    )
    expected = _quantize_cubic_groupwise_a8(staged, group_size)
    actual = _launch_cubic_moe_situ_cubic8_2bit(
        x,
        packed,
        weight_scale,
        topk_weights,
        sorted_ids,
        expert_ids,
        padded_count,
        logical_k=hidden,
        group_size=group_size,
        group_out=1,
        output_group_size=group_size,
        top_k=1,
        multiply_routed_weight=False,
        beta=4.0,
        linear_beta=25.0,
    )

    def decode_actual(carrier):
        code = carrier.codes.float().reshape(rows, -1, group_size)
        return (code * carrier.scales.unsqueeze(-1)).reshape(rows, -1)

    def decode_expected(carrier):
        code = carrier.values.float().reshape(rows, -1, group_size)
        return (code * carrier.scales.unsqueeze(-1)).reshape(rows, -1)

    torch.testing.assert_close(
        decode_actual(actual), decode_expected(expected), rtol=0.02, atol=0.02
    )
    assert torch.isfinite(actual.scales).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("bits", "group_size"),
    tuple((bits, group) for bits in range(1, 9) for group in GROUP_SIZES),
)
@pytest.mark.parametrize("grouped_routes", (1, 2))
@ONLINE_MLP_REMOVED
def test_cubic_online_groupwise_a8_consumer_matches_reference(
    bits: int,
    group_size: int,
    grouped_routes: int,
):
    if grouped_routes == 2 and group_size < 256:
        pytest.skip("production grouping heuristic uses singleton small groups")
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        CubicA8Carrier,
        _launch_cubic_moe_groupwise_a8,
    )

    device = torch.device("cuda")
    rows, outputs, input_size = 4, 37, 2 * group_size
    codes = _make_codes((1, outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    weight_scale = torch.linspace(
        0.001,
        0.02,
        outputs * 2,
        device=device,
        dtype=torch.float32,
    ).reshape(1, outputs, 2)
    fit_a = torch.full_like(weight_scale, 0.5, dtype=torch.float16)
    fit_b = torch.full_like(weight_scale, 0.25, dtype=torch.float16)
    carriers = _reference_carriers(
        codes,
        fit_a,
        fit_b,
        bits,
        group_size,
    )
    if bits == 3:
        levels = cubic_carrier_levels(bits, fit_a, fit_b)
        runtime_a = levels[..., 1].contiguous()
        runtime_b = levels[..., 2].contiguous()
    else:
        runtime_a, runtime_b = fit_a, fit_b
    values = torch.randint(
        -127,
        128,
        (rows, input_size),
        device=device,
        dtype=torch.int8,
    )
    activation_scale = torch.tensor(
        [
            [0.001, 0.017],
            [0.031, 0.004],
            [0.009, 0.023],
            [0.027, 0.006],
        ],
        device=device,
        dtype=torch.float32,
    )
    carrier = CubicA8Carrier(values, activation_scale, group_size)
    actual = torch.empty(rows, outputs, device=device, dtype=torch.bfloat16)
    topk_weights = torch.ones(rows, device=device)
    sorted_ids = torch.arange(rows, device=device, dtype=torch.int32)
    expert_ids = torch.zeros(rows, device=device, dtype=torch.int32)
    count = torch.tensor(rows, device=device, dtype=torch.int32)

    _launch_cubic_moe_groupwise_a8(
        carrier,
        packed,
        weight_scale,
        runtime_a,
        runtime_b,
        actual,
        topk_weights,
        sorted_ids,
        expert_ids,
        count,
        logical_k=input_size,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        top_k=1,
        multiply_routed_weight=False,
        sum_routes=False,
        route_ctas=rows,
        grouped_routes=grouped_routes,
    )

    expected = torch.zeros(rows, outputs, device=device, dtype=torch.float32)
    for group in range(2):
        start, end = group * group_size, (group + 1) * group_size
        partial = values[:, start:end].float() @ carriers[0, :, start:end].float().T
        expected += partial * (
            activation_scale[:, group, None] * weight_scale[0, :, group][None, :] / 127
        )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("bits", "group_size"),
    tuple((bits, group) for bits in range(1, 9) for group in (256, 512)),
)
@ONLINE_MLP_REMOVED
def test_cubic_online_a8_bounded_batch_matches_token_slices(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    group_size: int,
):
    """Exact Online Cubic and Dynamic-A8 fallbacks remain chunk invariant."""
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization import cubic_kernels

    device = torch.device("cuda")
    experts, tokens, top_k = 2, 17, 2
    hidden = intermediate = 2 * group_size

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
        packed = pack_cubic_codes(codes, bits)
        groups = shape[-1] // group_size
        scale = torch.full(
            (*shape[:-1], groups), 0.01, device=device, dtype=torch.float32
        )
        fit_a = torch.full_like(scale, 0.5, dtype=torch.float16)
        fit_b = torch.full_like(scale, 0.25, dtype=torch.float16)
        if bits == 3:
            levels = cubic_carrier_levels(bits, fit_a, fit_b)
            runtime_a = levels[..., 1].contiguous()
            runtime_b = levels[..., 2].contiguous()
        else:
            runtime_a, runtime_b = fit_a, fit_b
        return packed, scale, runtime_a, runtime_b

    w1, w1_scale, w1_a, w1_b = make_weight((experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((experts, hidden, intermediate))
    generator = torch.Generator(device=device).manual_seed(18700 + bits + group_size)
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.randint(
        experts,
        (tokens, top_k),
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=-1
    )
    expert_map = torch.arange(experts, device=device, dtype=torch.int32)

    monkeypatch.setattr(envs, "VLLM_CUBIC_DYNAMIC_A8_ONLINE", True)

    def run(start: int, end: int) -> torch.Tensor:
        return cubic_kernels.cubic_fused_moe_dynamic_a8(
            x[start:end],
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            topk_weights[start:end],
            topk_ids[start:end],
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

    monkeypatch.setattr(
        cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES", 1 << 60
    )
    expected = torch.cat([run(0, 8), run(8, 16), run(16, 17)])

    calls = 0
    original_launch = cubic_kernels._launch_cubic_moe_groupwise_a8

    def counted_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(cubic_kernels, "_launch_cubic_moe_groupwise_a8", counted_launch)
    bytes_per_token = top_k * hidden * x.element_size()
    monkeypatch.setattr(cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_MIN_BYTES", 1)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_A8_ROUTE_WORKSPACE_MAX_BYTES",
        3 * bytes_per_token,
    )
    actual = run(0, tokens)

    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    if bits == 2 and group_size == 512:
        assert calls >= math.ceil(tokens / 3)
    else:
        # Unsupported exact Cubic consumers must remain on established
        # Dynamic A8 rather than silently using a linear groupwise carrier.
        assert calls == 0
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_dynamic_a8_linear_matches_pytorch_reference(bits: int, group_size: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8,
    )

    device = torch.device("cuda")
    leading_shape, outputs, input_size = (4, 4), 7, group_size + 13
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    packed_before = packed.clone()
    groups = math.ceil(input_size / group_size)
    scale = (
        torch.linspace(0.01, 0.08, outputs * groups, device=device)
        .reshape(outputs, groups)
        .float()
    )
    a = torch.full((outputs, groups), 0.5, device=device, dtype=torch.float16)
    b = torch.full((outputs, groups), 0.25, device=device, dtype=torch.float16)
    generator = torch.Generator(device=device).manual_seed(2000 + bits + group_size)
    x = torch.randn(
        *leading_shape,
        input_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    x[0, 0].zero_()
    x[0, 1, 0] = 100

    activation_group_size = cubic_dynamic_a8_group_size(
        input_size=input_size,
        weight_group_size=group_size,
    )
    expected = _reference_dynamic_a8_linear(
        x,
        packed,
        scale,
        a,
        b,
        bits,
        group_size,
        activation_group_size,
    )
    actual = cubic_linear_dynamic_a8(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        input_size=input_size,
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(packed, packed_before, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0, 0]).item() == 0
    assert torch.isfinite(actual).all()
    assert actual.dtype == x.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("tokens", (1, 16))
def test_cubic_dynamic_a8_linear_supports_output_groups(bits: int, tokens: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8,
    )

    group_out, group_size = 128, 512
    outputs, input_size = 256, 1024
    device = torch.device("cuda")
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, input_size // group_size)
    scale = torch.linspace(
        0.01, 0.04, math.prod(metadata_shape), device=device
    ).reshape(metadata_shape)
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(tokens, input_size, device=device, dtype=torch.bfloat16)
    expanded_scale = scale.repeat_interleave(group_out, dim=0)
    expanded_a = a.repeat_interleave(group_out, dim=0)
    expanded_b = b.repeat_interleave(group_out, dim=0)

    expected = _reference_dynamic_a8_linear(
        x,
        packed,
        expanded_scale,
        expanded_a,
        expanded_b,
        bits,
        group_size,
        cubic_dynamic_a8_group_size(
            input_size=input_size,
            weight_group_size=group_size,
        ),
    )
    actual = cubic_linear_dynamic_a8(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=input_size,
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
def test_cubic_dynamic_a8_linear_supports_output_only_groups(bits: int):
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8,
    )

    group_out, group_size = 32, 1
    outputs = input_size = 64
    device = torch.device("cuda")
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, input_size)
    scale = torch.rand(metadata_shape, device=device, dtype=torch.float32) * 0.1
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(16, input_size, device=device, dtype=torch.bfloat16)

    expected = _reference_dynamic_a8_linear(
        x,
        packed,
        scale.repeat_interleave(group_out, dim=0),
        a.repeat_interleave(group_out, dim=0),
        b.repeat_interleave(group_out, dim=0),
        bits,
        group_size,
        cubic_dynamic_a8_group_size(
            input_size=input_size,
            weight_group_size=group_size,
        ),
    )
    actual = cubic_linear_dynamic_a8(
        x,
        packed,
        scale,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        input_size=input_size,
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("tokens", (1, 16, 256))
@pytest.mark.parametrize(("group_out", "group_size"), ((1, 128), (32, 1), (32, 128)))
def test_cubic_precomputed_carrier_matches_dynamic_a8(
    bits: int, tokens: int, group_out: int, group_size: int
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic import (
        materialize_cubic_a8_carrier,
    )

    outputs = input_size = 128
    device = torch.device("cuda")
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    groups = input_size // group_size
    metadata_shape = (outputs // group_out, groups)
    scale = torch.rand(*metadata_shape, device=device, dtype=torch.float32) * 0.1
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(tokens, input_size, device=device, dtype=torch.bfloat16)
    carrier = materialize_cubic_a8_carrier(
        packed,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        input_size=input_size,
        group_out=group_out,
    )
    expected_carrier = _reference_carriers(
        codes,
        a.repeat_interleave(group_out, dim=0),
        b.repeat_interleave(group_out, dim=0),
        bits,
        group_size,
    )
    torch.testing.assert_close(carrier, expected_carrier, rtol=0, atol=0)
    assert carrier.stride() == (1, outputs)

    kwargs = {
        "num_bits": bits,
        "group_size": group_size,
        "group_out": group_out,
        "input_size": input_size,
    }
    expected = cubic_kernels.cubic_linear_dynamic_a8(x, packed, scale, a, b, **kwargs)
    actual = cubic_kernels.cubic_linear_dynamic_a8_precomputed(
        x, carrier, scale, a, b, **kwargs
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("tokens", (1, 4))
@pytest.mark.parametrize("group_out", (1, 32))
def test_cubic_a16_packed_stream_matches_regular_linear(
    bits: int, tokens: int, group_out: int
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic_policy import (
        cubic_linear_token_bucket,
    )

    group_size, outputs, input_size = 128, 64, 128
    device = torch.device("cuda")
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, input_size // group_size)
    scale = torch.rand(*metadata_shape, device=device, dtype=torch.float32) * 0.1
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(tokens, input_size, device=device, dtype=torch.bfloat16)
    kwargs = {
        "num_bits": bits,
        "group_size": group_size,
        "group_out": group_out,
        "input_size": input_size,
    }
    expected = cubic_kernels.cubic_linear(x, packed, scale, a, b, **kwargs)
    key = (
        torch.accelerator.current_device_index(),
        False,
        False,
        bits,
        outputs,
        input_size,
        group_size,
        group_out,
        cubic_linear_token_bucket(tokens),
    )
    cubic_kernels._CUBIC_LINEAR_STREAM_TACTICS[key] = (8, 4)
    try:
        actual = cubic_kernels.cubic_linear(x, packed, scale, a, b, **kwargs)
    finally:
        cubic_kernels._CUBIC_LINEAR_STREAM_TACTICS.pop(key)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def test_cubic_linear_does_not_extrapolate_gemv_past_cuda_grid_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels
    from vllm.model_executor.layers.quantization.cubic_policy import (
        cubic_linear_token_bucket,
    )

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    key = (
        0,
        True,
        True,
        8,
        4608,
        4608,
        512,
        1,
        cubic_linear_token_bucket(65_536),
    )
    monkeypatch.setitem(cubic_kernels._CUBIC_LINEAR_EXECUTION_TACTICS, key, True)

    assert not cubic_kernels._cubic_linear_use_gemv(
        dynamic_a8=True,
        precomputed_carrier=True,
        num_bits=8,
        n=4608,
        k=4608,
        group_size=512,
        group_out=1,
        m=65_536,
        fallback=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    ("group_out", "group_size"),
    ((1, 128), (128, 1), (32, 64)),
)
def test_cubic_low_m_linear_is_batch_invariant(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    group_out: int,
    group_size: int,
) -> None:
    from vllm.model_executor.layers.quantization.cubic import (
        materialize_cubic_a8_carrier,
    )
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear,
        cubic_linear_dynamic_a8,
        cubic_linear_dynamic_a8_precomputed,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    outputs = input_size = 128
    device = torch.device("cuda")
    codes = _make_codes((outputs, input_size), bits).to(device)
    packed = pack_cubic_codes(codes, bits)
    metadata_shape = (outputs // group_out, input_size // group_size)
    scale = torch.rand(metadata_shape, device=device, dtype=torch.float32) * 0.1
    a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
    b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
    x = torch.randn(9, input_size, device=device, dtype=torch.bfloat16)
    carrier = materialize_cubic_a8_carrier(
        packed,
        a,
        b,
        num_bits=bits,
        group_size=group_size,
        input_size=input_size,
        group_out=group_out,
    )
    kwargs = {
        "num_bits": bits,
        "group_size": group_size,
        "group_out": group_out,
        "input_size": input_size,
    }

    for kernel, weight in (
        (cubic_linear, packed),
        (cubic_linear_dynamic_a8, packed),
        (cubic_linear_dynamic_a8_precomputed, carrier),
    ):
        batched = kernel(x, weight, scale, a, b, **kwargs)
        tokenwise = torch.cat(
            [kernel(row, weight, scale, a, b, **kwargs) for row in x.split(1)]
        )
        torch.testing.assert_close(batched, tokenwise, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_w8_large_linear_keeps_k_reduction_batch_invariant() -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_linear_dynamic_a8_precomputed,
    )

    outputs, input_size, group_size = 8192, 5120, 512
    carrier = torch.randint(
        -127,
        128,
        (outputs, input_size),
        device="cuda",
        dtype=torch.int8,
    )
    scale = torch.rand(
        outputs,
        input_size // group_size,
        device="cuda",
        dtype=torch.float32,
    )
    unused = torch.empty(0, device="cuda")
    row = torch.linspace(
        -1,
        1,
        input_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(1, -1)
    kwargs = {
        "num_bits": 8,
        "group_size": group_size,
        "group_out": 1,
        "input_size": input_size,
    }

    solo = cubic_linear_dynamic_a8_precomputed(
        row, carrier, scale, unused, unused, **kwargs
    )
    batched = cubic_linear_dynamic_a8_precomputed(
        row.repeat(256, 1), carrier, scale, unused, unused, **kwargs
    )[:1]

    torch.testing.assert_close(batched, solo, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("dynamic_a8", (False, True))
@pytest.mark.parametrize(
    "forced_backend", (None, True, False), ids=("auto", "gemv", "gemm")
)
def test_cubic_moe_is_exact_across_decode_batch_shapes(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    dynamic_a8: bool,
    forced_backend: bool | None,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization import cubic_kernels

    device = torch.device("cuda")
    experts, tokens, top_k = 2, 16, 2
    hidden = intermediate = group_size = 128
    generator = torch.Generator(device=device).manual_seed(9100 + bits)

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
        packed = pack_cubic_codes(codes, bits)
        metadata_shape = (*shape[:-1], shape[-1] // group_size)
        scale = torch.rand(
            metadata_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        scale = scale * 0.02 + 0.01
        a = torch.full(metadata_shape, 0.5, device=device, dtype=torch.float16)
        b = torch.full(metadata_shape, 0.25, device=device, dtype=torch.float16)
        return packed, scale, a, b

    w1, w1_scale, w1_a, w1_b = make_weight((experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((experts, hidden, intermediate))
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.tensor([[0, 1]], device=device, dtype=torch.int32).repeat(
        tokens, 1
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=-1
    )
    kernel = (
        cubic_kernels.cubic_fused_moe_dynamic_a8
        if dynamic_a8
        else cubic_kernels.cubic_fused_moe
    )
    if forced_backend is not None:
        device_index = torch.accelerator.current_device_index()
        for num_tokens in (1, tokens):
            key = (
                device_index,
                dynamic_a8,
                bits,
                hidden,
                intermediate,
                group_size,
                1,
                experts,
                cubic_kernels.cubic_token_bucket(num_tokens),
            )
            monkeypatch.setitem(
                cubic_kernels._CUBIC_MOE_EXECUTION_TACTICS,
                key,
                forced_backend,
            )

    def run(start: int, end: int) -> torch.Tensor:
        return kernel(
            x[start:end],
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            topk_weights[start:end],
            topk_ids[start:end],
            activation=MoEActivation.SITU,
            apply_router_weight_on_input=False,
            global_num_experts=experts,
            expert_map=None,
            num_bits=bits,
            group_size=group_size,
            group_out=1,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )

    solo = run(0, 1)
    batched = run(0, tokens)[:1]
    torch.testing.assert_close(batched, solo, rtol=0, atol=0)
    if forced_backend is None:
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        target = tokens // 2
        solo = run(target, target + 1)
        batched = run(0, tokens)[target : target + 1]
        torch.testing.assert_close(batched, solo, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_dynamic_a8_moe_ep_is_exact_across_batch_shapes() -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_fused_moe_dynamic_a8,
    )

    device = torch.device("cuda")
    bits, group_out, group_size = 4, 128, 512
    local_experts, global_experts, tokens, top_k = 2, 6, 16, 6
    hidden, intermediate = 4096, 2048
    generator = torch.Generator(device=device).manual_seed(19403)

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
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
        return packed, scale, a, b

    w1, w1_scale, w1_a, w1_b = make_weight((local_experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((local_experts, hidden, intermediate))
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.arange(global_experts, device=device, dtype=torch.int32).repeat(
        tokens, 1
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=-1
    )
    expert_map = torch.tensor([0, 1, -1, -1, -1, -1], device=device, dtype=torch.int32)

    def run(end: int) -> torch.Tensor:
        return cubic_fused_moe_dynamic_a8(
            x[:end],
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            topk_weights[:end],
            topk_ids[:end],
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=global_experts,
            expert_map=expert_map,
            num_bits=bits,
            group_size=group_size,
            group_out=group_out,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
        )

    solo = run(1)
    batched = run(tokens)[:1]
    torch.testing.assert_close(batched, solo, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("grouped_routes", (1, 2, 4))
@pytest.mark.parametrize("group_out", (1, 32, 128))
def test_cubic_dynamic_a8_situ_w2_supports_group_out(
    group_out: int, grouped_routes: int
) -> None:
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        _launch_cubic_moe_situ_gemv_2bit,
        per_token_quant_int8,
    )

    device = torch.device("cuda")
    tokens, hidden, intermediate = 4, 256, 128
    codes = _make_codes((1, 2 * intermediate, hidden), 2).to(device)
    packed = pack_cubic_codes(codes, 2)
    scale = torch.linspace(
        0.01,
        0.04,
        2 * intermediate // group_out,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 2 * intermediate // group_out, 1)
    inputs = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    topk_weights = torch.ones(tokens, 1, device=device)
    sorted_token_ids = torch.arange(tokens, device=device, dtype=torch.int32)
    expert_ids = torch.zeros(
        math.ceil(tokens / grouped_routes), device=device, dtype=torch.int32
    )
    count = torch.tensor([tokens], device=device, dtype=torch.int32)
    output = torch.empty(tokens, 1, intermediate, device=device, dtype=torch.bfloat16)

    _launch_cubic_moe_situ_gemv_2bit(
        inputs,
        packed,
        scale,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        count,
        logical_k=hidden,
        group_size=hidden,
        group_out=group_out,
        top_k=1,
        multiply_routed_weight=False,
        beta=4.0,
        linear_beta=25.0,
        dynamic_a8=True,
        grouped_routes=grouped_routes,
    )

    quantized, input_scale = per_token_quant_int8(inputs.contiguous())
    expanded_scale = scale.repeat_interleave(group_out, dim=1)
    weight = codes.float() * expanded_scale
    gate_up = torch.einsum("tk,eok->teo", quantized.float(), weight)
    gate_up *= input_scale.reshape(tokens, 1, 1)
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.to(torch.bfloat16).float()
    up = up.to(torch.bfloat16).float()
    expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected *= 25.0 * torch.tanh(up / 25.0)

    torch.testing.assert_close(
        output[:, 0].float(), expected[:, 0], rtol=0.02, atol=0.02
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dynamic_a8", (False, True))
def test_cubic_situ_warmup_accepts_legacy_scalar_group_size(
    monkeypatch: pytest.MonkeyPatch, dynamic_a8: bool
) -> None:
    from vllm.model_executor.layers.quantization import cubic_kernels

    device = torch.device("cuda")
    tokens, hidden, intermediate = 4, 256, 128
    codes = _make_codes((1, 2 * intermediate, hidden), 2).to(device)
    packed = pack_cubic_codes(codes, 2)
    scale = torch.full(
        (1, 2 * intermediate, 1), 0.02, device=device, dtype=torch.float32
    )
    coefficients = torch.empty_like(scale, dtype=torch.float16)
    inputs = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    topk_weights = torch.ones(tokens, 1, device=device)
    topk_ids = torch.zeros(tokens, 1, device=device, dtype=torch.int32)

    def do_bench(function, **_kwargs) -> float:
        function()
        torch.accelerator.synchronize()
        return 1.0

    monkeypatch.setattr(cubic_kernels.triton.testing, "do_bench", do_bench)
    cubic_kernels.calibrate_cubic_moe_route_ctas(
        inputs,
        packed,
        scale,
        coefficients,
        coefficients,
        topk_weights,
        topk_ids,
        None,
        dynamic_a8=dynamic_a8,
        global_num_experts=1,
        logical_k=hidden,
        num_bits=2,
        group_size=hidden,
        group_out=1,
        top_k=1,
        multiply_routed_weight=False,
        grouped_routes=1,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("expert_parallel", (False, True))
@pytest.mark.parametrize("group_size", (1, *GROUP_SIZES))
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("group_out", (1, 32))
def test_cubic_dynamic_a8_moe_matches_pytorch_reference(
    bits: int,
    group_size: int,
    expert_parallel: bool,
    group_out: int,
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_fused_moe_dynamic_a8,
    )

    device = torch.device("cuda")
    experts, tokens, top_k = 2, 16, 2
    hidden = intermediate = max(64, group_size)

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
        packed = pack_cubic_codes(codes, bits)
        groups = shape[-1] // group_size
        scale = torch.full(
            (*shape[:-2], shape[-2] // group_out, groups),
            0.1,
            device=device,
            dtype=torch.float32,
        )
        a = torch.full_like(scale, 0.5, dtype=torch.float16)
        b = torch.full_like(scale, 0.25, dtype=torch.float16)
        return packed, scale, a, b

    w1, w1_scale, w1_a, w1_b = make_weight((experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((experts, hidden, intermediate))
    w1_before, w2_before = w1.clone(), w2.clone()
    generator = torch.Generator(device=device).manual_seed(73)
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32).repeat(
        (tokens + 1) // 2, 1
    )[:tokens]
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=-1
    )

    actual = cubic_fused_moe_dynamic_a8(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SITU,
        apply_router_weight_on_input=False,
        global_num_experts=experts,
        expert_map=(
            torch.arange(experts, device=device, dtype=torch.int32)
            if expert_parallel
            else None
        ),
        num_bits=bits,
        group_size=group_size,
        group_out=group_out,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    expected = torch.zeros_like(actual, dtype=torch.float32)
    for token in range(tokens):
        for route in range(top_k):
            expert = topk_ids[token, route].item()
            gate_up = _reference_dynamic_a8_linear(
                x[token : token + 1],
                w1[expert],
                w1_scale[expert].repeat_interleave(group_out, dim=0),
                w1_a[expert].repeat_interleave(group_out, dim=0),
                w1_b[expert].repeat_interleave(group_out, dim=0),
                bits,
                group_size,
            )
            gate = gate_up[:, :intermediate].float()
            up = gate_up[:, intermediate:].float()
            activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            activated *= 25.0 * torch.tanh(up / 25.0)
            expert_output = _reference_dynamic_a8_linear(
                activated.to(torch.bfloat16),
                w2[expert],
                w2_scale[expert].repeat_interleave(group_out, dim=0),
                w2_a[expert].repeat_interleave(group_out, dim=0),
                w2_b[expert].repeat_interleave(group_out, dim=0),
                bits,
                group_size,
            )
            expected[token] += expert_output[0].float() * topk_weights[token, route]

    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(w1, w1_before, rtol=0, atol=0)
    torch.testing.assert_close(w2, w2_before, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cubic_dynamic_a8_w2_grouped_routes_single_token_is_bounded():
    """Cover EP compact-route and fused-sum hazards for grouped W2 routes."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.cubic_kernels import (
        cubic_fused_moe_dynamic_a8,
    )

    device = torch.device("cuda")
    experts, hidden, intermediate, group_size, bits = 3, 256, 256, 256, 2

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
        packed = pack_cubic_codes(codes, bits)
        groups = shape[-1] // group_size
        scale = torch.full(
            (*shape[:-1], groups), 0.1, device=device, dtype=torch.float32
        )
        a = torch.full_like(scale, 0.5, dtype=torch.float16)
        b = torch.full_like(scale, 0.25, dtype=torch.float16)
        return packed, scale, a, b

    w1, w1_scale, w1_a, w1_b = make_weight((experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((experts, hidden, intermediate))
    x = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4]], device=device)

    actual = cubic_fused_moe_dynamic_a8(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SITU,
        apply_router_weight_on_input=False,
        global_num_experts=experts,
        expert_map=torch.arange(experts, device=device, dtype=torch.int32),
        num_bits=bits,
        group_size=group_size,
        group_out=1,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", (1, 2, 3, 4, 8))
def test_cubic_dynamic_a8_chunked_down_projection_matches_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
):
    """The bounded W1-W8 workspace must preserve route math exactly."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization import cubic_kernels

    device = torch.device("cuda")
    experts, tokens, top_k = 3, 17, 2
    hidden = intermediate = group_size = 512

    def make_weight(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        codes = _make_codes(shape, bits).to(device)
        packed = pack_cubic_codes(codes, bits)
        groups = shape[-1] // group_size
        scale = torch.full(
            (*shape[:-1], groups), 0.1, device=device, dtype=torch.float32
        )
        a = torch.full_like(scale, 0.5, dtype=torch.float16)
        b = torch.full_like(scale, 0.25, dtype=torch.float16)
        return packed, scale, a, b

    w1, w1_scale, w1_a, w1_b = make_weight((experts, 2 * intermediate, hidden))
    w2, w2_scale, w2_a, w2_b = make_weight((experts, hidden, intermediate))
    generator = torch.Generator(device=device).manual_seed(9120 + bits)
    x = torch.randn(
        tokens, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.randint(
        experts,
        (tokens, top_k),
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator, device=device), dim=-1
    )
    expert_map = torch.arange(experts, device=device, dtype=torch.int32)

    def run() -> torch.Tensor:
        return cubic_kernels.cubic_fused_moe_dynamic_a8(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
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

    monkeypatch.setattr(
        cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES", 1 << 60
    )
    monkeypatch.setattr(
        cubic_kernels, "_CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES", 1 << 60
    )
    expected = run()

    cache1_bytes_per_token = top_k * (2 * intermediate) * x.element_size()
    monkeypatch.setattr(cubic_kernels, "_CUBIC_A8_PIPELINE_WORKSPACE_MIN_BYTES", 1)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES",
        3 * cache1_bytes_per_token,
    )
    pipeline_actual = run()
    torch.testing.assert_close(pipeline_actual, expected, rtol=0, atol=0)

    monkeypatch.setattr(
        cubic_kernels, "_CUBIC_A8_PIPELINE_WORKSPACE_MAX_BYTES", 1 << 60
    )

    bytes_per_token = top_k * hidden * x.element_size()
    monkeypatch.setattr(cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(cubic_kernels, "_CUBIC_A8_ROUTE_WORKSPACE_MIN_BYTES", 1)
    monkeypatch.setattr(
        cubic_kernels,
        "_CUBIC_A8_ROUTE_WORKSPACE_MAX_BYTES",
        3 * bytes_per_token,
    )
    actual = run()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
