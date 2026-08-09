# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_cubic_mixed_width_expert_config():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.cubic import CUBIC_FORMAT, CubicConfig

    config = CubicConfig.from_config(
        {
            "quant_method": "cubic",
            "format": CUBIC_FORMAT,
            "config_groups": {
                "w4": {
                    "targets": ["re:.*\\.layers\\.(?:2|3)\\.mlp\\.experts"],
                    "weights": {"num_bits": 4, "group_size": 256},
                },
                "w6": {
                    "targets": ["re:.*\\.layers\\.(?:0|1)\\.mlp\\.experts"],
                    "weights": {"num_bits": 6, "group_size": 256},
                },
            },
        }
    )
    experts = object.__new__(RoutedExperts)

    assert config._scheme_for(experts, "model.layers.2.mlp.experts").num_bits == 4
    assert config._scheme_for(experts, "model.layers.0.mlp.experts").num_bits == 6


@pytest.mark.parametrize(
    ("checkpoint_name", "expected"),
    (
        (
            "experts.gate_up_proj_packed",
            (
                ("experts.w13_weight_packed", "w1"),
                ("experts.w13_weight_packed", "w3"),
            ),
        ),
        (
            "experts.gate_up_proj_scale",
            (
                ("experts.w13_weight_scale", "w1"),
                ("experts.w13_weight_scale", "w3"),
            ),
        ),
        ("experts.down_proj_packed", (("experts.w2_weight_packed", "w2"),)),
        (
            "experts.7.gate_proj.weight_packed",
            (("experts.w13_weight_packed", "w1"),),
        ),
        ("experts.7.down_proj.weight_a", (("experts.w2_weight_a", "w2"),)),
    ),
)
def test_qwen3_5_cubic_expert_checkpoint_names_map_to_runtime_params(
    checkpoint_name: str,
    expected: tuple[tuple[str, str], ...],
):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    mapping = RoutedExperts.build_expert_params_mapping(
        "gate_proj",
        "down_proj",
        "up_proj",
        num_experts=256,
        routed_experts_prefix="",
        include_fused=True,
    )
    matches = []
    for param_name, weight_name, _, shard_id in mapping:
        if weight_name in checkpoint_name:
            matches.append((checkpoint_name.replace(weight_name, param_name), shard_id))

    assert tuple(matches) == expected


def test_cubic_fused_expert_metadata_keeps_native_layout():
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    loaded = []

    def weight_loader(**kwargs):
        loaded.append(kwargs["loaded_weight"])
        return True

    experts = Mock()
    experts.layer_name = "experts"
    experts.moe_config.hidden_dim_unpadded = 6
    experts.cubic_fused_checkpoint_layout = True
    experts.get_expert_mapping.return_value = [
        ("w13_weight_scale", "gate_up_proj_scale", 0, "w1")
    ]
    experts.w13_weight_scale.weight_loader = weight_loader
    checkpoint_tensor = torch.arange(2 * 8 * 3).reshape(2, 8, 3)

    list(
        RoutedExperts.load_weights(
            experts,
            [("gate_up_proj_scale", checkpoint_tensor)],
        )
    )

    assert len(loaded) == 2
    torch.testing.assert_close(loaded[0], checkpoint_tensor[0, :4])
    torch.testing.assert_close(loaded[1], checkpoint_tensor[1, :4])
