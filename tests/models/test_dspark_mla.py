# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.models.kimi_k3.nvidia.dspark_mla import ReplicatedDSparkMarkovHead


@pytest.mark.cpu_test
def test_k3_dspark_config_is_recognized_as_mla():
    config = ModelConfig(
        model="/team/llm_models/Inferact/Kimi-K3-DSpark",
        tokenizer="/team/llm_models/Inferact/Kimi-K3-DSpark",
        trust_remote_code=True,
    )

    assert config.use_mla
    assert config.is_deepseek_mla
    assert config.get_head_size() == 576


@pytest.mark.cpu_test
def test_dspark_markov_head_replication_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    sharded = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert sharded.markov_w1.tp_size == 8
    assert sharded.markov_w2.tp_size == 8

    replicated = ReplicatedDSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert isinstance(replicated, DSparkMarkovHead)
    assert replicated.markov_w1.tp_size == 1
    assert replicated.markov_w2.tp_size == 1
    assert replicated.markov_w1.weight.shape == (128, 8)
    assert replicated.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    processor = LogitsProcessor(128)
    monkeypatch.setattr(processor, "_gather_logits", fail_collective)

    markov_embed = replicated.embed(torch.tensor([1, 2]))
    bias = replicated.bias(markov_embed, processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)
