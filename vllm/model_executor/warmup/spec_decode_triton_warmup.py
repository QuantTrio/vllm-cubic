# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _warm_eagle_bookkeeping(model_runner: "GPUModelRunner") -> None:
    from vllm.v1.sample.rejection_sampler import (
        PLACEHOLDER_TOKEN_ID,
        expand_batch_to_tokens,
        rejection_greedy_sample_kernel,
    )
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        next_power_of_2,
    )

    spec_config = model_runner.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return

    device = model_runner.device
    num_spec_tokens = spec_config.num_speculative_tokens
    discard = torch.zeros(1, dtype=torch.bool, device=device)
    backup = model_runner.drafter.backup_next_token_ids.gpu[:1]
    next_tokens = torch.empty(1, dtype=torch.int32, device=device)
    valid_counts = torch.empty(1, dtype=torch.int32, device=device)
    for num_sampled_tokens in {num_spec_tokens, num_spec_tokens + 1}:
        sampled = torch.zeros(
            (1, num_sampled_tokens), dtype=torch.int32, device=device
        )
        eagle_prepare_next_token_padded_kernel[(1,)](
            sampled,
            discard,
            backup,
            next_tokens,
            valid_counts,
            model_runner.model_config.get_vocab_size(),
            num_sampled_tokens,
            1,
            sampled.stride(0),
            BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled_tokens),
        )

    num_sampled_tokens = num_spec_tokens + 1

    cu_draft = torch.tensor([num_spec_tokens], dtype=torch.int32, device=device)
    query_start = torch.tensor(
        [0, num_sampled_tokens], dtype=torch.int32, device=device
    )
    token_indices = torch.empty(1, dtype=torch.int32, device=device)
    rejected_counts = torch.empty(1, dtype=torch.int32, device=device)
    eagle_prepare_inputs_padded_kernel[(1,)](
        cu_draft,
        valid_counts,
        query_start,
        token_indices,
        rejected_counts,
        1,
    )

    draft_ids = torch.zeros(num_spec_tokens, dtype=torch.int32, device=device)
    target_argmax = torch.zeros(num_spec_tokens, dtype=torch.int64, device=device)
    bonus = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.full(
        (1, num_sampled_tokens),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    rejection_greedy_sample_kernel[(1,)](
        output,
        cu_draft,
        draft_ids,
        target_argmax,
        bonus,
        None,
        num_spec_tokens,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    cu_num_tokens = torch.tensor(
        [num_sampled_tokens], dtype=torch.int32, device=device
    )
    for dtype in (torch.float32, torch.int32):
        expand_batch_to_tokens(
            torch.ones(1, dtype=dtype, device=device),
            cu_num_tokens,
            num_sampled_tokens,
            replace_from=0,
            replace_to=1,
        )


def _warm_mamba_state_migration(model_runner: "GPUModelRunner") -> None:
    from vllm.v1.kv_cache_interface import MambaSpec

    spec_config = model_runner.speculative_config
    if (
        spec_config is None
        or not model_runner.model_config.is_hybrid
        or model_runner.cache_config.mamba_cache_mode != "align"
    ):
        return

    mamba_specs = [
        group.kv_cache_spec
        for group in model_runner.kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    if not mamba_specs:
        return

    mamba_group_ids = [
        index
        for index, group in enumerate(model_runner.kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    bufs = model_runner._get_mamba_bufs()
    context = bufs.postprocess_align
    assert context is not None
    assert context.mamba_state_idx_buf is not None
    assert context.num_scheduled_tokens_buf is not None
    assert context.num_computed_tokens_buf is not None
    assert context.num_draft_tokens_buf is not None
    assert context.precopy_src_col_buf is not None
    assert context.precopy_token_bias_buf is not None
    context.initialize_from_forward_context(
        model_runner.kv_cache_config,
        model_runner.compilation_config.static_forward_context,
        model_runner.model.get_mamba_state_copy_func(),
        [
            model_runner.input_batch.block_table[group_id].get_device_tensor(1)
            for group_id in mamba_group_ids
        ],
    )
    num_accepted = model_runner.num_accepted_tokens.gpu[:1]
    num_accepted.fill_(1)
    state_idx = context.mamba_state_idx_buf.gpu[:1]
    state_idx.fill_(-1)
    num_scheduled = context.num_scheduled_tokens_buf.gpu[:1]
    num_scheduled.fill_(1)
    num_computed = context.num_computed_tokens_buf.gpu[:1]
    num_computed.fill_(1)
    num_draft = context.num_draft_tokens_buf.gpu[:1]
    num_draft.fill_(1)
    context.run_fused_postprocess(
        num_reqs=1,
        num_accepted_tokens_gpu=num_accepted,
        mamba_state_idx_gpu=state_idx,
        num_scheduled_tokens_gpu=num_scheduled,
        num_computed_tokens_gpu=num_computed,
        num_draft_tokens_gpu=num_draft,
    )
    context.run_fused_precopy(
        num_reqs=1,
        state_idx_gpu=state_idx,
        src_col_gpu=context.precopy_src_col_buf.gpu[:1].fill_(-1),
        token_bias_gpu=context.precopy_token_bias_buf.gpu[:1].zero_(),
        idx_mapping=None,
    )


@torch.inference_mode()
def spec_decode_triton_warmup(model_runner: "GPUModelRunner") -> None:
    """Compile speculative bookkeeping kernels without mutating model state."""
    if model_runner.speculative_config is None:
        return
    logger.info_once("Warming up speculative decoding Triton kernels.")
    _warm_eagle_bookkeeping(model_runner)
    _warm_mamba_state_migration(model_runner)
    torch.accelerator.synchronize()
