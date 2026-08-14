# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _warm_eagle_bookkeeping(model_runner: "GPUModelRunner") -> None:
    from vllm.v1.sample.rejection_sampler import (
        PLACEHOLDER_TOKEN_ID,
        expand_kernel,
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
    num_sampled_tokens = num_spec_tokens + 1
    block_size_tokens = next_power_of_2(num_sampled_tokens)

    sampled = torch.zeros((1, num_sampled_tokens), dtype=torch.int32, device=device)
    discard = torch.zeros(1, dtype=torch.bool, device=device)
    backup = model_runner.drafter.backup_next_token_ids.gpu[:1]
    next_tokens = torch.empty(1, dtype=torch.int32, device=device)
    valid_counts = torch.empty(1, dtype=torch.int32, device=device)
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
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )

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

    expanded = torch.empty(num_sampled_tokens, dtype=torch.float32, device=device)
    expand_kernel[(1,)](
        expanded,
        torch.ones(1, dtype=torch.float32, device=device),
        torch.tensor([num_sampled_tokens], dtype=torch.int32, device=device),
        0,
        1,
        MAX_NUM_TOKENS=128,
    )


def _warm_mamba_state_migration(model_runner: "GPUModelRunner") -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.mamba_utils import (
        postprocess_mamba_fused_kernel,
        precopy_mamba_align_fused_kernel,
    )

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

    device = model_runner.device
    block_size = mamba_specs[0].block_size
    mamba_group_ids = [
        index
        for index, group in enumerate(model_runner.kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    block_table_stride = model_runner.input_batch.block_table[
        mamba_group_ids[0]
    ].get_device_tensor(1).stride(0)
    i32 = lambda value=0: torch.full((1,), value, dtype=torch.int32, device=device)
    i64 = lambda value=0: torch.full(
        (1,), value, dtype=torch.int64, device=device
    )
    bufs = model_runner._get_mamba_bufs()
    context = bufs.postprocess_align
    assert context is not None
    assert context.mamba_state_idx_buf is not None
    assert context.num_scheduled_tokens_buf is not None
    assert context.num_computed_tokens_buf is not None
    assert context.num_draft_tokens_buf is not None
    assert context.precopy_src_col_buf is not None
    assert context.precopy_token_bias_buf is not None
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
    block_table_ptrs = i64()
    state_base_addrs = i64()
    state_block_strides = i64()
    state_elem_sizes = i32()
    state_inner_sizes = i64()
    state_conv_widths = i32()
    state_group_indices = i32()
    state_dim_row_count = i32()
    state_dim_row_stride = i64()
    num_accepted_out = context.num_accepted_tokens_out[:1]

    postprocess_mamba_fused_kernel[(1, 1)](
        num_accepted,
        state_idx,
        num_scheduled,
        num_computed,
        num_draft,
        block_table_ptrs,
        block_table_stride,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
        num_accepted_out,
        None,
        block_table_stride,
        block_size=block_size,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
        HAS_IDX_MAPPING=False,
        PRECOMPUTED_NEW_COMPUTED=False,
    )
    precopy_mamba_align_fused_kernel[(1, 1)](
        state_idx,
        context.precopy_src_col_buf.gpu[:1].fill_(-1),
        context.precopy_token_bias_buf.gpu[:1].zero_(),
        block_table_ptrs,
        1,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
        None,
        1,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
        HAS_IDX_MAPPING=False,
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
