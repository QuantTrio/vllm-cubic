# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device("cuda")

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).to(dtype).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_decode_matches_spec_update_exactly(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    batch, num_heads, num_value_heads = 8, 4, 8
    key_dim = value_dim = 128
    qkv_dim = 2 * num_heads * key_dim + num_value_heads * value_dim
    device = torch.device("cuda")

    mixed_qkv = torch.randn(batch, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(batch, num_value_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(num_value_heads, device=device, dtype=dtype)
    dt_bias = torch.randn_like(A_log)
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    initial = torch.randn(
        batch + 1,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    )
    packed_state = initial.clone()
    spec_state = initial.clone()
    packed_output = torch.empty(
        batch,
        1,
        num_value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    q, k, v = torch.split(
        mixed_qkv,
        [num_heads * key_dim, num_heads * key_dim, num_value_heads * value_dim],
        dim=-1,
    )
    q = q.view(batch, 1, num_heads, key_dim)
    k = k.view(batch, 1, num_heads, key_dim)
    v = v.view(batch, 1, num_value_heads, value_dim)

    spec_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=spec_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=packed_state,
        out=packed_output,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(packed_output, spec_output, rtol=0, atol=0)
    torch.testing.assert_close(packed_state, spec_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_spec_update_matches_successive_decode_steps_exactly(
    dtype: torch.dtype,
) -> None:
    """Multi-token verification preserves cache-dtype recurrence boundaries."""
    torch.manual_seed(7)
    batch, tokens, num_heads, num_value_heads = 1, 2, 4, 8
    key_dim = value_dim = 128
    device = torch.device("cuda")
    q = torch.randn(
        batch, tokens, num_heads, key_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch, tokens, num_value_heads, value_dim, device=device, dtype=dtype
    )
    a = torch.randn(batch * tokens, num_value_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(num_value_heads, device=device, dtype=dtype)
    dt_bias = torch.randn_like(A_log)
    initial = torch.randn(
        batch,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    )

    multi_output, multi_states = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=initial,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    sequential_state = initial
    sequential_outputs = []
    for index in range(tokens):
        output, sequential_state = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a[index : index + 1],
            b=b[index : index + 1],
            dt_bias=dt_bias,
            q=q[:, index : index + 1],
            k=k[:, index : index + 1],
            v=v[:, index : index + 1],
            initial_state=sequential_state,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        sequential_outputs.append(output)
    sequential_output = torch.cat(sequential_outputs, dim=1)

    torch.testing.assert_close(multi_output, sequential_output, rtol=0, atol=0)
    torch.testing.assert_close(multi_states[-1:], sequential_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 8])
def test_varlen_spec_update_matches_per_sequence_exactly(
    dtype: torch.dtype, batch: int
) -> None:
    """Packed multi-sequence verification must preserve sequence isolation."""
    torch.manual_seed(11)
    tokens, num_heads, num_value_heads = 2, 4, 8
    key_dim = value_dim = 128
    device = torch.device("cuda")
    total_tokens = batch * tokens

    q = torch.randn(
        1, total_tokens, num_heads, key_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1,
        total_tokens,
        num_value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(total_tokens, num_value_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(num_value_heads, device=device, dtype=dtype)
    dt_bias = torch.randn_like(A_log)
    cu_seqlens = torch.arange(
        0, total_tokens + 1, tokens, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(
        1, total_tokens + 1, device=device, dtype=torch.int32
    ).view(batch, tokens)
    num_accepted_tokens = torch.ones(batch, device=device, dtype=torch.int32)
    initial = torch.randn(
        total_tokens + 1,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    )
    packed_state = initial.clone()

    packed_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=packed_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    reference_outputs = []
    reference_states = []
    for sequence in range(batch):
        start = sequence * tokens
        end = start + tokens
        state_index = state_indices[sequence, 0]
        output, states = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a[start:end],
            b=b[start:end],
            dt_bias=dt_bias,
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            initial_state=initial[state_index : state_index + 1],
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        reference_outputs.append(output)
        reference_states.append(states)
    reference_output = torch.cat(reference_outputs, dim=1)
    reference_states = torch.cat(reference_states, dim=0)

    torch.testing.assert_close(packed_output, reference_output, rtol=0, atol=0)
    for sequence in range(batch):
        for token in range(tokens):
            state_index = state_indices[sequence, token]
            torch.testing.assert_close(
                packed_state[state_index],
                reference_states[sequence * tokens + token],
                rtol=0,
                atol=0,
            )
