# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MLA prefill and decode epilogues for Kimi-K3.

Thin wrappers over the CUDA ops in
``csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu``, which
mirror ``fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_{bf16,fp8}_insert``.

- ``fused_mla_key_concat_kv_cache_insert`` (bf16): optionally apply RoPE,
  concat the full per-head key ``[k_nope | k_pe]`` into ``k_out``, and insert
  the latent ``[kv_c_normed | k_pe]`` into the paged cache.
- ``fused_mla_qkv_quant_kv_cache_fp8_insert`` (fp8): additionally quantize
  ``q``/``k``/``v`` to E4M3 with ``q_scale`` / ``k_scale`` / ``v_scale`` (the
  cache shares ``k_scale``, as in ``concat_and_cache_mla``).

The optional ``positions`` / ``cos_sin_cache`` pair enables GPT-J-style RoPE
inside the epilogue. Omitting both keeps the K3 NoPE fast path. The kernels use
Programmatic Dependent Launch to overlap the tail of the producing GEMMs on
sm_90+.
"""

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton


@triton.jit
def _insert_bf16_latent_cache_kernel(
    kv_c_ptr,
    k_pe_ptr,
    cache_ptr,
    slot_mapping_ptr,
    kv_lora_rank: tl.constexpr,
    rope_dim: tl.constexpr,
    entry_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    dimensions = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    slot = tl.load(slot_mapping_ptr + token)
    valid = (slot >= 0) & (dimensions < entry_dim)
    values = tl.where(
        dimensions < kv_lora_rank,
        tl.load(
            kv_c_ptr + token * kv_lora_rank + dimensions,
            mask=valid & (dimensions < kv_lora_rank),
            other=0.0,
        ),
        tl.load(
            k_pe_ptr + token * rope_dim + dimensions - kv_lora_rank,
            mask=valid & (dimensions >= kv_lora_rank),
            other=0.0,
        ),
    )
    tl.store(cache_ptr + slot * entry_dim + dimensions, values, mask=valid)


def _insert_bf16_latent_cache(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    ops.concat_and_cache_mla(
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        "auto",
        scale,
    )


def _apply_gptj_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    for _ in range(x.ndim - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x1 = x[..., ::2].float()
    x2 = x[..., 1::2].float()
    out1 = x1 * cos.float() - x2 * sin.float()
    out2 = x2 * cos.float() + x1 * sin.float()
    return torch.stack((out1, out2), dim=-1).flatten(-2).to(x.dtype)


def _decode_bf16_fallback(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    cache_scale: torch.Tensor,
) -> torch.Tensor:
    if positions is not None:
        assert cos_sin_cache is not None
        q_pe = _apply_gptj_rope(q_pe, positions, cos_sin_cache)
        k_pe = _apply_gptj_rope(k_pe, positions, cos_sin_cache)

    _insert_bf16_latent_cache(
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        cache_scale,
    )
    return torch.cat((ql_nope, q_pe), dim=-1)


def fused_mla_key_concat_kv_cache_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim], RoPE is applied in place
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + rope]
    slot_mapping: torch.Tensor,  # [Tp] int64
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
    cache_scale: torch.Tensor | None = None,
    cache_dtype: str = "auto",
) -> torch.Tensor:
    """Apply optional RoPE, concat K, and insert the paged latent.

    Returns the full key ``[Tp, H, qk_nope_head_dim + qk_rope_head_dim]``;
    optionally rotates ``q`` and writes ``kv_cache`` in place. Quantized cache
    writes use vLLM's generic cache op while q/k/v remain in the model dtype.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, qk_nope_head_dim = k_nope.shape
    qk_head_dim = qk_nope_head_dim + k_pe.shape[1]
    k_out = torch.empty(
        (tp, num_heads, qk_head_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    if tp == 0:
        return k_out

    if cache_dtype == "cubic8":
        if positions is not None:
            assert cos_sin_cache is not None
            rope_dim = k_pe.shape[-1]
            q[..., -rope_dim:] = _apply_gptj_rope(
                q[..., -rope_dim:], positions, cos_sin_cache
            )
            k_pe = _apply_gptj_rope(k_pe, positions, cos_sin_cache)
        from vllm.v1.attention.ops.cubic8_mla import (
            concat_and_cache_mla_cubic8,
        )

        concat_and_cache_mla_cubic8(kv_c_normed, k_pe, kv_cache, slot_mapping)
        return torch.cat((k_nope, k_pe[:, None, :].expand(-1, num_heads, -1)), dim=-1)

    op_name = "fused_kimi_k3_mla_key_concat_kv_cache_insert"
    if cache_dtype != "auto" or not hasattr(torch.ops._C, op_name):
        if cache_scale is None:
            cache_scale = torch.ones(1, dtype=torch.float32, device=q.device)
        if positions is not None:
            assert cos_sin_cache is not None
            rope_dim = k_pe.shape[-1]
            q[..., -rope_dim:] = _apply_gptj_rope(
                q[..., -rope_dim:], positions, cos_sin_cache
            )
            k_pe = _apply_gptj_rope(k_pe, positions, cos_sin_cache)

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            cache_dtype,
            cache_scale,
        )
        return torch.cat(
            (k_nope, k_pe[:, None, :].expand(-1, num_heads, -1)),
            dim=-1,
        )

    getattr(torch.ops._C, op_name)(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        k_out,
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return k_out


def fused_mla_key_concat_ds_mla_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim], RoPE is applied in place
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8 (fp8_ds_mla)
    slot_mapping: torch.Tensor,  # [Tp] int64
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> torch.Tensor:
    """Concat full K (bf16) and insert the latent in the fp8_ds_mla layout.

    The cache uses DeepSeek's 656-byte block-scaled layout (NoPE in 4 tiles of
    128 with per-tile dynamic fp8 scales, RoPE as bf16) -- self-scaling, so no
    scale argument. Returns the bf16 full key; optionally rotates ``q`` and
    writes ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, qk_nope_head_dim = k_nope.shape
    qk_head_dim = qk_nope_head_dim + k_pe.shape[1]
    k_out = torch.empty(
        (tp, num_heads, qk_head_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    if tp == 0:
        return k_out
    torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        k_out,
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return k_out


def fused_mla_qkv_quant_kv_cache_fp8_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim]
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    v: torch.Tensor,  # [Tp, H, v_head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + rope] fp8
    slot_mapping: torch.Tensor,  # [Tp] int64
    q_scale_inv: torch.Tensor,  # scalar fp32, 1 / q scale (attention query)
    k_scale_inv: torch.Tensor,  # scalar fp32, 1 / k scale (attention key)
    v_scale_inv: torch.Tensor,  # scalar fp32, 1 / v scale (attention value)
    cache_scale_inv: torch.Tensor,  # scalar fp32, 1 / kv scale (cache latent)
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize q/k/v to fp8 and insert the fp8 latent into the paged cache.

    The attention key ``k_fp8`` and the cache latent use *separate* scales
    (``k_scale_inv`` vs ``cache_scale_inv``): the cache must be quantized with
    ``_k_scale`` (read back by decode / context), while the prefill attention
    q/k/v currently stay unscaled (the prefill flash path does not dequantize).

    Returns ``(q_fp8, k_fp8, v_fp8)``; writes the fp8 ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, _ = q.shape
    qk_head_dim = q.shape[2]
    v_head_dim = v.shape[2]
    fp8 = torch.float8_e4m3fn
    q_fp8 = torch.empty((tp, num_heads, qk_head_dim), dtype=fp8, device=q.device)
    k_fp8 = torch.empty((tp, num_heads, qk_head_dim), dtype=fp8, device=q.device)
    v_fp8 = torch.empty((tp, num_heads, v_head_dim), dtype=fp8, device=q.device)
    if tp == 0:
        return q_fp8, k_fp8, v_fp8
    torch.ops._C.fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        kv_cache,
        slot_mapping,
        q_scale_inv,
        k_scale_inv,
        v_scale_inv,
        cache_scale_inv,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return q_fp8, k_fp8, v_fp8


def fused_mla_decode_q_concat_kv_cache_insert(
    ql_nope: torch.Tensor,  # [B, H, kv_lora_rank]  (BMM1 output, absorbed q)
    q_pe: torch.Tensor,  # [B, H, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [B, kv_lora_rank]
    k_pe: torch.Tensor,  # [B, qk_rope_head_dim] or [B, 1, qk_rope_head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, entry]
    slot_mapping: torch.Tensor,  # [B] int64
    *,
    ds_mla: bool = False,
    cache_only_fp8: bool = False,
    cubic8_cache: bool = False,
    q_scale_inv: torch.Tensor | None = None,  # scalar fp32, 1 / q scale
    cache_scale_inv: torch.Tensor | None = None,  # scalar fp32, 1 / kv scale
    bf16_cache_scale: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,  # [B] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> torch.Tensor:
    """Concat the latent decode query ``mqa_q = [ql_nope | q_pe]`` and insert the
    latent ``[kv_c_normed | k_pe]`` into the paged cache, in one launch (runs
    right before ``forward_mqa``).

    Dispatched by cache format:
      - bf16          -> bf16 mqa_q, bf16 cache
      - plain fp8     -> fp8 mqa_q (q_scale_inv), fp8 cache (cache_scale_inv)
      - cache-only fp8 -> bf16 mqa_q, fp8 cache (bf16_cache_scale)
      - fp8_ds_mla    -> bf16 mqa_q, 656B block-scaled cache
      - cubic8 MLA    -> bf16 mqa_q, per-token/group Cubic8 cache

    Returns ``mqa_q`` of shape ``[B, H, kv_lora_rank + qk_rope_head_dim]``;
    writes ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    b, num_heads, kv_lora_rank = ql_nope.shape
    entry = kv_lora_rank + q_pe.shape[-1]
    fp8_q = q_scale_inv is not None
    out_dtype = torch.float8_e4m3fn if fp8_q else ql_nope.dtype
    mqa_q = torch.empty((b, num_heads, entry), dtype=out_dtype, device=ql_nope.device)
    if b == 0:
        return mqa_q

    if cubic8_cache:
        if positions is not None:
            assert cos_sin_cache is not None
            q_pe = _apply_gptj_rope(q_pe, positions, cos_sin_cache)
            k_pe = _apply_gptj_rope(k_pe, positions, cos_sin_cache)
        from vllm.v1.attention.ops.cubic8_mla import (
            concat_and_cache_mla_cubic8,
        )

        concat_and_cache_mla_cubic8(kv_c_normed, k_pe, kv_cache, slot_mapping)
        mqa_q.copy_(torch.cat((ql_nope, q_pe), dim=-1))
    elif cache_only_fp8:
        assert bf16_cache_scale is not None
        if positions is not None:
            assert cos_sin_cache is not None
            q_pe = _apply_gptj_rope(q_pe, positions, cos_sin_cache)
            k_pe = _apply_gptj_rope(k_pe, positions, cos_sin_cache)
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            bf16_cache_scale,
        )
        mqa_q.copy_(torch.cat((ql_nope, q_pe), dim=-1))
    elif ds_mla:
        cache = (
            kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
        )
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            cache,
            slot_mapping,
            cache.shape[1],
            positions,
            cos_sin_cache,
        )
    elif fp8_q:
        assert cache_scale_inv is not None, "fp8 decode requires cache_scale_inv"
        cache = (
            kv_cache
            if kv_cache.dtype == torch.float8_e4m3fn
            else kv_cache.view(torch.float8_e4m3fn)
        )
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            cache,
            slot_mapping,
            q_scale_inv,
            cache_scale_inv,
            cache.shape[1],
            positions,
            cos_sin_cache,
        )
    else:
        op_name = "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
        if not hasattr(torch.ops._C, op_name):
            if bf16_cache_scale is None:
                bf16_cache_scale = torch.ones(
                    1,
                    dtype=torch.float32,
                    device=ql_nope.device,
                )
            return _decode_bf16_fallback(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                positions,
                cos_sin_cache,
                bf16_cache_scale,
            )
        getattr(torch.ops._C, op_name)(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            kv_cache,
            slot_mapping,
            kv_cache.shape[1],
            positions,
            cos_sin_cache,
        )
    return mqa_q
