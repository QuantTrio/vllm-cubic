# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-protected groupwise Cubic8 storage primitives for MLA KV cache."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import (
    CUBIC8_MLA_GROUP_SIZE,
    cubic8_mla_token_size_bytes,
)

_TRITON_GROUP_SIZE = tl.constexpr(64)
_TRITON_NUM_CURVES = tl.constexpr(4)
_CUBIC8_INVERSE_LUT_SIZE = 1 << 16
_CUBIC8_CURVES = ((1.0, 0.0), (0.5, 0.0), (0.5, 0.25), (0.75, -0.25))
_cubic8_inverse_lut_by_device: dict[tuple[str, int | None], torch.Tensor] = {}
CUBIC8_MLA_MATERIALIZE_TOKEN_CAP = 128 * 1024


def _build_cubic8_inverse_lut() -> torch.Tensor:
    normalized = torch.linspace(0.0, 1.0, _CUBIC8_INVERSE_LUT_SIZE, dtype=torch.float64)
    t = torch.arange(128, dtype=torch.float64) / 127.0
    tables = []
    for a, b in _CUBIC8_CURVES:
        levels = t * (a + t * (b + t * (1.0 - a - b)))
        thresholds = (levels[:-1] + levels[1:]) * 0.5
        tables.append(torch.searchsorted(thresholds, normalized).to(torch.uint8))
    return torch.stack(tables).contiguous()


_CUBIC8_INVERSE_LUT_CPU = _build_cubic8_inverse_lut()


def _get_cubic8_inverse_lut(device: torch.device) -> torch.Tensor:
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.accelerator.current_device_index()
    key = (device.type, index)
    lut = _cubic8_inverse_lut_by_device.get(key)
    if lut is None:
        target = torch.device(device.type, index) if index is not None else device
        lut = _CUBIC8_INVERSE_LUT_CPU.to(target, non_blocking=True)
        _cubic8_inverse_lut_by_device[key] = lut
    return lut


def cubic8_mla_materialize_token_count(
    batch_size: int, max_seq_len: int, page_size: int
) -> int:
    """Padded active-token count used by the bounded decode workspace."""
    request_stride = ((max_seq_len + page_size - 1) // page_size) * page_size
    return batch_size * request_stride


def cubic8_mla_max_materialize_tokens(
    max_batch_size: int, max_seq_len: int, page_size: int
) -> int:
    """Largest page-aligned materialization admitted by this configuration."""
    configured_tokens = cubic8_mla_materialize_token_count(
        max_batch_size, max_seq_len, page_size
    )
    page_aligned_cap = (CUBIC8_MLA_MATERIALIZE_TOKEN_CAP // page_size) * page_size
    return min(configured_tokens, page_aligned_cap)


def select_cubic8_mla_decode_tactic(
    batch_size: int,
    num_heads: int,
    num_splits: int,
    latent_dim: int,
    sm_count: int,
) -> tuple[int, int, int, int]:
    """Choose the consumer tile without encoding a GPU model or fixed batch."""
    latent_groups = latent_dim // CUBIC8_MLA_GROUP_SIZE
    base_ctas = batch_size * triton.cdiv(num_heads, 16) * num_splits
    target_ctas = max(1, (sm_count * 7) // 8)
    output_groups = 1
    for candidate in (2, 4, 8):
        if candidate > latent_groups:
            break
        if base_ctas * latent_groups // candidate < target_ctas:
            break
        output_groups = candidate
    selected_ctas = base_ctas * latent_groups // output_groups
    if selected_ctas < max(1, sm_count // 2):
        return output_groups, 4, 8, 64
    return output_groups, 16, 8, 32


@triton.jit
def _cubic_value(t, a, b):
    return t * (a + t * (b + t * (1.0 - a - b)))


@triton.jit
def _cubic8_curve_parameters(candidate: tl.constexpr):
    if candidate == 0:
        return 1.0, 0.0
    if candidate == 1:
        return 0.5, 0.0
    if candidate == 2:
        return 0.5, 0.25
    return 0.75, -0.25


@triton.jit
def _load_cubic8_metadata(cache_ptr, scale_offset, curve_offset):
    scale_bits = (
        tl.load(cache_ptr + scale_offset).to(tl.uint32)
        | (tl.load(cache_ptr + scale_offset + 1).to(tl.uint32) << 8)
        | (tl.load(cache_ptr + scale_offset + 2).to(tl.uint32) << 16)
        | (tl.load(cache_ptr + scale_offset + 3).to(tl.uint32) << 24)
    )
    scale = scale_bits.to(tl.float32, bitcast=True)
    curve = tl.load(cache_ptr + curve_offset).to(tl.int32)
    a = tl.where(curve == 0, 1.0, tl.where(curve == 3, 0.75, 0.5))
    b = tl.where(curve == 2, 0.25, tl.where(curve == 3, -0.25, 0.0))
    return scale, a, b


@triton.jit
def _store_cubic8_metadata(cache_ptr, scale_offset, curve_offset, scale, curve):
    scale_bits = scale.to(tl.float32).to(tl.uint32, bitcast=True)
    tl.store(cache_ptr + scale_offset, (scale_bits & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + scale_offset + 1, ((scale_bits >> 8) & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + scale_offset + 2, ((scale_bits >> 16) & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + scale_offset + 3, ((scale_bits >> 24) & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + curve_offset, curve.to(tl.uint8))


@triton.jit
def _cubic8_mla_insert_kernel(
    latent_ptr,
    rope_ptr,
    cache_ptr,
    slot_mapping_ptr,
    inverse_lut_ptr,
    latent_stride,
    rope_stride,
    cache_block_stride,
    cache_token_stride,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LUT_SIZE: tl.constexpr,
):
    """Bounded online Cubic8 contract from cubic_online_cubic_a8_rmse.md.

    Every token/group selects one of four fixed curves, performs exactly one
    direct scale reduction followed by code reassignment, and retains the
    linear curve as a mandatory candidate. Curve parameters are table constants;
    nearest levels come from precomputed inverse tables.
    """
    token = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, _TRITON_GROUP_SIZE)
    dimensions = group * _TRITON_GROUP_SIZE + offsets
    semantic_dim: tl.constexpr = LATENT_DIM + ROPE_DIM
    valid = dimensions < semantic_dim
    values = tl.where(
        dimensions < LATENT_DIM,
        tl.load(
            latent_ptr + token * latent_stride + dimensions,
            mask=valid & (dimensions < LATENT_DIM),
            other=0.0,
        ),
        tl.load(
            rope_ptr + token * rope_stride + dimensions - LATENT_DIM,
            mask=valid & (dimensions >= LATENT_DIM),
            other=0.0,
        ),
    ).to(tl.float32)
    absolute = tl.abs(values)
    amax = tl.maximum(tl.max(tl.where(valid, absolute, 0.0), axis=0), 1.0e-30)
    best_loss = float("inf")
    best_scale = amax
    best_curve = 0
    best_codes = tl.zeros([_TRITON_GROUP_SIZE], tl.int32)
    for candidate in tl.static_range(_TRITON_NUM_CURVES):
        a, b = _cubic8_curve_parameters(candidate)
        lut_base = candidate * LUT_SIZE
        normalized = tl.maximum(0.0, tl.minimum(1.0, absolute / amax))
        lut_index = tl.extra.cuda.libdevice.rint(normalized * (LUT_SIZE - 1)).to(
            tl.int32
        )
        codes = tl.load(inverse_lut_ptr + lut_base + lut_index).to(tl.int32)
        t = codes.to(tl.float32) * (1.0 / 127.0)
        q = _cubic_value(t, a, b)
        denominator = tl.sum(tl.where(valid, q * q, 0.0), axis=0)
        scale = tl.maximum(
            tl.sum(tl.where(valid, absolute * q, 0.0), axis=0)
            / tl.maximum(denominator, 1.0e-30),
            1.0e-30,
        )
        normalized = tl.maximum(0.0, tl.minimum(1.0, absolute / scale))
        lut_index = tl.extra.cuda.libdevice.rint(normalized * (LUT_SIZE - 1)).to(
            tl.int32
        )
        codes = tl.load(inverse_lut_ptr + lut_base + lut_index).to(tl.int32)
        t = codes.to(tl.float32) * (1.0 / 127.0)
        q = _cubic_value(t, a, b)
        error = absolute - scale * q
        loss = tl.sum(tl.where(valid, error * error, 0.0), axis=0)
        improved = loss < best_loss
        best_loss = tl.where(improved, loss, best_loss)
        best_scale = tl.where(improved, scale, best_scale)
        best_curve = tl.where(improved, candidate, best_curve)
        best_codes = tl.where(improved, codes, best_codes)

    signed_codes = tl.where(values < 0.0, -best_codes, best_codes)
    slot = tl.load(slot_mapping_ptr + token)
    block = slot // BLOCK_SIZE
    in_block = slot % BLOCK_SIZE
    token_base = block * cache_block_stride + in_block * cache_token_stride
    tl.store(
        cache_ptr + token_base + dimensions,
        (signed_codes & 0xFF).to(tl.uint8),
        mask=(slot >= 0) & valid,
    )
    total_groups: tl.constexpr = semantic_dim // _TRITON_GROUP_SIZE
    scale_offset = token_base + semantic_dim + group * 4
    curve_offset = token_base + semantic_dim + total_groups * 4 + group
    if slot >= 0:
        _store_cubic8_metadata(
            cache_ptr, scale_offset, curve_offset, best_scale, best_curve
        )


@triton.jit
def _cubic8_mla_dequant_kernel(
    cache_ptr,
    output_ptr,
    cache_token_stride,
    output_token_stride,
    SEMANTIC_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, _TRITON_GROUP_SIZE)
    dimensions = group * _TRITON_GROUP_SIZE + offsets
    valid = dimensions < SEMANTIC_DIM
    token_base = token * cache_token_stride
    raw = tl.load(cache_ptr + token_base + dimensions, mask=valid, other=0).to(tl.int32)
    codes = tl.where(raw < 128, raw, raw - 256)
    groups: tl.constexpr = SEMANTIC_DIM // _TRITON_GROUP_SIZE
    scale, a, b = _load_cubic8_metadata(
        cache_ptr,
        token_base + SEMANTIC_DIM + group * 4,
        token_base + SEMANTIC_DIM + groups * 4 + group,
    )
    t = tl.abs(codes).to(tl.float32) / 127.0
    values = tl.where(codes < 0, -1.0, 1.0) * scale * _cubic_value(t, a, b)
    tl.store(
        output_ptr + token * output_token_stride + dimensions,
        values,
        mask=valid,
    )


@triton.jit
def _cubic8_mla_gather_kernel(
    cache_ptr,
    output_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    token_to_seq_ptr,
    seq_starts_ptr,
    cache_block_stride,
    cache_token_stride,
    output_token_stride,
    block_table_stride,
    PAGE_SIZE: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
):
    output_token = tl.program_id(0)
    batch = tl.load(token_to_seq_ptr + output_token)
    sequence_offset = output_token - tl.load(cu_seq_lens_ptr + batch)
    if HAS_SEQ_STARTS:
        sequence_offset += tl.load(seq_starts_ptr + batch)
    block = tl.load(
        block_table_ptr + batch * block_table_stride + sequence_offset // PAGE_SIZE
    )
    token_base = (
        block * cache_block_stride + (sequence_offset % PAGE_SIZE) * cache_token_stride
    )
    semantic_dim: tl.constexpr = LATENT_DIM + ROPE_DIM
    latent_groups: tl.constexpr = LATENT_DIM // _TRITON_GROUP_SIZE
    latent_offsets = tl.arange(0, LATENT_DIM)
    raw_latent = tl.load(cache_ptr + token_base + latent_offsets).to(tl.int32)
    group_offsets = tl.arange(0, latent_groups)
    total_groups: tl.constexpr = latent_groups + 1
    scale_offsets = token_base + semantic_dim + group_offsets * 4
    curve_offsets = token_base + semantic_dim + total_groups * 4 + group_offsets
    scale, a, b = _load_cubic8_metadata(cache_ptr, scale_offsets, curve_offsets)
    raw_grouped = tl.reshape(raw_latent, [latent_groups, _TRITON_GROUP_SIZE])
    latent = _decode_cubic_codes(raw_grouped, scale[:, None], a[:, None], b[:, None])
    latent = tl.reshape(latent, [LATENT_DIM])
    tl.store(
        output_ptr + output_token * output_token_stride + latent_offsets,
        latent,
    )

    rope_offsets = tl.arange(0, ROPE_DIM)
    raw_rope = tl.load(cache_ptr + token_base + LATENT_DIM + rope_offsets).to(tl.int32)
    rope_scale, rope_a, rope_b = _load_cubic8_metadata(
        cache_ptr,
        token_base + semantic_dim + latent_groups * 4,
        token_base + semantic_dim + total_groups * 4 + latent_groups,
    )
    rope = _decode_cubic_codes(raw_rope, rope_scale, rope_a, rope_b)
    tl.store(
        output_ptr + output_token * output_token_stride + LATENT_DIM + rope_offsets,
        rope,
    )


@triton.jit
def _cubic8_mla_materialize_kernel(
    cache_ptr,
    output_ptr,
    block_table_ptr,
    seq_lens_ptr,
    cache_block_stride,
    cache_token_stride,
    output_token_stride,
    block_table_stride,
    REQUEST_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    linear_token = tl.program_id(0)
    batch = linear_token // REQUEST_STRIDE
    position = linear_token % REQUEST_STRIDE
    valid_token = position < tl.load(seq_lens_ptr + batch)
    block = tl.load(
        block_table_ptr + batch * block_table_stride + position // PAGE_SIZE,
        mask=valid_token,
        other=0,
    )
    token_base = (
        block * cache_block_stride + (position % PAGE_SIZE) * cache_token_stride
    )
    semantic_dim: tl.constexpr = LATENT_DIM + ROPE_DIM
    latent_groups: tl.constexpr = LATENT_DIM // _TRITON_GROUP_SIZE
    latent_offsets = tl.arange(0, LATENT_DIM)
    raw_latent = tl.load(
        cache_ptr + token_base + latent_offsets,
        mask=valid_token,
        other=0,
    ).to(tl.int32)
    group_offsets = tl.arange(0, latent_groups)
    total_groups: tl.constexpr = latent_groups + 1
    scale_offsets = token_base + semantic_dim + group_offsets * 4
    curve_offsets = token_base + semantic_dim + total_groups * 4 + group_offsets
    scale, a, b = _load_cubic8_metadata(cache_ptr, scale_offsets, curve_offsets)
    raw_grouped = tl.reshape(raw_latent, [latent_groups, _TRITON_GROUP_SIZE])
    latent = _decode_cubic_codes(raw_grouped, scale[:, None], a[:, None], b[:, None])
    latent = tl.reshape(latent, [LATENT_DIM])
    output_base = linear_token * output_token_stride
    tl.store(
        output_ptr + output_base + latent_offsets,
        latent,
        mask=valid_token,
    )

    rope_offsets = tl.arange(0, ROPE_DIM)
    raw_rope = tl.load(
        cache_ptr + token_base + LATENT_DIM + rope_offsets,
        mask=valid_token,
        other=0,
    ).to(tl.int32)
    rope_scale, rope_a, rope_b = _load_cubic8_metadata(
        cache_ptr,
        token_base + semantic_dim + latent_groups * 4,
        token_base + semantic_dim + total_groups * 4 + latent_groups,
    )
    rope = _decode_cubic_codes(raw_rope, rope_scale, rope_a, rope_b)
    tl.store(
        output_ptr + output_base + LATENT_DIM + rope_offsets,
        rope,
        mask=valid_token,
    )


@triton.jit
def _fill_contiguous_block_table_kernel(
    table_ptr,
    table_stride,
    NUM_PAGES: tl.constexpr,
):
    batch = tl.program_id(0)
    page = tl.program_id(1)
    tl.store(table_ptr + batch * table_stride + page, batch * NUM_PAGES + page)


@triton.jit
def _decode_cubic_codes(raw, scale, a, b):
    codes = tl.where(raw < 128, raw, raw - 256).to(tl.int32)
    t = tl.abs(codes).to(tl.float32) / 127.0
    return tl.where(codes < 0, -1.0, 1.0) * scale * _cubic_value(t, a, b)


@triton.jit
def _cubic8_mla_decode_stage1(
    Q,
    Cache,
    sm_scale,
    BlockTable,
    SeqLens,
    MidOut,
    stride_block_table_b,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_token,
    stride_mid_b,
    stride_mid_h,
    stride_mid_split,
    NUM_HEADS: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    OUTPUT_GROUPS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    combined_block = tl.program_id(1)
    split = tl.program_id(2)
    latent_groups: tl.constexpr = LATENT_DIM // _TRITON_GROUP_SIZE
    total_groups: tl.constexpr = latent_groups + 1
    output_tiles: tl.constexpr = latent_groups // OUTPUT_GROUPS
    output_tile = combined_block % output_tiles
    head_block = combined_block // output_tiles
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < NUM_HEADS
    output_width: tl.constexpr = OUTPUT_GROUPS * _TRITON_GROUP_SIZE
    output_offsets = tl.arange(0, output_width)
    output_dimensions = output_tile * output_width + output_offsets
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_rope = tl.load(
        Q
        + batch * stride_qb
        + heads[:, None] * stride_qh
        + LATENT_DIM
        + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
        cache_modifier=".ca",
    )
    seq_len = tl.load(SeqLens + batch)
    tokens_per_split = tl.cdiv(seq_len, NUM_SPLITS)
    split_start = tokens_per_split * split
    split_end = tl.minimum(split_start + tokens_per_split, seq_len)
    e_max = tl.full([BLOCK_H], -float("inf"), tl.float32)
    e_sum = tl.zeros([BLOCK_H], tl.float32)
    accumulator = tl.zeros([BLOCK_H, output_width], tl.float32)

    if split_end > split_start:
        for start_n in tl.range(split_start, split_end, BLOCK_N):
            token_offsets = start_n + tl.arange(0, BLOCK_N)
            token_mask = token_offsets < split_end
            pages = tl.load(
                BlockTable + batch * stride_block_table_b + token_offsets // PAGE_SIZE,
                mask=token_mask,
                other=0,
                cache_modifier=".ca",
            ).to(tl.int64)
            token_base = (
                pages * stride_cache_block
                + (token_offsets % PAGE_SIZE) * stride_cache_token
            )

            scores = tl.zeros([BLOCK_H, BLOCK_N], tl.float32)
            group_elements = tl.arange(0, _TRITON_GROUP_SIZE)
            for group in tl.static_range(latent_groups):
                dimensions = group * _TRITON_GROUP_SIZE + group_elements
                raw_group = tl.load(
                    Cache + token_base[None, :] + dimensions[:, None],
                    mask=token_mask[None, :],
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                scale_meta = token_base + LATENT_DIM + ROPE_DIM + group * 4
                curve_meta = (
                    token_base + LATENT_DIM + ROPE_DIM + total_groups * 4 + group
                )
                group_scale, group_a, group_b = _load_cubic8_metadata(
                    Cache, scale_meta, curve_meta
                )
                key_group = _decode_cubic_codes(
                    raw_group,
                    group_scale[None, :],
                    group_a[None, :],
                    group_b[None, :],
                ).to(Q.dtype.element_ty)
                query_group = tl.load(
                    Q
                    + batch * stride_qb
                    + heads[:, None] * stride_qh
                    + dimensions[None, :],
                    mask=head_mask[:, None],
                    other=0.0,
                    cache_modifier=".ca",
                )
                scores += tl.dot(query_group, key_group)

            raw_rope = tl.load(
                Cache + token_base[None, :] + LATENT_DIM + rope_offsets[:, None],
                mask=token_mask[None, :],
                other=0,
                cache_modifier=".cg",
            ).to(tl.int32)
            rope_group: tl.constexpr = total_groups - 1
            rope_scale_meta = token_base + LATENT_DIM + ROPE_DIM + rope_group * 4
            rope_curve_meta = (
                token_base + LATENT_DIM + ROPE_DIM + total_groups * 4 + rope_group
            )
            rope_scale, rope_a, rope_b = _load_cubic8_metadata(
                Cache, rope_scale_meta, rope_curve_meta
            )
            rope = _decode_cubic_codes(
                raw_rope,
                rope_scale[None, :],
                rope_a[None, :],
                rope_b[None, :],
            ).to(Q.dtype.element_ty)

            scores += tl.dot(q_rope, rope)
            scores *= sm_scale
            scores = tl.where(
                head_mask[:, None] & token_mask[None, :], scores, -float("inf")
            )
            new_max = tl.maximum(tl.max(scores, axis=1), e_max)
            old_scale = tl.exp(e_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            accumulator *= old_scale[:, None]
            raw_value_flat = tl.load(
                Cache + token_base[None, :] + output_dimensions[:, None],
                mask=token_mask[None, :],
                other=0,
                cache_modifier=".ca",
            ).to(tl.int32)
            output_group_offsets = tl.arange(0, OUTPUT_GROUPS)
            value_scale_offset = (
                token_base[None, :]
                + LATENT_DIM
                + ROPE_DIM
                + (output_tile * OUTPUT_GROUPS + output_group_offsets[:, None]) * 4
            )
            value_curve_offset = (
                token_base[None, :]
                + LATENT_DIM
                + ROPE_DIM
                + total_groups * 4
                + output_tile * OUTPUT_GROUPS
                + output_group_offsets[:, None]
            )
            value_scale, value_a, value_b = _load_cubic8_metadata(
                Cache, value_scale_offset, value_curve_offset
            )
            raw_value = tl.reshape(
                raw_value_flat, [OUTPUT_GROUPS, _TRITON_GROUP_SIZE, BLOCK_N]
            )
            value_grouped = _decode_cubic_codes(
                raw_value,
                value_scale[:, None, :],
                value_a[:, None, :],
                value_b[:, None, :],
            )
            value_groups = tl.reshape(value_grouped, [output_width, BLOCK_N]).to(
                Q.dtype.element_ty
            )
            accumulator += tl.dot(
                probabilities.to(value_groups.dtype), tl.trans(value_groups)
            )
            e_sum = e_sum * old_scale + tl.sum(probabilities, axis=1)
            e_max = new_max

        mid_offsets = (
            batch * stride_mid_b
            + heads[:, None] * stride_mid_h
            + split * stride_mid_split
            + output_dimensions[None, :]
        )
        tl.store(
            MidOut + mid_offsets,
            accumulator / e_sum[:, None],
            mask=head_mask[:, None],
        )
        if output_tile == 0:
            tl.store(
                MidOut
                + batch * stride_mid_b
                + heads * stride_mid_h
                + split * stride_mid_split
                + LATENT_DIM,
                e_max + tl.log(e_sum),
                mask=head_mask,
            )


def concat_and_cache_mla_cubic8(
    latent: torch.Tensor,
    rope: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write the production ``cubic8`` dtype without iterative fitting."""
    rope = rope.reshape(rope.shape[0], -1)
    latent = latent.reshape(latent.shape[0], -1)
    semantic_dim = latent.shape[1] + rope.shape[1]
    expected = cubic8_mla_token_size_bytes(semantic_dim)
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != expected:
        raise ValueError(
            "Cubic8 MLA cache requires uint8 storage with token stride "
            f"{expected}; received dtype={kv_cache.dtype}, "
            f"shape={tuple(kv_cache.shape)}"
        )
    if semantic_dim % CUBIC8_MLA_GROUP_SIZE != 0:
        raise NotImplementedError(
            "Cubic8 MLA requires semantic head size divisible by "
            f"{CUBIC8_MLA_GROUP_SIZE}; received {semantic_dim}."
        )
    inverse_lut = _get_cubic8_inverse_lut(latent.device)
    grid = (latent.shape[0], triton.cdiv(semantic_dim, CUBIC8_MLA_GROUP_SIZE))
    _cubic8_mla_insert_kernel[grid](
        latent,
        rope,
        kv_cache,
        slot_mapping,
        inverse_lut,
        latent.stride(0),
        rope.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        LATENT_DIM=latent.shape[1],
        ROPE_DIM=rope.shape[1],
        BLOCK_SIZE=kv_cache.shape[1],
        LUT_SIZE=_CUBIC8_INVERSE_LUT_SIZE,
        num_warps=1,
    )


def dequantize_cubic8_mla_cache(
    kv_cache: torch.Tensor, semantic_dim: int, output_dtype: torch.dtype
) -> torch.Tensor:
    """Materialize a contiguous cache for reference tests and fallback paths."""
    expected = cubic8_mla_token_size_bytes(semantic_dim)
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != expected:
        raise ValueError("Invalid Cubic8 MLA cache layout.")
    flat = kv_cache.reshape(-1, expected)
    output = torch.empty(
        (flat.shape[0], semantic_dim), dtype=output_dtype, device=kv_cache.device
    )
    grid = (flat.shape[0], triton.cdiv(semantic_dim, CUBIC8_MLA_GROUP_SIZE))
    _cubic8_mla_dequant_kernel[grid](
        flat,
        output,
        flat.stride(0),
        output.stride(0),
        SEMANTIC_DIM=semantic_dim,
        num_warps=1,
    )
    return output.reshape(*kv_cache.shape[:-1], semantic_dim)


def gather_cubic8_mla_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather paged Cubic8 entries and dequantize to a model-dtype workspace."""
    semantic_dim = dst.shape[-1]
    expected = cubic8_mla_token_size_bytes(semantic_dim)
    if src_cache.dtype != torch.uint8 or src_cache.shape[-1] != expected:
        raise ValueError("Invalid Cubic8 MLA gather cache layout.")
    if num_tokens == 0:
        return
    latent_dim = semantic_dim - CUBIC8_MLA_GROUP_SIZE
    if latent_dim not in (256, 512):
        raise NotImplementedError(
            "Cubic8 MLA gather supports a 256/512 latent plus 64 RoPE; "
            f"received semantic_dim={semantic_dim}."
        )
    _cubic8_mla_gather_kernel[(num_tokens,)](
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        seq_starts if seq_starts is not None else cu_seq_lens,
        src_cache.stride(0),
        src_cache.stride(1),
        dst.stride(0),
        block_table.stride(0),
        PAGE_SIZE=src_cache.shape[1],
        LATENT_DIM=latent_dim,
        ROPE_DIM=CUBIC8_MLA_GROUP_SIZE,
        HAS_SEQ_STARTS=seq_starts is not None,
        num_warps=8,
    )


def materialize_cubic8_mla_decode_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    contiguous_block_table: torch.Tensor,
    source_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize active Cubic8 pages into a contiguous model-dtype cache."""
    semantic_dim = dst.shape[-1]
    latent_dim = semantic_dim - CUBIC8_MLA_GROUP_SIZE
    if latent_dim not in (256, 512):
        raise NotImplementedError(
            "Cubic8 MLA materialization supports latent dimensions 256/512."
        )
    request_stride = (
        (max_seq_len + src_cache.shape[1] - 1) // src_cache.shape[1]
    ) * src_cache.shape[1]
    required_tokens = cubic8_mla_materialize_token_count(
        seq_lens.shape[0], max_seq_len, src_cache.shape[1]
    )
    if dst.shape[0] < required_tokens:
        raise ValueError(
            f"Cubic8 MLA workspace has {dst.shape[0]} tokens, requires "
            f"{required_tokens}."
        )
    num_pages = request_stride // src_cache.shape[1]
    if contiguous_block_table.shape[0] < seq_lens.shape[0] or (
        contiguous_block_table.shape[1] < num_pages
    ):
        raise ValueError("Cubic8 MLA contiguous block table workspace is too small.")
    output = dst[:required_tokens]
    _cubic8_mla_materialize_kernel[(required_tokens,)](
        src_cache,
        output,
        source_block_table,
        seq_lens,
        src_cache.stride(0),
        src_cache.stride(1),
        output.stride(0),
        source_block_table.stride(0),
        REQUEST_STRIDE=request_stride,
        PAGE_SIZE=src_cache.shape[1],
        LATENT_DIM=latent_dim,
        ROPE_DIM=CUBIC8_MLA_GROUP_SIZE,
        num_warps=8,
    )
    table = contiguous_block_table[: seq_lens.shape[0], :num_pages]
    _fill_contiguous_block_table_kernel[(seq_lens.shape[0], num_pages)](
        table,
        table.stride(0),
        NUM_PAGES=num_pages,
        num_warps=1,
    )
    return output.reshape(-1, src_cache.shape[1], semantic_dim), table


def cubic8_mla_decode_attention_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mid_output: torch.Tensor,
    num_splits: int,
    sm_scale: float,
    latent_dim: int,
    output_groups: int | None = None,
    block_h: int | None = None,
    num_warps: int | None = None,
    block_n: int | None = None,
) -> None:
    """BF16/FP16-query MLA decode directly from the Cubic8 paged cache."""
    from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

    semantic_dim = q.shape[-1]
    rope_dim = semantic_dim - latent_dim
    if latent_dim not in (256, 512) or rope_dim != 64:
        raise NotImplementedError(
            "Cubic8 MLA decode currently supports latent dimensions 256/512 "
            f"plus 64 RoPE dimensions; received {latent_dim}+{rope_dim}."
        )
    expected = cubic8_mla_token_size_bytes(semantic_dim)
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != expected:
        raise ValueError("Invalid Cubic8 MLA decode cache layout.")
    latent_groups = latent_dim // CUBIC8_MLA_GROUP_SIZE
    if output_groups is None or block_h is None or num_warps is None or block_n is None:
        properties = torch.cuda.get_device_properties(q.device)
        (
            automatic_output_groups,
            automatic_block_h,
            automatic_num_warps,
            automatic_block_n,
        ) = select_cubic8_mla_decode_tactic(
            q.shape[0],
            q.shape[1],
            num_splits,
            latent_dim,
            properties.multi_processor_count,
        )
        if output_groups is None:
            output_groups = automatic_output_groups
        if block_h is None:
            block_h = automatic_block_h
        if num_warps is None:
            num_warps = automatic_num_warps
        if block_n is None:
            block_n = automatic_block_n
    if block_n not in (16, 32, 64):
        raise ValueError(f"Invalid Cubic8 MLA block_n={block_n}.")
    if block_h not in (2, 4, 8, 16):
        raise ValueError(f"Invalid Cubic8 MLA block_h={block_h}.")
    if num_warps not in (4, 8):
        raise ValueError(f"Invalid Cubic8 MLA num_warps={num_warps}.")
    if output_groups not in (1, 2, 4, 8) or latent_groups % output_groups != 0:
        raise ValueError(
            f"Invalid Cubic8 MLA output_groups={output_groups} for "
            f"latent_dim={latent_dim}."
        )
    grid = (
        q.shape[0],
        triton.cdiv(q.shape[1], block_h) * (latent_groups // output_groups),
        num_splits,
    )
    _cubic8_mla_decode_stage1[grid](
        q,
        kv_cache,
        sm_scale,
        block_table,
        seq_lens,
        mid_output,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        NUM_HEADS=q.shape[1],
        LATENT_DIM=latent_dim,
        ROPE_DIM=rope_dim,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        OUTPUT_GROUPS=output_groups,
        NUM_SPLITS=num_splits,
        PAGE_SIZE=kv_cache.shape[1],
        num_warps=num_warps,
        num_stages=2,
    )
    block_dv = triton.next_power_of_2(latent_dim)
    _fwd_kernel_stage2[(q.shape[0], q.shape[1])](
        mid_output,
        output,
        lse,
        seq_lens,
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_splits,
        BLOCK_DV=block_dv,
        Lv=latent_dim,
        num_warps=4,
        num_stages=2,
    )


def cubic8_mla_materialized_decode_attention_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    source_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mid_output: torch.Tensor,
    materialize_workspace: torch.Tensor,
    block_table_workspace: torch.Tensor,
    num_splits: int,
    sm_scale: float,
    latent_dim: int,
    max_seq_len: int,
    cache_scale: torch.Tensor,
) -> None:
    """Materialize a bounded active cache, then use production BF16 MLA."""
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    pages, contiguous_block_table = materialize_cubic8_mla_decode_cache(
        kv_cache,
        materialize_workspace,
        block_table_workspace,
        source_block_table,
        seq_lens,
        max_seq_len,
    )
    decode_attention_fwd(
        q,
        pages.unsqueeze(2),
        pages[..., :latent_dim].unsqueeze(2),
        output,
        lse,
        contiguous_block_table,
        seq_lens,
        mid_output,
        num_splits,
        sm_scale,
        kv_cache.shape[1],
        k_scale=cache_scale,
        v_scale=cache_scale,
        is_mla=True,
    )
