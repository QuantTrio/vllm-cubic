# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(90),
    reason="Native BF16-Q / FP8-KV FlashMLA requires SM90",
)


def test_fp8_q16_uses_piecewise_cudagraph_boundary():
    from vllm.v1.attention.backends.mla.flashmla import (
        FlashMLAMetadataBuilder,
    )

    spec = SimpleNamespace(cache_dtype_str="fp8_q16")
    assert (
        FlashMLAMetadataBuilder.get_cudagraph_support(SimpleNamespace(), spec)
        == AttentionCGSupport.NEVER
    )


@pytest.mark.parametrize("query_len", [1, 5])
@torch.inference_mode()
def test_native_fp8_q16_shuffled_pages_matches_reference(query_len: int):
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache_fp8_q16,
        get_mla_metadata_dense_fp8,
    )

    torch.manual_seed(17)
    device = torch.device(current_platform.device_type)
    batch, heads, dim, value_dim, page_size = 3, 12, 576, 512, 64
    lengths = torch.tensor([37, 129, 511], device=device, dtype=torch.int32)
    max_pages = math.ceil(int(lengths.max().item()) / page_size)
    num_pages = batch * max_pages + 5
    scale = torch.tensor([0.25], device=device, dtype=torch.float32)

    source = torch.randn(num_pages, page_size, dim, device=device, dtype=torch.bfloat16)
    cache = (source.float() / scale).to(torch.float8_e4m3fn).unsqueeze(2)
    dequant = cache.squeeze(2).float() * scale
    permutation = torch.randperm(num_pages, device=device, dtype=torch.int64)
    block_table = (
        permutation[: batch * max_pages].to(torch.int32).reshape(batch, max_pages)
    )
    query = torch.randn(
        batch, query_len, heads, dim, device=device, dtype=torch.bfloat16
    )
    scheduler, split_offsets = get_mla_metadata_dense_fp8(lengths, query_len * heads, 1)

    output, lse = flash_mla_with_kvcache_fp8_q16(
        query,
        cache,
        block_table,
        lengths,
        value_dim,
        scheduler,
        split_offsets,
        scale,
        softmax_scale=1 / math.sqrt(dim),
    )

    reference = []
    for request in range(batch):
        length = int(lengths[request].item())
        pages = block_table[request, : math.ceil(length / page_size)].long()
        kv = dequant[pages].reshape(-1, dim)[:length]
        scores = query[request].float() @ kv.T / math.sqrt(dim)
        reference.append(torch.softmax(scores, dim=-1) @ kv[:, :value_dim])
    reference = torch.stack(reference)

    torch.testing.assert_close(output.float(), reference, atol=6e-3, rtol=1.5e-2)
    assert torch.isfinite(lse).all()


@torch.inference_mode()
def test_absorbed_fp8_q16_prefill_context_matches_expanded_mla():
    from vllm.v1.attention.backends.mla.flashmla import FlashMLAImpl

    torch.manual_seed(29)
    device = torch.device(current_platform.device_type)
    query_len, heads = 5, 4
    latent_dim, nope_dim, rope_dim, value_dim = 512, 128, 64, 128
    semantic_dim, page_size, context_len = latent_dim + rope_dim, 64, 93
    num_pages = math.ceil(context_len / page_size) + 3
    scale = torch.tensor([0.125], device=device, dtype=torch.float32)

    source = torch.randn(
        num_pages, page_size, semantic_dim, device=device, dtype=torch.bfloat16
    )
    cache = (source.float() / scale).to(torch.float8_e4m3fn)
    block_table = torch.randperm(num_pages, device=device, dtype=torch.int64)[
        : math.ceil(context_len / page_size)
    ].to(torch.int32)[None, :]
    context_lens = torch.tensor([context_len], device=device, dtype=torch.int32)

    q_nope = torch.randn(
        query_len, heads, nope_dim, device=device, dtype=torch.bfloat16
    )
    q_pe = torch.randn(query_len, heads, rope_dim, device=device, dtype=torch.bfloat16)
    W_UK_T = (
        torch.randn(heads, nope_dim, latent_dim, device=device) / latent_dim**0.5
    ).to(torch.bfloat16)
    W_UV = (
        torch.randn(heads, latent_dim, value_dim, device=device) / latent_dim**0.5
    ).to(torch.bfloat16)
    q_latent = torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)
    q_absorbed = torch.cat((q_latent, q_pe), dim=-1)

    impl = object.__new__(FlashMLAImpl)
    impl._fp8_cache_only = True
    impl.kv_lora_rank = latent_dim
    impl.scale = 1 / math.sqrt(nope_dim + rope_dim)
    metadata = SimpleNamespace(
        query_lens=[query_len],
        context_lens=context_lens,
        context_lens_cpu=[context_len],
        block_table=block_table,
    )
    latent_output, lse = impl._forward_fp8_q16_prefill_context(
        q_absorbed, cache, metadata, scale
    )
    absorbed_output = torch.bmm(latent_output.transpose(0, 1), W_UV).transpose(0, 1)

    pages = block_table[0].long()
    dequant = cache[pages].reshape(-1, semantic_dim)[:context_len].float() * scale
    latent = dequant[:, :latent_dim].to(torch.bfloat16)
    rope = dequant[:, latent_dim:].to(torch.bfloat16)
    k_nope = torch.einsum("tl,hpl->thp", latent, W_UK_T).to(torch.bfloat16)
    value = torch.einsum("tl,hlv->thv", latent, W_UV).to(torch.bfloat16)
    key = torch.cat((k_nope, rope[:, None, :].expand(-1, heads, -1)), dim=-1)
    query = torch.cat((q_nope, q_pe), dim=-1)
    scores = torch.einsum("qhd,thd->qht", query.float(), key.float()) * impl.scale
    reference = torch.einsum(
        "qht,thv->qhv", torch.softmax(scores, dim=-1), value.float()
    )

    torch.testing.assert_close(
        absorbed_output.float(), reference, atol=2.5e-2, rtol=2.5e-2
    )
    assert torch.isfinite(lse).all()
