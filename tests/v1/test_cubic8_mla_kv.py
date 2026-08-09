# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonMetadataBuilder,
)
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MLAAttentionSpec,
    cubic8_mla_token_size_bytes,
    get_kv_quant_mode,
)


def test_cubic8_mla_token_layout_sizes() -> None:
    # Kimi/DeepSeek-style 512 latent + 64 RoPE, and the other common MLA size.
    assert cubic8_mla_token_size_bytes(576) == 624
    assert cubic8_mla_token_size_bytes(320) == 352
    # The helper is generic and reserves one final partial group's metadata.
    assert cubic8_mla_token_size_bytes(577) == 633


def test_cubic8_mla_spec_uses_physical_byte_stride() -> None:
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=get_kv_quant_mode("cubic8"),
    )
    assert spec.kv_quant_mode == KVQuantMode.CUBIC8_GROUPWISE
    assert spec.real_page_size_bytes == 64 * 624
    assert MLACommonBackend.get_kv_cache_shape(
        7, 64, 1, 576, cache_dtype_str="cubic8"
    ) == (7, 64, 624)


def test_cubic8_mla_group_merge_preserves_physical_layout() -> None:
    """Capacity planning and backend shape must survive layer grouping."""
    specs = [
        MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            cache_dtype_str="cubic8",
            kv_quant_mode=KVQuantMode.CUBIC8_GROUPWISE,
            alignment=4096,
        )
        for _ in range(2)
    ]

    merged = specs[0].merge(specs)
    assert merged.kv_quant_mode == KVQuantMode.CUBIC8_GROUPWISE
    assert merged.cache_dtype_str == "cubic8"
    assert merged.alignment == 4096
    assert merged.real_page_size_bytes == 16 * 624
    assert merged.page_size_bytes == 3 * 4096


def test_cubic8_mla_group_merge_rejects_mixed_quantization_modes() -> None:
    cubic = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=KVQuantMode.CUBIC8_GROUPWISE,
    )
    mislabeled = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=KVQuantMode.NONE,
    )

    with pytest.raises(AssertionError, match="quantization mode"):
        cubic.merge([cubic, mislabeled])


def test_cubic8_mla_hybrid_page_planning_uses_packed_bytes() -> None:
    from vllm.v1.core.kv_cache_utils import (
        _estimate_max_model_len_from_groups,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    cubic = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=KVQuantMode.CUBIC8_GROUPWISE,
        indexes_kv_by_block_stride=True,
    )
    # A hybrid state smaller than the attention page is padded to the packed
    # Cubic page. It must not enlarge Cubic back to the semantic 576-value
    # layout or silently discard the per-group metadata bytes.
    mamba = MambaSpec(
        block_size=16,
        shapes=((5184,),),
        dtypes=(torch.uint8,),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        model_config=SimpleNamespace(max_model_len=1000),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
        ),
        kv_transfer_config=None,
    )

    groups = get_kv_cache_groups(vllm_config, {"mla": cubic, "mamba": mamba})
    assert len(groups) == 2
    assert {group.kv_cache_spec.page_size_bytes for group in groups} == {16 * 624}
    mamba_group = next(group for group in groups if group.layer_names == ["mamba"])
    assert mamba_group.kv_cache_spec.page_size_padded == 16 * 624

    available_memory = 10 * 16 * 624
    config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory)
    assert config.num_blocks == 10
    assert len(config.kv_cache_tensors) == 1
    assert config.kv_cache_tensors[0].size == available_memory
    assert set(config.kv_cache_tensors[0].shared_by) == {"mla", "mamba"}

    # Eleven shared pages hold ten 16-token MLA blocks plus the one resident
    # Mamba state page, so auto-fit must report exactly 160 tokens.
    assert (
        _estimate_max_model_len_from_groups(vllm_config, groups, 11 * 16 * 624) == 160
    )
    assert vllm_config.model_config.max_model_len == 1000


def test_cubic8_mla_padded_raw_pages_have_correct_physical_stride() -> None:
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=KVQuantMode.CUBIC8_GROUPWISE,
        page_size_padded=3 * 4096,
        indexes_kv_by_block_stride=True,
    )
    num_blocks = 3
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    view = _reshape_attention_kv_cache(
        raw,
        spec,
        (num_blocks, 16, 624),
        (0, 1, 2),
        num_blocks,
        None,
    )

    assert view.shape == (num_blocks, 16, 624)
    assert view.dtype == torch.uint8
    assert view.stride() == (spec.page_size_bytes, 624, 1)
    view[1, 0, 0] = 123
    assert raw.view(torch.uint8)[spec.page_size_bytes].item() == 123


@pytest.mark.parametrize(
    ("backend_name", "initial_block_size", "expected_block_size"),
    [("flashmla", 64, 1408), ("triton_mla", 16, 1408)],
)
def test_cubic8_mla_hybrid_alignment_uses_backend_granularity(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    initial_block_size: int,
    expected_block_size: int,
) -> None:
    """Hybrid alignment is derived from format bytes and backend constraints.

    The state geometry is the KDA sample shape at TP=8, but the alignment
    algorithm and assertion are model-name agnostic.
    """
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

    class HybridModel:
        @classmethod
        def get_mamba_state_shape_from_config(cls, _config):
            return ((4608, 3), (12, 128, 128))

        @classmethod
        def get_mamba_state_dtype_from_config(cls, _config):
            return (torch.bfloat16, torch.float32)

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *_args, **_kwargs: (HybridModel, "HybridModel"),
    )
    cache_config = SimpleNamespace(
        cache_dtype="cubic8",
        mamba_cache_mode="align",
        block_size=initial_block_size,
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
        kv_cache_dtype_skip_layers=None,
    )
    model_config = SimpleNamespace(
        use_mla=True,
        dtype=torch.bfloat16,
        architecture="HybridModel",
        get_num_kv_heads=lambda _parallel_config: 1,
        get_head_size=lambda: 576,
        get_mamba_chunk_size=lambda: 64,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        speculative_config=None,
    )
    backend = {
        "flashmla": FlashMLABackend,
        "triton_mla": TritonMLABackend,
    }[backend_name]

    current_platform._align_hybrid_block_size(vllm_config, backend)

    assert cache_config.block_size == expected_block_size
    assert cache_config.mamba_block_size == expected_block_size
    assert cache_config.mamba_page_size_padded == expected_block_size * 624


def test_cubic8_mla_dtype_is_query_protected() -> None:
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="cubic8"),
        attention_config=SimpleNamespace(use_prefill_query_quantization=True),
    )
    assert is_quantized_kv_cache("cubic8")
    assert kv_cache_dtype_str_to_dtype("cubic8", None) == torch.uint8
    assert (
        MLACommonMetadataBuilder.determine_prefill_query_data_type(
            config, torch.bfloat16, "cubic8"
        )
        == torch.bfloat16
    )


def test_cubic8_mla_uses_piecewise_cudagraph_boundary() -> None:
    from vllm.v1.attention.backends.mla.flashmla import (
        FlashMLAMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.triton_mla import (
        TritonMLAMetadataBuilder,
    )

    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="cubic8",
        kv_quant_mode=KVQuantMode.CUBIC8_GROUPWISE,
    )
    config = SimpleNamespace()
    assert (
        FlashMLAMetadataBuilder.get_cudagraph_support(config, spec)
        == AttentionCGSupport.NEVER
    )
    assert (
        TritonMLAMetadataBuilder.get_cudagraph_support(config, spec)
        == AttentionCGSupport.NEVER
    )


def test_cubic8_mla_decode_selector_is_shape_and_device_aware() -> None:
    from vllm.v1.attention.ops.cubic8_mla import (
        cubic8_mla_materialize_token_count,
        cubic8_mla_max_materialize_tokens,
        select_cubic8_mla_decode_tactic,
    )

    assert select_cubic8_mla_decode_tactic(1, 16, 2, 512, 132) == (1, 4, 8, 64)
    assert select_cubic8_mla_decode_tactic(16, 16, 2, 512, 132) == (2, 16, 8, 32)
    assert select_cubic8_mla_decode_tactic(16, 16, 4, 512, 132) == (4, 16, 8, 32)
    assert select_cubic8_mla_decode_tactic(64, 16, 2, 512, 132) == (8, 16, 8, 32)
    assert select_cubic8_mla_decode_tactic(64, 16, 2, 256, 108) == (4, 16, 8, 32)
    assert cubic8_mla_materialize_token_count(3, 65, 64) == 384
    assert cubic8_mla_max_materialize_tokens(3, 65, 64) == 384
    assert cubic8_mla_max_materialize_tokens(1024, 8192, 64) == 128 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
@torch.inference_mode()
def test_cubic8_mla_insert_gather_and_decode() -> None:
    from vllm.v1.attention.ops.cubic8_mla import (
        concat_and_cache_mla_cubic8,
        cubic8_mla_decode_attention_fwd,
        cubic8_mla_materialized_decode_attention_fwd,
        dequantize_cubic8_mla_cache,
        gather_cubic8_mla_cache,
        materialize_cubic8_mla_decode_cache,
    )

    torch.manual_seed(11)
    device = "cuda"
    latent_dim = 256
    semantic_dim = 320
    tokens = 97
    heads = 8
    num_splits = 2
    latent = torch.randn(tokens, latent_dim, device=device, dtype=torch.bfloat16)
    rope = torch.randn(tokens, 64, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(2, 64, 352, device=device, dtype=torch.uint8)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    concat_and_cache_mla_cubic8(latent, rope, cache, slots)

    materialized = dequantize_cubic8_mla_cache(cache, semantic_dim, torch.bfloat16)
    gathered = torch.empty(17, semantic_dim, device=device, dtype=torch.bfloat16)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, 17], device=device, dtype=torch.int32)
    token_to_seq = torch.zeros(17, device=device, dtype=torch.int32)
    seq_starts = torch.tensor([7], device=device, dtype=torch.int32)
    gather_cubic8_mla_cache(
        cache,
        gathered,
        block_table,
        cu_seq_lens,
        token_to_seq,
        17,
        seq_starts,
    )
    torch.testing.assert_close(gathered, materialized.reshape(-1, semantic_dim)[7:24])

    materialize_workspace = torch.empty(
        128, semantic_dim, device=device, dtype=torch.bfloat16
    )
    contiguous_table_workspace = torch.empty(1, 2, device=device, dtype=torch.int32)
    materialized_pages, contiguous_table = materialize_cubic8_mla_decode_cache(
        cache,
        materialize_workspace,
        contiguous_table_workspace,
        block_table,
        torch.tensor([tokens], device=device, dtype=torch.int32),
        tokens,
    )
    torch.testing.assert_close(
        materialized_pages.reshape(-1, semantic_dim)[:tokens],
        materialized.reshape(-1, semantic_dim)[:tokens],
    )
    torch.testing.assert_close(
        contiguous_table,
        torch.tensor([[0, 1]], device=device, dtype=torch.int32),
    )

    query = torch.randn(1, heads, semantic_dim, device=device, dtype=torch.bfloat16)
    output = torch.empty(1, heads, latent_dim, device=device, dtype=torch.bfloat16)
    lse = torch.empty(1, heads, device=device, dtype=torch.bfloat16)
    mid_output = torch.empty(
        1,
        heads,
        num_splits,
        latent_dim + 1,
        device=device,
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([tokens], device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(semantic_dim)
    cubic8_mla_decode_attention_fwd(
        query,
        cache,
        output,
        lse,
        block_table,
        seq_lens,
        mid_output,
        num_splits,
        scale,
        latent_dim,
    )
    values = materialized.reshape(-1, semantic_dim)[:tokens].float()
    scores = query[0].float() @ values.T * scale
    reference = scores.softmax(dim=-1) @ values[:, :latent_dim]
    torch.testing.assert_close(output[0].float(), reference, atol=6e-3, rtol=1.5e-2)
    assert torch.isfinite(lse).all()

    materialized_output = torch.empty_like(output)
    materialized_lse = torch.empty_like(lse)
    cubic8_mla_materialized_decode_attention_fwd(
        query,
        cache,
        materialized_output,
        materialized_lse,
        block_table,
        seq_lens,
        mid_output,
        materialize_workspace,
        contiguous_table_workspace,
        num_splits,
        scale,
        latent_dim,
        tokens,
        torch.ones(1, device=device, dtype=torch.float32),
    )
    torch.testing.assert_close(
        materialized_output[0].float(), reference, atol=6e-3, rtol=1.5e-2
    )
    assert torch.isfinite(materialized_lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
@torch.inference_mode()
def test_cubic8_mla_materialize_noncontiguous_variable_length_pages() -> None:
    """The fast decode path must preserve arbitrary paged-cache placement."""
    from vllm.v1.attention.ops.cubic8_mla import (
        concat_and_cache_mla_cubic8,
        cubic8_mla_materialized_decode_attention_fwd,
        dequantize_cubic8_mla_cache,
        materialize_cubic8_mla_decode_cache,
    )

    torch.manual_seed(19)
    device = "cuda"
    latent_dim = 256
    semantic_dim = 320
    page_size = 16
    seq_lens = torch.tensor([19, 11], device=device, dtype=torch.int32)
    block_table = torch.tensor([[3, 1], [4, 0]], device=device, dtype=torch.int32)
    total_tokens = int(seq_lens.sum().item())
    latent = torch.randn(total_tokens, latent_dim, device=device, dtype=torch.bfloat16)
    rope = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(5, page_size, 352, device=device, dtype=torch.uint8)

    slots: list[int] = []
    for batch, length in enumerate(seq_lens.tolist()):
        for position in range(length):
            slots.append(
                int(block_table[batch, position // page_size].item()) * page_size
                + position % page_size
            )
    concat_and_cache_mla_cubic8(
        latent,
        rope,
        cache,
        torch.tensor(slots, device=device, dtype=torch.int64),
    )
    physical = dequantize_cubic8_mla_cache(cache, semantic_dim, torch.bfloat16)

    request_stride = 32
    materialize_workspace = torch.empty(
        2 * request_stride, semantic_dim, device=device, dtype=torch.bfloat16
    )
    contiguous_table_workspace = torch.empty(2, 2, device=device, dtype=torch.int32)
    pages, contiguous_table = materialize_cubic8_mla_decode_cache(
        cache,
        materialize_workspace,
        contiguous_table_workspace,
        block_table,
        seq_lens,
        19,
    )
    assert pages.shape == (4, page_size, semantic_dim)
    torch.testing.assert_close(
        contiguous_table,
        torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32),
    )
    for batch, length in enumerate(seq_lens.tolist()):
        expected = torch.stack(
            [
                physical[
                    block_table[batch, position // page_size],
                    position % page_size,
                ]
                for position in range(length)
            ]
        )
        actual = pages.reshape(2, request_stride, semantic_dim)[batch, :length]
        torch.testing.assert_close(actual, expected)

    heads = 8
    num_splits = 2
    query = torch.randn(2, heads, semantic_dim, device=device, dtype=torch.bfloat16)
    output = torch.empty(2, heads, latent_dim, device=device, dtype=torch.bfloat16)
    lse = torch.empty(2, heads, device=device, dtype=torch.bfloat16)
    mid_output = torch.empty(
        2,
        heads,
        num_splits,
        latent_dim + 1,
        device=device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(semantic_dim)
    cubic8_mla_materialized_decode_attention_fwd(
        query,
        cache,
        output,
        lse,
        block_table,
        seq_lens,
        mid_output,
        materialize_workspace,
        contiguous_table_workspace,
        num_splits,
        scale,
        latent_dim,
        19,
        torch.ones(1, device=device, dtype=torch.float32),
    )
    for batch, length in enumerate(seq_lens.tolist()):
        values = pages.reshape(2, request_stride, semantic_dim)[batch, :length].float()
        scores = query[batch].float() @ values.T * scale
        reference = scores.softmax(dim=-1) @ values[:, :latent_dim]
        torch.testing.assert_close(
            output[batch].float(), reference, atol=6e-3, rtol=1.5e-2
        )
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
@torch.inference_mode()
def test_cubic8_mla_backend_dispatch_with_locked_workspace(monkeypatch) -> None:
    """Both advertised MLA backends must fit their pre-reserved workspace."""
    from vllm.v1.attention.backends.mla.flashmla import FlashMLAImpl
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl
    from vllm.v1.attention.ops.cubic8_mla import (
        concat_and_cache_mla_cubic8,
        cubic8_mla_materialize_token_count,
    )
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )

    torch.manual_seed(23)
    device = torch.device("cuda")
    batch = 2
    heads = 8
    latent_dim = 256
    semantic_dim = 320
    page_size = 16
    max_seq_len = 17
    seq_lens = torch.tensor([17, 9], device=device, dtype=torch.int32)
    block_table = torch.tensor([[2, 0], [3, 1]], device=device, dtype=torch.int32)
    total_tokens = int(seq_lens.sum().item())
    cache = torch.zeros(4, page_size, 352, device=device, dtype=torch.uint8)
    slots: list[int] = []
    for request, length in enumerate(seq_lens.tolist()):
        for position in range(length):
            slots.append(
                int(block_table[request, position // page_size].item()) * page_size
                + position % page_size
            )
    concat_and_cache_mla_cubic8(
        torch.randn(total_tokens, latent_dim, device=device, dtype=torch.bfloat16),
        torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16),
        cache,
        torch.tensor(slots, device=device, dtype=torch.int64),
    )

    query = torch.randn(batch, heads, semantic_dim, device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        max_seq_len=max_seq_len,
        decode=SimpleNamespace(block_table=block_table, seq_lens=seq_lens),
    )
    layer = SimpleNamespace(_k_scale=torch.ones(1, device=device, dtype=torch.float32))
    materialize_tokens = cubic8_mla_materialize_token_count(
        batch, max_seq_len, page_size
    )
    workspace_specs = [
        ((batch, heads, 1, latent_dim + 1), torch.float32),
        ((materialize_tokens, semantic_dim), torch.bfloat16),
        ((batch, materialize_tokens // batch // page_size), torch.int32),
    ]

    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        current_workspace_manager().get_simultaneous(*workspace_specs)
        lock_workspace()

        triton_impl = object.__new__(TritonMLAImpl)
        triton_impl.kv_cache_dtype = "cubic8"
        triton_impl.kv_lora_rank = latent_dim
        triton_impl.scale = 1.0 / math.sqrt(semantic_dim)
        triton_impl._sm_count = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
        triton_output, triton_lse = TritonMLAImpl.forward_mqa(
            triton_impl, query, cache, metadata, layer
        )

        flash_impl = object.__new__(FlashMLAImpl)
        flash_impl.kv_cache_dtype = "cubic8"
        flash_impl._cubic8_cache = True
        flash_impl.kv_lora_rank = latent_dim
        flash_impl.scale = triton_impl.scale
        flash_impl._sm_count = triton_impl._sm_count
        flash_output, flash_lse = FlashMLAImpl._forward_fp8_q16(
            flash_impl, query, cache, metadata, layer
        )

        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_lse = TritonMLAImpl.forward_mqa(
                triton_impl, query, cache, metadata, layer
            )
        graph.replay()
        torch.accelerator.synchronize()

        # Force the bounded materialization budget below this active set. This
        # exercises the allocation-free direct reader used by long contexts.
        import vllm.v1.attention.ops.cubic8_mla as cubic8_ops

        monkeypatch.setattr(cubic8_ops, "CUBIC8_MLA_MATERIALIZE_TOKEN_CAP", 0)
        triton_direct, triton_direct_lse = TritonMLAImpl.forward_mqa(
            triton_impl, query, cache, metadata, layer
        )
        flash_direct, flash_direct_lse = FlashMLAImpl._forward_fp8_q16(
            flash_impl, query, cache, metadata, layer
        )
    finally:
        reset_workspace_manager()

    torch.testing.assert_close(flash_output, triton_output)
    torch.testing.assert_close(flash_lse, triton_lse)
    torch.testing.assert_close(graph_output, triton_output)
    torch.testing.assert_close(graph_lse, triton_lse)
    torch.testing.assert_close(flash_direct, triton_direct)
    torch.testing.assert_close(flash_direct_lse, triton_direct_lse)
    torch.testing.assert_close(
        triton_direct.float(), triton_output.float(), atol=6e-3, rtol=1.5e-2
    )
    assert torch.isfinite(triton_output).all()
    assert torch.isfinite(triton_lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cubic8_mla_pages_are_opaque_for_copy_and_offload() -> None:
    """Prefix reuse/offload must copy the complete 624-byte token records."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.cubic8_mla import concat_and_cache_mla_cubic8

    torch.manual_seed(29)
    device = torch.device("cuda")
    blocks = 4
    page_size = 16
    tokens = blocks * page_size
    source = torch.zeros(blocks, page_size, 624, device=device, dtype=torch.uint8)
    concat_and_cache_mla_cubic8(
        torch.randn(tokens, 512, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, 64, device=device, dtype=torch.bfloat16),
        source,
        torch.arange(tokens, device=device, dtype=torch.int64),
    )
    block_bytes = source.element_size() * source.stride(0)
    mapping = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64, device="cpu")

    copied = torch.zeros_like(source)
    ops.swap_blocks(source, copied, block_bytes, mapping)
    torch.accelerator.synchronize()
    torch.testing.assert_close(copied[3], source[0])
    torch.testing.assert_close(copied[2], source[1])

    host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
    ops.swap_blocks(source, host, block_bytes, mapping)
    torch.accelerator.synchronize()
    torch.testing.assert_close(host[3], source[0].cpu())
    torch.testing.assert_close(host[2], source[1].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cubic8_mla_native_sm90_noncontiguous_pages() -> None:
    """The production FlashMLA path must consume physical Cubic8 records."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native Cubic8 FlashMLA currently requires SM90")

    from vllm.v1.attention.ops.cubic8_mla import (
        concat_and_cache_mla_cubic8,
        dequantize_cubic8_mla_cache,
    )
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache_cubic8,
        get_mla_metadata_dense_fp8,
    )

    torch.manual_seed(31)
    device = "cuda"
    batch, heads, page_size = 3, 12, 64
    latent_dim, semantic_dim = 512, 576
    seq_lens = torch.tensor([37, 129, 511], device=device, dtype=torch.int32)
    pages_per_seq = 8
    physical_pages = batch * pages_per_seq
    block_table = torch.randperm(
        physical_pages, device=device, dtype=torch.int32
    ).reshape(batch, pages_per_seq)
    cache = torch.zeros(
        physical_pages, page_size, 624, device=device, dtype=torch.uint8
    )

    token_count = int(seq_lens.sum().item())
    latent = torch.randn(token_count, latent_dim, device=device, dtype=torch.bfloat16)
    rope = torch.randn(token_count, 64, device=device, dtype=torch.bfloat16)
    slots: list[int] = []
    for batch_idx, length in enumerate(seq_lens.tolist()):
        for position in range(length):
            page = int(block_table[batch_idx, position // page_size].item())
            slots.append(page * page_size + position % page_size)
    concat_and_cache_mla_cubic8(
        latent,
        rope,
        cache,
        torch.tensor(slots, device=device, dtype=torch.int64),
    )

    query = torch.randn(batch, heads, semantic_dim, device=device, dtype=torch.bfloat16)
    scheduler, splits = get_mla_metadata_dense_fp8(seq_lens, heads, 1)
    scale = 1.0 / math.sqrt(semantic_dim)
    output, lse = flash_mla_with_kvcache_cubic8(
        query.unsqueeze(1),
        cache.unsqueeze(2),
        block_table,
        seq_lens,
        latent_dim,
        scheduler,
        splits,
        scale,
    )

    materialized = dequantize_cubic8_mla_cache(cache, semantic_dim, torch.bfloat16)
    references = []
    for batch_idx, length in enumerate(seq_lens.tolist()):
        page_ids = block_table[batch_idx, : math.ceil(length / page_size)].long()
        values = materialized[page_ids].reshape(-1, semantic_dim)[:length].float()
        scores = query[batch_idx].float() @ values.T * scale
        references.append(scores.softmax(dim=-1) @ values[:, :latent_dim])
    reference = torch.stack(references)
    torch.testing.assert_close(output[:, 0].float(), reference, atol=6e-3, rtol=1.5e-2)
    assert torch.isfinite(lse).all()
