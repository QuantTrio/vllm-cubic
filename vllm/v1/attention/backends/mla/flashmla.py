# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_cubic8,
    flash_mla_with_kvcache_fp8,
    flash_mla_with_kvcache_fp8_q16,
    get_mla_metadata,
    get_mla_metadata_dense_fp8,
    is_flashmla_dense_supported,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_MIN_WORK_PER_CACHE_ONLY_SPLIT = 512
_CACHE_ONLY_SPLIT_OCCUPANCY_MULTIPLIER = 2
_FP8_Q16_NATIVE_MAX_TOKEN_WORK = 32 * 1024


def _compute_cache_only_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    ideal = triton.next_power_of_2(
        max(1, max_seq_len // _MIN_WORK_PER_CACHE_ONLY_SPLIT)
    )
    return min(ideal, sm_count * _CACHE_ONLY_SPLIT_OCCUPANCY_MULTIPLIER)


class FlashMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_q16",
        "cubic8",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if use_sparse:
            from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

            return is_flashmla_sparse_supported()[1]
        else:
            from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported

            return is_flashmla_dense_supported()[1]


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    scheduler_metadata: FlashMLASchedMeta


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # ^ TODO(matt): tune this

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        if getattr(kv_cache_spec, "cache_dtype_str", None) == "cubic8":
            return AttentionCGSupport.NEVER
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        cache_dtype = getattr(kv_cache_spec, "cache_dtype_str", None)
        self.fp8_cache_only = cache_dtype == "fp8_q16"
        self.cubic8_cache = cache_dtype == "cubic8"
        self.native_cubic8_cache = cache_dtype == "cubic8"
        if self.fp8_cache_only or self.cubic8_cache:
            self.reorder_batch_threshold = 1
            cast(Any, self).query_len_support = QueryLenSupport.SINGLE_ONLY

        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None
        self.is_fp8_kvcache = is_quantized_kv_cache(
            vllm_config.cache_config.cache_dtype
        ) and not (self.fp8_cache_only or self.cubic8_cache)
        self.use_native_compressed_scheduler = (
            self.is_fp8_kvcache or self.fp8_cache_only or self.native_cubic8_cache
        )

        num_sms = num_compute_units(self.device.index)

        if (
            self.fp8_cache_only or (self.cubic8_cache and not self.native_cubic8_cache)
        ) and is_workspace_manager_initialized():
            max_splits = _compute_cache_only_num_kv_splits(
                self.model_config.max_model_len, num_sms
            )
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            workspace_specs: list[tuple[tuple[int, ...], Any]] = [
                (
                    (
                        max_seqs,
                        self.num_q_heads,
                        max_splits,
                        self.mla_dims.kv_lora_rank + 1,
                    ),
                    torch.float32,
                )
            ]
            if self.cubic8_cache:
                from vllm.v1.attention.ops.cubic8_mla import (
                    cubic8_mla_max_materialize_tokens,
                )

                semantic_dim = (
                    self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
                )
                page_size = vllm_config.cache_config.block_size
                max_tokens = cubic8_mla_max_materialize_tokens(
                    max_seqs, self.model_config.max_model_len, page_size
                )
                workspace_specs.extend(
                    [
                        ((max_tokens, semantic_dim), self.model_config.dtype),
                        ((max_tokens // page_size,), torch.int32),
                    ]
                )
            current_workspace_manager().get_simultaneous(*workspace_specs)

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.cg_buf_tile_scheduler_metadata = torch.zeros(
                # Upper bound on size (<= #SMs, TileSchedulerMetaDataSize)
                # TileSchedulerMetaDataSize = 8
                (num_sms, 8),
                device=self.device,
                dtype=torch.int32,
            )
            self.cg_buf_num_splits = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # we use the max but all should be the same due to uniform length requirement
        max_query_len = query_lens_cpu.max().item()
        num_q_heads = self.num_q_heads
        if self.dcp_world_size > 1:
            num_q_heads *= self.dcp_world_size
        num_q_tokens_per_head_k = max_query_len * num_q_heads // 1
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,  # MQA for the decode path
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        if self.use_native_compressed_scheduler:
            tile_scheduler_metadata, num_splits = get_mla_metadata_dense_fp8(
                seq_lens_device,
                num_q_tokens_per_head_k,
                1,  # MQA for the decode path
            )

            # Copy FP8 metadata into persistent CUDA graph buffers
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                assert self.cg_buf_tile_scheduler_metadata is not None
                assert self.cg_buf_num_splits is not None
                n = tile_scheduler_metadata.size(0)
                assert n <= self.cg_buf_tile_scheduler_metadata.size(0)
                self.cg_buf_tile_scheduler_metadata[:n].copy_(tile_scheduler_metadata)
                tile_scheduler_metadata = self.cg_buf_tile_scheduler_metadata[:n]

                n = num_splits.size(0)
                assert n <= self.cg_buf_num_splits.size(0)
                self.cg_buf_num_splits[:n].copy_(num_splits)
                num_splits = self.cg_buf_num_splits[:n]

            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata
            scheduler_metadata.num_splits = num_splits

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            scheduler_metadata=scheduler_metadata,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        is_supported, reason = is_flashmla_dense_supported()
        assert is_supported, reason

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashMLAImpl"
            )

        self._fp8_cache_only = self.kv_cache_dtype == "fp8_q16"
        self._cubic8_cache = self.kv_cache_dtype == "cubic8"
        self._native_cubic8_cache = self.kv_cache_dtype == "cubic8"
        if self._fp8_cache_only or self._cubic8_cache:
            self.supports_quant_query_input = False
            device_index = torch.accelerator.current_device_index()
            self._sm_count = num_compute_units(device_index)
            logger.info_once(
                "%s MLA cache-only decode keeps queries in model dtype and "
                "dequantizes cached values inside the attention kernel on GPU %d.",
                self.kv_cache_dtype,
                device_index,
            )

    def _forward_fp8_q16(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a model-dtype query against a compressed MLA cache."""
        assert q.ndim == 3, (
            "Compressed-cache FlashMLA supports single-token decode; "
            f"received query shape {tuple(q.shape)}"
        )
        assert attn_metadata.decode is not None

        if (
            self._fp8_cache_only
            and q.dtype == torch.bfloat16
            and q.shape[0] * attn_metadata.max_seq_len <= _FP8_Q16_NATIVE_MAX_TOKEN_WORK
        ):
            scheduler = attn_metadata.decode.scheduler_metadata
            assert scheduler.tile_scheduler_metadata is not None
            assert scheduler.num_splits is not None
            output, lse = flash_mla_with_kvcache_fp8_q16(
                q=q.unsqueeze(1),
                k_cache=kv_c_and_k_pe_cache.unsqueeze(2),
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler.tile_scheduler_metadata,
                num_splits=scheduler.num_splits,
                descale_k=layer._k_scale.reshape(1),
                softmax_scale=self.scale,
                causal=False,
            )
            return output.squeeze(1), lse.squeeze(-1)

        if (
            self._native_cubic8_cache
            and q.dtype == torch.bfloat16
            and q.shape[-1] == 576
            and kv_c_and_k_pe_cache.shape[1] == 64
        ):
            scheduler = attn_metadata.decode.scheduler_metadata
            assert scheduler.tile_scheduler_metadata is not None
            assert scheduler.num_splits is not None
            output, lse = flash_mla_with_kvcache_cubic8(
                q=q.unsqueeze(1),
                k_cache=kv_c_and_k_pe_cache.unsqueeze(2),
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler.tile_scheduler_metadata,
                num_splits=scheduler.num_splits,
                softmax_scale=self.scale,
                causal=False,
            )
            return output.squeeze(1), lse.squeeze(-1)
        batch, num_heads, _ = q.shape
        output = torch.zeros(
            batch, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(batch, num_heads, dtype=q.dtype, device=q.device)
        num_kv_splits = (
            1
            if envs.VLLM_BATCH_INVARIANT
            else _compute_cache_only_num_kv_splits(
                attn_metadata.max_seq_len, self._sm_count
            )
        )
        logits_shape = (
            batch,
            num_heads,
            num_kv_splits,
            self.kv_lora_rank + 1,
        )
        materialize_workspace = None
        contiguous_block_table = None
        materialize_tokens = 0
        if self._cubic8_cache:
            from vllm.v1.attention.ops.cubic8_mla import (
                CUBIC8_MLA_MATERIALIZE_TOKEN_CAP,
                cubic8_mla_materialize_token_count,
            )

            page_size = kv_c_and_k_pe_cache.shape[1]
            materialize_tokens = cubic8_mla_materialize_token_count(
                batch, attn_metadata.max_seq_len, page_size
            )
            use_materialized = materialize_tokens <= CUBIC8_MLA_MATERIALIZE_TOKEN_CAP
        else:
            use_materialized = False

        if is_workspace_manager_initialized():
            workspace_specs: list[tuple[tuple[int, ...], Any]] = [
                (logits_shape, torch.float32)
            ]
            if use_materialized:
                workspace_specs.extend(
                    [
                        ((materialize_tokens, q.shape[-1]), q.dtype),
                        (
                            (
                                batch,
                                materialize_tokens
                                // batch
                                // kv_c_and_k_pe_cache.shape[1],
                            ),
                            torch.int32,
                        ),
                    ]
                )
            workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
            attn_logits = workspaces[0]
            if use_materialized:
                materialize_workspace = workspaces[1]
                contiguous_block_table = workspaces[2]
        else:
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=q.device
            )
            if use_materialized:
                materialize_workspace = torch.empty(
                    materialize_tokens, q.shape[-1], dtype=q.dtype, device=q.device
                )
                contiguous_block_table = torch.empty(
                    batch,
                    materialize_tokens // batch // kv_c_and_k_pe_cache.shape[1],
                    dtype=torch.int32,
                    device=q.device,
                )

        if self._cubic8_cache:
            if use_materialized:
                assert materialize_workspace is not None
                assert contiguous_block_table is not None
                from vllm.v1.attention.ops.cubic8_mla import (
                    cubic8_mla_materialized_decode_attention_fwd,
                )

                cubic8_mla_materialized_decode_attention_fwd(
                    q,
                    kv_c_and_k_pe_cache,
                    output,
                    lse,
                    attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens,
                    attn_logits,
                    materialize_workspace,
                    contiguous_block_table,
                    num_kv_splits,
                    self.scale,
                    self.kv_lora_rank,
                    attn_metadata.max_seq_len,
                    layer._k_scale,
                )
            else:
                from vllm.v1.attention.ops.cubic8_mla import (
                    cubic8_mla_decode_attention_fwd,
                )

                cubic8_mla_decode_attention_fwd(
                    q,
                    kv_c_and_k_pe_cache,
                    output,
                    lse,
                    attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens,
                    attn_logits,
                    num_kv_splits,
                    self.scale,
                    self.kv_lora_rank,
                )
            return output, lse

        cache = kv_c_and_k_pe_cache.unsqueeze(2)
        decode_attention_fwd(
            q,
            cache,
            cache[..., : self.kv_lora_rank],
            output,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            cache.size(1),
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )
        return output, lse

    def _forward_fp8_q16_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        prefill_metadata,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend model-dtype absorbed queries directly to an FP8 cache."""
        assert self._fp8_cache_only
        assert q.dtype == torch.bfloat16
        assert prefill_metadata.query_lens_cpu is not None
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None

        cache = kv_c_and_k_pe_cache
        if cache.dtype != current_platform.fp8_dtype():
            cache = cache.view(current_platform.fp8_dtype())
        cache = cache.unsqueeze(2)

        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        query_offset = 0
        query_lens = prefill_metadata.query_lens_cpu.tolist()
        for request_idx, (query_len, context_len) in enumerate(
            zip(query_lens, chunked_context.context_lens_list)
        ):
            query_end = query_offset + query_len
            request_q = q[query_offset:query_end]
            if context_len == 0:
                outputs.append(
                    torch.zeros(
                        query_len,
                        q.shape[1],
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device,
                    )
                )
                lses.append(
                    torch.full(
                        (query_len, q.shape[1]),
                        -float("inf"),
                        dtype=torch.float32,
                        device=q.device,
                    )
                )
                query_offset = query_end
                continue

            request_context_len = chunked_context.context_lens[
                request_idx : request_idx + 1
            ]
            tile_metadata, num_splits = get_mla_metadata_dense_fp8(
                request_context_len,
                query_len * q.shape[1],
                1,
            )
            request_output, request_lse = flash_mla_with_kvcache_fp8_q16(
                q=request_q.unsqueeze(0),
                k_cache=cache,
                block_table=prefill_metadata.block_table[request_idx : request_idx + 1],
                cache_seqlens=request_context_len,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=tile_metadata,
                num_splits=num_splits,
                descale_k=k_scale.reshape(1),
                softmax_scale=self.scale,
                causal=False,
            )
            outputs.append(request_output.squeeze(0))
            lses.append(request_lse.squeeze(0).transpose(0, 1).contiguous())
            query_offset = query_end

        assert query_offset == q.shape[0]
        return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: (zyongye) decode function for mla here
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        # mypy assertion: q is now always a tensor
        assert isinstance(q, torch.Tensor)

        if self._fp8_cache_only or self._cubic8_cache:
            return self._forward_fp8_q16(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        num_decodes = attn_metadata.num_decodes
        q = reshape_query_for_spec_decode(q, num_decodes)

        scheduler_metadata = attn_metadata.decode.scheduler_metadata
        if envs.VLLM_BATCH_INVARIANT and not is_quantized_kv_cache(self.kv_cache_dtype):
            device = q.device
            dtype = torch.int32

            B = q.shape[0]
            # block_table shape: [batch_size, max_num_blocks_per_seq]
            # The number of blocks per sequence is in the second dimension
            topk = attn_metadata.decode.block_table.shape[-1]
            B_TOPK = 64
            assert topk % B_TOPK == 0, f"topk ({topk}) must be divisible by {B_TOPK}"
            end_block_idx = topk // B_TOPK

            # Single partition => num_sm_parts = 1
            # TileSchedulerMetaDataSize = 8, layout:
            # [begin_idx, begin_block_idx, end_idx, end_block_idx,
            #  begin_n_split_idx, _, _, _]
            tile_scheduler_metadata = torch.zeros((1, 8), dtype=dtype, device=device)
            tile_scheduler_metadata[0, 0] = 0  # begin_idx
            tile_scheduler_metadata[0, 1] = 0  # sched_begin_block_idx
            tile_scheduler_metadata[0, 2] = B - 1  # end_idx
            tile_scheduler_metadata[0, 3] = end_block_idx
            tile_scheduler_metadata[0, 4] = 0  # begin_n_split_idx
            # fields [5..7] stay 0

            # Non-split path ignores num_splits, but the API requires it:
            # zeros of length B+1
            num_splits = torch.zeros((B + 1,), dtype=dtype, device=device)
            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata
            scheduler_metadata.num_splits = num_splits

        if is_quantized_kv_cache(self.kv_cache_dtype):
            o, lse = flash_mla_with_kvcache_fp8(
                q=q,
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler_metadata.tile_scheduler_metadata,
                num_splits=scheduler_metadata.num_splits,
                softmax_scale=self.scale,
                causal=True,
                descale_q=layer._q_scale.reshape(1),
                descale_k=layer._k_scale.reshape(1),
            )
        else:
            o, lse = flash_mla_with_kvcache(
                q=q,
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler_metadata,
                softmax_scale=self.scale,
                causal=True,
                is_fp8_kvcache=False,
            )

        o = reshape_attn_output_for_spec_decode(o)

        return o, lse
