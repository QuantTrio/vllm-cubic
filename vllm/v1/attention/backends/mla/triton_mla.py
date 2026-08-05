# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# num_kv_splits selection (shared by forward_mqa and the workspace reservation
# so the two cannot drift). Both are hardware dependent.
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2


def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    # Power of 2 to avoid excessive kernel instantiations, capped by an SM-based
    # maximum (occupancy multiplier allows multiple blocks per SM
    # for latency hiding).
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        # Match FlashMLA: Cubic8 currently requires the dynamic MLA cache
        # boundary to remain outside full CUDA graphs.
        if getattr(kv_cache_spec, "cache_dtype_str", None) == "cubic8":
            return AttentionCGSupport.NEVER
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._reserve_attn_logits_workspace()

    def _reserve_attn_logits_workspace(self) -> None:
        """Pre-size the shared workspace for the decode split-KV attn logits.

        Reserving at the worst case (max_model_len -> max num_kv_splits,
        max_num_seqs decode tokens) before warmup/cudagraph capture means the
        per-call ``get_simultaneous`` in ``forward_mqa`` never has to grow the
        buffer at runtime (which would raise once the workspace is locked).
        """
        if not is_workspace_manager_initialized():
            return
        # Decode reorder threshold is 1, so decode tokens <= max_num_seqs.
        B = self.vllm_config.scheduler_config.max_num_seqs
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        max_splits = _compute_num_kv_splits(
            self.model_config.max_model_len,
            current_platform.num_compute_units(),
        )
        lse_dim = self.mla_dims.kv_lora_rank + 1
        workspace_specs: list[tuple[tuple[int, ...], Any]] = [
            ((B, q_num_heads, max_splits, lse_dim), torch.float32),
        ]
        if getattr(self.kv_cache_spec, "cache_dtype_str", None) == "cubic8":
            from vllm.v1.attention.ops.cubic8_mla import (
                cubic8_mla_max_materialize_tokens,
            )

            semantic_dim = self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
            page_size = self.vllm_config.cache_config.block_size
            max_materialize_tokens = cubic8_mla_max_materialize_tokens(
                B,
                self.model_config.max_model_len,
                page_size,
            )
            workspace_specs.extend(
                [
                    (
                        (max_materialize_tokens, semantic_dim),
                        self.model_config.dtype,
                    ),
                    ((max_materialize_tokens // page_size,), torch.int32),
                ]
            )
        current_workspace_manager().get_simultaneous(*workspace_specs)


class TritonMLABackend(MLACommonBackend):
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

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

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
        if kv_cache_dtype == "fp8_q16" and device_capability.to_int() < 89:
            return (
                "fp8_q16 requires reliable E4M3 load/dequant support "
                "(SM89+); an SM80 software reader is not implemented yet"
            )
        return None


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
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

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton MLA backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 "
                    f"(fp8e4nv) requires SM89+. Re-run with "
                    f"--kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported by the Triton MLA "
                    f"backend on {dev} (compute capability {cap_str}); "
                    f"bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = _compute_num_kv_splits(
                attn_metadata.max_seq_len, self._sm_count
            )

        # NOTE: the +1 stores the LogSumExp (LSE) that the stage2 kernel uses to
        # merge partial attention outputs across splits. The scratch is served
        # from the shared workspace (reserved at max in the metadata builder), so
        # there is no per-call allocation on the decode hot path. Fall back to a
        # direct allocation when the workspace manager is not initialized (e.g.
        # unit tests without a GPUModelRunner).
        logits_shape = (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1)
        materialize_workspace = None
        contiguous_block_table = None
        materialize_tokens = 0
        if self.kv_cache_dtype == "cubic8":
            from vllm.v1.attention.ops.cubic8_mla import (
                CUBIC8_MLA_MATERIALIZE_TOKEN_CAP,
                cubic8_mla_materialize_token_count,
            )

            page_size = kv_c_and_k_pe_cache.shape[1]
            materialize_tokens = cubic8_mla_materialize_token_count(
                B, attn_metadata.max_seq_len, page_size
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
                        (
                            (materialize_tokens, q.shape[-1]),
                            q.dtype,
                        ),
                        (
                            (
                                B,
                                materialize_tokens // B // kv_c_and_k_pe_cache.shape[1],
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
                    materialize_tokens,
                    q.shape[-1],
                    dtype=q.dtype,
                    device=q.device,
                )
                contiguous_block_table = torch.empty(
                    B,
                    materialize_tokens // B // kv_c_and_k_pe_cache.shape[1],
                    dtype=torch.int32,
                    device=q.device,
                )

        if self.kv_cache_dtype == "cubic8":
            if use_materialized:
                assert materialize_workspace is not None
                assert contiguous_block_table is not None
                from vllm.v1.attention.ops.cubic8_mla import (
                    cubic8_mla_materialized_decode_attention_fwd,
                )

                cubic8_mla_materialized_decode_attention_fwd(
                    q,
                    kv_c_and_k_pe_cache,
                    o,
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
                    o,
                    lse,
                    attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens,
                    attn_logits,
                    num_kv_splits,
                    self.scale,
                    self.kv_lora_rank,
                )
            return o, lse

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse
