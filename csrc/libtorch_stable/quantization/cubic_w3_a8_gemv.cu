// SPDX-License-Identifier: Apache-2.0
// Cubic W3/A8 subgroup DP4A GEMV for precomputed INT8 carrier levels.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

__device__ __forceinline__ int cubic_dp4a(int a, int b, int c) {
  int out;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

__device__ __forceinline__ int cubic_prmt(int a, int b, int selector) {
  int out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
  return out;
}

__device__ __forceinline__ unsigned cubic_w3_selector4(unsigned packed) {
  // Convert four adjacent base-8 digits into PRMT's four base-16 selector
  // nibbles.  For each pair, adding the upper 3-bit digit to itself moves it
  // from bit 3 to bit 4 without affecting the lower digit.
  unsigned lo = packed & 0x3fu;
  lo += lo & 0x38u;
  unsigned hi = (packed >> 6) & 0x3fu;
  hi += hi & 0x38u;
  return lo | (hi << 8);
}

__device__ __forceinline__ void cubic_w3_carrier_words(unsigned packed,
                                                       int lut0, int lut1,
                                                       int& lo, int& hi) {
  int selector = cubic_w3_selector4(packed);
  lo = cubic_prmt(lut0, lut1, selector);
  selector = cubic_w3_selector4(packed >> 12);
  hi = cubic_prmt(lut0, lut1, selector);
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
__global__ void cubic_w3_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const int8_t* __restrict__ level1, const int8_t* __restrict__ level2,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int top_k, int route_ctas, bool multiply_routed_weight) {
  constexpr int threads = 128;
  constexpr int outputs_per_block = threads / ThreadsPerOutput;
  constexpr int blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * 3 / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(blocks_per_group % ThreadsPerOutput == 0);

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  if (out_channel >= n) return;
  const int num_routes = *num_routes_ptr;
  const int input_words_per_row = k / 4;

  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token_id = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token_id / top_k;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* input_words =
        reinterpret_cast<const int*>(input) +
        static_cast<int64_t>(input_row) * input_words_per_row;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base = expert_output * num_groups;
    const float row_scale = GroupwiseScale ? 0.0f : input_scale[input_row];
    float accumulator = 0.0f;

    for (int group = 0; group < num_groups; ++group) {
      const int64_t meta = meta_base + group;
      const int l1 = static_cast<unsigned char>(level1[meta]);
      const int l2 = static_cast<unsigned char>(level2[meta]);
      const int lut0 = (l1 << 8) | (l2 << 16) | 0x7f000000;
      const int lut1 = 0x00008100 | ((-l2 & 0xff) << 16) | ((-l1 & 0xff) << 24);
      int dot = 0;
#pragma unroll
      for (int block_index = 0; block_index < blocks_per_thread;
           ++block_index) {
        const int code_block = lane + block_index * ThreadsPerOutput;
        const int word_base = group * words_per_group + code_block * 3;
        const unsigned w0 = weight_words[word_base];
        const unsigned w1 = weight_words[word_base + 1];
        const unsigned w2 = weight_words[word_base + 2];
        const unsigned packed[4] = {
            w0 & 0x00ffffffu, (w0 >> 24) | ((w1 & 0x0000ffffu) << 8),
            (w1 >> 16) | ((w2 & 0x000000ffu) << 16), w2 >> 8};
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
          int carrier_lo, carrier_hi;
          cubic_w3_carrier_words(packed[chunk], lut0, lut1, carrier_lo,
                                 carrier_hi);
          const int input_base =
              group * input_words_per_group + code_block * 8 + chunk * 2;
          dot = cubic_dp4a(carrier_lo, input_words[input_base], dot);
          dot = cubic_dp4a(carrier_hi, input_words[input_base + 1], dot);
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
        dot += __shfl_down_sync(mask, dot, delta, ThreadsPerOutput);
      }
      if (lane == 0) {
        const float activation_scale =
            GroupwiseScale
                ? input_scale[static_cast<int64_t>(input_row) * num_groups +
                              group]
                : row_scale;
        accumulator += static_cast<float>(dot) * activation_scale *
                       weight_scale[meta] * (1.0f / 127.0f);
      }
    }
    if (lane == 0) {
      if (multiply_routed_weight) accumulator *= topk_weights[token_id];
      output[static_cast<int64_t>(token_id) * n + out_channel] =
          __float2bfloat16(accumulator);
    }
  }
}

template <int GroupSize, int ThreadsPerOutput, bool Pair,
          bool GroupwiseScale = false>
__device__ __forceinline__ void cubic_w3_compute_route_block(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const int8_t* __restrict__ level1, const int8_t* __restrict__ level2,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    int token0, int token1, int expert, int out_channel, int n, int k,
    int num_groups, int packed_k, int top_k, bool multiply_routed_weight) {
  constexpr int blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * 3 / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;

  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int input_words_per_row = k / 4;
  const int row0 = token0 / top_k;
  const int row1 = Pair ? token1 / top_k : 0;
  const int64_t expert_output = static_cast<int64_t>(expert) * n + out_channel;
  const int* input0 = reinterpret_cast<const int*>(input) +
                      static_cast<int64_t>(row0) * input_words_per_row;
  const int* input1 = reinterpret_cast<const int*>(input) +
                      static_cast<int64_t>(row1) * input_words_per_row;
  const int* weight_words =
      reinterpret_cast<const int*>(weight + expert_output * packed_k);
  const int64_t meta_base = expert_output * num_groups;
  const float row_scale0 = GroupwiseScale ? 0.0f : input_scale[row0];
  const float row_scale1 = Pair && !GroupwiseScale ? input_scale[row1] : 0.0f;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;

  for (int group = 0; group < num_groups; ++group) {
    const int64_t meta = meta_base + group;
    const int l1 = static_cast<unsigned char>(level1[meta]);
    const int l2 = static_cast<unsigned char>(level2[meta]);
    const int lut0 = (l1 << 8) | (l2 << 16) | 0x7f000000;
    const int lut1 = 0x00008100 | ((-l2 & 0xff) << 16) | ((-l1 & 0xff) << 24);
    int dot0 = 0;
    int dot1 = 0;
#pragma unroll
    for (int block_index = 0; block_index < blocks_per_thread; ++block_index) {
      const int code_block = lane + block_index * ThreadsPerOutput;
      const int word_base = group * words_per_group + code_block * 3;
      const unsigned w0 = weight_words[word_base];
      const unsigned w1 = weight_words[word_base + 1];
      const unsigned w2 = weight_words[word_base + 2];
      const unsigned packed[4] = {
          w0 & 0x00ffffffu, (w0 >> 24) | ((w1 & 0x0000ffffu) << 8),
          (w1 >> 16) | ((w2 & 0x000000ffu) << 16), w2 >> 8};
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        int carrier_lo, carrier_hi;
        cubic_w3_carrier_words(packed[chunk], lut0, lut1, carrier_lo,
                               carrier_hi);
        const int input_base =
            group * input_words_per_group + code_block * 8 + chunk * 2;
        dot0 = cubic_dp4a(carrier_lo, input0[input_base], dot0);
        dot0 = cubic_dp4a(carrier_hi, input0[input_base + 1], dot0);
        if constexpr (Pair) {
          dot1 = cubic_dp4a(carrier_lo, input1[input_base], dot1);
          dot1 = cubic_dp4a(carrier_hi, input1[input_base + 1], dot1);
        }
      }
    }
    for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
      dot0 += __shfl_down_sync(mask, dot0, delta, ThreadsPerOutput);
      if constexpr (Pair) {
        dot1 += __shfl_down_sync(mask, dot1, delta, ThreadsPerOutput);
      }
    }
    if (lane == 0) {
      const float ws = weight_scale[meta];
      const float act_scale0 =
          GroupwiseScale
              ? input_scale[static_cast<int64_t>(row0) * num_groups + group]
              : row_scale0;
      accumulator0 +=
          static_cast<float>(dot0) * act_scale0 * ws * (1.0f / 127.0f);
      if constexpr (Pair) {
        const float act_scale1 =
            GroupwiseScale
                ? input_scale[static_cast<int64_t>(row1) * num_groups + group]
                : row_scale1;
        accumulator1 +=
            static_cast<float>(dot1) * act_scale1 * ws * (1.0f / 127.0f);
      }
    }
  }
  if (lane == 0) {
    if (multiply_routed_weight) accumulator0 *= topk_weights[token0];
    output[static_cast<int64_t>(token0) * n + out_channel] =
        __float2bfloat16(accumulator0);
    if constexpr (Pair) {
      if (multiply_routed_weight) accumulator1 *= topk_weights[token1];
      output[static_cast<int64_t>(token1) * n + out_channel] =
          __float2bfloat16(accumulator1);
    }
  }
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
__global__ void cubic_w3_grouped2_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const int8_t* __restrict__ level1, const int8_t* __restrict__ level2,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int num_valid_tokens, int n, int k,
    int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight) {
  constexpr int threads = 128;
  constexpr int outputs_per_block = threads / ThreadsPerOutput;

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  if (out_channel >= n) return;
  const int num_route_blocks = (*num_routes_ptr + 1) / 2;

  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    const int token0 = token_ids[route_block * 2];
    const int token1 = token_ids[route_block * 2 + 1];
    // CUDA-graph padding may use either the positive end sentinel emitted by
    // moe_align_block_size or a negative route id.  Treat both as padding;
    // accepting a negative id here would form a negative output address.
    const bool valid0 =
        static_cast<unsigned>(token0) < static_cast<unsigned>(num_valid_tokens);
    const bool valid1 =
        static_cast<unsigned>(token1) < static_cast<unsigned>(num_valid_tokens);
    const int expert = expert_ids[route_block];
    if (expert < 0 || (!valid0 && !valid1)) continue;
    const int single_token = valid0 ? token0 : token1;
    if (valid0 && valid1) {
      cubic_w3_compute_route_block<GroupSize, ThreadsPerOutput, true,
                                   GroupwiseScale>(
          input, input_scale, weight, weight_scale, level1, level2, output,
          topk_weights, token0, token1, expert, out_channel, n, k, num_groups,
          packed_k, top_k, multiply_routed_weight);
    } else {
      cubic_w3_compute_route_block<GroupSize, ThreadsPerOutput, false,
                                   GroupwiseScale>(
          input, input_scale, weight, weight_scale, level1, level2, output,
          topk_weights, single_token, 0, expert, out_channel, n, k, num_groups,
          packed_k, top_k, multiply_routed_weight);
    }
  }
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
void launch_cubic_w3_a8(const int8_t* input, const float* input_scale,
                        const uint8_t* weight, const float* weight_scale,
                        const int8_t* level1, const int8_t* level2,
                        __nv_bfloat16* output, const float* topk_weights,
                        const int* token_ids, const int* expert_ids,
                        const int* num_routes, int n, int k, int num_groups,
                        int packed_k, int top_k, int route_ctas,
                        bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w3_a8_gemv_kernel<GroupSize, ThreadsPerOutput, GroupwiseScale>
      <<<grid, 128, 0, stream>>>(
          input, input_scale, weight, weight_scale, level1, level2, output,
          topk_weights, token_ids, expert_ids, num_routes, n, k, num_groups,
          packed_k, top_k, route_ctas, multiply_routed_weight);
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
void launch_cubic_w3_grouped2_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, const int8_t* level1, const int8_t* level2,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int num_valid_tokens, int n,
    int k, int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w3_grouped2_a8_gemv_kernel<GroupSize, ThreadsPerOutput, GroupwiseScale>
      <<<grid, 128, 0, stream>>>(
          input, input_scale, weight, weight_scale, level1, level2, output,
          topk_weights, token_ids, expert_ids, num_routes, num_valid_tokens, n,
          k, num_groups, packed_k, top_k, route_ctas, multiply_routed_weight);
}

}  // namespace

void cubic_w3_a8_gemv(const torch::stable::Tensor& input,
                      const torch::stable::Tensor& input_scale,
                      const torch::stable::Tensor& weight,
                      const torch::stable::Tensor& weight_scale,
                      const torch::stable::Tensor& level1,
                      const torch::stable::Tensor& level2,
                      torch::stable::Tensor& output,
                      const torch::stable::Tensor& topk_weights,
                      const torch::stable::Tensor& token_ids,
                      const torch::stable::Tensor& expert_ids,
                      const torch::stable::Tensor& num_routes,
                      int64_t group_size, int64_t top_k,
                      bool multiply_routed_weight, int64_t route_ctas,
                      int64_t num_valid_tokens, int64_t routes_per_block) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w3_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          level1.scalar_type() == torch::headeronly::ScalarType::Char &&
          level2.scalar_type() == torch::headeronly::ScalarType::Char &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w3_a8_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                      weight_scale.is_contiguous() && level1.is_contiguous() &&
                      level2.is_contiguous() && output.is_contiguous(),
                  "cubic_w3_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK(
      (group_size == 128 || group_size == 256 || group_size == 512) &&
          k % group_size == 0 && weight.size(2) == k * 3 / 8,
      "cubic_w3_a8_gemv: unsupported shape or group size");
  const int groups = k / group_size;
  // A resident CTA grid may intentionally be larger than a tiny route
  // buffer.  Such CTAs exit at the route-loop condition before dereferencing
  // token_ids; only a positive launch dimension is required here.
  STD_TORCH_CHECK(
      (routes_per_block == 1 || routes_per_block == 2) && route_ctas > 0,
      "cubic_w3_a8_gemv: invalid route_ctas");
  const int desired_subgroup = k <= 4096 ? 4 : 8;
  const auto stream = get_current_cuda_stream(input.get_device_index());

#define CUBIC_ARGS                                                            \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      level1.const_data_ptr<int8_t>(), level2.const_data_ptr<int8_t>(),       \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      stream
#define CUBIC_GROUPED_ARGS                                                    \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      level1.const_data_ptr<int8_t>(), level2.const_data_ptr<int8_t>(),       \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),     \
      num_valid_tokens, n, k, groups, weight.size(2), top_k, route_ctas,      \
      multiply_routed_weight, stream
  if (routes_per_block == 2) {
    if (group_size == 128) {
      launch_cubic_w3_grouped2_a8<128, 4>(CUBIC_GROUPED_ARGS);
    } else if (group_size == 256 && desired_subgroup == 4) {
      launch_cubic_w3_grouped2_a8<256, 4>(CUBIC_GROUPED_ARGS);
    } else if (group_size == 256) {
      launch_cubic_w3_grouped2_a8<256, 8>(CUBIC_GROUPED_ARGS);
    } else if (desired_subgroup == 4) {
      launch_cubic_w3_grouped2_a8<512, 4>(CUBIC_GROUPED_ARGS);
    } else {
      launch_cubic_w3_grouped2_a8<512, 8>(CUBIC_GROUPED_ARGS);
    }
  } else {
    if (group_size == 128) {
      launch_cubic_w3_a8<128, 4>(CUBIC_ARGS);
    } else if (group_size == 256 && desired_subgroup == 4) {
      launch_cubic_w3_a8<256, 4>(CUBIC_ARGS);
    } else if (group_size == 256) {
      launch_cubic_w3_a8<256, 8>(CUBIC_ARGS);
    } else if (desired_subgroup == 4) {
      launch_cubic_w3_a8<512, 4>(CUBIC_ARGS);
    } else {
      launch_cubic_w3_a8<512, 8>(CUBIC_ARGS);
    }
  }
#undef CUBIC_GROUPED_ARGS
#undef CUBIC_ARGS
}

void cubic_w3_groupwise_a8_gemv(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale,
    const torch::stable::Tensor& level1, const torch::stable::Tensor& level2,
    torch::stable::Tensor& output, const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t group_size, int64_t top_k,
    bool multiply_routed_weight, int64_t route_ctas, int64_t num_valid_tokens,
    int64_t routes_per_block) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w3_groupwise_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          level1.scalar_type() == torch::headeronly::ScalarType::Char &&
          level2.scalar_type() == torch::headeronly::ScalarType::Char &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w3_groupwise_a8_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      level1.is_contiguous() && level2.is_contiguous() &&
                      output.is_contiguous(),
                  "cubic_w3_groupwise_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK(
      (group_size == 128 || group_size == 256 || group_size == 512) &&
          k % group_size == 0 && weight.size(2) == k * 3 / 8 &&
          input_scale.dim() == 2 && input_scale.size(0) == input.size(0) &&
          input_scale.size(1) == k / group_size,
      "cubic_w3_groupwise_a8_gemv: unsupported shape or scale layout");
  STD_TORCH_CHECK(
      (routes_per_block == 1 || routes_per_block == 2) && route_ctas > 0,
      "cubic_w3_groupwise_a8_gemv: invalid route_ctas");
  const int groups = k / group_size;
  const int desired_subgroup = k <= 4096 ? 4 : 8;
  const auto stream = get_current_cuda_stream(input.get_device_index());

#define CUBIC_ONLINE_ARGS                                                     \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      level1.const_data_ptr<int8_t>(), level2.const_data_ptr<int8_t>(),       \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      stream
#define CUBIC_ONLINE_GROUPED_ARGS                                             \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      level1.const_data_ptr<int8_t>(), level2.const_data_ptr<int8_t>(),       \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),     \
      num_valid_tokens, n, k, groups, weight.size(2), top_k, route_ctas,      \
      multiply_routed_weight, stream
  if (routes_per_block == 2) {
    if (group_size == 128) {
      launch_cubic_w3_grouped2_a8<128, 4, true>(CUBIC_ONLINE_GROUPED_ARGS);
    } else if (group_size == 256 && desired_subgroup == 4) {
      launch_cubic_w3_grouped2_a8<256, 4, true>(CUBIC_ONLINE_GROUPED_ARGS);
    } else if (group_size == 256) {
      launch_cubic_w3_grouped2_a8<256, 8, true>(CUBIC_ONLINE_GROUPED_ARGS);
    } else if (desired_subgroup == 4) {
      launch_cubic_w3_grouped2_a8<512, 4, true>(CUBIC_ONLINE_GROUPED_ARGS);
    } else {
      launch_cubic_w3_grouped2_a8<512, 8, true>(CUBIC_ONLINE_GROUPED_ARGS);
    }
  } else {
    if (group_size == 128) {
      launch_cubic_w3_a8<128, 4, true>(CUBIC_ONLINE_ARGS);
    } else if (group_size == 256 && desired_subgroup == 4) {
      launch_cubic_w3_a8<256, 4, true>(CUBIC_ONLINE_ARGS);
    } else if (group_size == 256) {
      launch_cubic_w3_a8<256, 8, true>(CUBIC_ONLINE_ARGS);
    } else if (desired_subgroup == 4) {
      launch_cubic_w3_a8<512, 4, true>(CUBIC_ONLINE_ARGS);
    } else {
      launch_cubic_w3_a8<512, 8, true>(CUBIC_ONLINE_ARGS);
    }
  }
#undef CUBIC_ONLINE_GROUPED_ARGS
#undef CUBIC_ONLINE_ARGS
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("cubic_w3_a8_gemv", TORCH_BOX(&cubic_w3_a8_gemv));
  m.impl("cubic_w3_groupwise_a8_gemv", TORCH_BOX(&cubic_w3_groupwise_a8_gemv));
}
