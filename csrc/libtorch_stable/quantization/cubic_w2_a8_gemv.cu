// SPDX-License-Identifier: Apache-2.0
// Cubic W2/A8 subgroup DP4A GEMV for large quantization groups.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cfloat>
#include <climits>

namespace {

__device__ __forceinline__ int cubic_w2_dp4a(int a, int b, int c) {
  int out;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

__device__ __forceinline__ int cubic_w2_prmt(int a, int b, int selector) {
  int out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
  return out;
}

__device__ __forceinline__ int cubic_w2_carrier_word(unsigned packed_byte) {
  packed_byte &= 0xff;
  unsigned selector = (packed_byte | (packed_byte << 4)) & 0x0f0f;
  selector = (selector | (selector << 2)) & 0x3333;
  return cubic_w2_prmt(0xff000100, 0, selector);
}

__device__ __forceinline__ float cubic8_curve(float t, float a, float b);

__device__ __forceinline__ int cubic8_moment_word(int packed_codes, int power) {
  unsigned result = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const int q = static_cast<int8_t>((packed_codes >> (byte * 8)) & 0xff);
    const int magnitude = q < 0 ? -q : q;
    int moment;
    if (power == 2) {
      moment = (magnitude * magnitude + 63) / 127;
    } else {
      moment = (magnitude * magnitude * magnitude + 8064) / 16129;
    }
    if (q < 0) moment = -moment;
    result |=
        (static_cast<unsigned>(static_cast<uint8_t>(moment)) << (byte * 8));
  }
  return static_cast<int>(result);
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
__global__ void cubic_w2_cubic8_moment_gemv_kernel(
    const int8_t* __restrict__ input_code,
    const float* __restrict__ input_scale, const __half* __restrict__ input_a,
    const __half* __restrict__ input_b, const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale, __nv_bfloat16* __restrict__ output,
    const float* __restrict__ topk_weights, const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_routes_ptr,
    int n, int k, int num_weight_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int words_per_weight_group = WeightGroupSize / 16;
  constexpr int words_per_lane = words_per_weight_group / ThreadsPerOutput;
  constexpr int words_per_activation_group = ActivationGroupSize / 16;
  constexpr float inverse_127 = 1.0f / 127.0f;
  static_assert(words_per_weight_group % ThreadsPerOutput == 0);

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out < n;
  const int safe_out = valid_output ? out : n - 1;
  const int input_words_per_row = k / 4;
  const int activation_groups = k / ActivationGroupSize;
  const int num_routes = *num_routes_ptr;
  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token / top_k;
    const int* input_words =
        reinterpret_cast<const int*>(input_code) +
        static_cast<int64_t>(input_row) * input_words_per_row;
    const int64_t weight_output = static_cast<int64_t>(expert) * n + safe_out;
    const int* weight_words = reinterpret_cast<const int*>(
        weight + weight_output * static_cast<int64_t>(packed_k));
    const int64_t weight_meta = weight_output * num_weight_groups;
    const int64_t activation_meta =
        static_cast<int64_t>(input_row) * activation_groups;
    float accumulator = 0.0f;
    for (int weight_group = 0; weight_group < num_weight_groups;
         ++weight_group) {
      const float ws = weight_scale[weight_meta + weight_group];
#pragma unroll
      for (int index = 0; index < words_per_lane; ++index) {
        const int word = lane + index * ThreadsPerOutput;
        const int global_word = weight_group * words_per_weight_group + word;
        const int activation_group = global_word / words_per_activation_group;
        const int64_t metadata = activation_meta + activation_group;
        const float as = input_scale[metadata];
        const float curve_a = __half2float(input_a[metadata]);
        const float curve_b = __half2float(input_b[metadata]);
        const float curve_c = 1.0f - curve_a - curve_b;
        const unsigned packed_weight = weight_words[global_word];
        int first = 0;
        int second = 0;
        int third = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          const int q = input_words[global_word * 4 + byte];
          const int carrier =
              cubic_w2_carrier_word(packed_weight >> (byte * 8));
          first = cubic_w2_dp4a(carrier, q, first);
          second = cubic_w2_dp4a(carrier, cubic8_moment_word(q, 2), second);
          third = cubic_w2_dp4a(carrier, cubic8_moment_word(q, 3), third);
        }
        accumulator += ws * as * inverse_127 *
                       (curve_a * first + curve_b * second + curve_c * third);
      }
    }
#pragma unroll
    for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
      accumulator +=
          __shfl_down_sync(0xffffffffu, accumulator, delta, ThreadsPerOutput);
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token];
      output[static_cast<int64_t>(token) * n + out] =
          __float2bfloat16(accumulator);
    }
  }
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
__global__ void cubic_w2_cubic8_lut_gemv_kernel(
    const int8_t* __restrict__ input_code,
    const float* __restrict__ input_scale, const __half* __restrict__ input_a,
    const __half* __restrict__ input_b, const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale, __nv_bfloat16* __restrict__ output,
    const float* __restrict__ topk_weights, const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_routes_ptr,
    int n, int k, int num_weight_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int words_per_weight_group = WeightGroupSize / 16;
  constexpr int words_per_activation_group = ActivationGroupSize / 16;
  constexpr int words_per_lane = words_per_activation_group / ThreadsPerOutput;
  constexpr int activation_groups_per_weight =
      WeightGroupSize / ActivationGroupSize;
  static_assert(WeightGroupSize % ActivationGroupSize == 0);
  static_assert(words_per_activation_group % ThreadsPerOutput == 0);
  __shared__ float decode_lut[256];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out < n;
  const int safe_out = valid_output ? out : n - 1;
  const int num_routes = *num_routes_ptr;
  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token / top_k;
    const int64_t weight_output = static_cast<int64_t>(expert) * n + safe_out;
    const int* weight_words = reinterpret_cast<const int*>(
        weight + weight_output * static_cast<int64_t>(packed_k));
    const int64_t weight_meta = weight_output * num_weight_groups;
    const int num_activation_groups = k / ActivationGroupSize;
    const int64_t activation_meta =
        static_cast<int64_t>(input_row) * num_activation_groups;
    float accumulator = 0.0f;
    for (int weight_group = 0; weight_group < num_weight_groups;
         ++weight_group) {
      float weight_partial = 0.0f;
#pragma unroll
      for (int activation_subgroup = 0;
           activation_subgroup < activation_groups_per_weight;
           ++activation_subgroup) {
        const int activation_group =
            weight_group * activation_groups_per_weight + activation_subgroup;
        const float as = input_scale[activation_meta + activation_group];
        const float curve_a =
            __half2float(input_a[activation_meta + activation_group]);
        const float curve_b =
            __half2float(input_b[activation_meta + activation_group]);
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int lut_index = threadIdx.x + item * kThreads;
          const int q = lut_index - 128;
          const float t = fabsf(static_cast<float>(q)) * (1.0f / 127.0f);
          const float decoded = as * cubic8_curve(t, curve_a, curve_b);
          decode_lut[lut_index] = q < 0 ? -decoded : decoded;
        }
        __syncthreads();
        float partial = 0.0f;
#pragma unroll
        for (int index = 0; index < words_per_lane; ++index) {
          const int word = lane + index * ThreadsPerOutput;
          const unsigned packed_weight =
              weight_words[weight_group * words_per_weight_group +
                           activation_subgroup * words_per_activation_group +
                           word];
          const int code_base = weight_group * WeightGroupSize +
                                activation_subgroup * ActivationGroupSize +
                                word * 16;
#pragma unroll
          for (int element = 0; element < 16; ++element) {
            const int weight_code = (packed_weight >> (element * 2)) & 3;
            const int ternary = (weight_code & 1) * (1 - (weight_code & 2));
            const int q = input_code[static_cast<int64_t>(input_row) * k +
                                     code_base + element];
            partial += static_cast<float>(ternary) * decode_lut[q + 128];
          }
        }
        __syncthreads();
        weight_partial += partial;
      }
      accumulator += weight_partial * weight_scale[weight_meta + weight_group];
    }
#pragma unroll
    for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
      accumulator +=
          __shfl_down_sync(0xffffffffu, accumulator, delta, ThreadsPerOutput);
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token];
      output[static_cast<int64_t>(token) * n + out] =
          __float2bfloat16(accumulator);
    }
    __syncthreads();
  }
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
__global__ void cubic_w2_cubic8_shared_gemv_kernel(
    const int8_t* __restrict__ input_code,
    const float* __restrict__ input_scale, const __half* __restrict__ input_a,
    const __half* __restrict__ input_b, const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale, __nv_bfloat16* __restrict__ output,
    const float* __restrict__ topk_weights, const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_routes_ptr,
    int n, int k, int num_weight_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int words_per_group = WeightGroupSize / 16;
  constexpr int words_per_lane = words_per_group / ThreadsPerOutput;
  constexpr int decode_items = WeightGroupSize / kThreads;
  __shared__ float decoded_activation[WeightGroupSize];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out < n;
  const int safe_out = valid_output ? out : n - 1;
  const int activation_groups = k / ActivationGroupSize;
  const int num_routes = *num_routes_ptr;
  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token / top_k;
    const int64_t weight_output = static_cast<int64_t>(expert) * n + safe_out;
    const int* weight_words = reinterpret_cast<const int*>(
        weight + weight_output * static_cast<int64_t>(packed_k));
    const int64_t weight_meta = weight_output * num_weight_groups;
    const int64_t activation_meta =
        static_cast<int64_t>(input_row) * activation_groups;
    float accumulator = 0.0f;
    for (int weight_group = 0; weight_group < num_weight_groups;
         ++weight_group) {
#pragma unroll
      for (int item = 0; item < decode_items; ++item) {
        const int local_k = threadIdx.x + item * kThreads;
        const int global_k = weight_group * WeightGroupSize + local_k;
        const int activation_group = global_k / ActivationGroupSize;
        const int64_t metadata = activation_meta + activation_group;
        const int q =
            input_code[static_cast<int64_t>(input_row) * k + global_k];
        const float t = fabsf(static_cast<float>(q)) * (1.0f / 127.0f);
        float value = input_scale[metadata] *
                      cubic8_curve(t, __half2float(input_a[metadata]),
                                   __half2float(input_b[metadata]));
        decoded_activation[local_k] = q < 0 ? -value : value;
      }
      __syncthreads();
      float partial = 0.0f;
#pragma unroll
      for (int index = 0; index < words_per_lane; ++index) {
        const int word = lane + index * ThreadsPerOutput;
        const unsigned packed_weight =
            weight_words[weight_group * words_per_group + word];
#pragma unroll
        for (int element = 0; element < 16; ++element) {
          const int weight_code = (packed_weight >> (element * 2)) & 3;
          const int ternary = (weight_code & 1) * (1 - (weight_code & 2));
          partial += static_cast<float>(ternary) *
                     decoded_activation[word * 16 + element];
        }
      }
      __syncthreads();
      accumulator += partial * weight_scale[weight_meta + weight_group];
    }
#pragma unroll
    for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
      accumulator +=
          __shfl_down_sync(0xffffffffu, accumulator, delta, ThreadsPerOutput);
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token];
      output[static_cast<int64_t>(token) * n + out] =
          __float2bfloat16(accumulator);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float cubic8_curve(float t, float a, float b) {
  return t * (a + t * (b + t * (1.0f - a - b)));
}

__device__ __forceinline__ float cubic8_warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, delta);
  return value;
}

__device__ __forceinline__ float cubic8_warp_max(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
  return value;
}

template <bool TakeMax, int BlockThreads>
__device__ __forceinline__ float cubic8_block_reduce(float value,
                                                     float* scratch) {
  constexpr int num_warps = BlockThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = TakeMax ? cubic8_warp_max(value) : cubic8_warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < num_warps ? scratch[lane] : 0.0f;
  if (warp == 0)
    value = TakeMax ? cubic8_warp_max(value) : cubic8_warp_sum(value);
  if (threadIdx.x == 0) scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

template <int WeightGroupSize, int OutputGroupSize>
__global__ void cubic_w2_situ_cubic8_producer_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    int8_t* __restrict__ output_code, float* __restrict__ output_scale,
    __half* __restrict__ output_a, __half* __restrict__ output_b,
    const float* __restrict__ topk_weights, const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_routes_ptr,
    int n, int k, int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta) {
  constexpr int kThreads = 256;
  constexpr int outputs_per_thread = OutputGroupSize / kThreads;
  static_assert(OutputGroupSize == 256 || OutputGroupSize == 512);
  static_assert(outputs_per_thread == 1 || outputs_per_thread == 2);
  constexpr int words_per_group = WeightGroupSize / 16;
  constexpr int input_words_per_group = WeightGroupSize / 4;
  __shared__ float reduction_scratch[8];

  const int output_group = blockIdx.x;
  const int output_base = output_group * OutputGroupSize;
  const int num_output_groups = n / OutputGroupSize;
  const int input_words_per_row = k / 4;
  const int num_routes = *num_routes_ptr;
  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token / top_k;
    const int* input_words =
        reinterpret_cast<const int*>(input) +
        static_cast<int64_t>(input_row) * input_words_per_row;
    const float activation_scale = input_scale[input_row];
    float activated[outputs_per_thread];
#pragma unroll
    for (int item = 0; item < outputs_per_thread; ++item) {
      const int out = output_base + threadIdx.x + item * kThreads;
      const int64_t gate_output = static_cast<int64_t>(expert) * (2 * n) + out;
      const int* gate_weight = reinterpret_cast<const int*>(
          weight + gate_output * static_cast<int64_t>(packed_k));
      const int* up_weight =
          gate_weight + static_cast<int64_t>(n) * packed_k / 4;
      const int64_t gate_meta = gate_output * num_groups;
      const int64_t up_meta = gate_meta + static_cast<int64_t>(n) * num_groups;
      float gate = 0.0f;
      float up = 0.0f;
      for (int group = 0; group < num_groups; ++group) {
        int gate_dot = 0;
        int up_dot = 0;
#pragma unroll
        for (int word = 0; word < words_per_group; ++word) {
          const unsigned gate_packed =
              gate_weight[group * words_per_group + word];
          const unsigned up_packed = up_weight[group * words_per_group + word];
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            const int input_word =
                input_words[group * input_words_per_group + word * 4 + byte];
            gate_dot =
                cubic_w2_dp4a(cubic_w2_carrier_word(gate_packed >> (byte * 8)),
                              input_word, gate_dot);
            up_dot =
                cubic_w2_dp4a(cubic_w2_carrier_word(up_packed >> (byte * 8)),
                              input_word, up_dot);
          }
        }
        gate += static_cast<float>(gate_dot) * activation_scale *
                weight_scale[gate_meta + group];
        up += static_cast<float>(up_dot) * activation_scale *
              weight_scale[up_meta + group];
      }
      if (multiply_routed_weight) {
        const float routed = topk_weights[token];
        gate *= routed;
        up *= routed;
      }
      gate = __bfloat162float(__float2bfloat16(gate));
      up = __bfloat162float(__float2bfloat16(up));
      const float gate_tanh =
          2.0f / (1.0f + __expf(-2.0f * gate / beta)) - 1.0f;
      gate = beta * gate_tanh / (1.0f + __expf(-gate));
      if (has_linear_beta)
        up = linear_beta *
             (2.0f / (1.0f + __expf(-2.0f * up / linear_beta)) - 1.0f);
      activated[item] = __bfloat162float(__float2bfloat16(gate * up));
    }

    float local_amax = 0.0f;
#pragma unroll
    for (int item = 0; item < outputs_per_thread; ++item)
      local_amax = fmaxf(local_amax, fabsf(activated[item]));
    const float amax = fmaxf(
        cubic8_block_reduce<true, kThreads>(local_amax, reduction_scratch),
        1.0e-30f);
    const float output_scale_value = amax * (1.0f / 127.0f);
    const float inverse_scale = 1.0f / output_scale_value;
#pragma unroll
    for (int item = 0; item < outputs_per_thread; ++item) {
      const float value = activated[item];
      const int code =
          max(-127, min(127, __float2int_rn(value * inverse_scale)));
      output_code[static_cast<int64_t>(token) * n + output_base + threadIdx.x +
                  item * kThreads] = static_cast<int8_t>(code);
    }
    if (threadIdx.x == 0) {
      const int64_t metadata =
          static_cast<int64_t>(token) * num_output_groups + output_group;
      output_scale[metadata] = output_scale_value;
      output_a[metadata] = __float2half(1.0f);
      output_b[metadata] = __float2half(0.0f);
    }
    __syncthreads();
  }
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false,
          int ActivationGroupSize = GroupSize>
__global__ void cubic_w2_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int top_k, int route_ctas, bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int activation_groups_per_weight = GroupSize / ActivationGroupSize;
  constexpr int words_per_activation_group = ActivationGroupSize / 16;
  constexpr int words_per_thread =
      words_per_activation_group / ThreadsPerOutput;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(GroupSize % ActivationGroupSize == 0);
  static_assert(words_per_activation_group % ThreadsPerOutput == 0);

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
    const int* input_words =
        reinterpret_cast<const int*>(input) +
        static_cast<int64_t>(input_row) * input_words_per_row;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base = expert_output * num_groups;
    const float row_scale = GroupwiseScale ? 0.0f : input_scale[input_row];
    float accumulator = 0.0f;

    for (int group = 0; group < num_groups; ++group) {
#pragma unroll
      for (int activation_group = 0;
           activation_group < activation_groups_per_weight;
           ++activation_group) {
        int dot = 0;
#pragma unroll
        for (int index = 0; index < words_per_thread; ++index) {
          const int word_in_activation_group = lane + index * ThreadsPerOutput;
          const int word_in_weight_group =
              activation_group * words_per_activation_group +
              word_in_activation_group;
          const unsigned packed =
              weight_words[group * (GroupSize / 16) + word_in_weight_group];
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            const int activation = input_words[group * input_words_per_group +
                                               word_in_weight_group * 4 + byte];
            dot = cubic_w2_dp4a(cubic_w2_carrier_word(packed >> (byte * 8)),
                                activation, dot);
          }
        }
        for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
          dot += __shfl_down_sync(mask, dot, delta, ThreadsPerOutput);
        if (lane == 0) {
          const int64_t meta = meta_base + group;
          const float activation_scale =
              GroupwiseScale
                  ? input_scale[static_cast<int64_t>(input_row) *
                                    (num_groups *
                                     activation_groups_per_weight) +
                                group * activation_groups_per_weight +
                                activation_group]
                  : row_scale;
          accumulator +=
              static_cast<float>(dot) * activation_scale * weight_scale[meta];
        }
      }
    }
    if (lane == 0) {
      if (multiply_routed_weight) accumulator *= topk_weights[token_id];
      output[static_cast<int64_t>(token_id) * n + out_channel] =
          __float2bfloat16(accumulator);
    }
  }
}

template <int WeightGroupSize, int OutputGroupSize, int ThreadsPerOutput,
          int RoutesPerBlock = 1>
__global__ void cubic_w2_situ_cubic8_subgroup_producer_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    int8_t* __restrict__ output_code, float* __restrict__ output_scale,
    __half* __restrict__ output_a, __half* __restrict__ output_b,
    const float* __restrict__ topk_weights, const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_routes_ptr,
    int num_valid_tokens, int n, int k, int num_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta) {
  constexpr int kThreads = OutputGroupSize * ThreadsPerOutput < 512
                               ? OutputGroupSize * ThreadsPerOutput
                               : 512;
  constexpr int outputs_per_wave = kThreads / ThreadsPerOutput;
  constexpr int output_waves = OutputGroupSize / outputs_per_wave;
  constexpr int words_per_group = WeightGroupSize / 16;
  constexpr int words_per_lane = words_per_group / ThreadsPerOutput;
  constexpr int input_words_per_group = WeightGroupSize / 4;
  static_assert(OutputGroupSize % outputs_per_wave == 0);
  static_assert(words_per_group % ThreadsPerOutput == 0);
  __shared__ float activated_group[RoutesPerBlock][OutputGroupSize];
  __shared__ float reduction_scratch[kThreads / 32];

  const int output_lane = threadIdx.x / ThreadsPerOutput;
  const int k_lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int num_output_groups = n / OutputGroupSize;
  const int input_words_per_row = k / 4;
  const int num_route_blocks =
      (*num_routes_ptr + RoutesPerBlock - 1) / RoutesPerBlock;
  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    int token[RoutesPerBlock];
    bool valid[RoutesPerBlock];
    int input_row[RoutesPerBlock];
    const int* input_words[RoutesPerBlock];
    float activation_scale[RoutesPerBlock];
#pragma unroll
    for (int route = 0; route < RoutesPerBlock; ++route) {
      token[route] = token_ids[route_block * RoutesPerBlock + route];
      valid[route] = static_cast<unsigned>(token[route]) <
                     static_cast<unsigned>(num_valid_tokens);
      input_row[route] = valid[route] ? token[route] / top_k : 0;
      input_words[route] =
          reinterpret_cast<const int*>(input) +
          static_cast<int64_t>(input_row[route]) * input_words_per_row;
      activation_scale[route] =
          valid[route] ? input_scale[input_row[route]] : 0.0f;
    }
    const int expert = expert_ids[route_block];
    const bool valid_expert = expert >= 0;
#pragma unroll
    for (int wave = 0; wave < output_waves; ++wave) {
      const int output_in_group = wave * outputs_per_wave + output_lane;
      const int out = blockIdx.x * OutputGroupSize + output_in_group;
      const int64_t gate_output =
          static_cast<int64_t>(valid_expert ? expert : 0) * (2 * n) + out;
      const int* gate_weight = reinterpret_cast<const int*>(
          weight + gate_output * static_cast<int64_t>(packed_k));
      const int* up_weight =
          gate_weight + static_cast<int64_t>(n) * packed_k / 4;
      const int64_t gate_meta = gate_output * num_groups;
      const int64_t up_meta = gate_meta + static_cast<int64_t>(n) * num_groups;
      float gate[RoutesPerBlock] = {};
      float up[RoutesPerBlock] = {};
      for (int group = 0; group < num_groups; ++group) {
        int gate_dot[RoutesPerBlock] = {};
        int up_dot[RoutesPerBlock] = {};
#pragma unroll
        for (int index = 0; index < words_per_lane; ++index) {
          const int word = k_lane + index * ThreadsPerOutput;
          const unsigned gate_packed =
              gate_weight[group * words_per_group + word];
          const unsigned up_packed = up_weight[group * words_per_group + word];
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            const int gate_carrier =
                cubic_w2_carrier_word(gate_packed >> (byte * 8));
            const int up_carrier =
                cubic_w2_carrier_word(up_packed >> (byte * 8));
#pragma unroll
            for (int route = 0; route < RoutesPerBlock; ++route) {
              const int input_word =
                  input_words[route]
                             [group * input_words_per_group + word * 4 + byte];
              gate_dot[route] =
                  cubic_w2_dp4a(gate_carrier, input_word, gate_dot[route]);
              up_dot[route] =
                  cubic_w2_dp4a(up_carrier, input_word, up_dot[route]);
            }
          }
        }
#pragma unroll
        for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
#pragma unroll
          for (int route = 0; route < RoutesPerBlock; ++route) {
            gate_dot[route] += __shfl_down_sync(0xffffffffu, gate_dot[route],
                                                delta, ThreadsPerOutput);
            up_dot[route] += __shfl_down_sync(0xffffffffu, up_dot[route], delta,
                                              ThreadsPerOutput);
          }
        }
        if (k_lane == 0) {
#pragma unroll
          for (int route = 0; route < RoutesPerBlock; ++route) {
            gate[route] += static_cast<float>(gate_dot[route]) *
                           activation_scale[route] *
                           weight_scale[gate_meta + group];
            up[route] += static_cast<float>(up_dot[route]) *
                         activation_scale[route] *
                         weight_scale[up_meta + group];
          }
        }
      }
      if (k_lane == 0) {
#pragma unroll
        for (int route = 0; route < RoutesPerBlock; ++route) {
          if (multiply_routed_weight && valid[route]) {
            const float routed = topk_weights[token[route]];
            gate[route] *= routed;
            up[route] *= routed;
          }
          gate[route] = __bfloat162float(__float2bfloat16(gate[route]));
          up[route] = __bfloat162float(__float2bfloat16(up[route]));
          const float gate_tanh =
              2.0f / (1.0f + __expf(-2.0f * gate[route] / beta)) - 1.0f;
          gate[route] = beta * gate_tanh / (1.0f + __expf(-gate[route]));
          if (has_linear_beta)
            up[route] =
                linear_beta *
                (2.0f / (1.0f + __expf(-2.0f * up[route] / linear_beta)) -
                 1.0f);
          activated_group[route][output_in_group] =
              valid_expert && valid[route]
                  ? __bfloat162float(__float2bfloat16(gate[route] * up[route]))
                  : 0.0f;
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int route = 0; route < RoutesPerBlock; ++route) {
      const bool fit_lane = threadIdx.x < OutputGroupSize;
      const float value = fit_lane ? activated_group[route][threadIdx.x] : 0.0f;
      const float amax = fmaxf(
          cubic8_block_reduce<true, kThreads>(fabsf(value), reduction_scratch),
          1.0e-30f);
      const float output_scale_value = amax * (1.0f / 127.0f);
      const float inverse_scale = 1.0f / output_scale_value;
      if (fit_lane && valid[route]) {
        const int code =
            max(-127, min(127, __float2int_rn(value * inverse_scale)));
        output_code[static_cast<int64_t>(token[route]) * n +
                    blockIdx.x * OutputGroupSize + threadIdx.x] =
            static_cast<int8_t>(code);
      }
      if (threadIdx.x == 0 && valid[route]) {
        const int64_t metadata =
            static_cast<int64_t>(token[route]) * num_output_groups + blockIdx.x;
        output_scale[metadata] = output_scale_value;
        output_a[metadata] = __float2half(1.0f);
        output_b[metadata] = __float2half(0.0f);
      }
      __syncthreads();
    }
  }
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
__global__ void cubic_w2_grouped2_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int num_valid_tokens, int n, int k,
    int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize / 16;
  constexpr int words_per_thread = words_per_group / ThreadsPerOutput;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(words_per_group % ThreadsPerOutput == 0);

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  if (out_channel >= n) return;
  const int num_route_blocks = (*num_routes_ptr + 1) / 2;
  const int input_words_per_row = k / 4;

  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    const int token0 = token_ids[route_block * 2];
    const int token1 = token_ids[route_block * 2 + 1];
    const bool valid0 =
        static_cast<unsigned>(token0) < static_cast<unsigned>(num_valid_tokens);
    const bool valid1 =
        static_cast<unsigned>(token1) < static_cast<unsigned>(num_valid_tokens);
    const int expert = expert_ids[route_block];
    if (expert < 0 || (!valid0 && !valid1)) continue;
    const int row0 = valid0 ? token0 / top_k : 0;
    const int row1 = valid1 ? token1 / top_k : 0;
    const int* input0 = reinterpret_cast<const int*>(input) +
                        static_cast<int64_t>(row0) * input_words_per_row;
    const int* input1 = reinterpret_cast<const int*>(input) +
                        static_cast<int64_t>(row1) * input_words_per_row;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base = expert_output * num_groups;
    const float row_scale0 =
        valid0 && !GroupwiseScale ? input_scale[row0] : 0.0f;
    const float row_scale1 =
        valid1 && !GroupwiseScale ? input_scale[row1] : 0.0f;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (int group = 0; group < num_groups; ++group) {
      int dot0 = 0;
      int dot1 = 0;
#pragma unroll
      for (int index = 0; index < words_per_thread; ++index) {
        const int word_in_group = lane + index * ThreadsPerOutput;
        const unsigned packed =
            weight_words[group * words_per_group + word_in_group];
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          const int carrier = cubic_w2_carrier_word(packed >> (byte * 8));
          const int input_index =
              group * input_words_per_group + word_in_group * 4 + byte;
          if (valid0) dot0 = cubic_w2_dp4a(carrier, input0[input_index], dot0);
          if (valid1) dot1 = cubic_w2_dp4a(carrier, input1[input_index], dot1);
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
        dot0 += __shfl_down_sync(mask, dot0, delta, ThreadsPerOutput);
        dot1 += __shfl_down_sync(mask, dot1, delta, ThreadsPerOutput);
      }
      if (lane == 0) {
        const float ws = weight_scale[meta_base + group];
        const float act_scale0 =
            valid0 && GroupwiseScale
                ? input_scale[static_cast<int64_t>(row0) * num_groups + group]
                : row_scale0;
        const float act_scale1 =
            valid1 && GroupwiseScale
                ? input_scale[static_cast<int64_t>(row1) * num_groups + group]
                : row_scale1;
        accumulator0 += (act_scale0 * ws) * static_cast<float>(dot0);
        accumulator1 += (act_scale1 * ws) * static_cast<float>(dot1);
      }
    }
    if (lane == 0) {
      if (valid0) {
        if (multiply_routed_weight) accumulator0 *= topk_weights[token0];
        output[static_cast<int64_t>(token0) * n + out_channel] =
            __float2bfloat16(accumulator0);
      }
      if (valid1) {
        if (multiply_routed_weight) accumulator1 *= topk_weights[token1];
        output[static_cast<int64_t>(token1) * n + out_channel] =
            __float2bfloat16(accumulator1);
      }
    }
  }
}

template <int GroupSize, int ThreadsPerOutput>
__global__ void cubic_w2_grouped2_situ_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int num_valid_tokens, int n, int k,
    int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize / 16;
  constexpr int words_per_thread = words_per_group / ThreadsPerOutput;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(words_per_group % ThreadsPerOutput == 0);

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  if (out_channel >= n) return;
  const int num_route_blocks = (*num_routes_ptr + 1) / 2;
  const int input_words_per_row = k / 4;

  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    const int token0 = token_ids[route_block * 2];
    const int token1 = token_ids[route_block * 2 + 1];
    const bool valid0 =
        static_cast<unsigned>(token0) < static_cast<unsigned>(num_valid_tokens);
    const bool valid1 =
        static_cast<unsigned>(token1) < static_cast<unsigned>(num_valid_tokens);
    const int expert = expert_ids[route_block];
    const int row0 = valid0 ? token0 / top_k : 0;
    const int row1 = valid1 ? token1 / top_k : 0;
    const int* input0 = reinterpret_cast<const int*>(input) +
                        static_cast<int64_t>(row0) * input_words_per_row;
    const int* input1 = reinterpret_cast<const int*>(input) +
                        static_cast<int64_t>(row1) * input_words_per_row;
    if (expert >= 0 && valid0) {
      const int64_t expert_output =
          static_cast<int64_t>(expert) * (2 * n) + out_channel;
      const int* gate_weight =
          reinterpret_cast<const int*>(weight + expert_output * packed_k);
      const int* up_weight =
          gate_weight + static_cast<int64_t>(n) * packed_k / 4;
      const int64_t gate_meta = expert_output * num_groups;
      const int64_t up_meta = gate_meta + static_cast<int64_t>(n) * num_groups;
      const float act_scale0 = input_scale[row0];
      const float act_scale1 = valid1 ? input_scale[row1] : 0.0f;
      float gate0 = 0.0f, gate1 = 0.0f, up0 = 0.0f, up1 = 0.0f;

      for (int group = 0; group < num_groups; ++group) {
        int gate_dot0 = 0, gate_dot1 = 0, up_dot0 = 0, up_dot1 = 0;
#pragma unroll
        for (int index = 0; index < words_per_thread; ++index) {
          const int word_in_group = lane + index * ThreadsPerOutput;
          const int weight_index = group * words_per_group + word_in_group;
          const unsigned gate_packed = gate_weight[weight_index];
          const unsigned up_packed = up_weight[weight_index];
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            const int gate_carrier =
                cubic_w2_carrier_word(gate_packed >> (byte * 8));
            const int up_carrier =
                cubic_w2_carrier_word(up_packed >> (byte * 8));
            const int input_index =
                group * input_words_per_group + word_in_group * 4 + byte;
            const int act = input0[input_index];
            gate_dot0 = cubic_w2_dp4a(gate_carrier, act, gate_dot0);
            up_dot0 = cubic_w2_dp4a(up_carrier, act, up_dot0);
            if (valid1) {
              const int act1 = input1[input_index];
              gate_dot1 = cubic_w2_dp4a(gate_carrier, act1, gate_dot1);
              up_dot1 = cubic_w2_dp4a(up_carrier, act1, up_dot1);
            }
          }
        }
        for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
          gate_dot0 +=
              __shfl_down_sync(mask, gate_dot0, delta, ThreadsPerOutput);
          gate_dot1 +=
              __shfl_down_sync(mask, gate_dot1, delta, ThreadsPerOutput);
          up_dot0 += __shfl_down_sync(mask, up_dot0, delta, ThreadsPerOutput);
          up_dot1 += __shfl_down_sync(mask, up_dot1, delta, ThreadsPerOutput);
        }
        if (lane == 0) {
          const float gs = weight_scale[gate_meta + group];
          const float us = weight_scale[up_meta + group];
          gate0 += (act_scale0 * gs) * static_cast<float>(gate_dot0);
          gate1 += (act_scale1 * gs) * static_cast<float>(gate_dot1);
          up0 += (act_scale0 * us) * static_cast<float>(up_dot0);
          up1 += (act_scale1 * us) * static_cast<float>(up_dot1);
        }
      }
      if (lane == 0) {
        auto store_situ = [&](int token, float gate, float up) {
          if (multiply_routed_weight) {
            const float routed = topk_weights[token];
            gate *= routed;
            up *= routed;
          }
          gate = __bfloat162float(__float2bfloat16(gate));
          up = __bfloat162float(__float2bfloat16(up));
          const float gate_tanh =
              2.0f / (1.0f + __expf(-2.0f * gate / beta)) - 1.0f;
          const float gate_sigmoid = 1.0f / (1.0f + __expf(-gate));
          gate = beta * gate_tanh * gate_sigmoid;
          if (has_linear_beta) {
            up = linear_beta *
                 (2.0f / (1.0f + __expf(-2.0f * up / linear_beta)) - 1.0f);
          }
          output[static_cast<int64_t>(token) * n + out_channel] =
              __float2bfloat16(gate * up);
        };
        store_situ(token0, gate0, up0);
        if (valid1) store_situ(token1, gate1, up1);
      }
    }
  }
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false,
          int ActivationGroupSize = GroupSize>
void launch_cubic_w2_a8(const int8_t* input, const float* input_scale,
                        const uint8_t* weight, const float* weight_scale,
                        __nv_bfloat16* output, const float* topk_weights,
                        const int* token_ids, const int* expert_ids,
                        const int* num_routes, int n, int k, int num_groups,
                        int packed_k, int top_k, int route_ctas,
                        bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_a8_gemv_kernel<GroupSize, ThreadsPerOutput, GroupwiseScale,
                          ActivationGroupSize><<<grid, 128, 0, stream>>>(
      input, input_scale, weight, weight_scale, output, topk_weights, token_ids,
      expert_ids, num_routes, n, k, num_groups, packed_k, top_k, route_ctas,
      multiply_routed_weight);
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
void launch_cubic_w2_cubic8_moment(
    const int8_t* input_code, const float* input_scale, const __half* input_a,
    const __half* input_b, const uint8_t* weight, const float* weight_scale,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int n, int k,
    int num_weight_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_cubic8_moment_gemv_kernel<WeightGroupSize, ActivationGroupSize,
                                     ThreadsPerOutput>
      <<<grid, 128, 0, stream>>>(input_code, input_scale, input_a, input_b,
                                 weight, weight_scale, output, topk_weights,
                                 token_ids, expert_ids, num_routes, n, k,
                                 num_weight_groups, packed_k, top_k, route_ctas,
                                 multiply_routed_weight);
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
void launch_cubic_w2_cubic8_lut(
    const int8_t* input_code, const float* input_scale, const __half* input_a,
    const __half* input_b, const uint8_t* weight, const float* weight_scale,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int n, int k, int num_groups,
    int packed_k, int top_k, int route_ctas, bool multiply_routed_weight,
    cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_cubic8_lut_gemv_kernel<WeightGroupSize, ActivationGroupSize,
                                  ThreadsPerOutput><<<grid, 128, 0, stream>>>(
      input_code, input_scale, input_a, input_b, weight, weight_scale, output,
      topk_weights, token_ids, expert_ids, num_routes, n, k, num_groups,
      packed_k, top_k, route_ctas, multiply_routed_weight);
}

template <int WeightGroupSize, int ActivationGroupSize, int ThreadsPerOutput>
void launch_cubic_w2_cubic8_shared(
    const int8_t* input_code, const float* input_scale, const __half* input_a,
    const __half* input_b, const uint8_t* weight, const float* weight_scale,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int n, int k,
    int num_weight_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_cubic8_shared_gemv_kernel<WeightGroupSize, ActivationGroupSize,
                                     ThreadsPerOutput>
      <<<grid, 128, 0, stream>>>(input_code, input_scale, input_a, input_b,
                                 weight, weight_scale, output, topk_weights,
                                 token_ids, expert_ids, num_routes, n, k,
                                 num_weight_groups, packed_k, top_k, route_ctas,
                                 multiply_routed_weight);
}

template <int GroupSize, int ThreadsPerOutput, bool GroupwiseScale = false>
void launch_cubic_w2_grouped2_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, __nv_bfloat16* output, const float* topk_weights,
    const int* token_ids, const int* expert_ids, const int* num_routes,
    int num_valid_tokens, int n, int k, int num_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_grouped2_a8_gemv_kernel<GroupSize, ThreadsPerOutput, GroupwiseScale>
      <<<grid, 128, 0, stream>>>(
          input, input_scale, weight, weight_scale, output, topk_weights,
          token_ids, expert_ids, num_routes, num_valid_tokens, n, k, num_groups,
          packed_k, top_k, route_ctas, multiply_routed_weight);
}

template <int GroupSize, int ThreadsPerOutput>
void launch_cubic_w2_grouped2_situ_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, __nv_bfloat16* output, const float* topk_weights,
    const int* token_ids, const int* expert_ids, const int* num_routes,
    int num_valid_tokens, int n, int k, int num_groups, int packed_k, int top_k,
    int route_ctas, bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w2_grouped2_situ_a8_gemv_kernel<GroupSize, ThreadsPerOutput>
      <<<grid, 128, 0, stream>>>(
          input, input_scale, weight, weight_scale, output, topk_weights,
          token_ids, expert_ids, num_routes, num_valid_tokens, n, k, num_groups,
          packed_k, top_k, route_ctas, multiply_routed_weight, beta,
          linear_beta, has_linear_beta);
}

template <int WeightGroupSize, int OutputGroupSize>
void launch_cubic_w2_situ_cubic8_producer(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, int8_t* output_code, float* output_scale,
    __half* output_a, __half* output_b, const float* topk_weights,
    const int* token_ids, const int* expert_ids, const int* num_routes, int n,
    int k, int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta, cudaStream_t stream) {
  dim3 grid(n / OutputGroupSize, route_ctas);
  cubic_w2_situ_cubic8_producer_kernel<WeightGroupSize, OutputGroupSize>
      <<<grid, 256, 0, stream>>>(
          input, input_scale, weight, weight_scale, output_code, output_scale,
          output_a, output_b, topk_weights, token_ids, expert_ids, num_routes,
          n, k, num_groups, packed_k, top_k, route_ctas, multiply_routed_weight,
          beta, linear_beta, has_linear_beta);
}

template <int WeightGroupSize, int OutputGroupSize, int ThreadsPerOutput,
          int RoutesPerBlock = 1>
void launch_cubic_w2_situ_cubic8_subgroup_producer(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, int8_t* output_code, float* output_scale,
    __half* output_a, __half* output_b, const float* topk_weights,
    const int* token_ids, const int* expert_ids, const int* num_routes, int n,
    int k, int num_groups, int packed_k, int top_k, int route_ctas,
    bool multiply_routed_weight, float beta, float linear_beta,
    bool has_linear_beta, cudaStream_t stream) {
  constexpr int threads = OutputGroupSize * ThreadsPerOutput < 512
                              ? OutputGroupSize * ThreadsPerOutput
                              : 512;
  dim3 grid(n / OutputGroupSize, route_ctas);
  cubic_w2_situ_cubic8_subgroup_producer_kernel<
      WeightGroupSize, OutputGroupSize, ThreadsPerOutput, RoutesPerBlock>
      <<<grid, threads, 0, stream>>>(
          input, input_scale, weight, weight_scale, output_code, output_scale,
          output_a, output_b, topk_weights, token_ids, expert_ids, num_routes,
          INT_MAX, n, k, num_groups, packed_k, top_k, route_ctas,
          multiply_routed_weight, beta, linear_beta, has_linear_beta);
}

}  // namespace

void cubic_w2_a8_gemv(const torch::stable::Tensor& input,
                      const torch::stable::Tensor& input_scale,
                      const torch::stable::Tensor& weight,
                      const torch::stable::Tensor& weight_scale,
                      torch::stable::Tensor& output,
                      const torch::stable::Tensor& topk_weights,
                      const torch::stable::Tensor& token_ids,
                      const torch::stable::Tensor& expert_ids,
                      const torch::stable::Tensor& num_routes,
                      int64_t group_size, int64_t top_k,
                      bool multiply_routed_weight, int64_t route_ctas) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w2_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w2_a8_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                      weight_scale.is_contiguous() && output.is_contiguous(),
                  "cubic_w2_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK((group_size == 256 || group_size == 512) &&
                      k % group_size == 0 && weight.size(2) == k / 4,
                  "cubic_w2_a8_gemv: unsupported shape or group size");
  STD_TORCH_CHECK(route_ctas > 0 && route_ctas <= token_ids.numel(),
                  "cubic_w2_a8_gemv: invalid route_ctas");
  const auto stream = get_current_cuda_stream(input.get_device_index());
  const int groups = k / group_size;
#define CUBIC_W2_A8_ARGS                                                      \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      stream
  if (group_size == 256 && k <= 4096) {
    launch_cubic_w2_a8<256, 4>(CUBIC_W2_A8_ARGS);
  } else if (group_size == 256) {
    launch_cubic_w2_a8<256, 8>(CUBIC_W2_A8_ARGS);
  } else if (k <= 4096) {
    launch_cubic_w2_a8<512, 4>(CUBIC_W2_A8_ARGS);
  } else {
    launch_cubic_w2_a8<512, 8>(CUBIC_W2_A8_ARGS);
  }
#undef CUBIC_W2_A8_ARGS
}

void cubic_w2_groupwise_a8_gemv(const torch::stable::Tensor& input,
                                const torch::stable::Tensor& input_scale,
                                const torch::stable::Tensor& weight,
                                const torch::stable::Tensor& weight_scale,
                                torch::stable::Tensor& output,
                                const torch::stable::Tensor& topk_weights,
                                const torch::stable::Tensor& token_ids,
                                const torch::stable::Tensor& expert_ids,
                                const torch::stable::Tensor& num_routes,
                                int64_t group_size, int64_t top_k,
                                bool multiply_routed_weight, int64_t route_ctas,
                                int64_t num_valid_tokens,
                                int64_t routes_per_block) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w2_groupwise_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w2_groupwise_a8_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      output.is_contiguous(),
                  "cubic_w2_groupwise_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  const int activation_groups = input_scale.size(1);
  const int activation_group_size = k / activation_groups;
  STD_TORCH_CHECK(
      (group_size == 256 || group_size == 512) && k % group_size == 0 &&
          weight.size(2) == k / 4 && input_scale.dim() == 2 &&
          input_scale.size(0) == input.size(0) &&
          (activation_group_size == group_size ||
           (group_size == 512 && activation_group_size == 256)),
      "cubic_w2_groupwise_a8_gemv: unsupported shape or scale layout");
  STD_TORCH_CHECK(
      (routes_per_block == 1 || routes_per_block == 2) && route_ctas > 0,
      "cubic_w2_groupwise_a8_gemv: invalid route_ctas");
  const auto stream = get_current_cuda_stream(input.get_device_index());
  const int groups = k / group_size;
#define CUBIC_W2_ONLINE_ARGS                                                  \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      stream
#define CUBIC_W2_ONLINE_GROUPED_ARGS                                          \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),     \
      num_valid_tokens, n, k, groups, weight.size(2), top_k, route_ctas,      \
      multiply_routed_weight, stream
  const bool small_k = k <= 4096;
  if (routes_per_block == 2) {
    STD_TORCH_CHECK(activation_group_size == group_size,
                    "paired routes require matching activation groups");
    if (group_size == 256 && small_k) {
      launch_cubic_w2_grouped2_a8<256, 4, true>(CUBIC_W2_ONLINE_GROUPED_ARGS);
    } else if (group_size == 256) {
      launch_cubic_w2_grouped2_a8<256, 8, true>(CUBIC_W2_ONLINE_GROUPED_ARGS);
    } else if (small_k) {
      launch_cubic_w2_grouped2_a8<512, 4, true>(CUBIC_W2_ONLINE_GROUPED_ARGS);
    } else {
      launch_cubic_w2_grouped2_a8<512, 8, true>(CUBIC_W2_ONLINE_GROUPED_ARGS);
    }
  } else if (group_size == 512 && activation_group_size == 256 && small_k) {
    launch_cubic_w2_a8<512, 4, true, 256>(CUBIC_W2_ONLINE_ARGS);
  } else if (group_size == 512 && activation_group_size == 256) {
    launch_cubic_w2_a8<512, 8, true, 256>(CUBIC_W2_ONLINE_ARGS);
  } else if (group_size == 256 && small_k) {
    launch_cubic_w2_a8<256, 4, true>(CUBIC_W2_ONLINE_ARGS);
  } else if (group_size == 256) {
    launch_cubic_w2_a8<256, 8, true>(CUBIC_W2_ONLINE_ARGS);
  } else if (small_k) {
    launch_cubic_w2_a8<512, 4, true>(CUBIC_W2_ONLINE_ARGS);
  } else {
    launch_cubic_w2_a8<512, 8, true>(CUBIC_W2_ONLINE_ARGS);
  }
#undef CUBIC_W2_ONLINE_GROUPED_ARGS
#undef CUBIC_W2_ONLINE_ARGS
}

void cubic_w2_cubic8_moment_gemv(
    const torch::stable::Tensor& input_code,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& input_a, const torch::stable::Tensor& input_b,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale, torch::stable::Tensor& output,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t weight_group_size,
    int64_t activation_group_size, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas) {
  STD_TORCH_CHECK(input_code.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w2_cubic8_moment_gemv: tensors must be CUDA");
  STD_TORCH_CHECK(
      input_code.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          input_a.scalar_type() == torch::headeronly::ScalarType::Half &&
          input_b.scalar_type() == torch::headeronly::ScalarType::Half &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w2_cubic8_moment_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input_code.is_contiguous() && input_scale.is_contiguous() &&
                      input_a.is_contiguous() && input_b.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      output.is_contiguous(),
                  "cubic_w2_cubic8_moment_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input_code.size(1);
  STD_TORCH_CHECK(
      (weight_group_size == 256 || weight_group_size == 512) &&
          (activation_group_size == 16 || activation_group_size == 32 ||
           activation_group_size == 64 || activation_group_size == 128 ||
           activation_group_size == 256 || activation_group_size == 512) &&
          k % weight_group_size == 0 && k % activation_group_size == 0 &&
          weight.size(2) == k / 4 &&
          input_scale.size(0) == input_code.size(0) &&
          input_scale.size(1) == k / activation_group_size &&
          input_a.size(0) == input_scale.size(0) &&
          input_a.size(1) == input_scale.size(1) &&
          input_b.size(0) == input_scale.size(0) &&
          input_b.size(1) == input_scale.size(1),
      "cubic_w2_cubic8_moment_gemv: unsupported shape");
  STD_TORCH_CHECK(route_ctas > 0 && route_ctas <= token_ids.numel(),
                  "cubic_w2_cubic8_moment_gemv: invalid route_ctas");
  const auto stream = get_current_cuda_stream(input_code.get_device_index());
  const int weight_groups = k / weight_group_size;
#define CUBIC_W2_CUBIC8_MOMENT_ARGS                                           \
  input_code.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),   \
      reinterpret_cast<const __half*>(input_a.const_data_ptr()),              \
      reinterpret_cast<const __half*>(input_b.const_data_ptr()),              \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, weight_groups, weight.size(2), top_k, route_ctas,                    \
      multiply_routed_weight, stream
#define CUBIC_W2_CUBIC8_ACTIVATION_CASE(WG, AG)                            \
  case AG:                                                                 \
    launch_cubic_w2_cubic8_moment<WG, AG, 8>(CUBIC_W2_CUBIC8_MOMENT_ARGS); \
    break
  if (weight_group_size == 256) {
    switch (activation_group_size) {
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 16);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 32);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 64);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 128);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 256);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(256, 512);
    }
  } else {
    switch (activation_group_size) {
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 16);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 32);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 64);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 128);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 256);
      CUBIC_W2_CUBIC8_ACTIVATION_CASE(512, 512);
    }
  }
#undef CUBIC_W2_CUBIC8_ACTIVATION_CASE
#undef CUBIC_W2_CUBIC8_MOMENT_ARGS
}

void cubic_w2_cubic8_lut_gemv(
    const torch::stable::Tensor& input_code,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& input_a, const torch::stable::Tensor& input_b,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale, torch::stable::Tensor& output,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t weight_group_size,
    int64_t activation_group_size, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas, int64_t threads_per_output) {
  STD_TORCH_CHECK(input_code.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w2_cubic8_lut_gemv: tensors must be CUDA");
  STD_TORCH_CHECK(
      input_code.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          input_a.scalar_type() == torch::headeronly::ScalarType::Half &&
          input_b.scalar_type() == torch::headeronly::ScalarType::Half &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w2_cubic8_lut_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input_code.is_contiguous() && input_scale.is_contiguous() &&
                      input_a.is_contiguous() && input_b.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      output.is_contiguous(),
                  "cubic_w2_cubic8_lut_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input_code.size(1);
  STD_TORCH_CHECK(
      (weight_group_size == 256 || weight_group_size == 512) &&
          (activation_group_size == 128 || activation_group_size == 256 ||
           activation_group_size == 512) &&
          weight_group_size % activation_group_size == 0 &&
          k % weight_group_size == 0 && k % activation_group_size == 0 &&
          weight.size(2) == k / 4 &&
          input_scale.size(0) == input_code.size(0) &&
          input_scale.size(1) == k / activation_group_size &&
          input_a.size(0) == input_scale.size(0) &&
          input_a.size(1) == input_scale.size(1) &&
          input_b.size(0) == input_scale.size(0) &&
          input_b.size(1) == input_scale.size(1),
      "cubic_w2_cubic8_lut_gemv: unsupported shape");
  STD_TORCH_CHECK(route_ctas > 0 && route_ctas <= token_ids.numel(),
                  "cubic_w2_cubic8_lut_gemv: invalid route_ctas");
  STD_TORCH_CHECK(threads_per_output == 2 || threads_per_output == 4 ||
                      threads_per_output == 8,
                  "cubic_w2_cubic8_lut_gemv: invalid threads_per_output");
  const auto stream = get_current_cuda_stream(input_code.get_device_index());
  const int groups = k / weight_group_size;
#define CUBIC_W2_CUBIC8_LUT_ARGS                                              \
  input_code.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),   \
      reinterpret_cast<const __half*>(input_a.const_data_ptr()),              \
      reinterpret_cast<const __half*>(input_b.const_data_ptr()),              \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      stream
#define CUBIC_W2_CUBIC8_LUT_LAUNCH(WG, AG)                           \
  if (threads_per_output == 2)                                       \
    launch_cubic_w2_cubic8_lut<WG, AG, 2>(CUBIC_W2_CUBIC8_LUT_ARGS); \
  else if (threads_per_output == 4)                                  \
    launch_cubic_w2_cubic8_lut<WG, AG, 4>(CUBIC_W2_CUBIC8_LUT_ARGS); \
  else                                                               \
    launch_cubic_w2_cubic8_lut<WG, AG, 8>(CUBIC_W2_CUBIC8_LUT_ARGS)
  if (weight_group_size == 256 && activation_group_size == 128) {
    CUBIC_W2_CUBIC8_LUT_LAUNCH(256, 128);
  } else if (weight_group_size == 256) {
    CUBIC_W2_CUBIC8_LUT_LAUNCH(256, 256);
  } else if (activation_group_size == 128) {
    CUBIC_W2_CUBIC8_LUT_LAUNCH(512, 128);
  } else if (activation_group_size == 256) {
    CUBIC_W2_CUBIC8_LUT_LAUNCH(512, 256);
  } else {
    CUBIC_W2_CUBIC8_LUT_LAUNCH(512, 512);
  }
#undef CUBIC_W2_CUBIC8_LUT_LAUNCH
#undef CUBIC_W2_CUBIC8_LUT_ARGS
}

void cubic_w2_cubic8_shared_gemv(
    const torch::stable::Tensor& input_code,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& input_a, const torch::stable::Tensor& input_b,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale, torch::stable::Tensor& output,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t weight_group_size,
    int64_t activation_group_size, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas) {
  STD_TORCH_CHECK(input_code.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w2_cubic8_shared_gemv: tensors must be CUDA");
  STD_TORCH_CHECK(
      input_code.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          input_a.scalar_type() == torch::headeronly::ScalarType::Half &&
          input_b.scalar_type() == torch::headeronly::ScalarType::Half &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w2_cubic8_shared_gemv: invalid tensor dtype");
  const int n = weight.size(1);
  const int k = input_code.size(1);
  STD_TORCH_CHECK(
      (weight_group_size == 256 || weight_group_size == 512) &&
          (activation_group_size == 16 || activation_group_size == 32 ||
           activation_group_size == 64 || activation_group_size == 128 ||
           activation_group_size == 256 || activation_group_size == 512) &&
          k % weight_group_size == 0 && k % activation_group_size == 0 &&
          weight.size(2) == k / 4 &&
          input_scale.size(0) == input_code.size(0) &&
          input_scale.size(1) == k / activation_group_size &&
          input_a.size(0) == input_scale.size(0) &&
          input_a.size(1) == input_scale.size(1) &&
          input_b.size(0) == input_scale.size(0) &&
          input_b.size(1) == input_scale.size(1),
      "cubic_w2_cubic8_shared_gemv: unsupported shape");
  STD_TORCH_CHECK(route_ctas > 0 && route_ctas <= token_ids.numel(),
                  "cubic_w2_cubic8_shared_gemv: invalid route_ctas");
  const auto stream = get_current_cuda_stream(input_code.get_device_index());
  const int weight_groups = k / weight_group_size;
#define CUBIC_W2_CUBIC8_SHARED_ARGS                                           \
  input_code.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),   \
      reinterpret_cast<const __half*>(input_a.const_data_ptr()),              \
      reinterpret_cast<const __half*>(input_b.const_data_ptr()),              \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, weight_groups, weight.size(2), top_k, route_ctas,                    \
      multiply_routed_weight, stream
#define CUBIC_W2_CUBIC8_SHARED_CASE(WG, AG)                                \
  case AG:                                                                 \
    launch_cubic_w2_cubic8_shared<WG, AG, 8>(CUBIC_W2_CUBIC8_SHARED_ARGS); \
    break
  if (weight_group_size == 256) {
    switch (activation_group_size) {
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 16);
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 32);
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 64);
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 128);
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 256);
      CUBIC_W2_CUBIC8_SHARED_CASE(256, 512);
    }
  } else {
    switch (activation_group_size) {
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 16);
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 32);
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 64);
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 128);
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 256);
      CUBIC_W2_CUBIC8_SHARED_CASE(512, 512);
    }
  }
#undef CUBIC_W2_CUBIC8_SHARED_CASE
#undef CUBIC_W2_CUBIC8_SHARED_ARGS
}

void cubic_w2_situ_cubic8_producer(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale,
    torch::stable::Tensor& output_code, torch::stable::Tensor& output_scale,
    torch::stable::Tensor& output_a, torch::stable::Tensor& output_b,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t group_size,
    int64_t output_group_size, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas, double beta, double linear_beta, bool has_linear_beta,
    int64_t threads_per_output) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output_code.device().is_cuda(),
                  "cubic_w2_situ_cubic8_producer: tensors must be CUDA");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output_code.scalar_type() == torch::headeronly::ScalarType::Char &&
          output_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          output_a.scalar_type() == torch::headeronly::ScalarType::Half &&
          output_b.scalar_type() == torch::headeronly::ScalarType::Half,
      "cubic_w2_situ_cubic8_producer: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      output_code.is_contiguous() &&
                      output_scale.is_contiguous() &&
                      output_a.is_contiguous() && output_b.is_contiguous(),
                  "cubic_w2_situ_cubic8_producer: tensors must be contiguous");
  const int n = output_code.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK((group_size == 256 || group_size == 512) &&
                      (output_group_size == 16 || output_group_size == 32 ||
                       output_group_size == 64 || output_group_size == 128 ||
                       output_group_size == 256 || output_group_size == 512) &&
                      k % group_size == 0 && n % output_group_size == 0 &&
                      weight.size(1) == 2 * n && weight.size(2) == k / 4 &&
                      output_scale.size(0) == output_code.size(0) &&
                      output_scale.size(1) == n / output_group_size &&
                      output_a.dim() == 2 && output_b.dim() == 2 &&
                      output_a.size(0) == output_scale.size(0) &&
                      output_a.size(1) == output_scale.size(1) &&
                      output_b.size(0) == output_scale.size(0) &&
                      output_b.size(1) == output_scale.size(1),
                  "cubic_w2_situ_cubic8_producer: unsupported shape");
  STD_TORCH_CHECK(route_ctas > 0 && route_ctas <= token_ids.numel(),
                  "cubic_w2_situ_cubic8_producer: invalid route_ctas");
  STD_TORCH_CHECK(threads_per_output == 1 || threads_per_output == 2 ||
                      threads_per_output == 4 || threads_per_output == 8,
                  "cubic_w2_situ_cubic8_producer: invalid threads_per_output");
  const auto stream = get_current_cuda_stream(input.get_device_index());
  const int groups = k / group_size;
#define CUBIC_W2_CUBIC8_PRODUCER_ARGS                                         \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      output_code.mutable_data_ptr<int8_t>(),                                 \
      output_scale.mutable_data_ptr<float>(),                                 \
      reinterpret_cast<__half*>(output_a.mutable_data_ptr()),                 \
      reinterpret_cast<__half*>(output_b.mutable_data_ptr()),                 \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), top_k, route_ctas, multiply_routed_weight,   \
      static_cast<float>(beta), static_cast<float>(linear_beta),              \
      has_linear_beta, stream
  if (group_size == 256 && output_group_size == 16)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 16, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 16)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 16, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 32)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 32, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 32)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 32, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 64)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 64, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 64)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 64, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 128)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 128, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 128)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 128, 8>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 256 &&
           threads_per_output == 2)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 256, 2>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 256)
    launch_cubic_w2_situ_cubic8_subgroup_producer<256, 256, 4>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 256 &&
           threads_per_output == 2)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 256, 2>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 256)
    launch_cubic_w2_situ_cubic8_subgroup_producer<512, 256, 4>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 256 && output_group_size == 512)
    launch_cubic_w2_situ_cubic8_producer<256, 512>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else if (group_size == 512 && output_group_size == 512)
    launch_cubic_w2_situ_cubic8_producer<512, 512>(
        CUBIC_W2_CUBIC8_PRODUCER_ARGS);
  else
    STD_TORCH_CHECK(false, "cubic_w2_situ_cubic8_producer: unsupported groups");
#undef CUBIC_W2_CUBIC8_PRODUCER_ARGS
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("cubic_w2_a8_gemv", TORCH_BOX(&cubic_w2_a8_gemv));
  m.impl("cubic_w2_groupwise_a8_gemv", TORCH_BOX(&cubic_w2_groupwise_a8_gemv));
  m.impl("cubic_w2_situ_cubic8_producer",
         TORCH_BOX(&cubic_w2_situ_cubic8_producer));
  m.impl("cubic_w2_cubic8_moment_gemv",
         TORCH_BOX(&cubic_w2_cubic8_moment_gemv));
  m.impl("cubic_w2_cubic8_lut_gemv", TORCH_BOX(&cubic_w2_cubic8_lut_gemv));
  m.impl("cubic_w2_cubic8_shared_gemv",
         TORCH_BOX(&cubic_w2_cubic8_shared_gemv));
}
