/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SGL_HIP_NORM_CUH_
#define SGL_HIP_NORM_CUH_

#include <numeric>

#include "utils.h"

namespace sgl_hip {

namespace norm {

template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(
    T* __restrict__ input,
    T* __restrict__ weight,
    T* __restrict__ output,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_output,
    float weight_bias,
    float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = WARP_SIZE;
  const uint32_t num_warps = blockDim.y;
  // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // TODO (Hubert)
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // sum_sq += float(input_vec[j]) * float(input_vec[j]);  // TODO (Hubert)
      sum_sq += to_float<T>(input_vec[j]) * to_float<T>(input_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    // sum_sq += math::shfl_xor_sync(sum_sq, offset);
    sum_sq += SGLANG_SHFL_XOR_SYNC(0xffffffff, sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      // sum_sq += math::shfl_xor_sync(sum_sq, offset);
      sum_sq += SGLANG_SHFL_XOR_SYNC(0xffffffff, sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  // float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);
  // float rms_rcp = rsqrtf(smem[0] / float(d) + eps); // TODO (Hubert): accuracy issue
  float mean_sq = smem[0] / float(d);
  // prefer accuracy: standard sqrt, not the fast rsqrt intrinsic
  float rms_rcp = 1.0f / sqrtf(mean_sq + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<T, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j])); // TODO (Hubert)
      output_vec[j] = from_float<T>(to_float<T>(input_vec[j]) * rms_rcp * (weight_bias + to_float<T>(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");  // TODO (Hubert)
#endif
}

template <typename T>
cudaError_t RMSNorm(
    T* input,
    T* weight,
    T* output,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_output,
    float eps = 1e-5,
    bool enable_pdl = false,
    cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  dim3 nblks(batch_size);
  dim3 nthrs(WARP_SIZE, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};
  SETUP_LAUNCH_CONFIG(nblks, nthrs, stream);  // creates  cfg
  cfg.shared_mem_bytes = smem_size;
  // hipLaunchConfig_t config; // hipLaunchConfig_t
  // config.gridDim = nblks;
  // config.blockDim = nthrs;
  // config.dynamicSmemBytes = smem_size;
  // config.stream = stream;
  // hipLaunchAttribute attrs[1]; // hipLaunchAttribute

  // #ifndef USE_ROCM
  //   attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  //   attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  // #else
  //   attrs[0].id  = hipLaunchAttributePriority;
  //   attrs[0].val.priority = enable_pdl ? 1 : 0;
  // #endif

  // config.numAttrs = 1;
  // config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    SGL_HIP_CALL(hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        // SGL_HIP_CALL(hipFuncSetAttribute(kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
    SGL_HIP_CALL(launch_kernel_non_coop(&cfg, kernel,
                                    input, weight, output, d, stride_input, stride_output, weight_bias, eps));
  });
  return cudaSuccess;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    T* __restrict__ weight,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    float weight_bias,
    float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = WARP_SIZE;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // TODO (Hubert)
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // float x = float(input_vec[j]); // TODO (Hubert)
      float x = to_float<T>(input_vec[j]);
      // x += float(residual_vec[j]);  // TODO (Hubert): no viable conversion from 'float' to 'half' (aka '__half'),
      // castToFloat & castFromFloat
      x += to_float<T>(residual_vec[j]);
      sum_sq += x * x;
      // residual_vec[j] = (T)x;
      residual_vec[j] = from_float<T>(x);
      x_vec[j] = x;  // keep FP32 copy
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    // sum_sq += math::shfl_xor_sync(sum_sq, offset);
    sum_sq += SGLANG_SHFL_XOR_SYNC(0xffffffff, sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      // sum_sq += math::shfl_xor_sync(sum_sq, offset);
      sum_sq += SGLANG_SHFL_XOR_SYNC(0xffffffff, sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  // float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);
  // float rms_rcp = rsqrtf(smem[0] / float(d) + eps); // TODO (Hubert): accuracy issue?
  float mean_sq = smem[0] / float(d);
  // prefer accuracy: standard sqrt, not the fast rsqrt intrinsic
  float rms_rcp = 1.0f / sqrtf(mean_sq + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j])); // TODO
      const float w = to_float<T>(weight_vec[j]);
      const float out = x_vec[j] * rms_rcp * (weight_bias + w);
      input_vec[j] = from_float<T>(out);  // write back in T
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");  // TODO (Hubert)
#endif
}

template <typename T>
cudaError_t FusedAddRMSNorm(
    T* input,
    T* residual,
    T* weight,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_residual,
    float eps = 1e-5,
    bool enable_pdl = false,
    cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  dim3 nblks(batch_size);
  dim3 nthrs(WARP_SIZE, num_warps);
  const uint32_t smem_size =
      (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);  // TODO: Exceed MI300X's per-block LDS limit (64 KB)
  float weight_bias = 0.f;
  void* args[] = {&input, &residual, &weight, &d, &stride_input, &stride_residual, &weight_bias, &eps};
  SETUP_LAUNCH_CONFIG(nblks, nthrs, stream);  // creates  cfg
  cfg.shared_mem_bytes = smem_size;
  // cudaLaunchConfig_t config;
  // config.gridDim = nblks;
  // config.blockDim = nthrs;
  // config.dynamicSmemBytes = smem_size;
  // config.stream = stream;
  // cudaLaunchAttribute attrs[1];

  // #ifndef USE_ROCM
  //   attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  //   attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  // #else
  //   attrs[0].id  = hipLaunchAttributePriority;
  //   attrs[0].val.priority = enable_pdl ? 1 : 0;
  // #endif
  // config.numAttrs = 1;
  // config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    SGL_HIP_CALL(hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    SGL_HIP_CALL(launch_kernel_non_coop(&cfg, kernel,
                                    input, residual, weight, d, stride_input, stride_residual, weight_bias, eps));
  });
  return cudaSuccess;
}

template <typename T>
cudaError_t GemmaRMSNorm(
    T* input,
    T* weight,
    T* output,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_output,
    float eps = 1e-5,
    bool enable_pdl = false,
    cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  dim3 nblks(batch_size);
  dim3 nthrs(WARP_SIZE, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 1.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};
  SETUP_LAUNCH_CONFIG(nblks, nthrs, stream);  // creates  cfg
  cfg.shared_mem_bytes = smem_size;
  // cudaLaunchConfig_t config;
  // config.gridDim = nblks;
  // config.blockDim = nthrs;
  // config.dynamicSmemBytes = smem_size;
  // config.stream = stream;
  // cudaLaunchAttribute attrs[1];
  // attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  // attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  // config.numAttrs = 1;
  // config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    // auto kernel = RMSNormKernel<VEC_SIZE, T>;
    // SGL_HIP_CALL(hipFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    // SGL_HIP_CALL(
    //     cudaLaunchKernelEx(&config, kernel, input, weight, output, d, stride_input, stride_output, weight_bias,
    //     eps));

    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    SGL_HIP_CALL(hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    SGL_HIP_CALL(launch_kernel_non_coop(&cfg, kernel,
                                    input, weight, output, d, stride_input, stride_output, weight_bias, eps));
  });
  return cudaSuccess;
}

template <typename T>
cudaError_t GemmaFusedAddRMSNorm(
    T* input,
    T* residual,
    T* weight,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_residual,
    float eps = 1e-5,
    bool enable_pdl = false,
    cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  dim3 nblks(batch_size);
  dim3 nthrs(WARP_SIZE, num_warps);
  // NOTE(Zihao): use ceil_div(num_warps, 4) * 4 for address alignment to 16 bytes
  const uint32_t smem_size =
      (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);  // TODO: Exceed MI300X's per-block LDS limit (64 KB)
  float weight_bias = 1.f;
  void* args[] = {&input, &residual, &weight, &d, &stride_input, &stride_residual, &weight_bias, &eps};
  SETUP_LAUNCH_CONFIG(nblks, nthrs, stream);  // creates  cfg
  cfg.shared_mem_bytes = smem_size;
// cudaLaunchConfig_t config;
// config.gridDim = nblks;
// config.blockDim = nthrs;
// config.dynamicSmemBytes = smem_size;
// config.stream = stream;
#ifndef USE_ROCM
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
#endif
  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;

    SGL_HIP_CALL(hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    SGL_HIP_CALL(launch_kernel_non_coop(&cfg, kernel,
                                    input, residual, weight, d, stride_input, stride_residual, weight_bias, eps));
  });

  // DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
  //   auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
  //   SGL_HIP_CALL(hipFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  //   SGL_HIP_CALL(cudaLaunchKernelEx(
  //       &config, kernel, input, residual, weight, d, stride_input, stride_residual, weight_bias, eps));
  // });

  return cudaSuccess;
}

}  // namespace norm

}  // namespace sgl_hip

#endif  // SGL_HIP_NORM_CUH_
