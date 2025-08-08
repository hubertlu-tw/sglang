/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#if USE_ROCM

#include <hip/hip_bf16.h>
#include <hip/hip_common.h>
#include <hip/hip_fp16.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)d

#define SGL_HIP_INLINE inline __attribute__((always_inline)) __device__

namespace sgl_hip {

template <typename float_t, size_t vec_size>
struct vec_t;

template <typename srcDtype, typename dstDtype, size_t vec_size>
SGL_HIP_INLINE void cast_load_impl(vec_t<dstDtype, vec_size>& dst, const srcDtype* src);

template <typename srcDtype, typename dstDtype, size_t vec_size>
SGL_HIP_INLINE void cast_store_impl(dstDtype* dst_ptr, const vec_t<srcDtype, vec_size>& src);

template <typename float_t, size_t vec_size>
struct vec_t {
  SGL_HIP_INLINE float_t& operator[](size_t i);
  SGL_HIP_INLINE const float_t& operator[](size_t i) const;
  SGL_HIP_INLINE void fill(float_t val);
  SGL_HIP_INLINE float_t* ptr();

  SGL_HIP_INLINE void load(const float_t* ptr);
  SGL_HIP_INLINE void store(float_t* ptr) const;

  template <typename T>
  SGL_HIP_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  SGL_HIP_INLINE void cast_load(const T* ptr);
  template <typename T>
  SGL_HIP_INLINE void cast_store(T* ptr) const;
};

}  // namespace sgl_hip

// **** impl *****

namespace sgl_hip {

template <typename DST, typename SRC>
struct vec_cast {
  template <size_t N>
  __device__ __forceinline__ static void cast(DST* dst, const SRC* src) {
#pragma unroll
    for (size_t i = 0; i < N; ++i)
      dst[i] = static_cast<DST>(src[i]);
  }
};

// float  <-->  half
template <>
struct vec_cast<float, __half> {
  template <size_t N>
  __device__ __forceinline__ static void cast(float* dst, const __half* src) {
    if constexpr (N == 1)
      dst[0] = __half2float(src[0]);
    else {
#pragma unroll
      for (size_t i = 0; i < N / 2; ++i)
        reinterpret_cast<float2*>(dst)[i] = __half22float2(reinterpret_cast<const __half2*>(src)[i]);
    }
  }
};
template <>
struct vec_cast<__half, float> {
  template <size_t N>
  __device__ __forceinline__ static void cast(__half* dst, const float* src) {
    if constexpr (N == 1)
      dst[0] = __float2half(src[0]);
    else {
#pragma unroll
      for (size_t i = 0; i < N / 2; ++i)
        reinterpret_cast<__half2*>(dst)[i] = __float22half2_rn(reinterpret_cast<const float2*>(src)[i]);
    }
  }
};

// float  <-->  bfloat16
template <>
struct vec_cast<float, __hip_bfloat16> {
  template <size_t N>
  __device__ __forceinline__ static void cast(float* dst, const __hip_bfloat16* src) {
    if constexpr (N == 1)
      dst[0] = __bfloat162float(src[0]);
    else {
#pragma unroll
      for (size_t i = 0; i < N / 2; ++i)
        reinterpret_cast<float2*>(dst)[i] = __bfloat1622float2(reinterpret_cast<const __hip_bfloat162*>(src)[i]);
    }
  }
};
template <>
struct vec_cast<__hip_bfloat16, float> {
  template <size_t N>
  __device__ __forceinline__ static void cast(__hip_bfloat16* dst, const float* src) {
    if constexpr (N == 1)
      dst[0] = __float2bfloat16(src[0]);
    else {
#pragma unroll
      for (size_t i = 0; i < N / 2; ++i)
        reinterpret_cast<__hip_bfloat162*>(dst)[i] = __float22bfloat162_rn(reinterpret_cast<const float2*>(src)[i]);
    }
  }
};

template <typename dst_t, typename src_t, size_t vec_size>
SGL_HIP_INLINE void cast_from_impl(vec_t<dst_t, vec_size>& dst, const vec_t<src_t, vec_size>& src) {
  if constexpr (std::is_same_v<dst_t, src_t>) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
      dst[i] = src[i];
  } else {
    // Delegate to vec_cast<dst,src>
    vec_cast<dst_t, src_t>::template cast<vec_size>(dst.ptr(), const_cast<src_t*>(&src[0]));
  }
}

template <typename srcDtype, typename dstDtype, size_t vec_size>
SGL_HIP_INLINE void cast_load_impl(vec_t<dstDtype, vec_size>& dst, const srcDtype* src_ptr) {
  if constexpr (std::is_same<srcDtype, dstDtype>::value) {
    dst.load(src_ptr);
  } else {
    vec_t<srcDtype, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename srcDtype, typename dstDtype, size_t vec_size>
SGL_HIP_INLINE void cast_store_impl(dstDtype* dst_ptr, const vec_t<srcDtype, vec_size>& src) {
  if constexpr (std::is_same<srcDtype, dstDtype>::value) {
    src.store(dst_ptr);
  } else {
    vec_t<dstDtype, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

template <typename float_t, size_t vec_size>
template <typename T>
SGL_HIP_INLINE void vec_t<float_t, vec_size>::cast_load(const T* ptr) {
  cast_load_impl(*this, ptr);
}

template <typename float_t, size_t vec_size>
template <typename T>
SGL_HIP_INLINE void vec_t<float_t, vec_size>::cast_store(T* ptr) const {
  cast_store_impl(ptr, *this);
}

}  // namespace sgl_hip

#include "impl/hip_vec_bf16_impl.h"
#include "impl/hip_vec_fp32_impl.h"
#include "impl/hip_vec_half_impl.h"
#endif
