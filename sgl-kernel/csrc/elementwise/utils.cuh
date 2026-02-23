// Adapted from https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh

#pragma once

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
using nv_bfloat16 = __hip_bfloat16;
#endif

#include <cstdint>

__forceinline__ __device__ int get_lane_id() {
#ifdef USE_ROCM
  return static_cast<int>(threadIdx.x & 31);
#else
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
#endif
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ void st_na_global_v1(const int* ptr, int v) {
#ifdef USE_ROCM
  *const_cast<int*>(ptr) = v;
#else
  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
#endif
}

__device__ __forceinline__ void st_na_global_v2(const int2* ptr, const int2& v) {
#ifdef USE_ROCM
  *const_cast<int2*>(ptr) = v;
#else
  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");
#endif
}

__device__ __forceinline__ void st_na_global_v4(const int4* ptr, const int4& v) {
#ifdef USE_ROCM
  *const_cast<int4*>(ptr) = v;
#else
  asm volatile(
      "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
      : "memory");
#endif
}

__device__ __forceinline__ int ld_na_global_v1(const int* ptr) {
#ifdef USE_ROCM
  return *ptr;
#else
  int r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#endif
  return r;
#endif
}

__device__ __forceinline__ int2 ld_na_global_v2(const int2* ptr) {
#ifdef USE_ROCM
  return *ptr;
#else
  int2 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#endif
  return r;
#endif
}

__device__ __forceinline__ int4 ld_na_global_v4(const int4* ptr) {
#ifdef USE_ROCM
  return *ptr;
#else
  int4 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#endif
  return r;
#endif
}

__device__ __forceinline__ void prefetch_L2(const void* p) {
#ifdef USE_ROCM
  (void)p;
#else
#if defined(ENABLE_L2_PREFETCH)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
#endif
}
