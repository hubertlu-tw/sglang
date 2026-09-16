// gfx1151 wave32 specialization for one-token packed GDN decode.
//
// Four V-slice workgroups are launched for every (batch, value-head). Each
// wave streams one BF16 state row at a time, keeping only four FP32 state
// elements per lane instead of a full [BV, K] tile. Q/K normalization is
// computed once per workgroup and shared by its four waves.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr int kGdnH = 16;
constexpr int kGdnHV = 48;
constexpr int kGdnK = 128;
constexpr int kGdnV = 128;
constexpr int kGdnWarps = 1;
constexpr int kGdnThreads = kGdnWarps * 32;

struct alignas(16) GdnBf16x8 {
  bf16_t values[8];
};

struct GdnPackedDecodeGfx1151Params {
  const bf16_t* __restrict__ mixed_qkv;
  const bf16_t* __restrict__ a;
  const bf16_t* __restrict__ b;
  const bf16_t* __restrict__ A_log;
  const bf16_t* __restrict__ dt_bias;
  bf16_t* __restrict__ out;
  bf16_t* __restrict__ state;
  const int32_t* __restrict__ indices;
  int64_t stride_mixed;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_state;
  float scale;
};

__device__ __forceinline__ float gdn_subgroup8_sum(float value) {
  constexpr uint64_t kFullMask = 0xffffffffffffffffull;
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float gdn_wave32_sum(float value) {
  constexpr uint64_t kFullMask = 0xffffffffffffffffull;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

template <int kVBlocks>
__global__ __launch_bounds__(kGdnThreads, 1) void gdn_packed_decode_gfx1151_kernel(
    const GdnPackedDecodeGfx1151Params __grid_constant__ params) {
  using namespace device;
  constexpr int kRowsPerBlock = kGdnV / kVBlocks;

  const int tile = static_cast<int>(blockIdx.x) & (kVBlocks - 1);
  const int nhv = static_cast<int>(blockIdx.x) / kVBlocks;
  const int n = nhv / kGdnHV;
  const int hv = nhv - n * kGdnHV;
  const int h = hv / (kGdnHV / kGdnH);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int k_lane = lane & 7;
  const int v_lane = lane >> 3;

  const int32_t state_idx = params.indices[n];
  bf16_t* out = params.out + (static_cast<int64_t>(n) * kGdnHV + hv) * kGdnV;
  if (state_idx < 0) {
    for (int row = tile * kRowsPerBlock + static_cast<int>(threadIdx.x);
         row < (tile + 1) * kRowsPerBlock;
         row += kGdnThreads) {
      out[row] = cast<bf16_t>(0.0f);
    }
    return;
  }

  const bf16_t* mixed = params.mixed_qkv + static_cast<int64_t>(n) * params.stride_mixed;
  __shared__ float shared_q[kGdnK];
  __shared__ float shared_k[kGdnK];
  float q_lane[4];
  float k_lane_values[4];
  float q_sq = 0.0f;
  float k_sq = 0.0f;
  const int lane_k0 = lane * 4;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    q_lane[e] = cast<fp32_t>(mixed[h * kGdnK + lane_k0 + e]);
    k_lane_values[e] =
        cast<fp32_t>(mixed[kGdnH * kGdnK + h * kGdnK + lane_k0 + e]);
    q_sq = fmaf(q_lane[e], q_lane[e], q_sq);
    k_sq = fmaf(k_lane_values[e], k_lane_values[e], k_sq);
  }
  const float q_norm = rsqrtf(gdn_wave32_sum(q_sq) + 1.0e-6f) * params.scale;
  const float k_norm = rsqrtf(gdn_wave32_sum(k_sq) + 1.0e-6f);
  float dot_kq = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    q_lane[e] *= q_norm;
    k_lane_values[e] *= k_norm;
    shared_q[lane_k0 + e] = q_lane[e];
    shared_k[lane_k0 + e] = k_lane_values[e];
    dot_kq = fmaf(q_lane[e], k_lane_values[e], dot_kq);
  }
  dot_kq = gdn_wave32_sum(dot_kq);
  __syncwarp();

  float q[16];
  float key[16];
#pragma unroll
  for (int ki = 0; ki < 2; ++ki) {
    const int k_start = k_lane * 8 + ki * 64;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      q[ki * 8 + e] = shared_q[k_start + e];
      key[ki * 8 + e] = shared_k[k_start + e];
    }
  }

  float decay = 0.0f;
  float beta = 0.0f;
  if (lane == 0) {
    const float gate_a = cast<fp32_t>(params.a[static_cast<int64_t>(n) * params.stride_a + hv]);
    const float gate_b = cast<fp32_t>(params.b[static_cast<int64_t>(n) * params.stride_b + hv]);
    const float gate_dt = cast<fp32_t>(params.dt_bias[hv]);
    const float exp_A = expf(cast<fp32_t>(params.A_log[hv]));
    const float x = gate_a + gate_dt;
    const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
    decay = expf(-exp_A * softplus);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_b));
    // The Triton baseline rounds sigmoid(beta) through bfloat16.
    beta = cast<fp32_t>(cast<bf16_t>(sigmoid));
  }
  constexpr uint64_t kFullMask = 0xffffffffffffffffull;
  decay = __shfl_sync(kFullMask, decay, 0);
  beta = __shfl_sync(kFullMask, beta, 0);

  const bf16_t* values = mixed + 2 * kGdnH * kGdnK + hv * kGdnV;
  bf16_t* state_head = params.state + static_cast<int64_t>(state_idx) * params.stride_state +
                       static_cast<int64_t>(hv) * kGdnV * kGdnK;
  const int row_end = (tile + 1) * kRowsPerBlock;
  for (int row = tile * kRowsPerBlock + v_lane; row < row_end; row += 4) {
    GdnBf16x8 packed[2];
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
      const int k_start = k_lane * 8 + ki * 64;
      packed[ki] = *reinterpret_cast<const GdnBf16x8*>(state_head + row * kGdnK + k_start);
    }
    float recurrent[16];
    float dot_hk = 0.0f;
    float dot_hq = 0.0f;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ki * 8 + e;
        recurrent[i] = cast<fp32_t>(packed[ki].values[e]) * decay;
        dot_hk = fmaf(recurrent[i], key[i], dot_hk);
        dot_hq = fmaf(recurrent[i], q[i], dot_hq);
      }
    }
    dot_hk = gdn_subgroup8_sum(dot_hk);
    dot_hq = gdn_subgroup8_sum(dot_hq);
    const float residual = (cast<fp32_t>(values[row]) - dot_hk) * beta;
    GdnBf16x8 stored[2];
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ki * 8 + e;
        stored[ki].values[e] = cast<bf16_t>(fmaf(residual, key[i], recurrent[i]));
      }
      const int k_start = k_lane * 8 + ki * 64;
      *reinterpret_cast<GdnBf16x8*>(state_head + row * kGdnK + k_start) = stored[ki];
    }
    if (k_lane == 0) {
      out[row] = cast<bf16_t>(fmaf(residual, dot_kq, dot_hq));
    }
  }
}

struct GdnPackedDecodeGfx1151Kernel {
  static void run(const tvm::ffi::TensorView mixed_qkv,
                  const tvm::ffi::TensorView a,
                  const tvm::ffi::TensorView b,
                  const tvm::ffi::TensorView A_log,
                  const tvm::ffi::TensorView dt_bias,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView state,
                  const tvm::ffi::TensorView indices,
                  double scale) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto Slots = SymbolicSize{"pool_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, 2 * kGdnH * kGdnK + kGdnHV * kGdnV})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(mixed_qkv);
    TensorMatcher({B, kGdnHV}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(a);
    TensorMatcher({B, kGdnHV}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({kGdnHV}).with_dtype<bf16_t>().with_device(device).with_strides({1}).verify(A_log);
    TensorMatcher({kGdnHV}).with_dtype<bf16_t>().with_device(device).with_strides({1}).verify(dt_bias);
    TensorMatcher({B, 1, kGdnHV, kGdnV}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({Slots, kGdnHV, kGdnV, kGdnK})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, kGdnV * kGdnK, kGdnK, 1})
        .verify(state);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(indices);

    const int batch = static_cast<int>(B.unwrap());
    if (batch == 0) return;
    const GdnPackedDecodeGfx1151Params params{
        .mixed_qkv = static_cast<const bf16_t*>(mixed_qkv.data_ptr()),
        .a = static_cast<const bf16_t*>(a.data_ptr()),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .A_log = static_cast<const bf16_t*>(A_log.data_ptr()),
        .dt_bias = static_cast<const bf16_t*>(dt_bias.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .state = static_cast<bf16_t*>(state.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .stride_mixed = mixed_qkv.stride(0),
        .stride_a = a.stride(0),
        .stride_b = b.stride(0),
        .stride_state = state.stride(0),
        .scale = static_cast<float>(scale),
    };
    if (batch == 1) {
      LaunchKernel(batch * kGdnHV * 8, kGdnThreads, device.unwrap())(
          gdn_packed_decode_gfx1151_kernel<8>, params);
    } else {
      LaunchKernel(batch * kGdnHV * 4, kGdnThreads, device.unwrap())(
          gdn_packed_decode_gfx1151_kernel<4>, params);
    }
  }
};

}  // namespace sglang
