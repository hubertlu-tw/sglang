// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_subblock_common.cuh"

torch::Tensor
a8w8_subblockwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_intrawave_v1(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);

  using DeviceGemmInstance = DeviceGemmHelper<
    64,
    16,
    16,
    128,
    16,
    16,
    1,
    1,
    S<8, 8, 1>,
    S<8, 8, 1>,
    S<1, 16, 1, 4>,
    S<4, 4, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    ck::tensor_operation::device::GemmSpecialization::MKPadding>;
  return gemm_a8w8_subblockwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}
