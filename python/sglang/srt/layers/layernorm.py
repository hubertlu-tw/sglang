# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging.version import Version

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
    supports_custom_op,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda:
    if _is_flashinfer_available:
        from flashinfer.norm import fused_add_rmsnorm
    else:
        from sgl_kernel import fused_add_rmsnorm
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    import vllm
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

    _vllm_version = Version(vllm.__version__)

logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            if _vllm_version < Version("0.9"):
                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
                return x, residual
            else:
                residual_out = torch.empty_like(x)
                output = torch.empty_like(x)
                fused_add_rms_norm(
                    output,
                    x,
                    residual_out,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return output, residual_out
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fused RMSNorm + residual + allreduce.

        Tries the Triton symmetric-memory fusion first (on ROCm when
        `enable_torch_symm_mem` and `enable_triton_allreduce_fusion` are set),
        then falls back to the FlashInfer fusion when
        `enable_flashinfer_allreduce_fusion` is enabled, and finally falls
        back to the unfused implementation.
        """
        if residual is not None:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            from sglang.srt.managers.schedule_batch import global_server_args_dict
            from sglang.srt.utils import (
                is_flashinfer_available,
                is_hip,
                supports_custom_op,
            )

            # Check if distributed environment is initialized
            try:
                world_size = get_tensor_model_parallel_world_size()
            except AssertionError:
                # Distributed environment not initialized, single GPU
                world_size = 1

            if world_size > 1:
                # Try Triton fused path on HIP devices when symmetric memory is enabled.
                if global_server_args_dict.get(
                    "enable_torch_symm_mem", False
                ) and global_server_args_dict.get(
                    "enable_triton_allreduce_fusion", False
                ):
                    # Debug logging
                    import logging

                    from sglang.srt.layers.triton_comm_fusion import (
                        triton_allreduce_residual_rmsnorm_wrapper,
                    )

                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"Attempting Triton fused allreduce for shape {x.shape}"
                    )

                    # Debug logging
                    # logger.debug(
                    #     f"Triton fusion - input shape: {x.shape}, contiguous: {x.is_contiguous()}, dtype: {x.dtype}"
                    # )
                    # logger.debug(
                    #     f"Triton fusion - residual shape: {residual.shape}, contiguous: {residual.is_contiguous()}, dtype: {residual.dtype}"
                    # )
                    # logger.debug(
                    #     f"Triton fusion - weight shape: {self.weight.shape}, contiguous: {self.weight.is_contiguous()}, dtype: {self.weight.dtype}"
                    # )

                    # Detailed tensor properties for accuracy debugging
                    # logger.debug(
                    #     f"Input strides: {x.stride()}, memory format: {x.layout}"
                    # )
                    # logger.debug(
                    #     f"Residual strides: {residual.stride()}, memory format: {residual.layout}"
                    # )
                    # logger.debug(
                    #     f"Weight strides: {self.weight.stride()}, memory format: {self.weight.layout}"
                    # )

                    # Sample data values for debugging accuracy issues
                    # logger.debug(
                    #     f"Input sample data (first 5 elements): {x.flatten()[:5].tolist()}"
                    # )
                    # logger.debug(
                    #     f"Residual sample data (first 5 elements): {residual.flatten()[:5].tolist()}"
                    # )
                    # logger.debug(
                    #     f"Weight sample data (first 5 elements): {self.weight.flatten()[:5].tolist()}"
                    # )

                    # Ensure tensors are contiguous like in forward_native
                    if not x.is_contiguous():
                        x = x.contiguous()
                        logger.debug("Made input tensor contiguous for Triton fusion")
                    if not residual.is_contiguous():
                        residual = residual.contiguous()
                        logger.debug(
                            "Made residual tensor contiguous for Triton fusion"
                        )

                    fused_result = triton_allreduce_residual_rmsnorm_wrapper(
                        input_tensor=x,
                        residual=residual,
                        weight=self.weight,
                        eps=self.variance_epsilon,
                        max_token_num=x.shape[0],
                    )
                    if fused_result is not None:
                        logger.debug(
                            f"Triton fused allreduce succeeded for shape {x.shape}"
                        )
                        # The wrapper returns (norm_output, residual_out)
                        # Unpack the tuple properly
                        norm_output, residual_out = fused_result
                        return norm_output, residual_out
                    else:
                        logger.debug(
                            f"Triton fused allreduce failed, falling back for shape {x.shape}"
                        )

                # Try FlashInfer fused path when available and enabled.
                if (
                    global_server_args_dict.get(
                        "enable_flashinfer_allreduce_fusion", False
                    )
                    and is_flashinfer_available()
                ):
                    from sglang.srt.layers.flashinfer_comm_fusion import (
                        flashinfer_allreduce_residual_rmsnorm,
                    )

                    fused_op = (
                        torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm
                        if supports_custom_op()
                        else flashinfer_allreduce_residual_rmsnorm
                    )
                    fused_result = fused_op(
                        input_tensor=x,
                        residual=residual,
                        weight=self.weight,
                        eps=self.variance_epsilon,
                    )
                    if fused_result[0] is not None:
                        return fused_result

        return self.forward(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x if residual is None else (x, residual)


class Gemma3RMSNorm(CustomOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


if not (
    _is_cuda or _is_hip or _is_npu or (_is_cpu and _is_cpu_amx_available) or _is_xpu
):
    logger.info(
        "sgl-kernel layernorm implementation is not available on current platform. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
