"""gfx1151 wave32 row-streaming specialization for packed GDN decode."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _is_gfx1151(device: torch.device) -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1151"


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gdn_packed_decode_gfx1151",
        cuda_files=["attention/gdn_packed_decode_gfx1151.cuh"],
        cuda_wrappers=[("run", "GdnPackedDecodeGfx1151Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_q_heads: int,
) -> bool:
    """Return whether inputs match the measured production specialization."""
    return (
        _is_gfx1151(mixed_qkv.device)
        and 0 < mixed_qkv.shape[0] <= 4
        and num_q_heads == 16
        and initial_state.ndim == 4
        and initial_state.shape[-3:] == (48, 128, 128)
        and mixed_qkv.shape == (mixed_qkv.shape[0], 2 * 16 * 128 + 48 * 128)
        and a.shape == b.shape == (mixed_qkv.shape[0], 48)
        and A_log.shape == dt_bias.shape == (48,)
        and out.shape == (mixed_qkv.shape[0], 1, 48, 128)
        and ssm_state_indices.shape == (mixed_qkv.shape[0],)
        and all(
            tensor.dtype == torch.bfloat16
            for tensor in (
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                initial_state,
                out,
            )
        )
        and ssm_state_indices.dtype == torch.int32
        and mixed_qkv.stride(-1) == 1
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and initial_state.stride()[-3:] == (128 * 128, 128, 1)
        and out.is_contiguous()
        and ssm_state_indices.is_contiguous()
    )


def gdn_packed_decode_gfx1151(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> None:
    """Update selected state slots in place and write ``out``."""
    _jit_module().run(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        ssm_state_indices,
        float(scale),
    )
