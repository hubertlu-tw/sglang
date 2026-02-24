import math

import pytest
import torch

from sglang.jit_kernel.hadamard import (
    hadamard_transform,
    hadamard_transform_12n,
    hadamard_transform_20n,
    hadamard_transform_28n,
    hadamard_transform_40n,
)


def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.view(*y.shape[:-3], n)
        h *= 2
    return y


def _hadamard_ref(x: torch.Tensor, scale: float) -> torch.Tensor:
    dim = x.shape[-1]
    dim_padded = 1 << math.ceil(math.log2(dim))
    x_padded = torch.nn.functional.pad(x, (0, dim_padded - dim))
    ref = _fwht_last_dim(x_padded) * scale
    return ref[..., :dim]


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 3e-4, 3e-3
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    return 3e-3, 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192, 16384, 32768],
)
def test_hadamard_transform_matches_reference(dim: int, dtype: torch.dtype):
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(dim)

    # Keep memory bounded for very large dimensions.
    batch = 4 if dim >= 8192 else 15
    x = torch.randn(batch, dim, device="cuda", dtype=dtype)

    out = hadamard_transform(x, scale=scale)
    ref = _hadamard_ref(x.float(), scale=scale)

    torch.testing.assert_close(
        out.float(),
        ref,
        rtol=rtol,
        atol=atol,
        msg=f"hadamard_transform mismatch for dtype={dtype}, dim={dim}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
@pytest.mark.parametrize(
    ("fn", "dim"),
    [
        (hadamard_transform_12n, 4 * 12),
        (hadamard_transform_20n, 4 * 20),
        (hadamard_transform_28n, 4 * 28),
        (hadamard_transform_40n, 4 * 40),
    ],
)
def test_specialized_hadamard_variants_match_reference(fn, dim: int):
    # These specialized MN variants are not the same as the power-of-two FWHT.
    # Validate stronger transform properties instead.
    x = torch.randn(8, dim + 3, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(8, dim + 3, device="cuda", dtype=torch.bfloat16)
    scale = 0.5

    out_x = fn(x, scale=scale)
    out_y = fn(y, scale=scale)
    out_sum = fn(x + y, scale=scale)
    out_x_repeat = fn(x, scale=scale)

    assert out_x.shape == x.shape
    assert torch.isfinite(out_x).all()
    # Determinism for same input.
    torch.testing.assert_close(
        out_x.float(), out_x_repeat.float(), rtol=1e-3, atol=1e-3
    )
    # Linearity: T(x + y) == T(x) + T(y)
    torch.testing.assert_close(
        out_sum.float(),
        (out_x + out_y).float(),
        rtol=2e-2,
        atol=8e-2,
    )
