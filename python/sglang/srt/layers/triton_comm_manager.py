"""
Manager for Triton-based allreduce fusion with symmetric memory.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_triton_symm_mem_buffer = None
_triton_symm_mem_enabled = False

try:
    import torch.distributed._symmetric_memory as symm_mem

    _symm_mem_available = True
except ImportError:
    _symm_mem_available = False
    logger.warning("torch.distributed._symmetric_memory not available")


def initialize_triton_symm_mem_buffer(
    max_tokens: int = 8192,
    hidden_dim: int = 8192,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
) -> Optional[torch.Tensor]:
    """
    Initialize the symmetric memory buffer for Triton allreduce fusion.

    Args:
        max_tokens: Maximum number of tokens to support
        hidden_dim: Hidden dimension of the model
        dtype: Data type for the buffer
        device: Device to allocate the buffer on

    Returns:
        The allocated symmetric memory buffer or None if not available
    """
    global _triton_symm_mem_buffer, _triton_symm_mem_enabled

    if not _symm_mem_available:
        logger.warning("Symmetric memory not available, Triton fusion disabled")
        return None

    if device is None:
        device = torch.cuda.current_device()

    try:
        # Allocate buffer for max_tokens * hidden_dim elements
        buffer_size = max_tokens * hidden_dim
        _triton_symm_mem_buffer = symm_mem.empty(
            buffer_size, device=device, dtype=dtype
        )
        _triton_symm_mem_enabled = True
        logger.info(
            f"Initialized Triton symm_mem buffer: size={buffer_size}, dtype={dtype}"
        )
        return _triton_symm_mem_buffer
    except Exception as e:
        logger.error(f"Failed to initialize Triton symm_mem buffer: {e}")
        _triton_symm_mem_enabled = False
        return None


def get_triton_symm_mem_buffer() -> Optional[torch.Tensor]:
    """Get the global symmetric memory buffer for Triton fusion."""
    return _triton_symm_mem_buffer


def is_triton_symm_mem_enabled() -> bool:
    """Check if Triton symmetric memory is enabled and available."""
    return _triton_symm_mem_enabled


def attach_symm_mem_buffer(
    tensor: torch.Tensor, buffer: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Attach symmetric memory buffer to a tensor as an attribute.

    Args:
        tensor: The tensor to attach the buffer to
        buffer: The buffer to attach (uses global buffer if None)

    Returns:
        The tensor with buffer attached
    """
    if buffer is None:
        buffer = _triton_symm_mem_buffer

    if buffer is not None:
        # Ensure buffer is large enough
        if buffer.numel() >= tensor.numel():
            # Preserve any existing attributes
            existing_attrs = {}
            if hasattr(tensor, "_sglang_needs_allreduce_fusion"):
                existing_attrs["_sglang_needs_allreduce_fusion"] = (
                    tensor._sglang_needs_allreduce_fusion
                )

            tensor._symm_mem_buffer = buffer[: tensor.numel()].view_as(tensor)

            # Restore existing attributes
            for attr_name, attr_value in existing_attrs.items():
                setattr(tensor, attr_name, attr_value)
        else:
            logger.warning(
                f"Symm mem buffer too small: {buffer.numel()} < {tensor.numel()}"
            )

    return tensor


def cleanup_triton_symm_mem():
    """Clean up the symmetric memory buffer."""
    global _triton_symm_mem_buffer, _triton_symm_mem_enabled
    _triton_symm_mem_buffer = None
    _triton_symm_mem_enabled = False
