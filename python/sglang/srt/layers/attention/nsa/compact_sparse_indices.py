"""
CUDA-graph-safe Triton kernel for compacting sparse indices by removing -1 values.
This is used by the aiter NSA decode backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_sparse_indices_kernel(
    input_ptr,  # [bs, topk]
    counts_ptr,  # [bs]
    output_ptr,  # [bs * topk] (pre-allocated)
    bs: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compact sparse indices by removing -1 values.
    
    Args:
        input_ptr: Input indices tensor [bs, topk] with -1 for invalid entries
        counts_ptr: Number of valid (non -1) entries per row [bs]
        output_ptr: Output buffer for compacted indices [total_valid]
        bs: Batch size
        topk: Number of indices per batch element
        BLOCK_SIZE: Block size for parallel processing
    """
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    if batch_idx >= bs:
        return
    
    # Calculate input offset for this batch element
    input_offset = batch_idx * topk
    
    # Calculate output offset (cumsum of counts up to batch_idx)
    output_offset = 0
    for i in range(batch_idx):
        output_offset += tl.load(counts_ptr + i)
    
    # Load count for this batch element
    count = tl.load(counts_ptr + batch_idx)
    
    # Compact the indices for this batch element
    # Use conditional write instead of break
    write_idx = 0
    for i in range(topk):
        idx = tl.load(input_ptr + input_offset + i)
        # Only write if: 1) idx is valid, and 2) we haven't written enough yet
        should_write = (idx != -1) & (write_idx < count)
        if should_write:
            tl.store(output_ptr + output_offset + write_idx, idx)
            write_idx += 1


def compact_sparse_indices_triton(
    input_indices: torch.Tensor,  # [bs, topk]
    counts: torch.Tensor,  # [bs]
    output_buffer: torch.Tensor,  # [bs * topk] pre-allocated
    bs: int,
    topk: int,
):
    """
    Compact sparse indices by removing -1 values using Triton.
    
    This is CUDA-graph-safe as it uses pre-allocated fixed-size buffers.
    
    Args:
        input_indices: Input indices [bs, topk] with -1 for padding
        counts: Number of valid indices per batch [bs]
        output_buffer: Pre-allocated output buffer [bs * topk]
        bs: Batch size
        topk: Top-k value (indices per batch)
    """
    assert input_indices.is_contiguous()
    assert counts.is_contiguous()
    assert output_buffer.is_contiguous()
    assert input_indices.shape == (bs, topk)
    assert counts.shape == (bs,)
    
    # Launch kernel with one program per batch element
    grid = (bs,)
    _compact_sparse_indices_kernel[grid](
        input_indices,
        counts,
        output_buffer,
        bs=bs,
        topk=topk,
        BLOCK_SIZE=256,
    )
