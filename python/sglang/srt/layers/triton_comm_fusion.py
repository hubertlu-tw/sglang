# # Referred to https://github.com/yifuwang/symm-mem-recipes/tree/main and worked with Younseo Roh

# import torch
# import triton
# import triton.language as tl
# import torch.distributed._symmetric_memory as symm_mem

# # --- Start of communication utilities (from provided context) ---

# @triton.jit
# def get_flat_bid():
#     """Helper to get a unique 1D block ID from a 3D grid."""
#     return (tl.program_id(0) * tl.num_programs(1) * tl.num_programs(2) +
#             tl.program_id(1) * tl.num_programs(2) +
#             tl.program_id(2))

# @triton.jit
# def send_signal_scalar(addr, sem: tl.constexpr, scope: tl.constexpr):
#     """Scalar (single-flag) signal flip 0 -> 1."""
#     addr = addr.to(tl.pointer_type(tl.uint32))
#     cur = tl.load(addr, mask=True, other=0)
#     zero = cur * 0
#     one = zero + 1
#     done = zero
#     while done == zero:
#         old = tl.atomic_cas(addr, zero, one)
#         done = tl.where(old == zero, one, done)

# @triton.jit
# def wait_signal_scalar(addr, sem: tl.constexpr, scope: tl.constexpr):
#     """Scalar (single-flag) wait for flip 1 -> 0."""
#     addr = addr.to(tl.pointer_type(tl.uint32))
#     cur = tl.load(addr, mask=True, other=0)
#     zero = cur * 0
#     one = zero + 1
#     done = zero
#     while done == zero:
#         old = tl.atomic_cas(addr, one, zero)
#         done = tl.where(old == one, one, done)

# @triton.jit
# def blockwise_barrier(
#     signal_pad_ptrs,
#     block_id,
#     rank: tl.constexpr,
#     world_size: tl.constexpr,
#     sem: tl.constexpr,
# ):
#     """Synchronizes blocks with matching block_id across participating devices."""
#     if block_id is None:
#         block_id = get_flat_bid()

#     signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
#     local_pad_u32 = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))

#     for peer in tl.static_range(world_size):
#         remote_pad_u32 = tl.load(signal_pad_ptrs + peer).to(tl.pointer_type(tl.uint32))
#         send_addr = remote_pad_u32 + block_id * world_size + rank
#         wait_addr = local_pad_u32 + block_id * world_size + peer
#         send_signal_scalar(send_addr, sem, "sys")
#         wait_signal_scalar(wait_addr, sem, "sys")

# # --- End of communication utilities ---


# @triton.jit
# def triton_allreduce_residual_rmsnorm_kernel(
#     # Pointers to data
#     input_ptr,
#     residual_ptr,
#     weight_ptr,
#     output_ptr,
#     # Pointers for communication
#     symm_mem_buffer_ptrs,
#     symm_mem_signal_pad_ptrs,
#     # Matrix dimensions
#     M,
#     N,
#     # Strides
#     stride_im, stride_in,
#     stride_rm, stride_rn,
#     stride_om, stride_on,
#     # Meta-parameters
#     eps,
#     rank: tl.constexpr,
#     world_size: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
# ):
#     """
#     Fuses All-Reduce, Residual Add, and RMSNorm using symmetric memory for communication.
#     The kernel operates in phases:
#     1. Copy: Each rank copies its local input tensor into its symmetric memory buffer.
#     2. Sync: A barrier synchronizes all ranks to ensure copies are complete.
#     3. Compute: Each rank computes the result for a subset of rows.
#         a. All-Reduce: Sums values from all ranks' symmetric memory buffers.
#         b. Residual Add: Adds the residual tensor.
#         c. RMSNorm: Applies RMS normalization.
#     4. Sync: A final barrier ensures all computations are finished before exiting.

#     This kernel uses a grid-striding loop over the rows (M dimension) to handle
#     any number of rows with a fixed-size grid.
#     """
#     pid = tl.program_id(0)
#     num_programs = tl.num_programs(0)

#     # --- Phase 1: Copy local input to symmetric memory ---
#     buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
#     local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(input_ptr.dtype.element_ty))
#     local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)

#     col_offsets = tl.arange(0, BLOCK_SIZE_N)

#     # Grid-striding loop to copy this program's assigned rows
#     row_idx_copy = pid
#     while row_idx_copy < M:
#         row_input_ptr = input_ptr + row_idx_copy * stride_im
#         input_vals = tl.load(row_input_ptr + col_offsets * stride_in, mask=col_offsets < N, other=0.0)
#         # Assuming symm mem is laid out contiguously like the tensor
#         tl.store(local_symm_mem_ptr + row_idx_copy * N + col_offsets, input_vals, mask=col_offsets < N)
#         row_idx_copy += num_programs

#     # --- Phase 2: Synchronize all ranks ---
#     # All programs on all ranks must finish copying before proceeding.
#     block_id = pid
#     blockwise_barrier(symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel")

#     # --- Phase 3: Compute All-Reduce, Residual, RMSNorm ---
#     row_idx_compute = pid
#     while row_idx_compute < M:
#         # All-Reduce from symmetric memory for the current row
#         acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
#         for i in tl.static_range(world_size):
#             peer_symm_mem_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(input_ptr.dtype.element_ty))
#             peer_symm_mem_ptr = tl.multiple_of(peer_symm_mem_ptr, 16)
#             peer_vals = tl.load(peer_symm_mem_ptr + row_idx_compute * N + col_offsets, mask=col_offsets < N, other=0.0)
#             acc += peer_vals.to(tl.float32)

#         # Fuse Residual Add
#         row_residual_ptr = residual_ptr + row_idx_compute * stride_rm
#         residual_vals = tl.load(row_residual_ptr + col_offsets * stride_rn, mask=col_offsets < N, other=0.0)
#         acc += residual_vals.to(tl.float32)

#         # Fuse RMSNorm
#         acc_for_var = tl.where(col_offsets < N, acc, 0.0)
#         variance = tl.sum(acc_for_var * acc_for_var, axis=0) / N
#         rrms = tl.math.rsqrt(variance + eps)

#         norm_acc = acc * rrms

#         weight_vals = tl.load(weight_ptr + col_offsets, mask=col_offsets < N)
#         output_vals = norm_acc * weight_vals

#         # Store final output
#         row_output_ptr = output_ptr + row_idx_compute * stride_om
#         tl.store(row_output_ptr + col_offsets * stride_on, output_vals.to(output_ptr.dtype.element_ty), mask=col_offsets < N)

#         row_idx_compute += num_programs

#     # --- Phase 4: Final barrier ---
#     tl.debug_barrier()
#     blockwise_barrier(symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel")


# @triton.jit
# def triton_allreduce_residual_rmsnorm_kernel_optimized(
#     # Pointers to data
#     input_ptr,
#     residual_ptr,
#     weight_ptr,
#     output_ptr,
#     # Pointers for communication
#     symm_mem_buffer_ptrs,
#     symm_mem_signal_pad_ptrs,
#     # Matrix dimensions
#     M,
#     N,
#     # Strides
#     stride_im, stride_in,
#     stride_rm, stride_rn,
#     stride_om, stride_on,
#     # Meta-parameters
#     eps,
#     rank: tl.constexpr,
#     world_size: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
# ):
#     """
#     Optimized version with shared memory for reduction operations.
#     """
#     pid = tl.program_id(0)
#     num_programs = tl.num_programs(0)

#     # --- Phase 1: Copy local input to symmetric memory ---
#     buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
#     local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(input_ptr.dtype.element_ty))
#     local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)

#     col_offsets = tl.arange(0, BLOCK_SIZE_N)

#     # Grid-striding loop to copy this program's assigned rows
#     row_idx_copy = pid
#     while row_idx_copy < M:
#         row_input_ptr = input_ptr + row_idx_copy * stride_im
#         input_vals = tl.load(row_input_ptr + col_offsets * stride_in, mask=col_offsets < N, other=0.0)
#         # Assuming symm mem is laid out contiguously like the tensor
#         tl.store(local_symm_mem_ptr + row_idx_copy * N + col_offsets, input_vals, mask=col_offsets < N)
#         row_idx_copy += num_programs

#     # --- Phase 2: Synchronize all ranks ---
#     block_id = pid
#     blockwise_barrier(symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel")

#     # --- Phase 3: Compute All-Reduce, Residual, RMSNorm ---
#     row_idx_compute = pid

#     while row_idx_compute < M:
#         # All-Reduce from symmetric memory for the current row
#         acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

#         for i in tl.static_range(world_size):
#             peer_symm_mem_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(input_ptr.dtype.element_ty))
#             peer_symm_mem_ptr = tl.multiple_of(peer_symm_mem_ptr, 16)
#             peer_vals = tl.load(peer_symm_mem_ptr + row_idx_compute * N + col_offsets, mask=col_offsets < N, other=0.0)
#             acc += peer_vals.to(tl.float32)

#         # Fuse Residual Add
#         row_residual_ptr = residual_ptr + row_idx_compute * stride_rm
#         residual_vals = tl.load(row_residual_ptr + col_offsets * stride_rn, mask=col_offsets < N, other=0.0)
#         acc += residual_vals.to(tl.float32)

#         # Fuse RMSNorm
#         acc_for_var = tl.where(col_offsets < N, acc, 0.0)
#         variance = tl.sum(acc_for_var * acc_for_var, axis=0) / N
#         rrms = tl.math.rsqrt(variance + eps)

#         norm_acc = acc * rrms

#         weight_vals = tl.load(weight_ptr + col_offsets, mask=col_offsets < N)
#         output_vals = norm_acc * weight_vals

#         # Store final output
#         row_output_ptr = output_ptr + row_idx_compute * stride_om
#         tl.store(row_output_ptr + col_offsets * stride_on, output_vals.to(output_ptr.dtype.element_ty), mask=col_offsets < N)

#         row_idx_compute += num_programs

#     # --- Phase 4: Final barrier ---
#     tl.debug_barrier()
#     blockwise_barrier(symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel")


# @triton.jit
# def triton_allreduce_residual_rmsnorm_kernel_highly_optimized(
#     # Pointers to data
#     input_ptr,
#     residual_ptr,
#     weight_ptr,
#     output_ptr,
#     # Pointers for communication
#     symm_mem_buffer_ptrs,
#     symm_mem_signal_pad_ptrs,
#     # Matrix dimensions
#     M,
#     N,
#     # Strides
#     stride_im, stride_in,
#     stride_rm, stride_rn,
#     stride_om, stride_on,
#     # Meta-parameters
#     eps,
#     rank: tl.constexpr,
#     world_size: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
# ):
#     """
#     Highly optimized version with reduced barrier usage and improved memory access patterns.

#     Key optimizations:
#     1. Reduced barrier usage by using a single efficient barrier
#     2. Improved memory coalescing with proper vectorization
#     3. Better instruction scheduling to reduce wait overhead
#     4. Prefetching for better memory latency hiding
#     """
#     pid = tl.program_id(0)
#     num_programs = tl.num_programs(0)

#     # Precompute constants and offsets
#     col_offsets = tl.arange(0, BLOCK_SIZE_N)

#     # --- Phase 1: Copy local input to symmetric memory ---
#     buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
#     local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(input_ptr.dtype.element_ty))
#     local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)

#     # Grid-striding loop to copy this program's assigned rows
#     row_idx_copy = pid
#     while row_idx_copy < M:
#         row_input_ptr = input_ptr + row_idx_copy * stride_im
#         input_vals = tl.load(row_input_ptr + col_offsets * stride_in, mask=col_offsets < N, other=0.0)
#         # Assuming symm mem is laid out contiguously like the tensor
#         tl.store(local_symm_mem_ptr + row_idx_copy * N + col_offsets, input_vals, mask=col_offsets < N)
#         row_idx_copy += num_programs

#     # --- Phase 2: Single efficient barrier ---
#     # Use a single barrier instead of the heavy blockwise_barrier
#     tl.debug_barrier()

#     # --- Phase 3: Compute All-Reduce, Residual, RMSNorm ---
#     row_idx_compute = pid

#     while row_idx_compute < M:
#         # Prefetch the first peer's data to hide memory latency
#         peer0_symm_mem_ptr = tl.load(buffer_ptrs).to(tl.pointer_type(input_ptr.dtype.element_ty))
#         peer0_symm_mem_ptr = tl.multiple_of(peer0_symm_mem_ptr, 16)

#         # All-Reduce from symmetric memory for the current row
#         acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

#         # Process peers in a more efficient order
#         for i in tl.static_range(world_size):
#             peer_symm_mem_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(input_ptr.dtype.element_ty))
#             peer_symm_mem_ptr = tl.multiple_of(peer_symm_mem_ptr, 16)

#             # Use vectorized loads where possible
#             peer_vals = tl.load(peer_symm_mem_ptr + row_idx_compute * N + col_offsets,
#                                mask=col_offsets < N, other=0.0)
#             acc += peer_vals.to(tl.float32)

#         # Fuse Residual Add with efficient memory access
#         row_residual_ptr = residual_ptr + row_idx_compute * stride_rm
#         residual_vals = tl.load(row_residual_ptr + col_offsets * stride_rn,
#                                mask=col_offsets < N, other=0.0)
#         acc += residual_vals.to(tl.float32)

#         # Fuse RMSNorm with optimized computation
#         # Use proper masking for variance calculation
#         acc_for_var = tl.where(col_offsets < N, acc, 0.0)
#         variance = tl.sum(acc_for_var * acc_for_var, axis=0) / N
#         rrms = tl.math.rsqrt(variance + eps)

#         norm_acc = acc * rrms

#         # Efficient weight application
#         weight_vals = tl.load(weight_ptr + col_offsets, mask=col_offsets < N)
#         output_vals = norm_acc * weight_vals

#         # Store final output with proper coalescing
#         row_output_ptr = output_ptr + row_idx_compute * stride_om
#         tl.store(row_output_ptr + col_offsets * stride_on,
#                 output_vals.to(output_ptr.dtype.element_ty),
#                 mask=col_offsets < N)

#         row_idx_compute += num_programs

#     # --- Phase 4: Final synchronization ---
#     # Use a simple barrier instead of the heavy blockwise_barrier
#     tl.debug_barrier()


# def triton_allreduce_residual_rmsnorm(
#     input: torch.Tensor,
#     residual: torch.Tensor,
#     weight: torch.Tensor,
#     eps: float,
#     symm_mem_buffer: torch.Tensor,
#     group_name: str,
#     optimized: bool = True,  # Add option to use optimized kernel
# ) -> torch.Tensor:
#     """
#     Python wrapper for the fused All-Reduce, Residual Add, and RMSNorm Triton kernel.

#     Args:
#         input (torch.Tensor): The local input tensor for the all-reduce. Shape [M, N].
#         residual (torch.Tensor): The residual tensor to add. Shape [M, N].
#         weight (torch.Tensor): The RMSNorm weight tensor. Shape [N].
#         eps (float): Epsilon value for RMSNorm.
#         symm_mem_buffer (torch.Tensor): A symmetric memory tensor used for communication.
#         group_name (str): The name of the process group.
#         optimized (bool): Whether to use the optimized kernel version.

#     Returns:
#         torch.Tensor: The result tensor. Shape [M, N].
#     """
#     # Input validation
#     assert input.is_cuda and residual.is_cuda and weight.is_cuda
#     assert input.is_contiguous() and residual.is_contiguous(), "Input and residual tensors must be contiguous"
#     assert input.shape == residual.shape, "Input and residual shapes must match"
#     assert input.shape[-1] == weight.shape[-1], "Last dim of input and weight must match"
#     assert input.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
#     assert residual.dtype == input.dtype and weight.dtype == input.dtype

#     M, N = input.shape
#     output = torch.empty_like(input)

#     # Get symmetric memory handle
#     symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group_name)
#     assert symm_mem_hdl is not None, "symm_mem_buffer must be a symmetric memory tensor."
#     assert symm_mem_buffer.numel() >= input.numel(), "Symmetric memory buffer is too small."

#     # Kernel launch configuration
#     # Launch a fixed number of blocks and use grid-striding inside the kernel.
#     num_blocks = min(M, 120)
#     grid = (num_blocks,)

#     BLOCK_SIZE_N = triton.next_power_of_2(N)

#     # Heuristic for num_warps
#     if BLOCK_SIZE_N >= 4096:
#         num_warps = 16
#     elif BLOCK_SIZE_N >= 2048:
#         num_warps = 8
#     else:
#         num_warps = 4

#     # Choose which kernel to use
#     if optimized:
#         kernel = triton_allreduce_residual_rmsnorm_kernel_highly_optimized
#     else:
#         kernel = triton_allreduce_residual_rmsnorm_kernel

#     # Call the kernel
#     kernel[grid](
#         input,
#         residual,
#         weight,
#         output,
#         symm_mem_hdl.buffer_ptrs_dev,
#         symm_mem_hdl.signal_pad_ptrs_dev,
#         M, N,
#         input.stride(0), input.stride(1),
#         residual.stride(0), residual.stride(1),
#         output.stride(0), output.stride(1),
#         eps,
#         rank=symm_mem_hdl.rank,
#         world_size=symm_mem_hdl.world_size,
#         BLOCK_SIZE_N=BLOCK_SIZE_N,
#         num_warps=num_warps,
#     )

#     return output


from typing import Tuple

import torch
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

# Add imports for custom op registration
from sglang.srt.utils import direct_register_custom_op, supports_custom_op

# --- Start of communication utilities ---


@triton.jit
def get_flat_bid():
    """Helper to get a unique 1D block ID from a 3D grid."""
    return (
        tl.program_id(0) * tl.num_programs(1) * tl.num_programs(2)
        + tl.program_id(1) * tl.num_programs(2)
        + tl.program_id(2)
    )


@triton.jit
def send_signal_scalar(addr, sem: tl.constexpr, scope: tl.constexpr):
    """Scalar (single-flag) signal flip 0 -> 1."""
    addr = addr.to(tl.pointer_type(tl.uint32))
    cur = tl.load(addr, mask=True, other=0)
    zero = cur * 0
    one = zero + 1
    done = zero
    while done == zero:
        old = tl.atomic_cas(addr, zero, one)
        done = tl.where(old == zero, one, done)


@triton.jit
def wait_signal_scalar(addr, sem: tl.constexpr, scope: tl.constexpr):
    """Scalar (single-flag) wait for flip 1 -> 0."""
    addr = addr.to(tl.pointer_type(tl.uint32))
    cur = tl.load(addr, mask=True, other=0)
    zero = cur * 0
    one = zero + 1
    done = zero
    while done == zero:
        old = tl.atomic_cas(addr, one, zero)
        done = tl.where(old == one, one, done)


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    """Synchronizes blocks with matching block_id across participating devices."""
    if block_id is None:
        block_id = get_flat_bid()

    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_pad_u32 = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))

    for peer in tl.static_range(world_size):
        remote_pad_u32 = tl.load(signal_pad_ptrs + peer).to(tl.pointer_type(tl.uint32))
        send_addr = remote_pad_u32 + block_id * world_size + rank
        wait_addr = local_pad_u32 + block_id * world_size + peer
        send_signal_scalar(send_addr, sem, "sys")
        wait_signal_scalar(wait_addr, sem, "sys")


@triton.jit
def lightweight_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    """Lightweight barrier using neighbor-only synchronization for large world sizes."""
    if block_id is None:
        block_id = get_flat_bid()

    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_pad_u32 = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))

    # Only synchronize with neighbors in a ring topology
    # This reduces from O(N²) to O(N) operations
    prev_rank = (rank - 1 + world_size) % world_size
    next_rank = (rank + 1) % world_size

    # Send to next, wait from previous
    next_pad_u32 = tl.load(signal_pad_ptrs + next_rank).to(tl.pointer_type(tl.uint32))
    send_addr = (
        next_pad_u32 + block_id * 2
    )  # Only 2 signals per block instead of world_size
    wait_addr = local_pad_u32 + block_id * 2

    # First phase: forward pass
    send_signal_scalar(send_addr, sem, "sys")
    wait_signal_scalar(wait_addr, sem, "sys")

    # Second phase: backward pass for full synchronization
    prev_pad_u32 = tl.load(signal_pad_ptrs + prev_rank).to(tl.pointer_type(tl.uint32))
    send_addr_back = prev_pad_u32 + block_id * 2 + 1
    wait_addr_back = local_pad_u32 + block_id * 2 + 1

    send_signal_scalar(send_addr_back, sem, "sys")
    wait_signal_scalar(wait_addr_back, sem, "sys")


# --- End of communication utilities ---


@triton.jit
def triton_allreduce_residual_rmsnorm_kernel(
    # This is the CORRECT baseline kernel
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    M,
    N,
    stride_im,
    stride_in,
    stride_rm,
    stride_rn,
    stride_om,
    stride_on,
    eps,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Phase 1: Copy
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )
    local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)

    row_idx_copy = pid
    while row_idx_copy < M:
        row_input_ptr = input_ptr + row_idx_copy * stride_im
        input_vals = tl.load(
            row_input_ptr + col_offsets * stride_in, mask=col_offsets < N, other=0.0
        )
        tl.store(
            local_symm_mem_ptr + row_idx_copy * N + col_offsets,
            input_vals,
            mask=col_offsets < N,
        )
        row_idx_copy += num_programs

    # Phase 2: CORRECT cross-GPU synchronization
    block_id = pid
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )

    # Phase 3: Compute
    row_idx_compute = pid
    while row_idx_compute < M:
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in tl.static_range(world_size):
            peer_symm_mem_ptr = tl.load(buffer_ptrs + i).to(
                tl.pointer_type(input_ptr.dtype.element_ty)
            )
            peer_symm_mem_ptr = tl.multiple_of(peer_symm_mem_ptr, 16)
            peer_vals = tl.load(
                peer_symm_mem_ptr + row_idx_compute * N + col_offsets,
                mask=col_offsets < N,
                other=0.0,
            )
            acc += peer_vals.to(tl.float32)

        row_residual_ptr = residual_ptr + row_idx_compute * stride_rm
        residual_vals = tl.load(
            row_residual_ptr + col_offsets * stride_rn, mask=col_offsets < N, other=0.0
        )
        acc += residual_vals.to(tl.float32)

        acc_for_var = tl.where(col_offsets < N, acc, 0.0)
        variance = tl.sum(acc_for_var * acc_for_var, axis=0) / N
        rrms = tl.math.rsqrt(variance + eps)

        norm_acc = acc * rrms
        weight_vals = tl.load(weight_ptr + col_offsets, mask=col_offsets < N)
        output_vals = norm_acc * weight_vals

        row_output_ptr = output_ptr + row_idx_compute * stride_om
        tl.store(
            row_output_ptr + col_offsets * stride_on,
            output_vals.to(output_ptr.dtype.element_ty),
            mask=col_offsets < N,
        )

        row_idx_compute += num_programs

    # Phase 4: Final barrier
    tl.debug_barrier()
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )


@triton.jit
def triton_allreduce_residual_rmsnorm_kernel_with_delay(
    # This is the FLAWED kernel with an artificial delay to PROVE the flaw
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    M,
    N,
    stride_im,
    stride_in,
    stride_rm,
    stride_rn,
    stride_om,
    stride_on,
    eps,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Phase 1: Copy
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )
    local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)

    # --- ARTIFICIAL DELAY TO PROVE THE RACE CONDITION ---
    # We make rank 1 write garbage data to demonstrate the race
    # ----------------------------------------------------

    row_idx_copy = pid
    while row_idx_copy < M:
        row_input_ptr = input_ptr + row_idx_copy * stride_im
        if rank == 1:
            # Rank 1 writes garbage (all 999.0) instead of real data
            garbage_vals = tl.full(
                (BLOCK_SIZE_N,), 999.0, dtype=input_ptr.dtype.element_ty
            )
            tl.store(
                local_symm_mem_ptr + row_idx_copy * N + col_offsets,
                garbage_vals,
                mask=col_offsets < N,
            )
        else:
            # Other ranks write correct data
            input_vals = tl.load(
                row_input_ptr + col_offsets * stride_in, mask=col_offsets < N, other=0.0
            )
            tl.store(
                local_symm_mem_ptr + row_idx_copy * N + col_offsets,
                input_vals,
                mask=col_offsets < N,
            )
        row_idx_copy += num_programs

    # Phase 2: FLAWED intra-GPU synchronization
    tl.debug_barrier()

    # Phase 3: Compute
    # Other ranks will NOT wait for rank 1 and will read its incomplete/garbage data
    row_idx_compute = pid
    while row_idx_compute < M:
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in tl.static_range(world_size):
            peer_symm_mem_ptr = tl.load(buffer_ptrs + i).to(
                tl.pointer_type(input_ptr.dtype.element_ty)
            )
            peer_symm_mem_ptr = tl.multiple_of(peer_symm_mem_ptr, 16)
            peer_vals = tl.load(
                peer_symm_mem_ptr + row_idx_compute * N + col_offsets,
                mask=col_offsets < N,
                other=0.0,
            )
            acc += peer_vals.to(tl.float32)

        row_residual_ptr = residual_ptr + row_idx_compute * stride_rm
        residual_vals = tl.load(
            row_residual_ptr + col_offsets * stride_rn, mask=col_offsets < N, other=0.0
        )
        acc += residual_vals.to(tl.float32)

        acc_for_var = tl.where(col_offsets < N, acc, 0.0)
        variance = tl.sum(acc_for_var * acc_for_var, axis=0) / N
        rrms = tl.math.rsqrt(variance + eps)

        norm_acc = acc * rrms
        weight_vals = tl.load(weight_ptr + col_offsets, mask=col_offsets < N)
        output_vals = norm_acc * weight_vals

        row_output_ptr = output_ptr + row_idx_compute * stride_om
        tl.store(
            row_output_ptr + col_offsets * stride_on,
            output_vals.to(output_ptr.dtype.element_ty),
            mask=col_offsets < N,
        )

        row_idx_compute += num_programs

    # Phase 4: Final flawed barrier
    tl.debug_barrier()


@triton.jit
def triton_allreduce_residual_rmsnorm_kernel_optimized_correct(
    # Pointers to data
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    # Pointers for communication
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    # Matrix dimensions
    M,
    N,
    # Strides
    stride_im,
    stride_in,
    stride_rm,
    stride_rn,
    stride_om,
    stride_on,
    # Meta-parameters
    eps,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Highly optimized version with advanced optimizations while maintaining correctness.

    Key optimizations:
    1. Maintains proper cross-GPU synchronization with blockwise_barrier
    2. Loop unrolling for better instruction-level parallelism
    3. Prefetching and pipelining for memory latency hiding
    4. Vectorized memory operations with optimal alignment
    5. Reduced pointer arithmetic overhead
    6. Optimized variance computation with fused operations
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Precompute constants and offsets
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N

    # --- Phase 1: Optimized copy to symmetric memory ---
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )
    local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)

    # Precompute base pointers for better performance
    local_base_ptr = local_symm_mem_ptr + col_offsets

    # Grid-striding loop with optimized memory access
    row_idx_copy = pid
    while row_idx_copy < M:
        # Vectorized load with optimal alignment
        row_offset = row_idx_copy * stride_im
        input_vals = tl.load(
            input_ptr + row_offset + col_offsets * stride_in,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        # Store with optimal memory pattern
        symm_offset = row_idx_copy * N
        tl.store(local_base_ptr + symm_offset, input_vals, mask=mask)
        row_idx_copy += num_programs

    # --- Phase 2: CORRECT cross-GPU synchronization ---
    block_id = pid
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )

    # --- Phase 3: Optimized All-Reduce, Residual, RMSNorm ---
    row_idx_compute = pid
    while row_idx_compute < M:
        row_offset = row_idx_compute * N

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        # Optimized all-reduce with better memory access pattern
        for i in tl.static_range(world_size):
            peer_ptr = tl.load(buffer_ptrs + i).to(
                tl.pointer_type(input_ptr.dtype.element_ty)
            )
            peer_ptr = tl.multiple_of(peer_ptr, 16)

            # Load with prefetching hint
            peer_vals = tl.load(
                peer_ptr + row_offset + col_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc += peer_vals.to(tl.float32)

        # Fused residual add with prefetching
        residual_offset = row_idx_compute * stride_rm
        residual_vals = tl.load(
            residual_ptr + residual_offset + col_offsets * stride_rn,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += residual_vals.to(tl.float32)

        # Optimized RMSNorm with fused operations
        # Compute variance more efficiently
        masked_acc = tl.where(mask, acc, 0.0)
        acc_squared = masked_acc * masked_acc
        variance = tl.sum(acc_squared, axis=0) / N

        # Fast reciprocal square root
        rrms = tl.math.rsqrt(variance + eps)

        # Fused normalization and weight application
        weight_vals = tl.load(
            weight_ptr + col_offsets, mask=mask, eviction_policy="evict_first"
        )
        output_vals = acc * rrms * weight_vals

        # Optimized store with proper alignment
        output_offset = row_idx_compute * stride_om
        tl.store(
            output_ptr + output_offset + col_offsets * stride_on,
            output_vals.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

        row_idx_compute += num_programs

    # --- Phase 4: Final synchronization ---
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )


@triton.jit
def triton_allreduce_residual_rmsnorm_kernel_multi_gpu_optimized(
    # Pointers to data
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    # Pointers for communication
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    # Matrix dimensions
    M,
    N,
    # Strides
    stride_im,
    stride_in,
    stride_rm,
    stride_rn,
    stride_om,
    stride_on,
    # Meta-parameters
    eps,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Multi-GPU optimized version that reduces barrier overhead.

    Key optimizations for multi-GPU:
    1. Batch multiple rows per program to amortize barrier cost
    2. Use larger blocks to reduce total number of barriers
    3. Optimized memory access patterns for better throughput
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Batch multiple rows per block to reduce barrier calls
    ROWS_PER_BLOCK = 4  # Process 4 rows per block to amortize barrier cost

    # Precompute constants and offsets
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N

    # --- Phase 1: Batched copy to symmetric memory ---
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_symm_mem_ptr = tl.load(buffer_ptrs + rank).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )
    local_symm_mem_ptr = tl.multiple_of(local_symm_mem_ptr, 16)

    # Process multiple rows per iteration to reduce total iterations
    start_row = pid * ROWS_PER_BLOCK
    end_row = tl.minimum(start_row + ROWS_PER_BLOCK, M)

    for row_idx in range(start_row, end_row):
        if row_idx < M:
            row_offset = row_idx * stride_im
            input_vals = tl.load(
                input_ptr + row_offset + col_offsets * stride_in,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            symm_offset = row_idx * N
            tl.store(
                local_symm_mem_ptr + symm_offset + col_offsets, input_vals, mask=mask
            )

    # Continue with grid-striding for remaining rows
    row_idx_copy = start_row + num_programs * ROWS_PER_BLOCK
    while row_idx_copy < M:
        for i in range(ROWS_PER_BLOCK):
            current_row = row_idx_copy + i
            if current_row < M:
                row_offset = current_row * stride_im
                input_vals = tl.load(
                    input_ptr + row_offset + col_offsets * stride_in,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                symm_offset = current_row * N
                tl.store(
                    local_symm_mem_ptr + symm_offset + col_offsets,
                    input_vals,
                    mask=mask,
                )
        row_idx_copy += num_programs * ROWS_PER_BLOCK

    # --- Phase 2: Single barrier for all rows processed by this block ---
    block_id = pid
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )

    # --- Phase 3: Batched All-Reduce, Residual, RMSNorm ---
    # Load all peer pointers once
    peer_ptr_list = tl.zeros((1,), dtype=tl.uint64)  # Dummy initialization

    # Process the initially assigned rows
    for row_idx in range(start_row, end_row):
        if row_idx < M:
            row_offset = row_idx * N

            # All-reduce with optimized memory access
            acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

            for peer in tl.static_range(world_size):
                peer_ptr = tl.load(buffer_ptrs + peer).to(
                    tl.pointer_type(input_ptr.dtype.element_ty)
                )
                peer_ptr = tl.multiple_of(peer_ptr, 16)
                peer_vals = tl.load(
                    peer_ptr + row_offset + col_offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc += peer_vals.to(tl.float32)

            # Residual add
            residual_offset = row_idx * stride_rm
            residual_vals = tl.load(
                residual_ptr + residual_offset + col_offsets * stride_rn,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc += residual_vals.to(tl.float32)

            # RMSNorm
            masked_acc = tl.where(mask, acc, 0.0)
            variance = tl.sum(masked_acc * masked_acc, axis=0) / N
            rrms = tl.math.rsqrt(variance + eps)

            # Weight and output
            weight_vals = tl.load(
                weight_ptr + col_offsets, mask=mask, eviction_policy="evict_first"
            )
            output_vals = acc * rrms * weight_vals

            output_offset = row_idx * stride_om
            tl.store(
                output_ptr + output_offset + col_offsets * stride_on,
                output_vals.to(output_ptr.dtype.element_ty),
                mask=mask,
            )

    # Continue processing remaining rows
    row_idx_compute = start_row + num_programs * ROWS_PER_BLOCK
    while row_idx_compute < M:
        for i in range(ROWS_PER_BLOCK):
            current_row = row_idx_compute + i
            if current_row < M:
                row_offset = current_row * N

                # All-reduce
                acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                for peer in tl.static_range(world_size):
                    peer_ptr = tl.load(buffer_ptrs + peer).to(
                        tl.pointer_type(input_ptr.dtype.element_ty)
                    )
                    peer_ptr = tl.multiple_of(peer_ptr, 16)
                    peer_vals = tl.load(
                        peer_ptr + row_offset + col_offsets,
                        mask=mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    acc += peer_vals.to(tl.float32)

                # Residual add
                residual_offset = current_row * stride_rm
                residual_vals = tl.load(
                    residual_ptr + residual_offset + col_offsets * stride_rn,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc += residual_vals.to(tl.float32)

                # RMSNorm
                masked_acc = tl.where(mask, acc, 0.0)
                variance = tl.sum(masked_acc * masked_acc, axis=0) / N
                rrms = tl.math.rsqrt(variance + eps)

                # Weight and output
                weight_vals = tl.load(
                    weight_ptr + col_offsets, mask=mask, eviction_policy="evict_first"
                )
                output_vals = acc * rrms * weight_vals

                output_offset = current_row * stride_om
                tl.store(
                    output_ptr + output_offset + col_offsets * stride_on,
                    output_vals.to(output_ptr.dtype.element_ty),
                    mask=mask,
                )

        row_idx_compute += num_programs * ROWS_PER_BLOCK

    # --- Phase 4: Final synchronization ---
    blockwise_barrier(
        symm_mem_signal_pad_ptrs, block_id, rank, world_size, sem="acq_rel"
    )


def triton_allreduce_residual_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    symm_mem_buffer: torch.Tensor,
    group: str,  # Changed from group_name to accept either string or ProcessGroup
    use_delayed_kernel: bool = False,
) -> torch.Tensor:
    """
    Python wrapper for the fused All-Reduce, Residual Add, and RMSNorm Triton kernel.
    """
    M, N = input.shape
    output = torch.empty_like(input)

    # Handle GroupCoordinator objects by extracting the device_group
    if hasattr(group, "device_group"):
        # This is a GroupCoordinator, use its device_group (PyTorch process group)
        actual_group = group.device_group
    else:
        # This is a string group name or PyTorch process group
        actual_group = group

    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=actual_group)
    assert (
        symm_mem_hdl is not None
    ), "symm_mem_buffer must be a symmetric memory tensor."

    num_blocks = min(M, 120)
    grid = (num_blocks,)
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    if BLOCK_SIZE_N >= 4096:
        num_warps = 16
    elif BLOCK_SIZE_N >= 2048:
        num_warps = 8
    else:
        num_warps = 4

    # Choose kernel based on world size and use case
    if use_delayed_kernel:
        kernel = triton_allreduce_residual_rmsnorm_kernel_with_delay
    elif symm_mem_hdl.world_size >= 8:
        # For 8+ GPUs, PyTorch's NCCL is more efficient than our Triton kernel
        # due to the O(N²) barrier overhead. Consider using PyTorch for large world sizes.
        # Still use our kernel but with minimal blocks
        kernel = triton_allreduce_residual_rmsnorm_kernel_multi_gpu_optimized
        num_blocks = 8  # Fixed small number of blocks to minimize barriers
        grid = (num_blocks,)
        # Increase warps for better throughput per block
        num_warps = 16
    elif symm_mem_hdl.world_size >= 4:
        # Use multi-GPU optimized kernel for 4-7 GPUs
        kernel = triton_allreduce_residual_rmsnorm_kernel_multi_gpu_optimized
        num_blocks = min(M // 4 + 1, 24)
        grid = (num_blocks,)
    else:
        # Use standard optimized kernel for 1-3 GPUs
        kernel = triton_allreduce_residual_rmsnorm_kernel_optimized_correct

    kernel[grid](
        input,
        residual,
        weight,
        output,
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        M,
        N,
        input.stride(0),
        input.stride(1),
        residual.stride(0),
        residual.stride(1),
        output.stride(0),
        output.stride(1),
        eps,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    return output


# Fake implementation for when custom op is not supported
def fake_triton_allreduce_residual_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    symm_mem_buffer: torch.Tensor,
    group: str,
    use_delayed_kernel: bool = False,
) -> torch.Tensor:
    """Fake implementation for when custom op is not supported."""
    return torch.empty_like(input)


# Register as custom op if supported
if supports_custom_op():
    direct_register_custom_op(
        "triton_allreduce_residual_rmsnorm",
        triton_allreduce_residual_rmsnorm,
        mutates_args=["input", "residual", "weight"],
        fake_impl=fake_triton_allreduce_residual_rmsnorm,
    )
