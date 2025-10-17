import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import get_device_capability, is_hip

_is_hip = is_hip()
symm_mem_one_shot_available = False
if _is_hip:
    try:
        import torch.distributed._symmetric_memory as torch_symm_mem
        import triton
        import triton.language as tl

        symm_mem_one_shot_available = True
    except ImportError:
        symm_mem_one_shot_available = False


logger = logging.getLogger(__name__)

KERNEL_REG_NAME = "sglang::triton_one_shot_all_reduce"


@triton.jit
def one_shot_all_reduce_kernel_optimized(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.int32,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    """Optimized one-shot all-reduce kernel that handles dynamic numel."""
    pid = tl.program_id(axis=0)
    tl.debug_barrier()
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))
    block_start = pid * BLOCK_SIZE
    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in tl.static_range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            val = tl.load(buffer_ptr + offsets, mask=mask, other=0.0)
            acc += val.to(tl.float32)
        tl.store(output_ptr + offsets, acc.to(tl.bfloat16), mask=mask)
        block_start += num_blocks * BLOCK_SIZE
    tl.debug_barrier()


@triton.jit
def one_shot_all_reduce_kernel_final(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.int32,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    """
    Production-ready kernel with best practices.
    This is the one you should use!
    """
    pid = tl.program_id(axis=0)
    tl.debug_barrier()

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))

    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        # Accumulate in fp32 for numerical stability
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Static range for compile-time unrolling
        for i in tl.static_range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)

            # Load with eviction hint for streaming access
            val = tl.load(
                buffer_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",  # Don't pollute cache
            ).to(tl.float32)

            acc += val

        # Store result
        tl.store(output_ptr + offsets, acc.to(tl.bfloat16), mask=mask)
        block_start += num_blocks * BLOCK_SIZE

    tl.debug_barrier()


if symm_mem_one_shot_available:
    torch.library.define(
        KERNEL_REG_NAME,
        "(Tensor input, Tensor buffer, int buffer_ptrs_dev, int signal_pad_ptrs_dev, int rank, int world_size, Tensor output) -> Tensor",
    )

    @torch.library.impl(KERNEL_REG_NAME, "CUDA")
    def triton_one_shot_all_reduce_cuda(
        input: torch.Tensor,
        buffer: torch.Tensor,
        buffer_ptrs_dev: int,
        signal_pad_ptrs_dev: int,
        rank: int,
        world_size: int,
        output: torch.Tensor,
    ):
        """The implementation now correctly receives the raw integer pointers."""
        buffer[: input.numel()].copy_(input.view(-1))
        # The barrier is handled by the caller before this op is invoked.

        BLOCK_SIZE = 2048
        MAX_NUM_BLOCKS = 32
        num_warps = 8
        num_blocks = min(triton.cdiv(input.numel(), BLOCK_SIZE), MAX_NUM_BLOCKS)

        kernel = one_shot_all_reduce_kernel_optimized[(num_blocks,)]
        kernel(
            buffer_ptrs_dev,
            signal_pad_ptrs_dev,
            output,
            numel=input.numel(),
            rank=rank,
            world_size=world_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_blocks=num_blocks,
            num_warps=num_warps,
        )
        return output


class SymmMemOneShotCommunicator:
    """Thin wrapper around the triton_one_shot_all_reduce custom op."""

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = True
        if not symm_mem_one_shot_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)

        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = get_device_capability(device)[0]
        if self.device_capability < 9:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported.",
                self.device_capability,
            )
            return

        self.max_size = 64 * 1024 * 1024
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize, device=self.device, dtype=self.dtype
        )
        self.buffer_handle = torch_symm_mem.rendezvous(
            self.buffer, self.group.group_name
        )
        self.disabled = False

        try:
            logger.info("Warming up one-shot all-reduce custom op...")
            dummy_size = 2048
            dummy_input = torch.zeros(
                (dummy_size,), dtype=self.dtype, device=self.device
            )
            dummy_output = torch.empty_like(dummy_input)

            # The caller is responsible for the barrier
            self.buffer[: dummy_input.numel()].copy_(dummy_input.view(-1))
            self.buffer_handle.barrier()

            # The call to the op remains the same, as it was already passing integers.
            torch.ops.sglang.triton_one_shot_all_reduce(
                dummy_input,
                self.buffer,
                self.buffer_handle.buffer_ptrs_dev,
                self.buffer_handle.signal_pad_ptrs_dev,
                self.buffer_handle.rank,
                self.buffer_handle.world_size,
                dummy_output,
            )
            torch.cuda.synchronize()
            logger.info("One-shot all-reduce custom op warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to warm up one-shot all-reduce custom op: {e}")
            self.disabled = True

    def should_one_shot_allreduce(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 8 != 0:
            return False
        return inp_size < self.max_size

    def one_shot_all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Performs all-reduce by calling the pre-warmed custom op."""
        if out is None:
            out = torch.empty_like(inp)

        try:
            # The caller handles the copy and barrier before calling the op
            self.buffer[: inp.numel()].copy_(inp.view(-1))
            self.buffer_handle.barrier()

            return torch.ops.sglang.triton_one_shot_all_reduce(
                inp,
                self.buffer,
                self.buffer_handle.buffer_ptrs_dev,
                self.buffer_handle.signal_pad_ptrs_dev,
                self.buffer_handle.rank,
                self.buffer_handle.world_size,
                out,
            )
        except Exception as e:
            logger.error(f"One-shot all-reduce custom op failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None
