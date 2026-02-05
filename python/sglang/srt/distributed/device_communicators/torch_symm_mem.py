import logging
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.utils import get_device_capability, is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_DEFAULT_HIP_MAX_SIZE = 64 * 1024 * 1024
_ONE_SHOT_ENV = "SGLANG_TORCH_SYMM_MEM_ONE_SHOT"

try:
    if _is_cuda or _is_hip:
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem_available = True
    else:
        torch_symm_mem_available = False
except ImportError:
    torch_symm_mem_available = False

_one_shot_available = False
if torch_symm_mem_available and _is_hip:
    try:
        import triton
        import triton.language as tl

        _one_shot_available = True
    except Exception:
        _one_shot_available = False

KERNEL_REG_NAME = "sglang::triton_one_shot_all_reduce"


def _register_one_shot_op() -> bool:
    if not _one_shot_available:
        return False
    try:
        torch.library.define(
            KERNEL_REG_NAME,
            "(Tensor input, Tensor buffer, int buffer_ptrs_dev, int signal_pad_ptrs_dev, int rank, int world_size, Tensor output) -> Tensor",
        )
    except Exception:
        return False

    @triton.jit
    def one_shot_all_reduce_kernel(
        buffer_ptrs,
        signal_pad_ptrs,
        output_ptr,
        numel: tl.int32,
        rank: tl.constexpr,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        num_blocks: tl.constexpr,
    ):
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
                val = tl.load(
                    buffer_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                acc += val
            tl.store(output_ptr + offsets, acc.to(tl.bfloat16), mask=mask)
            block_start += num_blocks * BLOCK_SIZE
        tl.debug_barrier()

    @torch.library.impl(KERNEL_REG_NAME, "CUDA")
    def triton_one_shot_all_reduce(
        input: torch.Tensor,
        buffer: torch.Tensor,
        buffer_ptrs_dev: int,
        signal_pad_ptrs_dev: int,
        rank: int,
        world_size: int,
        output: torch.Tensor,
    ):
        buffer[: input.numel()].copy_(input.view(-1))
        BLOCK_SIZE = 2048
        MAX_NUM_BLOCKS = 32
        num_warps = 8
        num_blocks = min(triton.cdiv(input.numel(), BLOCK_SIZE), MAX_NUM_BLOCKS)
        kernel = one_shot_all_reduce_kernel[(num_blocks,)]
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

    return True


_one_shot_registered = _register_one_shot_op() if _one_shot_available else False


class TorchSymmMemCommunicator:
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """

        self.disabled = True
        self._multicast_supported = False
        self._handle = None
        self._one_shot_enabled = False

        if not torch_symm_mem_available:
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

        capability = get_device_capability(device.index or 0)[0]
        self.device_capability = 0 if capability is None else capability

        if _is_cuda:
            if self.device_capability < 9:
                logger.warning(
                    "TorchSymmMemCommunicator: Device capability %s not supported, "
                    "communicator is not available.",
                    self.device_capability,
                )
                return
            if (
                self.world_size
                not in TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]
            ):
                logger.warning(
                    "TorchSymmMemCommunicator: World size %d not supported, "
                    "communicator is not available.",
                    self.world_size,
                )
                return
            self.max_size = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
                self.world_size
            ]
        else:
            # AMD GPUs: keep a conservative default until tuned.
            self.max_size = _DEFAULT_HIP_MAX_SIZE
            logger.warning(
                "TorchSymmMemCommunicator: Using default max size %d on AMD GPUs",
                self.max_size,
            )

        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
            handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        except Exception as e:
            logger.warning(
                "TorchSymmMemCommunicator: failed to initialize symmetric buffer: %s",
                e,
            )
            self.buffer = None
            return

        self._handle = handle
        self._multicast_supported = handle.multicast_ptr != 0
        if not self._multicast_supported:
            logger.warning(
                "TorchSymmMemCommunicator: torch symmetric memory multicast "
                "operations are not supported."
            )
            if _is_cuda:
                self.buffer = None
                return

        if (
            _is_hip
            and _one_shot_available
            and _one_shot_registered
            and os.environ.get(_ONE_SHOT_ENV, "0") == "1"
        ):
            try:
                self._warmup_one_shot()
                self._one_shot_enabled = True
                logger.info("TorchSymmMemCommunicator: one-shot path enabled")
            except Exception as e:
                logger.warning(
                    "TorchSymmMemCommunicator: one-shot warmup failed: %s", e
                )
                self._one_shot_enabled = False

        self.disabled = False

    def _warmup_one_shot(self) -> None:
        if not _one_shot_registered or self._handle is None:
            return
        dummy_size = 2048
        dummy_input = torch.zeros((dummy_size,), dtype=self.dtype, device=self.device)
        dummy_output = torch.empty_like(dummy_input)
        self.buffer[: dummy_input.numel()].copy_(dummy_input.view(-1))
        self._handle.barrier()
        torch.ops.sglang.triton_one_shot_all_reduce(
            dummy_input,
            self.buffer,
            self._handle.buffer_ptrs_dev,
            self._handle.signal_pad_ptrs_dev,
            self._handle.rank,
            self._handle.world_size,
            dummy_output,
        )
        torch.cuda.synchronize()

    def _should_one_shot_allreduce(self, inp: torch.Tensor) -> bool:
        if not self._one_shot_enabled or self._handle is None:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 8 != 0:
            return False
        return inp_size < self.max_size

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor; if omitted, a new tensor is allocated.

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Writes the result into 'out' and returns it.
        """
        if self.disabled:
            return None
        if out is None:
            out = torch.empty_like(inp)
        if self._should_one_shot_allreduce(inp):
            self.buffer[: inp.numel()].copy_(inp.view(-1))
            self._handle.barrier()
            torch.ops.sglang.triton_one_shot_all_reduce(
                inp,
                self.buffer,
                self._handle.buffer_ptrs_dev,
                self._handle.signal_pad_ptrs_dev,
                self._handle.rank,
                self._handle.world_size,
                out,
            )
            return out

        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if (
            _is_cuda
            and self._multicast_supported
            and self.world_size
            in self._WORLD_SIZES_MULTIMEM.get(self.device_capability, [])
        ):
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
