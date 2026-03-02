import os
from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.utils import is_hip


class RcclxCommunicator:
    """AMD-only RCCLX all-reduce wrapper over PyNccl communicator APIs."""

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        use_current_stream: bool = False,
    ):
        self.disabled = True
        self._pynccl = None

        if not is_hip():
            return

        library_path = os.environ.get("SGLANG_RCCLX_SO_PATH") or os.environ.get(
            "SGLANG_NCCL_SO_PATH"
        )
        if not library_path:
            return

        self._pynccl = PyNcclCommunicator(
            group=group,
            device=device,
            library_path=library_path,
            use_current_stream=use_current_stream,
        )
        self.disabled = self._pynccl.disabled

    def should_rcclx_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not inp.is_cuda:
            return False
        return inp.is_contiguous()

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if self.disabled:
            return None
        return self._pynccl.outplace_all_reduce(inp, out)
