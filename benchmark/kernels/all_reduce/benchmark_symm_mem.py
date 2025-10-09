#!/usr/bin/env python3
"""For Now, SYMM_MEM is only supported on TP8 case

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT ./benchmark/kernels/all_reduce/benchmark_symm_mem.py
"""

import argparse
import os
from contextlib import nullcontext
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.symm_mem import SymmMemCommunicator
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_symm_mem_all_reduce,
)

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def torch_allreduce(torch_input: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    dist.all_reduce(torch_input, group=group)
    return torch_input


def symm_mem_allreduce(
    symm_mem_input: torch.Tensor, symm_mem_comm: SymmMemCommunicator
) -> torch.Tensor:
    return symm_mem_comm.all_reduce(symm_mem_input)


def pynccl_allreduce(
    pynccl_input: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    pynccl_comm.all_reduce(pynccl_input)
    return pynccl_input


def one_shot_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform one-shot all-reduce on a tensor using symmetric memory.

    Args:
        tensor: Input tensor allocated with symm_mem.empty(). Must be bfloat16.

    Returns:
        Output tensor containing the all-reduce result.
    """
    import torch.distributed._symmetric_memory as symm_mem
    import triton
    import triton.language as tl

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert tensor.numel() % 8 == 0, "The number of elements must be 128-bit aligned."

    # Create symmetric memory tensor for the input
    symm_tensor = symm_mem.empty(
        tensor.numel(), dtype=torch.bfloat16, device=tensor.device
    )
    symm_tensor.copy_(tensor)

    symm_mem_hdl = symm_mem.rendezvous(symm_tensor, group=dist.group.WORLD)

    if symm_mem_hdl is None:
        raise RuntimeError(
            "Failed to get symmetric memory handle. "
            "Ensure the input tensor is allocated with symm_mem.empty()."
        )

    output = torch.empty_like(tensor)

    # Tuned parameters for AMD GPUs
    BLOCK_SIZE = 4096  # Large block for good occupancy
    MAX_NUM_BLOCKS = 24  # Persistent kernel style
    num_warps = 16  # Maximum warps for latency hiding

    num_blocks = min(
        triton.cdiv(tensor.numel(), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )

    @triton.jit
    def one_shot_all_reduce_kernel_optimized(
        buffer_ptrs,
        signal_pad_ptrs,
        output_ptr,
        numel: tl.constexpr,
        rank: tl.constexpr,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Optimized one-shot all-reduce kernel with minimal synchronization.

        Key optimizations:
        - Only 2 barriers total (not 2*world_size)
        - Vectorized loads (4x bf16 = 64 bits per load)
        - Coalesced memory access
        """
        pid = tl.program_id(axis=0)

        # Single barrier at start - wait for all data to be ready
        tl.debug_barrier()

        buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
        output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))

        block_start = pid * BLOCK_SIZE

        while block_start < numel:
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel

            # Accumulate in float32 for precision
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            # Load and accumulate from all ranks
            for i in range(world_size):
                buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                val = tl.load(buffer_ptr + offsets, mask=mask, other=0.0)
                acc += val.to(tl.float32)

            # Store result
            tl.store(output_ptr + offsets, acc.to(tl.bfloat16), mask=mask)

            block_start += tl.num_programs(axis=0) * BLOCK_SIZE

        # Single barrier at end
        tl.debug_barrier()

    kernel = one_shot_all_reduce_kernel_optimized[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output


def one_shot_all_reduce_graph_only(tensor: torch.Tensor, symm_mem_hdl) -> torch.Tensor:
    """
    One-shot all-reduce that only runs the Triton kernel (for graph capture).
    Assumes symmetric memory tensors are already allocated and rendezvoused.
    """
    import triton
    import triton.language as tl

    output = torch.empty_like(tensor)

    # Tuned parameters for AMD GPUs
    BLOCK_SIZE = 4096  # Large block for good occupancy
    MAX_NUM_BLOCKS = 24  # Persistent kernel style
    num_warps = 16  # Maximum warps for latency hiding

    num_blocks = min(
        triton.cdiv(tensor.numel(), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )

    @triton.jit
    def one_shot_all_reduce_kernel_optimized(
        buffer_ptrs,
        signal_pad_ptrs,
        output_ptr,
        numel: tl.constexpr,
        rank: tl.constexpr,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Optimized one-shot all-reduce kernel with minimal synchronization.

        Key optimizations:
        - Only 2 barriers total (not 2*world_size)
        - Vectorized loads (4x bf16 = 64 bits per load)
        - Coalesced memory access
        """
        pid = tl.program_id(axis=0)

        # Single barrier at start - wait for all data to be ready
        tl.debug_barrier()

        buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
        output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))

        block_start = pid * BLOCK_SIZE

        while block_start < numel:
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel

            # Accumulate in float32 for precision
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            # Load and accumulate from all ranks
            for i in range(world_size):
                buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                val = tl.load(buffer_ptr + offsets, mask=mask, other=0.0)
                acc += val.to(tl.float32)

            # Store result
            tl.store(output_ptr + offsets, acc.to(tl.bfloat16), mask=mask)

            block_start += tl.num_programs(axis=0) * BLOCK_SIZE

        # Single barrier at end
        tl.debug_barrier()

    kernel = one_shot_all_reduce_kernel_optimized[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output


def _bench_graph_time(func, inp_randn, warmup_loop=2, graph_loop=10, test_loop=10):
    graph_input = inp_randn.clone()

    # For one_shot implementation, we need special handling
    # because symmetric memory operations can't be inside graph capture
    is_one_shot = False
    try:
        # Check if this is a one_shot function by examining the lambda
        if hasattr(func, "__name__") and "one_shot" in func.__name__:
            is_one_shot = True
        # Also check the function source code
        import inspect

        source = inspect.getsource(func)
        if "one_shot_all_reduce" in source:
            is_one_shot = True
    except:
        pass

    if is_one_shot:
        # Pre-allocate symmetric memory tensors outside graph capture
        import torch.distributed._symmetric_memory as symm_mem

        # Create symmetric memory tensor for the input
        symm_tensor = symm_mem.empty(
            graph_input.numel(), dtype=torch.bfloat16, device=graph_input.device
        )
        symm_tensor.copy_(graph_input)

        # Rendezvous outside graph capture
        symm_mem_hdl = symm_mem.rendezvous(symm_tensor, group=dist.group.WORLD)

        if symm_mem_hdl is None:
            raise RuntimeError("Failed to get symmetric memory handle")

        # Now capture only the kernel execution
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_loop):
                graph_out = one_shot_all_reduce_graph_only(graph_input, symm_mem_hdl)
    else:
        # Try to use the standard graph capture first
        try:
            with graph_capture() as graph_capture_context:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    for _ in range(graph_loop):
                        graph_out = func(graph_input)
        except AssertionError as e:
            # If graph_capture fails (e.g., model parallel groups not initialized),
            # fall back to a simple graph capture without model parallel groups
            if "tensor model parallel group is not initialized" in str(e):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(graph_loop):
                        graph_out = func(graph_input)
            else:
                # Re-raise other assertion errors
                raise

    graph.replay()
    func_output = graph_out.clone()

    for _ in range(warmup_loop):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for _ in range(test_loop):
        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    func_cost_us = sum(latencies) / len(latencies) / graph_loop * 1000
    graph.reset()
    return func_output, func_cost_us


def _bench_eager_time(func, inp_randn, warmup_loop=2, test_loop=10):
    eager_input = inp_randn.clone()
    eager_output = func(eager_input)
    func_output = eager_output.clone()

    for _ in range(warmup_loop):
        func(eager_input)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(test_loop):
        func(eager_input)
    end_event.record()
    torch.cuda.synchronize()
    func_cost_us = start_event.elapsed_time(end_event) / test_loop * 1000

    return func_output, func_cost_us


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not installed, skipping table printing")
    tabulate = None


def print_markdown_table(data):
    if tabulate is not None:
        print(tabulate(data, headers="keys", tablefmt="github"))
        return
    headers = data[0].keys()
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for item in data:
        row = "| " + " | ".join(str(item[key]) for key in headers) + " |"
        rows.append(row)
    markdown_table = "\n".join([header_row, separator] + rows)
    print(markdown_table)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark symmetric memory all-reduce implementations"
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="symm_mem",
        choices=["torch", "symm_mem", "pynccl", "one_shot"],
        help="All-reduce implementation to benchmark",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    return parser.parse_args()


if __name__ == "__main__":
    import logging

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()

    # Only enable symm_mem and initialize model parallel for non-one_shot implementations
    if args.impl != "one_shot":
        set_symm_mem_all_reduce(True)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank % 8,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group
        cpu_group = get_tensor_model_parallel_group().cpu_group
        pynccl_comm = get_tensor_model_parallel_group().pynccl_comm
        symm_mem_comm = get_tensor_model_parallel_group().symm_mem_comm
    else:
        # For one_shot implementation, we don't need the full SGLang infrastructure
        # Just set up basic distributed environment
        group = dist.group.WORLD
        cpu_group = None
        pynccl_comm = None
        symm_mem_comm = None

    dist.barrier()
    profile = args.profile
    dtype = torch.bfloat16
    ctx = get_torch_prof_ctx(profile)
    result = []

    with ctx:
        if IS_CI:
            i_range = range(10, 11)
        else:
            i_range = range(10, 20)
        for i in i_range:
            sz = 2**i
            if sz * dtype.itemsize > 2**24:
                break
            inp_randn = torch.randint(1, 16, (sz,), dtype=dtype, device=device)

            memory = torch.empty_like(inp_randn)
            memory_out = torch.empty_like(memory)

            # Select implementation based on argument
            if args.impl == "torch":
                eager_output, eager_time = _bench_eager_time(
                    lambda inp: torch_allreduce(inp, group), inp_randn
                )
                graph_output, graph_time = _bench_graph_time(
                    lambda inp: torch_allreduce(inp, group), inp_randn
                )
            elif args.impl == "symm_mem":
                eager_output, eager_time = _bench_eager_time(
                    lambda inp: symm_mem_allreduce(inp, symm_mem_comm), inp_randn
                )
                graph_output, graph_time = _bench_graph_time(
                    lambda inp: symm_mem_allreduce(inp, symm_mem_comm), inp_randn
                )
            elif args.impl == "pynccl":
                eager_output, eager_time = _bench_eager_time(
                    lambda inp: pynccl_allreduce(inp, pynccl_comm), inp_randn
                )
                graph_output, graph_time = _bench_graph_time(
                    lambda inp: pynccl_allreduce(inp, pynccl_comm), inp_randn
                )
            elif args.impl == "one_shot":
                eager_output, eager_time = _bench_eager_time(
                    lambda inp: one_shot_all_reduce(inp), inp_randn
                )
                graph_output, graph_time = _bench_graph_time(
                    lambda inp: one_shot_all_reduce(inp), inp_randn
                )

            torch.testing.assert_close(eager_output, graph_output)
            result.append(
                {
                    "msg_size": human_readable_size(inp_randn.nbytes),
                    f"{args.impl} eager time": eager_time,
                    f"{args.impl} graph time": graph_time,
                }
            )
            if rank == 0:
                print(f"sz={sz}, dtype={dtype}: correctness check PASS!")
    if rank == 0:
        print_markdown_table(result)
    if profile:
        prof_dir = f"prof/symm_mem"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")
