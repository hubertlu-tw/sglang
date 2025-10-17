#!/usr/bin/env python3
"""Comprehensive All-Reduce Benchmark with Correctness Checking

Benchmarks various all-reduce implementations across different data sizes (1KB to 100MB)
and modes (eager vs graph) to determine the fastest implementation for each scenario.

Supported implementations:
- torch.distributed.all_reduce (Native)
- Symmetric Memory All-Reduce (symm_mem)
- Custom All-Reduce (custom_ar)
- Quick All-Reduce (quick_ar) - AMD GPUs only
- PyNccl (pynccl)
- PyMscclpp (pymscclpp) - NVIDIA GPUs only
- One-shot All-Reduce (symm_mem_one_shot) - AMD GPUs only

Usage:
  torchrun --nproc_per_node 2 --nnodes 1 \
    ./comprehensive_allreduce_benchmark.py \
    --min-size 10 --max-size 27 --step 1 --profile --dtype bfloat16 --check-correctness
"""

import argparse
import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_symm_mem_all_reduce,
)
from sglang.srt.utils import is_hip

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


class AllReduceBenchmark:
    """Comprehensive all-reduce benchmark class"""

    def __init__(self, world_size: int, rank: int, device: torch.device):
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.communicators: Dict[str, Any] = {}
        self.initialized = False

    def initialize_communicators(self):
        """Initialize all available communicators"""
        if self.initialized:
            return

        # Set device
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Enable all implementations
        set_symm_mem_all_reduce(True)
        set_custom_all_reduce(True)
        if not is_hip():
            set_mscclpp_all_reduce(True)

        # Initialize distributed environment
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.rank % torch.cuda.device_count(),
        )
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)

        group = get_tensor_model_parallel_group()

        # Map communicator names to their attributes
        comm_mapping = {
            "torch_native": "device_group",
            "pynccl": "pynccl_comm",
            "symm_mem": "symm_mem_comm",
            "custom_ar": "ca_comm",
            "quick_ar": "qr_comm",
            "symm_mem_one_shot": "symm_mem_one_shot_comm",
        }

        if not is_hip():
            comm_mapping["pymscclpp"] = "pymscclpp_comm"

        # Initialize communicators
        for name, attr_name in comm_mapping.items():
            try:
                comm = getattr(group, attr_name, None)
                if comm is not None:
                    # Try to enable disabled communicators
                    if hasattr(comm, "disabled") and comm.disabled:
                        try:
                            comm.disabled = False
                            if self.rank == 0:
                                print(f"Enabled {name} communicator")
                        except Exception:
                            if self.rank == 0:
                                print(f"Could not enable {name} communicator")
                    self.communicators[name] = comm
                else:
                    self.communicators[name] = None
                    if self.rank == 0:
                        print(f"Info: {name} communicator not available")
            except Exception as e:
                self.communicators[name] = None
                if self.rank == 0:
                    print(f"Info: {name} communicator initialization failed: {e}")

        self.initialized = True

    def get_implementation_function(self, impl_name: str) -> Callable:
        """Get the appropriate all-reduce function for the implementation"""
        comm = self.communicators.get(impl_name)

        if impl_name == "torch_native":

            def torch_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                dist.all_reduce(tensor, group=comm)
                return tensor

            return torch_allreduce

        elif impl_name == "symm_mem":

            def symm_mem_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None or (hasattr(comm, "disabled") and comm.disabled):
                    raise RuntimeError("symm_mem communicator is not available")
                if not comm.should_symm_mem_allreduce(tensor):
                    raise RuntimeError(
                        f"symm_mem cannot handle tensor: dtype={tensor.dtype}, size={tensor.numel() * tensor.element_size()}"
                    )
                return comm.all_reduce(tensor)

            return symm_mem_allreduce

        elif impl_name == "custom_ar":

            def custom_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None or (hasattr(comm, "disabled") and comm.disabled):
                    raise RuntimeError("custom_ar communicator is not available")
                result = comm.custom_all_reduce(tensor)
                if result is None:
                    raise RuntimeError("custom_ar cannot handle this tensor")
                return result

            return custom_allreduce

        elif impl_name == "quick_ar":

            def quick_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None:
                    raise RuntimeError("quick_ar communicator is not available")
                if hasattr(comm, "disabled") and comm.disabled:
                    comm.disabled = False
                if not comm.should_quick_allreduce(tensor):
                    raise RuntimeError(
                        f"quick_ar cannot handle tensor: dtype={tensor.dtype}, size={tensor.numel() * tensor.element_size()}"
                    )
                return comm.quick_all_reduce(tensor)

            return quick_allreduce

        elif impl_name == "pynccl":

            def pynccl_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None:
                    raise RuntimeError("pynccl communicator is not available")
                result = tensor.clone()
                with comm.change_state(enable=True, stream=torch.cuda.current_stream()):
                    comm.all_reduce(result)
                return result

            return pynccl_allreduce

        elif impl_name == "pymscclpp":

            def pymscclpp_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None or (hasattr(comm, "disabled") and comm.disabled):
                    raise RuntimeError("pymscclpp communicator is not available")
                return comm.all_reduce(tensor)

            return pymscclpp_allreduce

        elif impl_name == "symm_mem_one_shot":

            def symm_mem_one_shot_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if comm is None or (hasattr(comm, "disabled") and comm.disabled):
                    raise RuntimeError(
                        "symm_mem_one_shot communicator is not available"
                    )
                if not comm.should_one_shot_allreduce(tensor):
                    raise RuntimeError(
                        f"symm_mem_one_shot cannot handle tensor: dtype={tensor.dtype}, size={tensor.numel() * tensor.element_size()}"
                    )
                return comm.one_shot_all_reduce(tensor)

            return symm_mem_one_shot_allreduce

        else:
            raise ValueError(f"Unknown implementation: {impl_name}")

    def benchmark_eager_mode(
        self,
        impl_name: str,
        tensor: torch.Tensor,
        warmup_iters: int = 5,
        test_iters: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark all-reduce in eager mode

        Returns:
            Tuple of (output_tensor, time_per_iter_us)
        """
        func = self.get_implementation_function(impl_name)

        # Warmup
        for _ in range(warmup_iters):
            _ = func(tensor.clone())
            torch.cuda.synchronize()

        # Get output for correctness checking
        output_tensor = func(tensor.clone())

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        dist.barrier()

        start_event.record()
        for _ in range(test_iters):
            _ = func(tensor.clone())
        end_event.record()

        end_event.synchronize()

        # Convert to microseconds per iteration
        time_per_iter_us = start_event.elapsed_time(end_event) * 1000 / test_iters
        return output_tensor, time_per_iter_us

    def benchmark_graph_mode(
        self,
        impl_name: str,
        tensor: torch.Tensor,
        warmup_iters: int = 5,
        test_iters: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark all-reduce in graph mode

        Returns:
            Tuple of (output_tensor, time_per_iter_us)
        """
        # torch.distributed.all_reduce cannot work in CUDA graphs
        if impl_name == "torch_native":
            return self.benchmark_eager_mode(
                impl_name, tensor, warmup_iters, test_iters
            )

        func = self.get_implementation_function(impl_name)

        # Graph capture
        try:
            with graph_capture() as graph_capture_context:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    graph_out = func(tensor)
        except (AssertionError, RuntimeError) as e:
            if "tensor model parallel group is not initialized" in str(
                e
            ) or "CUDA" in str(e):
                return self.benchmark_eager_mode(
                    impl_name, tensor, warmup_iters, test_iters
                )
            # Fallback to simple graph capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = func(tensor)

        # Warmup
        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize()

        # Get output for correctness checking
        graph.replay()
        func_output = graph_out.clone()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies: List[float] = []
        for _ in range(test_iters):
            torch.cuda.synchronize()
            dist.barrier()
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        # Convert to microseconds per iteration
        time_per_iter_us = sum(latencies) / len(latencies) * 1000
        graph.reset()
        return func_output, time_per_iter_us


def human_readable_size(size_bytes: int, decimal_places: int = 1) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size_bytes < 1024.0 or unit == "PiB":
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.{decimal_places}f} {unit}"


def print_mode_results_table(
    results: List[Dict[str, Any]], mode: str, implementations: List[str]
):
    """Print benchmark results for a specific mode in a formatted table"""
    if not results:
        return

    # Filter implementations that have results for this mode
    mode_suffix = f"_{mode}"
    available_impls = [
        impl
        for impl in implementations
        if any(f"{impl}{mode_suffix}" in result for result in results)
    ]

    if not available_impls:
        return

    # Print header
    mode_title = mode.upper()
    print(f"\n{'='*80}")
    print(f"ALL-REDUCE BENCHMARK RESULTS - {mode_title} MODE")
    print(f"Time in microseconds (µs), lower is better")
    print(f"{'='*80}")

    # Create table header
    header = (
        "| Size      | "
        + " | ".join([f"{impl:>12}" for impl in available_impls])
        + " |"
    )
    separator = (
        "|-----------|-" + "-|-".join(["-" * 12 for _ in available_impls]) + "-|"
    )

    print(header)
    print(separator)

    # Print rows
    for result in results:
        row = f"| {result['size_human']:>9} | "
        for impl in available_impls:
            key = f"{impl}{mode_suffix}"
            time_val = result.get(key, "N/A")
            if isinstance(time_val, (int, float)):
                row += f"{time_val:>12.1f} | "
            else:
                row += f"{str(time_val):>12} | "
        print(row)

    print("=" * 80)


def print_mode_fastest(
    results: List[Dict[str, Any]], mode: str, implementations: List[str]
):
    """Find and print the fastest implementation for each size in a specific mode"""
    if not results:
        return

    mode_suffix = f"_{mode}"
    mode_title = mode.upper()

    print(f"\nFASTEST IMPLEMENTATIONS PER SIZE - {mode_title} MODE:")
    print("=" * 60)

    for result in results:
        size = result["size_human"]

        # Find implementation with minimum time for this mode
        impl_times = {}
        for impl in implementations:
            key = f"{impl}{mode_suffix}"
            if key in result and isinstance(result[key], (int, float)):
                impl_times[impl] = result[key]

        if impl_times:
            fastest_impl = min(impl_times, key=impl_times.get)
            fastest_time = impl_times[fastest_impl]
            print(f"{size:>9}: {fastest_impl:>20} ({fastest_time:>8.1f} µs)")
        else:
            print(f"{size:>9}: {'No successful runs':>20}")

    print("=" * 60)


def check_correctness(
    rank: int, mode: str, size_human: str, outputs: Dict[str, torch.Tensor]
):
    """Check correctness of all implementations against torch_native"""
    if rank != 0 or "torch_native" not in outputs:
        return

    reference_output = outputs["torch_native"]

    print(f"\nCorrectness Check - {mode.upper()} mode @ {size_human}:")
    print("-" * 60)

    for impl, output in outputs.items():
        if impl == "torch_native":
            continue

        try:
            torch.testing.assert_close(
                output,
                reference_output,
                rtol=1e-2,
                atol=1e-2,
                msg=f"{mode} mode: {impl} output doesn't match torch_native",
            )
            print(f"  ✓ {impl:>20}: PASSED")
        except AssertionError:
            max_diff = torch.max(torch.abs(output - reference_output)).item()
            mean_diff = torch.mean(torch.abs(output - reference_output)).item()
            print(
                f"  ✗ {impl:>20}: FAILED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})"
            )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive all-reduce benchmark across implementations and modes"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum size as power of 2 (e.g., 10 for 2^10 = 1KB)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=27,
        help="Maximum size as power of 2 (e.g., 27 for 2^27 = 128MB)",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Step size between power of 2 sizes"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for benchmark tensors",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["eager", "graph", "both"],
        help="Benchmark mode: eager, graph, or both",
    )
    parser.add_argument("--profile", action="store_true", help="Enable torch profiling")
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test-iters", type=int, default=20, help="Number of test iterations"
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Enable correctness checking (compares all implementations against torch_native)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Set device
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Create benchmark instance
    benchmark = AllReduceBenchmark(world_size, rank, device)
    benchmark.initialize_communicators()

    # Get dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Size range
    if IS_CI:
        size_range = range(10, 12)
    else:
        size_range = range(args.min_size, min(args.max_size, 30) + 1, args.step)

    # Available implementations
    implementations = ["torch_native", "symm_mem", "custom_ar", "pynccl"]
    if is_hip():
        implementations.extend(["quick_ar", "symm_mem_one_shot"])
    else:
        implementations.append("pymscclpp")

    # Filter to only available implementations
    implementations = [
        impl
        for impl in implementations
        if benchmark.communicators.get(impl) is not None
    ]

    results = []
    modes_to_run = ["eager", "graph"] if args.mode == "both" else [args.mode]

    # Profiler context
    prof_ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        if args.profile
        else nullcontext()
    )

    with prof_ctx as prof:
        for size_power in size_range:
            size_bytes = 2**size_power

            # Skip if too large
            if size_bytes * torch.tensor(0, dtype=dtype).element_size() > 2**30:
                if rank == 0:
                    print(
                        f"Skipping size {human_readable_size(size_bytes)} - too large"
                    )
                continue

            # Create test tensor
            tensor_size = size_bytes // torch.tensor(0, dtype=dtype).element_size()
            test_tensor = torch.randint(
                1, 16, (int(tensor_size),), dtype=dtype, device=device
            )

            if rank == 0:
                print(f"\n{'='*80}")
                print(
                    f"Benchmarking size: {human_readable_size(size_bytes)} ({tensor_size} elements)"
                )
                print(f"{'='*80}")

            result = {
                "size_bytes": size_bytes,
                "size_human": human_readable_size(size_bytes),
            }

            # Store outputs for correctness checking
            eager_outputs = {}
            graph_outputs = {}

            # Benchmark each implementation
            for impl in implementations:
                for mode in modes_to_run:
                    key = f"{impl}_{mode}"
                    try:
                        if mode == "eager":
                            output, time_us = benchmark.benchmark_eager_mode(
                                impl, test_tensor, args.warmup_iters, args.test_iters
                            )
                            eager_outputs[impl] = output
                        else:  # graph
                            output, time_us = benchmark.benchmark_graph_mode(
                                impl, test_tensor, args.warmup_iters, args.test_iters
                            )
                            graph_outputs[impl] = output

                        result[key] = time_us

                        if rank == 0:
                            print(f"  {impl:>20} ({mode:>5}): {time_us:>8.1f} µs")

                    except Exception as e:
                        result[key] = "FAILED"
                        if rank == 0:
                            error_msg = str(e)
                            if (
                                "communicator is not available" in error_msg
                                or "cannot handle tensor" in error_msg
                            ):
                                print(
                                    f"  {impl:>20} ({mode:>5}): SKIPPED - {error_msg}"
                                )
                            else:
                                print(f"  {impl:>20} ({mode:>5}): FAILED - {e}")

            # Correctness checking
            if args.check_correctness:
                if "eager" in modes_to_run and eager_outputs:
                    check_correctness(
                        rank, "eager", human_readable_size(size_bytes), eager_outputs
                    )
                if "graph" in modes_to_run and graph_outputs:
                    check_correctness(
                        rank, "graph", human_readable_size(size_bytes), graph_outputs
                    )

            results.append(result)

    # Print results on rank 0
    if rank == 0:
        # Print separate tables for each mode
        if "eager" in modes_to_run:
            print_mode_results_table(results, "eager", implementations)
            print_mode_fastest(results, "eager", implementations)

        if "graph" in modes_to_run:
            print_mode_results_table(results, "graph", implementations)
            print_mode_fastest(results, "graph", implementations)

    # Export profile if enabled
    if args.profile and rank == 0 and prof is not None:
        prof_dir = "prof/comprehensive_allreduce"
        os.makedirs(prof_dir, exist_ok=True)
        prof.export_chrome_trace(f"{prof_dir}/trace_rank{rank}.json.gz")
        print(f"\nProfiler trace saved to {prof_dir}/trace_rank{rank}.json.gz")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
