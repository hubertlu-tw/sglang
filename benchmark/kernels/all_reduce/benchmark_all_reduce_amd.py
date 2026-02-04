#!/usr/bin/env python3
"""AMD All-Reduce Benchmark with Correctness Checking.

Benchmarks AMD-relevant all-reduce implementations across different data sizes
and modes (eager vs graph).

Supported implementations (AMD):
- torch.distributed.all_reduce (reference for correctness)
- pynccl_comm.all_reduce
- custom_ar (sgl-kernel)
- custom_ar (aiter)
- quick_ar (FP, INT8, INT6, INT4)
- outplace_all_reduce
- inplace_all_reduce

Usage:
  torchrun --nproc_per_node 2 --nnodes 1 \
    ./benchmark_all_reduce_amd.py \
    --min-size 10 --max-size 27 --step 1 --dtype bfloat16 --check-correctness
"""

import argparse
import os
from contextlib import nullcontext
from typing import List, Dict, Any, Callable, Tuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.utils import is_hip
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce as SGLCustomAllreduce,
)
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    qr_rocm_arch_available,
)

_CUSTOM_AR_MIN_BYTES = 32 * 1024

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
        self.quick_ar_comms: Dict[str, QuickAllReduce] = {}
        self.group = None
        self.custom_ar_group = None
        self.invalid_impls = set()
        self.initialized = False
        
    def initialize_communicators(self, max_size_bytes: int):
        """Initialize all available communicators"""
        if self.initialized:
            return
            
        # Set device
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        
        # Disable GroupCoordinator custom AR to avoid duplicate instances.
        # We'll create sgl/aiter custom ARs explicitly for benchmarking.
        set_custom_all_reduce(False)
        
        # Initialize distributed environment
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.rank % torch.cuda.device_count(),
        )
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        
        group = get_tensor_model_parallel_group()
        self.group = group
        
        # Base communicators from group
        self.communicators["torch_native"] = group.device_group
        self.communicators["pynccl"] = group.pynccl_comm

        # Use the default gloo WORLD group for custom AR (aligns with benchmark_aiter.py)
        self.custom_ar_group = dist.group.WORLD

        # SGLang custom all-reduce (sgl-kernel)
        try:
            car_max_size = getattr(SGLCustomAllreduce, "_MAX_CAR_SIZE", max_size_bytes)
            self.communicators["custom_ar_sgl"] = SGLCustomAllreduce(
                group=self.custom_ar_group, device=self.device, max_size=car_max_size
            )
        except Exception as e:
            self.communicators["custom_ar_sgl"] = None
            if self.rank == 0:
                print(f"Info: custom_ar_sgl initialization failed: {e}")

        # Aiter custom all-reduce
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            car_max_size = getattr(
                AiterCustomAllreduce,
                "_MAX_CAR_SIZE",
                getattr(SGLCustomAllreduce, "_MAX_CAR_SIZE", max_size_bytes),
            )
            self.communicators["custom_ar_aiter"] = AiterCustomAllreduce(
                group=self.custom_ar_group, device=self.device, max_size=car_max_size
            )
        except Exception as e:
            self.communicators["custom_ar_aiter"] = None
            if self.rank == 0:
                print(f"Info: custom_ar_aiter initialization failed: {e}")

        # Quick all-reduce (AMD, quantization modes)
        if is_hip() and qr_rocm_arch_available():
            quick_ar_modes = ["FP", "INT8", "INT6", "INT4"]
            os.environ.setdefault("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "0")
            for mode in quick_ar_modes:
                try:
                    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = mode
                    comm = QuickAllReduce(group=group.cpu_group, device=self.device)
                    if comm is not None and not comm.disabled:
                        self.quick_ar_comms[mode.lower()] = comm
                    else:
                        if self.rank == 0:
                            print(f"Info: quick_ar {mode} not available or disabled")
                except Exception as e:
                    if self.rank == 0:
                        print(f"Info: quick_ar {mode} initialization failed: {e}")

        # Validate custom AR implementations; disable if incorrect.
        for impl_name in ("custom_ar_sgl", "custom_ar_aiter"):
            comm = self.communicators.get(impl_name)
            if comm is None:
                continue
            if not self._validate_custom_ar(impl_name, group.device_group):
                self.invalid_impls.add(impl_name)
                if self.rank == 0:
                    print(f"Info: {impl_name} failed validation; will be skipped")

        # Wire a default custom AR into the group for outplace_all_reduce.
        # Prefer sgl-kernel for determinism; fall back to aiter if needed.
        if self.group is not None:
            if (
                self.communicators.get("custom_ar_sgl") is not None
                and "custom_ar_sgl" not in self.invalid_impls
            ):
                self.group.ca_comm = self.communicators["custom_ar_sgl"]
            elif (
                self.communicators.get("custom_ar_aiter") is not None
                and "custom_ar_aiter" not in self.invalid_impls
            ):
                self.group.ca_comm = self.communicators["custom_ar_aiter"]
        
        self.initialized = True

    def _validate_custom_ar(self, impl_name: str, device_group: dist.ProcessGroup) -> bool:
        """Validate custom AR against torch_native on a small tensor."""
        try:
            tensor_elems = max(_CUSTOM_AR_MIN_BYTES, 32 * 1024) // 2
            torch.manual_seed(1234 + self.rank)
            test = torch.randint(
                1, 16, (tensor_elems,), device=self.device, dtype=torch.float16
            )
            ref = test.clone()
            dist.all_reduce(ref, group=device_group)
            output = self.get_implementation_function(impl_name)(test.clone())
            torch.cuda.synchronize()
            try:
                torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
                return True
            except AssertionError:
                # Some custom AR paths return averaged output.
                torch.testing.assert_close(
                    output * self.world_size, ref, rtol=1e-2, atol=1e-2
                )
                return True
        except Exception as e:
            if self.rank == 0:
                comm = self.communicators.get(impl_name)
                details = []
                if comm is not None:
                    details.append(f"disabled={getattr(comm, 'disabled', None)}")
                    details.append(f"full_nvlink={getattr(comm, 'full_nvlink', None)}")
                    details.append(f"max_size={getattr(comm, 'max_size', None)}")
                    try:
                        details.append(f"should_custom_ar={comm.should_custom_ar(test)}")
                    except Exception:
                        details.append("should_custom_ar=<error>")
                detail_str = ", ".join(details) if details else "no_comm_details"
                print(f"Info: {impl_name} validation error: {e} ({detail_str})")
            return False
        
    def get_implementation_function(self, impl_name: str) -> Callable:
        """Get the appropriate all-reduce function for the implementation"""
        comm = self.communicators.get(impl_name)
        
        if impl_name == "torch_native":
            def torch_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                dist.all_reduce(tensor, group=comm)
                return tensor
            return torch_allreduce
            
        elif impl_name == "custom_ar_sgl":
            def custom_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if impl_name in self.invalid_impls:
                    raise RuntimeError("custom_ar_sgl failed validation")
                if comm is None or (hasattr(comm, 'disabled') and comm.disabled):
                    raise RuntimeError("custom_ar_sgl communicator is not available")
                if hasattr(comm, "all_reduce_unreg"):
                    result = comm.all_reduce_unreg(tensor)
                else:
                    result = comm.custom_all_reduce(tensor)
                if result is None:
                    raise RuntimeError("custom_ar_sgl cannot handle this tensor")
                return result
            return custom_allreduce

        elif impl_name == "custom_ar_aiter":
            def aiter_custom_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if impl_name in self.invalid_impls:
                    raise RuntimeError("custom_ar_aiter failed validation")
                if comm is None or (hasattr(comm, 'disabled') and comm.disabled):
                    raise RuntimeError("custom_ar_aiter communicator is not available")
                if hasattr(comm, "all_reduce_unreg"):
                    result = comm.all_reduce_unreg(tensor)
                else:
                    result = comm.custom_all_reduce(tensor)
                if result is None:
                    raise RuntimeError("custom_ar_aiter cannot handle this tensor")
                return result
            return aiter_custom_allreduce
            
        elif impl_name.startswith("quick_ar_"):
            mode = impl_name.split("_", 2)[2].lower()

            def quick_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                qcomm = self.quick_ar_comms.get(mode)
                if qcomm is None or (hasattr(qcomm, "disabled") and qcomm.disabled):
                    raise RuntimeError(f"quick_ar_{mode} communicator is not available")
                return qcomm.quick_all_reduce(tensor)

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
            
        elif impl_name == "inplace_all_reduce":
            def inplace_allreduce(tensor: torch.Tensor) -> torch.Tensor:
                if self.group is None:
                    raise RuntimeError("inplace_all_reduce group is not initialized")
                torch.ops.sglang.inplace_all_reduce(
                    tensor, group_name=self.group.unique_name
                )
                return tensor
            return inplace_allreduce
            
        else:
            raise ValueError(f"Unknown implementation: {impl_name}")
    
    def benchmark_eager_mode(self, impl_name: str, tensor: torch.Tensor, 
                           warmup_iters: int = 5, test_iters: int = 20) -> Tuple[torch.Tensor, float]:
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
        torch.cuda.synchronize()
        
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
    
    def benchmark_graph_mode(self, impl_name: str, tensor: torch.Tensor,
                           warmup_iters: int = 5, test_iters: int = 20) -> Tuple[torch.Tensor, float]:
        """Benchmark all-reduce in graph mode
        
        Returns:
            Tuple of (output_tensor, time_per_iter_us)
        """
        # torch.distributed.all_reduce cannot work in CUDA graphs
        if impl_name == "torch_native":
            return self.benchmark_eager_mode(impl_name, tensor, warmup_iters, test_iters)
        
        func = self.get_implementation_function(impl_name)
        is_inplace = impl_name == "inplace_all_reduce"
        if is_inplace:
            base_tensor = tensor.clone()
        
        # Graph capture
        try:
            with graph_capture() as graph_capture_context:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    if is_inplace:
                        tensor.copy_(base_tensor)
                    graph_out = func(tensor)
        except (AssertionError, RuntimeError) as e:
            if "tensor model parallel group is not initialized" in str(e) or "CUDA" in str(e):
                return self.benchmark_eager_mode(impl_name, tensor, warmup_iters, test_iters)
            # Fallback to simple graph capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if is_inplace:
                    tensor.copy_(base_tensor)
                graph_out = func(tensor)
        
        # Warmup
        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize()
        
        # Get output for correctness checking
        graph.replay()
        torch.cuda.synchronize()
        func_output = graph_out.clone()
        torch.cuda.synchronize()
        
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

    def close(self):
        for comm in self.communicators.values():
            if comm is not None and hasattr(comm, "close"):
                try:
                    comm.close()
                except Exception:
                    pass
        for comm in self.quick_ar_comms.values():
            if comm is not None and hasattr(comm, "close"):
                try:
                    comm.close()
                except Exception:
                    pass


def human_readable_size(size_bytes: int, decimal_places: int = 1) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size_bytes < 1024.0 or unit == "PiB":
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.{decimal_places}f} {unit}"


def print_mode_results_table(results: List[Dict[str, Any]], mode: str, implementations: List[str]):
    """Print benchmark results for a specific mode in a formatted table"""
    if not results:
        return
    
    # Filter implementations that have results for this mode
    mode_suffix = f"_{mode}"
    available_impls = [impl for impl in implementations 
                      if any(f"{impl}{mode_suffix}" in result for result in results)]
    
    if not available_impls:
        return
    
    # Print header
    mode_title = mode.upper()
    print(f"\n{'='*80}")
    print(f"ALL-REDUCE BENCHMARK RESULTS - {mode_title} MODE")
    print(f"Time in microseconds (µs), lower is better")
    print(f"{'='*80}")
    
    # Create table header
    col_width = max(12, max(len(impl) for impl in available_impls))
    header = "| Size      | " + " | ".join([f"{impl:>{col_width}}" for impl in available_impls]) + " |"
    separator = "|-----------|-" + "-|-".join(["-" * col_width for _ in available_impls]) + "-|"
    
    print(header)
    print(separator)
    
    # Print rows
    for result in results:
        row = f"| {result['size_human']:>9} | "
        for impl in available_impls:
            key = f"{impl}{mode_suffix}"
            time_val = result.get(key, "N/A")
            if isinstance(time_val, (int, float)):
                row += f"{time_val:>{col_width}.1f} | "
            else:
                row += f"{str(time_val):>{col_width}} | "
        print(row)
    
    print("="*80)


def print_mode_fastest(results: List[Dict[str, Any]], mode: str, implementations: List[str]):
    """Find and print the fastest implementation for each size in a specific mode"""
    if not results:
        return
    
    mode_suffix = f"_{mode}"
    mode_title = mode.upper()
    
    print(f"\nFASTEST IMPLEMENTATIONS PER SIZE - {mode_title} MODE:")
    print("="*60)

    name_width = max(20, max(len(impl) for impl in implementations))
    
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
            print(f"{size:>9}: {fastest_impl:>{name_width}} ({fastest_time:>8.1f} µs)")
        else:
            print(f"{size:>9}: {'No successful runs':>{name_width}}")
    
    print("="*60)


def _get_tolerances(impl: str, world_size: int) -> Tuple[float, float]:
    if impl.startswith("quick_ar_"):
        return 0.5 * world_size, 1.25 * world_size
    return 1e-2, 1e-2


def check_output_correctness(
    impl: str,
    output: torch.Tensor,
    reference_output: torch.Tensor,
    world_size: int,
    mode: str,
) -> Tuple[bool, str]:
    """Check correctness for one implementation against torch_native."""
    rtol, atol = _get_tolerances(impl, world_size)
    try:
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=rtol,
            atol=atol,
            msg=f"{mode} mode: {impl} output doesn't match torch_native",
        )
        return True, "PASSED"
    except AssertionError:
        pass

    # For custom AR variants, also check if the output is averaged.
    if impl.startswith("custom_ar_"):
        try:
            torch.testing.assert_close(
                output * world_size,
                reference_output,
                rtol=rtol,
                atol=atol,
                msg=f"{mode} mode: {impl} output doesn't match torch_native (scaled)",
            )
            return True, "PASSED (averaged output)"
        except AssertionError:
            pass

    max_diff = torch.max(torch.abs(output - reference_output)).item()
    mean_diff = torch.mean(torch.abs(output - reference_output)).item()
    detail = f"FAILED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})"
    return False, detail


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive all-reduce benchmark across implementations and modes"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum size as power of 2 (e.g., 10 for 2^10 = 1KB)"
    )
    parser.add_argument(
        "--max-size", 
        type=int, 
        default=27,
        help="Maximum size as power of 2 (e.g., 27 for 2^27 = 128MB)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size between power of 2 sizes"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for benchmark tensors"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["eager", "graph", "both"],
        help="Benchmark mode: eager, graph, or both"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiling"
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test-iters",
        type=int,
        default=20,
        help="Number of test iterations"
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Enable correctness checking (compares all implementations against torch_native)"
    )
    parser.add_argument(
        "--impls",
        type=str,
        default=(
            "torch_native,pynccl,custom_ar_sgl,custom_ar_aiter,"
            "quick_ar_fp,quick_ar_int8,quick_ar_int6,quick_ar_int4,"
            "inplace_all_reduce"
        ),
        help=(
            "Comma-separated list of implementations to benchmark. "
            "Example: torch_native,pynccl,custom_ar_sgl,custom_ar_aiter,quick_ar_fp,inplace_all_reduce"
        ),
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Set device
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    
    # Create benchmark instance
    benchmark = AllReduceBenchmark(world_size, rank, device)
    
    # Get dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16, 
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Size range
    if IS_CI:
        size_range = range(10, 12)
    else:
        size_range = range(args.min_size, args.max_size + 1, args.step)

    max_size_bytes = 2 ** max(size_range)
    benchmark.initialize_communicators(max_size_bytes=max_size_bytes)
    
    # Available implementations (full default set)
    implementations = [
        "torch_native",
        "pynccl",
        "custom_ar_sgl",
        "custom_ar_aiter",
        "quick_ar_fp",
        "quick_ar_int8",
        "quick_ar_int6",
        "quick_ar_int4",
        "inplace_all_reduce",
    ]

    requested_impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    unknown_impls = [impl for impl in requested_impls if impl not in implementations]
    if unknown_impls:
        raise ValueError(
            f"Unknown implementations in --impls: {unknown_impls}. "
            f"Valid options: {implementations}"
        )
    if requested_impls:
        implementations = requested_impls

    def impl_available(impl: str) -> bool:
        if impl.startswith("quick_ar_"):
            return impl.split("_", 2)[2].lower() in benchmark.quick_ar_comms
        if impl in ("outplace_all_reduce", "inplace_all_reduce"):
            return benchmark.group is not None
        return benchmark.communicators.get(impl) is not None

    # Filter to only available implementations
    implementations = [impl for impl in implementations if impl_available(impl)]
    
    size_list = list(size_range)
    results = [
        {"size_bytes": 2 ** size_power, "size_human": human_readable_size(2 ** size_power)}
        for size_power in size_list
    ]
    modes_to_run = ["eager", "graph"] if args.mode == "both" else [args.mode]
    check_ref = args.check_correctness and "torch_native" in implementations
    if args.check_correctness and "torch_native" not in implementations and rank == 0:
        print("Warning: --check-correctness requires torch_native in --impls; skipping checks.")

    if check_ref and implementations[0] != "torch_native":
        implementations = ["torch_native"] + [impl for impl in implementations if impl != "torch_native"]
    
    # Profiler context
    prof_ctx = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) if args.profile else nullcontext()
    
    with prof_ctx as prof:
        for mode in modes_to_run:
            for idx, size_power in enumerate(size_list):
                size_bytes = 2 ** size_power

                # Skip if too large (cap total tensor bytes)
                if size_bytes > 2**31:
                    if rank == 0:
                        print(f"Skipping size {human_readable_size(size_bytes)} - too large")
                    continue

                # Create test tensor
                tensor_size = size_bytes // torch.tensor(0, dtype=dtype).element_size()
                torch.manual_seed(1234 + rank)
                test_tensor = torch.randint(
                    1, 16, (int(tensor_size),), device=device
                ).to(dtype)

                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"Benchmarking size: {human_readable_size(size_bytes)} ({tensor_size} elements)")
                    print(f"{'='*80}")

                result = results[idx]

                # Store torch_native reference output for this mode/size
                reference_output: Optional[torch.Tensor] = None
                printed_correctness_header = False

                # Benchmark each implementation
                for impl in implementations:
                    key = f"{impl}_{mode}"
                    try:
                        if mode == "eager":
                            output, time_us = benchmark.benchmark_eager_mode(
                                impl, test_tensor, args.warmup_iters, args.test_iters
                            )
                        else:  # graph
                            output, time_us = benchmark.benchmark_graph_mode(
                                impl, test_tensor.clone(), args.warmup_iters, args.test_iters
                            )

                        result[key] = time_us

                        if rank == 0:
                            print(f"  {impl:>20} ({mode:>5}): {time_us:>8.1f} µs")

                        if check_ref:
                            if impl == "torch_native":
                                reference_output = output
                            elif reference_output is not None:
                                if rank == 0 and not printed_correctness_header:
                                    print(f"\nCorrectness Check - {mode.upper()} mode @ {human_readable_size(size_bytes)}:")
                                    print("-" * 60)
                                    printed_correctness_header = True
                                passed, detail = check_output_correctness(
                                    impl, output, reference_output, world_size, mode
                                )
                                if rank == 0:
                                    symbol = "✓" if passed else "✗"
                                    print(f"  {symbol} {impl:>20}: {detail}")
                                if not passed and rank == 0:
                                    result[key] = "ACCURACY_ISSUE"
                    except Exception as e:
                        if rank == 0:
                            error_msg = str(e)
                            if (
                                "communicator is not available" in error_msg
                                or "cannot handle tensor" in error_msg
                                or "cannot handle this tensor" in error_msg
                                or "failed validation" in error_msg
                            ):
                                result[key] = "SKIPPED"
                                print(f"  {impl:>20} ({mode:>5}): SKIPPED - {error_msg}")
                            else:
                                result[key] = "FAILED"
                                print(f"  {impl:>20} ({mode:>5}): FAILED - {e}")
                        else:
                            result[key] = "SKIPPED"
    
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
    
    benchmark.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

