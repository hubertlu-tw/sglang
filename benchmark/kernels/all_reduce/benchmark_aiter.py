"""
Benchmark SGLang vs Aiter custom all-reduce across message sizes with correctness checking.
Usage:
    torchrun --nproc_per_node=2 benchmark_aiter.py
    torchrun --nproc_per_node=4 benchmark_aiter.py
    torchrun --nproc_per_node=8 benchmark_aiter.py
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

# Add SGLang distributed imports
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang vs Aiter custom all-reduce across message sizes."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="Process group backend for the custom-AR control path (must NOT be nccl).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per size per implementation.",
    )
    parser.add_argument(
        "--iters-small",
        type=int,
        default=50,
        help="Benchmark iterations for sizes <= 1MB.",
    )
    parser.add_argument(
        "--iters-large",
        type=int,
        default=20,
        help="Benchmark iterations for sizes > 1MB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration timings on rank 0 for debugging.",
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Enable correctness checking for all-reduce operations.",
    )
    parser.add_argument(
        "--graph-mode",
        action="store_true",
        help="Test in CUDA graph mode instead of eager mode.",
    )
    parser.add_argument(
        "--fp8-quant",
        action="store_true",
        help="Enable FP8 quantization for all-reduce operations (sets USE_AITER_CAR_FP8=1).",
    )
    parser.add_argument(
        "--use-tensor-model-parallel",
        action="store_true",
        help="Use tensor_model_parallel_all_reduce instead of direct custom all-reduce calls.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for tensors (default: bfloat16)",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="both",
        choices=["sglang", "aiter", "both"],
        help="Which implementation to test: sglang, aiter, or both (default: both)",
    )
    return parser.parse_args()


def get_env_rank_world() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def init_dist(backend: str):
    rank, world_size, local_rank = get_env_rank_world()
    if not dist.is_initialized():
        # Use SGLang's proper initialization instead of direct dist.init_process_group
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend=backend,
        )
        # Initialize model parallel for custom all-reduce support
        initialize_model_parallel(tensor_model_parallel_size=world_size)


def get_device(local_rank: int) -> torch.device:
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def human_size(num_bytes: int) -> str:
    units = [("B", 1), ("K", 1024), ("M", 1024 * 1024), ("G", 1024 * 1024 * 1024)]
    for suf, base in reversed(units):
        if num_bytes % base == 0 and num_bytes >= base:
            val = num_bytes // base
            return f"{val}{suf}"
    return f"{num_bytes}B"


def get_message_sizes() -> List[int]:
    return [
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
    ]


def check_correctness_allreduce(
    original: torch.Tensor,
    reduced: torch.Tensor,
    world_size: int,
    rank: int,
    pg: dist.ProcessGroup,
    name: str,
    size_bytes: int,
    dtype: torch.dtype,
) -> bool:
    """
    Check if all-reduce result is correct by gathering all inputs and computing expected sum.
    Returns True if correct, False if incorrect, and prints debug info on rank 0.
    """
    if original.shape != reduced.shape:
        if rank == 0:
            print(
                f"ERROR: Shape mismatch in {name} at size {human_size(size_bytes)}: "
                f"original {original.shape} != reduced {reduced.shape}"
            )
        return False

    # Convert to float32 for more precise comparison
    original_f32 = original.float()
    reduced_f32 = reduced.float()

    # Gather all original tensors to compute expected sum
    if world_size > 1:
        # Use all_gather to collect all inputs
        gathered = [torch.empty_like(original_f32) for _ in range(world_size)]
        dist.all_gather(gathered, original_f32, group=pg)

        # Compute expected sum
        expected_sum = torch.zeros_like(original_f32)
        for tensor in gathered:
            expected_sum += tensor
    else:
        # Single rank case
        expected_sum = original_f32

    # Use dtype-specific tolerance for floating point comparison
    if dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-8
    elif dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-5
    elif dtype == torch.bfloat16:
        rtol = 1e-2
        atol = 1e-4
    else:
        rtol = 1e-3
        atol = 1e-5

    is_correct = torch.allclose(reduced_f32, expected_sum, rtol=rtol, atol=atol)

    # Debug info on rank 0 for incorrect results
    if not is_correct and rank == 0:
        diff = torch.abs(reduced_f32 - expected_sum)
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()

        print(f"CORRECTNESS FAILURE in {name} at size {human_size(size_bytes)}:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Avg difference: {avg_diff:.6f}")
        print(
            f"  Expected range: [{torch.min(expected_sum).item():.3f}, {torch.max(expected_sum).item():.3f}]"
        )
        print(
            f"  Actual range:   [{torch.min(reduced_f32).item():.3f}, {torch.max(reduced_f32).item():.3f}]"
        )

        # Check for NaN or Inf values
        if torch.isnan(reduced_f32).any() or torch.isinf(reduced_f32).any():
            print(f"  WARNING: NaN or Inf values detected in result!")
        if torch.isnan(expected_sum).any() or torch.isinf(expected_sum).any():
            print(f"  WARNING: NaN or Inf values detected in expected sum!")

    return is_correct


@torch.inference_mode()
def run_custom_allreduce(
    comm, inp: torch.Tensor, fp8_quant: bool = False
) -> Optional[torch.Tensor]:
    """Run custom all-reduce and return None if disabled or falling back to NCCL."""
    # Check if custom all-reduce is disabled
    if hasattr(comm, "disabled") and comm.disabled:
        return None

    # Check if custom all-reduce should be used for this input size
    if hasattr(comm, "should_custom_ar"):
        should_use = comm.should_custom_ar(inp)
        if not should_use:
            return None

    # CRITICAL: If we reach here, the implementation claims it can handle this size
    # but we need to verify it actually works correctly
    inp_size = inp.numel() * inp.element_size()

    # For SGLang implementation, be extra careful with larger sizes
    if (
        hasattr(comm, "__class__")
        and "SGLCustomAllreduce" in str(comm.__class__)
        and inp_size >= (4 * 1024 * 1024)
    ):  # 4MB+ for SGLang
        # SGLang has known issues with larger sizes, force NCCL fallback
        return None

    if hasattr(comm, "all_reduce_unreg"):
        return comm.all_reduce_unreg(inp)
    if hasattr(comm, "custom_all_reduce"):
        # Check if the implementation supports the open_fp8_quant parameter
        if (
            hasattr(comm.custom_all_reduce, "__code__")
            and "open_fp8_quant" in comm.custom_all_reduce.__code__.co_varnames
        ):
            return comm.custom_all_reduce(inp, open_fp8_quant=fp8_quant)
        # Fall back to regular call - FP8 will be handled by USE_AITER_CAR_FP8 env var if set
        return comm.custom_all_reduce(inp)
    raise RuntimeError("No known all-reduce method found on the communicator.")


@torch.inference_mode()
def run_tensor_model_parallel_all_reduce(
    inp: torch.Tensor, fp8_quant: bool = False
) -> torch.Tensor:
    """Run tensor model parallel all-reduce using the real use case API."""
    try:
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_reduce,
        )

        result = tensor_model_parallel_all_reduce(inp)
        if hasattr(result, "is_cuda") and not result.is_cuda:
            print(f"WARNING: tensor_model_parallel_all_reduce returned non-CUDA tensor")
        return result
    except ImportError:
        try:
            from aiter.dist.communication_op import tensor_model_parallel_all_reduce

            result = tensor_model_parallel_all_reduce(inp, open_fp8_quant=fp8_quant)
            if hasattr(result, "is_cuda") and not result.is_cuda:
                print(
                    f"WARNING: aiter tensor_model_parallel_all_reduce returned non-CUDA tensor"
                )
            return result
        except ImportError:
            raise RuntimeError(
                "tensor_model_parallel_all_reduce not available in either SGLang or Aiter"
            )


@torch.inference_mode()
def bench_impl(
    name: str,
    comm,
    sizes: List[int],
    device: torch.device,
    warmup: int,
    iters_small: int,
    iters_large: int,
    verbose: bool,
    pg: Optional[dist.ProcessGroup] = None,
    correctness: bool = False,
    graph_mode: bool = False,
    fp8_quant: bool = False,
    use_tensor_model_parallel: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[int, Optional[float], Optional[bool]]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    results: List[Tuple[int, Optional[float], Optional[bool]]] = []

    # For graph mode, we need to capture the graph
    graph = None
    graph_stream = None

    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        graph_stream = torch.cuda.Stream()

    for size_bytes in sizes:
        # Calculate elements based on dtype element size
        if dtype == torch.float32:
            elems = size_bytes // 4  # 4 bytes per element for float32
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            elems = size_bytes // 2  # 2 bytes per element for float16/bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Use random input
        inp = torch.empty(elems, dtype=dtype, device=device)
        inp.uniform_(0, 1)

        # Store original for correctness checking
        original_inp = inp.clone() if correctness else None

        disabled = False
        dist.barrier(group=pg)

        # Warmup
        for _ in range(warmup):
            torch.cuda.synchronize()
            if use_tensor_model_parallel:
                out = run_tensor_model_parallel_all_reduce(inp, fp8_quant=fp8_quant)
            else:
                out = run_custom_allreduce(comm, inp, fp8_quant=fp8_quant)
            torch.cuda.synchronize()
            if out is None:
                disabled = True
                break
        dist.barrier(group=pg)

        if disabled:
            if rank == 0:
                print(
                    f"[{name}] {human_size(size_bytes)}: custom AR disabled (skipped)"
                )
            results.append((size_bytes, None, None))
            continue

        num_iters = iters_small if size_bytes <= (1 * 1024 * 1024) else iters_large

        times_ms: List[float] = []
        correct_results: List[bool] = []

        for it in range(num_iters):
            dist.barrier(group=pg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Check if custom all-reduce is actually being used
            if use_tensor_model_parallel:
                out = run_tensor_model_parallel_all_reduce(inp, fp8_quant=fp8_quant)
            else:
                out = run_custom_allreduce(comm, inp, fp8_quant=fp8_quant)

            # Debug: Check what's happening with custom all-reduce
            if rank == 0 and verbose:
                inp_size = inp.numel() * inp.element_size()
                print(
                    f"[{name}] {human_size(size_bytes)}: Input size {human_size(inp_size)}"
                )

                if hasattr(comm, "disabled"):
                    print(
                        f"[{name}] {human_size(size_bytes)}: comm.disabled = {comm.disabled}"
                    )

                if hasattr(comm, "should_custom_ar"):
                    should_use = comm.should_custom_ar(inp)
                    print(
                        f"[{name}] {human_size(size_bytes)}: should_custom_ar = {should_use}"
                    )

                if hasattr(comm, "max_size"):
                    print(
                        f"[{name}] {human_size(size_bytes)}: comm.max_size = {human_size(comm.max_size)}"
                    )

            # Debug: Check if custom all-reduce was actually used
            if out is None and rank == 0 and verbose:
                print(
                    f"[{name}] {human_size(size_bytes)}: Custom AR disabled or fell back to NCCL"
                )
            elif (
                out is not None
                and hasattr(comm, "should_custom_ar")
                and rank == 0
                and verbose
            ):
                should_use = comm.should_custom_ar(inp)
                print(
                    f"[{name}] {human_size(size_bytes)}: Custom AR {'enabled' if should_use else 'disabled'} by should_custom_ar"
                )

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dist.barrier(group=pg)

            if out is None:
                disabled = True
                break

            dt_ms = (t1 - t0) * 1000.0
            times_ms.append(dt_ms)

            # Check correctness if enabled
            if correctness and original_inp is not None:
                is_correct = check_correctness_allreduce(
                    original_inp, out, world_size, rank, pg, name, size_bytes, dtype
                )
                correct_results.append(is_correct)
                if not is_correct and rank == 0:
                    print(
                        f"WARNING: Incorrect result in {name} at size {human_size(size_bytes)} iter {it}"
                    )

                    # Debug: Check what the implementation claims about this size
                    inp_size = inp.numel() * inp.element_size()
                    print(f"  Input size: {human_size(inp_size)}")
                    if hasattr(comm, "max_size"):
                        print(f"  comm.max_size: {human_size(comm.max_size)}")
                    if hasattr(comm, "should_custom_ar"):
                        should_use = comm.should_custom_ar(inp)
                        print(f"  should_custom_ar: {should_use}")
                    if hasattr(comm, "disabled"):
                        print(f"  comm.disabled: {comm.disabled}")

            if verbose and rank == 0:
                print(
                    f"[{name}] size={human_size(size_bytes)} iter={it} time={dt_ms:.3f} ms"
                )

        if disabled or not times_ms:
            if rank == 0:
                print(
                    f"[{name}] {human_size(size_bytes)}: custom AR disabled (no timings)"
                )
            results.append((size_bytes, None, None))
            continue

        avg_ms_local = sum(times_ms) / len(times_ms)
        avg_tensor = torch.tensor([avg_ms_local], dtype=torch.float64, device=device)
        gather_list = [torch.zeros_like(avg_tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, avg_tensor, group=pg)

        # Check correctness across all ranks
        all_correct = True
        if correctness:
            correct_tensor = torch.tensor(
                [all(correct_results) if correct_results else True],
                dtype=torch.bool,
                device=device,
            )
            correct_gather = [
                torch.zeros_like(correct_tensor) for _ in range(world_size)
            ]
            dist.all_gather(correct_gather, correct_tensor, group=pg)
            all_correct = all(t.item() for t in correct_gather)

        if rank == 0:
            avg_ms = float(torch.stack(gather_list).mean().item())
            correctness_str = "✓" if all_correct else "✗" if correctness else "-"
            print(
                f"[{name}] {human_size(size_bytes)}: {avg_ms:.3f} ms {correctness_str} (avg across ranks)"
            )
            results.append((size_bytes, avg_ms, all_correct if correctness else None))
        else:
            results.append((size_bytes, None, all_correct if correctness else None))

    return results


def main():
    args = parse_args()
    rank, world_size, local_rank = get_env_rank_world()

    if world_size not in (2, 4, 6, 8):
        print(
            f"[rank {rank}] WARNING: world_size={world_size} not in supported set (2,4,6,8). "
            "Custom AR may disable itself.",
            file=sys.stderr,
        )

    init_dist(args.backend)
    device = get_device(local_rank)

    # Import after dist init; some libs query torch dist state on import
    sgl_comm = None
    aiter_comm = None
    HAVE_SGLANG = False
    HAVE_AITER = False

    # Determine what implementation to test based on command line argument
    if args.implementation == "aiter":
        use_aiter_car = True
        if rank == 0:
            print("Testing Aiter implementation only")
    elif args.implementation == "sglang":
        use_aiter_car = False
        if rank == 0:
            print("Testing SGLang implementation only")
    else:  # both
        use_aiter_car = os.environ.get("USE_AITER_CAR", "true").lower() == "true"
        if rank == 0:
            print(f"Testing both implementations (USE_AITER_CAR={use_aiter_car})")

    # Set USE_AITER_CAR environment variable for internal use
    os.environ["USE_AITER_CAR"] = "1" if use_aiter_car else "0"

    # Get SGLang implementation (which will use dispatch)
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as SGLCustomAllreduce,
        )

        HAVE_SGLANG = True
    except Exception as e:
        if rank == 0:
            print(f"SGLang CustomAllreduce import failed: {e}", file=sys.stderr)

    # Get Aiter implementation
    try:
        # New aiter car
        # from aiter.dist.device_communicators.custom_all_reduce import (
        #     CustomAllreduce as AiterCustomAllreduce,
        # )
        from aiter.dist.custom_all_reduce import (
            CustomAllreduce as AiterCustomAllreduce,  # Old aiter car
        )

        HAVE_AITER = True
    except Exception as e:
        if rank == 0:
            print(f"Aiter CustomAllreduce import failed: {e}", file=sys.stderr)

    if rank == 0:
        print(f"Initialized PG backend={args.backend} world_size={world_size}")
        print(f"Device: {device.type}:{device.index}")
        print(f"SGLang available: {HAVE_SGLANG}, Aiter available: {HAVE_AITER}")
        print(f"Correctness checking: {args.correctness}")
        print(f"Graph mode: {args.graph_mode}")
        print(f"FP8 quant: {args.fp8_quant}")
        print(f"Use tensor model parallel: {args.use_tensor_model_parallel}")

    # Set FP8 environment variable if requested
    if args.fp8_quant:
        os.environ["USE_AITER_CAR_FP8"] = "1"
        if rank == 0:
            print("SET USE_AITER_CAR_FP8=1 for global FP8 quantization")

    # Use tensor model parallel group for custom all-reduce instead of world group
    pg = get_tensor_model_parallel_group().device_group
    sizes = get_message_sizes()
    max_size = max(sizes) if sizes else (64 * 1024 * 1024)

    # Initialize communicators
    if (
        args.implementation == "sglang" or args.implementation == "both"
    ) and HAVE_SGLANG:
        # Create SGLang communicator for sglang or both modes
        try:
            sgl_comm = SGLCustomAllreduce(group=pg, device=device, max_size=max_size)
        except Exception as e:
            if rank == 0:
                print(
                    f"Failed to construct SGLang CustomAllreduce: {e}", file=sys.stderr
                )
            sgl_comm = None

    if (args.implementation == "aiter" or args.implementation == "both") and HAVE_AITER:
        # Create Aiter communicator for aiter or both modes
        try:
            aiter_comm = AiterCustomAllreduce(
                group=pg, device=device, max_size=max_size
            )
        except Exception as e:
            if rank == 0:
                print(
                    f"Failed to construct Aiter CustomAllreduce: {e}", file=sys.stderr
                )
            aiter_comm = None

    sgl_results: List[Tuple[int, Optional[float], Optional[bool]]] = []
    aiter_results: List[Tuple[int, Optional[float], Optional[bool]]] = []

    # Test implementations
    if sgl_comm is not None:
        impl_name = "SGLang"
        sgl_results = bench_impl(
            name=impl_name,
            comm=sgl_comm,
            sizes=sizes,
            device=device,
            warmup=args.warmup,
            iters_small=args.iters_small,
            iters_large=args.iters_large,
            verbose=args.verbose,
            pg=pg,
            correctness=args.correctness,
            graph_mode=args.graph_mode,
            fp8_quant=args.fp8_quant,
            use_tensor_model_parallel=args.use_tensor_model_parallel,
            dtype=torch.bfloat16,  # Assuming bfloat16 is the default dtype for SGLang
        )

    if aiter_comm is not None:
        impl_name = "Aiter"
        aiter_results = bench_impl(
            name=impl_name,
            comm=aiter_comm,
            sizes=sizes,
            device=device,
            warmup=args.warmup,
            iters_small=args.iters_small,
            iters_large=args.iters_large,
            verbose=args.verbose,
            pg=pg,
            correctness=args.correctness,
            graph_mode=args.graph_mode,
            fp8_quant=args.fp8_quant,
            use_tensor_model_parallel=args.use_tensor_model_parallel,
            dtype=torch.bfloat16,  # Assuming bfloat16 is the default dtype for Aiter
        )

    # Cleanup
    for comm in (sgl_comm, aiter_comm):
        if comm is not None and hasattr(comm, "close"):
            try:
                comm.close()
            except Exception:
                pass

    if dist.get_rank() == 0:
        print("\nResults (avg ms across ranks; None = disabled/unavailable):")
        header = f"{'Size':>8}  {'SGLang(ms)':>12}  {'Aiter(ms)':>11}  {'Correct' if args.correctness else ''}"
        print(header)
        print("-" * len(header))

        sgl_map = {s: (v, c) for s, v, c in sgl_results if v is not None}
        aiter_map = {s: (v, c) for s, v, c in aiter_results if v is not None}

        for s in sizes:
            sgl_data = sgl_map.get(s, (None, None))
            aiter_data = aiter_map.get(s, (None, None))

            sgl_ms, sgl_correct = sgl_data
            aiter_ms, aiter_correct = aiter_data

            sgl_str = f"{sgl_ms:.3f}" if sgl_ms is not None else "None"
            aiter_str = f"{aiter_ms:.3f}" if aiter_ms is not None else "None"

            if args.correctness:
                correct_str = ""
                if sgl_correct is not None and aiter_correct is not None:
                    correct_str = f"{'✓' if sgl_correct and aiter_correct else '✗'}"
                elif sgl_correct is not None:
                    correct_str = f"{'✓' if sgl_correct else '✗'} (SGL)"
                elif aiter_correct is not None:
                    correct_str = f"{'✓' if aiter_correct else '✗'} (Aiter)"
                else:
                    correct_str = "-"

                print(
                    f"{human_size(s):>8}  {sgl_str:>12}  {aiter_str:>11}  {correct_str:>7}"
                )
            else:
                print(f"{human_size(s):>8}  {sgl_str:>12}  {aiter_str:>11}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
