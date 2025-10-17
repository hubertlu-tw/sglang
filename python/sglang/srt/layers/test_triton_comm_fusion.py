import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from triton_comm_fusion import triton_allreduce_residual_rmsnorm


def setup_distributed():
    """Initializes the distributed environment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    # Create a process group with the specific name for symmetric memory
    group_name = "triton_fusion_test_group"
    ranks = list(range(world_size))
    process_group = dist.new_group(ranks, backend="nccl")

    return rank, world_size, local_rank, process_group


def benchmark_kernel(
    func,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    eps: float,
    symm_mem_buffer: torch.Tensor = None,
    process_group=None,
    use_delayed_kernel: bool = False,
    warmup_iters: int = 10,
    test_iters: int = 100,
) -> Tuple[torch.Tensor, float, float]:
    """Benchmark a kernel function and return output, mean time, and std time."""

    # Warmup
    for _ in range(warmup_iters):
        if symm_mem_buffer is not None:
            output = func(
                input_tensor,
                residual_tensor,
                weight_tensor,
                eps,
                symm_mem_buffer,
                process_group,
                use_delayed_kernel,
            )
        else:
            output = func(input_tensor, residual_tensor, weight_tensor, eps)
        torch.cuda.synchronize()

    # Get output for correctness checking
    if symm_mem_buffer is not None:
        output = func(
            input_tensor,
            residual_tensor,
            weight_tensor,
            eps,
            symm_mem_buffer,
            process_group,
            use_delayed_kernel,
        )
    else:
        output = func(input_tensor, residual_tensor, weight_tensor, eps)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(test_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if symm_mem_buffer is not None:
            _ = func(
                input_tensor,
                residual_tensor,
                weight_tensor,
                eps,
                symm_mem_buffer,
                process_group,
                use_delayed_kernel,
            )
        else:
            _ = func(input_tensor, residual_tensor, weight_tensor, eps)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return output, mean_time, std_time


def reference_implementation(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """A standard PyTorch implementation to serve as a ground truth."""
    reduced_tensor = input_tensor.clone().to(torch.float32)
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    sum_tensor = reduced_tensor + residual_tensor.to(torch.float32)
    variance = sum_tensor.pow(2).mean(-1, keepdim=True)
    hidden_states_norm = sum_tensor * torch.rsqrt(variance + eps)
    output = hidden_states_norm * weight_tensor.to(torch.float32)
    return output.to(input_tensor.dtype)


def run_test(rank: int, world_size: int, local_rank: int, process_group):
    if rank == 0:
        print(f"--- Running Test on {world_size} GPUs ---")

    # Test different problem sizes
    test_configs = [
        (128, 4096, "Small"),
        (512, 4096, "Medium"),
        (2048, 4096, "Large"),
    ]

    dtype = torch.bfloat16
    eps = 1e-5
    device = f"cuda:{local_rank}"

    for M, N, size_name in test_configs:
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

        input_tensor = torch.randn((M, N), device=device, dtype=dtype)
        residual_tensor = torch.randn((M, N), device=device, dtype=dtype)
        weight_tensor = torch.randn(N, device=device, dtype=dtype)
        symm_mem_buffer = symm_mem.empty(
            input_tensor.numel(), device=device, dtype=dtype
        )

        dist.barrier()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing {size_name} Size: M={M}, N={N}")
            print(f"{'='*60}")

        # Benchmark correct Triton kernel
        correct_output, correct_time, correct_std = benchmark_kernel(
            triton_allreduce_residual_rmsnorm,
            input_tensor,
            residual_tensor,
            weight_tensor,
            eps,
            symm_mem_buffer,
            process_group,
            False,
            warmup_iters=10,
            test_iters=50,
        )

        # Benchmark reference implementation
        reference_output, reference_time, reference_std = benchmark_kernel(
            reference_implementation,
            input_tensor.clone(),
            residual_tensor,
            weight_tensor,
            eps,
            warmup_iters=10,
            test_iters=50,
        )

        dist.barrier()

        if rank == 0:
            # Verification
            is_close = torch.allclose(
                correct_output, reference_output, atol=1e-2, rtol=1e-2
            )
            if is_close:
                print("✅ Correctness: PASSED")
            else:
                max_diff = (correct_output - reference_output).abs().max()
                print(f"❌ Correctness: FAILED (max diff: {max_diff.item():.4f})")

            # Performance Results
            print(f"\nPerformance:")
            print(f"  Optimized Triton: {correct_time:.3f} ± {correct_std:.3f} ms")
            print(f"  PyTorch Reference: {reference_time:.3f} ± {reference_std:.3f} ms")

            speedup = reference_time / correct_time
            print(f"  Speedup: {speedup:.2f}x")

            # Calculate bandwidth
            data_size_bytes = (
                input_tensor.numel()
                + residual_tensor.numel()
                + weight_tensor.numel()
                + correct_output.numel()
            ) * 2  # bfloat16 = 2 bytes
            triton_bandwidth = (data_size_bytes / 1e9) / (correct_time / 1000)  # GB/s
            reference_bandwidth = (data_size_bytes / 1e9) / (
                reference_time / 1000
            )  # GB/s

            print(f"\nMemory Bandwidth:")
            print(f"  Optimized Triton: {triton_bandwidth:.1f} GB/s")
            print(f"  PyTorch Reference: {reference_bandwidth:.1f} GB/s")

            # Calculate FLOPS (approximate)
            total_flops = M * N * (world_size + 1 + 5)
            triton_tflops = (total_flops / 1e12) / (correct_time / 1000)
            reference_tflops = (total_flops / 1e12) / (reference_time / 1000)

            print(f"\nCompute Throughput:")
            print(f"  Optimized Triton: {triton_tflops:.2f} TFLOPS")
            print(f"  PyTorch Reference: {reference_tflops:.2f} TFLOPS")

        dist.barrier()

    # Test flawed kernel once at the end for demonstration
    if rank == 0:
        print(f"\n{'='*60}")
        print("Testing Race Condition Demonstration")
        print(f"{'='*60}")

    M, N = 128, 4096  # Use small size for demo
    input_tensor = torch.randn((M, N), device=device, dtype=dtype)
    residual_tensor = torch.randn((M, N), device=device, dtype=dtype)
    weight_tensor = torch.randn(N, device=device, dtype=dtype)
    symm_mem_buffer = symm_mem.empty(input_tensor.numel(), device=device, dtype=dtype)

    flawed_output = triton_allreduce_residual_rmsnorm(
        input_tensor,
        residual_tensor,
        weight_tensor,
        eps,
        symm_mem_buffer,
        process_group,
        use_delayed_kernel=True,
    )
    reference_output = reference_implementation(
        input_tensor, residual_tensor, weight_tensor, eps
    )

    if rank == 0:
        is_close_flawed = torch.allclose(
            flawed_output, reference_output, atol=1e-2, rtol=1e-2
        )
        if not is_close_flawed:
            max_diff = (flawed_output - reference_output).abs().max()
            print(f"✅ Race condition demonstrated: max diff = {max_diff.item():.3f}")
        else:
            print("⚠️  Race condition not visible due to timing coincidence")

    dist.barrier()


if __name__ == "__main__":
    rank, world_size, local_rank, process_group = setup_distributed()
    try:
        run_test(rank, world_size, local_rank, process_group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
