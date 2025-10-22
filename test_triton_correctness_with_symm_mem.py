#!/usr/bin/env python3
"""
Test Triton All-Reduce Residual RMSNorm correctness with symmetric memory enabled.
This test properly sets up the environment for Triton fusion testing.
"""

import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment for multi-GPU testing."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def initialize_model_parallel_environment(world_size: int):
    """Initialize model parallel environment for SGLang."""
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Initialize distributed environment
    init_distributed_environment()

    # Initialize model parallel
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    print(f"✓ Model parallel environment initialized with TP={world_size}")


def enable_triton_fusion():
    """Enable Triton fusion by setting the required global flags."""
    from sglang.srt.managers.schedule_batch import global_server_args_dict

    # Enable Triton fusion
    global_server_args_dict["enable_torch_symm_mem"] = True
    global_server_args_dict["enable_triton_allreduce_fusion"] = True

    print(
        f"✓ Enabled Triton fusion: enable_torch_symm_mem={global_server_args_dict['enable_torch_symm_mem']}"
    )
    print(
        f"✓ Enabled Triton fusion: enable_triton_allreduce_fusion={global_server_args_dict['enable_triton_allreduce_fusion']}"
    )


def initialize_symmetric_memory():
    """Initialize symmetric memory buffer for Triton fusion."""
    from sglang.srt.layers.triton_comm_manager import initialize_triton_symm_mem_buffer

    # Initialize symmetric memory buffer
    buffer = initialize_triton_symm_mem_buffer(
        max_tokens=2048, hidden_dim=4096, dtype=torch.bfloat16
    )

    if buffer is not None:
        print(
            f"✓ Symmetric memory buffer initialized: shape={buffer.shape}, dtype={buffer.dtype}"
        )
    else:
        print("❌ Failed to initialize symmetric memory buffer")

    return buffer


def test_triton_correctness():
    """Test Triton kernel correctness with proper setup."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(
            "🧪 Testing Triton All-Reduce Residual RMSNorm Correctness with Symmetric Memory"
        )
        print("=" * 80)

    # Initialize model parallel environment
    if world_size > 1:
        initialize_model_parallel_environment(world_size)

    # Enable Triton fusion
    enable_triton_fusion()

    # Initialize symmetric memory
    buffer = initialize_symmetric_memory()

    if buffer is None:
        if rank == 0:
            print("❌ Cannot test Triton fusion - symmetric memory not available")
        return

    # Test configurations
    test_cases = [
        (128, 4096, torch.bfloat16, "Small"),
        (512, 4096, torch.bfloat16, "Medium"),
        (2048, 4096, torch.bfloat16, "Large"),
    ]

    eps = 1e-5

    for M, N, dtype, size_name in test_cases:
        if rank == 0:
            print(f"\n📊 Testing {size_name} Size: M={M}, N={N}, {dtype}")
            print("-" * 40)

        # Generate test data
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

        input_tensor = torch.randn((M, N), device=f"cuda:{local_rank}", dtype=dtype)
        residual_tensor = torch.randn((M, N), device=f"cuda:{local_rank}", dtype=dtype)
        weight_tensor = torch.randn(N, device=f"cuda:{local_rank}", dtype=dtype)

        # Log tensor properties for debugging
        if rank == 0:
            print(f"\n📊 Tensor properties for {size_name} test:")
            print(
                f"Input shape: {input_tensor.shape}, contiguous: {input_tensor.is_contiguous()}, dtype: {input_tensor.dtype}"
            )
            print(
                f"Residual shape: {residual_tensor.shape}, contiguous: {residual_tensor.is_contiguous()}, dtype: {residual_tensor.dtype}"
            )
            print(
                f"Weight shape: {weight_tensor.shape}, contiguous: {weight_tensor.is_contiguous()}, dtype: {weight_tensor.dtype}"
            )
            print(f"Input strides: {input_tensor.stride()}")
            print(f"Residual strides: {residual_tensor.stride()}")
            print(f"Weight strides: {weight_tensor.stride()}")
            print(f"Input sample: {input_tensor.flatten()[:5].tolist()}")
            print(f"Residual sample: {residual_tensor.flatten()[:5].tolist()}")
            print(f"Weight sample: {weight_tensor.flatten()[:5].tolist()}")

        # Run reference implementation
        def reference_implementation():
            # All-Reduce across all GPUs
            reduced_tensor = input_tensor.clone().to(torch.float32)
            if world_size > 1:
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

            # Add residual
            sum_tensor = reduced_tensor + residual_tensor.to(torch.float32)

            # RMS Normalization
            variance = sum_tensor.pow(2).mean(-1, keepdim=True)
            hidden_states_norm = sum_tensor * torch.rsqrt(variance + eps)

            # Apply weight
            output = hidden_states_norm * weight_tensor.to(torch.float32)

            return output.to(input_tensor.dtype)

        reference_output = reference_implementation()

        # Try to run Triton implementation
        try:
            from sglang.srt.layers.triton_comm_fusion import (
                triton_allreduce_residual_rmsnorm_wrapper,
            )

            triton_output = triton_allreduce_residual_rmsnorm_wrapper(
                input_tensor=input_tensor,
                residual=residual_tensor,
                weight=weight_tensor,
                eps=eps,
                max_token_num=M,
            )

            if triton_output is None:
                if rank == 0:
                    print("❌ Triton fusion not available (wrapper returned None)")
                continue

            # Calculate differences
            diff = (triton_output - reference_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Detailed accuracy analysis
            if rank == 0:
                print(
                    f"Reference output range: [{reference_output.min().item():.6e}, {reference_output.max().item():.6e}]"
                )
                print(
                    f"Triton output range: [{triton_output.min().item():.6e}, {triton_output.max().item():.6e}]"
                )
                print(f"Max absolute difference: {max_diff:.6e}")
                print(f"Mean absolute difference: {mean_diff:.6e}")
                print(
                    f"Relative difference (max/range): {max_diff / (reference_output.max().item() - reference_output.min().item()):.6e}"
                )

                # Check for NaN or Inf values
                if torch.isnan(triton_output).any():
                    print("❌ Triton output contains NaN values!")
                if torch.isinf(triton_output).any():
                    print("❌ Triton output contains Inf values!")

                # Check element-wise closeness
                close_mask = torch.isclose(
                    triton_output, reference_output, atol=1e-2, rtol=1e-2
                )
                close_percentage = close_mask.sum().item() / close_mask.numel() * 100
                print(
                    f"Percentage of close elements (atol=1e-2, rtol=1e-2): {close_percentage:.2f}%"
                )

            # Check correctness
            is_close = torch.allclose(
                triton_output, reference_output, atol=1e-2, rtol=1e-2
            )

            if rank == 0:
                if is_close:
                    print(
                        f"✅ PASS: Max diff={max_diff:.2e}, Mean diff={mean_diff:.2e}"
                    )
                else:
                    print(
                        f"❌ FAIL: Max diff={max_diff:.2e}, Mean diff={mean_diff:.2e}"
                    )

        except ImportError:
            if rank == 0:
                print("❌ Triton comm fusion module not available")
            break
        except Exception as e:
            if rank == 0:
                print(f"❌ Error running Triton kernel: {e}")
                import traceback

                traceback.print_exc()

    if rank == 0:
        print("\n🎉 Correctness test completed!")

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_triton_correctness()
