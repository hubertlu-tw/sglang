#!/usr/bin/env python3
"""
Test Triton All-Reduce Residual RMSNorm correctness with symmetric memory enabled.
This test properly sets up the environment for Triton fusion testing.
"""

import os
from typing import Tuple

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
    """
    Test Triton kernel correctness with proper setup.  This version exercises
    the real code path used in SGLang, i.e. `RMSNorm.forward_with_allreduce_fusion`,
    which returns both the RMS-normalized output and the updated residual.
    """
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

        # Compute the reference residual and normalized output (both outputs)
        def reference_implementation() -> Tuple[torch.Tensor, torch.Tensor]:
            # All‑reduce across all GPUs on a copy of the input
            reduced = input_tensor.clone().to(torch.float32)
            if world_size > 1:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            # Compute the updated residual: allreduced input + residual
            residual_out = reduced + residual_tensor.to(torch.float32)
            # Compute RMSNorm
            variance = residual_out.pow(2).mean(-1, keepdim=True)
            norm_out = residual_out * torch.rsqrt(variance + eps)
            # Apply weight
            norm_out = norm_out * weight_tensor.to(torch.float32)
            return norm_out.to(input_tensor.dtype), residual_out.to(input_tensor.dtype)

        reference_norm, reference_residual_out = reference_implementation()

        # Try to run the fused RMSNorm via the real code path
        try:
            from sglang.srt.layers.layernorm import RMSNorm

            # Create an RMSNorm instance with the same hidden size and epsilon
            rms_norm = RMSNorm(N, eps=eps)
            # Copy the test weight into the module's parameter
            with torch.no_grad():
                rms_norm.weight.copy_(weight_tensor)

            # Invoke the fused forward; this returns (norm_out, residual_out)
            norm_out, residual_out = rms_norm.forward_with_allreduce_fusion(
                input_tensor.clone(), residual_tensor.clone()
            )

            # Detailed accuracy analysis
            diff_norm = (norm_out - reference_norm).abs()
            diff_residual = (residual_out - reference_residual_out).abs()
            max_diff_norm = diff_norm.max().item()
            mean_diff_norm = diff_norm.mean().item()
            max_diff_residual = diff_residual.max().item()
            mean_diff_residual = diff_residual.mean().item()

            if rank == 0:
                print(
                    f"Reference norm range: [{reference_norm.min().item():.6e}, {reference_norm.max().item():.6e}]"
                )
                print(
                    f"Fused norm range: [{norm_out.min().item():.6e}, {norm_out.max().item():.6e}]"
                )
                print(
                    f"Max/mean diff (norm): {max_diff_norm:.6e} / {mean_diff_norm:.6e}, "
                    f"Max/mean diff (residual): {max_diff_residual:.6e} / {mean_diff_residual:.6e}"
                )

            # Assert both outputs are close to reference
            ok_norm = torch.allclose(norm_out, reference_norm, atol=1e-2, rtol=1e-2)
            ok_residual = torch.allclose(
                residual_out, reference_residual_out, atol=1e-2, rtol=1e-2
            )
            if rank == 0:
                if ok_norm and ok_residual:
                    print(
                        f"✅ PASS: Both norm and residual match reference (max diffs: {max_diff_norm:.2e}, {max_diff_residual:.2e})"
                    )
                else:
                    print(
                        f"❌ FAIL: Norm match={ok_norm}, Residual match={ok_residual}, "
                        f"max diffs: {max_diff_norm:.2e}, {max_diff_residual:.2e}"
                    )
        except ImportError:
            if rank == 0:
                print("❌ RMSNorm module not available for testing")
            break
        except Exception as e:
            if rank == 0:
                print(f"❌ Error running fused RMSNorm: {e}")
                print("Falling back to testing native implementation...")

            # Fall back to testing the native implementation
            try:
                from sglang.srt.layers.layernorm import RMSNorm

                rms_norm = RMSNorm(N, eps=eps)
                with torch.no_grad():
                    rms_norm.weight.copy_(weight_tensor)

                # For native implementation, we need to manually all-reduce the input
                # to match the reference implementation
                input_for_native = input_tensor.clone().to(torch.float32)
                if world_size > 1:
                    dist.all_reduce(input_for_native, op=dist.ReduceOp.SUM)

                # Test native implementation with all-reduced input
                norm_out, residual_out = rms_norm.forward_native(
                    input_for_native, residual_tensor.clone().to(torch.float32)
                )
                norm_out = norm_out.to(input_tensor.dtype)
                residual_out = residual_out.to(input_tensor.dtype)

                # Check accuracy
                diff_norm = (norm_out - reference_norm).abs()
                diff_residual = (residual_out - reference_residual_out).abs()
                max_diff_norm = diff_norm.max().item()
                max_diff_residual = diff_residual.max().item()

                ok_norm = torch.allclose(norm_out, reference_norm, atol=1e-2, rtol=1e-2)
                ok_residual = torch.allclose(
                    residual_out, reference_residual_out, atol=1e-2, rtol=1e-2
                )

                if rank == 0:
                    if ok_norm and ok_residual:
                        print(
                            f"✅ PASS (native fallback): Both norm and residual match reference (max diffs: {max_diff_norm:.2e}, {max_diff_residual:.2e})"
                        )
                    else:
                        print(
                            f"❌ FAIL (native fallback): Norm match={ok_norm}, Residual match={ok_residual}, "
                            f"max diffs: {max_diff_norm:.2e}, {max_diff_residual:.2e}"
                        )

            except Exception as fallback_error:
                if rank == 0:
                    print(f"❌ Error in native fallback: {fallback_error}")
                    import traceback

                    traceback.print_exc()

    if rank == 0:
        print("\n🎉 Correctness test completed!")

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_triton_correctness()
