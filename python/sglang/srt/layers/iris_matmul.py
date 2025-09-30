# SPDX-License-Identifier: Apache-2.0
# Integration layer for Iris GEMM + AllReduce kernels with SGLang
#
# This module provides integration between Iris fused GEMM+AllReduce kernels
# and SGLang's tensor parallel linear layers.
#
# Environment Variables:
#   SGL_USE_IRIS_MATMUL=1      : Enable iris matmul integration
#   SGL_IRIS_GEMM_AR_MODE=mode : Choose execution mode (persistent|oneshot, default: persistent)
#   SGL_IRIS_DEBUG=1           : Enable debug output from iris kernels

from __future__ import annotations
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional
import logging
import importlib.util

logger = logging.getLogger(__name__)


def _patch_iris_allocator():
    """
    Patch iris module to avoid CUDA allocator conflicts.
    
    This function monkey-patches torch.cuda.memory.change_current_allocator
    to prevent iris from changing the allocator when it's already initialized.
    """
    original_change_allocator = torch.cuda.memory.change_current_allocator
    
    def patched_change_allocator(allocator):
        """Skip allocator change if already initialized."""
        try:
            # Check if CUDA is already initialized
            if torch.cuda.is_initialized():
                logger.info(
                    "Skipping CUDA allocator change because PyTorch CUDA is already initialized. "
                    "Iris will use PyTorch's default allocator."
                )
                return
            # If not initialized, allow the change
            original_change_allocator(allocator)
        except RuntimeError as e:
            if "already initialized" in str(e).lower() or "allocator" in str(e).lower():
                logger.warning(
                    f"Could not change CUDA allocator (already initialized): {e}. "
                    "Continuing with PyTorch's default allocator."
                )
            else:
                raise
    
    # Apply the patch
    torch.cuda.memory.change_current_allocator = patched_change_allocator


def _import_iris_matmul():
    """
    Import iris matmul function with proper error handling.
    
    This function handles:
    1. CUDA allocator conflicts
    2. Numeric module names (09_gemm_one_shot_all_reduce)
    3. Missing dependencies
    """
    try:
        # Apply allocator patch before importing iris
        _patch_iris_allocator()
        
        # Add iris examples to Python path if needed
        # Try multiple possible locations for iris
        possible_iris_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'iris', 'examples'),
            '/sgl-workspace/iris/examples',
            os.path.expanduser('~/iris/examples'),
        ]
        
        iris_examples_path = None
        for path in possible_iris_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                iris_examples_path = abs_path
                break
        
        if iris_examples_path and iris_examples_path not in sys.path:
            sys.path.insert(0, iris_examples_path)
            logger.info(f"Added iris examples path: {iris_examples_path}")
        elif not iris_examples_path:
            logger.warning("Could not find iris examples directory")
        
        # Import iris matmul using importlib to handle numeric module names
        import importlib
        
        # First, make sure common utilities are available
        try:
            common_module = importlib.import_module('common.utils')
        except ImportError:
            try:
                common_module = importlib.import_module('examples.common.utils')
            except ImportError:
                # Create mock implementations for missing utilities
                import types
                common_module = types.ModuleType('common.utils')
                common_module.is_triton_interpret_set = lambda: False
                sys.modules['common.utils'] = common_module
                sys.modules['examples.common.utils'] = common_module
                logger.info("Created mock common.utils module")
        
        # Import the gemm kernel module using direct file import for numeric module names
        gemm_kernel_path = os.path.join(
            iris_examples_path or '/sgl-workspace/iris/examples',
            '09_gemm_one_shot_all_reduce',
            'gemm_one_shot_all_reduce.py'
        )
        
        if os.path.exists(gemm_kernel_path):
            spec = importlib.util.spec_from_file_location(
                'gemm_one_shot_all_reduce',
                gemm_kernel_path
            )
            if spec and spec.loader:
                gemm_module = importlib.util.module_from_spec(spec)
                sys.modules['gemm_one_shot_all_reduce'] = gemm_module
                spec.loader.exec_module(gemm_module)
                logger.info("Loaded gemm_one_shot_all_reduce module")
        
        # Import the matmul wrapper module
        matmul_wrapper_path = os.path.join(
            iris_examples_path or '/sgl-workspace/iris/examples',
            '09_gemm_one_shot_all_reduce',
            'matmul_wrapper.py'
        )
        
        if os.path.exists(matmul_wrapper_path):
            spec = importlib.util.spec_from_file_location(
                'matmul_wrapper',
                matmul_wrapper_path
            )
            if spec and spec.loader:
                matmul_module = importlib.util.module_from_spec(spec)
                sys.modules['matmul_wrapper'] = matmul_module
                spec.loader.exec_module(matmul_module)
                logger.info("Loaded matmul_wrapper module")
        else:
            raise ImportError(f"Could not find matmul_wrapper.py at {matmul_wrapper_path}")
        
        logger.info("Successfully imported iris matmul module")
        return matmul_module.matmul
        
    except Exception as e:
        logger.warning(f"Failed to import iris matmul: {e}")
        return None


# Lazy import - only import when actually needed
_IRIS_MATMUL_FN = None

# Global iris shared memory instance (set by ModelRunner)
_IRIS_SHMEM = None


def set_iris_shmem(shmem):
    """Set the global iris shared memory instance."""
    global _IRIS_SHMEM
    _IRIS_SHMEM = shmem


def get_iris_shmem():
    """Get the global iris shared memory instance."""
    return _IRIS_SHMEM


class IrisMatMulLayer:
    """
    Integration layer for Iris fused GEMM + AllReduce operations.
    
    This class provides a clean interface to iris's fused GEMM + AllReduce
    functionality for SGLang's RowParallelLinear layers. It handles:
    - Lazy importing to avoid allocator conflicts
    - Automatic fallback to standard PyTorch when iris is unavailable
    - Debug output and error handling
    - Support for both persistent and oneshot execution modes
    
    Args:
        tp_rank: Tensor parallel rank
        tp_size: Tensor parallel world size
        dtype: Data type for computations
        device: Device for computations
        name: Name for debugging
    
    Environment Variables:
        SGL_USE_IRIS_MATMUL: Set to 1 to enable iris matmul
        SGL_IRIS_GEMM_AR_MODE: "persistent" or "oneshot" (default: persistent)
        SGL_IRIS_DEBUG: Set to 1 to enable debug output
    """
    
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
        name: str = "iris_matmul",
        iris_shmem: Optional[any] = None,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dtype = dtype
        self.device = device
        self.name = name
        self._iris_matmul_fn = None
        self._use_iris = False
        self.iris_shmem = iris_shmem
        
        # Check if iris shared memory is available for TP>1
        if self.tp_size > 1 and iris_shmem is None:
            logger.warning(
                f"[{name}] Iris matmul disabled for TP={tp_size}: iris shared memory not initialized. "
                "Set SGL_USE_IRIS_MATMUL=1 to enable. Falling back to PyTorch."
            )
            self._use_iris = False
            return
        
        # Get configuration from environment
        self.mode = os.getenv("SGL_IRIS_GEMM_AR_MODE", "persistent").lower()
        if self.mode not in ("persistent", "oneshot"):
            logger.warning(f"Invalid iris mode '{self.mode}', using 'persistent'")
            self.mode = "persistent"
        
        self.debug = os.getenv("SGL_IRIS_DEBUG", "0") == "1"
        
    def _get_iris_matmul_fn(self):
        """Lazy import of iris matmul function."""
        if self._iris_matmul_fn is None:
            global _IRIS_MATMUL_FN
            if _IRIS_MATMUL_FN is None:
                _IRIS_MATMUL_FN = _import_iris_matmul()
            self._iris_matmul_fn = _IRIS_MATMUL_FN
            
            if self._iris_matmul_fn is None:
                logger.info(
                    f"[{self.name}] Iris matmul not available, using PyTorch fallback"
                )
                self._use_iris = False
            else:
                logger.info(f"[{self.name}] Iris matmul enabled in {self.mode} mode")
                self._use_iris = True
                
        return self._iris_matmul_fn
    
    def _fallback_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fallback implementation using standard PyTorch operations.
        
        This performs: output = x @ weight.T + bias
        followed by all-reduce across tensor parallel ranks.
        """
        # Standard matrix multiplication
        output = torch.matmul(x, weight.t())
        
        # All-reduce across tensor parallel group
        if self.tp_size > 1 and dist.is_initialized():
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            output = tensor_model_parallel_all_reduce(output)
        
        # Add bias if provided
        if bias is not None:
            output = output + bias
        
        return output
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with fused GEMM + AllReduce.
        
        Args:
            x: Input tensor of shape [M, K] (shard of input)
            weight: Weight tensor of shape [N, K] (row-parallel, each rank has full N rows)
            bias: Optional bias tensor of shape [N]
        
        Returns:
            Output tensor of shape [M, N] (fully reduced across ranks)
        """
        # Get iris matmul function (lazy import)
        iris_fn = self._get_iris_matmul_fn()
        
        # Fall back to PyTorch if iris is not available
        if not self._use_iris or iris_fn is None:
            return self._fallback_matmul(x, weight, bias)
        
        # Validate inputs
        assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
        # RowParallelLinear weight is [N, K], input is [M, K]
        # After transpose, weight becomes [K, N] for iris kernel
        assert x.shape[-1] == weight.shape[-1], (
            f"Input K dimension {x.shape[-1]} must match weight K dimension {weight.shape[-1]}"
        )
        
        try:
            M, K = x.shape
            N = weight.shape[0]
            
            # Iris expects weight in shape [K, N], but RowParallelLinear has [N, K]
            # We need to transpose the weight
            weight_transposed = weight.t().contiguous()  # [N, K] -> [K, N]
            
            # Create output tensors
            c = torch.empty((M, N), dtype=self.dtype, device=self.device)
            c_global = torch.empty((M, N), dtype=self.dtype, device=self.device)
            
            # Create synchronization tensors needed by iris for inter-GPU coordination
            # For TP>1, we need actual tensor allocations
            import triton
            total_blocks_M_prelim = triton.cdiv(M, 128)  # Using default BLK_M
            total_blocks_N_prelim = triton.cdiv(N, 128)  # Using default BLK_N
            total_tiles_prelim = total_blocks_M_prelim * total_blocks_N_prelim
            
            if self.tp_size > 1 and self.iris_shmem is not None:
                # Allocate in iris shared memory for multi-GPU
                tile_completed = self.iris_shmem.zeros((total_tiles_prelim,), device="cuda", dtype=torch.int32)
                locks = self.iris_shmem.zeros((304,), device="cuda", dtype=torch.int32)  # cu_count
                P = self.iris_shmem.zeros((total_tiles_prelim,), device="cuda", dtype=torch.int32)
                heap_bases_ptr = self.iris_shmem.get_heap_bases()
            else:
                # Single GPU - use empty tensors
                P = torch.empty(0, device=self.device)
                locks = torch.empty(0, device=self.device)
                tile_completed = torch.empty(0, device=self.device, dtype=torch.int32)
                heap_bases_ptr = torch.empty(0, device=self.device)
            
            # Kernel configuration
            # These parameters are tuned for performance
            BLK_M = 128
            BLK_N = 128
            BLK_K = 32
            gsize_m = 1
            two_tiles = True
            num_stages = 1
            num_warps = 8
            waves_per_eu = 0
            mfmaInstrSize = 16
            kpack = 2
            cu_count = 304  # For MI300X
            
            # Compute grid size
            import triton
            total_blocks_M = triton.cdiv(M, BLK_M)
            total_blocks_N = triton.cdiv(N, BLK_N)
            total_tiles = total_blocks_M * total_blocks_N
            grid = min(total_tiles, cu_count)
            
            if self.debug:
                logger.info(
                    f"[{self.name}] Calling iris matmul: "
                    f"x.shape={x.shape}, weight.shape={weight.shape}, "
                    f"weight_transposed.shape={weight_transposed.shape}, "
                    f"M={M}, N={N}, K={K}, grid={grid}, "
                    f"tp_rank={self.tp_rank}, tp_size={self.tp_size}"
                )
            
            # Set debug mode for iris
            if self.debug:
                iris_fn.set_debug(True)
            else:
                iris_fn.set_debug(False)
            
            # Call iris fused GEMM + AllReduce kernel
            result = iris_fn.apply(
                x,                      # a: input tensor [M, K]
                weight_transposed,      # b: weight tensor [K, N] (transposed from [N, K])
                c,                      # c: local output buffer [M, N]
                c_global,               # c_global: global output buffer [M, N]
                bias if bias is not None else torch.empty(0, device=self.device),
                P,                      # P: synchronization tensor
                locks,                  # locks: synchronization locks
                tile_completed,         # tile_completed: completion flags
                self.tp_rank,           # rank: current rank
                self.tp_size,           # world_size: total ranks
                grid,                   # grid: number of thread blocks
                BLK_M,                  # BLK_M: M dimension block size
                BLK_N,                  # BLK_N: N dimension block size
                BLK_K,                  # BLK_K: K dimension block size
                gsize_m,                # gsize_m: group size for M dimension
                two_tiles,              # two_tiles: use two-tile algorithm
                num_stages,             # num_stages: pipeline stages
                num_warps,              # num_warps: warps per thread block
                waves_per_eu,           # waves_per_eu: waves per execution unit
                mfmaInstrSize,          # mfmaInstrSize: matrix instruction size
                kpack,                  # kpack: K dimension packing factor
                heap_bases_ptr,         # heap_bases_ptr: shared memory heap bases
                cu_count,               # cu_count: compute unit count
                False,                  # COLLECT_TIMESTAMPS: disable timing
                None,                   # mm_begin_timestamp: begin timestamp
                None,                   # mm_end_timestamp: end timestamp
            )
            
            if self.debug:
                logger.info(f"[{self.name}] Iris matmul completed successfully")
            
            return result
            
        except Exception as e:
            logger.warning(
                f"[{self.name}] Iris matmul failed: {e}. Falling back to PyTorch."
            )
            # Fall back to standard PyTorch implementation
            return self._fallback_matmul(x, weight, bias)
    
    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Alias for forward method for compatibility."""
        return self.forward(x, weight, bias)
    
    def finalize(self):
        """Cleanup method for symmetry with other implementations."""
        pass


# Helper function to check if iris matmul should be used
def should_use_iris_matmul() -> bool:
    """
    Check if iris matmul should be used based on environment variables.
    
    Returns:
        True if SGL_USE_IRIS_MATMUL=1 is set, False otherwise
    """
    return os.getenv("SGL_USE_IRIS_MATMUL", "0") == "1"

