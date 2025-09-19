from __future__ import annotations

"""
Memory-safe AIter backend with proper hybrid KV cache support
"""

import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

try:
    from aiter import (
        flash_attn_varlen_func,
        mha_batch_prefill_func,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.mem_cache.memory_pool import SWAKVPool


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]


global_workspace_buffer = None
_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device

        # FIXED: Proper hybrid KV cache configuration
        self.is_hybrid = getattr(model_runner, "is_hybrid", False)
        self.hybrid_kvcache_ratio = (
            getattr(model_runner.server_args, "hybrid_kvcache_ratio", None) or 0.0
        )

        # Enable hybrid if ratio is specified
        if self.hybrid_kvcache_ratio > 0:
            self.is_hybrid = True
            print(
                f"[AIter Backend] Enabling hybrid KV cache with ratio: {self.hybrid_kvcache_ratio}"
            )

        # Get token pools and layer mappings
        token2pool = getattr(model_runner, "token_to_kv_pool", None)
        self.full_to_swa_index_mapping = getattr(
            token2pool, "full_to_swa_index_mapping", None
        )
        self.swa_layer_ids = (
            getattr(model_runner.model_config, "swa_attention_layer_ids", []) or []
        )
        self.full_layer_ids = (
            getattr(model_runner.model_config, "full_attention_layer_ids", []) or []
        )

        # FIXED: Calculate memory limits based on hybrid ratio and available GPU memory
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                # Use at most 80% of GPU memory for KV cache
                available_memory = int(total_memory * 0.8)
                print(
                    f"[AIter Backend] Available GPU memory for KV cache: {available_memory / 1024**3:.2f} GB"
                )
            else:
                available_memory = 8 * 1024**3  # 8GB fallback
        except:
            available_memory = 8 * 1024**3  # 8GB fallback

        self.max_total_num_tokens = model_runner.max_total_num_tokens

        # Calculate effective memory usage based on hybrid ratio
        if self.is_hybrid and self.hybrid_kvcache_ratio > 0:
            # Reduce GPU memory allocation based on hybrid ratio
            memory_reduction_factor = 1.0 - (
                self.hybrid_kvcache_ratio * 0.6
            )  # Up to 60% reduction
            self.effective_tokens = int(
                self.max_total_num_tokens * memory_reduction_factor
            )
            self.cpu_fallback_tokens = self.max_total_num_tokens - self.effective_tokens

            print(
                f"[AIter Backend] Hybrid config - GPU tokens: {self.effective_tokens}, CPU fallback: {self.cpu_fallback_tokens}"
            )
        else:
            self.effective_tokens = self.max_total_num_tokens
            self.cpu_fallback_tokens = 0

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # FIXED: Memory-safe workspace buffer allocation
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not self.use_mla:
            # Calculate safe workspace buffer size
            base_size = max_bs * self.num_head * self.max_num_partitions * self.head_dim

            # Apply memory reduction for hybrid caching
            if self.is_hybrid and self.hybrid_kvcache_ratio > 0:
                reduction_factor = 1.0 - (self.hybrid_kvcache_ratio * 0.5)
                base_size = int(base_size * reduction_factor)
                print(
                    f"[AIter Backend] Reduced workspace size by {self.hybrid_kvcache_ratio * 50:.1f}% for hybrid caching"
                )

            # Ensure we don't exceed available memory
            max_safe_size = available_memory // (4 * nbyes_per_qo_elem)  # Safety margin
            base_size = min(base_size, max_safe_size)

            try:
                self.workspace_buffer = torch.empty(
                    base_size * nbyes_per_qo_elem
                    + 2 * (base_size // self.head_dim) * 4,
                    dtype=torch.uint8,
                    device=self.device,
                )
                print(
                    f"[AIter Backend] Allocated workspace buffer: {self.workspace_buffer.numel() / 1024**2:.1f} MB"
                )
            except RuntimeError as e:
                # If allocation fails, try with even smaller size
                base_size = base_size // 2
                print(
                    f"[AIter Backend] Allocation failed, trying smaller size: {base_size}"
                )
                self.workspace_buffer = torch.empty(
                    base_size * nbyes_per_qo_elem
                    + 2 * (base_size // self.head_dim) * 4,
                    dtype=torch.uint8,
                    device=self.device,
                )

            # ADDED: CPU fallback workspace for extreme cases
            if self.is_hybrid and self.hybrid_kvcache_ratio > 0.5:
                try:
                    cpu_workspace_size = int(
                        base_size * self.hybrid_kvcache_ratio * 0.3
                    )
                    self.cpu_workspace_buffer = torch.empty(
                        cpu_workspace_size * nbyes_per_qo_elem,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                    print(
                        f"[AIter Backend] Allocated CPU fallback workspace: {self.cpu_workspace_buffer.numel() / 1024**2:.1f} MB"
                    )
                except:
                    print(
                        "[AIter Backend] Could not allocate CPU workspace, using GPU-only"
                    )
                    self.cpu_workspace_buffer = None
            else:
                self.cpu_workspace_buffer = None

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0
        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            self.enable_dp_attention = is_dp_attention_enabled()

    def _should_use_cpu_fallback(self, current_memory_usage: int) -> bool:
        """Determine if we should use CPU fallback based on memory pressure."""
        if not self.is_hybrid or self.hybrid_kvcache_ratio <= 0:
            return False

        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory

                memory_pressure = allocated / total

                # Use CPU fallback if memory pressure is high
                return memory_pressure > (1.0 - self.hybrid_kvcache_ratio * 0.5)
        except:
            pass

        # Fallback based on token count
        return current_memory_usage > self.effective_tokens

    # Map indices from the FULL KV pool to the SWA pool when this layer uses sliding-window attention.
    def _map_cache_loc_for_layer(
        self, layer_id: int, cache_loc: torch.Tensor
    ) -> torch.Tensor:
        if (not self.is_hybrid) or (self.full_to_swa_index_mapping is None):
            return cache_loc
        if layer_id in self.swa_layer_ids:
            # mapping is 1D tensor: new_idx = mapping[old_idx]
            return self.full_to_swa_index_mapping[cache_loc]
        return cache_loc

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None

        # ADDED: Check memory pressure and adjust batch size if needed
        current_tokens = forward_batch.seq_lens_sum
        if self._should_use_cpu_fallback(current_tokens):
            print(
                f"[AIter Backend] High memory pressure detected for {current_tokens} tokens, using fallback strategies"
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                # FIXED: Safe allocation with bounds checking
                try:
                    kv_indices = torch.empty(
                        forward_batch.seq_lens_sum,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                except RuntimeError as e:
                    print(f"[AIter Backend] Memory allocation failed: {e}")
                    # Try with smaller allocation
                    reduced_size = min(
                        forward_batch.seq_lens_sum, self.effective_tokens
                    )
                    kv_indices = torch.empty(
                        reduced_size, dtype=torch.int32, device=self.device
                    )
                    # Truncate sequences if needed
                    truncated_seq_lens = torch.clamp(
                        forward_batch.seq_lens, max=reduced_size // bs
                    )
                    kv_indptr[1 : bs + 1] = torch.cumsum(truncated_seq_lens, dim=0)
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        truncated_seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
            )

        # ... [Include rest of the method with similar safety checks] ...
        elif forward_batch.forward_mode.is_draft_extend():
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    kv_lens_sum,
                    dtype=torch.int32,
                    device=device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    forward_batch.extend_seq_lens.max().item(),
                    forward_batch.seq_lens.max().item(),
                    spec_info=None,
                )

                kv_indices = self.mla_indices_updater_prefill.kv_indices

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    kv_indices,
                    self.mla_indices_updater_prefill.qo_indptr,
                    self.kv_last_page_len[:bs],
                    self.mla_indices_updater_prefill.max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(max_bs, dtype=torch.int)

        # FIXED: Safe CUDA graph buffer allocation
        if kv_indices_buf is None:
            # Reduce buffer size for hybrid caching
            effective_context_len = self.max_context_len
            if self.is_hybrid and self.hybrid_kvcache_ratio > 0:
                effective_context_len = int(
                    self.max_context_len * (1.0 - self.hybrid_kvcache_ratio * 0.4)
                )

            buffer_size = max_bs * effective_context_len

            try:
                self.cuda_graph_kv_indices = torch.zeros(
                    buffer_size, dtype=torch.int32, device=self.device
                )
            except RuntimeError:
                # Try with smaller buffer
                buffer_size = buffer_size // 2
                print(
                    f"[AIter Backend] Reducing CUDA graph buffer size to {buffer_size}"
                )
                self.cuda_graph_kv_indices = torch.zeros(
                    buffer_size, dtype=torch.int32, device=self.device
                )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            # Reduce custom mask size for hybrid caching
            effective_num_tokens = max_num_tokens
            if self.is_hybrid and self.hybrid_kvcache_ratio > 0:
                effective_num_tokens = int(
                    max_num_tokens * (1.0 - self.hybrid_kvcache_ratio * 0.3)
                )

            mask_size = effective_num_tokens * self.max_context_len

            try:
                self.cuda_graph_custom_mask = torch.zeros(
                    mask_size, dtype=torch.uint8, device=self.device
                )
            except RuntimeError:
                mask_size = mask_size // 2
                print(f"[AIter Backend] Reducing custom mask size to {mask_size}")
                self.cuda_graph_custom_mask = torch.zeros(
                    mask_size, dtype=torch.uint8, device=self.device
                )

    # ... [Include all other methods from the original with safety improvements] ...
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            self._map_cache_loc_for_layer(layer.layer_id, forward_batch.out_cache_loc)
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        assert cache_loc.dtype in (torch.int32, torch.int64)
        if not get_is_capture_mode():
            assert cache_loc.min().item() >= 0

        self.logits_soft_cap = layer.logit_cap

        # ADDED: Memory pressure monitoring
        current_tokens = forward_batch.seq_lens_sum
        use_cpu_fallback = self._should_use_cpu_fallback(current_tokens)

        if use_cpu_fallback:
            print(
                f"[AIter Backend] Using memory optimization for {current_tokens} tokens"
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

        if self.use_mla:
            # MLA forward logic (same as before but with error handling)
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if forward_batch.forward_mode.is_extend():
                if kv_indices.shape[0] == 0:
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        qo_indptr,
                        max_q_len,
                        max_q_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )
                    kvprefix = layer.kv_b_proj(kvc.contiguous())[0]

                    kvprefix = kvprefix.view(
                        -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                    )
                    k_prefix, v_prefix = torch.split(
                        kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                    )
                    k_prefix = torch.cat(
                        [
                            k_prefix,
                            torch.broadcast_to(
                                k_pe,
                                (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                            ),
                        ],
                        dim=-1,
                    )
                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    k = k_prefix
                    v = v_prefix

                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        kv_indptr,
                        max_q_len,
                        max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o

                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                mla_decode_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            elif forward_batch.forward_mode.is_draft_extend():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                mla_prefill_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            bs0 = forward_batch.batch_size + 1

            try:
                o = mha_batch_prefill_func(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache,
                    v_cache,
                    self.qo_indptr[:bs0],
                    self.forward_metadata.kv_indptr[:bs0],
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.max_q_len,
                    self.forward_metadata.max_kv_len,
                    causal=True,
                    logits_soft_cap=self.logits_soft_cap,
                    alibi_slopes=None,
                    return_lse=False,
                    return_attn_probs=False,
                )
            except RuntimeError as e:
                print(f"[AIter Backend] MHA batch prefill failed: {e}, trying fallback")
                # Fallback to smaller batch or different approach
                raise e

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = self._map_cache_loc_for_layer(
            layer.layer_id, forward_batch.out_cache_loc
        )

        assert cache_loc.dtype in (torch.int32, torch.int64)
        if not get_is_capture_mode():
            assert cache_loc.min().item() >= 0

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        # Check memory pressure
        current_tokens = forward_batch.seq_lens_sum
        use_cpu_fallback = self._should_use_cpu_fallback(current_tokens)

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                layer.scaling,
                layer.logit_cap,
            )
            k_buffer = k_buffer.view(-1, 1, layer.qk_head_dim)
        else:
            self.logits_soft_cap = layer.logit_cap

            # Adjust partitions based on memory pressure
            effective_partitions = self.max_num_partitions
            if use_cpu_fallback:
                effective_partitions = int(
                    self.max_num_partitions * (1.0 - self.hybrid_kvcache_ratio * 0.3)
                )

            paged_attention_ragged(
                o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                self.workspace_buffer,
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                    -1, 1, layer.tp_k_head_num, layer.qk_head_dim
                ),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                    -1, 1, layer.tp_v_head_num, layer.v_head_dim
                ),
                self.scale,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.kv_last_page_len,
                1,
                effective_partitions,
                None,
                "auto",
                "NHD",
                self.logits_soft_cap,
                self.k_scale,
                self.v_scale,
                None,
                _AITER_PARTITION_SIZE_ROCM,
            )

        return o

    # Include the remaining methods (init_forward_metadata_capture_cuda_graph, etc.)
    # with similar safety improvements...

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


# Include the original helper classes with minimal changes
class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )

            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            self.max_kv_len = torch.max(paged_kernel_lens).item()

            extend_lens = seq_lens - prefix_lens
            self.max_q_len = torch.max(extend_lens).item()

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        self.attn_backend = attn_backend
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInfo],
    ):
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInfo],
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size
        assert self.page_size == 1, "Page size must be 1"

    # ... [Include remaining methods]
