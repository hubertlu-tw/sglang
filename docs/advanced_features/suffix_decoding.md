# Suffix Decoding & NGRAM Fixes for AMD GPUs

This document describes the **Suffix Decoding** feature added to SGLang, the **NGRAM sampling fixes** for AMD (HIP/ROCm) GPUs, and the associated configuration parameters.

---

## Table of Contents

- [Overview](#overview)
- [Suffix Decoding: Background](#suffix-decoding-background)
- [Suffix Decoding vs NGRAM Speculation](#suffix-decoding-vs-ngram-speculation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [AMD-Specific: Sampling Verification](#amd-specific-sampling-verification)
- [NGRAM Fixes for AMD GPUs](#ngram-fixes-for-amd-gpus)
- [Files Changed](#files-changed)
- [Implementation Details](#implementation-details)
- [Known Limitations](#known-limitations)
- [Supported Features](#supported-features)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

**Suffix Decoding** ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975)) is a training-free, model-free speculative decoding method that accelerates LLM inference by exploiting repetitive patterns in token sequences. It requires no draft model and runs its pattern-matching logic entirely on the CPU using suffix trees.

This changeset also documents the **sampling verification limitations** on **AMD MI300/MI350 GPUs** where the sampling kernels are not compiled for ROCm.

---

## Suffix Decoding: Background

### Core Concept

Suffix Decoding maintains **two suffix trees**:

1. **Per-request (prompt) tree**: Built from the prompt tokens of the current request. Captures intra-request repetition (e.g., repeated variable names in code).
2. **Global tree**: Built from previously completed responses across all requests. Captures inter-request repetition (e.g., similar code patterns across files in an agentic loop). Uses FIFO eviction when a capacity limit is reached.

### Speculation Process

1. **Pattern extraction**: Take the last N tokens of the current sequence as a query pattern.
2. **Suffix matching**: Search both trees for the longest matching suffix.
3. **Frequency-based ranking**: Gather candidate continuations from match points, ranked by observed frequency.
4. **Tree-structured proposal**: Build a tree of candidate tokens (with branches for alternative continuations).
5. **Frequency filtering**: Filter out candidates below a minimum probability threshold (`min_token_prob`).
6. **Adaptive length**: Limit speculation length via `max_spec_factor * prefix_match_length`.
7. **Parallel verification**: The target model verifies the entire candidate tree in a single forward pass.

### Adaptive Speculation

Unlike NGRAM or EAGLE which speculate a fixed number of tokens every step, Suffix Decoding **dynamically adjusts** per request per step:
- Long, confident matches produce more speculative tokens.
- No match results in zero speculation (falls back to normal decode).
- This avoids wasting GPU compute on low-confidence speculation.

---

## Suffix Decoding vs NGRAM Speculation

| Aspect | Suffix Decoding | NGRAM |
|---|---|---|
| **Data structure** | Suffix tree (compressed trie of all suffixes) | Hash table of fixed N-gram windows |
| **Context window** | Entire sequence history (prompt + all generated tokens) | Only last N tokens (fixed window) |
| **Match length** | Variable -- finds the longest common suffix | Fixed -- always matches exactly N-1 tokens |
| **Cross-request learning** | Yes, via global tree with FIFO eviction | No (each request is independent) |
| **Adaptive speculation** | Yes -- dynamic per-request, per-step | No -- fixed `draft_token_num` every step |
| **Best use case** | Repetitive agentic workloads, code editing, RL rollouts | General-purpose, works on any workload |
| **CPU overhead** | Higher (suffix tree construction/search) | Lower (simple hash lookups) |
| **Draft model needed** | No | No |
| **GPU memory overhead** | None (CPU-only speculation) | None (CPU-only speculation) |

---

## Installation

### Python Dependencies

Suffix Decoding requires the `arctic-inference` library:

```bash
pip install arctic-inference==0.1.1
```

### Do I Need to Reinstall SGLang or sgl-kernel?

**For Suffix Decoding and NGRAM on AMD (greedy verification):**
- **No `sgl-kernel` rebuild required.** All changes are Python-level. The existing `verify_tree_greedy` kernel is already compiled in the ROCm build.
- Just install `arctic-inference` and use the updated Python files.

**For sampling verification (temperature/top-p/top-k) on AMD:**
- **Not currently available.** The sampling kernels (`tree_speculative_sampling_target_only`, `top_k_renorm_prob`, `top_p_renorm_prob`) are not compiled for ROCm. They need to be compiled and registered in `common_extension_rocm.cc`. See [AMD-Specific: Sampling Verification](#amd-specific-sampling-verification) for details.

**For NVIDIA GPUs:**
- **No rebuild required.** All kernels (greedy + sampling) are already compiled in the NVIDIA build of `sgl-kernel`.

---

## Quick Start

### Basic Suffix Decoding

```bash
python -m sglang.launch_server \
    --model-path amd/Llama-3.1-405B-Instruct-FP8-KV \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 24
```

### Suffix Decoding on AMD MI300 (8-GPU TP)

```bash
python3 -m sglang.launch_server \
    --attention-backend triton \
    --model amd/Llama-3.1-405B-Instruct-FP8-KV \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.9 \
    --cuda-graph-max-bs 1 \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-metrics
```

### Advanced Configuration

```bash
python -m sglang.launch_server \
    --model-path amd/Llama-3.1-405B-Instruct-FP8-KV \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 32 \
    --speculative-suffix-max-tree-depth 24 \
    --speculative-suffix-max-cached-requests 10000 \
    --speculative-suffix-max-spec-factor 1.5 \
    --speculative-suffix-min-token-prob 0.05
```

---

## Configuration Parameters

### Suffix Decoding Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--speculative-algorithm SUFFIX` | str | None | Enable suffix decoding |
| `--speculative-num-draft-tokens` | int | `max_tree_depth` | Maximum number of speculative tokens per step |
| `--speculative-suffix-max-tree-depth` | int | 24 | Maximum depth of suffix trees (limits prefix match + speculation length) |
| `--speculative-suffix-max-cached-requests` | int | 10000 | Max requests in global suffix tree. 0 = disable global cache, -1 = unlimited |
| `--speculative-suffix-max-spec-factor` | float | 1.0 | Speculation length factor: `max_spec = factor * prefix_match_length` |
| `--speculative-suffix-min-token-prob` | float | 0.1 | Minimum frequency-based probability to speculate a token |

### AMD Sampling Parameter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--enable-speculative-sampling` | flag | False | Placeholder for sampling-based verification on AMD GPUs. Currently non-functional; sampling kernels need to be compiled for AMD (see [AMD-Specific: Sampling Verification](#amd-specific-sampling-verification)). |

### Performance Tuning Recommendations

**For high-repetition tasks** (code editing, agentic loops):
```bash
--speculative-suffix-max-spec-factor 2.0 \
--speculative-suffix-min-token-prob 0.05 \
--speculative-num-draft-tokens 32
```

**For mixed workloads** (balanced defaults):
```bash
--speculative-suffix-max-spec-factor 1.0 \
--speculative-suffix-min-token-prob 0.1 \
--speculative-num-draft-tokens 24
```

**For memory-constrained systems**:
```bash
--speculative-suffix-max-cached-requests 1000 \
--speculative-suffix-max-tree-depth 16
```

---

## AMD-Specific: Sampling Verification

### Current Status

On AMD GPUs (HIP/ROCm), speculative decoding verification is **forced to greedy mode**. This means `temperature`, `top_p`, and `top_k` settings are **silently ignored** during the verification phase. This affects both NGRAM and Suffix decoding.

### Root Cause

The sampling verification path requires three C++ kernels that are **not compiled** in the ROCm build of `sgl-kernel`:

- `tree_speculative_sampling_target_only` -- tree-based rejection sampling
- `top_k_renorm_prob` -- top-k probability renormalization
- `top_p_renorm_prob` -- top-p (nucleus) probability renormalization

These kernels are registered in `sgl-kernel/csrc/common_extension.cc` (NVIDIA build) but **not** in `sgl-kernel/csrc/common_extension_rocm.cc`. The source file `csrc/speculative/speculative_sampling.cu` exists but is not listed in `sgl-kernel/setup_rocm.py`.

The greedy verification kernel (`verify_tree_greedy`) **is** available on ROCm, so greedy-mode speculative decoding works correctly.

### The `--enable-speculative-sampling` Flag

A flag `--enable-speculative-sampling` has been added as a **placeholder**. Currently, setting this flag will produce a warning and fall back to greedy verification because the required kernels are not compiled for AMD.

### Why These Kernels Are Not Available on ROCm

The three sampling kernels are **not compiled** in the ROCm build of `sgl-kernel`:

- `tree_speculative_sampling_target_only` -- source exists at `csrc/speculative/speculative_sampling.cu` but is not listed in `setup_rocm.py`
- `top_k_renorm_probs` / `top_p_renorm_probs` -- no standalone source files in `sgl-kernel/csrc/`; declared in the header but only linked in the NVIDIA build

These kernels need to be compiled and registered in `common_extension_rocm.cc` to enable sampling-based verification on AMD.

---

## NGRAM Fixes for AMD GPUs

The following changes fix issues with NGRAM speculative decoding on AMD (HIP) GPUs:

### 1. Greedy Verification Works on AMD

The existing `verify_tree_greedy` kernel **is** compiled in `common_extension_rocm.cc` and works correctly. Both NGRAM and Suffix decoding use greedy verification on AMD by default.

### 2. Sampling Kernels NOT Available on AMD (`ngram_info.py`)

The sampling verification kernels (`tree_speculative_sampling_target_only`, `top_k_renorm_prob`, `top_p_renorm_prob`) are **not compiled** in the ROCm build of `sgl-kernel`. The HIP import block in `ngram_info.py` only imports `verify_tree_greedy`.

This means temperature, top-p, and top-k are **ignored** during speculative verification on AMD.

### 3. `TREE_SPEC_KERNEL_AVAILABLE` (`spec_utils.py`)

`TREE_SPEC_KERNEL_AVAILABLE` remains set to `_is_cuda` only, since the full sampling kernel suite is not available on HIP. On AMD, the code always falls through to `_greedy_verify`.

### 4. Placeholder Flag `--enable-speculative-sampling` (`ngram_info.py`, `server_args.py`)

A CLI flag `--enable-speculative-sampling` has been added. Currently it produces a runtime warning and falls back to greedy because the sampling kernels are not compiled for AMD.

---

## Files Changed

### New Files

| File | Description |
|---|---|
| `python/sglang/srt/speculative/suffix_cache_adapter.py` | Adapter wrapping `arctic-inference` SuffixDecodingCache to match SGLang's NgramCache interface |
| `python/sglang/srt/speculative/suffix_info.py` | `SuffixVerifyInput` data structure for suffix verification (extends `NgramVerifyInput`) |
| `python/sglang/srt/speculative/suffix_worker.py` | `SuffixWorker` class managing suffix speculation lifecycle (extends `NGRAMWorker`) |
| `docs/advanced_features/suffix_decoding.md` | This documentation file |

### Modified Files

| File | Change Summary |
|---|---|
| `python/sglang/srt/speculative/spec_info.py` | Added `SUFFIX` enum, `is_suffix()`, `SUFFIX_VERIFY` type, worker factory |
| `python/sglang/srt/server_args.py` | Added SUFFIX CLI args, validation, `--enable-speculative-sampling` |
| `python/sglang/srt/managers/scheduler.py` | Treat SUFFIX like NGRAM in disaggregation path |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | SUFFIX-aware CUDA graph capture/replay/spec-info |
| `python/sglang/srt/speculative/ngram_info.py` | AMD sampling gating logic with fallback warning |
| `python/sglang/srt/speculative/spec_utils.py` | `TREE_SPEC_KERNEL_AVAILABLE` remains CUDA-only (sampling kernels not compiled for ROCm) |
| `python/sglang/srt/utils/common.py` | Added `is_arctic_inference_available()` utility |

---

## Implementation Details

### Architecture

```
SuffixWorker (extends NGRAMWorker)
  |
  +-- SuffixCacheAdapter (wraps arctic-inference SuffixDecodingCache)
  |     |-- start_request(req_id, prompt)     # init per-request tree
  |     |-- add_active_response(req_id, tokens) # update with generated tokens
  |     |-- speculate(req_id, pattern, ...)    # get draft tree
  |     +-- stop_request(req_id)               # cleanup
  |
  +-- SuffixVerifyInput (extends NgramVerifyInput)
  |     |-- spec_input_type = SUFFIX_VERIFY
  |     +-- _fill_requests() with debug logging
  |
  +-- Inherits from NGRAMWorker:
        |-- _init_preallocated_tensors()
        |-- forward_batch_generation()
        +-- Tree mask construction & verification pipeline
```

### Verification Flow

1. `SuffixWorker._prepare_draft_tokens()` calls `SuffixCacheAdapter.batch_get()` on CPU.
2. `SuffixWorker._prepare_for_speculative_decoding()` converts draft tree to GPU tensors, sets `batch.spec_algorithm = SUFFIX`, creates `SuffixVerifyInput`.
3. Target model runs a single forward pass over all draft tokens.
4. `NgramVerifyInput.verify()` (inherited) runs greedy or sampling verification:
   - On NVIDIA (CUDA): sampling if `temperature > 0`, else greedy. Both paths have compiled kernels.
   - On AMD (HIP/ROCm): **always greedy**. The sampling kernels (`tree_speculative_sampling_target_only`, `top_k_renorm_prob`, `top_p_renorm_prob`) are not compiled for ROCm. See [AMD-Specific: Sampling Verification](#amd-specific-sampling-verification).
5. Accepted tokens are committed; rejected tokens' KV cache is freed.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **External dependency** | Requires `arctic-inference` pip package |
| **No DP attention** | Data-parallel attention is not supported with suffix decoding |
| **No overlap scheduling** | Overlap scheduler is disabled (`disable_overlap_schedule = True`) |
| **No mixed chunked prefill** | Mixed chunk is disabled (`enable_mixed_chunk = False`) |
| **Python-level batch loop** | `SuffixCacheAdapter.batch_get` iterates in Python; potential CPU bottleneck at large batch sizes |
| **CPU-GPU round-trip** | Suffix tree lives on CPU; each decode step has CPU-GPU data transfer overhead |
| **AMD sampling not available** | Sampling kernels not compiled for ROCm; temperature/top-p are always ignored on AMD during speculative verification (greedy only). These kernels need to be compiled for AMD. |
| **CUDA only** | Suffix decoding requires CUDA-compatible devices (NVIDIA or AMD ROCm/HIP) |

---

## Supported Features

| Feature | Status |
|---|---|
| Suffix tree speculation (via arctic-inference) | Supported |
| Cross-request caching (global tree) | Supported |
| Adaptive speculation length | Supported |
| Frequency-based filtering | Supported |
| Tree-structured candidates (branching) | Supported |
| CUDA graph capture/replay | Supported |
| Triton attention backend | Supported |
| FlashInfer attention backend | Supported |
| Greedy verification | Supported (always used on AMD) |
| Sampling verification (temperature/top-p/top-k) | Supported on NVIDIA; **not available on AMD** (kernels need to be compiled for ROCm) |
| Grammar / structured output | Supported |
| Tensor parallelism | Supported |
| Debug logging | Supported (set `SUFFIX_DEBUG_TREE=1` env var) |

---

## Troubleshooting

### Import Error: Arctic Inference Not Found

```
ImportError: Arctic Inference is required for suffix decoding
```

**Solution**: `pip install arctic-inference==0.1.1`

### Temperature Being Ignored on AMD

On AMD GPUs, temperature/top-p/top-k are **always ignored** during speculative verification. Greedy verification is used instead because the sampling kernels are not compiled for ROCm. See [AMD-Specific: Sampling Verification](#amd-specific-sampling-verification) for details.

### Low Acceptance Rate

1. Increase `--speculative-suffix-max-spec-factor` (try 1.5 or 2.0).
2. Lower `--speculative-suffix-min-token-prob` (try 0.05).
3. Increase `--speculative-suffix-max-tree-depth` (try 32).
4. Verify the workload has repetitive patterns -- suffix decoding works best with repetition.

### No Speedup

1. Check task type: suffix decoding benefits most from repetitive workloads.
2. Monitor acceptance rate via `/get_server_info` endpoint (`avg_spec_accept_length`).
3. For general-purpose tasks, consider using NGRAM or EAGLE instead.

### Memory Issues

1. Reduce `--speculative-suffix-max-cached-requests` (try 1000).
2. Reduce `--speculative-suffix-max-tree-depth` (try 16).
3. Disable global cache: `--speculative-suffix-max-cached-requests 0`.

---

## References

- [Suffix Decoding Paper (arXiv:2411.04975)](https://arxiv.org/abs/2411.04975)
- [Arctic Inference GitHub](https://github.com/snowflakedb/ArcticInference)
- [Snowflake Blog: SuffixDecoding at Production Scale](https://www.snowflake.com/content/snowflake-site/global/en/engineering-blog/suffixdecoding-arctic-inference-vllm)
- [SGLang Speculative Decoding Docs](https://docs.sglang.ai/backend/speculative_decoding.html)
- [vLLM Suffix Decoding PR #25784](https://github.com/vllm-project/vllm/pull/25784)
- [SGLang Suffix Decoding PR #13553](https://github.com/sgl-project/sglang/pull/13553)
