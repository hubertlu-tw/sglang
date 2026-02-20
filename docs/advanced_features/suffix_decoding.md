# Suffix Decoding on AMD GPUs

**In simple terms:** Instead of generating tokens one at a time, Suffix Decoding looks at what the model has already said (and what it said in previous requests) to guess what comes next — like autocomplete. If the same pattern appeared before, it proposes multiple tokens at once and the model verifies them in a single step, making inference faster.

**How is it different from NGRAM?** NGRAM only looks at the last few tokens (a fixed window) to guess the next ones, and each request starts from scratch. Suffix Decoding searches the *entire* history using a suffix tree, finds matches of any length, learns across requests, and dynamically adjusts how many tokens to speculate — proposing more when it's confident and falling back to normal decoding when it's not.

Both methods are draft-model-free and run on CPU, but Suffix Decoding shines on **repetitive workloads** (code editing, agentic loops) where the same patterns keep appearing.

For more details, see the [Suffix Decoding paper (arXiv:2411.04975)](https://arxiv.org/abs/2411.04975).

---

## Installation

### 1. Stash Local Changes & Switch to the Suffix Decoding Branch

```bash
cd /sgl-workspace/sglang
git stash
git remote add amd-fork https://github.com/amd-pedghazi/sglang.git
git fetch amd-fork feat/suffix-decoding-amd
git checkout amd-fork/feat/suffix-decoding-amd
```

### 2. Install sgl-kernel (ROCm)

```bash
cd /sgl-workspace/sglang/sgl-kernel
pip uninstall sgl-kernel
python setup_rocm.py install
```

### 3. Install Python Dependencies

```bash
pip install arctic-inference==0.1.1
```

---

## Quick Start — GLM-4.7-FP8

Launch the server with suffix decoding. Tested on 8× AMD MI300 GPUs using **lmsysorg/sglang:v0.5.7-rocm700-mi30x** image:

```bash
AITER_ONLINE_TUNE=1 SGLANG_AITER_MLA_PERSIST=1 SGLANG_USE_AITER=1 SAFETENSORS_FAST_GPU=1 \
python3 -m sglang.launch_server \
    --model zai-org/GLM-4.7-FP8 \
    --tp 8 \
    --enable-metrics \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --trust-remote-code \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 4 \
    --speculative-suffix-max-spec-factor 2.0 \
    --speculative-suffix-min-token-prob 0.2
```

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `--speculative-algorithm SUFFIX` | — | Enable suffix decoding |
| `--speculative-num-draft-tokens` | 24 | Max speculative tokens per step |
| `--speculative-suffix-max-tree-depth` | 24 | Max depth of suffix trees |
| `--speculative-suffix-max-cached-requests` | 10000 | Max requests in global suffix tree (0 = disable, -1 = unlimited) |
| `--speculative-suffix-max-spec-factor` | 1.0 | Speculation length = factor × prefix match length |
| `--speculative-suffix-min-token-prob` | 0.1 | Min frequency-based probability to speculate a token |

### Tuning Tips

- **High-repetition tasks** (code editing, agentic loops): use `--speculative-suffix-max-spec-factor 2.0 --speculative-num-draft-tokens 32`
- **Memory-constrained**: reduce `--speculative-suffix-max-cached-requests 1000 --speculative-suffix-max-tree-depth 16`

---

## Performance Optimizations

The suffix cache adapter and worker include several optimizations to reduce Python-side overhead:

- **Numpy-based mask construction**: The full attention mask is built on CPU with numpy and transferred to GPU in a single copy, replacing per-request `torch.ones().cuda()` + `torch.cat()` allocations.
- **O(n) BFS ancestor propagation**: Tree masks are built with a single forward pass over BFS-ordered nodes instead of O(n×d) nested while-loops.
- **Throttled cleanup**: Inactive-request pruning in the cache adapter runs every 8 calls instead of every step.

---

## AMD-Specific Notes

- **Greedy verification** works out of the box on AMD (ROCm).
- **Sampling verification** (temperature / top-p / top-k) is **not available** on AMD — the sampling kernels are not compiled for ROCm. Speculative verification always uses greedy on AMD GPUs.

---

## Changed and New Files

Below is the complete list of files modified or added by the suffix decoding feature branch.

### New Files

| File | Description |
|---|---|
| `python/sglang/srt/speculative/suffix_worker.py` | Suffix decoding worker (extends NGRAMWorker). Overrides draft token preparation and mask construction with optimized numpy-based implementations. All verification logic is inherited from NGRAMWorker. |
| `python/sglang/srt/speculative/suffix_info.py` | Data structures for suffix decoding verification (extends NgramVerifyInput). Adds SUFFIX_VERIFY spec input type and debug logging of accepted tokens. |
| `python/sglang/srt/speculative/suffix_cache_adapter.py` | Adapter wrapping `arctic_inference.SuffixDecodingCache` to match the NgramCache interface. Handles BFS reordering, root injection, and tree mask construction. |
| `docs/advanced_features/suffix_decoding.md` | This documentation file. |

### Modified Files

| File | Changes |
|---|---|
| `python/sglang/srt/speculative/spec_info.py` | Added `SUFFIX` to `SpeculativeAlgorithm` enum, `SUFFIX_VERIFY` to `SpecInputType`, `is_suffix()` method, and `SuffixWorker` factory in `create_worker()`. |
| `python/sglang/srt/speculative/spec_utils.py` | Updated comment: sampling kernel not compiled for HIP/ROCm yet. |
| `python/sglang/srt/speculative/ngram_info.py` | Added AMD/ROCm fallback: forces greedy verification when sampling kernels are unavailable. Added warning when `--enable-speculative-sampling` is set on AMD. |
| `python/sglang/srt/managers/scheduler.py` | Added `spec_algorithm.is_suffix()` check so suffix decoding skips draft KV pool allocation (like NGRAM). |
| `python/sglang/srt/server_args.py` | Added suffix-specific CLI args (`--speculative-suffix-max-tree-depth`, `--speculative-suffix-max-cached-requests`, `--speculative-suffix-max-spec-factor`, `--speculative-suffix-min-token-prob`, `--enable-speculative-sampling`). Added SUFFIX validation logic. |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | Added `is_suffix()` checks for CUDA graph capture and replay, mirroring the existing NGRAM path. |
| `python/sglang/srt/layers/attention/aiter_backend.py` | Added non-MLA speculative decoding support for AMD: custom mask / mask_indptr fields in ForwardMetadata, direct metadata computation for DRAFT_EXTEND and TARGET_VERIFY modes, CUDA graph capture path for non-MLA tree verification. |
| `python/sglang/srt/utils/common.py` | Added `is_arctic_inference_available()` utility function. |
| `sgl-kernel/csrc/common_extension_rocm.cc` | Registered `verify_tree_greedy` kernel for the ROCm build. |
| `sgl-kernel/setup_rocm.py` | Added `common_extension_rocm.cc` to the ROCm build sources. |

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Requires `arctic-inference` | External pip dependency |
| No DP attention | Data-parallel attention not supported |
| No overlap scheduling | Overlap scheduler is disabled |
| AMD sampling unavailable | Temperature/top-p ignored during speculative verification on AMD |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ImportError: Arctic Inference is required` | `pip install arctic-inference==0.1.1` |
| Temperature ignored on AMD | Expected — greedy verification only on ROCm |
| Low acceptance rate | Increase `--speculative-suffix-max-spec-factor` (try 2.0), lower `--speculative-suffix-min-token-prob` (try 0.05) |
| No speedup | Suffix decoding benefits most from repetitive workloads; check acceptance rate. |

---

## References

- [Suffix Decoding Paper (arXiv:2411.04975)](https://arxiv.org/abs/2411.04975)
- [vLLM Suffix Decoding PR #25784](https://github.com/vllm-project/vllm/pull/25784)
- [SGLang Suffix Decoding PR #13553](https://github.com/sgl-project/sglang/pull/13553)
