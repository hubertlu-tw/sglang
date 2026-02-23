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

## How the Pipeline Works

Suffix decoding adds a **speculate → verify** loop around the normal autoregressive decode step. Every decode iteration runs through the stages below. The entire pipeline is **synchronous** - the scheduler blocks until all stages finish before scheduling the next batch.

```
  One Suffix Decoding Iteration (synchronous)

  ┌──────────────────┐  ┌────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐
  │ 1. Draft Prep    │→ │ 2. CPU→GPU │→ │ 3. Index       │→ │ 4. Full Attn   │→ │ 5. Model Forward │→ │ 6. Tree Verify │
  │                  │  │  Transfer  │  │  Reconstruction│  │  Mask          │  │                  │  │                │
  │ • update cache   │  │            │  │                │  │                │  │ target model on  │  │ greedy verify  │
  │ • suffix tree    │  │ masks +    │  │ build indices, │  │ [ones|tree]    │  │ all draft tokens │  │ walk accepted  │
  │   lookup         │  │ draft IDs  │  │ positions      │  │ per request    │  │ produce logits   │  │ path, select   │
  │ • BFS reorder    │  │            │  │                │  │ numpy → GPU    │  │                  │  │ final logits   │
  │ • mask build     │  │            │  │                │  │                │  │                  │  │                │
  └──────────────────┘  └────────────┘  └────────────────┘  └────────────────┘  └──────────────────┘  └────────────────┘
        CPU only            H2D              GPU only          CPU+GPU (AMD)          GPU only              GPU only
```

### Stage Details

| Stage | Device | What Happens |
|---|---|---|
| **1. Draft preparation** | CPU | Update suffix cache with verified tokens, search the suffix tree for pattern matches (`suffix_cache.speculate`), BFS-reorder nodes, inject root node, build tree mask (numpy). This is the most expensive stage on CPU, ~20% of the total workload. |
| **2. CPU→GPU transfer** | CPU→GPU | Copy tree mask and draft token IDs to GPU. |
| **3. Index reconstruction** | GPU | `reconstruct_indices_from_tree_mask` kernel builds retrieval indices and position offsets. |
| **4. Full attention mask** | CPU+GPU | *(AMD / non-MLA only)* Build per-request `[ones | tree_mask]` with numpy, transfer to GPU in one copy. |
| **5. Model forward pass** | GPU | Run the full target model on all draft tokens (`batch_size × draft_token_num` tokens). This is the most expensive stage on GPU, ~60% of the total workload. |
| **6. Tree verification** | GPU | Greedy verification kernel walks the draft tree, accepts matching tokens, selects final logits. |

The result of each iteration is **1–N accepted tokens** per request (vs. exactly 1 in normal decode).
**Trade-off: Each step is longer than normal Decode (more tokens through the model), but produces more (1-N) tokens, so wall-clock time per token is lower when the acceptance rate is high.**


---

## AMD-Specific Notes

The core suffix decoding logic is adapted from [PR #13553](https://github.com/sgl-project/sglang/pull/13553) (NVIDIA-only). This PR adds full AMD/ROCm support and several performance optimizations.

### AMD/ROCm Porting

Changes made to enable suffix decoding on AMD:

- **Non-MLA attention backend** (`aiter_backend.py`): Speculative decoding support for the AMD attention path — custom mask handling, metadata computation for draft-extend and target-verify modes, and CUDA graph capture for non-MLA tree verification.
- **Greedy-only verification** (`ngram_info.py`): Forces greedy verification on ROCm because the sampling kernels are not compiled for HIP. Temperature / top-p / top-k are ignored during verification on AMD.
- **ROCm kernel registration** (`common_extension_rocm.cc`, `setup_rocm.py`): Registered `verify_tree_greedy` in the ROCm build of sgl-kernel.

### Performance Optimizations

Optimizations to reduce Python-side overhead compared to the base PR:

- **Numpy-based mask construction** (`suffix_worker.py`): Full attention mask built on CPU with numpy and transferred in a single copy, replacing per-request GPU allocations.
- **O(n) BFS ancestor propagation** (`suffix_cache_adapter.py`): Tree masks built in one forward pass over BFS-ordered nodes instead of nested while-loops.
- **No-copy output_ids** (`suffix_worker.py`, `suffix_cache_adapter.py`): Prompt and output tokens passed separately to avoid list concatenation per request per step.
- **Throttled cleanup** (`suffix_cache_adapter.py`): Inactive-request pruning runs every 8 calls instead of every step.

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
| `test/registered/spec/test_suffix_speculative_decoding.py` | Test suite adapted from [PR #13553](https://github.com/sgl-project/sglang/pull/13553). Config validation, algorithm registration, `SuffixVerifyInput` unit tests, and GSM8K integration tests for fa3/triton/flashinfer backends. |

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

## Testing

Unit tests (no GPU or server needed):

```bash
python -m pytest test/registered/spec/test_suffix_speculative_decoding.py -v -k "Configuration or VerifyInput or Registration"
```

Integration tests (launches a server and runs GSM8K evaluation automatically):

```bash
python -m pytest test/registered/spec/test_suffix_speculative_decoding.py -v -s -k "Decoding"
```

Or run the eval manually against your own running server:

```bash
# Terminal 1 — start the server (see Quick Start above)

# Terminal 2 — run GSM8K eval once the server is up
python -m sglang.test.few_shot_gsm8k \
    --num-shots 5 --num-questions 200 --max-new-tokens 512 \
    --parallel 128 --host http://127.0.0.1 --port 30000

# Check speculative acceptance length
curl http://127.0.0.1:30000/server_info | python -m json.tool | grep spec_accept
```

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
