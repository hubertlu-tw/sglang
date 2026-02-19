# Suffix Decoding on AMD GPUs

**In simple terms:** Instead of generating tokens one at a time, Suffix Decoding looks at what the model has already said (and what it said in previous requests) to guess what comes next — like autocomplete. If the same pattern appeared before, it proposes multiple tokens at once and the model verifies them in a single step, making inference faster.

**How is it different from NGRAM?** NGRAM only looks at the last few tokens (a fixed window) to guess the next ones, and each request starts from scratch. Suffix Decoding searches the *entire* history using a suffix tree, finds matches of any length, learns across requests, and dynamically adjusts how many tokens to speculate — proposing more when it's confident and falling back to normal decoding when it's not.

Both methods are draft-model-free and run on CPU, but Suffix Decoding shines on **repetitive workloads** (code editing, agentic loops) where the same patterns keep appearing.

For more details, see the [Suffix Decoding paper (arXiv:2411.04975)](https://arxiv.org/abs/2411.04975).

---

## Installation

### 1. Stash Local Changes & Switch to the Suffix Decoding Branch

```bash
cd sglang
git stash
git remote add amd-fork https://github.com/amd-pedghazi/sglang.git
git fetch amd-fork feat/suffix-decoding-amd
git checkout amd-fork/feat/suffix-decoding-amd
```

### 2. Install sgl-kernel (ROCm)

```bash
cd sgl-kernel
pip uninstall sgl-kernel
python setup_rocm.py install
```

### 3. Install Python Dependencies

```bash
pip install arctic-inference==0.1.1
```

---

## Quick Start — GLM-4.7-FP8

Launch the server with suffix decoding on 8× AMD MI300/MI325/MI355 GPUs:

```bash
AITER_ONLINE_TUNE=1 SGLANG_AITER_MLA_PERSIST=1 SGLANG_USE_AITER=1 SAFETENSORS_FAST_GPU=1 \
python3 -m sglang.launch_server \
    --model zai-org/GLM-4.7-FP8 \
    --tp 8 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-metrics \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --trust-remote-code \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 16 \
    --speculative-suffix-max-spec-factor 2.0
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

## AMD-Specific Notes

- **Greedy verification** works out of the box on AMD (ROCm).
- **Sampling verification** (temperature / top-p / top-k) is **not available** on AMD — the sampling kernels are not compiled for ROCm. Speculative verification always uses greedy on AMD GPUs.

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
| No speedup | Suffix decoding benefits most from repetitive workloads; check acceptance rate via `/get_server_info` |

---

## References

- [Suffix Decoding Paper (arXiv:2411.04975)](https://arxiv.org/abs/2411.04975)
- [vLLM Suffix Decoding PR #25784](https://github.com/vllm-project/vllm/pull/25784)
- [SGLang Suffix Decoding PR #13553](https://github.com/sgl-project/sglang/pull/13553)
