# Qwen3.8-27B gfx1151 optimization journey

All Python data points use Hubert's tree, not `/sgl-workspace/sglang`:

```bash
export PYTHONPATH=/sgl-workspace/hubert_sgl/python/sglang/kernels/aot/python:/sgl-workspace/hubert_sgl/python
export HF_HUB_OFFLINE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_WARMUP_TIMEOUT=${SGLANG_WARMUP_TIMEOUT:-1800}
```

`sgl_kernel` is a **separate** compiled package (`import sgl_kernel`). Putting only `.../hubert_sgl/python` on `PYTHONPATH` picks up SGLang, not the AOT extension. The extra `.../kernels/aot/python` entry is required after an in-place kernel build.

Shared launch knobs (keep these identical across steps unless a step documents a change):

- `--attention-backend triton`
- `--host 0.0.0.0 --port 30000`
- `--mem-fraction-static 0.93`
- `--max-running-requests 4`
- `--chunked-prefill-size 4096`
- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder`

After the server is healthy, measure every row with:

```bash
python3 /sgl-workspace/hubert_sgl/benchmark/gsm8k/bench_sglang.py --port 30000 --num-questions 10 --parallel 10
```

Record output tok/s and accuracy in the table at the bottom. Do not compare rows until that command has been run for that exact launch.

---

## 1. Baseline — `b556a3cc72` + Qwen3.8-27B BF16

Tree: `/sgl-workspace/hubert_sgl` at `b556a3cc72`.

Model: `/home/hubertlu/.cache/huggingface/hub/Qwen/Qwen3.8-27B` (not the `amd/` Quark checkpoint).

```bash
MODEL_PATH=${MODEL_PATH:-/home/hubertlu/.cache/huggingface/hub/Qwen/Qwen3.8-27B}
SGLANG_MAMBA_SSM_DTYPE=bfloat16 python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.75 \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

---

## 2. + Quark W4A16 — `08cf09194f` + INT4 checkpoint

Tree: `/sgl-workspace/hubert_sgl` at `08cf09194f` (current `gfx1151_optim` tip).

Model: `/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`.

```bash
MODEL_PATH=${MODEL_PATH:-/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.93 \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

---

## 3. + Quark W4A16 — `08cf09194f` + INT4 checkpoint + MTP

Tree: `/sgl-workspace/hubert_sgl` at `08cf09194f` (current `gfx1151_optim` tip).

Model: `/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`.

```bash
MODEL_PATH=${MODEL_PATH:-/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.93 \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

---

## 4. + Clint skinny GEMM (`wvSplitK`) — same Quark checkpoint

Python + kernel sources: [clintg6/sglang@3c8a486](https://github.com/clintg6/sglang/commit/3c8a486f63b91733df8091758cc75cc53ff49345).

Those files are **not** in `hubert_sgl` yet (`skinny_gemms*.cu`, `rocm_wv_split_k.py`). Apply that commit (or cherry-pick) onto `08cf09194f` **before** rebuilding kernels. Until `setup_rocm.py` compiles `csrc/gemm/skinny_gemms.cu` and `skinny_gemms_int4.cu`, the verify snippet below prints `False` and SGLang stays on Triton/hipBLAS.

### Why not `pip install` the wheel into site-packages?

`PYTHONPATH=/sgl-workspace/hubert_sgl/python` does **not** isolate `sgl_kernel`. A global

`pip install --force-reinstall --no-deps .../sglang_kernel-*.whl`

replaces the extension for **every** interpreter, including `/sgl-workspace/sglang`. Copying Hubert Python diffs into `/sgl-workspace/sglang` and then building that tree does the same: it mutates the default workspace wheel.

Build **in place** under Hubert's AOT tree, then put that package **first** on `PYTHONPATH`.

### Rebuild kernels used only by Hubert's `PYTHONPATH`

```bash
# 1) Apply Clint sources into /sgl-workspace/hubert_sgl first.

export PYTHONPATH=/sgl-workspace/hubert_sgl/python/sglang/kernels/aot/python:/sgl-workspace/hubert_sgl/python
cd /sgl-workspace/hubert_sgl/python/sglang/kernels/aot
rm -rf build
AMDGPU_TARGET=gfx1151 python3 setup_rocm.py build_ext --inplace
```

`build_ext --inplace` drops `common_ops*.so` next to `python/sgl_kernel/` in this tree. No `pip install` is required.

Optional wheel, still isolated (do **not** omit `--target`):

```bash
cd /sgl-workspace/hubert_sgl/python/sglang/kernels/aot
wheel_dir=$(mktemp -d)
AMDGPU_TARGET=gfx1151 python3 setup_rocm.py bdist_wheel --dist-dir "$wheel_dir"
mkdir -p /sgl-workspace/hubert_sgl/.local_kernels
pip install --force-reinstall --no-deps --target /sgl-workspace/hubert_sgl/.local_kernels "$wheel_dir"/sglang_kernel-*.whl
export PYTHONPATH=/sgl-workspace/hubert_sgl/.local_kernels:/sgl-workspace/hubert_sgl/python
```

Verify **with the same `PYTHONPATH`** you will use to launch. The file path must be under `/sgl-workspace/hubert_sgl`, not site-packages:

```bash
PYTHONPATH=/sgl-workspace/hubert_sgl/python/sglang/kernels/aot/python:/sgl-workspace/hubert_sgl/python python3 - <<'PY'
import sgl_kernel
import torch
print(sgl_kernel.__file__)
print(hasattr(torch.ops.sgl_kernel, "wvSplitK"))
print(hasattr(torch.ops.sgl_kernel, "wvSplitK_int4_g"))
PY
```

Both ops should print `True`. Otherwise the Quark linear path falls back to slower Triton/hipBLAS GEMM.

### Launch

```bash
MODEL_PATH=${MODEL_PATH:-/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend triton \
    --speculative-draft-attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.93 \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

---

## 5. + DFlash2 draft — Quark target + DFlash2 draft

Target: same Quark INT4 checkpoint as steps 3–4.

Draft: `/root/.cache/huggingface/hub/incoai/Qwen3.8-27B-DFlash2`.

```bash
MODEL_PATH=${MODEL_PATH:-/home/hubertlu/.cache/huggingface/hub/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/root/.cache/huggingface/hub/incoai/Qwen3.8-27B-DFlash2}
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --attention-backend triton \
    --speculative-draft-attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.93 \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-num-draft-tokens 8
```

---

## Results (fill after each GSM8K run)

| Step | Stack | Tree | Target weights | Spec | GSM8K output tok/s | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | BF16 baseline | `b556a3cc72` | `Qwen/Qwen3.8-27B` | none | 2.528 | 1.000 |
| 2 | + Quark W4A16 | `08cf09194f` | `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | none | 4.329 | 1.000 |
| 3 | + Quark W4A16 + MTP/EAGLE | `08cf09194f` | `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | EAGLE 3/1/4 | 9.315 | 1.000 |
| 4 | + Clint `wvSplitK` | `08cf09194f` + [3c8a486](https://github.com/clintg6/sglang/commit/3c8a486f63b91733df8091758cc75cc53ff49345) | same Quark | same + draft attn triton | | |
| 5 | + DFlash2 | same as 4 | Quark + `incoai/Qwen3.8-27B-DFlash2` | DFLASH, 8 draft tokens | | |
