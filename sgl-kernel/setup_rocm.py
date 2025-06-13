# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/torch_extension_rocm.cc",
    "csrc/speculative/eagle_utils.cu",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

supported_arches = {"gfx942", "gfx950"}
have_hip_runtime = torch.version.hip is not None
have_gpu = have_hip_runtime and torch.cuda.is_available()


amdgpu_target = None
if have_gpu:
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if amdgpu_target not in supported_arches:
            print(
                f"[setup_rocm] Unsupported AMD GPU '{amdgpu_target}'. "
                f"Supported: {', '.join(sorted(supported_arches))}. "
                "Skipping ROCm extension build."
            )
            have_gpu = False
    except (AssertionError, RuntimeError, AttributeError) as exc:
        print(
            f"[setup_rocm] Could not query GPU properties ({exc}). "
            "Skipping ROCm extension build."
        )
        have_gpu = False
else:
    if not have_hip_runtime:
        print(
            "[setup_rocm] CPU-only PyTorch build detected. "
            "Skipping ROCm extension build."
        )
    else:
        print(
            "[setup_rocm] No GPU visible to PyTorch. " "Skipping ROCm extension build."
        )

if have_gpu:
    hipcc_flags = [
        "-DNDEBUG",
        f"-DOPERATOR_NAMESPACE={operator_namespace}",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "-D__HIP_PLATFORM_AMD__=1",
        f"--amdgpu-target={amdgpu_target}",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
    ]

    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.common_ops",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": hipcc_flags,
                "cxx": cxx_flags,
            },
            libraries=libraries,
            extra_link_args=extra_link_args,
            py_limited_api=False,
        ),
    ]

    setup(
        name="sgl-kernel",
        version=_get_version(),
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        options={"bdist_wheel": {"py_limited_api": "cp39"}},
    )
