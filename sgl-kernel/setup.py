import os
from pathlib import Path
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

root = Path(__file__).parent.resolve()


def _update_wheel_platform_tag():
    wheel_dir = Path("dist")
    if wheel_dir.exists() and wheel_dir.is_dir():
        old_wheel = next(wheel_dir.glob("*.whl"))
        new_wheel = wheel_dir / old_wheel.name.replace(
            "linux_x86_64", "manylinux2014_x86_64"
        )
        old_wheel.rename(new_wheel)


def _get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def _get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0

def is_cuda() -> bool:
    """Return whether it is CUDA on the NVIDIA CUDA platform."""
    return torch.cuda.is_available() and torch.version.cuda


def is_hip() -> bool:
    """Return whether it is HIP on the AMD ROCm platform."""
    return torch.cuda.is_available() and torch.version.hip

def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')

if is_cuda():
    cutlass_default = root / "3rdparty" / "cutlass"
    cutlass = Path(os.environ.get("CUSTOM_CUTLASS_SRC_DIR", default=cutlass_default))
    flashinfer = root / "3rdparty" / "flashinfer"
    turbomind = root / "3rdparty" / "turbomind"
    include_dirs = [
        cutlass.resolve() / "include",
        cutlass.resolve() / "tools" / "util" / "include",
        root / "src" / "sgl-kernel" / "csrc",
        flashinfer.resolve() / "include",
        flashinfer.resolve() / "include" / "gemm",
        flashinfer.resolve() / "csrc",
        turbomind.resolve(),
        turbomind.resolve() / "src",
    ]
    nvcc_flags = [
        "-DNDEBUG",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-std=c++17",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
    ]
    nvcc_flags_fp8 = [
        "-DFLASHINFER_ENABLE_FP8",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    sources = [
        "src/sgl-kernel/csrc/trt_reduce_internal.cu",
        "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
        "src/sgl-kernel/csrc/moe_align_kernel.cu",
        "src/sgl-kernel/csrc/int8_gemm_kernel.cu",
        "src/sgl-kernel/csrc/sampling_scaling_penalties.cu",
        "src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu",
        "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
        "src/sgl-kernel/csrc/rotary_embedding.cu",
        "src/sgl-kernel/csrc/fused_add_rms_norm.cu",
        "3rdparty/flashinfer/csrc/activation.cu",
        "3rdparty/flashinfer/csrc/bmm_fp8.cu",
        "3rdparty/flashinfer/csrc/group_gemm.cu",
        "3rdparty/flashinfer/csrc/norm.cu",
        "3rdparty/flashinfer/csrc/sampling.cu",
        "3rdparty/flashinfer/csrc/renorm.cu",
    ]

    enable_bf16 = os.getenv("SGL_KERNEL_ENABLE_BF16", "0") == "1"
    enable_fp8 = os.getenv("SGL_KERNEL_ENABLE_FP8", "0") == "1"
    enable_sm90a = os.getenv("SGL_KERNEL_ENABLE_SM90A", "0") == "1"
    cuda_version = _get_cuda_version()
    sm_version = _get_device_sm()

    if torch.cuda.is_available():
        if cuda_version >= (12, 0) and sm_version >= 90:
            nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
            sources.append("3rdparty/flashinfer/csrc/group_gemm_sm90.cu")
        if sm_version >= 90:
            nvcc_flags.extend(nvcc_flags_fp8)
        if sm_version >= 80:
            nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")
    else:
        # compilation environment without GPU
        if enable_sm90a:
            nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
            sources.append("3rdparty/flashinfer/csrc/group_gemm_sm90.cu")
        if enable_fp8:
            nvcc_flags.extend(nvcc_flags_fp8)
        if enable_bf16:
            nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")

    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    cxx_flags = ["-O3"]
    libraries = ["c10", "torch", "torch_python", "cuda"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ]

    setup(
        name="sgl-kernel",
        version=_get_version(),
        packages=find_packages(),
        package_dir={"": "src"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
elif is_hip():
    cutlass = root / "3rdparty" / "cutlass"
    flashinfer = root / "3rdparty" / "flashinfer"
    turbomind = root / "3rdparty" / "turbomind"
    include_dirs = [
        cutlass.resolve() / "include",
        cutlass.resolve() / "tools" / "util" / "include",
        root / "src" / "sgl-kernel" / "csrc",
        # TODO: enable the following 3rdparty libraries
        # flashinfer.resolve() / "include",
        # flashinfer.resolve() / "include" / "gemm",
        # flashinfer.resolve() / "csrc",
        # turbomind.resolve(),
        # turbomind.resolve() / "src",
    ]
    
    ck = root / "3rdparty" / "composable_kernel"
    ck_include_dirs = [
        ck.resolve() / "include",
        root / "src" / "sgl-kernel-amd" / "csrc",
    ]
    
    hipcc_flags = [
        "-D__HIP_PLATFORM_AMD__=1",
        "-DLEGACY_HIPBLAS_DIRECT",
        "--amdgpu-target=gfx90a,gfx940,gfx941,gfx942",
    ]
    
    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=[
                "src/sgl-kernel/csrc/moe_align_kernel.cu",
                "src/sgl-kernel-amd/csrc/ck/gemm_a8w8/gemm_a8w8_subblock.cu",
                "src/sgl-kernel-amd/csrc/sgl_kernel_amd_ops.cu",
            ]
            + glob.glob("src/sgl-kernel-amd/csrc/ck/gemm_a8w8/impl/*.cu"),
            include_dirs=ck_include_dirs,
            extra_compile_args={
                "nvcc": hipcc_flags
                + [
                    "-O3",
                    "-fPIC",
                ],
                "cxx": ["-O3"],
            },
            libraries=["hiprtc", "amdhip64", "c10", "torch", "torch_python"],
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        ),
    ]

    setup(
        name="sgl-kernel",
        version=_get_version(),
        packages=["sgl_kernel", "sgl_kernel.ops", "sgl_kernel.csrc"],
        package_dir={"sgl_kernel": "src/sgl-kernel-amd"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"],
    )
else:
    raise RuntimeError(
        "Neither CUDA nor ROCm environment detected. Set either CUDA_HOME or ROCM_PATH"
    )

_update_wheel_platform_tag()