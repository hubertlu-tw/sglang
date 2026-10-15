"""Compare fused all-reduce + residual add + RMSNorm backends on ROCm.

Rows compare, for each (hidden, tokens) shape:

* ``ca-2stage``: AITER CustomAllreduce fused AR+RMSNorm (2-stage), lossless.
* ``qr-fused``: AITER QuickReduce fused AR+RMSNorm with the requested codec.
* ``qr-unfused``: AITER QuickReduce all-reduce followed by a separate
  ``rmsnorm2d_fwd_with_add``, the path SGLang takes when the fused kernel is
  skipped.

Usage:
    torchrun --standalone --nproc_per_node=8 \
        benchmark/kernels/all_reduce/benchmark_quick_ar_rmsnorm_rocm.py \
        --codec INT4 --hidden 4096 8192
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

MB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec", default="INT4", choices=["FP", "FP8", "INT6", "INT4"]
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-size-mb", type=int, default=128)
    return parser.parse_args()


def reference(
    allreduced: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = allreduced.float() + residual.float()
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    out = residual_out * torch.rsqrt(variance + eps) * weight.float()
    return out, residual_out


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = torch.linalg.vector_norm(actual.float() - expected)
    return (err / torch.linalg.vector_norm(expected).clamp_min(1e-30)).item()


def time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end) / iters], device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.AVG)
    return elapsed.item()


def all_ranks(flag: bool) -> bool:
    t = torch.tensor([int(flag)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = args.codec
    os.environ["AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = str(args.max_size_mb)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    cpu_group = dist.new_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    import aiter
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce

    ca = CustomAllreduce(group=cpu_group, device=device, max_size=args.max_size_mb * MB)
    qr = QuickAllReduce(group=cpu_group, device=device)
    eps = 1e-6

    rows: List[Dict[str, object]] = []
    for hidden in args.hidden:
        weight = torch.randn(hidden, dtype=dtype, device=device) * 0.1 + 1.0
        for tokens in args.tokens:
            gen = torch.Generator(device=device).manual_seed(1234 + rank)
            x = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=gen)
            gen_res = torch.Generator(device=device).manual_seed(99)
            res = torch.randn(
                tokens, hidden, dtype=dtype, device=device, generator=gen_res
            )
            size = x.numel() * x.element_size()

            allreduced = x.float().clone()
            dist.all_reduce(allreduced)
            ref_out, ref_res = reference(allreduced, res, weight, eps)

            def run_ca() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
                return ca.custom_fused_ar_rms(x, res, weight, eps, use_1stage=False)

            def run_qr_fused() -> Tuple[torch.Tensor, torch.Tensor]:
                return qr.quick_all_reduce_rmsnorm(x, res, weight, eps, hidden)

            out_buf = torch.empty_like(x)
            res_buf = torch.empty_like(res)

            def run_qr_unfused() -> Tuple[torch.Tensor, torch.Tensor]:
                reduced = qr.quick_all_reduce(x)
                aiter.rmsnorm2d_fwd_with_add(
                    out_buf, reduced, res, res_buf, weight, eps
                )
                return out_buf, res_buf

            candidates = {
                "ca-2stage": (
                    run_ca,
                    all_ranks(ca.should_custom_ar(x) and size <= 64 * MB),
                ),
                "qr-fused": (
                    run_qr_fused,
                    all_ranks(
                        qr.should_quick_allreduce_rmsnorm(x, res, weight, hidden)
                    ),
                ),
                "qr-unfused": (run_qr_unfused, all_ranks(qr.should_quick_allreduce(x))),
            }

            row: Dict[str, object] = {"hidden": hidden, "tokens": tokens, "size": size}
            for name, (fn, eligible) in candidates.items():
                if not eligible:
                    continue
                out, res_out = fn()
                torch.cuda.synchronize()
                row[f"{name}:l2"] = max(
                    relative_l2(out, ref_out), relative_l2(res_out, ref_res)
                )
                row[f"{name}:ms"] = time_ms(fn, args.warmup, args.iters)
            rows.append(row)

    if rank == 0:
        names = ["ca-2stage", "qr-fused", "qr-unfused"]
        print(
            f"\n=== TP{world_size} {args.dtype} codec={args.codec} "
            "(eager, ms; l2 = max relative L2 of out/residual vs fp32 reference) ==="
        )
        header = f"{'hidden':>6} {'tokens':>6} {'size':>9}"
        for n in names:
            header += f" {n + ' ms':>14} {n + ' l2':>14}"
        header += f" {'fused/ca':>9} {'fused/unf':>9}"
        print(header)
        for row in rows:
            size = row["size"]
            size_s = f"{size / MB:.2f}MiB" if size >= MB else f"{size // 1024}KiB"
            line = f"{row['hidden']:>6} {row['tokens']:>6} {size_s:>9}"
            for n in names:
                ms = row.get(f"{n}:ms")
                l2 = row.get(f"{n}:l2")
                line += f" {ms:>14.4f}" if ms is not None else f" {'-':>14}"
                line += f" {l2:>14.2e}" if l2 is not None else f" {'-':>14}"
            fused = row.get("qr-fused:ms")
            ca_ms = row.get("ca-2stage:ms")
            unf = row.get("qr-unfused:ms")
            line += f" {fused / ca_ms:>8.2f}x" if fused and ca_ms else f" {'-':>9}"
            line += f" {fused / unf:>8.2f}x" if fused and unf else f" {'-':>9}"
            print(line)

    qr.close()
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
