"""Compare fused all-reduce + residual add + RMSNorm backends on ROCm.

For each (hidden, tokens) shape, the bf16-output variants are:

* ``ca-2stage``: AITER CustomAllreduce fused AR+RMSNorm (2-stage), lossless.
* ``qr-fused``: AITER QuickReduce fused AR+RMSNorm with the requested codec.
* ``qr-unfused``: AITER QuickReduce all-reduce followed by a separate
  ``rmsnorm2d_fwd_with_add``.

With ``--quant`` the variants feed a per-1x128 FP8 consumer instead:

* ``ca-pgq``: AITER CustomAllreduce fused AR+RMSNorm+per-group FP8 quant.
* ``ca-2stage+q``: CustomAllreduce fused AR+RMSNorm, then per-1x128 quant.
* ``qr-fused+q``: QuickReduce fused AR+RMSNorm, then per-1x128 quant.
* ``qr-unfused+q``: QuickReduce all-reduce, RMSNorm, then per-1x128 quant.

``--mode graph`` times CUDA-graph replays (decode-style); ``eager`` times
direct launches (prefill-style).

Usage:
    torchrun --standalone --nproc_per_node=8 \
        benchmark/kernels/all_reduce/benchmark_quick_ar_rmsnorm_rocm.py \
        --codec INT4 --hidden 4096 8192 --mode eager graph
"""

import argparse
import os
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

MB = 1024 * 1024
GROUP_SIZE = 128


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
    parser.add_argument(
        "--mode", nargs="+", default=["eager"], choices=["eager", "graph"]
    )
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=10)
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


def dequant(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = fp8.shape
    scale = scale.reshape(m, k // GROUP_SIZE)
    grouped = fp8.float().reshape(m, k // GROUP_SIZE, GROUP_SIZE)
    return (grouped * scale.float().unsqueeze(-1)).reshape(m, k)


def all_ranks(flag: bool) -> bool:
    t = torch.tensor([int(flag)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def reduce_avg(value: float) -> float:
    t = torch.tensor([value], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def time_eager(fn: Callable[[], object], warmup: int, iters: int) -> float:
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
    return reduce_avg(start.elapsed_time(end) / iters)


def time_graph(
    fn: Callable[[], object], ca, calls: int, warmup: int, iters: int
) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                fn()
    torch.cuda.synchronize()
    for _ in range(max(1, warmup // calls)):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    replays = max(1, iters // calls)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return reduce_avg(start.elapsed_time(end) / (replays * calls))


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
    pg_quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    fp8 = aiter.dtypes.fp8
    eps = 1e-6

    names = (
        ["ca-pgq", "ca-2stage+q", "qr-fused+q", "qr-unfused+q"]
        if args.quant
        else ["ca-2stage", "qr-fused", "qr-unfused"]
    )
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

            out_buf = torch.empty_like(x)
            res_buf = torch.empty_like(res)

            def ca_2stage() -> Tuple[torch.Tensor, torch.Tensor]:
                return ca.custom_fused_ar_rms(x, res, weight, eps, use_1stage=False)

            def qr_fused() -> Tuple[torch.Tensor, torch.Tensor]:
                return qr.quick_all_reduce_rmsnorm(x, res, weight, eps, hidden)

            def qr_unfused() -> Tuple[torch.Tensor, torch.Tensor]:
                reduced = qr.quick_all_reduce(x)
                aiter.rmsnorm2d_fwd_with_add(
                    out_buf, reduced, res, res_buf, weight, eps
                )
                return out_buf, res_buf

            def with_quant(fn):
                def run():
                    out, res_out = fn()
                    q, scale = pg_quant(out, quant_dtype=fp8, transpose_scale=False)
                    return q, scale, res_out

                return run

            def ca_pgq():
                q, res_out, scale = ca.custom_fused_ar_rms_per_group_quant(
                    x, res, weight, eps, GROUP_SIZE, use_1stage=False
                )
                return q, scale, res_out

            ca_ok = all_ranks(ca.should_custom_ar(x) and size <= 64 * MB)
            qr_fused_ok = all_ranks(
                qr.should_quick_allreduce_rmsnorm(x, res, weight, hidden)
            )
            qr_ok = all_ranks(qr.should_quick_allreduce(x))
            candidates = {
                "ca-2stage": (ca_2stage, ca_ok),
                "qr-fused": (qr_fused, qr_fused_ok),
                "qr-unfused": (qr_unfused, qr_ok),
                "ca-pgq": (ca_pgq, ca_ok),
                "ca-2stage+q": (with_quant(ca_2stage), ca_ok),
                "qr-fused+q": (with_quant(qr_fused), qr_fused_ok),
                "qr-unfused+q": (with_quant(qr_unfused), qr_ok),
            }

            for mode in args.mode:
                row: Dict[str, object] = {
                    "mode": mode,
                    "hidden": hidden,
                    "tokens": tokens,
                    "size": size,
                }
                for name in names:
                    fn, eligible = candidates[name]
                    if not eligible:
                        continue
                    result = fn()
                    torch.cuda.synchronize()
                    if args.quant:
                        q, scale, res_out = result
                        out = dequant(q, scale)
                    else:
                        out, res_out = result
                    row[f"{name}:l2"] = max(
                        relative_l2(out, ref_out), relative_l2(res_out, ref_res)
                    )
                    if mode == "eager":
                        row[f"{name}:ms"] = time_eager(fn, args.warmup, args.iters)
                    else:
                        row[f"{name}:ms"] = time_graph(
                            fn, ca, args.graph_calls, args.warmup, args.iters
                        )
                rows.append(row)

    if rank == 0:
        base = names[0]
        for mode in args.mode:
            print(
                f"\n=== TP{world_size} {args.dtype} codec={args.codec} mode={mode} "
                f"(ms; l2 = max relative L2 of out/residual vs fp32 reference; "
                f"ratio = variant / {base}) ==="
            )
            header = f"{'hidden':>6} {'tokens':>6} {'size':>9}"
            for n in names:
                header += f" {n + ' ms':>16} {'l2':>8}"
            for n in names[1:]:
                header += f" {n + '/' + base:>22}"
            print(header)
            for row in rows:
                if row["mode"] != mode:
                    continue
                size = row["size"]
                size_s = f"{size / MB:.2f}MiB" if size >= MB else f"{size // 1024}KiB"
                line = f"{row['hidden']:>6} {row['tokens']:>6} {size_s:>9}"
                for n in names:
                    ms = row.get(f"{n}:ms")
                    l2 = row.get(f"{n}:l2")
                    line += f" {ms:>16.4f}" if ms is not None else f" {'-':>16}"
                    line += f" {l2:>8.1e}" if l2 is not None else f" {'-':>8}"
                base_ms = row.get(f"{base}:ms")
                for n in names[1:]:
                    ms = row.get(f"{n}:ms")
                    line += (
                        f" {ms / base_ms:>21.2f}x"
                        if ms is not None and base_ms
                        else f" {'-':>22}"
                    )
                print(line)

    qr.close()
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
