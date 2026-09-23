"""
Benchmark fused and unfused all-reduce + RMSNorm on AMD.

Two entry points:

* Legacy single-dispatch run (used by test_aiter_allreduce_fusion_amd.py).
  Times the public fused API against SGLang's unfused all-reduce plus RMSNorm.
* Fusion study (``--sweep-output-dir``). Times each candidate directly, with
  the slowest rank's average per-iteration GPU-event time, and writes a
  markdown report.

The study covers the residual widths of the models on this machine:

* 4096: Qwen3.5-397B (``/data/models/amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2``)
* 7168: DeepSeek-V4-Pro and DeepSeek-R1-MXFP4

Fused QuickReduce RMSNorm also requires the bf16/fp16 row to divide a 32 KiB
tile. 4096 does. 7168 does not, so DeepSeek stays on custom all-reduce fusion.

Usage:
  python benchmark/kernels/all_reduce/benchmark_fused_ar_rms_amd.py \\
    --sweep-output-dir benchmark/kernels/all_reduce/results/fused_ar_rms_int4_YYYYMMDD
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.environ import envs
from sglang.test.test_utils import publish_build_topology

Shape = Tuple[int, int]

# 128 DeepSeek tokens of bf16 residual used to be the 1-stage cutoff.
# The 2026-09-22 study replaced that rule. See stage_limit_bytes.
DEEPSEEK_HIDDEN = 7168
QWEN_HIDDEN = 4096

STUDY_PATHS = (
    "fused_qr_int4",
    "fused_car_1stage",
    "fused_car_2stage",
    "split_car",
    "split_qr_int4",
    "fused_api",
)
PATH_LABELS = {
    "fused_qr_int4": "Fused QR INT4",
    "fused_car_1stage": "Fused CAR 1-stage",
    "fused_car_2stage": "Fused CAR 2-stage",
    "split_car": "Split CAR",
    "split_qr_int4": "Split QR INT4",
    "fused_api": "Fused API",
}


def stage_limit_bytes(world_size: int) -> int:
    """Match GroupCoordinator._fused_ar_rmsnorm_use_1stage.

    TP2 keeps 1-stage across the custom-AR window. TP4 and TP8 keep it
    through 128 KiB. The 2026-09-22 study measured the previous
    ``128 * 7168 * 2 // world_size`` rule and is what motivated this.
    """
    if world_size <= 2:
        return 64 * 1024 * 1024
    return 128 * 1024


def policy_uses_1stage(nbytes: int, world_size: int) -> bool:
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        return bool(envs.SGLANG_USE_1STAGE_ALLREDUCE.get())
    return nbytes <= stage_limit_bytes(world_size)


def _quickreduce_regime() -> str:
    return os.environ.get(
        "AITER_QUICK_REDUCE_QUANTIZATION",
        os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE"),
    )


def _quickreduce_cast_bf16() -> bool:
    return bool(
        int(
            os.environ.get(
                "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16",
                os.environ.get("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1"),
            )
        )
    )


def parse_shapes(raw: str) -> List[Shape]:
    shapes: List[Shape] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if "x" not in item:
            raise ValueError(f"Invalid shape '{item}', expected MxN format.")
        m_str, n_str = item.split("x", 1)
        m = int(m_str)
        n = int(n_str)
        if m <= 0 or n <= 0:
            raise ValueError(f"Invalid shape '{item}', both dims must be positive.")
        shapes.append((m, n))
    if not shapes:
        raise ValueError("Empty shape list is not allowed.")
    return shapes


def parse_int_list(raw: str) -> List[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Expected positive integers, got {raw!r}")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _shape_bytes(shape: Shape, dtype: torch.dtype) -> int:
    m, n = shape
    return m * n * torch.empty((), dtype=dtype).element_size()


def _barrier(device: torch.device) -> None:
    try:
        dist.barrier(device_ids=[device.index])
    except TypeError:
        dist.barrier()


def _all_true(value: bool, device: torch.device) -> bool:
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(int(tensor.item()))


def _max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def slowest_rank_average_us(times_us: Sequence[float], device: torch.device) -> float:
    """Match benchmark_rocm.py: mean on each rank, then the slowest rank."""
    stats = torch.tensor(
        [
            sum(times_us) / len(times_us),
            min(times_us),
            max(times_us),
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats)
    return float(torch.stack(gathered)[:, 0].max().item())


def time_per_iteration_us(
    fn: Callable[[], object], warmup: int, iters: int
) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def check_close(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    dtype: torch.dtype,
    world_size: int,
    quantized: bool,
) -> Tuple[bool, str, float, float]:
    if quantized:
        rtol, atol = 0.5 * world_size, 1.25 * world_size
    elif dtype == torch.bfloat16:
        rtol, atol = 2e-2, 1.25e-1
    else:
        rtol, atol = 1e-2, 2e-2
    diff = (candidate.float() - reference.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    try:
        torch.testing.assert_close(candidate, reference, rtol=rtol, atol=atol)
        return True, "PASS", max_diff, mean_diff
    except AssertionError:
        return (
            False,
            f"FAIL(max={max_diff:.6f},mean={mean_diff:.6f})",
            max_diff,
            mean_diff,
        )


def _make_inputs(
    shape: Shape,
    dtype: torch.dtype,
    seed: int,
    rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = shape
    torch.manual_seed(seed + rank * 17)
    values = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
    residual = values.clone()
    weight = torch.randn((n,), dtype=torch.float32, device=device).to(dtype)
    return values, residual, weight


def _rccl_reference(
    values: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduced = values.clone()
    dist.all_reduce(reduced, group=get_tp_group().device_group)
    residual_out = reduced + residual
    out = F.rms_norm(
        residual_out,
        (residual_out.shape[-1],),
        weight,
        eps,
    )
    return out, residual_out


def _split_dispatched(
    values: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduced = tensor_model_parallel_all_reduce(values.clone())
    residual_out = reduced + residual
    out = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, eps)
    return out, residual_out


def _as_pair(result: object) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if result is None:
        return None
    if isinstance(result, tuple) and len(result) >= 2:
        out, residual_out = result[0], result[1]
        if out is None or residual_out is None:
            return None
        return out, residual_out
    return None


def _call_path(
    name: str,
    values: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    group = get_tp_group()
    try:
        if name == "fused_qr_int4":
            qr_comm = group.qr_comm
            if (
                qr_comm is None
                or getattr(qr_comm, "disabled", True)
                or not qr_comm.should_quick_allreduce_rmsnorm(
                    values, residual, weight, weight.numel()
                )
            ):
                return None
            return _as_pair(
                qr_comm.quick_all_reduce_rmsnorm(
                    values, residual, weight, eps, weight.numel()
                )
            )
        if name == "fused_car_1stage":
            return _call_car(values, residual, weight, eps, True)
        if name == "fused_car_2stage":
            return _call_car(values, residual, weight, eps, False)
        if name == "split_car":
            ca_comm = group.ca_comm
            if ca_comm is None or getattr(ca_comm, "disabled", True):
                return None
            reduced = ca_comm.custom_all_reduce(values)
            if reduced is None:
                return None
            residual_out = reduced + residual
            out = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, eps)
            return out, residual_out
        if name == "split_qr_int4":
            qr_comm = group.qr_comm
            if (
                qr_comm is None
                or getattr(qr_comm, "disabled", True)
                or not qr_comm.should_quick_allreduce(values)
            ):
                return None
            reduced = qr_comm.quick_all_reduce(values)
            residual_out = reduced + residual
            out = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, eps)
            return out, residual_out
        if name == "fused_api":
            return _as_pair(
                tensor_model_parallel_fused_allreduce_rmsnorm(
                    values, residual, weight, eps
                )
            )
    except Exception as exc:
        if dist.get_rank() == 0:
            print(f"[{name}] unavailable: {exc}", file=sys.stderr)
        return None
    raise ValueError(f"Unknown path {name}")


def _call_car(
    values: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_1stage: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ca_comm = get_tp_group().ca_comm
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        return None
    if not ca_comm.should_custom_ar(values):
        return None
    return _as_pair(
        ca_comm.custom_fused_ar_rms(values, residual, weight, eps, use_1stage)
    )


def _quantized_path(name: str) -> bool:
    return name in ("fused_qr_int4", "split_qr_int4")


def _selected_fused_backend(
    values: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> str:
    group = get_tp_group()
    use_1stage = group._fused_ar_rmsnorm_use_1stage(values)
    if group._quick_allreduce_rmsnorm_eligible(values, residual, weight, use_1stage):
        return "quick"
    ca_comm = group.ca_comm
    if ca_comm is not None and not getattr(ca_comm, "disabled", True):
        return "custom"
    return "fallback"


def _init_distributed(dtype_name: str) -> Tuple[int, int, torch.device, torch.dtype]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    dtype = dtype_from_name(dtype_name)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(world_rank=rank, tp_size=world_size)
    initialize_model_parallel()
    return rank, world_size, device, dtype


def _shutdown() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def measure_path(
    name: str,
    values: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    warmup: int,
    iters: int,
    mode: str,
    device: torch.device,
) -> Dict[str, object]:
    world_size = dist.get_world_size()
    quantized = _quantized_path(name)
    if name == "fused_api":
        backend = _selected_fused_backend(values, residual, weight)
        regime = _quickreduce_regime()
        quantized = backend == "quick" and regime not in ("", "NONE", "FP")
    probe = _call_path(name, values.clone(), residual.clone(), weight, eps)
    available = _all_true(probe is not None, device)
    result: Dict[str, object] = {
        "path": name,
        "available": available,
        "latency_us": None,
        "correctness_ok": True,
        "correctness_detail": "SKIP",
        "out_max_abs": 0.0,
        "residual_max_abs": 0.0,
    }
    if not available:
        result["correctness_detail"] = "SKIP(unavailable)"
        return result

    ref_out, ref_residual = _rccl_reference(values, residual, weight, eps)
    if mode == "eager":

        def invoke() -> None:
            _call_path(name, values, residual, weight, eps)

        check_out, check_residual = _call_path(
            name, values.clone(), residual.clone(), weight, eps
        )
    else:
        graph_values = values.clone()
        graph_residual = residual.clone()
        graph_holder: Dict[str, torch.Tensor] = {}
        with graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                captured = _call_path(name, graph_values, graph_residual, weight, eps)
                if captured is None:
                    raise RuntimeError(f"{name} returned None during graph capture")
                graph_holder["out"], graph_holder["residual"] = captured

        def invoke() -> None:
            graph.replay()

        invoke()
        torch.cuda.synchronize()
        check_out, check_residual = graph_holder["out"], graph_holder["residual"]

    assert check_out is not None and check_residual is not None
    out_ok, out_detail, out_max, _ = check_close(
        check_out, ref_out, values.dtype, world_size, quantized
    )
    res_ok, res_detail, res_max, _ = check_close(
        check_residual, ref_residual, values.dtype, world_size, quantized
    )
    result["correctness_ok"] = _all_true(out_ok and res_ok, device)
    result["correctness_detail"] = f"out={out_detail}, residual={res_detail}"
    result["out_max_abs"] = _max_across_ranks(out_max, device)
    result["residual_max_abs"] = _max_across_ranks(res_max, device)

    _barrier(device)
    times_us = time_per_iteration_us(invoke, warmup, iters)
    result["latency_us"] = slowest_rank_average_us(times_us, device)
    return result


def run_study_worker(args: argparse.Namespace) -> None:
    rank, world_size, device, dtype = _init_distributed(args.dtype)
    tokens = parse_int_list(args.tokens)
    hiddens = parse_int_list(args.hiddens)
    modes: Sequence[str] = ("eager", "graph") if args.mode == "both" else (args.mode,)
    rows: List[Dict[str, object]] = []
    if rank == 0:
        print(
            f"Study world_size={world_size} dtype={dtype} regime={_quickreduce_regime()} "
            f"cast_bf16={_quickreduce_cast_bf16()} warmup={args.warmup} iters={args.iters} "
            f"stage_limit_bytes={stage_limit_bytes(world_size)}",
            flush=True,
        )
    for mode in modes:
        for hidden in hiddens:
            for token_count in tokens:
                shape = (token_count, hidden)
                values, residual, weight = _make_inputs(
                    shape, dtype, args.seed, rank, device
                )
                nbytes = _shape_bytes(shape, dtype)
                backend = _selected_fused_backend(values, residual, weight)
                stage = "1stage" if policy_uses_1stage(nbytes, world_size) else "2stage"
                if rank == 0:
                    print(
                        f"\n{mode} {token_count}x{hidden} bytes={nbytes} "
                        f"policy={stage} api={backend}",
                        flush=True,
                    )
                for path in STUDY_PATHS:
                    measured = measure_path(
                        path,
                        values,
                        residual,
                        weight,
                        args.eps,
                        args.warmup,
                        args.iters,
                        mode,
                        device,
                    )
                    if rank == 0:
                        latency = measured["latency_us"]
                        latency_text = (
                            f"{float(latency):.1f}" if latency is not None else "—"
                        )
                        print(
                            f"  {path:18} {latency_text:>10} us  "
                            f"ok={measured['correctness_ok']} {measured['correctness_detail']}",
                            flush=True,
                        )
                        rows.append(
                            {
                                "mode": mode,
                                "world_size": world_size,
                                "dtype": str(dtype),
                                "hidden": hidden,
                                "tokens": token_count,
                                "shape": f"{token_count}x{hidden}",
                                "bytes_per_rank": nbytes,
                                "path": path,
                                "latency_us": (
                                    ""
                                    if measured["latency_us"] is None
                                    else f"{float(measured['latency_us']):.4f}"
                                ),
                                "available": measured["available"],
                                "correctness_ok": measured["correctness_ok"],
                                "correctness_detail": measured["correctness_detail"],
                                "out_max_abs": measured["out_max_abs"],
                                "residual_max_abs": measured["residual_max_abs"],
                                "policy_stage": stage,
                                "policy_backend": backend,
                                "quickreduce_regime": _quickreduce_regime(),
                                "warmup": args.warmup,
                                "iters": args.iters,
                            }
                        )
                    _barrier(device)
    if rank == 0 and args.csv_out:
        _write_csv(args.csv_out, rows)
        print(f"Saved CSV to {args.csv_out}", flush=True)
    _barrier(device)
    _shutdown()


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_legacy_worker(args: argparse.Namespace) -> None:
    """One dispatched fused measurement per shape. Preserves the CI CSV schema."""
    rank, world_size, device, dtype = _init_distributed(args.dtype)
    if args.force_stage is not None:
        envs.SGLANG_USE_1STAGE_ALLREDUCE.set(args.force_stage == "1")
    prefill_shapes = parse_shapes(args.prefill_shapes)
    decode_shapes = parse_shapes(args.decode_shapes)
    modes: Sequence[str] = ("eager", "graph") if args.mode == "both" else (args.mode,)
    rows: List[Dict[str, object]] = []
    for mode in modes:
        shapes = prefill_shapes if mode == "eager" else decode_shapes
        for shape in shapes:
            values, residual, weight = _make_inputs(
                shape, dtype, args.seed, rank, device
            )
            fused = _legacy_pair(
                "fused",
                lambda: _as_pair(
                    tensor_model_parallel_fused_allreduce_rmsnorm(
                        values, residual, weight, args.eps
                    )
                ),
                lambda: _split_dispatched(values, residual, weight, args.eps),
                values,
                residual,
                weight,
                args.eps,
                args.warmup,
                args.iters,
                mode,
                device,
            )
            if rank != 0:
                continue
            fused_us = fused["fused_us"]
            split_us = fused["split_us"]
            rows.append(
                {
                    "mode": mode,
                    "selected_backend": fused["selected_backend"],
                    "shape": f"{shape[0]}x{shape[1]}",
                    "m": shape[0],
                    "n": shape[1],
                    "bytes_per_rank": _shape_bytes(shape, dtype),
                    "split_p50_us": split_us if split_us is not None else "",
                    "fused_p50_us": fused_us if fused_us is not None else "",
                    "speedup_split_over_fused": (
                        split_us / fused_us
                        if split_us is not None and fused_us not in (None, 0)
                        else ""
                    ),
                    "fused_available": fused["fused_available"],
                    "correctness_ok": fused["correctness_ok"],
                    "correctness_detail": fused["correctness_detail"],
                    "out_max_abs": fused["out_max_abs"],
                    "out_mean_abs": fused["out_mean_abs"],
                    "residual_max_abs": fused["residual_max_abs"],
                    "residual_mean_abs": fused["residual_mean_abs"],
                    "dtype": str(dtype),
                    "world_size": world_size,
                    "quickreduce_regime": _quickreduce_regime(),
                    "cast_bf16_to_fp16": _quickreduce_cast_bf16(),
                    "force_stage": args.force_stage or "auto",
                    "residual_mode": args.residual_mode,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "aggregation": "slowest_rank_average",
                }
            )
    if rank == 0 and args.csv_out:
        _write_csv(args.csv_out, rows)
        print(f"Saved CSV to {args.csv_out}", flush=True)
    failed = any(not row["correctness_ok"] for row in rows) if rank == 0 else False
    failed_tensor = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    dist.broadcast(failed_tensor, src=0)
    _barrier(device)
    _shutdown()
    if args.fail_on_correctness and bool(failed_tensor.item()):
        raise RuntimeError("At least one fused result failed correctness")


def _legacy_pair(
    _label: str,
    fused_fn: Callable[[], Optional[Tuple[torch.Tensor, torch.Tensor]]],
    split_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    values: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    warmup: int,
    iters: int,
    mode: str,
    device: torch.device,
) -> Dict[str, object]:
    backend = _selected_fused_backend(values, residual, weight)
    ref_out, ref_residual = _rccl_reference(values, residual, weight, eps)
    quantized = _quickreduce_regime() not in ("", "NONE")

    def time_callable(fn: Callable[[], object]) -> float:
        _barrier(device)
        return slowest_rank_average_us(time_per_iteration_us(fn, warmup, iters), device)

    if mode == "graph":
        split_values = values.clone()
        split_residual = residual.clone()
        with graph_capture() as capture:
            split_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(split_graph, stream=capture.stream):
                split_fn_values = _split_dispatched(
                    split_values, split_residual, weight, eps
                )
                del split_fn_values
        split_us = time_callable(split_graph.replay)
    else:
        split_us = time_callable(
            lambda: _split_dispatched(values, residual, weight, eps)
        )

    probe = fused_fn()
    fused_available = _all_true(probe is not None, device)
    fused_us: Optional[float] = None
    correctness_ok = True
    correctness_detail = "SKIP(fused_unavailable)"
    out_max = out_mean = res_max = res_mean = 0.0
    if fused_available:
        if mode == "graph":
            fused_values = values.clone()
            fused_residual = residual.clone()
            holder: Dict[str, torch.Tensor] = {}
            with graph_capture() as capture:
                fused_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(fused_graph, stream=capture.stream):
                    captured = fused_fn()
                    if captured is None:
                        raise RuntimeError("fused path returned None during capture")
                    holder["out"], holder["residual"] = captured

            def replay() -> None:
                fused_graph.replay()

            replay()
            torch.cuda.synchronize()
            check_out, check_residual = holder["out"], holder["residual"]
            fused_us = time_callable(replay)
        else:
            check_out, check_residual = fused_fn()
            assert check_out is not None and check_residual is not None
            fused_us = time_callable(lambda: fused_fn())
        out_ok, out_detail, out_max, out_mean = check_close(
            check_out, ref_out, values.dtype, dist.get_world_size(), quantized
        )
        res_ok, res_detail, res_max, res_mean = check_close(
            check_residual, ref_residual, values.dtype, dist.get_world_size(), quantized
        )
        correctness_ok = _all_true(out_ok and res_ok, device)
        correctness_detail = f"out={out_detail}, residual={res_detail}"
    return {
        "split_us": split_us,
        "fused_us": fused_us,
        "fused_available": fused_available,
        "selected_backend": backend,
        "correctness_ok": correctness_ok,
        "correctness_detail": correctness_detail,
        "out_max_abs": _max_across_ranks(out_max, device),
        "out_mean_abs": _max_across_ranks(out_mean, device),
        "residual_max_abs": _max_across_ranks(res_max, device),
        "residual_mean_abs": _max_across_ranks(res_mean, device),
    }


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.sweep_output_dir)
    csv_dir = output_dir / "csv"
    log_dir = output_dir / "logs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tp_values = parse_int_list(args.sweep_tp)
    jobs = [(tp, "both") for tp in tp_values]
    for index, (tp, mode) in enumerate(jobs, start=1):
        stem = f"tp{tp}_{args.dtype}"
        csv_path = csv_dir / f"{stem}.csv"
        log_path = log_dir / f"{stem}.log"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={tp}",
            str(Path(__file__).resolve()),
            "--study",
            "--dtype",
            args.dtype,
            "--mode",
            mode,
            "--hiddens",
            args.hiddens,
            "--tokens",
            args.tokens,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--eps",
            str(args.eps),
            "--seed",
            str(args.seed),
            "--csv-out",
            str(csv_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
        env["AITER_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
        env.pop("SGLANG_USE_1STAGE_ALLREDUCE", None)
        env.pop("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", None)
        print(f"[{index}/{len(jobs)}] {stem}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            try:
                returncode = process.wait(timeout=args.worker_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise RuntimeError(f"{stem} timed out; see {log_path}") from None
        if returncode != 0:
            raise RuntimeError(f"{stem} failed with exit {returncode}; see {log_path}")
    report_path = (
        Path(args.report_out)
        if args.report_out
        else output_dir / "fused_ar_rms_report.md"
    )
    write_markdown_report(csv_dir, report_path)
    print(f"Sweep complete: {report_path}")


def _fmt_us(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def _load_study_rows(csv_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _latency(row: Dict[str, str]) -> Optional[float]:
    text = row.get("latency_us", "")
    if text == "" or row.get("available") not in ("True", "true", "1"):
        return None
    return float(text)


def write_markdown_report(csv_dir: Path, report_path: Path) -> None:
    rows = _load_study_rows(csv_dir)
    if not rows:
        raise RuntimeError(f"No study CSV rows in {csv_dir}")
    groups: Dict[Tuple[int, str, int], List[Dict[str, str]]] = {}
    for row in rows:
        key = (int(row["world_size"]), row["mode"], int(row["hidden"]))
        groups.setdefault(key, []).append(row)

    lines: List[str] = [
        "# Fused all-reduce + RMSNorm benchmark",
        "",
        "Per-call GPU-event latency in microseconds. The primary metric is the slowest rank's mean of per-iteration times. `—` means the path is ineligible or did not return a result.",
        "",
        "QuickReduce rows use `AITER_QUICK_REDUCE_QUANTIZATION=INT4` with BF16 cast to FP16. Custom all-reduce does not read that variable; it is timed in the same process so both communicators are live.",
        "",
        "## Models",
        "",
        "| Hidden | Model | Fused QR RMSNorm |",
        "| --- | --- | --- |",
        "| 4096 | Qwen3.5-397B-A17B (`/data/models/amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2`) | Eligible. Row is 8192 bytes, which divides the 32 KiB QuickReduce tile. |",
        "| 7168 | DeepSeek-V4-Pro (`/data2/deepseek-ai/DeepSeek-V4-Pro`) and DeepSeek-R1-MXFP4 (`/data2/amd/DeepSeek-R1-MXFP4-Preview`) | Ineligible at every token count. Row is 14336 bytes, and 32768 % 14336 != 0. |",
        "",
        "The fused op is residual-stream all-reduce plus RMSNorm after row-parallel `o_proj` and dense `down_proj`. Its width is `hidden_size`, not the MoE intermediate size.",
        "",
        "Other residual widths that divide 16384 (so the bf16 row divides 32 KiB) can use fused QuickReduce. 8192 can. 2880, 5120, and 6144 cannot. Those widths are correctness-tested elsewhere and are not in this timing sweep.",
        "",
        "## Summary",
        "",
        "| TP | Mode | Hidden | Rows | CAR 1-stage faster | Policy mismatches | Fused QR faster than best CAR | API matches best |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cells_for(
        group_rows: List[Dict[str, str]],
    ) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
        table: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
        for row in group_rows:
            table.setdefault((row["tokens"], row["shape"]), {})[row["path"]] = _latency(
                row
            )
        return table

    summary_notes: List[str] = []
    ordered = sorted(groups)
    for tp, mode, hidden in ordered:
        group_rows = groups[(tp, mode, hidden)]
        table = cells_for(group_rows)
        car_1_wins = 0
        mismatches = 0
        qr_wins = 0
        qr_compared = 0
        api_matches = 0
        api_compared = 0
        limit = stage_limit_bytes(tp)
        for (token_text, _shape), paths in table.items():
            one = paths.get("fused_car_1stage")
            two = paths.get("fused_car_2stage")
            qr = paths.get("fused_qr_int4")
            api = paths.get("fused_api")
            nbytes = int(token_text) * hidden * 2
            if one is not None and two is not None:
                faster_is_1 = one <= two
                if faster_is_1:
                    car_1_wins += 1
                policy_1 = nbytes <= limit
                if policy_1 != faster_is_1:
                    mismatches += 1
                best_car = min(one, two)
            else:
                best_car = one if one is not None else two
            if qr is not None and best_car is not None:
                qr_compared += 1
                if qr < best_car:
                    qr_wins += 1
            candidates = [
                value for value in paths.values() if value is not None and value > 0
            ]
            # API is one of the candidates; compare API to the best explicit kernel.
            explicit = [
                paths.get(name)
                for name in (
                    "fused_qr_int4",
                    "fused_car_1stage",
                    "fused_car_2stage",
                    "split_car",
                    "split_qr_int4",
                )
            ]
            explicit_values = [value for value in explicit if value is not None]
            if api is not None and explicit_values:
                api_compared += 1
                best_explicit = min(explicit_values)
                if api <= best_explicit * 1.02:
                    api_matches += 1
            del candidates
        lines.append(
            f"| TP{tp} | {mode} | {hidden} | {len(table)} | {car_1_wins}/{len(table)} | {mismatches}/{len(table)} | {qr_wins}/{qr_compared or 0} | {api_matches}/{api_compared or 0} |"
        )
        summary_notes.append(
            f"TP{tp} {mode} hidden {hidden}: current 1-stage cutoff is {limit} bytes "
            f"({limit / (hidden * 2):.1f} tokens)."
        )

    lines.extend(["", "## Current 1-stage cutoff", ""])
    lines.append(
        "The runtime 1-stage rule is: always at TP2, and only through 128 KiB at TP4 and TP8. "
        "It comes from the MI355X study of hidden 4096 and 7168. "
        "A policy mismatch means that rule picks the slower custom-allreduce stage."
    )
    lines.append("")
    for note in summary_notes:
        lines.append(f"- {note}")
    lines.append("")

    for tp, mode, hidden in ordered:
        group_rows = groups[(tp, mode, hidden)]
        model = "Qwen3.5" if hidden == QWEN_HIDDEN else "DeepSeek-V4 / R1"
        limit = stage_limit_bytes(tp)
        lines.extend(
            [
                f"## TP{tp} · {mode} · hidden {hidden} ({model})",
                "",
                f"Aggregation: `slowest_rank_average` · warmups: {group_rows[0]['warmup']} · timed iterations: {group_rows[0]['iters']} · 1-stage cutoff: {limit} bytes",
                "",
                "### Latency (us)",
                "",
                "| Tokens | Bytes | Policy | "
                + " | ".join(PATH_LABELS[name] for name in STUDY_PATHS)
                + " | Fastest |",
                "| --- | --- | --- | "
                + " | ".join("---" for _ in STUDY_PATHS)
                + " | --- |",
            ]
        )
        by_token: Dict[int, Dict[str, Optional[float]]] = {}
        policy_by_token: Dict[int, str] = {}
        correctness_fail = False
        for row in group_rows:
            token_count = int(row["tokens"])
            by_token.setdefault(token_count, {})[row["path"]] = _latency(row)
            policy_by_token[token_count] = row["policy_stage"]
            if row["available"] in ("True", "true") and row["correctness_ok"] not in (
                "True",
                "true",
            ):
                correctness_fail = True
        for token_count in sorted(by_token):
            paths = by_token[token_count]
            nbytes = token_count * hidden * 2
            ranked = [(name, paths.get(name)) for name in STUDY_PATHS]
            available = [(name, value) for name, value in ranked if value is not None]
            fastest = min(available, key=lambda item: item[1])[0] if available else "—"
            fastest_label = PATH_LABELS.get(fastest, fastest)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(token_count),
                        str(nbytes),
                        policy_by_token[token_count],
                        *[_fmt_us(paths.get(name)) for name in STUDY_PATHS],
                        fastest_label,
                    ]
                )
                + " |"
            )
        lines.append("")
        if correctness_fail:
            lines.append(
                "At least one available path failed its RCCL tolerance on this section."
            )
        else:
            lines.append(
                "Every available path passed its RCCL tolerance on this section."
            )
        lines.append("")

    lines.extend(
        [
            "## How to read a cutoff",
            "",
            "- Fused CAR 1-stage versus 2-stage is the custom-allreduce stage decision. The policy column is what the current byte cutoff would launch.",
            "- Fused QR INT4 is a lossy codec. It can only run when the shape passes AITER's size floor and the 32 KiB row-tile rule. On hidden 7168 that tile rule fails, so the column stays empty.",
            "- Split CAR is custom all-reduce plus a separate PyTorch RMSNorm, with QuickReduce not used for that call.",
            "- Split QR INT4 is plain QuickReduce plus a separate PyTorch RMSNorm. Its size floor is coarser than fused QuickReduce's tile rule, so it can appear on hidden 7168.",
            "- Fused API is `tensor_model_parallel_fused_allreduce_rmsnorm` with the current QR-before-CAR rule and the current stage cutoff. It should match the faster legal fused kernel. A gap means the cutoff or the QR gate is leaving time on the table.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused and unfused all-reduce + RMSNorm on AMD."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "float16", "bfloat16"],
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--residual-mode", type=str, default="self", choices=["self", "random", "zero"]
    )
    parser.add_argument("--prefill-shapes", type=str, default="2048x4096,2048x7168")
    parser.add_argument(
        "--decode-shapes", type=str, default="1x4096,16x4096,1x7168,16x7168"
    )
    parser.add_argument(
        "--hiddens", type=str, default=f"{QWEN_HIDDEN},{DEEPSEEK_HIDDEN}"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="1,8,16,24,32,48,64,96,128,256,512,1024,2048,4096",
        help="Token counts swept for every hidden size in study mode.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--force-stage", type=str, choices=["1", "2"], default=None)
    parser.add_argument("--require-quick-rmsnorm", action="store_true")
    parser.add_argument("--fail-on-correctness", action="store_true")
    parser.add_argument(
        "--mode", type=str, default="both", choices=["eager", "graph", "both"]
    )
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument(
        "--study", action="store_true", help="Time every fused and unfused candidate."
    )
    parser.add_argument("--sweep-output-dir", type=str, default=None)
    parser.add_argument("--sweep-tp", type=str, default="2,4,8")
    parser.add_argument("--report-out", type=str, default=None)
    parser.add_argument("--report-input-dir", type=str, default=None)
    parser.add_argument("--worker-timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_input_dir:
        report_path = (
            Path(args.report_out)
            if args.report_out
            else Path(args.report_input_dir).parent / "fused_ar_rms_report.md"
        )
        write_markdown_report(Path(args.report_input_dir), report_path)
        print(f"Wrote {report_path}")
        return
    if args.sweep_output_dir:
        run_sweep(args)
        return
    if args.study:
        run_study_worker(args)
        return
    run_legacy_worker(args)


if __name__ == "__main__":
    main()
