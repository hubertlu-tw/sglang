"""Benchmark the gfx1151 grouped-three-head ``chunk_fwd_o`` specialization.

The eight rows are the exact packed variable-length shapes observed in the
Qwen3.8 GDN extend stage. Correctness against the existing kernel is checked
before timing. Baseline and guarded-candidate launches are interleaved.

Run:
    python benchmark/kernels/gdn/benchmark_chunk_o_grouped_three_head.py
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch

from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o


CASES = [
    (75,),
    (89,),
    (95,),
    (104,),
    (139,),
    (146,),
    (867,),
    (860, 838, 829),
]


def _make_inputs(seq_lens: tuple[int, ...]):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260916 + sum(seq_lens))
    total_tokens = sum(seq_lens)
    total_chunks = sum(math.ceil(length / 64) for length in seq_lens)

    def randn(shape, scale=1.0, dtype=torch.bfloat16):
        return (
            torch.randn(
                shape,
                generator=generator,
                device="cuda",
                dtype=dtype,
            )
            * scale
        )

    q = randn((1, total_tokens, 16, 128), 0.125)
    k = randn((1, total_tokens, 16, 128), 0.125)
    v = randn((1, total_tokens, 48, 128), 0.25)
    h = randn((1, total_chunks, 48, 128, 128), 0.0625)

    # Match chunk_local_cumsum output: FP32, non-increasing within each chunk.
    g = torch.empty((1, total_tokens, 48), device="cuda", dtype=torch.float32)
    offset = 0
    for length in seq_lens:
        for chunk_start in range(0, length, 64):
            size = min(64, length - chunk_start)
            decay = torch.rand(
                (size, 48), generator=generator, device="cuda", dtype=torch.float32
            )
            g[0, offset + chunk_start : offset + chunk_start + size] = (
                -0.03 * decay
            ).cumsum(0)
        offset += length

    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.long,
    )
    return q, k, v, h, g, cu_seqlens


def _interleaved_event_bench(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for i in range(warmup):
        if i % 2:
            candidate()
            baseline()
        else:
            baseline()
            candidate()
    torch.cuda.synchronize()

    baseline_events = []
    candidate_events = []
    for i in range(iterations):
        ordered = (
            (candidate, candidate_events),
            (baseline, baseline_events),
        )
        if i % 2 == 0:
            ordered = tuple(reversed(ordered))
        for fn, records in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            records.append((start, end))

    torch.cuda.synchronize()
    baseline_ms = sum(start.elapsed_time(end) for start, end in baseline_events)
    candidate_ms = sum(start.elapsed_time(end) for start, end in candidate_events)
    return baseline_ms / iterations, candidate_ms / iterations


def _run_case(seq_lens: tuple[int, ...], warmup: int, iterations: int):
    q, k, v, h, g, cu_seqlens = _make_inputs(seq_lens)

    def baseline():
        return chunk_fwd_o(
            q,
            k,
            v,
            h,
            g,
            cu_seqlens=cu_seqlens,
            use_gfx1151_grouped_three_head=False,
        )

    def candidate():
        # Default dispatch exercises the production geometry/architecture guard.
        return chunk_fwd_o(q, k, v, h, g, cu_seqlens=cu_seqlens)

    reference = baseline()
    actual = candidate()
    torch.cuda.synchronize()
    difference = (reference.float() - actual.float()).abs()
    max_abs = difference.max().item()
    max_rel = (difference / reference.float().abs().clamp_min(1e-2)).max().item()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)

    baseline_ms, candidate_ms = _interleaved_event_bench(
        baseline, candidate, warmup, iterations
    )
    return max_abs, max_rel, baseline_ms, candidate_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        raise SystemExit("At least 20 warmups and 100 timed iterations are required.")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This benchmark requires a ROCm GPU.")

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if not arch.startswith("gfx1151"):
        raise SystemExit(f"This specialization is gfx1151-only, found {arch}.")

    print(f"device={torch.cuda.get_device_name()} arch={arch}")
    print(f"warmup={args.warmup} iterations={args.iterations}")
    print(
        "seq_lens,total_tokens,total_chunks,max_abs,max_rel,"
        "baseline_ms,candidate_ms,speedup"
    )
    all_win = True
    for seq_lens in CASES:
        max_abs, max_rel, baseline_ms, candidate_ms = _run_case(
            seq_lens, args.warmup, args.iterations
        )
        speedup = baseline_ms / candidate_ms
        all_win &= speedup > 1.0
        label = "+".join(map(str, seq_lens))
        print(
            f"{label},{sum(seq_lens)},"
            f"{sum(math.ceil(length / 64) for length in seq_lens)},"
            f"{max_abs:.6g},{max_rel:.6g},"
            f"{baseline_ms:.6f},{candidate_ms:.6f},{speedup:.4f}"
        )
    if not all_win:
        raise SystemExit("FAIL: at least one guarded row did not win.")


if __name__ == "__main__":
    main()
