"""Focused gfx1151 benchmark for the packed GDN decode HIP-JIT route."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from sglang.kernels.ops.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode_kernel as baseline_kernel,
)
from sglang.kernels.ops.attention.gdn_packed_decode_gfx1151 import (  # noqa: E402
    gdn_packed_decode_gfx1151 as candidate,
)

H, HV, K, V = 16, 48, 128, 128
CANDIDATE_A_US = {1: 107.61, 2: 136.96, 3: 141.13, 4: 163.17}


def unchanged_baseline(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
):
    """Launch the pre-dispatch Triton kernel with its unchanged configuration."""
    batch = mixed_qkv.shape[0]
    baseline_kernel[(4, batch * HV)](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=128,
        BV=32,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=1,
        num_stages=3,
    )


def make_case(batch: int, *, envelope: bool, negative: bool, seed: int = 123):
    torch.manual_seed(seed + batch)
    device = "cuda"
    dtype = torch.bfloat16
    mixed = torch.randn(batch, 2 * H * K + HV * V, device=device, dtype=dtype) * 0.1
    a = torch.randn(batch, HV, device=device, dtype=dtype) * 0.2
    b = torch.randn(batch, HV, device=device, dtype=dtype) * 0.2
    A_log = torch.randn(HV, device=device, dtype=dtype) * 0.1
    dt_bias = torch.randn(HV, device=device, dtype=dtype) * 0.1
    slots = 8
    dense_stride = HV * V * K
    slot_stride = dense_stride + (257 if envelope else 0)
    backing = torch.randn(slots * slot_stride, device=device, dtype=dtype) * 0.01
    state = torch.as_strided(
        backing,
        (slots, HV, V, K),
        (slot_stride, V * K, K, 1),
    )
    indices = torch.arange(batch, device=device, dtype=torch.int32)
    if negative:
        indices[-1] = -1
    return mixed, a, b, A_log, dt_bias, backing, state, indices


def run(which, case, out):
    mixed, a, b, A_log, dt_bias, _, state, indices = case
    if which == "baseline":
        unchanged_baseline(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            K**-0.5,
            state,
            out,
            indices,
        )
    else:
        candidate(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            K**-0.5,
            state,
            out,
            indices,
        )


def clone_case(case):
    mixed, a, b, A_log, dt_bias, backing, state, indices = case
    cloned_backing = backing.clone()
    cloned_state = torch.as_strided(
        cloned_backing, state.shape, state.stride(), state.storage_offset()
    )
    return mixed, a, b, A_log, dt_bias, cloned_backing, cloned_state, indices


def assert_matches(base_case, cand_case, base_out, cand_out, label):
    torch.testing.assert_close(cand_out, base_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(
        cand_case[5], base_case[5], atol=2e-2, rtol=1e-2
    )
    out_diff = (cand_out.float() - base_out.float()).abs().max().item()
    state_diff = (cand_case[5].float() - base_case[5].float()).abs().max().item()
    print(f"{label}: PASS (out_max={out_diff:.6g}, state_max={state_diff:.6g})")


def check_correctness(replays: int):
    # Every batch size covers envelope slot pitch; B>=2 also covers -1.
    for batch in (1, 2, 3, 4):
        original = make_case(batch, envelope=True, negative=batch >= 2)
        base_case, cand_case = clone_case(original), clone_case(original)
        base_out = torch.full(
            (batch, 1, HV, V), 7, device="cuda", dtype=torch.bfloat16
        )
        cand_out = base_out.clone()
        run("baseline", base_case, base_out)
        run("candidate", cand_case, cand_out)
        torch.cuda.synchronize()
        if batch >= 2:
            assert torch.count_nonzero(cand_out[-1]).item() == 0
        assert_matches(
            base_case, cand_case, base_out, cand_out, f"B={batch} envelope/-1"
        )

    # Compile eagerly before capture, then validate a production-like fixed
    # address graph through 1,000 sequential state updates.
    original = make_case(4, envelope=True, negative=False, seed=777)
    base_case, cand_case = clone_case(original), clone_case(original)
    base_out = torch.empty(4, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    cand_out = torch.empty_like(base_out)
    run("baseline", base_case, base_out)
    run("candidate", cand_case, cand_out)
    torch.cuda.synchronize()
    base_case, cand_case = clone_case(original), clone_case(original)
    base_graph, cand_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(base_graph):
        run("baseline", base_case, base_out)
    with torch.cuda.graph(cand_graph):
        run("candidate", cand_case, cand_out)
    for _ in range(replays):
        base_graph.replay()
        cand_graph.replay()
    torch.cuda.synchronize()
    assert_matches(
        base_case,
        cand_case,
        base_out,
        cand_out,
        f"graph sequential replays={replays}",
    )


def time_batch(batch: int, warmups: int, iterations: int):
    original = make_case(batch, envelope=True, negative=False, seed=999)
    base_case, cand_case = clone_case(original), clone_case(original)
    base_out = torch.empty(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    cand_out = torch.empty_like(base_out)
    for _ in range(warmups):
        run("baseline", base_case, base_out)
        run("candidate", cand_case, cand_out)
    torch.cuda.synchronize()

    records = {"baseline": [], "candidate": []}
    for iteration in range(iterations):
        order = (
            ("baseline", "candidate")
            if iteration % 2 == 0
            else ("candidate", "baseline")
        )
        for which in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(
                which,
                base_case if which == "baseline" else cand_case,
                base_out if which == "baseline" else cand_out,
            )
            end.record()
            records[which].append((start, end))
    torch.cuda.synchronize()
    samples = {
        name: [start.elapsed_time(end) * 1000 for start, end in pairs]
        for name, pairs in records.items()
    }
    base_us = statistics.mean(samples["baseline"])
    cand_us = statistics.mean(samples["candidate"])
    return base_us, cand_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1000)
    args = parser.parse_args()
    if args.warmups < 20 or args.iterations < 100 or args.replays < 1000:
        parser.error(
            "requires >=20 warmups, >=100 iterations, and >=1000 replays"
        )

    props = torch.cuda.get_device_properties(0)
    print(
        f"device={props.name} arch={props.gcnArchName} CUs={props.multi_processor_count}"
    )
    check_correctness(args.replays)
    print("B baseline_us candidate_us speedup vs_candidate_A")
    passed = True
    for batch in (1, 2, 3, 4):
        base_us, cand_us = time_batch(batch, args.warmups, args.iterations)
        speedup = base_us / cand_us
        passed &= speedup > 1.0
        print(
            f"{batch} {base_us:.3f} {cand_us:.3f} {speedup:.4f} "
            f"{CANDIDATE_A_US[batch] / cand_us:.4f}"
        )
    print(f"production_gate={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
