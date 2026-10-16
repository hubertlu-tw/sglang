"""Fused AR+RMSNorm sites dispatch to AITER QuickReduce+RMSNorm on ROCm.

Launches itself through torchrun so the real SGLang communicators are built.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd")

QUICK_REDUCE_L2_LIMIT = 0.2
CUSTOM_ALL_REDUCE_L2_LIMIT = 1e-2


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = torch.linalg.vector_norm(actual.float() - expected)
    return (err / torch.linalg.vector_norm(expected).clamp_min(1e-30)).item()


def _run_distributed_check() -> None:
    import torch.distributed as dist

    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import is_gfx95_supported

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    initialize_model_parallel()

    group = get_parallel().tp_group
    assert group.qr_comm is not None and not group.qr_comm.disabled
    calls = {"qr": 0}
    qr_fused = group.qr_comm.quick_all_reduce_rmsnorm

    def counting_qr_fused(*args, **kwargs):
        calls["qr"] += 1
        return qr_fused(*args, **kwargs)

    group.qr_comm.quick_all_reduce_rmsnorm = counting_qr_fused

    dtype, eps = torch.bfloat16, 1e-6
    failures = []

    def check(tokens: int, hidden: int, expect: str, label: str) -> None:
        """``expect`` is ``qr`` (fused QuickReduce kernel), ``ca``
        (CustomAllReduce kernel) or ``fallback`` (None, so the caller runs
        the all-reduce and the RMSNorm separately)."""
        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        x = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=gen)
        gen_res = torch.Generator(device=device).manual_seed(7)
        residual = torch.randn(
            tokens, hidden, dtype=dtype, device=device, generator=gen_res
        )
        weight = torch.rand(hidden, dtype=dtype, device=device) + 0.5
        dist.broadcast(weight, src=0)

        reduced = x.float()
        dist.all_reduce(reduced)
        ref_residual = reduced + residual.float()
        ref_out = (
            ref_residual
            * torch.rsqrt(ref_residual.pow(2).mean(-1, keepdim=True) + eps)
            * weight.float()
        )

        before = calls["qr"]
        result = tensor_model_parallel_fused_allreduce_rmsnorm(x, residual, weight, eps)
        torch.cuda.synchronize()
        if result is None:
            got = "fallback"
        else:
            got = "qr" if calls["qr"] > before else "ca"
        if got != expect:
            failures.append(f"{label}: took {got}, expected {expect}")
            return
        if result is None:
            if rank == 0:
                print(f"{label}: {got}", flush=True)
            return
        out, residual_out = result
        limit = QUICK_REDUCE_L2_LIMIT if got == "qr" else CUSTOM_ALL_REDUCE_L2_LIMIT
        l2 = max(_relative_l2(out, ref_out), _relative_l2(residual_out, ref_residual))
        if not l2 <= limit:
            failures.append(f"{label}: relative L2 {l2:.3e} > {limit:.1e}")
        if rank == 0:
            print(f"{label}: {got} l2={l2:.3e}", flush=True)

    check(2048, 4096, expect="qr", label="2048x4096")
    check(2048, 8192, expect="qr", label="2048x8192")
    # 4 MiB: QuickReduce takes it at TP2/TP4 but runs unfused; at TP8 it is
    # below QuickReduce's INT4 minimum.
    check(
        512,
        4096,
        expect="ca" if world_size == 8 else "fallback",
        label="512x4096",
    )
    check(8, 4096, expect="ca", label="8x4096 (1-stage)")
    # 7168 * 2 bytes does not divide QuickReduce's 32 KiB tile.
    check(2048, 7168, expect="ca", label="2048x7168 (row size)")
    with envs.SGLANG_ENABLE_QUICK_AR_RMSNORM_FUSION.override(False):
        check(2048, 4096, expect="ca", label="2048x4096 (opt-out)")

    # FP8-consumer sites: large sizes leave the CustomAllReduce
    # AR+RMSNorm+quant kernel for QuickReduce plus a separate quant.
    if is_gfx95_supported():
        for tokens, expect_none in ((2048, True), (16, False)):
            x = torch.randn(tokens, 4096, dtype=dtype, device=device)
            pgq = tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
                x,
                torch.randn_like(x),
                torch.ones(4096, dtype=dtype, device=device),
                eps,
            )
            if (pgq is None) != expect_none:
                failures.append(
                    f"per-group quant {tokens}x4096: returned "
                    f"{'None' if pgq is None else 'a result'}"
                )
            elif rank == 0:
                print(
                    f"per-group quant {tokens}x4096: "
                    f"{'deferred to QuickReduce' if pgq is None else 'CustomAllReduce'}",
                    flush=True,
                )

    # Graph replay must reproduce the eager QuickReduce result.
    tokens, hidden = 2048, 4096
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    residual = torch.randn(tokens, hidden, dtype=dtype, device=device)
    weight = torch.ones(hidden, dtype=dtype, device=device)
    eager_out, eager_residual = tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, weight, eps
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as context:
        with torch.cuda.graph(graph, stream=context.stream):
            graph_out, graph_residual = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, eps
            )
    graph.replay()
    torch.cuda.synchronize()
    if not (
        torch.equal(graph_out, eager_out)
        and torch.equal(graph_residual, eager_residual)
    ):
        failures.append("graph replay differs from eager QuickReduce output")

    local_ok = torch.tensor([int(not failures)], device=device)
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    for failure in failures:
        print(f"[rank {rank}] FAIL {failure}", flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()
    if not local_ok.item():
        sys.exit(1)
    if rank == 0:
        print("ALL CHECKS PASSED", flush=True)


class TestQuickArRmsNormFusion(CustomTestCase):
    def test_fused_sites_use_quick_allreduce_rmsnorm(self):
        nproc = min(torch.cuda.device_count(), 8)
        if nproc < 2:
            self.skipTest("This test requires at least 2 GPUs.")
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_USE_AITER": "1",
                "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
                "SGLANG_ENABLE_QUICK_AR_RMSNORM_FUSION": "1",
            }
        )
        env.pop("ROCM_QUICK_REDUCE_QUANTIZATION", None)
        env.pop("SGLANG_USE_1STAGE_ALLREDUCE", None)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            __file__,
            "--distributed",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("ALL CHECKS PASSED", result.stdout)


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        _run_distributed_check()
    else:
        unittest.main()
