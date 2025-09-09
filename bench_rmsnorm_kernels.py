#!/usr/bin/env python3
# Benchmark RMSNorm (with and without residual) comparing
# SGLANG_USE_AITER=0 (vLLM kernels) vs 1 (AIter kernels).
import argparse
import json
import math
import os
import re
import subprocess
import sys
from statistics import median
from typing import Dict, List, Tuple

CHILD_MARK = "--__child_run__"

DEFAULT_NUM_TOKENS = [7, 83, 4096]
DEFAULT_HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
DEFAULT_DTYPES = ["float16", "bfloat16"]  # strings for JSON transport
DEFAULT_ADD_RESIDUAL = [0, 1]  # 0: no residual, 1: with residual

def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]

def str2dtype_list(arg: str) -> List[str]:
    if arg in ("", None):
        return []
    toks = [x.strip().lower() for x in arg.split(",") if x.strip()]
    for t in toks:
        if t not in ("float16", "bfloat16", "half", "bf16"):
            raise argparse.ArgumentTypeError(f"Unsupported dtype: {t}")
    out = []
    for t in toks:
        if t in ("float16", "half"):
            out.append("float16")
        elif t in ("bfloat16", "bf16"):
            out.append("bfloat16")
    return out

def make_configs(num_tokens: List[int], hidden_sizes: List[int], dtypes: List[str], add_residual: List[int]):
    # Each config is a tuple: (num_tokens, hidden_size, dtype_str, add_residual)
    return [(nt, hs, dt, ar) for nt in num_tokens for hs in hidden_sizes for dt in dtypes for ar in add_residual]

def run_child(mode: int,
              configs: List[Tuple[int, int, str, int]],
              warmup: int,
              repeats: int,
              check: bool,
              gemma: bool) -> Dict[Tuple, float]:
    """Spawn child process with SGLANG_USE_AITER=mode and run configs.
    Returns: {(num_tokens, hidden_size, dtype, add_residual): median_us}"""
    env = os.environ.copy()
    env["SGLANG_USE_AITER"] = str(mode)
    payload = {
        "configs": configs,
        "warmup": warmup,
        "repeats": repeats,
        "check": check,
        "gemma": gemma,
    }
    cmd = [sys.executable, sys.argv[0], CHILD_MARK, f"--mode={mode}"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    stdout, stderr = proc.communicate(json.dumps(payload))
    if proc.returncode != 0:
        print("Child stderr:", file=sys.stderr)
        print(stderr or "<empty>", file=sys.stderr)
        raise RuntimeError(f"Child benchmark failed for mode={mode} (exit {proc.returncode})")

    results: Dict[Tuple, float] = {}
    for line in stdout.strip().splitlines():
        try:
            rec = json.loads(line)
            if "event" in rec:
                continue
            key = (rec["num_tokens"], rec["hidden_size"], rec["dtype"], rec["add_residual"])
            results[key] = float(rec["median_us"])
        except Exception as e:
            print(f"Warning: failed to parse line: {line} ({e})", file=sys.stderr)
    return results

def print_summary(res0: Dict[Tuple, float], res1: Dict[Tuple, float]):
    keys = sorted(set(res0.keys()) & set(res1.keys()))
    if not keys:
        print("No overlapping results to compare.")
        return
    print("dtype      NTokens  HSize   Resid  t_vllm(us)   t_aiter(us)  speedup(×)")
    gm_logs = []
    for k in keys:
        t0 = res0[k]
        t1 = res1[k]
        spd = t0 / t1 if (t1 > 0 and math.isfinite(t0) and math.isfinite(t1)) else float("nan")
        if spd > 0 and math.isfinite(spd):
            gm_logs.append(math.log(spd))
        nt, hs, dt, ar = k
        print(f"{dt:9s} {nt:7d} {hs:7d} {('yes' if ar else 'no '):>5s} {t0:12.2f} {t1:12.2f} {spd:10.3f}")
    if gm_logs:
        gm = math.exp(sum(gm_logs) / len(gm_logs))
        print(f"\nGeometric-mean speedup (AIter vs vLLM): {gm:.3f}× over {len(gm_logs)} configs")
    else:
        print("\nNo valid timings to compute geometric mean.")

def child_worker():
    import json as _json
    import os as _os  # fix: ensure os is available in child
    import torch

    raw = sys.stdin.read()
    payload = _json.loads(raw)
    configs = payload["configs"]
    warmup = int(payload.get("warmup", 5))
    repeats = int(payload.get("repeats", 20))
    check = bool(payload.get("check", False))
    gemma = bool(payload.get("gemma", False))

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available in child\n")
        sys.exit(3)

    try:
        from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
    except Exception as e:
        sys.stderr.write(f"Failed to import SGLang RMSNorm: {e}\n")
        sys.exit(4)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch.set_default_device("cuda")

    def time_forwards(fn, warmup_iters: int, repeats_iters: int) -> float:
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        times_ms = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats_iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        return 1000.0 * median(times_ms)

    # Informational line: which mode is active
    print(_json.dumps({"event": "info", "aiter": _os.environ.get("SGLANG_USE_AITER", "unset")}))

    for (num_tokens, hidden_size, dtype_s, add_residual) in configs:
        try:
            dtype = dtype_map[dtype_s]
            seed = (hash((num_tokens, hidden_size, dtype_s, add_residual)) & 0xFFFFFFFF)
            torch.manual_seed(seed)

            LayerCls = GemmaRMSNorm if gemma else RMSNorm
            layer = LayerCls(hidden_size).to(dtype=dtype)

            with torch.no_grad():
                layer.weight.data.normal_(mean=1.0, std=0.1)

            scale = 1.0 / (2.0 * hidden_size)
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
            residual = torch.randn_like(x) * scale if add_residual else None

            if check:
                with torch.inference_mode():
                    ref_out = layer.forward_native(x, residual)
                    out = layer(x, residual)
                if add_residual:
                    ok = torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2) and \
                         torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
                else:
                    ok = torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
                if not ok:
                    print(_json.dumps({
                        "event": "warn",
                        "msg": "mismatch vs forward_native",
                        "num_tokens": num_tokens, "hidden_size": hidden_size,
                        "dtype": dtype_s, "add_residual": add_residual
                    }))

            def fn_forward():
                with torch.inference_mode():
                    layer(x, residual)

            median_us = time_forwards(fn_forward, warmup, repeats)
            print(_json.dumps({
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "dtype": dtype_s,
                "add_residual": add_residual,
                "median_us": float(median_us),
            }))
        except Exception as e:
            print(_json.dumps({
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "dtype": dtype_s,
                "add_residual": add_residual,
                "median_us": float("nan"),
                "error": str(e),
            }))

def main():
    if CHILD_MARK in sys.argv:
        # Fix: remove sentinel and ignore unknown args to avoid argparse failure
        sys.argv = [arg for arg in sys.argv if arg != CHILD_MARK]
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", type=int, required=True)
        # parse_known_args to ignore anything unexpected
        parser.parse_known_args()
        child_worker()
        return

    parser = argparse.ArgumentParser("RMSNorm benchmark: SGLANG_USE_AITER=0 (vLLM) vs 1 (AIter)")
    parser.add_argument("--num_tokens", type=str2int_list, default=DEFAULT_NUM_TOKENS,
                        help=f"CSV list. Default: {','.join(map(str, DEFAULT_NUM_TOKENS))}")
    parser.add_argument("--hidden_sizes", type=str2int_list, default=DEFAULT_HIDDEN_SIZES,
                        help=f"CSV list. Default: {','.join(map(str, DEFAULT_HIDDEN_SIZES))}")
    parser.add_argument("--dtypes", type=str2dtype_list, default=DEFAULT_DTYPES,
                        help="CSV list among: float16,bfloat16")
    parser.add_argument("--residual", type=str2int_list, default=DEFAULT_ADD_RESIDUAL,
                        help="CSV list with 0,1 for no/with residual. Default: 0,1")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per shape")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per shape (median reported)")
    parser.add_argument("--check", action="store_true", help="Validate against forward_native before timing")
    parser.add_argument("--gemma", action="store_true", help="Use GemmaRMSNorm instead of RMSNorm")
    args = parser.parse_args()

    configs = make_configs(args.num_tokens, args.hidden_sizes, args.dtypes, args.residual)

    print("Running SGLANG_USE_AITER=0 (vLLM kernels)...", file=sys.stderr)
    res0 = run_child(0, configs, args.warmup, args.repeats, args.check, args.gemma)
    print("Running SGLANG_USE_AITER=1 (AIter kernels)...", file=sys.stderr)
    res1 = run_child(1, configs, args.warmup, args.repeats, args.check, args.gemma)

    print_summary(res0, res1)

if __name__ == "__main__":
    main()
