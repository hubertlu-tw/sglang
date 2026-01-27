# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch

from sglang.test.test_utils import call_generate_srt_raw, kill_process_tree, popen_launch_server


TEST_MODEL = "openai-community/gpt2"
BASE_URL = "http://127.0.0.1:31001"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fastsafetensors requires NVIDIA/AMD GPUs"
)
def test_fastsafetensors_loader():
    try:
        import fastsafetensors  # noqa: F401
    except Exception:
        pytest.skip("fastsafetensors not installed")

    env = os.environ.copy()
    other_args = ["--load-format", "fastsafetensors", "--skip-server-warmup"]

    proc = None
    try:
        proc = popen_launch_server(
            TEST_MODEL,
            BASE_URL,
            timeout=300,
            other_args=other_args,
            env=env,
        )
        # Simple generate request to validate server works.
        output = call_generate_srt_raw(
            "Hello, my name is",
            temperature=0.8,
            max_tokens=8,
            url=f"{BASE_URL}/generate",
        )
        assert output
    finally:
        if proc is not None and proc.poll() is None:
            try:
                kill_process_tree(proc.pid)
            except Exception:
                pass
            time.sleep(2)
