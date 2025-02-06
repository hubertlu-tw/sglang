#!/bin/bash

# ============================================
# LLaMA 3.1 Benchmarking
# ============================================

# Model configurations
# LLAMA_MODEL="amd/Meta-Llama-3.1-8B-Instruct-FP8-KV"
# LLAMA_MODEL_PATH="/data/llama3.1/Llama-3.1-8B-Instruct-FP8-KV/"

# Run benchmarking for LLaMA 3.1
# rocprofv3 --hip-runtime-trace --kernel-trace --output-format pftrace -- python3 -m sglang.bench_one_batch \
# python3 -m sglang.bench_one_batch \
#     --batch-size 32 \
#     --input 32 \
#     --output 32 \
#     --model-path "$LLAMA_MODEL_PATH" \
#     --quantization fp8 \
#     --attention-backend wave \
#     --disable-cuda-graph \
#     --tp 8 2>&1 | tee "llama3.1-8B_tp8_wave_$(date +'%Y%m%d_%H%M%S').txt"

# ============================================
# Grok Benchmarking
# ============================================

# Remove existing wave cache
rm -r ~/.wave/

# Model configurations
GROK_MODEL_PATH="/data/lmzheng-grok-1/"
TOKENIZER_PATH="Xenova/grok-1-tokenizer"

# Run benchmarking for Grok 
# Accuracy: add --correctness-test \
python -m sglang.bench_one_batch  \
    --batch-size 32 \
    --input 128 \
    --output 128 \
    --model "$GROK_MODEL_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --tp 8 \
    --trust-remote-code \
    --quantization fp8 \
    --attention-backend wave \
    --enable-nan-detection 2>&1 | tee "grok_log_$(date +'%Y%m%d_%H%M%S').txt"
