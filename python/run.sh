# Llama 3.1
# MODELS=(
#     "amd/Meta-Llama-3.1-8B-Instruct-FP8-KV"
# )

# MODELS_PATH=(
#     "/data/llama3.1/Llama-3.1-8B-Instruct-FP8-KV/"
# )

# rocprofv3 --hip-runtime-trace --kernel-trace --output-format pftrace -- python3 -m sglang.bench_one_batch \
# python3 -m sglang.bench_one_batch \
#     --batch-size 32 \
#     --input 32 \
#     --output 32 \
#     --model-path "$MODELS_PATH" \
#     --quantization fp8 \
#     --attention-backend wave \
#     --disable-cuda-graph \
#     --tp 8 2>&1 | tee llama3.1-8B_tp1wave_log.txt


rm -r ~/.wave/
# python3 -m sglang.bench_one_batch --model dummy_grok1/ --load-format dummy --tokenizer-path Xenova/grok-1-tokenizer --tp 8 --batch-size 32 --input 1024 --output 128 --quantization fp8 --attention-backend wave --enable-nan-detection 2>&1 | tee wave_log.txt
#python -m sglang.bench_one_batch --batch-size 1 --input 2048 --output 256 --model /data//lmzheng-grok-1/ --tp 8 --trust-remote-code --quantization fp8 --correctness-test --attention-backend wave 2>&1 | tee wave_log.txt

#WAVE_CACHE_ON=0 
python -m sglang.bench_one_batch  --batch-size 32 --input 128 --output 128 --model /data/lmzheng-grok-1/ --tokenizer-path Xenova/grok-1-tokenizer --tp 8 --trust-remote-code --quantization fp8 --attention-backend wave --correctness-test --enable-nan-detection 2>&1 | tee log.txt

#python -m sglang.bench_one_batch  --model dummy_grok1/ --load-format dummy --tokenizer-path Xenova/grok-1-tokenizer --tp 8 --batch-size 32 --input 1024 --output 128 --quantization fp8 --attention-backend wave --enable-nan-detection --correctness-test 2>&1 | tee wave_log.txt
