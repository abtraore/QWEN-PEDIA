#!/bin/bash
# Qwen3.8-27B (NVFP4), 2x RTX 5090 (sm120), TP=2. No patches: the official
# vLLM v0.28.0 image is enough. This was the rig's daily driver for two weeks.
#
# Measured: decode 157 tok/s single-stream (212-236 on code-shaped prompts,
# MTP acceptance 0.74-0.92), flat at depth; prefill ~4,600-5,050 tok/s at
# 115-137K; KV pool 833,295 tokens (3.18x at the native 262,144 context).
#
# Why v0.28.0 matters: its XQA decode kernels for sm120 keep FULL cudagraphs
# with speculative decoding. On older images spec decode falls back to
# PIECEWISE graphs and the same config decodes ~95-99 tok/s instead.
set -e
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8014}"

docker run -d --restart unless-stopped --name vllm-qwen38 \
  --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -p "$PORT":8000 \
  vllm/vllm-openai:v0.28.0 \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --served-model-name vllm/qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory 15851561472 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

echo "launched on :$PORT (first request after boot is cold-start slow; discard it)"
