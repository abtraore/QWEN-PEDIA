#!/bin/bash
# Qwen3.8-Flash-Next (nvidia NVFP4) on 4x RTX 5090, build r9. Measured 2026-09-05 (NOTES.md):
# 95 engine steps/s, 296-303 tok/s decode on code, prefill ~3,200 tok/s at 36K and 100K,
# KV pool 736,359 tokens at the 393,216-token YaRN context, boot ~10 min.
# Build first:  docker build -t qwen-pedia/fnext-vllm:r9 .
# Needs: 4x 32 GB sm120 cards, ~135 GB disk for the checkpoint, 125 GB host RAM (the 48 GB
# n-gram table is read into page cache and mlocked at boot), ulimit memlock unlimited.
set -e
CTX="${CTX:-393216}"          # 262144 = native, no YaRN
KV_BYTES="${KV_BYTES:-6000000000}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8036}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"   # RadixArk/Qwen3.8-Flash-Next-NVFP4 works too
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPEC='{"method":"qwen4_exp_mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
YARN_ARGS=()
if [ "$CTX" -gt 262144 ]; then
  FACTOR=$(python3 -c "print($CTX/262144)")
  YARN_ARGS=(--hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":262144}}")
  # The drafter does not inherit --hf-overrides; without this pin its acceptance is 0.000 past 262144.
  SPEC="{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":3,\"use_local_argmax_reduction\":true,\"max_model_len\":$CTX}"
fi

docker run -d --restart unless-stopped --name fnext-vllm \
  --gpus all --cap-add=SYS_PTRACE --ipc=host --ulimit memlock=-1:-1 \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_CUSTOM_AR_ALLOW_PCIE=1 -e VLLM_CUSTOM_ALL_GATHER=1 -e NCCL_PROTO=Simple \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_DISABLE_EAGLE_BLOCK_DROP=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_PINNED=1 -e VLLM_PLE_MMAP_SERIAL=8192 -e VLLM_PLE_MMAP_WORKERS=32 \
  -e VLLM_PLE_MMAP_PREWARM=1 -e VLLM_PLE_MMAP_MADV_RANDOM=1 -e VLLM_PLE_MMAP_MLOCK=1 \
  -e VLLM_QWEN4_EXP_SKINNY_GEMM_SM120=1 -e VLLM_QWEN4_EXP_SKINNY_GEMM_PLANS_FILE=/plans/qwen4exp-skinny-sm120-fused.json \
  -e VLLM_GDN_FUSE_IN_PROJ=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "$HERE/plans":/plans:ro \
  -p "$PORT":8000 \
  qwen-pedia/fnext-vllm:r9 \
  --model "$MODEL" \
  --served-model-name vllm/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt \
  --tensor-parallel-size 4 --enable-expert-parallel --async-scheduling \
  --max-model-len "$CTX" "${YARN_ARGS[@]}" \
  --kv-cache-dtype fp8 --kv-cache-memory "$KV_BYTES" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,24,32]}' \
  --speculative-config "$SPEC"

echo "launched on :$PORT, model name vllm/qwen3.8-flash-next; healthy in ~10 min"
