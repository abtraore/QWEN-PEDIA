#!/bin/bash
# Qwen3.8-Flash-Next, 4x RTX 5090 (sm120), TP=4 + expert parallel.
# Measured on this config: decode 203-211 tok/s on code, flat to 186K depth;
# prefill ~2,350-2,690 tok/s at 99-186K; KV pool 786,432 tokens at the
# 393,216-token YaRN context (fp8 KV + pinned pool).
#
# Build the image first from the Dockerfile in this directory:
#   docker build -t qwen-pedia/fnext-vllm:r1 .
#
# Knobs (all measured, see NOTES.md):
#   CTX=262144  native, no YaRN penalty on short prompts (default here: 393216)
#   KV=auto     reverts the fp8-KV patch to stock bf16 (pool drops ~45%)
#   KV_BYTES    leave >=1 GiB under vLLM's "fully utilize" advisory: it misses
#               the PLE worker's ~682 MiB CUDA context AND the vision encoder
set -e
CTX="${CTX:-393216}"
KV="${KV:-fp8}"
KV_BYTES="${KV_BYTES:-6400000000}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8036}"

YARN_ARGS=()
SPEC='{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":3}'
if [ "$CTX" -gt 262144 ]; then
  FACTOR=$(python3 -c "print($CTX/262144)")
  YARN_ARGS=(--hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":262144}}")
  # The drafter does not inherit --hf-overrides. Without this pin its
  # acceptance silently drops to 0.000 past 262144.
  SPEC="{\"method\":\"qwen3_8_flash_next_mtp\",\"num_speculative_tokens\":3,\"max_model_len\":$CTX}"
fi

docker run -d --restart unless-stopped --name fnext-vllm \
  --gpus all --cap-add=SYS_PTRACE --ipc=host \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_FORCE_FP8=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_DISABLE_EAGLE_BLOCK_DROP=1 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -p "$PORT":8000 \
  qwen-pedia/fnext-vllm:r1 \
  --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name vllm/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --max-model-len "$CTX" \
  --kv-cache-dtype "$KV" \
  --kv-cache-memory "$KV_BYTES" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,24,32]}' \
  --speculative-config "$SPEC"

echo "launched on :$PORT (model loads in ~15 min; needs ~75 GiB host RAM transient"
echo " for the PLE table: keep a swapfile if the host runs other RAM-heavy work)"
