#!/bin/bash
# Qwen3.8-Flash-Next GGUF lane: llama.cpp + Unsloth's MTP draft head, 4x RTX 5090.
# Measured 2026-09-02 (greedy, tools/llm-bench, ctx 131,072): code 130-156 tok/s
# at n-max 2-3 (spec off: 91-99), copy-heavy code 218 at n-max 5, prose 101-104
# at n-max 2. About 1.3-1.5x slower than the vLLM lane in launch.sh; use this
# one when you want a GGUF, not the fast path.
#
# Build the image first (llama.cpp PR #28243, pinned commit, sm120 only):
#   git clone https://github.com/danielhanchen/llama.cpp llama.cpp-mtp
#   git -C llama.cpp-mtp checkout 2857e51143bd88ec6fc0246246f42a5d0394d98a
#   docker build --target server --build-arg CUDA_DOCKER_ARCH=120 \
#     -t qwen-pedia/fnext-llamacpp-mtp:r1 -f llama.cpp-mtp/.devops/cuda.Dockerfile llama.cpp-mtp
#
# Weights (the head is a separate file; the trunk shards did not change):
#   hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
#
# Knobs (see NOTES.md, "llama.cpp lane"):
#   NMAX=2   draft length. 2 is the safe default; 3 for code; 5 only for
#            copy-heavy edits (it is slower than spec-off on prose).
#   CTX=131072  fits 4x 32 GB with the draft. 262,144 is untested with the head.
set -e
HF="${HF_HOME:-$HOME/.cache/huggingface}"
SNAP=$(ls -d "$HF"/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/*/ | tail -1)
MODEL="${MODEL:-$SNAP/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
DRAFT="${DRAFT:-$(ls "$HF"/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/*/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | tail -1)}"
NMAX="${NMAX:-2}"
CTX="${CTX:-131072}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8038}"

docker run -d --restart unless-stopped --name fnext-llamacpp-mtp \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$HF":"$HF":ro \
  -p "$PORT":8000 \
  qwen-pedia/fnext-llamacpp-mtp:r1 \
  -m "$MODEL" \
  -md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max "$NMAX" \
  --alias llamacpp/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0

echo "serving on http://localhost:$PORT/v1 as llamacpp/qwen3.8-flash-next (n-max $NMAX, ctx $CTX)"
echo "expect one 'borrow_shared_tensor' error line at startup: it is the memory fitter probing the head alone, drafting still runs"
