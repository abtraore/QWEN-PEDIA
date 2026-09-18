#!/bin/bash
# Ternary Bonsai 2 27B on ONE RTX 5090: PrismML llama.cpp fork + our decode patches, n-gram speculation.
# Measured 2026-09-19 (server-side, single stream, ctx 262,144): 139-141 tok/s on fresh code,
# 118 @30K depth, 87 @100K; 540 tok/s on copy-heavy edits when the request turns thinking off
# (n-gram drafts, acceptance 0.94); prefill 3,445 tok/s @30K, 2,303 @100K. Loads in about 10 s.
#
# Build the image first from the Dockerfile in this directory:
#   docker build -t qwen-pedia/bonsai27b-llamacpp:s1 .
# Weights (7.2 GB + 0.63 GB vision tower):
#   hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
#
# Knobs (see NOTES.md):
#   GPU=0        one card. Select it with CUDA_VISIBLE_DEVICES, not --gpus device=N (see NOTES.md).
#   CTX=262144   fits with room to spare: 24.9 GB resident with the full KV.
#   NGRAM=1      n-gram speculation on. It costs nothing on fresh text and 3.7x-es edit traffic.
set -e
HF="${HF_HOME:-$HOME/.cache/huggingface}"
SNAP=$(ls -d "$HF"/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/*/ | tail -1)
MODEL="${MODEL:-$SNAP/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
MMPROJ="${MMPROJ:-$SNAP/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"
GPU="${GPU:-0}"
CTX="${CTX:-262144}"
PORT="${PORT:-8041}"
NGRAM="${NGRAM:-1}"

SPEC=()
if [ "$NGRAM" = "1" ]; then
  SPEC=(--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 32 --spec-ngram-mod-n-max 48)
fi

docker run -d --restart unless-stopped --name bonsai27b-llamacpp \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" \
  -v "$HF":"$HF":ro \
  -p "$PORT":8000 \
  qwen-pedia/bonsai27b-llamacpp:s1 \
  -m "$MODEL" --mmproj "$MMPROJ" \
  --alias llamacpp/ternary-bonsai-2-27b \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  --temp 0.6 --top-p 0.95 \
  "${SPEC[@]}"

echo "serving on http://localhost:$PORT/v1 as llamacpp/ternary-bonsai-2-27b (ctx $CTX, n-gram speculation $NGRAM)"
echo "for edits and refactors send \"chat_template_kwargs\": {\"enable_thinking\": false}: that is where the n-gram drafts pay"
