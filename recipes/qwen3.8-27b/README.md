# Qwen3.8-27B on 2x RTX 5090

The easy one: official vLLM v0.28.0 image, zero patches, one command.

Measured on this config: **157 tok/s** single-stream decode (212-236 on
code-shaped prompts, MTP acceptance 0.74-0.92), flat at depth; prefill
**4,600-5,050 tok/s** at 115-137K context; KV pool **833,295 tokens**
(3.18x concurrency at the native 262,144 context).

## Contents

- `launch.sh`: the launch command with `GPUS`/`PORT` as env knobs
- `NOTES.md`: why each flag is there, the traps (wrong tool parser, TP=4
  on PCIe, the YaRN drafter pitfall), and scaling numbers

## Start the server

Copy-paste (adjust `CUDA_VISIBLE_DEVICES` and the port to taste; needs
2x 32 GB sm120 GPUs and the ~21 GB checkpoint download on first run):

```bash
docker run -d --restart unless-stopped --name vllm-qwen38 \
  --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8014:8000 \
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
```

Serving on `http://localhost:8014/v1`, model name `vllm/qwen3.8-27b`.
The first request after boot is cold-start slow (~20 tok/s); discard it.
