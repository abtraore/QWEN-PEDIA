# Qwen3.8-Flash-Next on 4x RTX 5090

The 176B/6B-active Qwen4 preview at **203-211 tok/s** decode, flat to 186K
context, with a **786,432-token KV pool** at a 393,216-token YaRN context.
Needs a patched vLLM nightly (the model's support PRs are unmerged), built
here from a pinned digest so you can read every line of what you run.

Measured on this config: decode 203-211 tok/s on code (MTP acceptance up to
0.93), prefill 2,350-2,690 tok/s at 99-186K, needle retrieval 3/3 at 336K.

## Contents

- `Dockerfile`: pinned nightly base + five patch files baked on top, each
  documented with its upstream thread and its retirement condition
- `patches/`: the patch files themselves (PLE FP8 gate, prefix-cache
  block-drop fix, fp8 KV for QSA)
- `launch.sh`: parameterized launcher (`CTX`/`KV`/`GPUS`/`PORT` knobs, the
  YaRN drafter pin handled automatically)
- `NOTES.md`: the six boot walls in order, the knob sweeps, and the
  operational rules (host RAM is part of this recipe)

## Start the server

Needs 4x 32 GB sm120 GPUs, ~130 GB disk for the checkpoint, and ~75 GB of
host RAM headroom during load (the 51B n-gram table lives in host RAM;
keep a swapfile if the host runs other RAM-heavy work).

```bash
git clone https://github.com/abtraore/QWEN-PEDIA && cd QWEN-PEDIA/recipes/qwen3.8-flash-next
docker build -t qwen-pedia/fnext-vllm:r1 .
./launch.sh
```

Or the full command without the script (native 262,144 context variant):

```bash
docker run -d --restart unless-stopped --name fnext-vllm \
  --gpus all --cap-add=SYS_PTRACE --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_FORCE_FP8=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_DISABLE_EAGLE_BLOCK_DROP=1 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8036:8000 \
  qwen-pedia/fnext-vllm:r1 \
  --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name vllm/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory 6400000000 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,24,32]}' \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":3}'
```

Serving on `http://localhost:8036/v1`, model name
`vllm/qwen3.8-flash-next`, after ~15 minutes of model and n-gram table
loading. For the 393,216-token context use `./launch.sh` (default), which
adds the YaRN override and the drafter's `max_model_len` pin together;
setting one without the other silently zeroes drafter acceptance past
262,144 (details in NOTES.md).
