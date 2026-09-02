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
- `launch-llamacpp-mtp.sh`: the GGUF lane (llama.cpp + Unsloth's MTP draft
  head), measured below; slower than vLLM, useful when you want a GGUF

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

## GGUF lane: llama.cpp with Unsloth's MTP head

Unsloth published a separate MTP draft head for this model on 2026-09-02
(`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 2.6 GB; it borrows the
trunk's embeddings). It needs llama.cpp
[PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243), not a mainline
build. The UD-Q4_K_XL trunk shards did not change (same LFS oids as before),
so only the head is a new download. Measured on 4x RTX 5090, ctx 131,072,
greedy, `tools/llm-bench` with a fresh salt per run:

| decode tok/s | spec off | n-max 2 | n-max 3 | n-max 5 |
|---|---|---|---|---|
| code, fresh generation | 90.9 | 130.5 | 132.9 | 127.5 |
| code, copy-heavy refactor | 99.4 | 137.5 | 155.5 | 217.9 |
| prose (two stories) | 88.4 / 89.5 | 100.6 / 103.2 | 93.5 / 103.7 | 82.2 / 82.0 |
| prose acceptance | | 0.56 | 0.43-0.51 | 0.31 |
| prefill tok/s at 4.3K | 1,456 | 1,691 | 1,685 | 1,712 |

n-max 2 is the safe default (Unsloth's recommendation holds here): +40% on
code, +14% on prose. n-max 5 only pays on copy-heavy edits and is slower
than no drafter on prose. For comparison the vLLM lane above does 203-211
tok/s on code with prefill at 2,350-2,690, so this lane is the fallback,
not the fast path.

```bash
git clone https://github.com/danielhanchen/llama.cpp llama.cpp-mtp
git -C llama.cpp-mtp checkout 2857e51143bd88ec6fc0246246f42a5d0394d98a
docker build --target server --build-arg CUDA_DOCKER_ARCH=120 \
  -t qwen-pedia/fnext-llamacpp-mtp:r1 -f llama.cpp-mtp/.devops/cuda.Dockerfile llama.cpp-mtp
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
./launch-llamacpp-mtp.sh          # NMAX=3 for code-heavy use
```

Serving on `http://localhost:8038/v1`, model name
`llamacpp/qwen3.8-flash-next`, about 3 minutes after a warm load (the first
cold load of 111 GB off disk takes longer). One `borrow_shared_tensor`
error line at startup is expected: the memory fitter probes the head on its
own before the trunk exists. Drafting still runs; `draft acceptance = ...`
lines in the log confirm it.
