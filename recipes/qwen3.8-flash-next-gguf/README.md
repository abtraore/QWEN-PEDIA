# Qwen3.8-Flash-Next GGUF on 4x RTX 5090 (llama.cpp + MTP head)

Unsloth's UD-Q4_K_XL GGUF of the 176B/6B-active Qwen4 preview with the MTP
draft head they published on 2026-09-02, on llama.cpp
[PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) (mainline has no
MTP graph for this architecture yet). Measured, greedy, single stream, ctx
131,072: **130-156 tok/s on code** at draft length 2-3 (91-99 with the
drafter off), 218 on copy-heavy edits at draft length 5, 101-104 on prose.
The vLLM recipe next door (`../qwen3.8-flash-next/`) does 203-211 on the same
four cards; use this one when you want a GGUF.

| decode tok/s | spec off | n-max 2 | n-max 3 | n-max 5 |
|---|---|---|---|---|
| code, fresh generation | 90.9 | 130.5 | 132.9 | 127.5 |
| code, copy-heavy refactor | 99.4 | 137.5 | 155.5 | 217.9 |
| prose (two stories) | 88.4 / 89.5 | 100.6 / 103.2 | 93.5 / 103.7 | 82.2 / 82.0 |
| prose acceptance | | 0.56 | 0.43-0.51 | 0.31 |
| prefill tok/s at 4.3K | 1,456 | 1,691 | 1,685 | 1,712 |

n-max 2 is the safe default (Unsloth's recommendation holds here): +40% on
code, +14% on prose. n-max 5 only pays on copy-heavy edits and is slower
than no drafter on prose. All numbers from `tools/llm-bench` with a fresh
salt per run, warmup discarded, 19*23 exact on every variant.

## Contents

- `Dockerfile`: the fork's own server build at a pinned commit, pinned base
  digests, sm120 only; retires when #28243 merges
- `launch.sh`: parameterized launcher (`NMAX`/`CTX`/`GPUS`/`PORT` knobs)
- `NOTES.md`: the `-md` path trap, the expected startup error line, why
  draft length is workload-shaped, and how this lane compares with vLLM

## Start the server

Needs 4x 32 GB sm120 GPUs and ~114 GB of disk (111 GB trunk + 2.6 GB head).
The trunk shards are unchanged since the model's release; only the head is
a new download.

```bash
git clone https://github.com/abtraore/QWEN-PEDIA && cd QWEN-PEDIA/recipes/qwen3.8-flash-next-gguf
docker build -t qwen-pedia/fnext-llamacpp-mtp:r1 .
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
./launch.sh          # NMAX=3 for code-heavy use
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
SNAP=$(ls -d $HF/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/*/ | tail -1)
docker run -d --restart unless-stopped --name fnext-llamacpp-mtp \
  --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v $HF:$HF:ro -p 8038:8000 \
  qwen-pedia/fnext-llamacpp-mtp:r1 \
  -m $SNAP/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md $SNAP/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --alias llamacpp/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Serving on `http://localhost:8038/v1`, model name
`llamacpp/qwen3.8-flash-next`, about 3 minutes after a warm load (the first
cold load of 111 GB off disk takes longer). One `borrow_shared_tensor`
error line at startup is expected: the memory fitter probes the head on its
own before the trunk exists. Drafting still runs; `draft acceptance = ...`
lines in the log confirm it.
