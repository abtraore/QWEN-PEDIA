# Qwen3.8-27B ternary (Bonsai 2) on one RTX 5090 (llama.cpp, PrismML fork)

[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is Qwen3.8-27B at 1.72 bits per weight: ternary weights with an fp16 scale
per 128, embeddings and LM head included. 7.2 GB on disk, so one 32 GB card
holds the model, the vision tower and a full 262,144-token KV with 7 GB to
spare. Measured on GPU 5 of the box, server-side, single stream:

| | tok/s |
|---|---|
| decode, fresh code | 139-141 |
| decode at 30K / 100K depth | 118 / 87 |
| decode, copy-heavy edit, thinking off, n-gram drafts (acceptance 0.94) | **540** |
| prefill at 30K / 100K | 3,445 / 2,303 |
| load time from NVMe | about 10 s |

Gates on the served model: 19*23 exact, tool call parses, needle found at
60% of a 100K prompt, an AIME 2024 problem answered correctly, generated
code passes its own asserts. Quality beyond that is Prism's claim (98.2% of
fp16 on their 14-benchmark harness, vision 66.2 vs 71.4), not ours.

The two-GPU vLLM recipe next door (`../qwen3.8-27b/`) does 212-236 tok/s on
code with its drafter. Use this one when you have one card, want a very
deep context on it, or want a 27B sidecar next to a bigger model.

## Contents

- `Dockerfile`: the PrismML fork at a pinned commit plus `patches/`, sm120
  only. Stock llama.cpp refuses these files (private ggml types 142/143).
- `patches/`: four commits on top of the fork. Patch 4 is the one that
  changes speed: it restores the gate/up/GLU fusion on the 24 layers where
  the fork's memory check refused it, and runs `ssm_alpha`+`ssm_beta` and
  `attn_k`+`attn_v` as one launch each. 1,542 to 1,430 kernel launches per
  token, output byte-identical. Patches 1-3 are a launch counter and an
  experimental matvec kernel, both off unless an env var asks for them.
- `launch.sh`: parameterized launcher (`GPU`/`CTX`/`PORT`/`NGRAM` knobs)
- `NOTES.md`: why no model drafter, what else was tried and measured, and
  the traps

## Start the server

Needs one 32 GB sm120 GPU and 8 GB of disk.

```bash
git clone https://github.com/abtraore/QWEN-PEDIA && cd QWEN-PEDIA/recipes/qwen3.8-27b-ternary-gguf
docker build -t qwen-pedia/bonsai27b-llamacpp:s1 .
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
./launch.sh          # GPU=5 PORT=8041 to move it
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
SNAP=$(ls -d $HF/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/*/ | tail -1)
docker run -d --restart unless-stopped --name bonsai27b-llamacpp \
  --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -v $HF:$HF:ro -p 8041:8000 \
  qwen-pedia/bonsai27b-llamacpp:s1 \
  -m $SNAP/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj $SNAP/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --alias llamacpp/ternary-bonsai-2-27b \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -fa on --jinja -c 262144 --parallel 1 --metrics \
  --temp 0.6 --top-p 0.95 \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 32 --spec-ngram-mod-n-max 48
```

Serving on `http://localhost:8041/v1`, model name
`llamacpp/ternary-bonsai-2-27b`.

## Getting the 540 tok/s

N-gram speculation drafts from text the server has already seen, so it pays
when the output repeats the prompt: edits, refactors, "return the file with
this change". It needs thinking off for that request, otherwise most of the
output is reasoning that repeats nothing (measured: 147 tok/s with thinking
on, 540 with it off, same prompt). Per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

On fresh text the drafter proposes nothing about 97% of the time and costs
nothing, so it stays on by default.
