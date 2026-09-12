# Qwen3.8-Flash-Next on 4x RTX 5090 (build r10)

The 176B/6B-active Qwen4 preview (`qwen4_exp` architecture) at **106 engine
steps/s**, **319-331 tok/s decode on code**, prefill about **7,800 tok/s at
both 36K and 100K depth**, with a **736,359-token KV pool** at the default
393,216-token YaRN context. Boot takes about **10 minutes**. Built here from
a pinned nightly digest with the overlay applied as plain Python files, so
you can read every line of what you run.

Default checkpoint is **`nvidia/Qwen3.8-Flash-Next-NVFP4`**, the official
ModelOpt release with a published eval table. **RadixArk/Qwen3.8-Flash-Next-NVFP4**
is the alternative (same speed, no published eval table); pick either with
the `MODEL` knob below.

## What is in this recipe

- `Dockerfile`: pinned nightly base (`vllm/vllm-openai`, digest in the file)
  plus a Python-only overlay copied on top, no recompilation needed.
- `image/OVERLAY-FILES.txt`: every file the overlay touches, relative to
  vLLM's own package tree.
- `image/pr-54873.diff`, `image/pr-54129.diff`: the two upstream vLLM pull
  requests this recipe carries (QSA sparse-attention kernels, mmap-backed
  PLE table loading), applied as three-way diffs onto the pinned nightly.
- `image/vllm/`, `image/flashinfer/`: the overlay source itself, staged so
  the Dockerfile can `COPY` it straight into the image's site-packages.
- `plans/qwen4exp-skinny-sm120-fused.json`: the tuned skinny-GEMM plan table
  for this rig's cards (see NOTES.md to regenerate on different hardware).
- `launch.sh`: the exact production launch command, parameterized.
- `legacy-r1/`: the previous recipe (RadixArk checkpoint, older nightly, no
  YaRN, no skinny GEMM). Keep it if you are pinned to that nightly; it still
  runs at 203-211 tok/s. See its own README and NOTES.md.

The GGUF lane (llama.cpp with Unsloth's MTP draft head, roughly a third
slower on the same cards) is `../qwen3.8-flash-next-gguf/`.

## Requirements

- 4x 32 GB sm120 cards (tested on RTX 5090, PCIe, no NVLink).
- About 135 GB disk for the checkpoint.
- 125 GB host RAM. The 48 GB n-gram (PLE) table is read into page cache at
  boot and mlocked there, so it never pages out; set the container's
  memlock ulimit to unlimited (`--ulimit memlock=-1:-1` in `launch.sh`) or
  the mlock step fails.

## Build and launch

```bash
git clone https://github.com/abtraore/QWEN-PEDIA && cd QWEN-PEDIA/recipes/qwen3.8-flash-next
docker build -t qwen-pedia/fnext-vllm:r10 .
./launch.sh
```

Serves on `http://localhost:8036/v1` (default port), model name
`vllm/qwen3.8-flash-next`, after roughly 10 minutes of weight and PLE-table
loading plus compile and CUDA-graph capture.

### Knobs (environment variables, all optional)

| knob | default | meaning |
|---|---|---|
| `CTX` | `393216` | max model length. `262144` is the native length, no YaRN. Anything above 262144 turns on static YaRN with `factor = CTX / 262144`, applied to both the model and the drafter. |
| `KV_BYTES` | `6000000000` | `--kv-cache-memory` pin, in bytes. |
| `MODEL` | `nvidia/Qwen3.8-Flash-Next-NVFP4` | checkpoint repo. `RadixArk/Qwen3.8-Flash-Next-NVFP4` is the alternative, same speed. |
| `GPUS` | `0,1,2,3` | `CUDA_VISIBLE_DEVICES` value. |
| `PORT` | `8036` | host port, mapped to the container's 8000. |

## The change table: 75.7 to 95 engine steps/s

Nine changes, in the order they landed, each measured on this rig
(2026-09-04 and 2026-09-05). All numbers are engine steps/s on the same
393,216-token YaRN profile unless stated otherwise.

| change | before -> after | mechanism | overlay file |
|---|---|---|---|
| PLE table mlock | 61-66 (degrading between boots) -> 75.7 | pins the prewarmed 47.68 GiB n-gram table in RAM so kswapd cannot evict it under transparent-hugepage compaction pressure, even with 60+ GB of host RAM reported free | `vllm/models/qwen4_exp/nvidia/ple_mmap.py` |
| local-argmax draft-token reduction | 75.7 -> 81.1 | draft steps argmax the local vocab shard and gather (value, id) pairs instead of moving the full bf16 logits tensor for every draft step | `vllm/models/qwen4_exp/nvidia/mtp.py` |
| tuned sm120 skinny GEMM | 81.1 -> 87.9-89.0 | opens upstream's low-latency CuTe DSL GEMM kernel, normally gated to sm90/sm103, to sm120, using a per-shape plan table swept against cuBLAS | `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`, `plans/qwen4exp-skinny-sm120-fused.json` |
| custom IPC one-shot all-gather for logits | 90.2 -> 94.3 | routes the target's final logits all-gather through vLLM's existing custom-all-reduce IPC buffers, a kernel that shipped upstream but was never wired up for this, instead of NCCL | `vllm/distributed/device_communicators/cuda_communicator.py` |
| custom all-reduce over PCIe at TP=4 | 56.5-56.8 -> 61.3-62.4 | bypasses vLLM's NVLink-only gate so the one-shot IPC kernel replaces NCCL's ring for the small payloads at 4-way | `vllm/envs.py` (adds `VLLM_CUSTOM_AR_ALLOW_PCIE`), custom-AR patch inside `vllm/distributed` |
| PLE table prewarm plus disabled readahead | 61.3-62.4 -> 71.2 | streams the whole table into page cache at load, then disables the kernel's 128 KB readahead per 160-byte row access, removing per-token cold-storage reads | `vllm/models/qwen4_exp/nvidia/ple_mmap.py` |
| newer nightly base plus upstream PR #54873 and PR #54129 | 61.3-62.4 (decode regressed here, prefill up) | sparse-GQA kernels speed prefill; mmap-gathered PLE rows replace a 51 GB anonymous host copy with evictable page-cache pages | `image/pr-54873.diff`, `image/pr-54129.diff` |
| NCCL_PROTO=Simple | 88.6 -> 89.8-90.0 | Simple beats the LL protocol for the one remaining NCCL payload (a 2 MB logits gather) once all-reduces had already moved to custom AR | compose/launch env only, no code change |
| fused qkvz+ba GDN input projection | 89.9 -> 90.7 | concatenates the two GDN input-projection weights per rank after load and re-points the originals at views, so one skinny GEMM replaces two on all 36 GDN layers | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` |

Current production: 95.1-95.6 engine steps/s, 296-303 tok/s decode on code.
Fine print: the port-base row above is measured against the pre-port
baseline (56.5-62.4 steps/s on the older nightly), while every row after it
compounds on the same port build, which is why the numbers do not chain as
a single before/after column; see the model's own dated ledger for the full
sequence if you want the exact intermediate boots.

## The nvidia checkpoint

`nvidia/Qwen3.8-Flash-Next-NVFP4` became the default on 2026-09-05. Why: it
is the official ModelOpt 0.46 quantization recipe (MSE-calibrated expert
scales, calibration on CNN-DailyMail plus Nemotron-Post-Training-Dataset-v2,
both public) and ships a published evaluation table against its own FP8
checkpoint, which RadixArk's repackage does not. Same decode speed as
RadixArk on this rig: 95.1-95.5 engine steps/s, code 296-303 tok/s at
acceptance 0.79-0.83, versus RadixArk's 95.1-95.6 steps/s, 285-294 tok/s at
0.75-0.81.

Loading it with the drafter needed three loader fixes, all carried in this
recipe's overlay:

1. FP8 PLE support in mixed ModelOpt checkpoints (upstream vLLM #54882, 8
   lines, cherry-picked).
2. The MTP module remaps `mtp.layers.0` to `mtp.layers.48` for exclude
   lists but not for the mixed-precision `quantized_layers` map, so the MTP
   experts resolved to the wrong quantization method. Fixed in `mtp.py`.
3. The MTP routed experts use `FP8_BLOCK_SCALES` (128x128 `weight_scale_inv`,
   DeepSeek layout), an algorithm vLLM's ModelOpt path did not route
   anywhere upstream (nvidia's own serving command never loads the
   drafter). Routed to vLLM's block-FP8 MoE method inside the mixed
   config. Fixed in `modelopt.py`.

Our eval baseline, measured on our own harness (GPQA Diamond, IFBench,
SciCode, AA-LCR, thinking on, temperature 0, 65,536-token generation
budget):

| benchmark | our measurement |
|---|---|
| GPQA Diamond | 84.3 (167/198) |
| IFBench | 68.3 strict / 76.3 loose, prompt-level |
| SciCode | 21.3 sub-problem |
| AA-LCR | 79.0 (79/100) |

Caveat: nvidia's own harness, sampling budget and prompting are unstated,
so these rows compare against nvidia's published NVFP4 numbers (GPQA
Diamond 91.5, IFBench 81.0, SciCode 18.8, AA-LCR 74.1) only to within a few
points, not as an apples-to-apples reproduction. SciCode and AA-LCR come
out above nvidia's rows here, GPQA and IFBench below; that is consistent
with differing budgets, effort, and judges rather than a regression.

## Negative result: fp8 per-channel on the attention projections

Converting the QSA full-attention q/k/v/o projections (1.23 GB bf16, the
smallest of the excluded bf16 groups) to FP8_PER_CHANNEL_PER_TOKEN on top
of the nvidia checkpoint is correct (gates pass, acceptance unchanged in
band) but **0.7 percent slower**: 94.5-94.6 engine steps/s and 288-295
tok/s on code, versus 95.1-95.5 and 296-303 bf16. At the M of 1 to 4 tokens
these layers run at during decode, the fp8 path pays a per-token
activation-quantization kernel plus a cuBLAS fp8 GEMM that together cost
more than the tuned bf16 skinny kernel those layers already had. The 0.6
GB per rank saved falls below the launch-latency floor, not a bandwidth
win. The same mechanism applies to the larger GDN linear-attention
projections (4.2 GB), so quantizing those would trade accuracy risk for no
measured speed gain on this hardware. Parked; it would reopen only with a
fused quant-plus-GEMM kernel for fp8, or on a batch-heavy workload (M of 16
or more) where these GEMMs become bandwidth-bound instead of launch-bound.

## Also see

- `legacy-r1/`: the older recipe, for anyone still on the pre-port nightly
  or the RadixArk-only config.
- `../qwen3.8-flash-next-gguf/`: the llama.cpp lane, for hardware or setups
  that cannot run the vLLM overlay.
- `NOTES.md`: field notes, the decode-step profile, dead ends, the skinny
  GEMM tuner, and how to verify a boot.

## r10 (2026-09-12): rebase onto nightly eed1f3d0

Same overlay rebased onto the 2026-09-12 nightly. Upstream now carries the FlashInfer GDN prefill
kernel on SM12x (#55715), which is where the prefill jump comes from (3,200 to 7,800 tok/s: the 36
GDN layers ran a Triton fallback before), the FP8 indexer cache (#54890), no torch.compile on the
NVIDIA path (#55272), the block-FP8 MTP fixes (#55513) and UVA PLE offload (#54371). Measured on
the same rig and prompts as r9:

| | r9 (09-05) | r10 (09-12) |
|---|---|---|
| engine steps/s | 95.1 to 95.5 | 105.7 to 106.0 |
| decode on code, tok/s | 296 to 303 | 319 to 331 |
| prefill at 36K / 100K, tok/s | 3,218 / 3,185 | 7,841 / 7,767 |
| needle at 50 percent of 336K | found, 129 s | found, 46 s |
| KV pool at 393,216 context | 736,359 | 736,359 |

The n-gram table is now served by upstream's UVA offload (`--engram-config '{"cpu_offload": true}'`,
pinned host memory, the GPU reads rows directly); the mmap, prewarm and mlock path of r9 is gone.
Four r9 patches retired upstream (QSA sparse GQA, FP8 PLE in mixed checkpoints, block-FP8 MTP
experts, the MTP per-layer algo remap). The r9 overlay stays in git history for the 27a94d1 nightly.
