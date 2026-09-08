# Qwen3.8-Flash-Next on 4x RTX 5090: field notes (build r9)

Measured on the same rig as `legacy-r1/NOTES.md` (4x RTX 5090 32 GB, sm120,
PCIe, no NVLink, TP=4 plus expert parallel). Dates are 2026-09-04 and
2026-09-05 unless stated otherwise. The nvidia NVFP4 checkpoint is the
default; RadixArk measures the same speed.

## Boot walls: what changed since legacy-r1

Three of the six walls in `legacy-r1/NOTES.md` are resolved or replaced on
this build:

- **Custom all-reduce is now ON**, not disabled. Legacy-r1 needed
  `--disable-custom-all-reduce` because vLLM gates custom AR off on more
  than two PCIe-only GPUs when NVML reports no NVLink on any pair, true for
  every 6x6 pair on this rig. This recipe carries a patch to
  `custom_all_reduce.py` that adds `VLLM_CUSTOM_AR_ALLOW_PCIE=1`; the env
  var alone does nothing against stock vLLM, the mounted patch is what
  makes it work. It also needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`:
  expandable-segments memory is VMM-backed and `cudaIpcGetMemHandle` cannot
  export it, so custom AR's IPC buffer registration fails with `invalid
  argument` at CUDA-graph-buffer registration time whenever expandable
  segments is on. With it off, kernel warmup needs about 1 GB more
  KV-cache headroom than with it on.
- **PLE table handling is mmap plus prewarm plus mlock, not CPU offload.**
  Legacy-r1's `VLLM_PLE_CPU_OFFLOAD=1` copied the whole table into an
  anonymous 51 GB host allocation, needing `--cap-add=SYS_PTRACE` for the
  offload worker's `pidfd_getfd` handoff (still needed here for other
  reasons, see below). This build gathers PLE rows from a memory-mapped
  safetensors file at input prep instead (`VLLM_PLE_MMAP=1`), so table
  pages are ordinary evictable page-cache pages. Three more knobs matter:
  `VLLM_PLE_MMAP_PREWARM=1` streams the full 47.68 GiB table into page
  cache at load; `VLLM_PLE_MMAP_MLOCK=1` then pins it so `kswapd` cannot
  evict it under transparent-hugepage pressure (needs the memlock ulimit,
  `--ulimit memlock=-1:-1`); on a host with `transparent_hugepage/enabled`
  set to `always`, also run `sudo sysctl -w vm.watermark_boost_factor=0`,
  since `kswapd` otherwise wakes for high-order THP allocations rather than
  actual memory pressure and reclaims file pages, including the freshly
  prewarmed table, even with 60+ GB reported free. `VLLM_PLE_MMAP_SERIAL=8192`
  and `VLLM_PLE_MMAP_WORKERS=32` tune the row-gather thread pool;
  `VLLM_PLE_MMAP_MADV_RANDOM=1` disables the kernel's 128 KB readahead per
  160-byte row access, which otherwise multiplies cold-row cost.
- **Checkpoint choice is nvidia vs RadixArk, not RadixArk vs Inferact.**
  Legacy-r1 ruled out Inferact's checkpoint because its PLE table ships in
  BF16 with no scale tensor and cannot load under the fp8 PLE path. That
  reasoning still applies, but the live choice today is nvidia's official
  ModelOpt release versus RadixArk's repackage of the same NVFP4 recipe;
  see the README for why nvidia is now the default.

Three walls are unchanged from legacy-r1 and still apply: `hf_xet` needs to
be recent enough to pull the multi-hundred-GB checkpoint at speed; NVFP4
MoE needs `--enable-expert-parallel` because the expert intermediate size
does not divide evenly across tensor-parallel shards; `VLLM_USE_DEEP_GEMM=0`
and `VLLM_MOE_USE_DEEP_GEMM=0` stay off (DeepGEMM's blockwise-fp8 kernels
fail on consumer sm12x parts, a rig-wide issue, not specific to this
model). `--cap-add=SYS_PTRACE` also stays required, now for the PLE mmap
gather path's handoff rather than the old offload worker's.

## Traps that still apply

- **The drafter does not inherit `--hf-overrides`.** Under YaRN, pin
  `max_model_len` separately inside `--speculative-config` (see
  `launch.sh`), matching the model's own extended length. Without the pin,
  acceptance silently drops to 0.000 past the native 262,144 length;
  nothing fails loudly, only the acceptance metric shows it.
- **Draft depths above 4 need an explicit `--block-size`.** The QSA ring
  capacity is `4*ceil((4+depth)/4)`: 8 at depth 3-4, 12 at depth 5-7, 16 at
  depth 9-12. The default block size is not a multiple of 12, so depth 5-7
  refuses to boot without `--block-size 1632` (a multiple of 8, 12, and
  16). Moot in production: depth 5 measured a loss on every workload (see
  Dead ends below), so depth 3 is what ships.

## Decode-step profile

A torch-profiler trace of a production decode step (150 engine steps,
greedy code generation, taken while the server was at 75.7 engine
steps/s, before the skinny-GEMM and fusion rounds below) found the GPU
busy union was 12.95 ms against a 13.2 ms wall step: no launch gaps left
to recover at that config. The step was about 2,000 tiny kernels inside 3
CUDA graphs:

| bucket | ms | launches | note |
|---|---|---|---|
| bf16 GEMMs on one sm80 WMMA cuBLAS kernel | 4.7 | 528 | the ModelOpt-excluded projections: linear attention, self-attention, hyper-connection mixers, MoE gate, shared expert, MTP. 7.86 GB bf16 total, about 2 GB per rank, versus a 1.2 ms bandwidth floor |
| other GEMM and split-K | 1.9 | ~100 | |
| NCCL all-gather (logits) | 2.2 | 10 | since replaced by the custom IPC all-gather |
| custom all-reduce | 2.2 | 106 | about 20 us each, roughly half of that is rank skew |
| MoE grouped GEMM and routing | 1.3 | 304 | |
| elementwise, norms, hyper-connection | 1.0 | 487 | |
| GDN state update | 0.5 | 48 | |
| QSA attention | 0.16 | 44 | |

Skinny GEMM, the qkvz+ba fusion, and the custom logits all-gather all
landed after this trace and have already closed part of the gap it
implied: at trace time the estimated ceiling was about 8.5 ms/step
(roughly 117.6 engine steps/s); current production is 94.3-95.6 steps/s,
so the remaining headroom to that ceiling is closer to 1.25x now, not the
1.55x the raw trace suggested. About 4 ms of communication (custom
all-reduce plus whatever residual NCCL traffic remains) stays on the
critical path regardless: this rig has no NVLink, so every collective, no
matter how it is routed, pays PCIe latency.

## Measured dead ends

- **Draft depth 5.** Needed `--block-size 1632` just to boot. Once
  booted, a loss on every workload: code fresh generation 208 to 123
  tok/s, code with a 4K prompt 246 to 188 (acceptance 0.82 to 0.64), code
  at 31K depth 244 to 167 (0.81 to 0.48), prose 159-189 to 15-20 (0.41-0.54
  to 0.31-0.39). Engine steps/s 71-72 to 49, KV pool 708,497 to 676,055.
  Per-token acceptance fell at every position, not only from depth
  compounding; the cause was not root-caused further. Depth 3 restored.
- **MTP n=4 versus n=3.** Shallow 170 to 162 tok/s, code 211 to 182. Wins
  only past roughly 115K depth (186 to 203). n=3 kept as the default.
- **`--max-num-batched-tokens 8192` versus the default 2048.** Prefill up
  4-6 percent (2,360 to 2,460-2,487) but KV pool down 23 percent (to
  552,269). Rejected.
- **FlashInfer trtllm all-reduce at TP=4.** Engine steps/s 56.7 to 54.6
  (-4 percent), correct output, but crashed production once with a 40 MiB
  CUDA OOM: the FlashInfer workspace ate the runtime margin. Reverted to
  custom AR.
- **`NCCL_PROTO=LL`.** Engine steps/s 55.4 to 45.2 (-18 percent), measured
  before custom AR existed, when all-reduces were still PyNccl. NCCL's
  default Simple protocol was already the right choice for PCIe at these
  payload sizes.
- **GPU-resident PLE row cache.** 92.3 percent hit rate in production
  traffic, still slower overall: 44.7 engine steps/s, p50 gather 10 ms. A
  second cache tier cannot beat the Linux page cache at this job; prewarm
  plus mlock was the real fix.
- **PP=2xTP=2.** Dead at boot on this model, structurally: the PLE n-gram
  embedding layer requires `pipeline_parallel_size=1`, since non-first
  pipeline ranks never see raw input_ids.
- **fp8 per-channel on the attention projections.** Correct, 0.7 percent
  slower. See the README for the mechanism.

## The skinny-GEMM tuner

The tuned plan table in `plans/qwen4exp-skinny-sm120-fused.json` is
specific to the GPU it was swept on. To regenerate it on a different sm120
card, run the tuner inside the serving image on one free GPU (its own
docstring, `image/vllm/tools/qwen4_exp_skinny_gemm_tune.py`):

```bash
docker run --rm --gpus '"device=4"' -v "$PWD/plans":/patches \
    --entrypoint python3 qwen-pedia/fnext-vllm:r9 \
    /usr/local/lib/python3.12/dist-packages/vllm/tools/qwen4_exp_skinny_gemm_tune.py \
    --out /patches/qwen4exp-skinny-sm120.json
```

For every (N, K) shape in the model's GEMM plan table and every M in {1,
2, 4, 8, 16}, it times each candidate kernel config under CUDA-graph
replay, both hot and after an L2 cache flush, against
`torch.nn.functional.linear`, and keeps only configs that beat cuBLAS on
both. On this rig's cards, 26 of 45 (shape, M) points won; the rest fell
back to cuBLAS. Never use the kernel outside a configuration the tuner
actually passed: several hundred candidate configs were rejected by the
correctness check (`allclose` against the reference) during tuning,
meaning they compute wrong results on sm120 despite compiling cleanly.
Tuning takes on the order of 30 minutes on one GPU.

## How to verify a boot

Cheapest first, run these on every restart:

1. **Math.** A fixed-answer arithmetic prompt, for example `19*23=437`,
   comes back exact.
2. **Tool call.** A tool-call-shaped prompt parses into a structured
   `tool_calls` field via the `qwen3_xml` parser, with the right
   `finish_reason`.
3. **Needle at depth.** A needle-in-haystack retrieval test at roughly
   50 percent and 95 percent of the max context length comes back found,
   exact string.
4. **Acceptance band.** Read
   `vllm:spec_decode_num_{accepted,draft}_tokens_total` from `/metrics`
   around a request. Expect roughly 0.55-0.90 depending on workload, code
   higher, prose lower. Identical greedy prompts can show 0.57-0.83 noise
   across repeated rounds from prefix-cache state and spec-batching
   nondeterminism, so treat sub-0.1 deltas as noise unless repeated. A
   reading of exactly 0.000 past the native context length means the
   drafter's `max_model_len` pin is missing.
5. **Engine steps/s probe.** Compare against the current baseline,
   95.1-95.6 steps/s on the 393,216 YaRN profile. A regression usually
   means a mistuned PLE thread-pool setting, a missing mlock, or the wrong
   all-reduce configuration; check the boot log for the `['CUSTOM',
   'PYNCCL']` pair and the skinny-GEMM and fusion engagement lines,
   grepping for one pattern at a time (a combined grep can hide the line
   you are looking for behind a flood of unrelated matches).

A KV-pool sanity check is a good sixth gate: roughly 736,359 tokens at the
393,216-token YaRN context, at the 6.0 GB `--kv-cache-memory` pin. A
materially lower number usually means an all-reduce or KV-dtype
configuration change silently ate into the runtime margin.
