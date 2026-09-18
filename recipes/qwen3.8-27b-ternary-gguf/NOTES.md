# Notes: Ternary Bonsai 2 27B on one 5090

Everything here was measured on 2026-09-17 to 2026-09-19 on one RTX 5090
(sm120, 450 W cap, driver 610.57), PrismML fork at 5d80cff0.

## No model drafter works on the ternary target

- `z-lab/Qwen3.8-27B-DFlash2-GGUF` does not load: the fork branched mainline
  one commit before DFlash2 support, so its loader knows neither the conv
  nor the selector tensors ("expected 81, got 58").
- `magnitudedev/Qwen3.8-27B-DSpark-GGUF` loads and drafts, and the target
  accepts 0.5% of it. Decode drops from 139 to 47 tok/s.
- Control: the same DSpark file on stock `unsloth/Qwen3.8-27B` UD-Q4_K_XL in
  the same fork accepts 26-30%. So the speculative path works and the
  ternary target is the cause. These drafters condition on hidden states
  from five target layers, and training the weights down to ternary moved
  those states off the distribution the drafter learned. Vocabulary, hidden
  size and layer count all match. A drafter for this model has to be trained
  against the ternary weights.
- Stripping the stray YaRN keys from the drafter GGUF changes nothing.

N-gram speculation needs no model and is the lever that works (README).

## Measured and dropped

| tried | result |
|---|---|
| PTQ1_0 packing (5.95 GB instead of 7.21) | decode -4%, prefill halves. Prism's "PQ2_0 wins on Blackwell" holds |
| `-ub 2048 -b 4096` | prefill unchanged (3,068 vs 3,046 tok/s at 53K) |
| `ngram-map-k4v` instead of `ngram-mod` | acceptance 0.26, no gain |
| a purpose-built PQ2_0 matvec kernel (patch 3, `GGML_CUDA_PQ2_GEMV=1`) | correct to 2e-7, 20% slower than the fork's kernel in four variants |
| matvec block geometry, 2 / 4 / 8 warps (`GGML_CUDA_PQ2_WIDE`) | within 2% on every shape |
| sharing the q8_1 activation between matvecs (`GGML_CUDA_MMVQ_SHARE_Q8=1`) | 64 cache hits per token, output identical, 0.00 ms gained |
| vLLM with the NVFP4 checkpoint on one GPU | 68 tok/s undrafted at 32K; with the drafter it does not fit a useful context (the checkpoint is 21.8 GB, its linear-attention weights stay bf16) |

Why the kernel work stalls at +2%: the fork's ternary matvec already streams
weights at 84% of the card's peak memory bandwidth on the ffn shapes and 71%
on the 8 MB projections, and a kernel launch removed from the captured
decode graph is worth about 1 us (734 launches removed = 0.75 ms), far less
than the 2.6-5 us an isolated profile of those kernels suggests, because
back-to-back kernels overlap. What is left in launch fusion is about 0.6 ms
of a 7.1 ms token.

## Patch 4 in one paragraph each

**GLU fusion.** llama.cpp fuses `ffn_gate`, `ffn_up` and the GLU into one
launch, guarded by a check that the output does not overlap any input. The
graph allocator likes to place the GLU output on the memory of the shared
activation, which dies at `ffn_up`, so the check refused the fusion on 24 of
64 layers (layers 1, 4, 6 mod 8). On the quantized matvec path the kernel
reads a private q8_1 copy of the activation made before launch, never the
original, so that overlap is harmless. The patch ignores it for that path
only; the float path keeps the strict check. `GGML_CUDA_FUSE_GLU_RELAXED=0`
turns it off.

**Dual-output matvec.** `ssm_alpha` and `ssm_beta` (bf16, 48 rows, bound by
launch cost) read the same input through same-shape weights, and so do
`attn_k` and `attn_v`. One launch now computes both and writes the second
result to a second output. `GGML_CUDA_FUSE_DUAL_MATVEC=0` turns it off.

Both were validated the same way: greedy output hash identical with the flag
on and off, then the gates above on the built image.

## Traps

- **`--gpus device=N` can land on the wrong card.** Docker's CDI spec maps
  indices and UUIDs to `/dev/nvidiaN` nodes at the time it was generated. A
  reboot that reorders the PCIe bus leaves it stale, and the symptom is
  "cudaMalloc failed: out of memory" on a GPU that `nvidia-smi` shows empty.
  `--gpus all -e CUDA_VISIBLE_DEVICES=N` is immune; `sudo nvidia-ctk cdi
  generate --output=/etc/cdi/nvidia.yaml` fixes the spec.
- **Build link error** (`undefined reference to cuMemCreate`): the CUDA devel
  image has no `libcuda.so.1` stub. The Dockerfile links with
  `--allow-shlib-undefined`, the same flag upstream's own CUDA Dockerfile
  uses.
- **An idle second server on the same GPU slows the first one**: the matvec
  bench read 17.0 us with an idle 25 GB server resident and 15.7 us without.
  Benchmark on an otherwise empty card.
- Do not mix this fork's `ggml-*` libraries with a stock llama.cpp build
  (Prism's own warning), and do not build the fork's `prism-v6` branch.
