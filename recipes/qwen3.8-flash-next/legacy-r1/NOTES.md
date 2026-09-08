# Qwen3.8-Flash-Next on 4x RTX 5090: field notes

Everything below was measured on saturn (4x RTX 5090 32 GB, sm120, PCIe,
TP=4 + expert parallel, RadixArk NVFP4 checkpoint). Dates are 2026-08-28/29.

## The six boot walls (in the order you will hit them)

1. `hf_xet` below 1.6.0 cannot download the 102 GB n-gram shard. Upgrade
   first; 1.6.0 pulled at 294 MB/s where 1.4.2 crawled at 7.7 MB/s.
2. NVFP4 MoE breaks under plain TP: `moe_intermediate_size` 640 is not
   divisible by the 64-wide quant groups after TP4 sharding. Fix:
   `--enable-expert-parallel` (whole experts per rank).
3. `--disable-custom-all-reduce` is mandatory on multi-GPU consumer
   Blackwell: without it, init dies with `custom_all_reduce.cuh:164
   'invalid argument'`.
4. `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0`: DeepGEMM's gate
   admits the sm12x family, then its blockwise-FP8 kernels fail on consumer
   parts, surfaced only as `Engine core initialization failed` (vllm#51884,
   vllm#54125).
5. `--cap-add=SYS_PTRACE`: the PLE offload handoff uses `pidfd_getfd`,
   blocked by Docker's default seccomp profile.
6. Checkpoint choice decides whether PLE offload fits your host:
   `RadixArk/...-NVFP4` stores the 51B n-gram table as FP8 with a scale
   (~51 GB host RAM, needs the `VLLM_PLE_FORCE_FP8` patch); `Inferact/...`
   stores it BF16 with no scale (~102 GB host RAM, incompatible with the
   FP8 patch). On a 128 GB host, RadixArk is the one that works.

## Knobs that were swept so you do not have to

- MTP `num_speculative_tokens`: 3 beats 4 on interactive traffic (211 vs
  182 tok/s on code at shallow/mid depth). 4 wins only past ~115K context.
- `--max-num-batched-tokens`: leave at default. 8192 bought +4-6% prefill
  and cost 23% of the KV pool on 32 GB cards.
- `--kv-cache-memory` pinning: undercut vLLM's "fully utilize" advisory by
  at least 1 GiB. The advisory misses the PLE worker's ~682 MiB CUDA
  context (OOMs at warmup) and the vision encoder's runtime allocation
  (survived every text benchmark, then OOMed on an image request at 155K
  context beside a second stream). Advisory minus ~760 MiB is stable.
- fp8 KV (`--kv-cache-dtype fp8`, needs the patches in this recipe): pool
  389,084 to 692,910 tokens, greedy outputs md5-identical to bf16, MTP
  acceptance unchanged, prefill -8%, needle 3/3 at 140K. Also the only way
  to reach 393,216 context: the bf16 pool (389,084) cannot hold one
  max-length request.
- YaRN past native: the drafter does not inherit `--hf-overrides`. Pin
  `"max_model_len"` inside `--speculative-config` or acceptance silently
  drops to 0.000 past 262,144 while outputs stay correct. Verified 0.74-0.90
  past native with the pin, needle 3/3 at 336K.

## Operational rules

- Host RAM is part of the recipe: the PLE table wants ~51 GB anonymous plus
  load transients (~75 GB peak). If the host runs other RAM-heavy work,
  keep a swapfile: we watched memory pressure freeze a TP worker past the
  engine watchdog (driver-level NV_ERR_NO_MEMORY on host allocations hours
  before the crash). `--restart unless-stopped` makes such a death a 15-min
  self-heal instead of an outage.
- Boot is the fragile window. Restarting under heavy co-tenant RAM load can
  wedge the load phase; pause the co-tenant or rely on the swapfile.
- Expect ~15 min from `docker run` to serving (model + 51 GB table).
