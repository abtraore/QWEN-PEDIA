# Qwen3.8-27B on 2x RTX 5090: field notes

Measured on saturn (2x RTX 5090 32 GB, sm120, PCIe, TP=2), unsloth NVFP4
checkpoint, vLLM v0.28.0. Dates 2026-08-14 to 08-28.

## What makes this recipe fast

- **v0.28.0 or newer is the whole story for decode.** Its XQA kernels for
  sm12x let FlashInfer keep FULL cudagraphs with speculative decoding; the
  same flags on the prior image fall back to PIECEWISE graphs and decode
  95-99 tok/s instead of 157. MTP n=3 beats n=2 only under full graphs:
  on the old image n=3 was a loss.
- **The MTP drafter ships inside the checkpoint** (`model_mtp.safetensors`,
  referenced by the weight index; the config field is
  `text_config.mtp_num_hidden_layers`, not the usual
  `num_nextn_predict_layers`). Enable with method `mtp`. Verify it works
  from `/metrics` (`vllm:spec_decode_num_accepted_tokens_total`), not from
  tok/s alone.
- **fp8 KV is safe here because the checkpoint carries static scales**
  (`kv_cache_scheme` with calibrated factors, which the official FP8 repo
  lacks). Needle test passed 3/3 at ~516K on the YaRN variant. Note fp8
  does not halve KV on this hybrid model: the 48 linear-attention layers
  keep bf16 state, so it buys ~1.75x, not 2x.
- **Pinned `--kv-cache-memory` beats a utilization fraction**: the value is
  vLLM's own advisory for this config; util 0.95 over-committed past what
  vLLM considers safe.

## Traps

- 640-wide expert FFNs make some TP widths illegal on quantized variants;
  TP=2 avoids the whole class here.
- `--disable-custom-all-reduce` is mandatory on multi-GPU consumer
  Blackwell (`custom_all_reduce.cuh:164 'invalid argument'` at init).
- `qwen3_coder` is the right tool parser for THIS model (its template emits
  `<function=...>` XML). Flash-Next uses `qwen3_xml` instead; they are not
  interchangeable.
- First request after boot is ~20 tok/s. Cold-start, not the real number.
- YaRN variants: the drafter does not inherit `--hf-overrides`. Pin
  `"max_model_len"` inside `--speculative-config` or drafter acceptance
  silently drops to 0.000 past native length while outputs stay correct.
- Known wart: with any MTP-family drafter, each warm prefix-cache hit drops
  its last matched block, ~1,600 tokens on this hybrid layout (vllm#53670).
  Costs ~0.3 s per warm turn; far cheaper than dropping the drafter.
- TP=4 on PCIe: only single-stream decode improves (+12%); prefill drops
  ~3x and 16-stream aggregate drops 2.6x. TP=2 is the right width for this
  model on consumer PCIe boxes.

## Scaling notes

- 16 concurrent streams: ~1,285 tok/s aggregate at TP=2.
- Two TP=2 replicas via `--data-parallel-size 2` (4 GPUs) measured 3,157
  tok/s aggregate at 64 streams where a single TP=2 collapses past its
  knee. Worth it only for sustained multi-agent traffic.
