# QWEN-PEDIA

A hub for fast, verified Qwen serving recipes on consumer hardware. Every
number here was measured on real machines with the exact commands published
next to it. No screenshots without flags, no "trust me" throughput.

Community project, not affiliated with Alibaba or the Qwen team.

## Hardware

Current fleet: one box, "saturn": 6x RTX 5090 32 GB (sm120, PCIe, no NVLink),
AMD 5955WX, 125 GB RAM. Recipes state how many of the six cards they use.
More hardware will join as it arrives; recipes are tagged per configuration.

## Recipes

Newest first: the top row is always the most recently added or updated
recipe.

| model | engine | hardware | context | decode | prefill @depth | KV pool | recipe |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL GGUF) | llama.cpp PR #28243 + Unsloth MTP head | 4x 5090 | 131,072 | 130-156 tok/s on code (n-max 2-3), 218 on copy-heavy edits | 1,456-1,712 tok/s @4K | 1 slot x 131,072 | [recipe](recipes/qwen3.8-flash-next-gguf/) |
| Qwen3.8-Flash-Next (NVFP4) | vLLM nightly + patches | 4x 5090 | 393,216 (YaRN 1.5) | 203-211 tok/s, flat to 186K | 2,350-2,690 tok/s | 786,432 tok | [recipe](recipes/qwen3.8-flash-next/) |
| Qwen3.8-27B (NVFP4) | vLLM v0.28.0, no patches | 2x 5090 | 262,144 native | 157 tok/s (212-236 on code) | 4,600-5,050 tok/s | 833,295 tok | [recipe](recipes/qwen3.8-27b/) |

Decode is single-stream on code-shaped prompts, greedy, warmup discarded.
Prefill is cold, measured at 99K-186K context. Full methodology in
[tools/](tools/): every recipe is benchmarked with `llm-bench`, seeded and
cache-honest (fresh salt per run against a live server).

## Related: the EXL3 lane, by Mia'a AI Lab

[Mia'a AI Lab](https://x.com/MiaAI_lab) serves the same Qwen3.8-27B through
an in-house [exllamav3 fork](https://github.com/MiaAI-Lab/exllamav3)
(DFlash2 and MTP drafting, NVFP4/FP8 KV cache, GB10/aarch64 as well as x86
CUDA), with matching quants:
[EXL3 3.5bpw target](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw)
and a
[DFlash2 EXL3 5.0bpw draft](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw),
plus a
[deployment kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw).
Mia'a AI Lab's numbers: 47.5 tok/s on a DGX Spark (4.43 tokens accepted
per step) and ~30-34.5 tok/s on 24 GB cards, hardware this repo does not
cover. That fork is also where our fp8 QSA-KV patch learned the sm12x rule
that fp8 operands need a bf16 upcast before `tl.dot`. Credit where it is
due.

## Patched images

Some recipes need patches that have not merged upstream yet. We do not ship
prebuilt images: each recipe carries a small Dockerfile (pinned base digest +
COPY of the patch files, all published in the repo) so you build in one
command and can read every line of what you run. Each patch names its
upstream thread and retires when the fix merges.

## Upstream trail

The findings behind these recipes are contributed back where they belong:

- 2026-08-28 [vllm-project/recipes#870](https://github.com/vllm-project/recipes/pull/870): Flash-Next verified at 200 tok/s decode on 4x RTX 5090, with the sm120 knobs explained
- 2026-08-28 [vllm-project/vllm#54275](https://github.com/vllm-project/vllm/pull/54275): fix for the kv-cache-memory advisory that OOMs when other processes hold GPU memory
- 2026-08-28 [fp8 KV for QSA](https://gist.github.com/abtraore/329547468a6eb04ecedac38250148093), offered on [vllm#53896](https://github.com/vllm-project/vllm/pull/53896): pool +78%, outputs md5-identical to bf16
