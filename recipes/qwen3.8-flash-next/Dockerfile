# Qwen3.8-Flash-Next on consumer Blackwell (sm120): patched vLLM nightly, build r9 (2026-09-05).
#
# Base: vLLM nightly 27a94d1 (2026-09-03, the qwen4_exp model tree), pinned by digest.
# On top, as a Python-only overlay (no recompilation), image/vllm and image/flashinfer carry:
#   - vllm PR #54873 (QSA sparse GQA) and PR #54129 (mmap-backed PLE n-gram table), applied as
#     three-dot diffs (image/pr-54873.diff, image/pr-54129.diff);
#   - the fixes and speed patches listed in image/OVERLAY-FILES.txt and explained in NOTES.md:
#     fp8 main-KV for QSA, prefix-cache block-drop ablation, custom all-reduce on PCIe,
#     custom IPC all-gather routing, sm120 skinny GEMM plans, fused GDN qkvz+ba projection,
#     local-argmax draft sampling, PLE prewarm + mlock, FlashInfer sm120 all-reduce table,
#     mixed-precision loader fixes for nvidia/Qwen3.8-Flash-Next-NVFP4, a loader error message.
# Every patch names its upstream thread in the source; the image retires file by file as they merge.
# Rebuild: docker build -t qwen-pedia/fnext-vllm:r9 .
FROM vllm/vllm-openai@sha256:7d5128a92db05e6d70661a18f10e40c17cd2d8fcaad9226e170a0d69f822b795
COPY image/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY image/flashinfer/ /usr/local/lib/python3.12/dist-packages/flashinfer/
