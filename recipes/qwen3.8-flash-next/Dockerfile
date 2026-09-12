# Qwen3.8-Flash-Next on consumer Blackwell (sm120): patched vLLM nightly, build r10 (2026-09-12).
#
# Base: vLLM nightly eed1f3d0 (2026-09-12), which already carries the FlashInfer GDN prefill kernel
# on SM12x (#55715), the FP8 indexer cache (#54890), the removal of torch.compile from the NVIDIA
# path (#55272), the block-FP8 MTP fixes for ModelOpt mixed checkpoints (#55513) and UVA PLE
# offload (#54371). The Python-only overlay in image/vllm adds what is still ours (see NOTES.md):
# fp8 main-KV for QSA, fused multi-step draft, prefix-cache block-drop ablation, local-argmax
# draft sampling, sm120 skinny GEMM plans + tuner, fused GDN qkvz+ba projection, custom IPC
# all-gather routing, custom all-reduce on PCIe, a loader error message.
# Rebuild: docker build -t qwen-pedia/fnext-vllm:r10 .
FROM vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
COPY image/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
