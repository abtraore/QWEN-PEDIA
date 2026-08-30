# Qwen3.8-Flash-Next on consumer Blackwell (sm120), patched vLLM nightly.
#
# Base: the dedicated Flash-Next nightly (PRs vllm#53896 + vllm#53899),
# pinned by digest because the tag is a rolling build.
# Patches baked on top (each inert unless its flag/env enables it):
#   ple_layer.py            VLLM_PLE_FORCE_FP8 gate + _offload_weight_scale
#                           capture + optional VLLM_PLE_MMAP_FILE table
#                           (required for FP8-PLE checkpoints like RadixArk;
#                            from the vllm#53896 discussion lineage)
#   kv_cache_coordinator.py VLLM_DISABLE_EAGLE_BLOCK_DROP=1 recovers ~1,600
#                           cached tokens per warm prefix hit (vllm#53670)
#   qsa.py / ops_qsa.py /   fp8 e4m3 main-KV cache for QSA, enabled by
#   qsa_cache.py            --kv-cache-dtype fp8: pool 389,084 -> 692,910
#                           tokens, greedy outputs md5-identical to bf16
#                           (patch offered upstream on vllm#53896)
#
# SUNSET: this image retires when vllm#53896 merges with fp8-KV support.
# Rebuild: docker build -t qwen-pedia/fnext-vllm:r1 .

FROM vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

COPY patches/ple_layer.py            $SITE/models/qwen3_8_flash_next/nvidia/ple_layer.py
COPY patches/kv_cache_coordinator.py $SITE/v1/core/kv_cache_coordinator.py
COPY patches/qsa.py                  $SITE/models/qwen3_8_flash_next/nvidia/qsa.py
COPY patches/ops_qsa.py              $SITE/models/qwen3_8_flash_next/nvidia/ops/qsa.py
COPY patches/qsa_cache.py            $SITE/models/qwen3_8_flash_next/common/qsa_cache.py
