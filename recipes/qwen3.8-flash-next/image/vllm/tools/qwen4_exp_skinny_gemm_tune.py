# SPDX-License-Identifier: Apache-2.0
"""Sweep skinny GEMM configs for the Qwen4Exp bf16 dense shapes on the local GPU.

Run INSIDE the serving image on a free GPU (needs cuteDSL + quack):

    docker run --rm --gpus '"device=4"' -v "$PWD/plans":/patches \\
        --entrypoint python3 qwen-pedia/fnext-vllm:r9 \\
        /usr/local/lib/python3.12/dist-packages/vllm/tools/qwen4_exp_skinny_gemm_tune.py \\
        --out /patches/qwen4exp-skinny-sm120.json

For every (N, K) in the Qwen4Exp TP=4 plan table and M in {1, 2, 4, 8, 16} it
times each candidate config under CUDA graph replay (hot cache and after an
L2 flush) against torch.nn.functional.linear and keeps only configs that win
both. The JSON it writes is loaded through VLLM_QWEN4_EXP_SKINNY_GEMM_PLANS_FILE.
"""

import argparse
import itertools
import json
import time

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import QWEN4_EXP_GEMM_PLANS

SHAPES = sorted(QWEN4_EXP_GEMM_PLANS) + [(320, 10240), (4120, 2560)]  # 4120 = fused qkvz+ba, TP=4
MS = (1, 2, 4, 8, 16)
BLOCK_SIZES = (32, 64, 128, 256)
OUTPUTS = (1, 2, 3, 4)
UNROLLS = (1, 2, 4)
VECTORS = (4, 8)


def candidates(m: int, n: int, k: int):
    for bs, opb, ku, vw in itertools.product(BLOCK_SIZES, OUTPUTS, UNROLLS, VECTORS):
        if n % opb or k % (bs * vw):
            continue
        yield SkinnyGemmConfig(m, bs, opb, k_unroll=ku, vector_width=vw)
        yield SkinnyGemmConfig(m, bs, opb, k_unroll=ku, vector_width=vw, static_k=k)


def time_graph(fn, iters: int, flush: torch.Tensor | None) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return sum(times[: max(1, len(times) // 2)]) / max(1, len(times) // 2) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--shapes", default="", help="comma list NxK to restrict, e.g. 4120x2560")
    ap.add_argument("--merge", default="", help="existing plans JSON to extend")
    ap.add_argument("--margin", type=float, default=0.97, help="keep if <= margin x cublas")
    args = ap.parse_args()
    assert shape_dynamic_skinny_gemm.is_available(), "cuteDSL not importable"
    dev = torch.device("cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    result: dict[str, dict[str, dict]] = {}
    if args.merge:
        result = json.load(open(args.merge))
    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(v) for v in x.split("x")) for x in args.shapes.split(",")]
    for n, k in shapes:
        w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            ref = torch.nn.functional.linear(x, w)
            base_hot = time_graph(lambda: torch.nn.functional.linear(x, w), args.iters, None)
            base_cold = time_graph(lambda: torch.nn.functional.linear(x, w), args.iters, flush)
            best = None
            for cfg in candidates(m, n, k):
                try:
                    out = shape_dynamic_skinny_gemm(x, w, cfg)
                except Exception as e:  # compile or launch failure: skip config
                    print(f"  skip {cfg}: {type(e).__name__}: {str(e)[:80]}")
                    continue
                if not torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2):
                    print(f"  WRONG {cfg}")
                    continue
                hot = time_graph(lambda: shape_dynamic_skinny_gemm(x, w, cfg), args.iters, None)
                cold = time_graph(lambda: shape_dynamic_skinny_gemm(x, w, cfg), args.iters, flush)
                if hot <= args.margin * base_hot and cold <= args.margin * base_cold:
                    score = hot + cold
                    if best is None or score < best[0]:
                        best = (score, cfg, hot, cold)
            line = f"N={n} K={k} M={m}: cublas hot {base_hot:.1f} us cold {base_cold:.1f} us"
            if best:
                _, cfg, hot, cold = best
                line += f" -> skinny hot {hot:.1f} cold {cold:.1f} {cfg}"
                result.setdefault(f"{n}x{k}", {})[str(m)] = {
                    "num_rows": cfg.num_rows,
                    "block_size": cfg.block_size,
                    "outputs_per_block": cfg.outputs_per_block,
                    "k_unroll": cfg.k_unroll,
                    "vector_width": cfg.vector_width,
                    "static_k": cfg.static_k,
                }
            else:
                line += " -> keep cublas"
            print(line, flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
