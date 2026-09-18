# Ternary Bonsai 2 27B (Qwen3.8-27B at 1.72 bits/weight) on one RTX 5090.
# The GGUF uses PrismML's private ggml types (PQ2_0 / PTQ1_0), so stock llama.cpp cannot load it:
# this builds the PrismML fork at a pinned commit, sm120 only, plus our patch series (patches/).
# Retirement condition: when the fork (or mainline) carries the two decode fusions in patch 4,
# drop the patches and bump the pin.
ARG LLAMA_REPO=https://github.com/PrismML-Eng/llama.cpp
ARG LLAMA_SHA=5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6
ARG CUDA_DOCKER_ARCH=120

FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS build
ARG LLAMA_REPO
ARG LLAMA_SHA
ARG CUDA_DOCKER_ARCH
RUN apt-get update && apt-get install -y gcc-14 g++-14 build-essential cmake python3 git libssl-dev libgomp1 libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
RUN git clone ${LLAMA_REPO} /app && git -C /app checkout ${LLAMA_SHA}
WORKDIR /app
COPY patches/ /patches/
RUN git -c user.name=qwen-pedia -c user.email=qwen-pedia@users.noreply.github.com am /patches/*.patch
# --allow-shlib-undefined: the devel image has no libcuda.so.1 stub and ggml's VMM pool links against the driver API
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . \
    && cmake --build build --config Release -j$(nproc) --target llama-server llama-cli llama-bench

FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867 AS server
RUN apt-get update && apt-get install -y libgomp1 curl libcurl4 && apt clean -y && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/build/bin/ /app/
WORKDIR /app
ENV LD_LIBRARY_PATH=/app
ENV LLAMA_ARG_HOST=0.0.0.0
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8000/health" ]
ENTRYPOINT [ "/app/llama-server" ]
