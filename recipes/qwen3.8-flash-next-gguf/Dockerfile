# Qwen3.8-Flash-Next GGUF lane: llama.cpp with MTP draft-head support.
# Upstream thread: ggml-org/llama.cpp#28243 (branch qwen4exp/mtp on
# danielhanchen/llama.cpp). Retirement condition: when #28243 merges, switch
# the clone to ggml-org/llama.cpp at the merge commit and drop the fork.
# Mirrors the fork's own .devops/cuda.Dockerfile (server target) at a pinned
# commit, with pinned base digests and sm120 as the only CUDA arch.
ARG LLAMA_REPO=https://github.com/danielhanchen/llama.cpp
ARG LLAMA_SHA=2857e51143bd88ec6fc0246246f42a5d0394d98a
ARG CUDA_DOCKER_ARCH=120

FROM docker.io/node:24@sha256:be23f54a88d34e8824c741b19b91064094f92c1c97b194144bfc8b50d67258e2 AS web
ARG LLAMA_REPO
ARG LLAMA_SHA
RUN git clone ${LLAMA_REPO} /src && git -C /src checkout ${LLAMA_SHA}
WORKDIR /src/tools/ui
RUN npm ci && LLAMA_BUILD_NUMBER=${LLAMA_SHA} npm run build

FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS build
ARG LLAMA_REPO
ARG LLAMA_SHA
ARG CUDA_DOCKER_ARCH
RUN apt-get update && apt-get install -y gcc-14 g++-14 build-essential cmake python3 python3-pip git libssl-dev libgomp1
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
RUN git clone ${LLAMA_REPO} /app && git -C /app checkout ${LLAMA_SHA}
WORKDIR /app
COPY --from=web /src/tools/ui/dist tools/ui/dist
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . \
    && cmake --build build --config Release -j$(nproc)
RUN mkdir -p /app/lib && find build -name "*.so*" -exec cp -P {} /app/lib \; \
    && mkdir -p /app/full && cp build/bin/* /app/full

FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867 AS server
RUN apt-get update && apt-get install -y libgomp1 curl && apt clean -y && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-server /app
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8000/health" ]
ENTRYPOINT [ "/app/llama-server" ]
