#!/usr/bin/env bash
# Serve Qwen3-Next-80B-A3B-Instruct-FP8 with vLLM (OpenAI-compatible API). Used for probe generation, probe updates
# and listwise reranking. Usage: bash scripts/serve_vllm.sh <GPUS, e.g. 0,1> <PORT, e.g. 8000>
set -euo pipefail
GPUS="${1:-0,1}"
PORT="${2:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
N_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${N_GPUS}" \
    --port "${PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --dtype bfloat16
