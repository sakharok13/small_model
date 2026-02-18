#!/usr/bin/env bash
set -euo pipefail

# Resume-safe 8-GPU HotpotQA reasoning generation.
#
# It launches 8 independent rank processes (0..7), each bound to one GPU via
# CUDA_VISIBLE_DEVICES. The Python script skips already-generated rows in each
# rank output directory by default.
#
# Example:
#   bash run_8gpu_hotpot_reasoning_resume.sh
#
# Optional overrides:
#   OUT_DIR=/home/jovyan/gambashidze/small_model/data/hotpot_reasoning_qwen14b \
#   MAX_SAMPLES=0 PROMPT_VERSION=mix BATCH_SIZE=8 \
#   bash run_8gpu_hotpot_reasoning_resume.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/build_hotpot_reasoning_traces_qwen14b.py}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
OUT_DIR="${OUT_DIR:-/home/jovyan/gambashidze/small_model/data/hotpot_reasoning_qwen14b}"

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"

DATASET_NAME="${DATASET_NAME:-hotpot_qa}"
DATASET_CONFIG="${DATASET_CONFIG:-distractor}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"

CONTEXT_MODE="${CONTEXT_MODE:-all}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-6000}"
PROMPT_VERSION="${PROMPT_VERSION:-mix}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

SHARD_SIZE="${SHARD_SIZE:-50000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
PARQUET_COMPRESSION="${PARQUET_COMPRESSION:-zstd}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "PY_SCRIPT=${PY_SCRIPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "GPU_OFFSET=${GPU_OFFSET}"

pids=()
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((GPU_OFFSET + rank))
  log_path="${OUT_DIR}.rank${rank}.log"

  echo "Launching rank=${rank} on GPU=${gpu} (log: ${log_path})"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python3 "${PY_SCRIPT}" \
    --use_vllm \
    --model_name "${MODEL_NAME}" \
    --out_dir "${OUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${rank}" \
    --max_samples "${MAX_SAMPLES}" \
    --seed "${SEED}" \
    --context_mode "${CONTEXT_MODE}" \
    --max_context_chars "${MAX_CONTEXT_CHARS}" \
    --prompt_version "${PROMPT_VERSION}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --shard_size "${SHARD_SIZE}" \
    --save_every "${SAVE_EVERY}" \
    --parquet_compression "${PARQUET_COMPRESSION}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_dtype "${VLLM_DTYPE}" \
    ${EXTRA_ARGS} \
    > "${log_path}" 2>&1 &

  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  echo "One or more ranks failed. Check ${OUT_DIR}.rank*.log"
  exit "${rc}"
fi

echo "All ranks finished."
