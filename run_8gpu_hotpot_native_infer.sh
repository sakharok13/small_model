#!/usr/bin/env bash
set -euo pipefail

# 8-GPU parallel inference for HotpotQA train split using Qwen3 native
# <|thinking_start|>/<|thinking_end|> and <|answer_start|>/<|answer_end|>.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/infer_qwen3_native_reasoning.py}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
OUT_JSONL="${OUT_JSONL:-${SCRIPT_DIR}/outputs/hotpot_native_reasoning_train.jsonl}"

NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"

DATASET_NAME="${DATASET_NAME:-hotpot_qa}"
DATASET_CONFIG="${DATASET_CONFIG:-distractor}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-0}" # 0 = all samples in each shard (so globally all train)

CONTEXT_MODE="${CONTEXT_MODE:-all}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-6000}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/hotpot_native_infer_logs}"
STARTUP_CHECK_SECS="${STARTUP_CHECK_SECS:-20}"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUT_JSONL}")"

echo "PY_SCRIPT=${PY_SCRIPT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GPU_OFFSET=${GPU_OFFSET}"
echo "SPLIT=${SPLIT}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"

pids=()
ranks=()
logs=()

for rank in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((GPU_OFFSET + rank))
  log_path="${LOG_DIR}/worker_rank${rank}.log"
  echo "Launching rank=${rank} on GPU=${gpu}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
  python3 "${PY_SCRIPT}" \
    --use_vllm \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --context_mode "${CONTEXT_MODE}" \
    --max_context_chars "${MAX_CONTEXT_CHARS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --num_shards "${NUM_WORKERS}" \
    --shard_index "${rank}" \
    --distributed_output_mode per_rank \
    --output_jsonl "${OUT_JSONL}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_dtype "${VLLM_DTYPE}" \
    > "${log_path}" 2>&1 &

  pids+=("$!")
  ranks+=("${rank}")
  logs+=("${log_path}")
done

echo "Launched ${#pids[@]} workers. Checking startup in ${STARTUP_CHECK_SECS}s..."
sleep "${STARTUP_CHECK_SECS}"

startup_failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  rank="${ranks[$i]}"
  log_path="${logs[$i]}"
  if ! kill -0 "${pid}" 2>/dev/null; then
    startup_failed=1
    echo "Rank ${rank} exited early. Last log lines:"
    tail -n 120 "${log_path}" || true
  fi
done

if [[ "${startup_failed}" -ne 0 ]]; then
  echo "Startup failures detected. Logs are under ${LOG_DIR}"
  exit 1
fi

rc=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  rank="${ranks[$i]}"
  if ! wait "${pid}"; then
    rc=1
    echo "Rank ${rank} failed. Last log lines:"
    tail -n 120 "${logs[$i]}" || true
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  echo "One or more workers failed. See ${LOG_DIR}/worker_rank*.log"
  exit "${rc}"
fi

echo "All workers finished."
echo "Per-rank outputs:"
echo "  ${OUT_JSONL%.*}.rank*.jsonl"
echo
echo "To merge shards:"
echo "  cat ${OUT_JSONL%.*}.rank*.jsonl > ${OUT_JSONL}"
