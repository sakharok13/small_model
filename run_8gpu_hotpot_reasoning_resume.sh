#!/usr/bin/env bash
set -euo pipefail

# Global rebalance + 8-GPU resume for HotpotQA reasoning generation.
#
# Steps:
# 1) Aggregate completed rows across all rank outputs.
# 2) Compute globally missing tasks.
# 3) Redistribute missing tasks evenly to workers.
# 4) Launch workers on 8 GPUs and process only missing tasks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/build_hotpot_reasoning_traces_qwen14b.py}"
PREP_SCRIPT="${PREP_SCRIPT:-${SCRIPT_DIR}/prepare_hotpot_reasoning_rebalance.py}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
OUT_DIR="${OUT_DIR:-/home/jovyan/gambashidze/small_model/data/hotpot_reasoning_qwen14b}"
PLAN_DIR="${PLAN_DIR:-${OUT_DIR}.rebalance_plan}"

NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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

STARTUP_CHECK_SECS="${STARTUP_CHECK_SECS:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "PY_SCRIPT=${PY_SCRIPT}"
echo "PREP_SCRIPT=${PREP_SCRIPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "PLAN_DIR=${PLAN_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GPU_OFFSET=${GPU_OFFSET}"

mkdir -p "${PLAN_DIR}"

prep_log="${PLAN_DIR}/prepare.log"
summary_json="${PLAN_DIR}/summary.json"

echo "Preparing global rebalance plan..."
if ! PYTHONUNBUFFERED=1 python3 "${PREP_SCRIPT}" \
  --out_dir "${OUT_DIR}" \
  --plan_dir "${PLAN_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --num_shards "${NUM_SHARDS}" \
  --num_workers "${NUM_WORKERS}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --context_mode "${CONTEXT_MODE}" \
  --max_context_chars "${MAX_CONTEXT_CHARS}" \
  --prompt_version "${PROMPT_VERSION}" \
  > "${prep_log}" 2>&1; then
  echo "Plan preparation failed. Last log lines:"
  tail -n 120 "${prep_log}" || true
  exit 1
fi

cat "${prep_log}"

if [[ ! -f "${summary_json}" ]]; then
  echo "Missing summary file: ${summary_json}"
  exit 1
fi

missing_total="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("missing_total", 0))' "${summary_json}")"
if [[ "${missing_total}" -eq 0 ]]; then
  echo "No missing tasks. Nothing to run."
  exit 0
fi

echo "Missing tasks to run: ${missing_total}"

pids=()
ranks=()
logs=()

for rank in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((GPU_OFFSET + rank))
  task_file="${PLAN_DIR}/tasks_rank${rank}.jsonl"
  log_path="${PLAN_DIR}/worker_rank${rank}.log"

  if [[ ! -s "${task_file}" ]]; then
    echo "Skipping rank=${rank}: no assigned tasks (${task_file})"
    continue
  fi

  echo "Launching rank=${rank} on GPU=${gpu} with tasks=${task_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
  python3 "${PY_SCRIPT}" \
    --use_vllm \
    --tasks_jsonl "${task_file}" \
    --model_name "${MODEL_NAME}" \
    --out_dir "${OUT_DIR}" \
    --num_shards "${NUM_WORKERS}" \
    --shard_index "${rank}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
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
  ranks+=("${rank}")
  logs+=("${log_path}")
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No workers launched (all task files empty)."
  exit 0
fi

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
  echo "Startup failures detected. Logs are under ${PLAN_DIR}/worker_rank*.log"
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
  echo "One or more workers failed. See ${PLAN_DIR}/worker_rank*.log"
  exit "${rc}"
fi

echo "All workers finished."
