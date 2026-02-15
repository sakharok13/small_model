#!/usr/bin/env bash
set -euo pipefail

# Single-node multi-GPU validation for reasoning-SFT checkpoints.
# Default server layout:
#   SCC_EVAL_ROOT=/home/jovyan/avgalichin/scc-eval

SCC_EVAL_ROOT="${SCC_EVAL_ROOT:-/home/jovyan/avgalichin/scc-eval}"
SMALL_MODEL_DIR="${SMALL_MODEL_DIR:-${SCC_EVAL_ROOT}/small_model}"

RUNS_ROOT="${RUNS_ROOT:-${SCC_EVAL_ROOT}/runs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUNS_ROOT}/reasoning_validation_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SUMMARY_DIR}/eval_outputs}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-6000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-all}"  # all|latest|final
EVAL_DTYPE="${EVAL_DTYPE:-auto}"             # auto|float16|bfloat16
EVAL_CONTEXT_MODE="${EVAL_CONTEXT_MODE:-all}"
EVAL_MAX_CONTEXT_CHARS="${EVAL_MAX_CONTEXT_CHARS:-6000}"
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-6000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

mkdir -p "${SUMMARY_DIR}" "${OUTPUT_ROOT}"

CMD=(
  python3 "${SMALL_MODEL_DIR}/validate_reasoning_checkpoints.py"
  --runs_root "${RUNS_ROOT}"
  --groups "v1,v2,v3"
  --expected_train_rows "${EXPECTED_TRAIN_ROWS}"
  --eval_checkpoints "${EVAL_CHECKPOINTS}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --eval_dtype "${EVAL_DTYPE}"
  --eval_context_mode "${EVAL_CONTEXT_MODE}"
  --eval_max_context_chars "${EVAL_MAX_CONTEXT_CHARS}"
  --output_root "${OUTPUT_ROOT}"
  --summary_dir "${SUMMARY_DIR}"
  --jobs_per_gpu "${JOBS_PER_GPU}"
)

if [[ -n "${GPUS}" ]]; then
  CMD+=(--gpus "${GPUS}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip_existing)
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  CMD+=(--local_files_only)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

echo "SCC_EVAL_ROOT=${SCC_EVAL_ROOT}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "SUMMARY_DIR=${SUMMARY_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "GPUS=${GPUS:-auto}"
echo "JOBS_PER_GPU=${JOBS_PER_GPU}"
echo "EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES}"
echo "EVAL_CHECKPOINTS=${EVAL_CHECKPOINTS}"
echo "Running:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
