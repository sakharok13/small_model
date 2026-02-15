#!/usr/bin/env bash
set -euo pipefail

# Single-GPU (GPU 0) training of reasoning-SFT for v1/v2/v3 on two models:
#   1. Qwen/Qwen3-0.6B
#   2. PleIAs/Baguettotron
#
# Data: parquets with columns (query, context, thinking, answer, prompt_version, reference_answer)
#       spread across hotpot_traces_all3.rank0..rank7 directories.
#
# Each version is trained separately (--prompt_versions v1, v2, v3).
# filter_correct_only=True (default) keeps only rows where answer == reference_answer.
#
# Usage:
#   bash run_reasoning_sft_3versions_gpu0.sh
#   DATA_DIR=/custom/path RUNS_DIR=/custom/runs bash run_reasoning_sft_3versions_gpu0.sh

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/jovyan/gambashidze/data}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/hotpot_traces_all3.rank*}"
RUNS_DIR="${RUNS_DIR:-${DATA_DIR}/../runs/reasoning_sft}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"

VERSIONS=("v1" "v2" "v3")

run_sft() {
    local model_script="$1"
    local model_tag="$2"
    local version="$3"

    local run_name="${model_tag}_${version}"
    local output_dir="${RUNS_DIR}/${run_name}"

    if [[ -f "${output_dir}/sft_reasoning_summary.json" ]]; then
        echo "=== SKIP (already done): ${run_name} ==="
        echo "  summary: ${output_dir}/sft_reasoning_summary.json"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "  Training: ${run_name}"
    echo "  Script:   ${model_script}"
    echo "  Version:  ${version}"
    echo "  Output:   ${output_dir}"
    echo "============================================================"

    python3 "${SCRIPT_DIR}/${model_script}" \
        --train_files "${TRAIN_FILES}" \
        --prompt_versions "${version}" \
        --output_dir "${output_dir}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --per_device_eval_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LR}" \
        --num_train_epochs "${EPOCHS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --eval_ratio "${EVAL_RATIO}" \
        --evaluation_strategy steps \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 2 \
        --logging_steps "${LOGGING_STEPS}" \
        --seed "${SEED}" \
        --report_to none \
        --bf16 \
        --gradient_checkpointing \
        --run_name "${run_name}"

    echo "=== DONE: ${run_name} ==="
    echo ""
}

echo "Data pattern: ${TRAIN_FILES}"
echo "Runs dir:     ${RUNS_DIR}"
echo "GPU:          ${CUDA_VISIBLE_DEVICES}"
echo "Versions:     ${VERSIONS[*]}"
echo ""

mkdir -p "${RUNS_DIR}"

# --- Qwen3-0.6B ---
for v in "${VERSIONS[@]}"; do
    run_sft "finetune_qwen3_600m_reasoning_sft.py" "qwen3_600m" "$v"
done

# --- Baguettotron ---
for v in "${VERSIONS[@]}"; do
    run_sft "finetune_baguettotron_reasoning_sft.py" "baguettotron" "$v"
done

echo ""
echo "============================================================"
echo "  All 6 training runs complete."
echo "  Results in: ${RUNS_DIR}"
echo "============================================================"
ls -1d "${RUNS_DIR}"/*/sft_reasoning_summary.json 2>/dev/null || true
