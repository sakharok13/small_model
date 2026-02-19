#!/usr/bin/env bash
set -euo pipefail

# 8-GPU accelerate launcher for reasoning-trace SFT.
#
# Defaults:
# - MODEL_SCRIPT=finetune_qwen3_600m_reasoning_sft.py
# - TRAIN_FILES=/home/jovyan/gambashidze/small_model/outputs/hotpot_native_reasoning_train.rank*.jsonl
#
# Usage examples:
#   bash run_8gpu_reasoning_sft.sh
#   MODEL_SCRIPT=finetune_baguettotron_reasoning_sft.py bash run_8gpu_reasoning_sft.sh
#   OUTPUT_DIR=/home/jovyan/gambashidze/small_model/runs/qwen3_600m_reasoning_sft_v2 \
#   MAX_SEQ_LEN=4096 EVAL_RATIO=0.005 bash run_8gpu_reasoning_sft.sh

MODEL_SCRIPT="${MODEL_SCRIPT:-finetune_qwen3_600m_reasoning_sft.py}"
TRAIN_FILES="${TRAIN_FILES:-/home/jovyan/gambashidze/small_model/outputs/hotpot_native_reasoning_train.rank*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/gambashidze/small_model/runs/reasoning_sft_8gpu}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-3072}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1.0}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
REPORT_TO="${REPORT_TO:-tensorboard}"
SEED="${SEED:-42}"

ACCELERATE_EXTRA_ARGS="${ACCELERATE_EXTRA_ARGS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "MODEL_SCRIPT=${MODEL_SCRIPT}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  ${ACCELERATE_EXTRA_ARGS} \
  "${MODEL_SCRIPT}" \
  --train_files "${TRAIN_FILES}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --eval_ratio "${EVAL_RATIO}" \
  --evaluation_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --seed "${SEED}" \
  --report_to "${REPORT_TO}" \
  --bf16 \
  --ddp_backend nccl \
  ${TRAIN_EXTRA_ARGS}
