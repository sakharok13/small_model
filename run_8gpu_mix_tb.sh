#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline:
# 1) prepare grouped train/val splits
# 2) run a predefined experiment grid on 8 GPUs (DDP, 1 epoch max)
# 3) log to TensorBoard
# 4) collect and rank results

# -------------------------------
# Config (override via env vars)
# -------------------------------
INPUT_GLOB="${INPUT_GLOB:-/home/jovyan/gambashidze/small_model/data/synth_ctx_idk.rank*/part-*.parquet}"
SPLITS_DIR="${SPLITS_DIR:-/home/jovyan/gambashidze/small_model/data/sft_v1}"
RUN_ROOT="${RUN_ROOT:-/home/jovyan/gambashidze/small_model/runs}"
RUN_TAG="${RUN_TAG:-qwen3_06b_8gpu_mix_$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="${RUN_ROOT}/${RUN_TAG}"
TB_ROOT="${RUNS_DIR}/tensorboard"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-42}"
VAL_PER_GROUP="${VAL_PER_GROUP:-5000}"

# Predefined mixture
MIX_RATIOS="${MIX_RATIOS:-correct:0.70,refusal:0.15,negative:0.15}"

# Subset controls (to avoid training on all data)
# If TARGET_TRAIN_SAMPLES is empty, it is auto-derived from stats and TARGET_FRACTION.
TARGET_TRAIN_SAMPLES="${TARGET_TRAIN_SAMPLES:-}"
TARGET_FRACTION="${TARGET_FRACTION:-0.50}"
MAX_TARGET_SAMPLES="${MAX_TARGET_SAMPLES:-800000}"

# Train config
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EPOCHS="${EPOCHS:-1.0}"  # hard cap: must be <= 1.0
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

# Predefined experiment grid
LRS_CSV="${LRS_CSV:-1e-5,2e-5,5e-5}"
GRAD_ACCUMS_CSV="${GRAD_ACCUMS_CSV:-8,16}"

mkdir -p "${RUNS_DIR}" "${TB_ROOT}"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is not available. Install/activate your training env first."
  exit 1
fi

python3 - <<PY
epochs = float("${EPOCHS}")
if epochs > 1.0:
    raise SystemExit("EPOCHS must be <= 1.0")
if epochs <= 0.0:
    raise SystemExit("EPOCHS must be > 0")
PY

GPU_COUNT="$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "Requested ${NPROC_PER_NODE} GPUs but only detected ${GPU_COUNT}."
  exit 1
fi

echo "[1/4] Preparing grouped splits"
if [[ ! -f "${SPLITS_DIR}/stats.json" || "${REBUILD_SPLITS:-0}" == "1" ]]; then
  mkdir -p "${SPLITS_DIR}"
  python3 prepare_sft_splits.py \
    --input_glob "${INPUT_GLOB}" \
    --output_dir "${SPLITS_DIR}" \
    --val_per_group "${VAL_PER_GROUP}" \
    --seed "${SEED}" \
    --allow_nonempty_output_dir
else
  echo "Using existing splits at ${SPLITS_DIR}"
fi

if [[ -z "${TARGET_TRAIN_SAMPLES}" ]]; then
  TARGET_TRAIN_SAMPLES="$(python3 - <<PY
import json
import math

stats_path = "${SPLITS_DIR}/stats.json"
ratio_spec = "${MIX_RATIOS}"
target_fraction = float("${TARGET_FRACTION}")
max_target = int("${MAX_TARGET_SAMPLES}")

with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)
counts = stats["train_counts_by_group"]

ratios = {}
for part in ratio_spec.split(","):
    key, value = part.split(":", 1)
    ratios[key.strip()] = float(value)
ratio_sum = sum(ratios.values())
ratios = {k: v / ratio_sum for k, v in ratios.items()}

full_target = min(counts[g] / ratios[g] for g in ("correct", "refusal", "negative"))
target = max(1, int(full_target * target_fraction))
if max_target > 0:
    target = min(target, max_target)
print(target)
PY
)"
fi

echo "[2/4] Configuration"
echo "RUNS_DIR=${RUNS_DIR}"
echo "TB_ROOT=${TB_ROOT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MIX_RATIOS=${MIX_RATIOS}"
echo "TARGET_TRAIN_SAMPLES=${TARGET_TRAIN_SAMPLES}"
echo "EPOCHS=${EPOCHS} (max 1.0 enforced)"

SUMMARY_JSONL="${RUNS_DIR}/summary.jsonl"
: > "${SUMMARY_JSONL}"

IFS=',' read -r -a LRS <<< "${LRS_CSV}"
IFS=',' read -r -a GAS <<< "${GRAD_ACCUMS_CSV}"

TOTAL_RUNS=$(( ${#LRS[@]} * ${#GAS[@]} ))
RUN_IDX=0

echo "[3/4] Launching experiments (${TOTAL_RUNS} runs)"
for LR in "${LRS[@]}"; do
  for GA in "${GAS[@]}"; do
    RUN_NAME="$(printf "run_%03d_lr%s_ga%s_ep%s_mixcustom" "${RUN_IDX}" "${LR}" "${GA}" "${EPOCHS}")"
    OUT_DIR="${RUNS_DIR}/${RUN_NAME}"
    LOG_DIR="${TB_ROOT}/${RUN_NAME}"
    mkdir -p "${OUT_DIR}" "${LOG_DIR}"

    echo ">> [${RUN_IDX}/${TOTAL_RUNS}] ${RUN_NAME}"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" finetune_qwen3_base.py \
      --model_name "${MODEL_NAME}" \
      --output_dir "${OUT_DIR}" \
      --run_name "${RUN_NAME}" \
      --logging_dir "${LOG_DIR}" \
      --train_correct_files "${SPLITS_DIR}/train_correct/*.parquet" \
      --train_refusal_files "${SPLITS_DIR}/train_refusal/*.parquet" \
      --train_negative_files "${SPLITS_DIR}/train_negative/*.parquet" \
      --eval_correct_files "${SPLITS_DIR}/val_correct/*.parquet" \
      --eval_refusal_files "${SPLITS_DIR}/val_refusal/*.parquet" \
      --eval_negative_files "${SPLITS_DIR}/val_negative/*.parquet" \
      --mix_strategy custom \
      --mix_ratios "${MIX_RATIOS}" \
      --target_train_samples "${TARGET_TRAIN_SAMPLES}" \
      --num_train_epochs "${EPOCHS}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
      --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH}" \
      --gradient_accumulation_steps "${GA}" \
      --learning_rate "${LR}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --lr_scheduler_type "${LR_SCHEDULER}" \
      --evaluation_strategy steps \
      --eval_steps "${EVAL_STEPS}" \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit 1 \
      --logging_steps "${LOGGING_STEPS}" \
      --report_to tensorboard \
      --seed "${SEED}" \
      --bf16 \
      --gradient_checkpointing

    python3 - <<PY
import json
import os

run_name = "${RUN_NAME}"
out_dir = "${OUT_DIR}"
summary_path = "${SUMMARY_JSONL}"
metrics_path = os.path.join(out_dir, "all_eval_metrics.json")
row = {"run_name": run_name, "out_dir": out_dir, "status": "ok"}
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    row["metrics"] = metrics
    losses = {}
    for g in ("correct", "refusal", "negative"):
        block = metrics.get(g, {})
        key = f"eval_{g}_loss"
        if key in block:
            losses[key] = block[key]
    if losses:
        row["losses"] = losses
        row["mean_group_loss"] = sum(losses.values()) / len(losses)
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY

    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo "[4/4] Ranking results"
python3 - <<PY
import csv
import json

summary_jsonl = "${SUMMARY_JSONL}"
rows = []
with open(summary_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
rows = sorted(rows, key=lambda r: r.get("mean_group_loss", float("inf")))

ranked_json = "${RUNS_DIR}/summary_ranked.json"
with open(ranked_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

ranked_csv = "${RUNS_DIR}/summary_ranked.csv"
with open(ranked_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "rank",
        "run_name",
        "mean_group_loss",
        "eval_correct_loss",
        "eval_refusal_loss",
        "eval_negative_loss",
        "out_dir",
    ])
    for i, row in enumerate(rows, start=1):
        losses = row.get("losses", {})
        w.writerow([
            i,
            row.get("run_name", ""),
            row.get("mean_group_loss", ""),
            losses.get("eval_correct_loss", ""),
            losses.get("eval_refusal_loss", ""),
            losses.get("eval_negative_loss", ""),
            row.get("out_dir", ""),
        ])

print(f"Wrote {ranked_json}")
print(f"Wrote {ranked_csv}")
if rows:
    print("Best run:", rows[0].get("run_name"), "mean_group_loss=", rows[0].get("mean_group_loss"))
PY

echo "Done."
echo "TensorBoard command:"
echo "tensorboard --logdir ${TB_ROOT} --host 0.0.0.0 --port 6006"
