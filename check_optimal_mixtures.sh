#!/usr/bin/env bash
set -euo pipefail

# Compare a curated set of train mixture splits for context-grounded QA.
# Assumes grouped splits already exist (train_correct/refusal/negative, val_*).

SPLITS_DIR="${SPLITS_DIR:-/home/jovyan/gambashidze/small_model/data/sft_v1}"
RUN_ROOT="${RUN_ROOT:-/home/jovyan/gambashidze/small_model/runs}"
RUN_TAG="${RUN_TAG:-qwen3_06b_mixcheck_$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="${RUN_ROOT}/${RUN_TAG}"
TB_ROOT="${RUNS_DIR}/tensorboard"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-42}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

# Fixed train setup so the only variable is the mixture split.
EPOCHS="${EPOCHS:-1.0}"   # must be <= 1.0
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Fraction of the largest feasible no-oversample dataset used for all runs.
# 1.0 = max fair target across all tested mixtures.
TARGET_FRACTION="${TARGET_FRACTION:-1.0}"

# Curated splits for your goal:
# - strong context answer quality (higher correct)
# - still enough refusal/negative for robustness
MIXTURES=(
  "correct:0.75,refusal:0.10,negative:0.15"
  "correct:0.70,refusal:0.15,negative:0.15"
  "correct:0.65,refusal:0.20,negative:0.15"
  "correct:0.60,refusal:0.20,negative:0.20"
)

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found in PATH."
  exit 1
fi

python3 - <<PY
epochs = float("${EPOCHS}")
if not (0.0 < epochs <= 1.0):
    raise SystemExit("EPOCHS must be in (0, 1.0]")
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
  echo "Need ${NPROC_PER_NODE} GPUs, found ${GPU_COUNT}."
  exit 1
fi

REPORT_TO="$(python3 - <<'PY'
try:
    import tensorboard  # noqa: F401
    print("tensorboard")
except Exception:
    try:
        import tensorboardX  # noqa: F401
        print("tensorboard")
    except Exception:
        print("none")
PY
)"
if [[ "${REPORT_TO}" == "none" ]]; then
  echo "TensorBoard package not found; falling back to --report_to none."
fi

if [[ ! -f "${SPLITS_DIR}/stats.json" ]]; then
  echo "Missing ${SPLITS_DIR}/stats.json. Run prepare_sft_splits.py first."
  exit 1
fi

mkdir -p "${RUNS_DIR}" "${TB_ROOT}"
SUMMARY_JSONL="${RUNS_DIR}/summary.jsonl"
: > "${SUMMARY_JSONL}"

# Pick a single fair target train size valid for every mixture (no oversampling).
TARGET_TRAIN_SAMPLES="$(python3 - <<PY
import json

stats_path = "${SPLITS_DIR}/stats.json"
target_fraction = float("${TARGET_FRACTION}")
mixtures = [
    "correct:0.75,refusal:0.10,negative:0.15",
    "correct:0.70,refusal:0.15,negative:0.15",
    "correct:0.65,refusal:0.20,negative:0.15",
    "correct:0.60,refusal:0.20,negative:0.20",
]

with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)
counts = stats["train_counts_by_group"]

def feasible_target(spec: str) -> int:
    ratios = {}
    for part in spec.split(","):
        k, v = part.split(":", 1)
        ratios[k.strip()] = float(v)
    s = sum(ratios.values())
    ratios = {k: v / s for k, v in ratios.items()}
    return int(min(counts[g] / ratios[g] for g in ("correct", "refusal", "negative")))

base = min(feasible_target(m) for m in mixtures)
target = max(1, int(base * target_fraction))
print(target)
PY
)"

echo "RUNS_DIR=${RUNS_DIR}"
echo "TARGET_TRAIN_SAMPLES=${TARGET_TRAIN_SAMPLES}"
echo "EPOCHS=${EPOCHS}"
echo "REPORT_TO=${REPORT_TO}"
echo "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
echo "Checking ${#MIXTURES[@]} mixture splits..."

IDX=0
for MIX in "${MIXTURES[@]}"; do
  RUN_NAME="$(printf "mix_%02d_%s" "${IDX}" "${MIX//,/__}")"
  RUN_NAME="${RUN_NAME//:/-}"
  OUT_DIR="${RUNS_DIR}/${RUN_NAME}"
  LOG_DIR="${TB_ROOT}/${RUN_NAME}"
  mkdir -p "${OUT_DIR}" "${LOG_DIR}"

  echo ">> ${RUN_NAME}"
  EXTRA_FLAGS=()
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    EXTRA_FLAGS+=(--local_files_only)
  fi
  torchrun --standalone --master_addr 127.0.0.1 --nproc_per_node="${NPROC_PER_NODE}" finetune_qwen3_base.py \
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
    --mix_ratios "${MIX}" \
    --target_train_samples "${TARGET_TRAIN_SAMPLES}" \
    --num_train_epochs "${EPOCHS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 1 \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to "${REPORT_TO}" \
    --seed "${SEED}" \
    --bf16 \
    --gradient_checkpointing \
    "${EXTRA_FLAGS[@]}"

  python3 - <<PY
import json
import os

run_name = "${RUN_NAME}"
out_dir = "${OUT_DIR}"
mix = "${MIX}"
summary = "${SUMMARY_JSONL}"
row = {"run_name": run_name, "out_dir": out_dir, "mix": mix}

metrics_path = os.path.join(out_dir, "all_eval_metrics.json")
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
    row["losses"] = losses
    if losses:
        row["mean_group_loss"] = sum(losses.values()) / len(losses)

with open(summary, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY

  IDX=$((IDX + 1))
done

python3 - <<PY
import csv
import json

summary = "${SUMMARY_JSONL}"
rows = []
with open(summary, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

rows = sorted(rows, key=lambda r: r.get("mean_group_loss", float("inf")))
out_json = "${RUNS_DIR}/mixture_ranked.json"
out_csv = "${RUNS_DIR}/mixture_ranked.csv"

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rank", "run_name", "mix", "mean_group_loss", "eval_correct_loss", "eval_refusal_loss", "eval_negative_loss"])
    for i, r in enumerate(rows, start=1):
        ls = r.get("losses", {})
        w.writerow([i, r.get("run_name", ""), r.get("mix", ""), r.get("mean_group_loss", ""), ls.get("eval_correct_loss", ""), ls.get("eval_refusal_loss", ""), ls.get("eval_negative_loss", "")])

print(f"Wrote {out_json}")
print(f"Wrote {out_csv}")
if rows:
    print("Best:", rows[0].get("run_name"), rows[0].get("mix"), "mean_group_loss=", rows[0].get("mean_group_loss"))
PY

if [[ "${REPORT_TO}" == "tensorboard" ]]; then
  echo "TensorBoard:"
  echo "tensorboard --logdir ${TB_ROOT} --host 0.0.0.0 --port 6006"
fi
