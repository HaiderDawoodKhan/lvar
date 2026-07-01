#!/usr/bin/env bash
set -euo pipefail

cat <<'USAGE' >/dev/null
Train the binary latent-depth STOP/CONTINUE controller from fixed-THINK sweep outputs.

Common usage:
  LVAR_CHECKPOINT_PATH=/path/to/lvar.pth bash run_scripts/run_latent_depth_controller_training.sh

Environment overrides:
  CONFIG                  Config path, default configs/qwen2vl_m3cot.yaml
  SWEEP_OUTPUT_ROOT       Fixed-THINK sweep root, default outputs/inference/fixed_think_sweep
  TRAIN_OUTPUT_ROOT       Output root, default outputs/latent_depth_controller_m3cot
  DATASET_PARTITION       Dataset partition to train from, default test
  CHECKPOINT_NAME         Sweep checkpoint folder to consume, default lvar
  MODEL_CHECKPOINT_PATH   Model checkpoint for feature extraction. Defaults to LVAR_CHECKPOINT_PATH.
  FIXED_THINK_GLOB        Explicit JSONL glob. Defaults to the sweep path for partition/checkpoint.
  MAX_DEPTH               Maximum latent depth, default 10
  TARGET_POLICY           earliest_correct or all_correct, default earliest_correct
  CONTEXT                 global, coarse, full_context, or global_mean; default global
  IMAGE_SIZE              Image size, default 280
  LIMIT                   Optional dataset limit for smoke runs
  SEED                    Seed, default 42
  NUM_EPOCHS              Number of epochs, default 1
  LEARNING_RATE           Learning rate, default 0.0001
  VALIDATION_FRACTION     Held-out fraction from supervision examples, default 0.1
  CONTROLLER_HIDDEN_SIZE  Transformer width, default 512
  CONTROLLER_LAYERS       Transformer layers, default 2
  CONTROLLER_HEADS        Transformer heads, default 8
  MAX_PROMPT_TOKENS       Prompt tail length, default 10
USAGE

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-outputs/inference/fixed_think_sweep}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/latent_depth_controller_m3cot}"
DATASET_PARTITION="${DATASET_PARTITION:-test}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-lvar}"
MODEL_CHECKPOINT_PATH="${MODEL_CHECKPOINT_PATH:-${LVAR_CHECKPOINT_PATH:-}}"
MAX_DEPTH="${MAX_DEPTH:-10}"
TARGET_POLICY="${TARGET_POLICY:-earliest_correct}"
CONTEXT="${CONTEXT:-global}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
CONTROLLER_HIDDEN_SIZE="${CONTROLLER_HIDDEN_SIZE:-512}"
CONTROLLER_LAYERS="${CONTROLLER_LAYERS:-2}"
CONTROLLER_HEADS="${CONTROLLER_HEADS:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-10}"

FIXED_THINK_GLOB="${FIXED_THINK_GLOB:-${SWEEP_OUTPUT_ROOT}/m3cot/${DATASET_PARTITION}/${CHECKPOINT_NAME}/fixed_think_steps_*/*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_ROOT}/${DATASET_PARTITION}/${CHECKPOINT_NAME}_${TARGET_POLICY}}"
MANIFEST_PATH="${MANIFEST_PATH:-${TRAIN_OUTPUT_ROOT}/latent_depth_controller_runs.jsonl}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${MANIFEST_PATH}")"

args=(
  --config "${CONFIG}"
  --fixed-think-glob "${FIXED_THINK_GLOB}"
  --max-depth "${MAX_DEPTH}"
  --target-policy "${TARGET_POLICY}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-partition "${DATASET_PARTITION}"
  --seed "${SEED}"
  --num-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --context "${CONTEXT}"
  --image-size "${IMAGE_SIZE}"
  --controller-hidden-size "${CONTROLLER_HIDDEN_SIZE}"
  --controller-layers "${CONTROLLER_LAYERS}"
  --controller-heads "${CONTROLLER_HEADS}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
)

if [[ -n "${MODEL_CHECKPOINT_PATH}" ]]; then
  args+=(--checkpoint-path "${MODEL_CHECKPOINT_PATH}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi

echo "Training latent-depth controller"
echo "  fixed-think glob: ${FIXED_THINK_GLOB}"
echo "  output dir:       ${OUTPUT_DIR}"
python lvar_scripts/train_latent_depth_controller.py "${args[@]}"

python - "${MANIFEST_PATH}" "${OUTPUT_DIR}" "${CONFIG}" "${FIXED_THINK_GLOB}" \
  "${DATASET_PARTITION}" "${CHECKPOINT_NAME}" "${TARGET_POLICY}" "${MAX_DEPTH}" "${CONTEXT}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    output_dir,
    config,
    fixed_think_glob,
    dataset_partition,
    checkpoint_name,
    target_policy,
    max_depth,
    context,
) = sys.argv[1:]

row = {
    "output_dir": output_dir,
    "config": config,
    "fixed_think_glob": fixed_think_glob,
    "dataset_partition": dataset_partition,
    "checkpoint_name": checkpoint_name,
    "target_policy": target_policy,
    "max_depth": int(max_depth),
    "context": context,
    "checkpoint_path": str(Path(output_dir) / "latent_depth_controller.pt"),
    "summary_path": str(Path(output_dir) / "latent_depth_controller_summary.json"),
    "loss_history_path": str(Path(output_dir) / "latent_depth_controller_losses.jsonl"),
}
with open(manifest_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
PY

echo "Done. Manifest: ${MANIFEST_PATH}"
