#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-outputs/inference/fixed_think_sweep}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/latent_depth_controller_m3cot/eval}"
DATASET_PARTITION="${DATASET_PARTITION:-test}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-lvar}"
CONTROLLER_CHECKPOINT="${CONTROLLER_CHECKPOINT:-outputs/latent_depth_controller_m3cot/test/lvar_all_correct/latent_depth_controller.pt}"
MODEL_CHECKPOINT_PATH="${MODEL_CHECKPOINT_PATH:-${LVAR_CHECKPOINT_PATH:-}}"
FIXED_THINK_GLOB="${FIXED_THINK_GLOB:-${SWEEP_OUTPUT_ROOT}/m3cot/${DATASET_PARTITION}/${CHECKPOINT_NAME}/fixed_think_steps_*/*.jsonl}"
MAX_DEPTH="${MAX_DEPTH:-10}"
TARGET_POLICY="${TARGET_POLICY:-all_correct}"
CONTEXT="${CONTEXT:-global}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
THRESHOLD="${THRESHOLD:-0.5}"
SEED="${SEED:-42}"

if [[ ! -f "${CONTROLLER_CHECKPOINT}" ]]; then
  echo "Controller checkpoint not found: ${CONTROLLER_CHECKPOINT}" >&2
  exit 2
fi

output_dir="${EVAL_OUTPUT_ROOT}/${DATASET_PARTITION}/${CHECKPOINT_NAME}_${TARGET_POLICY}"
mkdir -p "${output_dir}"
output_path="${output_dir}/latent_depth_controller_${DATASET_PARTITION}_predictions.jsonl"

args=(
  --config "${CONFIG}"
  --controller-checkpoint "${CONTROLLER_CHECKPOINT}"
  --fixed-think-glob "${FIXED_THINK_GLOB}"
  --output "${output_path}"
  --dataset-partition "${DATASET_PARTITION}"
  --max-depth "${MAX_DEPTH}"
  --target-policy "${TARGET_POLICY}"
  --context "${CONTEXT}"
  --image-size "${IMAGE_SIZE}"
  --threshold "${THRESHOLD}"
  --seed "${SEED}"
)

if [[ -n "${MODEL_CHECKPOINT_PATH}" ]]; then
  args+=(--checkpoint-path "${MODEL_CHECKPOINT_PATH}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi

echo "Evaluating latent-depth controller"
echo "  controller:       ${CONTROLLER_CHECKPOINT}"
echo "  fixed-think glob: ${FIXED_THINK_GLOB}"
echo "  output:           ${output_path}"
python lvar_scripts/eval_latent_depth_controller.py "${args[@]}"
