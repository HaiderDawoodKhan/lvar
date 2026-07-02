#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
DATASET_PARTITION="${DATASET_PARTITION:-validation}"
THINK_STEPS="${THINK_STEPS:-2}"
CONTEXT="${CONTEXT:-global}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference/${DATASET_PARTITION}_fixed_think}"

run_dir="${OUTPUT_ROOT}/evaluated_by_lvar_ckpt/fixed_think_steps_${THINK_STEPS}"
mkdir -p "${run_dir}"

output_path="${run_dir}/m3cot_${DATASET_PARTITION}_predictions_evaluated-by_lvar_fixed-think-${THINK_STEPS}_${CONTEXT}.jsonl"

ARGS=(
  --config "${CONFIG}"
  --dataset-partition "${DATASET_PARTITION}"
  --num-think-steps "${THINK_STEPS}"
  --context "${CONTEXT}"
  --image-size "${IMAGE_SIZE}"
  --output "${output_path}"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

if [[ "${ADD_ANSWER_INSTRUCTION:-0}" == "1" ]]; then
  ARGS+=(--add-answer-instruction)
fi

python lvar_scripts/infer_fixed_think_m3cot.py "${ARGS[@]}"
