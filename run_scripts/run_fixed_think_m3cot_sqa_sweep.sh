#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 && ( -z "${M3COT_CHECKPOINT_PATH:-}" || -z "${SQA_CHECKPOINT_PATH:-}" ) ]]; then
  cat <<'USAGE' >&2
Usage:
  run_scripts/run_fixed_think_m3cot_sqa_sweep.sh <m3cot_checkpoint> <sqa_checkpoint>

Environment overrides:
  M3COT_CHECKPOINT_PATH       M3CoT checkpoint path if not passed as arg 1
  SQA_CHECKPOINT_PATH         ScienceQA checkpoint path if not passed as arg 2
  OUTPUT_ROOT                 Output root, default outputs/inference/fixed_think_sweep
  CONTEXT                     global, coarse, full_context, or global_mean; default global
  IMAGE_SIZE                  Image size passed to inference; default 280
  LIMIT                       Optional dataset limit for quick smoke runs
  SEED                        Optional seed override
  ADD_ANSWER_INSTRUCTION      Set to 1 to append tagged-answer instruction
USAGE
  exit 1
fi

M3COT_CHECKPOINT_PATH="${1:-${M3COT_CHECKPOINT_PATH:-}}"
SQA_CHECKPOINT_PATH="${2:-${SQA_CHECKPOINT_PATH:-}}"

if [[ -z "${M3COT_CHECKPOINT_PATH}" ]]; then
  echo "Missing M3CoT checkpoint path." >&2
  exit 1
fi
if [[ -z "${SQA_CHECKPOINT_PATH}" ]]; then
  echo "Missing ScienceQA checkpoint path." >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference/fixed_think_sweep}"
CONTEXT="${CONTEXT:-global}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
PARTITIONS=(test validation)
THINK_STEPS=(0 1 2 3 4 5 6 7 8 9 10)

run_one() {
  local dataset_key="$1"
  local config_path="$2"
  local checkpoint_path="$3"
  local partition="$4"
  local think_steps="$5"

  local run_dir="${OUTPUT_ROOT}/${dataset_key}/${partition}/fixed_think_steps_${think_steps}"
  local output_path="${run_dir}/${dataset_key}_${partition}_predictions_fixed-think-${think_steps}_${CONTEXT}.jsonl"
  mkdir -p "${run_dir}"

  local args=(
    --config "${config_path}"
    --dataset-partition "${partition}"
    --num-think-steps "${think_steps}"
    --context "${CONTEXT}"
    --image-size "${IMAGE_SIZE}"
    --checkpoint-path "${checkpoint_path}"
    --output "${output_path}"
  )

  if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${SEED:-}" ]]; then
    args+=(--seed "${SEED}")
  fi
  if [[ "${ADD_ANSWER_INSTRUCTION:-0}" == "1" ]]; then
    args+=(--add-answer-instruction)
  fi

  echo "Running ${dataset_key} ${partition} fixed THINK=${think_steps}"
  python lvar_scripts/infer_fixed_think_m3cot.py "${args[@]}"
}

for dataset_key in m3cot sqa; do
  if [[ "${dataset_key}" == "m3cot" ]]; then
    config_path="configs/qwen2vl_m3cot.yaml"
    checkpoint_path="${M3COT_CHECKPOINT_PATH}"
  else
    config_path="configs/qwen2vl_sqa.yaml"
    checkpoint_path="${SQA_CHECKPOINT_PATH}"
  fi

  for partition in "${PARTITIONS[@]}"; do
    for think_steps in "${THINK_STEPS[@]}"; do
      run_one "${dataset_key}" "${config_path}" "${checkpoint_path}" "${partition}" "${think_steps}"
    done
  done
done
