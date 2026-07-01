#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
DATASET_PARTITION="${DATASET_PARTITION:-test}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
CONTROLLER_CHECKPOINT="${CONTROLLER_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference/${DATASET_PARTITION}_controller_rollouts}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-1.5}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
STEP_ENTROPY_TOP_K="${STEP_ENTROPY_TOP_K:-}"
USE_COARSE_CONTEXT="${USE_COARSE_CONTEXT:-false}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${LVAR_PHASE1_CHECKPOINT}" ]]; then
  echo "LVAR phase 1 checkpoint not found: ${LVAR_PHASE1_CHECKPOINT}" >&2
  exit 2
fi

case "${DATASET_PARTITION}" in
  train|validation|test)
    ;;
  *)
    echo "DATASET_PARTITION must be one of: train, validation, test" >&2
    exit 2
    ;;
esac

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

controller_args=()
if [[ -n "${CONTROLLER_CHECKPOINT}" ]]; then
  if [[ ! -f "${CONTROLLER_CHECKPOINT}" ]]; then
    echo "Controller checkpoint not found: ${CONTROLLER_CHECKPOINT}" >&2
    exit 2
  fi
  controller_args=(--controller-path "${CONTROLLER_CHECKPOINT}")
fi

step_entropy_args=()
if [[ -n "${STEP_ENTROPY_TOP_K}" ]]; then
  step_entropy_args=(--step-entropy-top-k "${STEP_ENTROPY_TOP_K}")
fi

coarse_args=()
context_suffix="full"
if [[ "${USE_COARSE_CONTEXT}" == "true" ]]; then
  coarse_args=(--use-coarse-context)
  context_suffix="coarse"
elif [[ "${USE_COARSE_CONTEXT}" != "false" ]]; then
  echo "USE_COARSE_CONTEXT must be true or false" >&2
  exit 2
fi

run_dir="${OUTPUT_ROOT}/G${NUM_ROLLOUTS}_temp${TEMPERATURE}_seed${SEED}_${context_suffix}"
output_path="${run_dir}/m3cot_${DATASET_PARTITION}_controller_rollouts_G${NUM_ROLLOUTS}_temp${TEMPERATURE}_${context_suffix}.jsonl"
mkdir -p "${run_dir}"

echo "Running sampled-controller rollout inference"
echo "  partition: ${DATASET_PARTITION}"
echo "  rollouts:  ${NUM_ROLLOUTS}"
echo "  temp:      ${TEMPERATURE}"
echo "  output:    ${output_path}"

python lvar_scripts/infer_lvar_m3cot_rollouts.py \
  --config "${CONFIG}" \
  --dataset-partition "${DATASET_PARTITION}" \
  --vlm-path "${LVAR_PHASE1_CHECKPOINT}" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --output "${output_path}" \
  "${controller_args[@]}" \
  "${step_entropy_args[@]}" \
  "${coarse_args[@]}" \
  "${limit_args[@]}"

echo "Done. Rollout outputs are under ${run_dir}"
