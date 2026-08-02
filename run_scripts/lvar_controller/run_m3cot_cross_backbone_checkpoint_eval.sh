#!/usr/bin/env bash
set -euo pipefail

# Evaluate the epoch checkpoints from the cross-backbone controller-SFT run.
#
# The four controller checkpoints from each training method are evaluated with
# both backbone checkpoints:
#   lvar/train/controller_sft_epoch_{4,8,12,16}.pt   x  LVAR, IVTLR
#   ivtlr/train/controller_sft_epoch_{4,8,12,16}.pt  x  LVAR, IVTLR
#
# This produces 16 sequential inference runs. Epoch 20 is intentionally not
# included. Set SKIP_EXISTING=true to resume without overwriting outputs.
#
# Example:
#   bash run_scripts/lvar_controller/run_m3cot_cross_backbone_checkpoint_eval.sh
#
# Path overrides are useful when running on a different machine:
#   OUTPUT_ROOT=/path/to/outputs/controller_sft_m3cot_train_cross_backbone \
#   LVAR_CHECKPOINT=/path/to/lvar_backbone.pth \
#   IVTLR_CHECKPOINT=/path/to/ivtlr_backbone.pth \
#   bash run_scripts/lvar_controller/run_m3cot_cross_backbone_checkpoint_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/controller_sft_m3cot_train_cross_backbone}"

# These defaults match the paths used by the training run/configuration. They
# can be overridden with paths appropriate to the host running this script.
LVAR_CHECKPOINT="${LVAR_CHECKPOINT:-D:/Haider/IVTLR-Baseline/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}"
IVTLR_CHECKPOINT="${IVTLR_CHECKPOINT:-D:/Haider/IVTLR-Baseline/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth}"

EPOCHS=(4 8 12 16)
SKIP_EXISTING="${SKIP_EXISTING:-false}"
DRY_RUN="${DRY_RUN:-false}"

INFERENCE_SCRIPT="${ROOT_DIR}/lvar_scripts/infer_lvar_m3cot.py"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Required config not found: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${INFERENCE_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFERENCE_SCRIPT}" >&2
  exit 2
fi
for required_path in "${LVAR_CHECKPOINT}" "${IVTLR_CHECKPOINT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required backbone checkpoint not found: ${required_path}" >&2
    exit 2
  fi
done

for trained_method in lvar ivtlr; do
  for epoch in "${EPOCHS[@]}"; do
    controller_checkpoint="${OUTPUT_ROOT}/${trained_method}/train/controller_sft_epoch_${epoch}.pt"
    if [[ ! -f "${controller_checkpoint}" ]]; then
      echo "Required controller checkpoint not found: ${controller_checkpoint}" >&2
      exit 2
    fi
  done
done

run_inference() {
  local trained_method="$1"
  local epoch="$2"
  local evaluated_model="$3"
  local backbone_checkpoint="$4"
  local controller_checkpoint="${OUTPUT_ROOT}/${trained_method}/train/controller_sft_epoch_${epoch}.pt"
  local output_dir="${OUTPUT_ROOT}/${trained_method}/eval_${evaluated_model}/epoch_${epoch}"
  local output_path="${output_dir}/m3cot_test_predictions.jsonl"

  if [[ "${SKIP_EXISTING}" == "true" && -s "${output_path}" ]]; then
    echo "Skipping existing output: trained=${trained_method} epoch=${epoch} model=${evaluated_model} path=${output_path}"
    skipped_count=$((skipped_count + 1))
    return
  fi

  mkdir -p "${output_dir}"
  echo "Evaluating ${trained_method}-trained checkpoint at epoch ${epoch} with ${evaluated_model} backbone"
  echo "  controller: ${controller_checkpoint}"
  echo "  backbone:   ${backbone_checkpoint}"
  echo "  output:     ${output_path}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    planned_count=$((planned_count + 1))
    return
  fi

  launched_count=$((launched_count + 1))
  "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    --config "${CONFIG}" \
    --checkpoint-path "${backbone_checkpoint}" \
    --controller-path "${controller_checkpoint}" \
    --output "${output_path}" \
    --no-vlm-checkpoint \
    --no-nucleus-insertion
}

launched_count=0
planned_count=0
skipped_count=0
for trained_method in lvar ivtlr; do
  for epoch in "${EPOCHS[@]}"; do
    run_inference "${trained_method}" "${epoch}" "lvar" "${LVAR_CHECKPOINT}"
    run_inference "${trained_method}" "${epoch}" "ivtlr" "${IVTLR_CHECKPOINT}"
  done
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run complete. Would launch ${planned_count} evaluation runs under ${OUTPUT_ROOT}."
else
  echo "Complete. Launched ${launched_count} evaluation runs and skipped ${skipped_count} existing outputs under ${OUTPUT_ROOT}."
fi
