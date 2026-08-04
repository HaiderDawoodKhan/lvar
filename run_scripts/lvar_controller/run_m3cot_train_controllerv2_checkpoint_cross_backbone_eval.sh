#!/usr/bin/env bash
set -euo pipefail

# Train ControllerV2 twice on the complete mined M3CoT train set, saving the
# four epoch checkpoints from each run, then evaluate every saved controller
# with both baseline backbones:
#   LVAR-trained  epochs {2, 4, 6, 8} x {LVAR, IVTLR}
#   IVTLR-trained epochs {2, 4, 6, 8} x {LVAR, IVTLR}
# This produces 8 controller checkpoints and 16 test-set evaluations.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="${ROOT_DIR}/configs/qwen2vl_m3cot.yaml"
TRAIN_SCRIPT="${ROOT_DIR}/lvar_scripts/train_controller_sft.py"
INFERENCE_SCRIPT="${ROOT_DIR}/lvar_scripts/infer_lvar_m3cot.py"

# Fixed baseline checkpoints supplied for the M3CoT controller experiment.
LVAR_CHECKPOINT="D:/Haider/IVTLR-Baseline/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth"
IVTLR_CHECKPOINT="D:/Haider/IVTLR-Baseline/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth"
TRAIN_TRACE_PATH="${ROOT_DIR}/outputs/oracle_dataset/train/lvar_ckpt/m3cot_train_traces_lvar_global.jsonl"
OUTPUT_ROOT="${ROOT_DIR}/outputs/controller_sft_m3cot_controllerv2_cross_backbone"

SEED=42
NUM_EPOCHS=8
CHECKPOINT_EVERY=2
CONTROLLER_LR=0.00005
WEIGHT_DECAY=0.01
EPOCHS=(2 4 6 8)

for required_path in "${CONFIG}" "${TRAIN_SCRIPT}" "${INFERENCE_SCRIPT}" "${TRAIN_TRACE_PATH}" "${LVAR_CHECKPOINT}" "${IVTLR_CHECKPOINT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path}" >&2
    exit 2
  fi
done

evaluate_checkpoint() {
  local trained_backbone="$1"
  local epoch="$2"
  local evaluated_backbone="$3"
  local evaluated_checkpoint="$4"
  local controller_checkpoint="${OUTPUT_ROOT}/${trained_backbone}/train/controller_sft_epoch_${epoch}.pt"
  local output_path="${OUTPUT_ROOT}/${trained_backbone}/eval_${evaluated_backbone}/epoch_${epoch}/m3cot_test_predictions.jsonl"

  if [[ ! -f "${controller_checkpoint}" ]]; then
    echo "Expected controller checkpoint was not produced: ${controller_checkpoint}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${output_path}")"
  echo "Evaluating ${trained_backbone}-trained ControllerV2 at epoch ${epoch} with ${evaluated_backbone} backbone"
  "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    --config "${CONFIG}" \
    --checkpoint-path "${evaluated_checkpoint}" \
    --controller-path "${controller_checkpoint}" \
    --output "${output_path}" \
    --no-vlm-checkpoint \
    --no-nucleus-insertion
}

train_and_evaluate_checkpoints() {
  local trained_backbone="$1"
  local training_checkpoint="$2"
  local train_output_dir="${OUTPUT_ROOT}/${trained_backbone}/train"

  mkdir -p "${train_output_dir}"
  echo "Training ControllerV2 on the complete mined train set with ${trained_backbone} backbone"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --checkpoint-path "${training_checkpoint}" \
    --trace-jsonl "${TRAIN_TRACE_PATH}" \
    --output-dir "${train_output_dir}" \
    --seed "${SEED}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --phase3-override "dataset_partition=train" \
    --phase3-override "phase4_vlm_checkpoint_path=null" \
    --phase3-override "controller_max_steps=22" \
    --phase3-override "num_epochs=${NUM_EPOCHS}" \
    --phase3-override "controller_lr=${CONTROLLER_LR}" \
    --phase3-override "weight_decay=${WEIGHT_DECAY}" \
    --phase3-override "use_one_replay_setting=true" \
    --phase3-override "replay_setting=global" \
    --phase3-override "decision_block_normalized=true" \
    --phase3-override "multi_hot_patch_labels=false" \
    --phase3-override "use_type_loss_weights=true" \
    --phase3-v2-override "enabled=false"

  for epoch in "${EPOCHS[@]}"; do
    evaluate_checkpoint "${trained_backbone}" "${epoch}" "lvar" "${LVAR_CHECKPOINT}"
    evaluate_checkpoint "${trained_backbone}" "${epoch}" "ivtlr" "${IVTLR_CHECKPOINT}"
  done
}

train_and_evaluate_checkpoints "lvar" "${LVAR_CHECKPOINT}"
train_and_evaluate_checkpoints "ivtlr" "${IVTLR_CHECKPOINT}"

echo "Complete: 2 training runs, 8 ControllerV2 epoch checkpoints, and 16 test evaluations under ${OUTPUT_ROOT}."
