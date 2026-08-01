#!/usr/bin/env bash
set -euo pipefail

# One-command controller experiment:
#   1. train on the complete mined M3CoT train set with the LVAR backbone,
#      then evaluate on LVAR and IVTLR baselines;
#   2. train on the same full dataset with the IVTLR backbone,
#      then evaluate on LVAR and IVTLR baselines.
#
# The baseline checkpoints are intentionally fixed here so no checkpoint paths
# need to be passed at invocation time.

CONFIG="configs/qwen2vl_m3cot.yaml"
LVAR_CHECKPOINT="D:/Haider/IVTLR-Baseline/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth"
IVTLR_CHECKPOINT="D:/Haider/IVTLR-Baseline/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth"
TRAIN_TRACE_PATH="outputs/oracle_dataset/train/lvar_ckpt/m3cot_train_traces_lvar_global.jsonl"
OUTPUT_ROOT="outputs/controller_sft_m3cot_train_cross_backbone"
SEED=42
NUM_EPOCHS=20
CONTROLLER_LR=0.00005
WEIGHT_DECAY=0.01
CHECKPOINT_EVERY=4

for required_path in "${CONFIG}" "${TRAIN_TRACE_PATH}" "${LVAR_CHECKPOINT}" "${IVTLR_CHECKPOINT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path}" >&2
    exit 2
  fi
done

run_inference() {
  local trained_backbone="$1"
  local controller_checkpoint="$2"
  local baseline_name="$3"
  local baseline_checkpoint="$4"
  local output_path="${OUTPUT_ROOT}/${trained_backbone}/eval_${baseline_name}/m3cot_test_predictions.jsonl"

  mkdir -p "$(dirname "${output_path}")"
  echo "Evaluating ${trained_backbone}-trained controller using ${baseline_name} baseline"
  python lvar_scripts/infer_lvar_m3cot.py \
    --config "${CONFIG}" \
    --checkpoint-path "${baseline_checkpoint}" \
    --controller-path "${controller_checkpoint}" \
    --output "${output_path}" \
    --no-nucleus-insertion
}

train_and_evaluate() {
  local backbone_name="$1"
  local backbone_checkpoint="$2"
  local train_output_dir="${OUTPUT_ROOT}/${backbone_name}/train"
  local controller_checkpoint="${train_output_dir}/controller_sft.pt"

  mkdir -p "${train_output_dir}"
  echo "Training controller on ${backbone_name} backbone with ${TRAIN_TRACE_PATH}"
  python lvar_scripts/train_controller_sft.py \
    --config "${CONFIG}" \
    --checkpoint-path "${backbone_checkpoint}" \
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

  if [[ ! -f "${controller_checkpoint}" ]]; then
    echo "Training did not produce ${controller_checkpoint}" >&2
    exit 1
  fi

  run_inference "${backbone_name}" "${controller_checkpoint}" "lvar" "${LVAR_CHECKPOINT}"
  run_inference "${backbone_name}" "${controller_checkpoint}" "ivtlr" "${IVTLR_CHECKPOINT}"
}

train_and_evaluate "lvar" "${LVAR_CHECKPOINT}"
train_and_evaluate "ivtlr" "${IVTLR_CHECKPOINT}"

echo "Complete. Two training runs and four test inference runs are under ${OUTPUT_ROOT}."
