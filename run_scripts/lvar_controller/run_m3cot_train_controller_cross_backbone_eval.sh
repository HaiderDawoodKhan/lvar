#!/usr/bin/env bash
set -euo pipefail

# Train the three-action (PATCH / THINK / STOP) controller on the complete
# mined M3CoT train set, then evaluate that same controller on both baselines.
#
# Usage:
#   BACKBONE=lvar LVAR_CHECKPOINT=/path/lvar.pth IVTLR_CHECKPOINT=/path/ivtlr.pth \
#     bash run_scripts/lvar_controller/run_m3cot_train_controller_cross_backbone_eval.sh
#
# Set BACKBONE=ivtlr to train against the IVTLR backbone. The train traces can
# be changed with TRAIN_TRACE_PATH when a separate IVTLR-mined dataset exists.

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
BACKBONE="${BACKBONE:-lvar}"
LVAR_CHECKPOINT="${LVAR_CHECKPOINT:-}"
IVTLR_CHECKPOINT="${IVTLR_CHECKPOINT:-}"
TRAIN_TRACE_PATH="${TRAIN_TRACE_PATH:-outputs/oracle_dataset/train/lvar_ckpt/m3cot_train_traces_lvar_global.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/controller_sft_m3cot_train_cross_backbone}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
CONTROLLER_LR="${CONTROLLER_LR:-0.00005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
LIMIT="${LIMIT:-}"

case "${BACKBONE}" in
  lvar) TRAIN_BACKBONE_CHECKPOINT="${LVAR_CHECKPOINT}" ;;
  ivtlr) TRAIN_BACKBONE_CHECKPOINT="${IVTLR_CHECKPOINT}" ;;
  *)
    echo "BACKBONE must be 'lvar' or 'ivtlr', got: ${BACKBONE}" >&2
    exit 2
    ;;
esac

for required_path in "${CONFIG}" "${TRAIN_TRACE_PATH}" "${LVAR_CHECKPOINT}" "${IVTLR_CHECKPOINT}" "${TRAIN_BACKBONE_CHECKPOINT}"; do
  if [[ -z "${required_path}" || ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path:-<unset>}" >&2
    exit 2
  fi
done

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

RUN_ROOT="${OUTPUT_ROOT}/${BACKBONE}"
TRAIN_OUTPUT_DIR="${RUN_ROOT}/train"
CONTROLLER_CHECKPOINT="${TRAIN_OUTPUT_DIR}/controller_sft.pt"
mkdir -p "${TRAIN_OUTPUT_DIR}" "${RUN_ROOT}/eval_lvar" "${RUN_ROOT}/eval_ivtlr"

echo "Training controller on ${BACKBONE} backbone with ${TRAIN_TRACE_PATH}"
python lvar_scripts/train_controller_sft.py \
  --config "${CONFIG}" \
  --checkpoint-path "${TRAIN_BACKBONE_CHECKPOINT}" \
  --trace-jsonl "${TRAIN_TRACE_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  --seed "${SEED}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --phase3-override "dataset_partition=train" \
  --phase3-override "phase4_vlm_checkpoint_path=null" \
  --phase3-override "num_epochs=${NUM_EPOCHS}" \
  --phase3-override "controller_lr=${CONTROLLER_LR}" \
  --phase3-override "weight_decay=${WEIGHT_DECAY}" \
  --phase3-override "use_one_replay_setting=true" \
  --phase3-override "replay_setting=global" \
  --phase3-override "decision_block_normalized=true" \
  --phase3-override "multi_hot_patch_labels=true" \
  --phase3-override "multi_hot_patch_target_mode=binary" \
  --phase3-override "use_type_loss_weights=true" \
  --phase3-v2-override "enabled=false" \
  "${LIMIT_ARGS[@]}"

if [[ ! -f "${CONTROLLER_CHECKPOINT}" ]]; then
  echo "Training did not produce ${CONTROLLER_CHECKPOINT}" >&2
  exit 1
fi

run_inference() {
  local baseline_name="$1"
  local baseline_checkpoint="$2"
  local output_path="${RUN_ROOT}/eval_${baseline_name}/m3cot_test_predictions.jsonl"

  echo "Evaluating ${BACKBONE}-trained controller using ${baseline_name} baseline"
  python lvar_scripts/infer_lvar_m3cot.py \
    --config "${CONFIG}" \
    --checkpoint-path "${baseline_checkpoint}" \
    --controller-path "${CONTROLLER_CHECKPOINT}" \
    --output "${output_path}" \
    --nucleus-insertion \
    --nucleus-insertion-scope patch \
    --nucleus-insertion-top-p 0.9 \
    --nucleus-insertion-max-indices 4 \
    "${LIMIT_ARGS[@]}"
}

run_inference "lvar" "${LVAR_CHECKPOINT}"
run_inference "ivtlr" "${IVTLR_CHECKPOINT}"

echo "Complete. Outputs: ${RUN_ROOT}"
