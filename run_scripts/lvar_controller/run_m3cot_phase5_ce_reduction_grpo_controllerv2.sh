#!/usr/bin/env bash
set -euo pipefail

# Refine the SFT-trained ControllerV2 (the MLP controller, never the
# transformer controller) with the dense, sequential CE-reduction GRPO reward,
# then evaluate the resulting controller on M3CoT's test partition.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="${CONFIG:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
TRAIN_SCRIPT="${ROOT_DIR}/lvar_scripts/train_grpo.py"
INFERENCE_SCRIPT="${ROOT_DIR}/lvar_scripts/infer_lvar_m3cot.py"

# Override these for the machine that owns the checkpoints. The defaults mirror
# the model/controller paths in configs/qwen2vl_m3cot.yaml.
BASE_MODEL_CHECKPOINT="${BASE_MODEL_CHECKPOINT:-D:/Haider/IVTLR-Baseline/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}"
SFT_CONTROLLER_CHECKPOINT="${SFT_CONTROLLER_CHECKPOINT:-D:/Haider/lvar/outputs/controller_sft_m3cot/controller_sft.pt}"
PHASE4_VLM_CHECKPOINT="${PHASE4_VLM_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/grpo_phase5_m3cot_ce_reduction_controllerv2}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_DIR}/eval_test/m3cot_test_predictions.jsonl}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
CONTROLLER_MAX_STEPS="${CONTROLLER_MAX_STEPS:-22}"

for required_path in "${CONFIG}" "${TRAIN_SCRIPT}" "${INFERENCE_SCRIPT}" "${BASE_MODEL_CHECKPOINT}" "${SFT_CONTROLLER_CHECKPOINT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path}" >&2
    exit 2
  fi
done
if [[ -n "${PHASE4_VLM_CHECKPOINT}" && ! -f "${PHASE4_VLM_CHECKPOINT}" ]]; then
  echo "Phase 4 VLM checkpoint not found: ${PHASE4_VLM_CHECKPOINT}" >&2
  exit 2
fi

train_args=(
  --config "${CONFIG}"
  --checkpoint-path "${BASE_MODEL_CHECKPOINT}"
  --model-override "controller_architecture=mlp"
  --phase5-override "controller_checkpoint_path=${SFT_CONTROLLER_CHECKPOINT}"
  --phase5-override "phase4_vlm_checkpoint_path=null"
  --phase5-override "output_dir=${OUTPUT_DIR}"
  --phase5-override "reward_mode=ce_reduction"
  --phase5-override "use_baseline_advantage_weighting=false"
  --phase5-override "controller_max_steps=${CONTROLLER_MAX_STEPS}"
  --phase5-override "ce_reduction_max_steps=5"
  --phase5-override "ce_reduction_aggregation=mean"
  --checkpoint-every "${CHECKPOINT_EVERY}"
)
if [[ -n "${PHASE4_VLM_CHECKPOINT}" ]]; then
  train_args+=(--phase5-override "phase4_vlm_checkpoint_path=${PHASE4_VLM_CHECKPOINT}")
fi
if [[ -n "${MAX_EXAMPLES}" ]]; then
  train_args+=(--phase5-override "max_examples=${MAX_EXAMPLES}")
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${EVAL_OUTPUT}")"
echo "Training ControllerV2 from SFT with sequential CE-reduction GRPO"
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${train_args[@]}"

FINAL_CONTROLLER_CHECKPOINT="${OUTPUT_DIR}/phase5_controller.pt"
if [[ ! -f "${FINAL_CONTROLLER_CHECKPOINT}" ]]; then
  echo "GRPO did not produce ${FINAL_CONTROLLER_CHECKPOINT}" >&2
  exit 1
fi

eval_args=(
  --config "${CONFIG}"
  --checkpoint-path "${BASE_MODEL_CHECKPOINT}"
  --model-override "controller_architecture=mlp"
  --controller-path "${FINAL_CONTROLLER_CHECKPOINT}"
  --output "${EVAL_OUTPUT}"
  --max-controller-steps "${CONTROLLER_MAX_STEPS}"
  --no-nucleus-insertion
)
if [[ -n "${PHASE4_VLM_CHECKPOINT}" ]]; then
  eval_args+=(--vlm-path "${PHASE4_VLM_CHECKPOINT}")
else
  eval_args+=(--no-vlm-checkpoint)
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  eval_args+=(--limit "${EVAL_LIMIT}")
fi

echo "Evaluating the CE-reduction-refined ControllerV2 on the M3CoT test partition"
"${PYTHON_BIN}" "${INFERENCE_SCRIPT}" "${eval_args[@]}"

echo "Complete: ${FINAL_CONTROLLER_CHECKPOINT} and ${EVAL_OUTPUT}"
