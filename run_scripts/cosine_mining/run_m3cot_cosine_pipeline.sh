#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosine_similarity_m3cot}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
TOP_K=3
MAX_RATIONALE_STEPS=8
MAX_CONTROLLER_STEPS="${MAX_CONTROLLER_STEPS:-}"
MINING_MODE="${MINING_MODE:-single_pass}"
if [[ -z "${MAX_CONTROLLER_STEPS:-}" ]]; then
  if [[ "${MINING_MODE}" == "sequential" ]]; then
    MAX_CONTROLLER_STEPS=33
  else
    MAX_CONTROLLER_STEPS=25
  fi
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH or pass the base LVAR checkpoint as the first argument." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

test_trace="${OUTPUT_ROOT}/mined/test/m3cot_test_cosine_k3.jsonl"
train_trace="${OUTPUT_ROOT}/mined/train/m3cot_train_cosine_k3.jsonl"
think_replay="${OUTPUT_ROOT}/replay/test/think/m3cot_test_predictions.jsonl"
no_think_replay="${OUTPUT_ROOT}/replay/test/no_think/m3cot_test_predictions.jsonl"
controller_dir="${OUTPUT_ROOT}/controller_sft"
controller_checkpoint="${controller_dir}/controller_sft.pt"
controller_predictions="${OUTPUT_ROOT}/controller_inference/test/m3cot_test_predictions.jsonl"

mkdir -p \
  "$(dirname "${test_trace}")" \
  "$(dirname "${train_trace}")" \
  "$(dirname "${think_replay}")" \
  "$(dirname "${no_think_replay}")" \
  "${controller_dir}" \
  "$(dirname "${controller_predictions}")"

echo "[1/6] Mining M3CoT test traces with top-${TOP_K} cosine patches..."
python lvar_scripts/mine_cosine_similarity.py \
  --config "${CONFIG}" \
  --dataset-partition test \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --top-k "${TOP_K}" \
  --max-steps "${MAX_RATIONALE_STEPS}" \
  --mining-mode "${MINING_MODE}" \
  --seed "${SEED}" \
  --output "${test_trace}" \
  --resume \
  "${limit_args[@]}"

echo "[2/6] Replaying test traces (raw ${MINING_MODE} trace)..."
python lvar_scripts/eval_mined_traces_m3cot.py \
  --config "${CONFIG}" \
  --dataset-partition test \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --context global \
  --trace-variant raw \
  --seed "${SEED}" \
  --trace-path "${test_trace}" \
  --output "${think_replay}" \
  "${limit_args[@]}"

echo "[3/6] Replaying test traces without THINK..."
python lvar_scripts/eval_mined_traces_m3cot.py \
  --config "${CONFIG}" \
  --dataset-partition test \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --context global \
  --trace-variant no_think \
  --seed "${SEED}" \
  --trace-path "${test_trace}" \
  --output "${no_think_replay}" \
  "${limit_args[@]}"

echo "[4/6] Mining M3CoT train traces with top-${TOP_K} cosine patches..."
python lvar_scripts/mine_cosine_similarity.py \
  --config "${CONFIG}" \
  --dataset-partition train \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --top-k "${TOP_K}" \
  --max-steps "${MAX_RATIONALE_STEPS}" \
  --mining-mode "${MINING_MODE}" \
  --seed "${SEED}" \
  --output "${train_trace}" \
  --resume \
  "${limit_args[@]}"

echo "[5/6] Training the controller with sequential cross-entropy targets..."
python lvar_scripts/train_controller_sft.py \
  --config "${CONFIG}" \
  --trace-jsonl "${train_trace}" \
  --output-dir "${controller_dir}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --seed "${SEED}" \
  --phase3-override "dataset_partition=train" \
  --phase3-override "phase4_vlm_checkpoint_path=null" \
  --phase3-override "phase3_v2=false" \
  --phase3-override "controller_max_steps=${MAX_CONTROLLER_STEPS}" \
  --phase3-override "use_one_replay_setting=true" \
  --phase3-override "replay_setting=global" \
  --phase3-override "full_context_probability=1.0" \
  --phase3-override "decision_block_normalized=false" \
  --phase3-override "use_type_loss_weights=false" \
  --phase3-override "visual_block_dropout_p=0.0" \
  --phase3-override "multi_hot_patch_labels=false" \
  "${limit_args[@]}"

echo "[6/6] Running trained-controller inference on M3CoT test..."
python lvar_scripts/infer_lvar_m3cot.py \
  --config "${CONFIG}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --use-checkpoint \
  --controller-path "${controller_checkpoint}" \
  --max-controller-steps "${MAX_CONTROLLER_STEPS}" \
  --output "${controller_predictions}" \
  --no-nucleus-insertion \
  "${limit_args[@]}"

echo "Pipeline complete. Accuracy summaries:"
echo "  THINK replay:       ${think_replay%.jsonl}_summary.json"
echo "  No-THINK replay:    ${no_think_replay%.jsonl}_summary.json"
echo "  Controller inference: ${controller_predictions%.jsonl}_summary.json"
