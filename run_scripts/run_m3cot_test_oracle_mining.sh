#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
IVTLR_CHECKPOINT="${IVTLR_CHECKPOINT:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"

if [[ -z "${LVAR_PHASE1_CHECKPOINT}" ]]; then
  echo "Set LVAR_PHASE1_CHECKPOINT or pass it as the first argument." >&2
  exit 2
fi

if [[ ! -f "${IVTLR_CHECKPOINT}" ]]; then
  echo "IVTLR checkpoint not found: ${IVTLR_CHECKPOINT}" >&2
  exit 2
fi

if [[ ! -f "${LVAR_PHASE1_CHECKPOINT}" ]]; then
  echo "LVAR phase 1 checkpoint not found: ${LVAR_PHASE1_CHECKPOINT}" >&2
  exit 2
fi

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

run_one() {
  local model_key="$1"
  local checkpoint_path="$2"
  local context_label="$3"
  local initial_visual_mode="$4"

  local trace_dir="outputs/oracle_dataset/test/${model_key}_ckpt"
  local trace_path="${trace_dir}/m3cot_test_traces_${model_key}_${context_label}.jsonl"

  mkdir -p "${trace_dir}"

  echo "Mining ${model_key} checkpoint on M3CoT test with ${context_label} context..."
  python lvar_scripts/mine_phase2.py \
    --config "${CONFIG}" \
    --dataset-partition test \
    --checkpoint-path "${checkpoint_path}" \
    --initial-visual-mode "${initial_visual_mode}" \
    --seed "${SEED}" \
    --output "${trace_path}" \
    --resume \
    "${limit_args[@]}"

  eval_mined_traces "${model_key}" "${model_key}" "${checkpoint_path}" "${context_label}"
}

eval_mined_traces() {
  local mined_by_key="$1"
  local evaluated_by_key="$2"
  local evaluator_checkpoint_path="$3"
  local context_label="$4"

  local trace_path="outputs/oracle_dataset/test/${mined_by_key}_ckpt/m3cot_test_traces_${mined_by_key}_${context_label}.jsonl"
  local variants=("raw" "filtered_cap" "filtered_no_cap")

  for trace_variant in "${variants[@]}"; do
    local inference_dir="outputs/inference/test_oracle/mined_by_${mined_by_key}_ckpt/evaluated_by_${evaluated_by_key}_ckpt/trace_variant_${trace_variant}"
    local output_path="${inference_dir}/m3cot_test_predictions_mined-by_${mined_by_key}_evaluated-by_${evaluated_by_key}_${context_label}_${trace_variant}.jsonl"

    mkdir -p "${inference_dir}"

    echo "Evaluating ${context_label} ${trace_variant} traces mined by ${mined_by_key} checkpoint using ${evaluated_by_key} checkpoint..."
    python lvar_scripts/eval_mined_traces_m3cot.py \
      --config "${CONFIG}" \
      --dataset-partition test \
      --checkpoint-path "${evaluator_checkpoint_path}" \
      --context "${context_label}" \
      --trace-variant "${trace_variant}" \
      --seed "${SEED}" \
      --trace-path "${trace_path}" \
      --output "${output_path}" \
      "${limit_args[@]}"
  done
}

run_one "ivtlr" "${IVTLR_CHECKPOINT}" "global" "global"
run_one "ivtlr" "${IVTLR_CHECKPOINT}" "coarse" "coarse"
run_one "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global" "global"
run_one "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse" "coarse"

echo "Done. Trace datasets are under outputs/oracle_dataset/test. Self and cross evals are under outputs/inference/test_oracle/mined_by_*_ckpt/evaluated_by_*_ckpt."
