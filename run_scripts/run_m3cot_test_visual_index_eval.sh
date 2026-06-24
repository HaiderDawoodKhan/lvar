#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
IVTLR_CHECKPOINT="${IVTLR_CHECKPOINT:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"

for checkpoint in "${IVTLR_CHECKPOINT}" "${LVAR_PHASE1_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
done

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

eval_visual_index_modes() {
  local model_key="$1"
  local checkpoint_path="$2"
  local context_label="$3"
  local trace_path="outputs/oracle_dataset/test/${model_key}_ckpt/m3cot_test_traces_${model_key}_${context_label}.jsonl"
  local visual_index_modes=("random" "last")

  if [[ ! -f "${trace_path}" ]]; then
    echo "Mined trace file not found: ${trace_path}" >&2
    exit 2
  fi

  for visual_index_mode in "${visual_index_modes[@]}"; do
    local inference_dir="outputs/inference/test_oracle_visual_index/mined_by_${model_key}_ckpt/evaluated_by_${model_key}_ckpt/context_${context_label}/visual_index_mode_${visual_index_mode}"
    local output_path="${inference_dir}/m3cot_test_predictions_mined-by_${model_key}_evaluated-by_${model_key}_${context_label}_raw_${visual_index_mode}.jsonl"

    mkdir -p "${inference_dir}"
    echo "Evaluating ${model_key} ${context_label} trace with visual-index mode ${visual_index_mode}..."
    python lvar_scripts/eval_mined_traces_m3cot.py \
      --config "${CONFIG}" \
      --dataset-partition test \
      --checkpoint-path "${checkpoint_path}" \
      --context "${context_label}" \
      --trace-variant raw \
      --visual-index-mode "${visual_index_mode}" \
      --seed "${SEED}" \
      --trace-path "${trace_path}" \
      --output "${output_path}" \
      "${limit_args[@]}"
  done
}

# Four mined traces, each replayed by the same model that mined it.
eval_visual_index_modes "ivtlr" "${IVTLR_CHECKPOINT}" "global"
eval_visual_index_modes "ivtlr" "${IVTLR_CHECKPOINT}" "coarse"
eval_visual_index_modes "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global"
eval_visual_index_modes "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse"

echo "Done. Wrote eight runs under outputs/inference/test_oracle_visual_index."
