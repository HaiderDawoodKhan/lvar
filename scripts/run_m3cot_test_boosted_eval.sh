#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
IVTLR_CHECKPOINT="${IVTLR_CHECKPOINT:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
PHASE4_VLM_CHECKPOINT="${PHASE4_VLM_CHECKPOINT:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth}"
CONTROLLER_CHECKPOINT="${CONTROLLER_CHECKPOINT:-/home/csalt/Haider/DVLM/lvar/outputs/controller_sft_m3cot/controller_sft.pt}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"

for checkpoint in \
  "${IVTLR_CHECKPOINT}" \
  "${LVAR_PHASE1_CHECKPOINT}" \
  "${PHASE4_VLM_CHECKPOINT}" \
  "${CONTROLLER_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
done

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

boost_settings() {
  local target layer_mode alpha
  for target in trace_visual trace_all; do
    for layer_mode in all latter_half; do
      for alpha in 0.01 0.1 0.2 0.3 0.4 0.5; do
        echo "${target} ${layer_mode} ${alpha}"
      done
    done
  done
}
# boost_settings() {
#   local target layer_mode alpha
#   for target in trace_visual trace_all; do
#     for layer_mode in all latter_half; do
#       for alpha in 0.01 0.1 0.2 0.4; do
#         echo "${target} ${layer_mode} ${alpha}"
#       done
#     done
#   done
# }

run_full_lvar() {
  local target="$1"
  local layer_mode="$2"
  local alpha="$3"
  local inference_dir="outputs/inference/current_lvar_model_boosted/target_${target}/layers_${layer_mode}/alpha_${alpha}"
  local output_path="${inference_dir}/m3cot_lvar_predictions.jsonl"

  mkdir -p "${inference_dir}"
  echo "Running full LVAR pipeline: target=${target}, layers=${layer_mode}, alpha=${alpha}"
  python scripts/infer_lvar_m3cot.py \
    --config "${CONFIG}" \
    --phase4-vlm-checkpoint-path "${PHASE4_VLM_CHECKPOINT}" \
    --controller-checkpoint-path "${CONTROLLER_CHECKPOINT}" \
    --trace-boost \
    --trace-boost-target "${target}" \
    --trace-boost-layer-mode "${layer_mode}" \
    --trace-boost-alpha "${alpha}" \
    --output "${output_path}" \
    "${limit_args[@]}"
}

eval_mined_trace_setting() {
  local mined_by_key="$1"
  local evaluated_by_key="$2"
  local evaluator_checkpoint_path="$3"
  local context_label="$4"
  local target="$5"
  local layer_mode="$6"
  local alpha="$7"

  local trace_path="outputs/oracle_dataset/test/${mined_by_key}_ckpt/m3cot_test_traces_${mined_by_key}_${context_label}.jsonl"
  local inference_dir="outputs/inference/test_oracle_boosted/mined_by_${mined_by_key}_ckpt/evaluated_by_${evaluated_by_key}_ckpt/trace_variant_raw/target_${target}/layers_${layer_mode}/alpha_${alpha}"
  local output_path="${inference_dir}/m3cot_test_predictions_mined-by_${mined_by_key}_evaluated-by_${evaluated_by_key}_${context_label}_raw.jsonl"

  if [[ ! -f "${trace_path}" ]]; then
    echo "Mined trace file not found: ${trace_path}" >&2
    exit 2
  fi
  mkdir -p "${inference_dir}"
  echo "Evaluating raw ${context_label} traces mined by ${mined_by_key} with ${evaluated_by_key}: target=${target}, layers=${layer_mode}, alpha=${alpha}"
  python scripts/eval_mined_traces_m3cot.py \
    --config "${CONFIG}" \
    --dataset-partition test \
    --checkpoint-path "${evaluator_checkpoint_path}" \
    --context "${context_label}" \
    --trace-variant raw \
    --seed "${SEED}" \
    --trace-path "${trace_path}" \
    --trace-boost \
    --trace-boost-target "${target}" \
    --trace-boost-layer-mode "${layer_mode}" \
    --trace-boost-alpha "${alpha}" \
    --output "${output_path}" \
    "${limit_args[@]}"
}


while read -r target layer_mode alpha; do
  eval_mined_trace_setting "lvar" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global" "${target}" "${layer_mode}" "${alpha}"
  eval_mined_trace_setting "ivtlr" "ivtlr" "${IVTLR_CHECKPOINT}" "global" "${target}" "${layer_mode}" "${alpha}"
#   eval_mined_trace_setting "lvar" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse" "${target}" "${layer_mode}" "${alpha}"
#   eval_mined_trace_setting "ivtlr" "ivtlr" "${IVTLR_CHECKPOINT}" "coarse" "${target}" "${layer_mode}" "${alpha}"
done < <(boost_settings)

while read -r target layer_mode alpha; do
  run_full_lvar "${target}" "${layer_mode}" "${alpha}"
done < <(boost_settings)

echo "Done. Full-pipeline outputs are under outputs/inference/current_lvar_model_boosted."
echo "Raw oracle outputs are under outputs/inference/test_oracle_boosted."
