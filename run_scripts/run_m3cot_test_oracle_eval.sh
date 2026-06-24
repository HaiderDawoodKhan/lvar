
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

    echo "Evaluating ${context_label} ${trace_variant} traces mined by ${mined_by_key} checkpoint using ${evaluated_by_key} checkpoint with entropy tracking..."
    python scripts/eval_mined_traces_m3cot.py \
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

eval_mined_traces "ivtlr" "ivtlr" "${IVTLR_CHECKPOINT}" "global"
eval_mined_traces "ivtlr" "ivtlr" "${IVTLR_CHECKPOINT}" "coarse"
eval_mined_traces "lvar" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global"
eval_mined_traces "lvar" "ivtlr" "${IVTLR_CHECKPOINT}" "global"

echo "Running cross-checkpoint oracle-forced evals..."
eval_mined_traces "lvar" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse"
eval_mined_traces "lvar" "ivtlr" "${IVTLR_CHECKPOINT}" "coarse"
eval_mined_traces "ivtlr" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global"
eval_mined_traces "ivtlr" "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse"

echo "Done. Existing trace datasets were read from outputs/oracle_dataset/test. Self and cross evals are under outputs/inference/test_oracle/mined_by_*_ckpt/evaluated_by_*_ckpt."
echo "Entropy sidecars are written next to each predictions JSONL as *_entropy_tracking.json."
