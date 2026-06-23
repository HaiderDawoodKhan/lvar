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

  local trace_dir="outputs/oracle_dataset/validation/${model_key}_ckpt"
  local trace_path="${trace_dir}/m3cot_validation_traces_${model_key}_${context_label}.jsonl"

  mkdir -p "${trace_dir}"

  echo "Mining ${model_key} checkpoint on M3CoT validation with ${context_label} context..."
  python scripts/mine_phase2.py \
    --config "${CONFIG}" \
    --dataset-partition validation \
    --checkpoint-path "${checkpoint_path}" \
    --initial-visual-mode "${initial_visual_mode}" \
    --seed "${SEED}" \
    --output "${trace_path}" \
    --resume \
    "${limit_args[@]}"

  eval_mined_traces "${model_key}" "${model_key}" "${checkpoint_path}" "${context_label}"
}

run_one "lvar" "${LVAR_PHASE1_CHECKPOINT}" "global" "global"
run_one "lvar" "${LVAR_PHASE1_CHECKPOINT}" "coarse" "coarse"

echo "Done. Trace datasets are under outputs/oracle_dataset/validation."