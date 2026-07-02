#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/qwen2vl_sqa.yaml"
SQA_CHECKPOINT="/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_sqa_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
TRACE_DIR="${TRACE_DIR:-outputs/oracle_dataset/test/sqa_ckpt}"
TRACE_PATH="${TRACE_PATH:-${TRACE_DIR}/sqa_test_traces_sqa_global.jsonl}"

if [[ ! -f "${SQA_CHECKPOINT}" ]]; then
  echo "SQA checkpoint not found: ${SQA_CHECKPOINT}" >&2
  exit 2
fi

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

mkdir -p "${TRACE_DIR}"

echo "Mining SQA checkpoint on ScienceQA test with global context..."
python lvar_scripts/mine_phase2.py \
  --config "${CONFIG}" \
  --dataset-partition test \
  --checkpoint-path "${SQA_CHECKPOINT}" \
  --initial-visual-mode global \
  --seed "${SEED}" \
  --output "${TRACE_PATH}" \
  --resume \
  "${limit_args[@]}"

echo "Done. Trace dataset written to ${TRACE_PATH}"
