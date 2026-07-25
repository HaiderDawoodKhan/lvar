#!/usr/bin/env bash
# Replay each retained beam rank separately to compare rank-wise oracle accuracy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
TRACE_PATH="${2:-${ROOT_DIR}/outputs/beam_oracle_dataset/m3cot/m3cot_test_beam_traces.jsonl}"
OUTPUT_DIR="${3:-${ROOT_DIR}/outputs/beam_oracle_dataset/m3cot/replay_by_rank}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Override for a non-default beam width, e.g. BEAM_RANKS="1 2" ./...sh
BEAM_RANKS="${BEAM_RANKS:-1 2 3}"

if [[ ! -f "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" ]]; then
  echo "Could not locate the trace replay script below ${ROOT_DIR}." >&2
  exit 1
fi
if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "Beam trace dataset not found: ${TRACE_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for beam_rank in ${BEAM_RANKS}; do
  output_path="${OUTPUT_DIR}/m3cot_test_beam_rank_${beam_rank}_replay.jsonl"
  echo "Replaying beam rank ${beam_rank} across all prompts that retained it..."
  "${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" \
    --config "${CONFIG_PATH}" \
    --trace-path "${TRACE_PATH}" \
    --dataset-partition test \
    --beam-rank "${beam_rank}" \
    --context full_context \
    --output "${output_path}"
done

echo "Rank-wise replay summaries are in ${OUTPUT_DIR}."
