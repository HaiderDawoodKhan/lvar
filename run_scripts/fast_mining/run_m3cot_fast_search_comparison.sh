#!/usr/bin/env bash
# Compare rank-once greedy and beam-search oracle mining on the M3CoT test set.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs/fast_search_comparison/m3cot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PATCH_TOP_K=100

if [[ ! -f "${ROOT_DIR}/lvar_scripts/mine_phase2_fast.py" ]]; then
  echo "Could not locate lvar_scripts/mine_phase2_fast.py below ${ROOT_DIR}." >&2
  exit 1
fi

GREEDY_TRACES="${OUTPUT_DIR}/m3cot_test_greedy_fast_top100_traces.jsonl"
GREEDY_REPLAY="${OUTPUT_DIR}/m3cot_test_greedy_fast_top100_replay.jsonl"
BEAM_TRACES="${OUTPUT_DIR}/m3cot_test_beam_search_fast_top100_traces.jsonl"
BEAM_REPLAY="${OUTPUT_DIR}/m3cot_test_beam_search_fast_top100_replay.jsonl"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/4] Mining greedy_fast test traces with patch_top_k=${PATCH_TOP_K}..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_fast.py" \
  --strategy greedy_fast \
  --config "${CONFIG_PATH}" \
  --dataset-partition test \
  --patch-top-k "${PATCH_TOP_K}" \
  --output "${GREEDY_TRACES}" \
  --resume

echo "[2/4] Replaying greedy_fast test traces..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" \
  --config "${CONFIG_PATH}" \
  --trace-path "${GREEDY_TRACES}" \
  --dataset-partition test \
  --context full_context \
  --output "${GREEDY_REPLAY}"

echo "[3/4] Mining beam_search_fast test traces with patch_top_k=${PATCH_TOP_K}..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_fast.py" \
  --strategy beam_search_fast \
  --config "${CONFIG_PATH}" \
  --dataset-partition test \
  --patch-top-k "${PATCH_TOP_K}" \
  --output "${BEAM_TRACES}" \
  --resume

echo "[4/4] Replaying rank-1 beam_search_fast test traces..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" \
  --config "${CONFIG_PATH}" \
  --trace-path "${BEAM_TRACES}" \
  --dataset-partition test \
  --beam-rank 1 \
  --context full_context \
  --output "${BEAM_REPLAY}"

echo "Done. Each replay command prints accuracy and writes a *_summary.json sidecar in ${OUTPUT_DIR}."
