#!/usr/bin/env bash
# End-to-end greedy counterpart to the beam-oracle M3CoT pipeline.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs/greedy_oracle_dataset/m3cot}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${ROOT_DIR}/lvar_scripts/mine_phase2_greedy.py" ]]; then
  echo "Could not locate lvar_scripts/mine_phase2_greedy.py below ${ROOT_DIR}." >&2
  exit 1
fi

TEST_TRACES="${OUTPUT_DIR}/m3cot_test_greedy_traces.jsonl"
TEST_REPLAY="${OUTPUT_DIR}/m3cot_test_greedy_replay.jsonl"
TRAIN_TRACES="${OUTPUT_DIR}/m3cot_train_greedy_traces.jsonl"
CONTROLLER_DIR="${OUTPUT_DIR}/controller_sft"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/4] Mining M3CoT test traces with greedy width-one search..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_greedy.py" \
  --config "${CONFIG_PATH}" \
  --dataset-partition test \
  --output "${TEST_TRACES}" \
  --resume

echo "[2/4] Replaying greedy test traces..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" \
  --config "${CONFIG_PATH}" \
  --trace-path "${TEST_TRACES}" \
  --dataset-partition test \
  --context full_context \
  --output "${TEST_REPLAY}"

echo "[3/4] Mining M3CoT train traces with greedy width-one search..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_greedy.py" \
  --config "${CONFIG_PATH}" \
  --dataset-partition train \
  --output "${TRAIN_TRACES}" \
  --resume

echo "[4/4] Training the controller on greedy oracle traces..."
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/train_controller_sft.py" \
  --config "${CONFIG_PATH}" \
  --trace-jsonl "${TRAIN_TRACES}" \
  --output-dir "${CONTROLLER_DIR}"
