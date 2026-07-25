#!/usr/bin/env bash
# Mine optional beam-oracle traces, replay test traces, then train on train traces.
set -euo pipefail

# This script lives in run_scripts/beam_mining/, so the repository root is
# two levels above it (not one level above, which is run_scripts/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "${ROOT_DIR}/lvar_scripts/mine_phase2_beam.py" ]]; then
  echo "Could not locate lvar_scripts/mine_phase2_beam.py below ${ROOT_DIR}." >&2
  exit 1
fi
CONFIG_PATH="${1:-${ROOT_DIR}/configs/qwen2vl_m3cot.yaml}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs/beam_oracle_dataset/m3cot}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEST_TRACES="${OUTPUT_DIR}/m3cot_test_beam_traces.jsonl"
TEST_REPLAY="${OUTPUT_DIR}/m3cot_test_beam_replay.jsonl"
TRAIN_TRACES="${OUTPUT_DIR}/m3cot_train_beam_traces.jsonl"
CONTROLLER_DIR="${OUTPUT_DIR}/controller_sft"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# 1. Mine M3CoT test traces.
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_beam.py" \
  --config "${CONFIG_PATH}" \
  --dataset-partition test \
  --output "${TEST_TRACES}" \
  --no-resume

# 2. Replay rank-1 beam trajectories and write per-example results plus summary.
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/eval_mined_traces_m3cot.py" \
  --config "${CONFIG_PATH}" \
  --trace-path "${TEST_TRACES}" \
  --dataset-partition test \
  --context full_context \
  --output "${TEST_REPLAY}"

# 3. Mine M3CoT train traces.
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/mine_phase2_beam.py" \
  --config "${CONFIG_PATH}" \
  --dataset-partition train \
  --output "${TRAIN_TRACES}" \
  --no-resume

# 4. Train the controller.  The loader expands all surviving beam trajectories.
"${PYTHON_BIN}" "${ROOT_DIR}/lvar_scripts/train_controller_sft.py" \
  --config "${CONFIG_PATH}" \
  --trace-jsonl "${TRAIN_TRACES}" \
  --output-dir "${CONTROLLER_DIR}"
