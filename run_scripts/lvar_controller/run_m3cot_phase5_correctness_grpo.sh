#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

args=(--config "${CONFIG}")
if [[ -n "${CHECKPOINT_EVERY}" ]]; then
  args+=(--checkpoint-every "${CHECKPOINT_EVERY}")
fi

echo "Running Phase 5 correctness-only GRPO"
echo "  config: ${CONFIG}"
echo "  metrics: phase5.metrics_path or <phase5.output_dir>/grpo_training_metrics.jsonl"
echo "  summary: phase5.summary_path or <phase5.output_dir>/grpo_training_summary.json"

"${PYTHON_BIN}" lvar_scripts/train_grpo.py "${args[@]}"
