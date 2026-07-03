#!/usr/bin/env bash
set -euo pipefail

ABLATION_SCRIPT="${ABLATION_SCRIPT:-run_scripts/run_m3cot_validation_num_rollouts_ablation.sh}"
ORACLE_MINING_SCRIPT="${ORACLE_MINING_SCRIPT:-run_scripts/run_sqa_test_val_oracle_mining.sh}"

if [[ ! -f "${ABLATION_SCRIPT}" ]]; then
  echo "Ablation script not found: ${ABLATION_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${ORACLE_MINING_SCRIPT}" ]]; then
  echo "Oracle mining script not found: ${ORACLE_MINING_SCRIPT}" >&2
  exit 2
fi

echo "Step 1/2: running M3CoT num-rollouts ablation"
bash "${ABLATION_SCRIPT}"

echo
echo "Step 2/2: running SQA test/validation oracle mining"
bash "${ORACLE_MINING_SCRIPT}"

echo
echo "Done. Completed num-rollouts ablation and SQA oracle mining."
