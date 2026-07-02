#!/usr/bin/env bash
set -euo pipefail

# Evaluate controllers produced by run_controller_sft_variant_sweep.sh on the
# validation set, with nucleus insertion explicitly disabled.
#
# Useful overrides:
#   ONLY_VARIANT=multihot_binary_unweighted_unnormalized bash run_scripts/run_controller_validation_eval.sh
#   ONLY_TRACE_SOURCE=global_test SKIP_EXISTING=true bash run_scripts/run_controller_validation_eval.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/controller_sft_m3cot_test_sweeps}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/controller_runs.jsonl}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/eval_validation}"
LIMIT="${LIMIT:-}"
PHASE4_VLM_CHECKPOINT_PATH="${PHASE4_VLM_CHECKPOINT_PATH:-}"
USE_COARSE_CONTEXT="${USE_COARSE_CONTEXT:-false}"
ONLY_TRACE_SOURCE="${ONLY_TRACE_SOURCE:-}"
ONLY_VARIANT="${ONLY_VARIANT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Controller manifest not found: ${MANIFEST_PATH}" >&2
  echo "Run run_scripts/run_controller_sft_variant_sweep.sh first, or set MANIFEST_PATH." >&2
  exit 2
fi

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

phase4_args=()
if [[ -n "${PHASE4_VLM_CHECKPOINT_PATH}" ]]; then
  phase4_args=(--phase4-vlm-checkpoint-path "${PHASE4_VLM_CHECKPOINT_PATH}")
fi

coarse_args=()
if [[ "${USE_COARSE_CONTEXT}" == "true" ]]; then
  coarse_args=(--use-coarse-context)
fi

mkdir -p "${EVAL_ROOT}"

while IFS=$'\t' read -r trace_source variant config_path checkpoint_path; do
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${trace_source}/${variant}: ${config_path}" >&2
    exit 2
  fi
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing controller checkpoint for ${trace_source}/${variant}: ${checkpoint_path}" >&2
    exit 2
  fi

  output_dir="${EVAL_ROOT}/${trace_source}/${variant}"
  output_path="${output_dir}/m3cot_validation_predictions.jsonl"
  mkdir -p "${output_dir}"

  if [[ "${SKIP_EXISTING}" == "true" && -s "${output_path}" ]]; then
    echo "Skipping existing validation output: trace_source=${trace_source} variant=${variant} path=${output_path}"
    continue
  fi

  echo "Evaluating controller on validation: trace_source=${trace_source} variant=${variant}"
  python lvar_scripts/infer_lvar_m3cot.py \
    --config "${config_path}" \
    --use-validation-set \
    --controller-path "${checkpoint_path}" \
    --output "${output_path}" \
    --no-nucleus-insertion \
    "${phase4_args[@]}" \
    "${coarse_args[@]}" \
    "${limit_args[@]}"
done < <(
  python - "${MANIFEST_PATH}" "${ONLY_TRACE_SOURCE}" "${ONLY_VARIANT}" <<'PY'
import json
import sys

manifest_path, only_trace_source, only_variant = sys.argv[1:]
with open(manifest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if only_trace_source and row.get("trace_source") != only_trace_source:
            continue
        if only_variant and row.get("variant") != only_variant:
            continue
        print(
            row["trace_source"],
            row["variant"],
            row["config_path"],
            row["controller_checkpoint_path"],
            sep="\t",
        )
PY
)

echo "Done. Validation outputs are under ${EVAL_ROOT}"
