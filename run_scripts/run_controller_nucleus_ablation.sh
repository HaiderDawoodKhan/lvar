#!/usr/bin/env bash
set -euo pipefail

# Evaluate each trained controller under nucleus-insertion ablations:
#   1. off
#   2. patch only
#   3. region only
#
# The "both" setting is intentionally omitted because the requested ablation is
# off vs patch-only vs region-only.

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/controller_sft_m3cot_test_sweeps}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/controller_runs.jsonl}"
ABLATION_ROOT="${ABLATION_ROOT:-${OUTPUT_ROOT}/nucleus_ablation_validation}"
LIMIT="${LIMIT:-}"
PHASE4_VLM_CHECKPOINT_PATH="${PHASE4_VLM_CHECKPOINT_PATH:-}"
USE_COARSE_CONTEXT="${USE_COARSE_CONTEXT:-false}"
TOP_P="${TOP_P:-0.9}"
MAX_INDICES="${MAX_INDICES:-4}"

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

mkdir -p "${ABLATION_ROOT}"

run_ablation() {
  local trace_source="$1"
  local variant="$2"
  local config_path="$3"
  local checkpoint_path="$4"
  local ablation="$5"

  local output_dir="${ABLATION_ROOT}/${trace_source}/${variant}/${ablation}"
  local output_path="${output_dir}/m3cot_validation_predictions.jsonl"
  mkdir -p "${output_dir}"

  ablation_args=()
  case "${ablation}" in
    off)
      ablation_args=(--no-nucleus-insertion)
      ;;
    patch_only)
      ablation_args=(
        --nucleus-insertion
        --nucleus-insertion-scope patch
        --nucleus-insertion-top-p "${TOP_P}"
        --nucleus-insertion-max-indices "${MAX_INDICES}"
      )
      ;;
    region_only)
      ablation_args=(
        --nucleus-insertion
        --nucleus-insertion-scope region
        --nucleus-insertion-top-p "${TOP_P}"
        --nucleus-insertion-max-indices "${MAX_INDICES}"
      )
      ;;
    *)
      echo "Unknown ablation: ${ablation}" >&2
      exit 2
      ;;
  esac

  echo "Evaluating nucleus ablation: trace_source=${trace_source} variant=${variant} ablation=${ablation}"
  python lvar_scripts/infer_lvar_m3cot.py \
    --config "${config_path}" \
    --use-validation-set \
    --controller-path "${checkpoint_path}" \
    --output "${output_path}" \
    "${ablation_args[@]}" \
    "${phase4_args[@]}" \
    "${coarse_args[@]}" \
    "${limit_args[@]}"
}

while IFS=$'\t' read -r trace_source variant config_path checkpoint_path; do
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${trace_source}/${variant}: ${config_path}" >&2
    exit 2
  fi
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing controller checkpoint for ${trace_source}/${variant}: ${checkpoint_path}" >&2
    exit 2
  fi

  run_ablation "${trace_source}" "${variant}" "${config_path}" "${checkpoint_path}" "off"
  run_ablation "${trace_source}" "${variant}" "${config_path}" "${checkpoint_path}" "patch_only"
  run_ablation "${trace_source}" "${variant}" "${config_path}" "${checkpoint_path}" "region_only"
done < <(
  python - "${MANIFEST_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        print(
            row["trace_source"],
            row["variant"],
            row["config_path"],
            row["controller_checkpoint_path"],
            sep="\t",
        )
PY
)

echo "Done. Nucleus ablation outputs are under ${ABLATION_ROOT}"
