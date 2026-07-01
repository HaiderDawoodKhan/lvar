#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
DATASET_PARTITION="${DATASET_PARTITION:-validation}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference/${DATASET_PARTITION}_oracle_ablations}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
STEP_ENTROPY_TOP_K="${STEP_ENTROPY_TOP_K:-}"
GLOBAL_REPLAY_CONTEXT="${GLOBAL_REPLAY_CONTEXT:-global}"
COARSE_REPLAY_CONTEXT="${COARSE_REPLAY_CONTEXT:-coarse}"
CONTROLLER_SWEEP_ROOT="${CONTROLLER_SWEEP_ROOT:-/home/csalt/Haider/DVLM/lvar/outputs/controller_sft_m3cot_test_sweeps}"
CONTROLLER_TRACE_SOURCE="${CONTROLLER_TRACE_SOURCE:-global_test}"
CONTROLLER_TRACE_ROOT="${CONTROLLER_TRACE_ROOT:-${CONTROLLER_SWEEP_ROOT}/eval_validation/${CONTROLLER_TRACE_SOURCE}}"
CONTROLLER_MANIFEST_PATH="${CONTROLLER_MANIFEST_PATH:-${CONTROLLER_SWEEP_ROOT}/controller_runs.jsonl}"
REPLAY_CONTROLLER_TRACES="${REPLAY_CONTROLLER_TRACES:-true}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
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

step_entropy_args=()
if [[ -n "${STEP_ENTROPY_TOP_K}" ]]; then
  step_entropy_args=(--step-entropy-top-k "${STEP_ENTROPY_TOP_K}")
fi

run_variant() {
  local trace_context="$1"
  local variant="$2"
  local track_prefix_entropy="$3"
  local replay_context
  local replay_suffix=""
  if [[ "${trace_context}" == "global" ]]; then
    replay_context="${GLOBAL_REPLAY_CONTEXT}"
  else
    replay_context="${COARSE_REPLAY_CONTEXT}"
  fi
  if [[ "${replay_context}" != "${trace_context}" ]]; then
    replay_suffix="_replayed-under_${replay_context}"
  fi
  local trace_path="outputs/oracle_dataset/${DATASET_PARTITION}/lvar_ckpt/m3cot_${DATASET_PARTITION}_traces_lvar_${trace_context}.jsonl"
  local run_dir="${OUTPUT_ROOT}/mined_by_lvar_ckpt/evaluated_by_lvar_ckpt/trace_variant_${variant}"
  local output_path="${run_dir}/m3cot_${DATASET_PARTITION}_predictions_mined-by_lvar_evaluated-by_lvar_${trace_context}_${variant}${replay_suffix}.jsonl"
  local tracking_args=()

  if [[ ! -f "${trace_path}" ]]; then
    echo "Mined trace file not found: ${trace_path}" >&2
    exit 2
  fi

  if [[ "${track_prefix_entropy}" == "true" ]]; then
    tracking_args=(--track-step-hidden-entropy --track-prefix-rollouts)
  else
    tracking_args=(--no-track-step-hidden-entropy --no-track-prefix-rollouts)
  fi

  mkdir -p "${run_dir}"
  echo "Running model=lvar, trace_context=${trace_context}, replay_context=${replay_context}, variant=${variant}, track_prefix_entropy=${track_prefix_entropy}"
  python lvar_scripts/eval_mined_traces_m3cot.py \
    --config "${CONFIG}" \
    --dataset-partition "${DATASET_PARTITION}" \
    --checkpoint-path "${LVAR_PHASE1_CHECKPOINT}" \
    --trace-path "${trace_path}" \
    --context "${replay_context}" \
    --trace-variant "${variant}" \
    --seed "${SEED}" \
    "${tracking_args[@]}" \
    --output "${output_path}" \
    "${step_entropy_args[@]}" \
    "${limit_args[@]}"
}

run_controller_trace_replays() {
  if [[ "${REPLAY_CONTROLLER_TRACES}" != "true" ]]; then
    echo "Skipping controller-trace replay because REPLAY_CONTROLLER_TRACES=${REPLAY_CONTROLLER_TRACES}"
    return
  fi
  if [[ ! -f "${CONTROLLER_MANIFEST_PATH}" ]]; then
    echo "Controller manifest not found: ${CONTROLLER_MANIFEST_PATH}" >&2
    echo "Set CONTROLLER_MANIFEST_PATH or REPLAY_CONTROLLER_TRACES=false." >&2
    exit 2
  fi
  if [[ ! -d "${CONTROLLER_TRACE_ROOT}" ]]; then
    echo "Controller trace root not found: ${CONTROLLER_TRACE_ROOT}" >&2
    echo "Set CONTROLLER_TRACE_ROOT or REPLAY_CONTROLLER_TRACES=false." >&2
    exit 2
  fi

  while IFS=$'\t' read -r trace_source variant config_path checkpoint_path; do
    local trace_path="${CONTROLLER_TRACE_ROOT}/${variant}/m3cot_${DATASET_PARTITION}_predictions.jsonl"
    local run_dir="${OUTPUT_ROOT}/controller_traces_from_lvar_model/${trace_source}/${variant}"
    local output_path="${run_dir}/m3cot_${DATASET_PARTITION}_predictions_replayed-controller-traces_${trace_source}_${variant}.jsonl"

    if [[ ! -f "${trace_path}" ]]; then
      echo "Controller trace file not found for ${trace_source}/${variant}: ${trace_path}" >&2
      exit 2
    fi
    if [[ ! -f "${config_path}" ]]; then
      echo "Controller run config not found for ${trace_source}/${variant}: ${config_path}" >&2
      exit 2
    fi
    if [[ ! -f "${checkpoint_path}" ]]; then
      echo "Controller checkpoint not found for ${trace_source}/${variant}: ${checkpoint_path}" >&2
      exit 2
    fi

    mkdir -p "${run_dir}"
    echo "Replaying LVAR controller traces with entropy tracking: trace_source=${trace_source}, variant=${variant}"
    python lvar_scripts/eval_mined_traces_m3cot.py \
      --config "${config_path}" \
      --dataset-partition "${DATASET_PARTITION}" \
      --checkpoint-path "${LVAR_PHASE1_CHECKPOINT}" \
      --controller-path "${checkpoint_path}" \
      --trace-path "${trace_path}" \
      --context "${GLOBAL_REPLAY_CONTEXT}" \
      --trace-variant "raw" \
      --seed "${SEED}" \
      --track-step-hidden-entropy \
      --track-prefix-rollouts \
      --output "${output_path}" \
      "${step_entropy_args[@]}" \
      "${limit_args[@]}"
  done < <(
    python - "${CONTROLLER_MANIFEST_PATH}" "${CONTROLLER_TRACE_SOURCE}" "${CONTROLLER_SWEEP_ROOT}" "${CONTROLLER_TRACE_ROOT}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, wanted_trace_source, sweep_root, trace_root, default_config = sys.argv[1:]
rows_by_variant = {}

with open(manifest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("trace_source") != wanted_trace_source:
            continue
        rows_by_variant[row["variant"]] = (
            row["trace_source"],
            row["variant"],
            row["config_path"],
            row["controller_checkpoint_path"],
        )

# If the manifest was rewritten by a later one-off training run, recover older
# evaluated controllers from eval_validation/{trace_source}/{variant}.
trace_root = Path(trace_root)
sweep_root = Path(sweep_root)
if trace_root.exists():
    for run_dir in sorted(path for path in trace_root.iterdir() if path.is_dir()):
        variant = run_dir.name
        if variant in rows_by_variant:
            continue
        trace_path = run_dir / "m3cot_validation_predictions.jsonl"
        if not trace_path.exists():
            continue
        checkpoint_path = sweep_root / "train" / wanted_trace_source / variant / "controller_sft.pt"
        rows_by_variant[variant] = (
            wanted_trace_source,
            variant,
            default_config,
            str(checkpoint_path),
        )

for variant in sorted(rows_by_variant):
    trace_source, variant, config_path, checkpoint_path = rows_by_variant[variant]
    print(
        trace_source,
        variant,
        config_path,
        checkpoint_path,
        sep="\t",
    )
PY
  )
}

# Two untouched-trace runs with the expensive per-prefix measurements enabled.
run_variant "global" "raw" "true"
run_controller_trace_replays
