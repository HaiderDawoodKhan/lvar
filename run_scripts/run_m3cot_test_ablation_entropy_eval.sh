#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
DATASET_PARTITION="${DATASET_PARTITION:-test}"
LVAR_PHASE1_CHECKPOINT="${LVAR_PHASE1_CHECKPOINT:-${1:-/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference/${DATASET_PARTITION}_oracle_ablations}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
STEP_ENTROPY_TOP_K="${STEP_ENTROPY_TOP_K:-}"
GLOBAL_REPLAY_CONTEXT="${GLOBAL_REPLAY_CONTEXT:-global}"
COARSE_REPLAY_CONTEXT="${COARSE_REPLAY_CONTEXT:-coarse}"

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

# Five trace-ablation runs. These produce normal final-answer entropy sidecars,
# but do not pay the extra cost of measuring every intermediate action prefix.
run_variant "global" "no_reasoning" "false"
run_variant "global" "shuffled" "false"
run_variant "global" "no_visual" "false"
run_variant "coarse" "shuffled" "false"
run_variant "coarse" "no_visual" "false"

# Two untouched-trace runs with the expensive per-prefix measurements enabled.
run_variant "global" "raw" "true"
run_variant "coarse" "raw" "true"

echo "Done. Completed 7 LVAR self-evaluation runs under ${OUTPUT_ROOT}."
