#!/usr/bin/env bash
set -euo pipefail

# Train 8 controller SFT variants:
#   4 objective variants x 2 test trace sources.
#
# All runs replay traces under the same single global/full-image context:
#   phase3.use_one_replay_setting=true
#   phase3.replay_setting=global
#
# Override paths/settings via environment variables, for example:
#   GLOBAL_TRACE_PATH=... COARSE_TRACE_PATH=... LIMIT=200 bash run_scripts/run_controller_sft_variant_sweep.sh

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/controller_sft_m3cot_test_sweeps}"
TRAIN_ROOT="${TRAIN_ROOT:-${OUTPUT_ROOT}/train}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/controller_runs.jsonl}"
GLOBAL_TRACE_PATH="${GLOBAL_TRACE_PATH:-outputs/oracle_dataset/test/lvar_ckpt/m3cot_test_traces_lvar_global.jsonl}"
COARSE_TRACE_PATH="${COARSE_TRACE_PATH:-outputs/oracle_dataset/test/lvar_ckpt/m3cot_test_traces_lvar_coarse.jsonl}"
PHASE4_VLM_CHECKPOINT_PATH="${PHASE4_VLM_CHECKPOINT_PATH:-}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
DATASET_PARTITION="${DATASET_PARTITION:-test}"
ORDER_DECAY="${ORDER_DECAY:-0.5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"

mkdir -p "${TRAIN_ROOT}"
mkdir -p "$(dirname "${MANIFEST_PATH}")"
: > "${MANIFEST_PATH}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

phase4_override_args=()
if [[ -n "${PHASE4_VLM_CHECKPOINT_PATH}" ]]; then
  phase4_override_args=(--phase3-override "phase4_vlm_checkpoint_path=${PHASE4_VLM_CHECKPOINT_PATH}")
fi

append_manifest() {
  local trace_source="$1"
  local variant="$2"
  local trace_path="$3"
  local output_dir="$4"
  local use_weights="$5"
  local multi_hot="$6"
  local multi_hot_mode="$7"

  python - "${MANIFEST_PATH}" "${trace_source}" "${variant}" "${trace_path}" "${output_dir}" \
    "${CONFIG}" "${use_weights}" "${multi_hot}" "${multi_hot_mode}" "${ORDER_DECAY}" "${CHECKPOINT_EVERY}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    trace_source,
    variant,
    trace_path,
    output_dir,
    config_path,
    use_weights,
    multi_hot,
    multi_hot_mode,
    order_decay,
    checkpoint_every,
) = sys.argv[1:]

row = {
    "trace_source": trace_source,
    "variant": variant,
    "trace_path": trace_path,
    "output_dir": output_dir,
    "config_path": config_path,
    "controller_checkpoint_path": str(Path(output_dir) / "controller_sft.pt"),
    "summary_path": str(Path(output_dir) / "controller_sft_summary.json"),
    "loss_history_path": str(Path(output_dir) / "controller_sft_losses.jsonl"),
    "phase3_overrides": {
        "dataset_partition": "test",
        "use_one_replay_setting": True,
        "replay_setting": "global",
        "decision_block_normalized": True,
        "visual_block_dropout_p": 0.0,
        "use_type_loss_weights": use_weights.lower() == "true",
        "multi_hot_patch_labels": multi_hot.lower() == "true",
        "multi_hot_patch_target_mode": multi_hot_mode,
        "multi_hot_patch_order_decay": float(order_decay),
        "checkpoint_every": int(checkpoint_every),
    },
    "phase3_v2_overrides": {
        "enabled": False,
        "remove_global": False,
        "mask_immediate_repeats": False,
    },
}
with open(manifest_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
PY
}

run_one() {
  local trace_source="$1"
  local trace_path="$2"
  local variant="$3"
  local use_weights="$4"
  local multi_hot="$5"
  local multi_hot_mode="$6"

  if [[ ! -f "${trace_path}" ]]; then
    echo "Trace file not found for ${trace_source}: ${trace_path}" >&2
    exit 2
  fi

  local output_dir="${TRAIN_ROOT}/${trace_source}/${variant}"
  mkdir -p "${output_dir}"

  echo "Training controller: trace_source=${trace_source} variant=${variant} replay_setting=global"
  python lvar_scripts/train_controller_sft.py \
    --config "${CONFIG}" \
    --trace-jsonl "${trace_path}" \
    --output-dir "${output_dir}" \
    --seed "${SEED}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --phase3-override "dataset_partition=${DATASET_PARTITION}" \
    --phase3-override "use_one_replay_setting=true" \
    --phase3-override "replay_setting=global" \
    --phase3-override "decision_block_normalized=true" \
    --phase3-override "visual_block_dropout_p=0.0" \
    --phase3-override "use_type_loss_weights=${use_weights}" \
    --phase3-override "multi_hot_patch_labels=${multi_hot}" \
    --phase3-override "multi_hot_patch_target_mode=${multi_hot_mode}" \
    --phase3-override "multi_hot_patch_order_decay=${ORDER_DECAY}" \
    --phase3-v2-override "enabled=false" \
    --phase3-v2-override "remove_global=false" \
    --phase3-v2-override "mask_immediate_repeats=false" \
    "${phase4_override_args[@]}" \
    "${limit_args[@]}"

  append_manifest "${trace_source}" "${variant}" "${trace_path}" "${output_dir}" \
    "${use_weights}" "${multi_hot}" "${multi_hot_mode}"
}

run_suite_for_trace_source() {
  local trace_source="$1"
  local trace_path="$2"

  run_one "${trace_source}" "${trace_path}" "normal" "false" "false" "binary"
  run_one "${trace_source}" "${trace_path}" "weighted" "true" "false" "binary"
  run_one "${trace_source}" "${trace_path}" "multihot_binary" "true" "true" "binary"
  run_one "${trace_source}" "${trace_path}" "multihot_ordered" "true" "true" "ordered"
}

run_suite_for_trace_source "global_test" "${GLOBAL_TRACE_PATH}"
# run_suite_for_trace_source "coarse_test_replayed_global" "${COARSE_TRACE_PATH}"

echo "Done. Controller sweep manifest: ${MANIFEST_PATH}"
