#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen2vl_m3cot.yaml}"
PARTITION="${PARTITION:-validation}"
INDEX="${INDEX:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/qwen_lvar_vs_reference_${PARTITION}_${INDEX}.json}"

ARGS=(
  --config "${CONFIG}"
  --partition "${PARTITION}"
  --index "${INDEX}"
  --image-size "${IMAGE_SIZE}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi

if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

if [[ -n "${IMAGE_PATH:-}" || -n "${PROMPT:-}" ]]; then
  ARGS=(
    --config "${CONFIG}"
    --image-path "${IMAGE_PATH:-}"
    --prompt "${PROMPT:-}"
    --image-size "${IMAGE_SIZE}"
    --output-json "${OUTPUT_JSON}"
  )
  if [[ -n "${EXAMPLE_ID:-}" ]]; then
    ARGS+=(--example-id "${EXAMPLE_ID}")
  fi
  if [[ -n "${GOLD_ANSWER:-}" ]]; then
    ARGS+=(--gold-answer "${GOLD_ANSWER}")
  fi
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
  fi
  if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
    ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  fi
fi

python lvar_scripts/diagnose_qwen_lvar_vs_reference.py "${ARGS[@]}"
