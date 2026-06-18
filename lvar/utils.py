import re
import argparse
from typing import Dict, Iterable, List, Optional

# Discrete action ids used by the controller and trace utilities.
ACTION_THINK = 0
ACTION_STOP = 1
ACTION_GLOBAL = 2
ACTION_REGION = 3
ACTION_PATCH = 4

# Stable id->name and name->id maps used by model code and debug output.
ACTION_NAMES: Dict[int, str] = {
    ACTION_THINK: "THINK",
    ACTION_STOP: "STOP",
    ACTION_GLOBAL: "GLOBAL",
    ACTION_REGION: "REGION",
    ACTION_PATCH: "PATCH",
}
ACTION_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in ACTION_NAMES.items()}

ACTION_NAMES_NO_GLOBAL: Dict[int, str] = {
    0: "THINK",
    1: "STOP",
    2: "REGION",
    3: "PATCH",
}
ACTION_NAME_TO_ID_NO_GLOBAL: Dict[str, int] = {name: idx for idx, name in ACTION_NAMES_NO_GLOBAL.items()}


def normalize_action_names(action_names: Optional[object] = None) -> Dict[int, str]:
    """Return a stable id->name action map from config metadata."""
    if action_names is None:
        return dict(ACTION_NAMES)
    if isinstance(action_names, dict):
        return {int(idx): str(name).upper() for idx, name in action_names.items()}
    if isinstance(action_names, (list, tuple)):
        return {idx: str(name).upper() for idx, name in enumerate(action_names)}
    raise TypeError("controller action_names must be a dict, list, tuple, or None.")

# Regex used everywhere to extract the final tagged answer from model text.
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_tagged_answer(text: Optional[str]) -> str:
    """Extract the content inside <answer>...</answer>; fallback to stripped raw text."""
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer_text(text: Optional[str]) -> str:
    """Normalize answers for reward/eval by lowercasing and collapsing whitespace."""
    extracted = extract_tagged_answer(text)
    normalized = re.sub(r"\s+", " ", extracted).strip().lower()
    return normalized


def _format_probability_distribution(
    values: Optional[Iterable[object]],
    labels: Optional[Iterable[object]] = None,
    precision: int = 2,
    top_k: Optional[int] = None,
) -> str:
    """Render probability values as a compact [label:prob, ...] list."""
    if values is None:
        return "[]"
    value_list = list(values)
    label_list = list(labels) if labels is not None else list(range(len(value_list)))
    pairs = [(label, float(value)) for label, value in zip(label_list, value_list)]
    if top_k is not None and top_k > 0 and len(pairs) > top_k:
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
    rendered = []
    for label, value in pairs:
        rendered.append(f"{label}:{value:.{precision}f}")
    return "[" + ", ".join(rendered) + "]"


def format_trace_step(step_trace: Dict[str, object]) -> str:
    """Render one reasoning step dict into a compact human-readable debug line."""
    action = step_trace.get("action", "UNKNOWN")
    trace_action_names = step_trace.get("action_names")
    if isinstance(trace_action_names, dict):
        action_name_map = {int(idx): str(name) for idx, name in trace_action_names.items()}
        action_labels = [action_name_map[idx] for idx in sorted(action_name_map)]
    elif isinstance(trace_action_names, (list, tuple)):
        action_labels = [str(name) for name in trace_action_names]
    else:
        action_probs = step_trace.get("action_probs")
        if isinstance(action_probs, list) and len(action_probs) == len(ACTION_NAMES_NO_GLOBAL):
            action_labels = [ACTION_NAMES_NO_GLOBAL[idx] for idx in sorted(ACTION_NAMES_NO_GLOBAL)]
        else:
            action_labels = [ACTION_NAMES[idx] for idx in sorted(ACTION_NAMES)]
    action_probs = _format_probability_distribution(step_trace.get("action_probs"), action_labels)
    region_probs = _format_probability_distribution(step_trace.get("region_probs"), top_k=2)
    patch_probs = _format_probability_distribution(step_trace.get("patch_probs"), top_k=2)
    parts: List[str] = [
        f"step={step_trace.get('step_idx', '?')}",
        f"act={action}",
        # f"seq_len={step_trace.get('sequence_length_after', step_trace.get('sequence_length_before', '?'))}",
        f"act_prob={action_probs}",
        f"region_prob={region_probs}",
        f"patch_prob={patch_probs}",
    ]
    if step_trace.get("region_index") is not None:
        parts.append(f"region={step_trace['region_index']}")
    if step_trace.get("patch_index") is not None:
        parts.append(f"patch={step_trace['patch_index']}")
    return " | ".join(parts)


def format_trace(trace: Iterable[Dict[str, object]]) -> str:
    """Render an entire action trace as multi-line text for CLI debugging."""
    return "\n".join(format_trace_step(step) for step in trace)


def add_model_loading_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI overrides for Qwen/LVAR checkpoint loading."""
    parser.add_argument(
        "--use-checkpoint",
        dest="use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override model.use_checkpoint from the config.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Override model.checkpoint_path from the config.",
    )
    parser.add_argument(
        "--merge-lora",
        dest="merge_lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override model.merge_lora from the config.",
    )


def apply_model_loading_overrides(model_cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """Apply optional CLI model-loading overrides to a model config copy."""
    updated_cfg = dict(model_cfg)
    if args.use_checkpoint is not None:
        updated_cfg["use_checkpoint"] = args.use_checkpoint
    if args.checkpoint_path is not None:
        updated_cfg["checkpoint_path"] = args.checkpoint_path
    if args.merge_lora is not None:
        updated_cfg["merge_lora"] = args.merge_lora
    return updated_cfg
