import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import verify_choice_output
from lvar.utils import (
    add_model_loading_args,
    add_trace_boost_args,
    apply_model_loading_overrides,
    apply_trace_boost_overrides,
    boosted_output_path,
    trace_boost_is_enabled,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def entropy_tracking_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_entropy_tracking.json")


def build_entropy_tracking_row(
    example_id: Any,
    example: Dict[str, Any],
    decoded: Dict[str, Any],
    is_correct: bool,
    context: str,
    context_mode: str,
    trace_variant: str,
    visual_index_mode: str,
) -> Dict[str, Any]:
    option_entropy = decoded.get("answer_option_entropy") or {}
    return {
        "example_id": example_id,
        "correct": is_correct,
        "gold_answer": example["gold_answer"],
        "raw_answer": example.get("answer", ""),
        "decoded_answer": decoded.get("answer"),
        "context": context,
        "context_mode": context_mode,
        "trace_variant": trace_variant,
        "visual_index_mode": visual_index_mode,
        "num_trace_actions": len(decoded["trace"]),
        "num_output_tokens": len(decoded["generated_ids"]),
        "answer_option_entropy": option_entropy.get("entropy"),
        "answer_option_probabilities": option_entropy.get("softmax_option_probabilities"),
        "answer_option_raw_probabilities": option_entropy.get("raw_option_probabilities"),
        "answer_option_token_ids": option_entropy.get("option_token_ids"),
        "answer_option_selected_option": option_entropy.get("selected_option"),
        "answer_option_selected_token_id": option_entropy.get("selected_token_id"),
        "answer_option_decoded_token_index": option_entropy.get("decoded_token_index"),
        "decoded_token_entropies": decoded["token_entropies"],
        "decoded_token_entropy_mean": decoded["token_entropy_mean"],
        "decoded_token_entropy_median": decoded["token_entropy_median"],
        "decoded_token_entropy_max": decoded["token_entropy_max"],
        "trace_attention_mass": decoded.get("trace_attention_mass"),
        "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
        "think_attention_mass": decoded.get("think_attention_mass"),
    }


def normalize_context_mode(context: str) -> str:
    mode = str(context).strip().lower()
    if mode in {"global", "full", "full_image", "full_context"}:
        return "full_context"
    if mode in {"coarse", "coarse_context", "global_mean", "global_token"}:
        return "global_mean"
    raise ValueError("context must be one of: global, coarse, full_context, global_mean.")


def rewrite_visual_action_index(
    action: Dict[str, Any],
    index_mode: str,
    num_regions: int,
    num_patches: int,
    rng: Any = random,
) -> Dict[str, Any]:
    """Copy one replay action, changing only its visual-bank index when requested."""
    rewritten = dict(action)
    action_type = str(rewritten.get("type", "")).upper()
    mode = str(index_mode).strip().lower()
    if mode not in {"original", "random", "last"}:
        raise ValueError("index_mode must be one of: original, random, last.")

    if action_type == "REGION":
        if num_regions <= 0:
            raise ValueError("Cannot replay a REGION action with an empty region bank.")
        if mode == "random":
            rewritten["region_idx"] = rng.randrange(num_regions)
        elif mode == "last":
            rewritten["region_idx"] = num_regions - 1
    elif action_type == "PATCH":
        if num_patches <= 0:
            raise ValueError("Cannot replay a PATCH action with an empty patch bank.")
        if mode == "random":
            rewritten["patch_idx"] = rng.randrange(num_patches)
        elif mode == "last":
            rewritten["patch_idx"] = num_patches - 1
    return rewritten


def collect_dataset_by_id(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    examples_by_id: Dict[str, Dict[str, Any]] = {}
    for index in range(len(dataset)):  # type: ignore[arg-type]
        example = dataset[index]  # type: ignore[index]
        examples_by_id[str(example.get("id", index))] = example
    return examples_by_id


def decision_improvement(decision: Dict[str, Any]) -> float:
    if "improvement" in decision:
        return float(decision.get("improvement") or 0.0)
    ce_noop = decision.get("ce_noop")
    ce_selected = decision.get("ce_selected")
    if ce_noop is None or ce_selected is None:
        return 0.0
    return float(ce_noop) - float(ce_selected)


def is_visual_block(actions: List[Dict[str, Any]]) -> bool:
    return any(str(action.get("type", "")).upper() in {"PATCH", "REGION", "GLOBAL"} for action in actions)


def build_filtered_replay_trace(
    decisions: Iterable[Dict[str, Any]],
    visual_or_region_min_improvement: float,
    think_min_improvement: float,
    apply_cap: bool,
    max_decision_blocks_per_example: int,
    max_primitive_actions_per_example: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Tuple[int, float, List[Dict[str, Any]]]] = []
    skipped = Counter()

    for index, decision in enumerate(decisions):
        raw_actions = decision.get("actions") or []
        if not raw_actions:
            skipped["noop"] += 1
            continue

        actions = []
        for action in raw_actions:
            copied = dict(action)
            copied["type"] = str(copied.get("type", "")).upper()
            actions.append(copied)

        improvement = decision_improvement(decision)
        action_names = {str(action.get("type", "")).upper() for action in actions}
        if action_names == {"THINK"}:
            if improvement < think_min_improvement:
                skipped["weak_think"] += 1
                continue
        elif is_visual_block(actions):
            if improvement < visual_or_region_min_improvement:
                skipped["weak_visual"] += 1
                continue
        else:
            skipped["unsupported"] += 1
            continue

        candidates.append((index, improvement, actions))

    if apply_cap:
        selected_blocks: List[Tuple[int, float, List[Dict[str, Any]]]] = []
        primitive_count = 0
        for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
            actions = candidate[2]
            if len(selected_blocks) >= max_decision_blocks_per_example:
                skipped["cap_blocks"] += 1
                continue
            if primitive_count + len(actions) > max_primitive_actions_per_example:
                skipped["cap_primitives"] += 1
                continue
            selected_blocks.append(candidate)
            primitive_count += len(actions)
        selected_blocks.sort(key=lambda item: item[0])
    else:
        selected_blocks = sorted(candidates, key=lambda item: item[0])

    trace = []
    for _, _, actions in selected_blocks:
        trace.extend(actions)
    trace.append({"type": "STOP"})

    metrics = {
        "candidate_blocks": len(candidates),
        "kept_blocks": len(selected_blocks) + 1,
        "kept_non_stop_blocks": len(selected_blocks),
        "kept_primitives": len(trace),
        "skipped_blocks": dict(skipped),
        "cap_applied": bool(apply_cap),
        "converted_noop_to_stop": 0,
        "removed_global": False,
        "visual_dropout_applied": False,
    }
    return trace, metrics


def build_replay_trace(
    trace_row: Dict[str, Any],
    trace_variant: str,
    visual_or_region_min_improvement: float,
    think_min_improvement: float,
    max_decision_blocks_per_example: int,
    max_primitive_actions_per_example: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if trace_variant == "raw":
        trace = [dict(action) for action in (trace_row.get("trace") or [])]
        return trace, {
            "variant": trace_variant,
            "source": "trace",
            "candidate_blocks": None,
            "kept_blocks": None,
            "kept_non_stop_blocks": None,
            "kept_primitives": len(trace),
            "skipped_blocks": {},
            "cap_applied": False,
            "converted_noop_to_stop": 0,
            "removed_global": False,
            "visual_dropout_applied": False,
        }

    apply_cap = trace_variant == "filtered_cap"
    trace, metrics = build_filtered_replay_trace(
        trace_row.get("decisions", []),
        visual_or_region_min_improvement=visual_or_region_min_improvement,
        think_min_improvement=think_min_improvement,
        apply_cap=apply_cap,
        max_decision_blocks_per_example=max_decision_blocks_per_example,
        max_primitive_actions_per_example=max_primitive_actions_per_example,
    )
    metrics.update({"variant": trace_variant, "source": "decisions"})
    return trace, metrics


def replay_trace_and_decode(
    model: QwenLVAR,
    example: Dict[str, Any],
    trace: List[Dict[str, Any]],
    context_mode: str,
    image_size: Optional[int],
    add_answer_instruction: bool,
    visual_index_mode: str = "original",
    rng: Any = random,
) -> Dict[str, Any]:
    prepared = model.prepare_inputs(
        example["image"],
        example["question"],
        add_answer_instruction=add_answer_instruction,
        image_size=image_size,
    )
    image_tokens = model.get_projected_image_tokens(prepared)
    prepared["projected_image_tokens"] = image_tokens
    bank = model.build_visual_bank(image_tokens)
    if context_mode == "full_context":
        state = model.build_initial_state(prepared)
    else:
        state = model.build_coarse_initial_state(prepared, bank)

    replayed_trace = []
    for step_idx, action in enumerate(trace):
        action_type = str(action.get("type", "")).upper()
        replay_action = rewrite_visual_action_index(
            action,
            index_mode=visual_index_mode,
            num_regions=int(bank["raw_regions"].size(0)),
            num_patches=int(bank["patches"].size(0)),
            rng=rng,
        )
        before = int(state["inputs_embeds"].size(1))
        model.apply_mined_actions(state, bank, [replay_action])
        after = int(state["inputs_embeds"].size(1))
        step_info = {
            "step_idx": step_idx,
            "action": action_type,
            "sequence_length_before": before,
            "sequence_length_after": after,
        }
        if replay_action.get("region_idx") is not None:
            step_info["region_index"] = int(replay_action["region_idx"])
            if visual_index_mode != "original" and action.get("region_idx") is not None:
                step_info["source_region_index"] = int(action["region_idx"])
        if replay_action.get("patch_idx") is not None:
            step_info["patch_index"] = int(replay_action["patch_idx"])
            if visual_index_mode != "original" and action.get("patch_idx") is not None:
                step_info["source_patch_index"] = int(action["patch_idx"])
        replayed_trace.append(step_info)
        if action_type == "STOP":
            break

    if model.use_control_tokens:
        state = model.drop_act_token(state)
    with torch.no_grad():
        decoded = model.decode_answer(model._build_decode_state(state))
    return {
        "generated_text": decoded["generated_text"],
        "answer": decoded["answer"],
        "generated_ids": decoded["generated_ids"],
        "token_entropies": decoded["token_entropies"],
        "token_entropy_mean": decoded["token_entropy_mean"],
        "token_entropy_median": decoded["token_entropy_median"],
        "token_entropy_max": decoded["token_entropy_max"],
        "answer_option_entropy": decoded["answer_option_entropy"],
        "trace_attention_mass": decoded.get("trace_attention_mass"),
        "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
        "think_attention_mass": decoded.get("think_attention_mass"),
        "trace_boost_attention_observations": decoded.get("trace_boost_attention_observations", 0),
        "trace_boost_softmax_hits": decoded.get("trace_boost_softmax_hits", 0),
        "trace": replayed_trace,
        "decode_prefix_length": decoded["decode_prefix_length"],
        "final_sequence_length": decoded["final_sequence_length"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay mined M3CoT traces as oracle controller actions, decode answers, and report accuracy."
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-partition", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--trace-variant",
        default="raw",
        choices=["raw", "filtered_cap", "filtered_no_cap"],
        help="Replay raw mined trace, threshold-filtered trace with caps, or threshold-filtered trace without caps.",
    )
    parser.add_argument(
        "--context",
        default="coarse",
        choices=["global", "coarse", "full_context", "global_mean"],
        help="'global'/'full_context' replays on the full image-token prompt; 'coarse'/'global_mean' replays on one pooled visual token.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--visual-index-mode",
        default="original",
        choices=["original", "random", "last"],
        help=(
            "Choose visual indices while preserving the mined trace order: "
            "'original' uses mined indices, 'random' samples each PATCH/REGION index, "
            "and 'last' always uses the final patch or region."
        ),
    )
    parser.add_argument("--add-answer-instruction", action="store_true", default=False)
    add_model_loading_args(parser)
    add_trace_boost_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    config["model"] = apply_trace_boost_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    phase2_cfg = config.get("phase2", {})
    phase3_cfg = config.get("phase3", {})
    inference_cfg = config.get("inference", {})

    seed = int(args.seed if args.seed is not None else phase2_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trace_path = Path(args.trace_path)
    trace_rows = read_jsonl(trace_path)
    dataset = build_dataset(dataset_cfg, limit=args.limit, partition=args.dataset_partition)
    examples_by_id = collect_dataset_by_id(dataset)
    context_mode = normalize_context_mode(args.context)
    image_size = inference_cfg.get("image_size", phase2_cfg.get("image_size", 280))
    visual_or_region_min_improvement = float(phase3_cfg.get("visual_or_region_min_improvement", 0.05))
    think_min_improvement = float(phase3_cfg.get("think_min_improvement", 0.03))
    max_decision_blocks = int(phase3_cfg.get("max_decision_blocks_per_example", 6))
    max_primitive_actions = int(phase3_cfg.get("max_primitive_actions_per_example", 8))

    model = QwenLVAR(config["model"])
    model.eval()

    output_path = Path(
        boosted_output_path(
            args.output,
            enabled=trace_boost_is_enabled(config["model"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entropy_path = entropy_tracking_path(output_path)
    entropy_rows = []

    total = 0
    correct = 0
    missing = []
    total_trace_actions = 0
    total_output_tokens = 0
    transform_totals: Counter[str] = Counter()
    skipped_block_totals: Counter[str] = Counter()
    attention_mass_values = {
        "trace_attention_mass": [],
        "visual_trace_attention_mass": [],
        "think_attention_mass": [],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        for trace_row in tqdm(trace_rows, total=len(trace_rows), desc="Replaying traces"):
            example_id = trace_row.get("example_id")
            example = examples_by_id.get(str(example_id))
            if example is None:
                missing.append(example_id)
                continue

            replay_trace, transform_metrics = build_replay_trace(
                trace_row,
                trace_variant=args.trace_variant,
                visual_or_region_min_improvement=visual_or_region_min_improvement,
                think_min_improvement=think_min_improvement,
                max_decision_blocks_per_example=max_decision_blocks,
                max_primitive_actions_per_example=max_primitive_actions,
            )

            with torch.no_grad():
                decoded = replay_trace_and_decode(
                    model=model,
                    example=example,
                    trace=replay_trace,
                    context_mode=context_mode,
                    image_size=image_size,
                    add_answer_instruction=bool(args.add_answer_instruction),
                    visual_index_mode=args.visual_index_mode,
                )

            generated_text = decoded["generated_text"]
            is_correct = verify_choice_output(generated_text, example["gold_answer"])
            correct += int(is_correct)
            total += 1
            total_trace_actions += len(decoded["trace"])
            total_output_tokens += len(decoded["generated_ids"])
            for key in ("candidate_blocks", "kept_blocks", "kept_non_stop_blocks", "kept_primitives", "converted_noop_to_stop"):
                value = transform_metrics.get(key)
                if value is not None:
                    transform_totals[key] += int(value)
            skipped_block_totals.update(transform_metrics.get("skipped_blocks", {}) or {})

            row = {
                "example_id": example_id,
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "raw_answer": example.get("answer", ""),
                "domain": example.get("domain"),
                "topic": example.get("topic"),
                "context": args.context,
                "context_mode": context_mode,
                "trace_variant": args.trace_variant,
                "visual_index_mode": args.visual_index_mode,
                "correct": is_correct,
                "num_trace_actions": len(decoded["trace"]),
                "num_output_tokens": len(decoded["generated_ids"]),
                "generated_text": generated_text,
                "decoded_answer": decoded["answer"],
                "transform": transform_metrics,
                "trace": decoded["trace"],
                "trace_attention_mass": decoded.get("trace_attention_mass"),
                "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
                "think_attention_mass": decoded.get("think_attention_mass"),
            }
            write_jsonl_row(handle, row)
            for key in attention_mass_values:
                value = decoded.get(key)
                if value is not None:
                    attention_mass_values[key].append(float(value))
            entropy_rows.append(
                build_entropy_tracking_row(
                    example_id=example_id,
                    example=example,
                    decoded=decoded,
                    is_correct=is_correct,
                    context=args.context,
                    context_mode=context_mode,
                    trace_variant=args.trace_variant,
                    visual_index_mode=args.visual_index_mode,
                )
            )

    accuracy = correct / total if total else 0.0
    write_json(entropy_path, entropy_rows)
    summary = {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"),
        "dataset_partition": args.dataset_partition,
        "trace_path": str(trace_path),
        "output_path": str(output_path),
        "entropy_tracking_path": str(entropy_path),
        "context": args.context,
        "context_mode": context_mode,
        "trace_variant": args.trace_variant,
        "visual_index_mode": args.visual_index_mode,
        "filtering": {
            "visual_or_region_min_improvement": visual_or_region_min_improvement,
            "think_min_improvement": think_min_improvement,
            "max_decision_blocks_per_example": max_decision_blocks,
            "max_primitive_actions_per_example": max_primitive_actions,
            "no_op_to_stop_conversion": False,
            "remove_global": False,
            "visual_dropout": False,
        },
        "checkpoint_path": config["model"].get("checkpoint_path"),
        "use_checkpoint": config["model"].get("use_checkpoint"),
        "trace_boost": model.trace_boost_config.to_dict(),
        "add_answer_instruction": bool(args.add_answer_instruction),
        "num_trace_rows": len(trace_rows),
        "num_missing_examples": len(missing),
        "missing_example_ids": missing[:50],
        "transform_totals": dict(transform_totals),
        "skipped_block_totals": dict(skipped_block_totals),
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_trace_actions": round(total_trace_actions / total, 2) if total else 0.0,
            "avg_output_tokens": round(total_output_tokens / total, 2) if total else 0.0,
            "trace_attention_mass": (
                sum(attention_mass_values["trace_attention_mass"])
                / len(attention_mass_values["trace_attention_mass"])
                if attention_mass_values["trace_attention_mass"] else None
            ),
            "visual_trace_attention_mass": (
                sum(attention_mass_values["visual_trace_attention_mass"])
                / len(attention_mass_values["visual_trace_attention_mass"])
                if attention_mass_values["visual_trace_attention_mass"] else None
            ),
            "think_attention_mass": (
                sum(attention_mass_values["think_attention_mass"])
                / len(attention_mass_values["think_attention_mass"])
                if attention_mass_values["think_attention_mass"] else None
            ),
        },
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)

    print(f"Wrote {total} oracle-forced predictions to {output_path}")
    print(f"Wrote entropy tracking to {entropy_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    if missing:
        print(f"Skipped {len(missing)} traces whose example_id was not found in the dataset partition.")


if __name__ == "__main__":
    main()
