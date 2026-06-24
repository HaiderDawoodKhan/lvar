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
        "action_prefix_metrics": decoded.get("action_prefix_metrics", []),
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


def flatten_replay_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten decision-level replay blocks into primitive controller actions."""
    return [dict(action) for block in blocks for action in (block.get("actions") or [])]


def raw_replay_blocks(trace_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover oracle decision blocks, falling back to one block per primitive action."""
    decisions = trace_row.get("decisions") or []
    blocks = [
        {
            "label": str(decision.get("selected") or "").upper(),
            "actions": [dict(action) for action in (decision.get("actions") or [])],
        }
        for decision in decisions
        if decision.get("actions")
    ]
    decision_trace = flatten_replay_blocks(blocks)
    raw_trace = [dict(action) for action in (trace_row.get("trace") or [])]
    raw_without_stop = [action for action in raw_trace if str(action.get("type", "")).upper() != "STOP"]
    if not blocks or decision_trace != raw_without_stop:
        blocks = [
            {"label": str(action.get("type", "")).upper(), "actions": [dict(action)]}
            for action in raw_without_stop
        ]
    if any(str(action.get("type", "")).upper() == "STOP" for action in raw_trace):
        blocks.append({"label": "STOP", "actions": [{"type": "STOP"}]})
    return blocks


def transform_replay_blocks(
    blocks: List[Dict[str, Any]],
    trace_variant: str,
    rng: Any = random,
) -> List[Dict[str, Any]]:
    """Apply inference-only ablations to decision-level oracle blocks."""
    copied = [
        {"label": str(block.get("label", "")).upper(), "actions": [dict(a) for a in block.get("actions", [])]}
        for block in blocks
    ]
    if trace_variant == "no_visual":
        transformed = []
        for block in copied:
            actions = [
                action
                for action in block["actions"]
                if str(action.get("type", "")).upper() not in {"PATCH", "REGION", "GLOBAL"}
            ]
            if actions:
                transformed.append({"label": block["label"], "actions": actions})
        return transformed
    if trace_variant == "shuffled":
        terminal = [block for block in copied if any(str(a.get("type", "")).upper() == "STOP" for a in block["actions"])]
        nonterminal = [block for block in copied if block not in terminal]
        rng.shuffle(nonterminal)
        return nonterminal + terminal
    if trace_variant == "no_reasoning":
        return []
    return copied


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
    rng: Any = random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if trace_variant in {"raw", "no_visual", "no_reasoning", "shuffled"}:
        source_blocks = raw_replay_blocks(trace_row)
        blocks = transform_replay_blocks(source_blocks, trace_variant=trace_variant, rng=rng)
        trace = flatten_replay_blocks(blocks)
        source_trace = flatten_replay_blocks(source_blocks)
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
            "removed_visual_actions": len(source_trace) - len(trace) if trace_variant == "no_visual" else 0,
            "removed_global": trace_variant == "no_visual" and any(
                str(action.get("type", "")).upper() == "GLOBAL" for action in source_trace
            ),
            "visual_dropout_applied": False,
            "shuffled": trace_variant == "shuffled",
            "no_reasoning": trace_variant == "no_reasoning",
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


def build_replay_blocks(
    trace_row: Dict[str, Any],
    replay_trace: List[Dict[str, Any]],
    trace_variant: str,
    rng: Any = random,
) -> List[Dict[str, Any]]:
    """Build decision-level blocks for replay and optional per-prefix measurements."""
    if trace_variant in {"raw", "no_visual", "no_reasoning", "shuffled"}:
        return transform_replay_blocks(raw_replay_blocks(trace_row), trace_variant=trace_variant, rng=rng)
    return [
        {"label": str(action.get("type", "")).upper(), "actions": [dict(action)]}
        for action in replay_trace
    ]


def next_token_entropy_from_state(
    model: QwenLVAR,
    state: Dict[str, Any],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure next-token LM-head entropy after an oracle action block is inserted."""
    outputs = model.backbone(
        inputs_embeds=state["inputs_embeds"],
        attention_mask=state["attention_mask"],
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    logits = outputs.logits[:, -1, :].float()
    vocab_size = int(logits.size(-1))
    if top_k is None or top_k <= 0 or top_k >= vocab_size:
        return {
            "entropy": model._entropy_from_logits(logits),
            "top_k": None,
            "vocab_size": vocab_size,
            "retained_probability_mass": 1.0,
        }

    k = min(int(top_k), vocab_size)
    top_values = torch.topk(logits, k=k, dim=-1).values
    top_log_probs = torch.log_softmax(top_values, dim=-1)
    top_probs = top_log_probs.exp()
    entropy = -(top_probs * top_log_probs).sum(dim=-1)
    retained_mass = torch.exp(torch.logsumexp(top_values, dim=-1) - torch.logsumexp(logits, dim=-1))
    return {
        "entropy": float(entropy.squeeze().detach().cpu().item()),
        "top_k": k,
        "vocab_size": vocab_size,
        "retained_probability_mass": float(retained_mass.squeeze().detach().cpu().item()),
    }


def decode_current_replay_state(model: QwenLVAR, state: Dict[str, Any]) -> Dict[str, Any]:
    """Decode from a detached view without mutating the ongoing replay state."""
    decode_state = model._build_decode_state(state)
    if model.use_control_tokens:
        decode_state = model.drop_act_token(decode_state)
    return model.decode_answer(decode_state)


def decode_without_reasoning(
    model: QwenLVAR,
    example: Dict[str, Any],
    image_size: Optional[int],
    add_answer_instruction: bool,
) -> Dict[str, Any]:
    """Run the normal multimodal VLM prompt with no latent/controller reasoning."""
    prepared = model.prepare_inputs(
        example["image"],
        example["question"],
        add_answer_instruction=add_answer_instruction,
        image_size=image_size,
    )
    image_tokens = model.get_projected_image_tokens(prepared)
    prepared["projected_image_tokens"] = image_tokens
    inputs_embeds, attention_mask = model._build_multimodal_embeddings(prepared)
    state = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "latent_pos": None,
        "act_pos": None,
        "trace_all_positions": [],
        "trace_visual_positions": [],
    }
    return model.decode_answer(state)


def replay_trace_and_decode(
    model: QwenLVAR,
    example: Dict[str, Any],
    trace: List[Dict[str, Any]],
    context_mode: str,
    image_size: Optional[int],
    add_answer_instruction: bool,
    visual_index_mode: str = "original",
    rng: Any = random,
    replay_blocks: Optional[List[Dict[str, Any]]] = None,
    no_reasoning: bool = False,
    track_step_hidden_entropy: bool = False,
    step_entropy_top_k: Optional[int] = None,
    track_prefix_rollouts: bool = False,
) -> Dict[str, Any]:
    if no_reasoning:
        decoded = decode_without_reasoning(
            model,
            example,
            image_size=image_size,
            add_answer_instruction=add_answer_instruction,
        )
        return {
            **decoded,
            "trace": [],
            "action_prefix_metrics": [],
        }

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
    action_prefix_metrics = []
    blocks = replay_blocks or [
        {"label": str(action.get("type", "")).upper(), "actions": [action]}
        for action in trace
    ]
    primitive_step_idx = 0
    prefix_block_labels: List[str] = []
    for block_idx, block in enumerate(blocks):
        block_label = str(block.get("label", "")).upper()
        block_actions = block.get("actions") or []
        rewritten_actions = []
        terminal = False
        for action in block_actions:
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
                "step_idx": primitive_step_idx,
                "block_idx": block_idx,
                "block_action": block_label,
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
            rewritten_actions.append(dict(replay_action))
            primitive_step_idx += 1
            if action_type == "STOP":
                terminal = True
                break

        if terminal:
            break
        if not rewritten_actions:
            continue
        prefix_block_labels.append(block_label)
        if not track_step_hidden_entropy and not track_prefix_rollouts:
            continue

        prefix_metric: Dict[str, Any] = {
            "block_idx": block_idx,
            "action": block_label,
            "prefix_trace": list(prefix_block_labels),
            "actions": rewritten_actions,
            "num_primitive_actions": len(rewritten_actions),
            "sequence_length": int(state["inputs_embeds"].size(1)),
        }
        if track_step_hidden_entropy:
            prefix_metric["next_token_entropy"] = next_token_entropy_from_state(
                model,
                state,
                top_k=step_entropy_top_k,
            )
        if track_prefix_rollouts:
            rollout = decode_current_replay_state(model, state)
            rollout_correct = verify_choice_output(rollout["generated_text"], example["gold_answer"])
            prefix_metric["rollout"] = {
                "generated_text": rollout["generated_text"],
                "decoded_answer": rollout["answer"],
                "correct": rollout_correct,
                "generated_ids": rollout["generated_ids"],
                "token_entropies": rollout["token_entropies"],
                "token_entropy_mean": rollout["token_entropy_mean"],
                "token_entropy_median": rollout["token_entropy_median"],
                "token_entropy_max": rollout["token_entropy_max"],
                "answer_option_entropy": rollout["answer_option_entropy"],
                "trace_attention_mass": rollout.get("trace_attention_mass"),
                "visual_trace_attention_mass": rollout.get("visual_trace_attention_mass"),
                "think_attention_mass": rollout.get("think_attention_mass"),
                "decode_prefix_length": rollout["decode_prefix_length"],
                "final_sequence_length": rollout["final_sequence_length"],
            }
        action_prefix_metrics.append(prefix_metric)

    with torch.no_grad():
        decoded = decode_current_replay_state(model, state)
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
        "action_prefix_metrics": action_prefix_metrics,
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
        choices=["raw", "filtered_cap", "filtered_no_cap", "no_visual", "no_reasoning", "shuffled"],
        help=(
            "Replay the raw/filtered oracle trace, remove visual actions, bypass reasoning entirely, "
            "or shuffle oracle decision blocks while keeping STOP terminal."
        ),
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
    parser.add_argument(
        "--track-step-hidden-entropy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After every inserted oracle action block, measure next-token LM-head entropy.",
    )
    parser.add_argument(
        "--step-entropy-top-k",
        type=int,
        default=None,
        help="Renormalize stepwise LM entropy over only the top-k tokens; omit for full-vocabulary entropy.",
    )
    parser.add_argument(
        "--track-prefix-rollouts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode and score a complete answer after every inserted oracle action block.",
    )
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
    track_step_hidden_entropy = bool(
        inference_cfg.get("track_step_hidden_entropy", False)
        if args.track_step_hidden_entropy is None
        else args.track_step_hidden_entropy
    )
    track_prefix_rollouts = bool(
        inference_cfg.get("track_prefix_rollouts", False)
        if args.track_prefix_rollouts is None
        else args.track_prefix_rollouts
    )
    step_entropy_top_k = (
        args.step_entropy_top_k
        if args.step_entropy_top_k is not None
        else inference_cfg.get("step_entropy_top_k")
    )
    if step_entropy_top_k is not None and int(step_entropy_top_k) <= 0:
        raise ValueError("step_entropy_top_k must be a positive integer or null for full-vocabulary entropy.")
    step_entropy_top_k = int(step_entropy_top_k) if step_entropy_top_k is not None else None
    effective_context_mode = "full_context" if args.trace_variant == "no_reasoning" else context_mode

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
    rollout_total = 0
    rollout_correct = 0
    rollout_by_block: Dict[int, Counter[str]] = {}
    step_hidden_entropy_values: List[float] = []
    rollout_token_entropy_values: List[float] = []
    rollout_option_entropy_values: List[float] = []
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

            replay_seed = f"{seed}:{example_id}:{args.trace_variant}"
            replay_trace, transform_metrics = build_replay_trace(
                trace_row,
                trace_variant=args.trace_variant,
                visual_or_region_min_improvement=visual_or_region_min_improvement,
                think_min_improvement=think_min_improvement,
                max_decision_blocks_per_example=max_decision_blocks,
                max_primitive_actions_per_example=max_primitive_actions,
                rng=random.Random(replay_seed),
            )
            replay_blocks = build_replay_blocks(
                trace_row,
                replay_trace=replay_trace,
                trace_variant=args.trace_variant,
                rng=random.Random(replay_seed),
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
                    rng=random.Random(f"{replay_seed}:visual-indices"),
                    replay_blocks=replay_blocks,
                    no_reasoning=args.trace_variant == "no_reasoning",
                    track_step_hidden_entropy=track_step_hidden_entropy,
                    step_entropy_top_k=step_entropy_top_k,
                    track_prefix_rollouts=track_prefix_rollouts,
                )

            generated_text = decoded["generated_text"]
            is_correct = verify_choice_output(generated_text, example["gold_answer"])
            correct += int(is_correct)
            total += 1
            total_trace_actions += len(decoded["trace"])
            total_output_tokens += len(decoded["generated_ids"])
            for prefix_metric in decoded.get("action_prefix_metrics", []):
                hidden_entropy = prefix_metric.get("next_token_entropy") or {}
                if hidden_entropy.get("entropy") is not None:
                    step_hidden_entropy_values.append(float(hidden_entropy["entropy"]))
                rollout = prefix_metric.get("rollout")
                if not rollout:
                    continue
                rollout_total += 1
                rollout_correct += int(rollout["correct"])
                block_idx = int(prefix_metric["block_idx"])
                block_counts = rollout_by_block.setdefault(block_idx, Counter())
                block_counts["total"] += 1
                block_counts["correct"] += int(rollout["correct"])
                if rollout.get("token_entropy_mean") is not None:
                    rollout_token_entropy_values.append(float(rollout["token_entropy_mean"]))
                rollout_option_entropy = rollout.get("answer_option_entropy") or {}
                if rollout_option_entropy.get("entropy") is not None:
                    rollout_option_entropy_values.append(float(rollout_option_entropy["entropy"]))
            for key in (
                "candidate_blocks",
                "kept_blocks",
                "kept_non_stop_blocks",
                "kept_primitives",
                "converted_noop_to_stop",
                "removed_visual_actions",
            ):
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
                "context_mode": effective_context_mode,
                "trace_variant": args.trace_variant,
                "visual_index_mode": args.visual_index_mode,
                "correct": is_correct,
                "num_trace_actions": len(decoded["trace"]),
                "num_output_tokens": len(decoded["generated_ids"]),
                "generated_text": generated_text,
                "decoded_answer": decoded["answer"],
                "transform": transform_metrics,
                "trace": decoded["trace"],
                "action_prefix_metrics": decoded.get("action_prefix_metrics", []),
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
                    context_mode=effective_context_mode,
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
        "context_mode": effective_context_mode,
        "trace_variant": args.trace_variant,
        "visual_index_mode": args.visual_index_mode,
        "filtering": {
            "visual_or_region_min_improvement": visual_or_region_min_improvement,
            "think_min_improvement": think_min_improvement,
            "max_decision_blocks_per_example": max_decision_blocks,
            "max_primitive_actions_per_example": max_primitive_actions,
            "no_op_to_stop_conversion": False,
            "remove_global": args.trace_variant == "no_visual",
            "remove_all_visual_trace_actions": args.trace_variant == "no_visual",
            "visual_dropout": False,
        },
        "checkpoint_path": config["model"].get("checkpoint_path"),
        "use_checkpoint": config["model"].get("use_checkpoint"),
        "trace_boost": model.trace_boost_config.to_dict(),
        "add_answer_instruction": bool(args.add_answer_instruction),
        "stepwise_metrics": {
            "track_step_hidden_entropy": track_step_hidden_entropy,
            "step_entropy_top_k": step_entropy_top_k,
            "track_prefix_rollouts": track_prefix_rollouts,
        },
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
            "prefix_rollouts": {
                "total": rollout_total,
                "correct": rollout_correct,
                "accuracy": round(rollout_correct / rollout_total, 4) if rollout_total else None,
                "by_block": {
                    str(block_idx): {
                        "total": counts["total"],
                        "correct": counts["correct"],
                        "accuracy": round(counts["correct"] / counts["total"], 4) if counts["total"] else None,
                    }
                    for block_idx, counts in sorted(rollout_by_block.items())
                },
                "mean_decoded_token_entropy": (
                    sum(rollout_token_entropy_values) / len(rollout_token_entropy_values)
                    if rollout_token_entropy_values else None
                ),
                "mean_answer_option_entropy": (
                    sum(rollout_option_entropy_values) / len(rollout_option_entropy_values)
                    if rollout_option_entropy_values else None
                ),
            },
            "step_hidden_entropy_mean": (
                sum(step_hidden_entropy_values) / len(step_hidden_entropy_values)
                if step_hidden_entropy_values else None
            ),
            "step_hidden_entropy_count": len(step_hidden_entropy_values),
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
