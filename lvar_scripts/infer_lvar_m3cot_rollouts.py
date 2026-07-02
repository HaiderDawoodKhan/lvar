import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_controller_checkpoint, load_vlm_lora_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import extract_choice_candidates, verify_choice_output
from lvar.utils import (
    ACTION_NAMES_NO_GLOBAL,
    add_model_loading_args,
    add_trace_boost_args,
    apply_model_loading_overrides,
    apply_trace_boost_overrides,
    boosted_output_path,
    normalize_answer_text,
    trace_boost_is_enabled,
)
from lvar_scripts.infer_lvar_m3cot import (
    build_entropy_tracking_row,
    compute_controller_tokens,
    entropy_tracking_path,
    load_config,
    write_json,
    write_jsonl,
)


def aggregate_values(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(mean(cleaned)),
        "median": float(median(cleaned)),
        "max": float(max(cleaned)),
    }


def next_token_entropy_from_state(
    model: QwenLVAR,
    state: Dict[str, Any],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure LM next-token entropy from the current recurrent state."""
    outputs = model.backbone(
        inputs_embeds=state["inputs_embeds"],
        attention_mask=state["attention_mask"],
        **model._state_position_kwargs(state),
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


def run_sampled_rollout(
    model: QwenLVAR,
    example: Dict[str, Any],
    image_size: Optional[int],
    use_coarse_context: bool,
    add_answer_instruction: bool,
    step_entropy_top_k: Optional[int],
) -> Dict[str, Any]:
    """Run one sampled-controller rollout and collect hidden-step entropy only."""
    prepared = model.prepare_inputs(
        example["image"],
        example["question"],
        add_answer_instruction=add_answer_instruction,
        image_size=image_size,
    )
    image_tokens = model.get_projected_image_tokens(prepared)
    prepared["projected_image_tokens"] = image_tokens
    bank = model.build_visual_bank(image_tokens)
    if use_coarse_context:
        state = model.build_coarse_initial_state(prepared, bank)
    else:
        state = model.build_initial_state(prepared)
    state["sample_actions"] = True

    stopped = False
    hidden_step_entropy: List[Dict[str, Any]] = []
    for step_idx in range(model.max_steps):
        state, _, stopped, step_trace = model.forward_reasoning_step(state, bank, step_idx)
        entropy = next_token_entropy_from_state(model, state, top_k=step_entropy_top_k)
        hidden_step_entropy.append(
            {
                "step_idx": step_idx,
                "action": step_trace.get("action"),
                "entropy": entropy.get("entropy"),
                "top_k": entropy.get("top_k"),
                "vocab_size": entropy.get("vocab_size"),
                "retained_probability_mass": entropy.get("retained_probability_mass"),
            }
        )
        if stopped:
            break

    if model.use_control_tokens:
        state = model.drop_act_token(state)
    decoded = model.decode_answer(model._build_decode_state(state))
    controller_entropy_tracking = model._controller_entropy_tracking(state["trace"])
    hidden_summary = aggregate_values(metric.get("entropy") for metric in hidden_step_entropy)
    return {
        "answer": decoded["answer"],
        "generated_text": decoded["generated_text"],
        "generated_ids": decoded["generated_ids"],
        "token_entropies": decoded.get("token_entropies", []),
        "token_entropy_mean": decoded.get("token_entropy_mean"),
        "token_entropy_median": decoded.get("token_entropy_median"),
        "token_entropy_max": decoded.get("token_entropy_max"),
        "answer_option_entropy": decoded.get("answer_option_entropy"),
        "trace_attention_mass": decoded.get("trace_attention_mass"),
        "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
        "think_attention_mass": decoded.get("think_attention_mass"),
        "trace": state["trace"],
        "num_steps": len(state["trace"]),
        "stopped": stopped,
        "decode_prefix_length": decoded.get("decode_prefix_length"),
        "final_sequence_length": decoded.get("final_sequence_length"),
        "hidden_step_entropy": hidden_step_entropy,
        "hidden_step_entropy_mean": hidden_summary["mean"],
        "hidden_step_entropy_median": hidden_summary["median"],
        "hidden_step_entropy_max": hidden_summary["max"],
        **controller_entropy_tracking,
    }


def trace_for_output(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tracing = []
    for step in trace:
        step_info = {
            "step_idx": step["step_idx"],
            "action": step["action"],
            "action_id": step["action_id"],
            "action_probs": step["action_probs"],
        }
        if step.get("region_index") is not None:
            step_info["region_index"] = step["region_index"]
        if step.get("region_indices"):
            step_info["region_indices"] = step["region_indices"]
        if step.get("patch_index") is not None:
            step_info["patch_index"] = step["patch_index"]
        if step.get("patch_indices"):
            step_info["patch_indices"] = step["patch_indices"]
        step_info["nucleus_insertion_applied"] = bool(step.get("nucleus_insertion_applied", False))
        step_info["sequence_length_before"] = step["sequence_length_before"]
        step_info["sequence_length_after"] = step["sequence_length_after"]
        tracing.append(step_info)
    return tracing


def answer_key(generated_text: str, decoded_answer: Optional[str] = None) -> str:
    candidates = sorted(extract_choice_candidates(generated_text))
    if candidates:
        return candidates[0]
    if decoded_answer:
        decoded_candidates = sorted(extract_choice_candidates(decoded_answer))
        if decoded_candidates:
            return decoded_candidates[0]
        return normalize_answer_text(decoded_answer)
    return normalize_answer_text(generated_text)


def common_answer_key(rollouts: List[Dict[str, Any]]) -> str:
    counts = Counter(rollout["answer_key"] for rollout in rollouts)
    return counts.most_common(1)[0][0]


def select_variant_rollouts(
    rollouts: List[Dict[str, Any]],
    variant: str,
    rng: random.Random,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    if not rollouts:
        return False, [], {}
    if variant == "best_of_n":
        selected_key = common_answer_key(rollouts)
        selected = [rollout for rollout in rollouts if rollout["answer_key"] == selected_key]
        return bool(selected and selected[0]["correct"]), selected, {"selected_answer_key": selected_key}
    if variant == "oracle":
        correct_rollouts = [rollout for rollout in rollouts if rollout["correct"]]
        if correct_rollouts:
            return True, correct_rollouts, {"oracle_found_correct": True}
        return False, rollouts, {"oracle_found_correct": False}
    if variant == "random":
        selected = [rng.choice(rollouts)]
        return bool(selected[0]["correct"]), selected, {"selected_rollout_idx": selected[0]["rollout_idx"]}
    raise ValueError(f"Unknown rollout accuracy variant: {variant}")


def build_variant_row(
    example: Dict[str, Any],
    variant: str,
    is_correct: bool,
    selected_rollouts: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    option_entropies = [
        ((rollout.get("answer_option_entropy") or {}).get("entropy"))
        for rollout in selected_rollouts
    ]
    decoded_values = [rollout.get("token_entropy_mean") for rollout in selected_rollouts]
    hidden_values = [rollout.get("hidden_step_entropy_mean") for rollout in selected_rollouts]
    controller_values = [rollout.get("controller_entropy_mean") for rollout in selected_rollouts]
    return {
        "example_id": example["id"],
        "variant": variant,
        "correct": bool(is_correct),
        "gold_answer": example["gold_answer"],
        "raw_answer": example.get("answer", ""),
        "num_selected_rollouts": len(selected_rollouts),
        "selected_rollout_indices": [rollout["rollout_idx"] for rollout in selected_rollouts],
        "selected_answer_keys": [rollout["answer_key"] for rollout in selected_rollouts],
        "decoded_token_entropy_mean": aggregate_values(decoded_values)["mean"],
        "decoded_token_entropy_median": aggregate_values(decoded_values)["median"],
        "decoded_token_entropy_max": aggregate_values(decoded_values)["max"],
        "answer_option_entropy_mean": aggregate_values(option_entropies)["mean"],
        "answer_option_entropy_median": aggregate_values(option_entropies)["median"],
        "answer_option_entropy_max": aggregate_values(option_entropies)["max"],
        "hidden_step_entropy_mean": aggregate_values(hidden_values)["mean"],
        "hidden_step_entropy_median": aggregate_values(hidden_values)["median"],
        "hidden_step_entropy_max": aggregate_values(hidden_values)["max"],
        "controller_entropy_mean": aggregate_values(controller_values)["mean"],
        "controller_entropy_median": aggregate_values(controller_values)["median"],
        "controller_entropy_max": aggregate_values(controller_values)["max"],
        **extra,
    }


def rollout_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_rollouts.jsonl")


def variants_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_accuracy_variants.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple sampled-controller LVAR rollouts per M3CoT prompt."
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--vlm-path", default=None)
    parser.add_argument("--controller-path", default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-coarse-context", action="store_true", default=False)
    parser.add_argument("--dataset-partition", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--use-validation-set", action="store_true", help="Use validation set for inference")
    parser.add_argument("--step-entropy-top-k", type=int, default=None)
    add_model_loading_args(parser)
    add_trace_boost_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    config["model"] = apply_trace_boost_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})

    phase3_cfg = config.get("phase3", {})
    phase3_v2_cfg = config.get("phase3_v2", {})
    phase3_v2_enabled = bool(phase3_cfg.get("phase3_v2", phase3_v2_cfg.get("enabled", False)))
    phase3_v2_removes_global = bool(phase3_v2_cfg.get("remove_global", phase3_cfg.get("remove_global", True)))
    if phase3_v2_enabled and phase3_v2_removes_global:
        config["model"]["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
    if "mask_immediate_repeats" in inference_cfg:
        config["model"]["mask_immediate_repeats"] = bool(inference_cfg["mask_immediate_repeats"])

    num_rollouts = args.num_rollouts or int(inference_cfg.get("num_rollouts", 8))
    if num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive.")
    temperature = args.temperature
    if temperature is None:
        temperature = float(inference_cfg.get("rollout_temperature", config["model"].get("controller_temperature", 1.0)))
    if temperature <= 0.0:
        raise ValueError("--temperature must be greater than 0.")
    config["model"]["controller_temperature"] = temperature
    config["model"]["action_selection"] = "sample"

    seed = args.seed
    if seed is None:
        seed = int(inference_cfg.get("rollout_seed", train_cfg.get("seed", 42)))
    random_rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    step_entropy_top_k = args.step_entropy_top_k
    if step_entropy_top_k is None:
        step_entropy_top_k = inference_cfg.get("step_entropy_top_k")
    if step_entropy_top_k is not None and int(step_entropy_top_k) <= 0:
        raise ValueError("--step-entropy-top-k must be a positive integer or omitted.")
    step_entropy_top_k = int(step_entropy_top_k) if step_entropy_top_k is not None else None

    dataset_partition = args.dataset_partition or inference_cfg.get("dataset_partition", "test")
    dataset_partition = "validation" if args.use_validation_set else dataset_partition
    split_seed = int(inference_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(inference_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    dataset_limit = args.limit if args.limit is not None else inference_cfg.get("limit")
    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(dataset_options, limit=dataset_limit, partition=dataset_partition)
    print(f"Loaded {len(dataset)} examples from partition '{dataset_partition}'")

    model = QwenLVAR(config["model"])
    phase4_vlm_checkpoint_path = args.vlm_path or inference_cfg.get(
        "phase4_vlm_checkpoint_path",
        config.get("phase5", {}).get("phase4_vlm_checkpoint_path", ""),
    )
    if phase4_vlm_checkpoint_path:
        loaded = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        print(
            f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}"
            if loaded
            else f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}"
        )

    controller_checkpoint_path = args.controller_path or inference_cfg.get(
        "controller_checkpoint_path",
        config.get("phase5", {}).get("controller_checkpoint_path", ""),
    )
    if controller_checkpoint_path:
        loaded = load_controller_checkpoint(model, controller_checkpoint_path)
        print(
            f"Loaded controller checkpoint: {controller_checkpoint_path}"
            if loaded
            else f"Controller checkpoint not found: {controller_checkpoint_path}"
        )

    model.eval()
    image_size = inference_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    print(f"Using inference image size: {image_size}x{image_size}")
    print(f"Sampling {num_rollouts} rollouts per prompt at controller temperature {temperature}")

    requested_output = args.output or inference_cfg.get(
        "rollout_output_path",
        "outputs/m3cot_lvar_rollout_predictions.jsonl",
    )
    output_path = Path(
        boosted_output_path(
            str(requested_output),
            enabled=trace_boost_is_enabled(config["model"]),
        )
    )

    rollout_rows: List[Dict[str, Any]] = []
    entropy_rows: List[Dict[str, Any]] = []
    variant_rows: List[Dict[str, Any]] = []
    variant_correct = Counter()
    variant_total = Counter()
    total_rollout_correct = 0
    total_rollouts = 0

    for example in tqdm(dataset, total=len(dataset), desc="Rollout inference"):
        example_rollouts: List[Dict[str, Any]] = []
        for rollout_idx in range(num_rollouts):
            with torch.no_grad():
                output = run_sampled_rollout(
                    model,
                    example,
                    image_size=image_size,
                    use_coarse_context=args.use_coarse_context,
                    add_answer_instruction=False,
                    step_entropy_top_k=step_entropy_top_k,
                )
            generated_text = output["generated_text"]
            is_correct = verify_choice_output(generated_text, example["gold_answer"])
            answer = answer_key(generated_text, output.get("answer"))
            num_controller_tokens = compute_controller_tokens(output["trace"])
            num_output_tokens = len(output["generated_ids"])

            total_rollouts += 1
            total_rollout_correct += int(is_correct)
            rollout_row = {
                "example_id": example["id"],
                "rollout_idx": rollout_idx,
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "raw_answer": example.get("answer", ""),
                "domain": example.get("domain"),
                "topic": example.get("topic"),
                "answer_key": answer,
                "correct": is_correct,
                "num_steps": output["num_steps"],
                "num_controller_tokens": num_controller_tokens,
                "num_output_tokens": num_output_tokens,
                "num_total_tokens": num_controller_tokens + num_output_tokens,
                "generated_text": generated_text,
                "trace": trace_for_output(output["trace"]),
                "hidden_step_entropy": output["hidden_step_entropy"],
                "hidden_step_entropy_mean": output["hidden_step_entropy_mean"],
                "hidden_step_entropy_median": output["hidden_step_entropy_median"],
                "hidden_step_entropy_max": output["hidden_step_entropy_max"],
                "trace_attention_mass": output.get("trace_attention_mass"),
                "visual_trace_attention_mass": output.get("visual_trace_attention_mass"),
                "think_attention_mass": output.get("think_attention_mass"),
                "token_entropy_mean": output.get("token_entropy_mean"),
                "answer_option_entropy": output.get("answer_option_entropy"),
                "controller_entropy_mean": output.get("controller_entropy_mean"),
            }
            rollout_rows.append(rollout_row)
            example_rollouts.append(rollout_row)
            entropy_row = build_entropy_tracking_row(example, output, is_correct)
            entropy_row["rollout_idx"] = rollout_idx
            entropy_row["answer_key"] = answer
            entropy_row["hidden_step_entropy"] = output["hidden_step_entropy"]
            entropy_row["hidden_step_entropy_mean"] = output["hidden_step_entropy_mean"]
            entropy_row["hidden_step_entropy_median"] = output["hidden_step_entropy_median"]
            entropy_row["hidden_step_entropy_max"] = output["hidden_step_entropy_max"]
            entropy_rows.append(entropy_row)

        for variant in ("best_of_n", "oracle", "random"):
            is_correct, selected_rollouts, extra = select_variant_rollouts(
                example_rollouts,
                variant=variant,
                rng=random_rng,
            )
            variant_correct[variant] += int(is_correct)
            variant_total[variant] += 1
            variant_rows.append(build_variant_row(example, variant, is_correct, selected_rollouts, extra))

    write_jsonl(rollout_output_path(output_path), rollout_rows)
    write_json(entropy_tracking_path(output_path), entropy_rows)
    write_jsonl(variants_output_path(output_path), variant_rows)

    num_examples = len(dataset)
    summary = {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"),
        "dataset_partition": dataset_partition,
        "num_examples": num_examples,
        "num_rollouts": num_rollouts,
        "controller_temperature": temperature,
        "seed": seed,
        "coarse_context": args.use_coarse_context,
        "phase4_vlm_checkpoint": phase4_vlm_checkpoint_path,
        "controller_checkpoint": controller_checkpoint_path,
        "step_entropy_top_k": step_entropy_top_k,
        "track_prefix_rollouts": False,
        "rollout_predictions_path": str(rollout_output_path(output_path)),
        "entropy_tracking_path": str(entropy_tracking_path(output_path)),
        "accuracy_variants_path": str(variants_output_path(output_path)),
        "metrics": {
            "rollout_total": total_rollouts,
            "rollout_correct": total_rollout_correct,
            "rollout_accuracy": round(total_rollout_correct / total_rollouts, 4) if total_rollouts else 0.0,
            "best_of_n_accuracy": (
                round(variant_correct["best_of_n"] / variant_total["best_of_n"], 4)
                if variant_total["best_of_n"] else 0.0
            ),
            "oracle_accuracy": (
                round(variant_correct["oracle"] / variant_total["oracle"], 4)
                if variant_total["oracle"] else 0.0
            ),
            "random_accuracy": (
                round(variant_correct["random"] / variant_total["random"], 4)
                if variant_total["random"] else 0.0
            ),
        },
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)

    print(f"Wrote rollout predictions to {rollout_output_path(output_path)}")
    print(f"Wrote entropy tracking to {entropy_tracking_path(output_path)}")
    print(f"Wrote accuracy variants to {variants_output_path(output_path)}")
    print(f"Wrote summary to {summary_path}")
    print("\nResults")
    print(f"  Rollout accuracy: {summary['metrics']['rollout_accuracy']:.4f}")
    print(f"  Best-of-N accuracy: {summary['metrics']['best_of_n_accuracy']:.4f}")
    print(f"  Oracle accuracy: {summary['metrics']['oracle_accuracy']:.4f}")
    print(f"  Random accuracy: {summary['metrics']['random_accuracy']:.4f}")


if __name__ == "__main__":
    main()
