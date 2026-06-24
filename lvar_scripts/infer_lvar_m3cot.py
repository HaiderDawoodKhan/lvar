import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_controller_checkpoint, load_vlm_lora_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import verify_choice_output
from lvar.utils import (
    ACTION_NAMES_NO_GLOBAL,
    add_model_loading_args,
    add_trace_boost_args,
    apply_model_loading_overrides,
    apply_trace_boost_overrides,
    boosted_output_path,
    trace_boost_is_enabled,
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify_output(generated_text: str, gold_answer: str) -> bool:
    return verify_choice_output(generated_text, gold_answer)


def compute_controller_tokens(trace: list) -> int:
    return sum(
        step["sequence_length_after"] - step["sequence_length_before"]
        for step in trace
    )


def entropy_tracking_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_entropy_tracking.json")


def build_entropy_tracking_row(example, output, is_correct: bool):
    option_entropy = output.get("answer_option_entropy") or {}
    return {
        "example_id": example["id"],
        "correct": is_correct,
        "gold_answer": example["gold_answer"],
        "raw_answer": example.get("answer", ""),
        "decoded_answer": output.get("answer"),
        "num_output_tokens": len(output["generated_ids"]),
        "answer_option_entropy": option_entropy.get("entropy"),
        "answer_option_probabilities": option_entropy.get("softmax_option_probabilities"),
        "answer_option_raw_probabilities": option_entropy.get("raw_option_probabilities"),
        "answer_option_token_ids": option_entropy.get("option_token_ids"),
        "answer_option_selected_option": option_entropy.get("selected_option"),
        "answer_option_selected_token_id": option_entropy.get("selected_token_id"),
        "answer_option_decoded_token_index": option_entropy.get("decoded_token_index"),
        "decoded_token_entropies": output["token_entropies"],
        "decoded_token_entropy_mean": output["token_entropy_mean"],
        "decoded_token_entropy_median": output["token_entropy_median"],
        "decoded_token_entropy_max": output["token_entropy_max"],
        "controller_action_entropies": output["controller_action_entropy_values"],
        "controller_action_entropy_mean": output["controller_action_entropy_mean"],
        "controller_action_entropy_median": output["controller_action_entropy_median"],
        "controller_action_entropy_max": output["controller_action_entropy_max"],
        "controller_region_entropies": output["controller_region_entropy_values"],
        "controller_region_entropy_mean": output["controller_region_entropy_mean"],
        "controller_region_entropy_median": output["controller_region_entropy_median"],
        "controller_region_entropy_max": output["controller_region_entropy_max"],
        "controller_patch_entropies": output["controller_patch_entropy_values"],
        "controller_patch_entropy_mean": output["controller_patch_entropy_mean"],
        "controller_patch_entropy_median": output["controller_patch_entropy_median"],
        "controller_patch_entropy_max": output["controller_patch_entropy_max"],
        "controller_entropies": output["controller_entropy_values"],
        "controller_entropy_mean": output["controller_entropy_mean"],
        "controller_entropy_median": output["controller_entropy_median"],
        "controller_entropy_max": output["controller_entropy_max"],
        "trace_attention_mass": output.get("trace_attention_mass"),
        "visual_trace_attention_mass": output.get("visual_trace_attention_mass"),
        "think_attention_mass": output.get("think_attention_mass"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LVAR inference on the M3CoT test split with controller traces."
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--phase4-vlm-checkpoint-path", default=None)
    parser.add_argument("--controller-checkpoint-path", default=None)
    parser.add_argument("--use-coarse-context", action="store_true", default=False)
    parser.add_argument(
        "--nucleus-insertion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Insert the complete top-p patch/region nucleus for each selected visual action.",
    )
    parser.add_argument("--nucleus-insertion-scope", choices=["patch", "region", "both"], default=None)
    parser.add_argument("--nucleus-insertion-top-p", type=float, default=None)
    parser.add_argument("--nucleus-insertion-max-indices", type=int, default=None)
    parser.add_argument("--use_validation_set", action="store_true", help="Use validation set for inference")
    add_model_loading_args(parser)
    add_trace_boost_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    config["model"] = apply_trace_boost_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})

    if "action_selection" in inference_cfg:
        config["model"]["action_selection"] = inference_cfg["action_selection"]
    if bool(config.get("phase3", {}).get("phase3_v2", False)) or bool(config.get("phase3", {}).get("remove_global", False)):
        config["model"]["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
    if "mask_immediate_repeats" in inference_cfg:
        config["model"]["mask_immediate_repeats"] = bool(inference_cfg["mask_immediate_repeats"])
    for key in (
        "nucleus_insertion_enabled",
        "nucleus_insertion_scope",
        "nucleus_insertion_top_p",
        "nucleus_insertion_max_indices",
    ):
        if key in inference_cfg:
            config["model"][key] = inference_cfg[key]
    if args.nucleus_insertion is not None:
        config["model"]["nucleus_insertion_enabled"] = bool(args.nucleus_insertion)
    if args.nucleus_insertion_scope is not None:
        config["model"]["nucleus_insertion_scope"] = args.nucleus_insertion_scope
    if args.nucleus_insertion_top_p is not None:
        config["model"]["nucleus_insertion_top_p"] = args.nucleus_insertion_top_p
    if args.nucleus_insertion_max_indices is not None:
        config["model"]["nucleus_insertion_max_indices"] = args.nucleus_insertion_max_indices

    dataset_partition = inference_cfg.get("dataset_partition", "test")
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

    phase4_vlm_checkpoint_path = args.phase4_vlm_checkpoint_path or inference_cfg.get(
        "phase4_vlm_checkpoint_path",
        config.get("phase5", {}).get("phase4_vlm_checkpoint_path", ""),
    )
    if phase4_vlm_checkpoint_path:
        loaded = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        if loaded:
            print(f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}")
        else:
            print(f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}")

    controller_checkpoint_path = args.controller_checkpoint_path or inference_cfg.get(
        "controller_checkpoint_path",
        config.get("phase5", {}).get("controller_checkpoint_path", ""),
    )
    if controller_checkpoint_path:
        loaded = load_controller_checkpoint(model, controller_checkpoint_path)
        if loaded:
            print(f"Loaded controller checkpoint: {controller_checkpoint_path}")
        else:
            print(f"Controller checkpoint not found: {controller_checkpoint_path}")

    model.eval()
    image_size = inference_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    print(f"Using inference image size: {image_size}x{image_size}")
    rows = []
    entropy_rows = []
    total = 0
    correct = 0
    total_controller_tokens = 0
    total_output_tokens = 0
    total_steps = 0
    attention_mass_values = {
        "trace_attention_mass": [],
        "visual_trace_attention_mass": [],
        "think_attention_mass": [],
    }

    for example in tqdm(dataset, total=len(dataset), desc="Inferring"):
        total += 1
        with torch.no_grad():
            output = model.forward(
                images=example["image"],
                questions=example["question"],
                add_answer_instruction=False,
                use_coarse_context=args.use_coarse_context,
                image_size=image_size,
            )

        generated_text = output["generated_text"]
        is_correct = verify_output(generated_text, example["gold_answer"])
        if is_correct:
            correct += 1

        num_steps = output["num_steps"]
        num_output_tokens = len(output["generated_ids"])
        num_controller_tokens = compute_controller_tokens(output["trace"])

        total_steps += num_steps
        total_output_tokens += num_output_tokens
        total_controller_tokens += num_controller_tokens

        tracing = []
        for step in output["trace"]:
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

        row = {
            "example_id": example["id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "raw_answer": example.get("answer", ""),
            "domain": example.get("domain"),
            "topic": example.get("topic"),
            "correct": is_correct,
            "num_steps": num_steps,
            "num_controller_tokens": num_controller_tokens,
            "num_output_tokens": num_output_tokens,
            "num_total_tokens": num_controller_tokens + num_output_tokens,
            "generated_text": generated_text,
            "trace": tracing,
            "trace_attention_mass": output.get("trace_attention_mass"),
            "visual_trace_attention_mass": output.get("visual_trace_attention_mass"),
            "think_attention_mass": output.get("think_attention_mass"),
        }
        rows.append(row)
        entropy_rows.append(build_entropy_tracking_row(example, output, is_correct))
        for key in attention_mass_values:
            value = output.get(key)
            if value is not None:
                attention_mass_values[key].append(float(value))

    requested_output = args.output or inference_cfg.get("output_path", "outputs/m3cot_lvar_predictions.jsonl")
    output_path = Path(
        boosted_output_path(
            str(requested_output),
            enabled=trace_boost_is_enabled(config["model"]),
        )
    )
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    entropy_path = entropy_tracking_path(output_path)
    write_json(entropy_path, entropy_rows)
    print(f"Wrote entropy tracking to {entropy_path}")

    accuracy = correct / total if total > 0 else 0.0
    avg_steps = total_steps / total if total > 0 else 0.0
    avg_controller_tokens = total_controller_tokens / total if total > 0 else 0.0
    avg_output_tokens = total_output_tokens / total if total > 0 else 0.0
    avg_total_tokens = (total_controller_tokens + total_output_tokens) / total if total > 0 else 0.0

    summary = {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"),
        "dataset_partition": dataset_partition,
        "num_examples": total,
        "coarse_context": args.use_coarse_context,
        "phase4_vlm_checkpoint": phase4_vlm_checkpoint_path,
        "controller_checkpoint": controller_checkpoint_path,
        "trace_boost": model.trace_boost_config.to_dict(),
        "nucleus_insertion": {
            "enabled": model.nucleus_insertion_enabled,
            "scope": model.nucleus_insertion_scope,
            "top_p": model.nucleus_insertion_top_p,
            "max_indices": model.nucleus_insertion_max_indices,
        },
        "entropy_tracking_path": str(entropy_path),
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_controller_actions": round(avg_steps, 2),
            "avg_controller_tokens": round(avg_controller_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "avg_total_tokens": round(avg_total_tokens, 2),
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
    summary_json_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_json_path, summary)
    print(f"Wrote summary to {summary_json_path}")

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"  Total:       {total}")
    print(f"  Correct:     {correct}")
    print(f"  Accuracy:    {accuracy:.4f} ({correct}/{total})")
    print(f"  Avg actions: {avg_steps:.2f}")
    print(f"  Avg ctrl tokens:  {avg_controller_tokens:.2f}")
    print(f"  Avg output tokens: {avg_output_tokens:.2f}")
    print(f"  Avg total tokens:  {avg_total_tokens:.2f}")


if __name__ == "__main__":
    main()
