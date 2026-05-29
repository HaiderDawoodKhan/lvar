import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_trainable_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides

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
    cleaned_text = re.sub(
        r"(?<=answer:)\s*(\n+\s*)?assistant\b",
        "",
        generated_text,
        flags=re.IGNORECASE,
    )
    letter_matches = re.finditer(
        r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = {match.group(1).upper() for match in letter_matches}

    digit_matches = re.finditer(
        r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*(\d)",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in digit_matches:
        digit = int(match.group(1))
        if 0 <= digit <= 3:
            candidates.add(chr(ord("A") + digit))

    gt_answer = gold_answer.strip().upper()
    if gt_answer.isdigit():
        digit = int(gt_answer)
        if 0 <= digit <= 3:
            gt_answer = chr(ord("A") + digit)

    return gt_answer in candidates


def compute_controller_tokens(trace: list) -> int:
    return sum(
        step["sequence_length_after"] - step["sequence_length_before"]
        for step in trace
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LVAR inference on the M3CoT test split with controller traces."
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--controller-checkpoint-path", default=None)
    parser.add_argument("--use-coarse-context", action="store_true", default=False)
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})

    if "action_selection" in inference_cfg:
        config["model"]["action_selection"] = inference_cfg["action_selection"]

    dataset_partition = inference_cfg.get("dataset_partition", "test")
    split_seed = int(inference_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(inference_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    dataset_limit = args.limit if args.limit is not None else inference_cfg.get("limit")
    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(dataset_options, limit=dataset_limit, partition=dataset_partition)
    print(f"Loaded {len(dataset)} examples from partition '{dataset_partition}'")

    model = QwenLVAR(config["model"])

    controller_checkpoint_path = args.controller_checkpoint_path or inference_cfg.get(
        "controller_checkpoint_path", ""
    )
    if controller_checkpoint_path:
        loaded = load_trainable_checkpoint(model, controller_checkpoint_path)
        if loaded:
            print(f"Loaded controller checkpoint: {controller_checkpoint_path}")
        else:
            print(f"Controller checkpoint not found: {controller_checkpoint_path}")

    model.eval()
    rows = []
    total = 0
    correct = 0
    total_controller_tokens = 0
    total_output_tokens = 0
    total_steps = 0

    for example in tqdm(dataset, total=len(dataset), desc="Inferring"):
        total += 1
        with torch.no_grad():
            output = model.forward(
                images=example["image"],
                questions=example["question"],
                add_answer_instruction=False,
                use_coarse_context=args.use_coarse_context,
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
            if step.get("patch_index") is not None:
                step_info["patch_index"] = step["patch_index"]
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
        }
        rows.append(row)

    output_path = Path(args.output) if args.output else Path(inference_cfg.get("output_path", "outputs/m3cot_lvar_predictions.jsonl"))
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")

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
        "controller_checkpoint": controller_checkpoint_path,
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_controller_actions": round(avg_steps, 2),
            "avg_controller_tokens": round(avg_controller_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "avg_total_tokens": round(avg_total_tokens, 2),
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
