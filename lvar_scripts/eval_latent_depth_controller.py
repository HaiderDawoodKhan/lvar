import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_vlm_lora_checkpoint
from lvar.latent_depth import build_latent_depth_supervision, load_fixed_think_rows
from lvar.latent_depth_controller import LatentDepthController
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides
from lvar_scripts.train_latent_depth_controller import (
    extract_features_for_example,
    group_rows_by_example,
    latent_tokens_for_depth,
    normalize_context_mode,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_depth_row_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    index: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        example_id = str(row.get("example_id"))
        depth = int(row.get("latent_depth", row.get("num_think_steps", -1)))
        if example_id and depth >= 0:
            index[example_id][depth] = row
    return index


def load_controller(
    checkpoint_path: str | Path,
    input_hidden_size: int,
    device: torch.device,
) -> tuple[LatentDepthController, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    controller = LatentDepthController(
        input_hidden_size=input_hidden_size,
        controller_hidden_size=int(metadata.get("controller_hidden_size", 512)),
        num_layers=int(metadata.get("controller_layers", 2)),
        num_heads=int(metadata.get("controller_heads", 8)),
        max_prompt_tokens=int(metadata.get("max_prompt_tokens", 10)),
        max_latent_steps=int(metadata.get("max_depth", 10)),
    ).to(device)
    controller.load_state_dict(state_dict)
    controller.eval()
    return controller, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained binary latent-depth controller.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--fixed-think-jsonl", action="append", default=[])
    parser.add_argument("--fixed-think-glob", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-partition", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--target-policy", choices=["earliest_correct", "all_correct"], default=None)
    parser.add_argument("--context", choices=["global", "coarse", "full_context", "global_mean"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--phase4-vlm-checkpoint-path", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    add_model_loading_args(parser)
    args = parser.parse_args()

    if not args.fixed_think_jsonl and not args.fixed_think_glob:
        raise ValueError("Provide at least one --fixed-think-jsonl or --fixed-think-glob.")

    config = load_config(args.config)
    model_cfg = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = dict(config["dataset"])
    train_cfg = config.get("train", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = QwenLVAR(model_cfg)
    phase4_vlm_checkpoint_path = args.phase4_vlm_checkpoint_path
    if phase4_vlm_checkpoint_path is None:
        phase4_vlm_checkpoint_path = config.get("phase5", {}).get("phase4_vlm_checkpoint_path", "")
    loaded_phase4_vlm = False
    if phase4_vlm_checkpoint_path:
        loaded_phase4_vlm = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        print(
            f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}"
            if loaded_phase4_vlm
            else f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}"
        )
    model.eval()

    controller, metadata = load_controller(args.controller_checkpoint, model.hidden_size, model.device)
    max_depth = int(args.max_depth if args.max_depth is not None else metadata.get("max_depth", 10))
    target_policy = args.target_policy or str(metadata.get("target_policy", "earliest_correct"))
    context = args.context or str(metadata.get("context", "global"))
    context_mode = normalize_context_mode(context)
    max_prompt_tokens = int(metadata.get("max_prompt_tokens", 10))
    image_size = args.image_size
    if image_size is None:
        image_size = int(metadata.get("image_size") or config.get("inference", {}).get("image_size", config.get("phase2", {}).get("image_size", 280)))

    fixed_rows, load_summary = load_fixed_think_rows(args.fixed_think_jsonl, args.fixed_think_glob)
    supervision, supervision_summary = build_latent_depth_supervision(
        fixed_rows,
        max_depth=max_depth,
        target_policy=target_policy,
    )
    if not supervision:
        raise ValueError("No supervision rows were built from the provided validation fixed-THINK files.")

    dataset = build_dataset(dataset_cfg, limit=args.limit, partition=args.dataset_partition)
    example_index = {str(example.get("id")): example for example in dataset}
    grouped = group_rows_by_example(supervision)
    depth_row_index = build_depth_row_index(fixed_rows)

    output_rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    binary_correct = 0
    binary_total = 0
    chosen_correct = 0
    chosen_total = 0
    oracle_correct = 0
    skipped_missing = 0
    predicted_depths: Counter[int] = Counter()
    target_depths: Counter[int] = Counter()

    with torch.no_grad():
        for example_id, rows_for_example in tqdm(sorted(grouped.items()), desc="Evaluating latent-depth controller"):
            source = example_index.get(str(example_id))
            if source is None:
                skipped_missing += 1
                continue
            rows_for_example = sorted(rows_for_example, key=lambda item: int(item["depth"]))
            features = extract_features_for_example(
                model,
                source,
                question=rows_for_example[0].get("question") or source["question"],
                image_size=image_size,
                context_mode=context_mode,
                max_depth=max_depth,
                max_prompt_tokens=max_prompt_tokens,
            )

            depth_predictions: List[Dict[str, Any]] = []
            first_predicted_stop: Optional[int] = None
            for row in rows_for_example:
                depth = int(row["depth"])
                visual = features["visual_token"].to(model.device).unsqueeze(0)
                prompt = features["prompt_tokens"].to(model.device).unsqueeze(0)
                latents = latent_tokens_for_depth(features, depth, model.hidden_size).to(model.device).unsqueeze(0)
                target = torch.tensor([float(row["target_stop"])], device=model.device)
                logit = controller(visual, prompt, latents)
                probability = float(torch.sigmoid(logit).item())
                predicted_stop = probability >= float(args.threshold)
                loss = F.binary_cross_entropy_with_logits(logit.float(), target.float())
                losses.append(float(loss.item()))
                binary_correct += int(predicted_stop == bool(row["target_stop"]))
                binary_total += 1
                if predicted_stop and first_predicted_stop is None:
                    first_predicted_stop = depth
                depth_predictions.append(
                    {
                        "depth": depth,
                        "stop_logit": float(logit.item()),
                        "stop_probability": probability,
                        "predicted_stop": bool(predicted_stop),
                        "target_stop": bool(row["target_stop"]),
                    }
                )

            correct_depths = [int(depth) for depth in rows_for_example[0]["correct_depths"]]
            predicted_depth = int(first_predicted_stop if first_predicted_stop is not None else rows_for_example[-1]["depth"])
            predicted_depths[predicted_depth] += 1
            target_depths[int(rows_for_example[0]["earliest_correct_depth"])] += 1
            oracle_correct += int(bool(correct_depths))
            selected_fixed_row = depth_row_index.get(str(example_id), {}).get(predicted_depth, {})
            selected_correct = bool(selected_fixed_row.get("correct", predicted_depth in correct_depths))
            chosen_correct += int(selected_correct)
            chosen_total += 1
            output_rows.append(
                {
                    "example_id": example_id,
                    "question": rows_for_example[0].get("question"),
                    "gold_answer": rows_for_example[0].get("gold_answer"),
                    "correct_depths": correct_depths,
                    "earliest_correct_depth": int(rows_for_example[0]["earliest_correct_depth"]),
                    "predicted_depth": predicted_depth,
                    "predicted_depth_correct": selected_correct,
                    "selected_generated_text": selected_fixed_row.get("generated_text"),
                    "selected_decoded_answer": selected_fixed_row.get("decoded_answer"),
                    "depth_predictions": depth_predictions,
                }
            )

    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)
    summary = {
        "controller_checkpoint": str(args.controller_checkpoint),
        "controller_metadata": metadata,
        "dataset_partition": args.dataset_partition,
        "num_dataset_examples": len(dataset),
        "num_evaluated_examples": chosen_total,
        "skipped_missing_examples": skipped_missing,
        "fixed_think_load_summary": load_summary,
        "supervision_summary": supervision_summary,
        "max_depth": max_depth,
        "target_policy": target_policy,
        "context": context,
        "context_mode": context_mode,
        "image_size": image_size,
        "threshold": float(args.threshold),
        "loaded_phase4_vlm": loaded_phase4_vlm,
        "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
        "metrics": {
            "binary_loss": sum(losses) / len(losses) if losses else None,
            "binary_accuracy": binary_correct / binary_total if binary_total else None,
            "binary_total": binary_total,
            "predicted_depth_accuracy": chosen_correct / chosen_total if chosen_total else None,
            "predicted_depth_correct": chosen_correct,
            "predicted_depth_total": chosen_total,
            "oracle_accuracy": oracle_correct / chosen_total if chosen_total else None,
            "predicted_depth_distribution": dict(predicted_depths),
            "target_earliest_depth_distribution": dict(target_depths),
        },
        "predictions_path": str(output_path),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote predictions to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Binary accuracy: {summary['metrics']['binary_accuracy']}")
    print(f"Predicted-depth accuracy: {summary['metrics']['predicted_depth_accuracy']}")


if __name__ == "__main__":
    main()
