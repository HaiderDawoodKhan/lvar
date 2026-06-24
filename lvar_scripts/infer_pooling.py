import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/infer_pooling.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import correctness_reward
from lvar.utils import add_model_loading_args, apply_model_loading_overrides


def load_config(config_path: str):
    """Load YAML config file used to parameterize dataset/model/inference paths."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(path: Path, rows):
    """Persist prediction rows as JSONL so each example is one appendable line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    """Persist a JSON object with stable indentation for readability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _slugify(text: str) -> str:
    """Convert dataset identifiers into filename-safe lowercase tokens."""
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    return cleaned.strip("_")


def build_dataset_tag(dataset_cfg) -> str:
    """Build a compact dataset tag used to disambiguate output artifact names."""
    dataset_type = str(dataset_cfg.get("type", "dataset"))
    dataset_name = str(dataset_cfg.get("name", "dataset"))
    tail_name = dataset_name.rsplit("/", 1)[-1]
    tail_tag = _slugify(tail_name)
    type_tag = _slugify(dataset_type)
    if tail_tag and tail_tag != "dataset":
        return tail_tag
    return type_tag or "dataset"


def resolve_output_path(cli_output: str, inference_cfg, dataset_cfg) -> Path:
    """Choose output path and inject dataset tag into filename when needed."""
    dataset_tag = build_dataset_tag(dataset_cfg)
    if cli_output:
        return Path(cli_output)

    configured_output = Path(inference_cfg.get("output_path", "outputs/predictions.jsonl"))
    if dataset_tag in configured_output.stem:
        return configured_output
    return configured_output.with_name(f"{configured_output.stem}_{dataset_tag}{configured_output.suffix}")


def build_accuracy_summary(totals, count: int, dataset_cfg, partition: str):
    """Build a serializable summary payload for metrics sidecar artifacts."""
    def metric(correct_key: str):
        correct = int(totals[correct_key])
        return {
            "correct": correct,
            "total": int(count),
            "accuracy": float(correct / count) if count else 0.0,
        }

    return {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"),
        "dataset_partition": partition,
        "num_examples": int(count),
        "metrics": {
            "full_image": metric("full_image_correct"),
            "mean_pooled": metric("mean_pooled_correct"),
            "max_pooled": metric("max_pooled_correct"),
            "region_mean_pooled": metric("region_mean_pooled_correct"),
            "region_max_pooled": metric("region_max_pooled_correct"),
        },
    }

def main() -> None:
    """Run inference comparing full-image, pooled-image, and region-token baselines."""
    # Allow script-level limit/output overrides while defaulting to config values.
    parser = argparse.ArgumentParser(
        description="Compare accuracy for full-image, pooled-image, and region-token inputs."
    )
    parser.add_argument("--config", default="configs/qwen2vl_clevr.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})
    dataset_partition = inference_cfg.get("dataset_partition", dataset_cfg.get("eval_partition", "test"))
    split_seed = int(inference_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(inference_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    dataset_limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(dataset_options, limit=dataset_limit, partition=dataset_partition)
    model = QwenLVAR(config["model"])

    # Evaluate one example at a time (v1 design) and collect metrics for JSONL output.
    rows = []
    totals = {
        "full_image_correct": 0,
        "mean_pooled_correct": 0,
        "max_pooled_correct": 0,
        "region_mean_pooled_correct": 0,
        "region_max_pooled_correct": 0,
    }
    for example in tqdm(dataset, total=len(dataset), desc="Inferring"):
        full_image_output = model.generate_baseline(example["image"], example["question"])
        mean_pooled_output = model.generate_pooled_baseline(example["image"], example["question"], pooling="mean")
        max_pooled_output = model.generate_pooled_baseline(example["image"], example["question"], pooling="max")
        region_mean_output = model.generate_region_baseline(example["image"], example["question"], pooling="mean")
        region_max_output = model.generate_region_baseline(example["image"], example["question"], pooling="max")

        full_image_correct = correctness_reward(full_image_output["prediction"], example["gold_answer"])
        mean_pooled_correct = correctness_reward(mean_pooled_output["prediction"], example["gold_answer"])
        max_pooled_correct = correctness_reward(max_pooled_output["prediction"], example["gold_answer"])
        region_mean_correct = correctness_reward(region_mean_output["prediction"], example["gold_answer"])
        region_max_correct = correctness_reward(region_max_output["prediction"], example["gold_answer"])
        totals["full_image_correct"] += int(full_image_correct)
        totals["mean_pooled_correct"] += int(mean_pooled_correct)
        totals["max_pooled_correct"] += int(max_pooled_correct)
        totals["region_mean_pooled_correct"] += int(region_mean_correct)
        totals["region_max_pooled_correct"] += int(region_max_correct)
        row = {
            "example_id": example["id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "full_image_prediction": full_image_output["prediction"],
            "mean_pooled_prediction": mean_pooled_output["prediction"],
            "max_pooled_prediction": max_pooled_output["prediction"],
            "region_mean_pooled_prediction": region_mean_output["prediction"],
            "region_max_pooled_prediction": region_max_output["prediction"],
            "full_image_correct": bool(full_image_correct),
            "mean_pooled_correct": bool(mean_pooled_correct),
            "max_pooled_correct": bool(max_pooled_correct),
            "region_mean_pooled_correct": bool(region_mean_correct),
            "region_max_pooled_correct": bool(region_max_correct),
            "full_image_generated_text": full_image_output["generated_text"],
            "mean_pooled_generated_text": mean_pooled_output["generated_text"],
            "max_pooled_generated_text": max_pooled_output["generated_text"],
            "region_mean_pooled_generated_text": region_mean_output["generated_text"],
            "region_max_pooled_generated_text": region_max_output["generated_text"],
            "num_region_tokens": region_mean_output["num_region_tokens"],
        }
        rows.append(row)

    # Write all rows once at the end for a clean single output artifact.
    output_path = resolve_output_path(args.output, inference_cfg, dataset_cfg)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")

    summary = build_accuracy_summary(
        totals,
        count=len(rows),
        dataset_cfg=dataset_cfg,
        partition=dataset_partition,
    )
    summary_json_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_json_path, summary)
    print(f"Wrote accuracy summary JSON to {summary_json_path}")

    if rows:
        count = len(rows)
        print("Accuracy summary:")
        print(f"  full_image: {totals['full_image_correct'] / count:.4f} ({totals['full_image_correct']}/{count})")
        print(f"  mean_pooled: {totals['mean_pooled_correct'] / count:.4f} ({totals['mean_pooled_correct']}/{count})")
        print(f"  max_pooled: {totals['max_pooled_correct'] / count:.4f} ({totals['max_pooled_correct']}/{count})")
        print(
            "  region_mean_pooled: "
            f"{totals['region_mean_pooled_correct'] / count:.4f} "
            f"({totals['region_mean_pooled_correct']}/{count})"
        )
        print(
            "  region_max_pooled: "
            f"{totals['region_max_pooled_correct'] / count:.4f} "
            f"({totals['region_max_pooled_correct']}/{count})"
        )


if __name__ == "__main__":
    main()
