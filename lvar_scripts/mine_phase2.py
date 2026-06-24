import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/mine_phase2.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.oracle_mining import OracleTraceMiner
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl_row(handle, row) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def collect_example_ids(dataset) -> list:
    ids = []
    for index in range(len(dataset)):
        ids.append(dataset[index].get("id", index))
    return ids


def read_completed_example_ids(path: Path) -> set:
    """Read existing JSONL output and return mined example ids."""
    if not path.exists():
        return set()
    completed = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            example_id = row.get("example_id")
            if example_id is not None:
                completed.add(example_id)
    return completed


def iter_dataset_indices(dataset_length: int, start_from_end: bool = False) -> list:
    """Return dataset indices in forward or reverse mining order."""
    if start_from_end:
        return list(range(dataset_length - 1, -1, -1))
    return list(range(dataset_length))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Phase 2 oracle traces from M3CoT.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-partition", default=None, help="Dataset partition/split to mine, e.g. train or test.")
    parser.add_argument(
        "--initial-visual-mode",
        default=None,
        choices=["global_mean", "full_context", "global", "coarse"],
        help="Initial image context used during mining. 'global_mean'/'coarse' uses one pooled visual token; 'full_context'/'global' uses full image tokens.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--start-from-end",
        "--reverse",
        dest="start_from_end",
        action="store_true",
        help="Mine examples from the end of the dataset toward the beginning.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Read the existing output JSONL and skip already mined example ids. Enabled by default.",
    )
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    phase2_cfg = config.get("phase2", {})
    dataset_cfg = config["dataset"]

    seed = int(args.seed if args.seed is not None else phase2_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)

    limit = args.limit if args.limit is not None else phase2_cfg.get("limit", dataset_cfg.get("limit"))
    dataset_partition = args.dataset_partition or phase2_cfg.get("dataset_partition")
    dataset = build_dataset(dataset_cfg, limit=limit, partition=dataset_partition)
    output_path = Path(args.output or phase2_cfg.get("output_path", "outputs/phase2_m3cot_traces.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume = bool(phase2_cfg.get("resume", True)) if args.resume is None else bool(args.resume)
    start_from_end = bool(args.start_from_end or phase2_cfg.get("start_from_end", False))
    completed_ids = read_completed_example_ids(output_path) if resume else set()

    model = QwenLVAR(config["model"])
    model.eval()
    miner = OracleTraceMiner(
        model=model,
        selection_delta=float(phase2_cfg.get("selection_delta", 0.03)),
        patch_k_choices=phase2_cfg.get("patch_k_choices", [1, 2, 3, 4]),
        max_steps=int(phase2_cfg.get("max_steps", config["model"].get("max_steps", 8))),
        rng=random.Random(seed),
        initial_visual_mode=str(args.initial_visual_mode or phase2_cfg.get("initial_visual_mode", "global_mean")),
        image_size=phase2_cfg.get("image_size", 280),
        counterfactual_negative_source=str(phase2_cfg.get("counterfactual_negative_source", "same_image")),
    )

    example_ids = collect_example_ids(dataset)
    indices = iter_dataset_indices(len(dataset), start_from_end=start_from_end)
    pending_indices = []
    skipped_existing = 0
    for index in indices:
        example = dataset[index]
        if example.get("id", index) in completed_ids:
            skipped_existing += 1
            continue
        pending_indices.append(index)

    file_mode = "a" if resume else "w"
    with open(output_path, file_mode, encoding="utf-8") as handle:
        for index in tqdm(pending_indices, total=len(pending_indices), desc="Mining Phase 2"):
            example = dataset[index]
            row = miner.mine_example(example, negative_global_example_ids=example_ids)
            write_jsonl_row(handle, row)

    summary = miner.get_summary()
    summary.update(
        {
            "output_path": str(output_path),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_name": dataset_cfg.get("name"),
            "dataset_partition": dataset_partition or dataset_cfg.get("split"),
            "num_examples": len(dataset),
            "num_mined_this_run": len(pending_indices),
            "num_skipped_existing": skipped_existing,
            "resume": resume,
            "start_from_end": start_from_end,
            "seed": seed,
        }
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote Phase 2 traces to {output_path}")
    print(f"Wrote Phase 2 summary to {summary_path}")


if __name__ == "__main__":
    main()
