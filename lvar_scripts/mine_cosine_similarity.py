import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.cosine_mining import CosineSimilarityTraceMiner, summarize_cosine_trace_rows
from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides
from lvar_scripts.mine_phase2 import iter_dataset_indices, read_completed_example_ids


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl_row(handle, row) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def read_jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
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


def pending_dataset_indices(dataset, completed_ids, start_from_end: bool = False):
    indices = iter_dataset_indices(len(dataset), start_from_end=start_from_end)
    pending = [index for index in indices if dataset[index].get("id", index) not in completed_ids]
    return indices, pending


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine M3CoT patch traces using rationale-step cosine similarity.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--dataset-partition", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-from-end", "--reverse", dest="start_from_end", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip example ids already present in the output JSONL (default: enabled).",
    )
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    phase2_cfg = config.get("phase2", {})
    seed = int(args.seed if args.seed is not None else phase2_cfg.get("seed", 42))
    set_seed(seed)

    dataset = build_dataset(dataset_cfg, limit=args.limit, partition=args.dataset_partition)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = read_completed_example_ids(output_path) if args.resume else set()
    indices, pending_indices = pending_dataset_indices(
        dataset,
        completed_ids,
        start_from_end=args.start_from_end,
    )

    model = QwenLVAR(config["model"])
    model.eval()
    miner = CosineSimilarityTraceMiner(
        model,
        top_k=args.top_k,
        max_steps=args.max_steps,
        image_size=args.image_size if args.image_size is not None else phase2_cfg.get("image_size", 280),
    )

    file_mode = "a" if args.resume else "w"
    with open(output_path, file_mode, encoding="utf-8") as handle:
        for index in tqdm(pending_indices, total=len(pending_indices), desc="Mining cosine traces"):
            write_jsonl_row(handle, miner.mine_example(dataset[index]))

    summary = summarize_cosine_trace_rows(
        read_jsonl_rows(output_path),
        top_k=args.top_k,
        max_steps=args.max_steps,
    )
    summary.update(
        {
            "output_path": str(output_path),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_name": dataset_cfg.get("name"),
            "dataset_partition": args.dataset_partition,
            "num_dataset_examples": len(dataset),
            "num_mined_this_run": len(pending_indices),
            "num_skipped_existing": len(indices) - len(pending_indices),
            "resume": bool(args.resume),
            "start_from_end": bool(args.start_from_end),
            "seed": seed,
        }
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote cosine traces to {output_path}")
    print(f"Wrote cosine mining summary to {summary_path}")


if __name__ == "__main__":
    main()
