"""Mine five-stage PATCH/THINK oracle traces with greedy (width-one) search."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.beam_oracle_mining import BeamOracleTraceMiner
from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides
from lvar_scripts.mine_phase2 import iter_dataset_indices, read_completed_example_ids, set_seed


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine greedy five-stage PATCH/THINK Phase 2 oracle traces.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-partition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-from-end", "--reverse", dest="start_from_end", action="store_true")
    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=None)
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    greedy_cfg = dict(config.get("phase2_greedy", {}))
    dataset_cfg = config["dataset"]
    seed = int(args.seed if args.seed is not None else greedy_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)
    limit = args.limit if args.limit is not None else greedy_cfg.get("limit", dataset_cfg.get("limit"))
    dataset_partition = args.dataset_partition or greedy_cfg.get("dataset_partition")
    dataset = build_dataset(dataset_cfg, limit=limit, partition=dataset_partition)
    output_path = Path(args.output or greedy_cfg.get("output_path", "outputs/phase2_m3cot_greedy_traces.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume = bool(greedy_cfg.get("resume", True)) if args.resume is None else bool(args.resume)
    completed_ids = read_completed_example_ids(output_path) if resume else set()
    indices = iter_dataset_indices(len(dataset), start_from_end=bool(args.start_from_end or greedy_cfg.get("start_from_end", False)))
    pending = [index for index in indices if dataset[index].get("id", index) not in completed_ids]

    model = QwenLVAR(config["model"])
    model.eval()
    # Width one preserves the same action space, CE objective, thresholding,
    # stages, and per-stage action cap as beam mining, but makes every choice
    # greedily from the currently best single trajectory.
    miner = BeamOracleTraceMiner(
        model=model,
        beam_width=1,
        max_steps=int(greedy_cfg.get("max_steps", 5)),
        max_actions_per_stage=int(greedy_cfg.get("max_actions_per_stage", 4)),
        patch_top_k=int(greedy_cfg.get("patch_top_k", 32)),
        ce_improvement_threshold=float(greedy_cfg.get("ce_improvement_threshold", 0.01)),
        rationale_ce_weight=float(greedy_cfg.get("rationale_ce_weight", 0.4)),
        image_size=greedy_cfg.get("image_size", 280),
    )
    mode = "a" if resume else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        for index in tqdm(pending, total=len(pending), desc="Mining Phase 2 greedy traces"):
            row = miner.mine_example(dataset[index])
            row["search_strategy"] = "greedy"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    summary = miner.get_summary()
    summary.update({
        "search_strategy": "greedy", "output_path": str(output_path), "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"), "dataset_partition": dataset_partition or dataset_cfg.get("split"),
        "num_examples": len(dataset), "num_mined_this_run": len(pending),
        "num_skipped_existing": len(indices) - len(pending), "resume": resume, "seed": seed,
    })
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote Phase 2 greedy traces to {output_path}")
    print(f"Wrote Phase 2 greedy summary to {summary_path}")


if __name__ == "__main__":
    main()
