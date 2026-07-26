"""Mine rank-once fast oracle traces in greedy or beam-search mode."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.fast_oracle_mining import BeamSearchFastOracleTraceMiner, GreedyFastOracleTraceMiner
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
    parser = argparse.ArgumentParser(description="Mine rank-once fast Phase 2 oracle traces.")
    parser.add_argument("--strategy", choices=["greedy_fast", "beam_search_fast"], required=True)
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-partition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--patch-top-k",
        type=int,
        default=None,
        help="Override the configured attended-patch candidate count for this run.",
    )
    parser.add_argument("--start-from-end", "--reverse", dest="start_from_end", action="store_true")
    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=None)
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    config_key = "phase2_greedy_fast" if args.strategy == "greedy_fast" else "phase2_beam_search_fast"
    fast_cfg = dict(config.get(config_key, {}))
    dataset_cfg = config["dataset"]
    seed = int(args.seed if args.seed is not None else fast_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)
    limit = args.limit if args.limit is not None else fast_cfg.get("limit", dataset_cfg.get("limit"))
    dataset_partition = args.dataset_partition or fast_cfg.get("dataset_partition")
    dataset = build_dataset(dataset_cfg, limit=limit, partition=dataset_partition)
    default_output = f"outputs/phase2_m3cot_{args.strategy}_traces.jsonl"
    output_path = Path(args.output or fast_cfg.get("output_path", default_output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume = bool(fast_cfg.get("resume", True)) if args.resume is None else bool(args.resume)
    completed_ids = read_completed_example_ids(output_path) if resume else set()
    indices = iter_dataset_indices(len(dataset), start_from_end=bool(args.start_from_end or fast_cfg.get("start_from_end", False)))
    pending = [index for index in indices if dataset[index].get("id", index) not in completed_ids]

    model = QwenLVAR(config["model"])
    model.eval()
    common_kwargs = {
        "max_steps": int(fast_cfg.get("max_steps", 5)),
        "max_actions_per_stage": int(fast_cfg.get("max_actions_per_stage", 4)),
        "patch_top_k": int(args.patch_top_k if args.patch_top_k is not None else fast_cfg.get("patch_top_k", 32)),
        "proposal_count": int(fast_cfg.get("proposal_count", 6)),
        "ce_improvement_threshold": float(fast_cfg.get("ce_improvement_threshold", 0.01)),
        "rationale_ce_weight": float(fast_cfg.get("rationale_ce_weight", 0.4)),
        "image_size": fast_cfg.get("image_size", 280),
    }
    if args.strategy == "greedy_fast":
        miner = GreedyFastOracleTraceMiner(model=model, **common_kwargs)
    else:
        miner = BeamSearchFastOracleTraceMiner(
            model=model,
            beam_width=int(fast_cfg.get("beam_width", 3)),
            **common_kwargs,
        )

    mode = "a" if resume else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        for index in tqdm(pending, total=len(pending), desc=f"Mining Phase 2 {args.strategy} traces"):
            row = miner.mine_example(dataset[index])
            row["search_strategy"] = args.strategy
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    summary = miner.get_summary()
    summary.update(
        {
            "search_strategy": args.strategy,
            "output_path": str(output_path),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_name": dataset_cfg.get("name"),
            "dataset_partition": dataset_partition or dataset_cfg.get("split"),
            "num_examples": len(dataset),
            "num_mined_this_run": len(pending),
            "num_skipped_existing": len(indices) - len(pending),
            "resume": resume,
            "seed": seed,
        }
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote Phase 2 {args.strategy} traces to {output_path}")
    print(f"Wrote Phase 2 {args.strategy} summary to {summary_path}")


if __name__ == "__main__":
    main()
