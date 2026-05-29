import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/train_counterfactual.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.controller_sft import build_example_index, set_seed
from lvar.counterfactual_training import (
    CounterfactualMetricTracker,
    load_counterfactual_pairs,
    phase4_parameter_groups,
    replay_counterfactual_pair_loss,
    save_phase4_checkpoint,
    set_phase4_trainable,
    validate_negative_type_probs,
)
from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides


def load_config(config_path: str):
    """Load YAML config values shared across scripts."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_load_controller_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> bool:
    """Load optional Phase 3 controller weights before Phase 4 training."""
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Phase 3 controller checkpoint not found, skipping: {path}")
        return False
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded controller checkpoint: {path}")
    print("Missing keys while loading controller checkpoint:", len(missing))
    print("Unexpected keys while loading controller checkpoint:", len(unexpected))
    return True


def main() -> None:
    """Train Phase 4 with visual counterfactual ranking."""
    parser = argparse.ArgumentParser(description="Train Phase 4 counterfactual ranking over mined traces.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--trace-jsonl", default=None, help="Override phase4.trace_path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit flattened counterfactual pairs.")
    parser.add_argument("--seed", type=int, default=None, help="Override phase4.seed.")
    parser.add_argument("--output-dir", default=None, help="Override phase4.output_dir.")
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    phase4_cfg = config.get("phase4", {})
    dataset_cfg = config["dataset"]
    model_cfg = apply_model_loading_overrides(config["model"], args)

    if "controller_max_steps" in phase4_cfg:
        model_cfg["controller_max_steps"] = int(phase4_cfg["controller_max_steps"])

    seed = int(args.seed if args.seed is not None else phase4_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)
    rng = random.Random(seed)

    trace_path = Path(args.trace_jsonl or phase4_cfg.get("trace_path", "outputs/phase2_m3cot_traces.jsonl"))
    output_dir = Path(args.output_dir or phase4_cfg.get("output_dir", "outputs/counterfactual_m3cot"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_limit = args.limit if args.limit is not None else phase4_cfg.get("max_pairs")
    pairs = load_counterfactual_pairs(trace_path, limit=pair_limit)
    if not pairs:
        raise ValueError(f"No counterfactual pairs found in {trace_path}.")

    dataset_partition = phase4_cfg.get("dataset_partition")
    dataset_limit = phase4_cfg.get("dataset_limit", dataset_cfg.get("limit"))
    dataset = build_dataset(dict(dataset_cfg), limit=dataset_limit, partition=dataset_partition)
    example_index = build_example_index(dataset)

    model = QwenLVAR(model_cfg)
    loaded_controller = maybe_load_controller_checkpoint(model, phase4_cfg.get("controller_checkpoint_path"))
    model.train()
    train_controller = bool(phase4_cfg.get("train_controller", False))
    trainable_params = set_phase4_trainable(model, train_controller=train_controller)
    if not trainable_params:
        raise ValueError("No trainable Phase 4 parameters were found.")
    parameter_groups = phase4_parameter_groups(model)
    controller_lr = float(phase4_cfg.get("controller_lr", phase4_cfg.get("learning_rate", 1e-4)))
    lora_lr = float(phase4_cfg.get("lora_lr", phase4_cfg.get("learning_rate", 2e-5)))
    optimizer_groups = []
    if parameter_groups["controller"]:
        optimizer_groups.append({"params": parameter_groups["controller"], "lr": controller_lr})
    if parameter_groups["lora"]:
        optimizer_groups.append({"params": parameter_groups["lora"], "lr": lora_lr})

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(phase4_cfg.get("weight_decay", 0.0)),
    )

    negative_type_probs = validate_negative_type_probs(
        phase4_cfg.get(
            "negative_type_probs",
            {"same_image_wrong": 0.5, "different_image_random": 0.35, "same_image_noisy": 0.15},
        )
    )
    num_epochs = int(phase4_cfg.get("num_epochs", 1))
    grad_clip_norm = float(phase4_cfg.get("grad_clip_norm", 1.0))
    log_every = int(phase4_cfg.get("log_every", 10))
    image_size = phase4_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    context_full_probability = float(phase4_cfg.get("context_full_probability", 0.2))
    rank_margin = float(phase4_cfg.get("rank_margin", 0.1))
    positive_ce_weight = float(phase4_cfg.get("positive_ce_weight", 0.2))
    rank_weight = float(phase4_cfg.get("rank_weight", 0.3))
    controller_loss_weight = float(phase4_cfg.get("controller_loss_weight", 1.0 if train_controller else 0.0))
    noise_scale = float(phase4_cfg.get("noise_scale", 0.5))

    tracker = CounterfactualMetricTracker()
    negative_bank_cache = {}
    global_step = 0
    skipped_missing = 0

    for epoch in range(num_epochs):
        epoch_trained_pairs = 0
        progress = tqdm(
            pairs,
            total=len(pairs),
            desc=f"Phase 4 counterfactual epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
        )
        for pair in progress:
            example_id = str(pair.get("example_id"))
            source_example = example_index.get(example_id)
            if source_example is None:
                skipped_missing += 1
                tracker.update({"skip_reason": "missing_source_example"})
                progress.set_postfix(trained=epoch_trained_pairs, skipped=skipped_missing)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = replay_counterfactual_pair_loss(
                model,
                pair,
                source_example,
                example_index,
                rng,
                negative_type_probs,
                context_full_probability=context_full_probability,
                image_size=image_size,
                rank_margin=rank_margin,
                positive_ce_weight=positive_ce_weight,
                rank_weight=rank_weight,
                controller_loss_weight=controller_loss_weight,
                noise_scale=noise_scale,
                negative_bank_cache=negative_bank_cache,
            )
            tracker.update(metrics)
            if loss is None:
                progress.set_postfix(
                    trained=epoch_trained_pairs,
                    skipped=skipped_missing,
                    skip=metrics.get("skip_reason", "pair_skipped"),
                )
                continue

            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()

            global_step += 1
            epoch_trained_pairs += 1
            summary = tracker.summary()["global"]
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                margin=f"{metrics['margin']:.4f}",
                mean_margin=f"{summary.get('margin', 0.0):.4f}",
                trained=epoch_trained_pairs,
                skipped=skipped_missing,
            )
            if log_every > 0 and global_step % log_every == 0:
                tqdm.write(
                    f"step={global_step} epoch={epoch + 1}/{num_epochs} "
                    f"loss={metrics['loss']:.4f} ce_pos={metrics['ce_pos']:.4f} "
                    f"ce_neg={metrics['ce_neg']:.4f} "
                    f"ce_pos_answer={metrics['ce_pos_answer']:.4f} margin={metrics['margin']:.4f} "
                    f"neg={metrics['negative_type']} context={metrics['context_mode']} "
                    f"mean_margin={summary.get('margin', 0.0):.4f}"
                )

        if tracker.count > 0:
            epoch_metadata = {
                "phase": "phase4_counterfactual",
                "epoch": epoch + 1,
                "trace_path": str(trace_path),
                "num_pairs": len(pairs),
                "trained_pairs": tracker.count,
                "trained_pairs_this_epoch": epoch_trained_pairs,
                "skipped_missing_examples": skipped_missing,
                "loaded_controller_checkpoint": loaded_controller,
                "negative_type_probs": negative_type_probs,
                "controller_lr": controller_lr,
                "lora_lr": lora_lr,
                "train_controller": train_controller,
                "context_full_probability": context_full_probability,
                "rank_margin": rank_margin,
                "positive_ce_weight": positive_ce_weight,
                "rank_weight": rank_weight,
                "controller_loss_weight": controller_loss_weight,
                "noise_scale": noise_scale,
                "summary": tracker.summary(),
                "seed": seed,
            }
            epoch_checkpoint_path = output_dir / f"counterfactual_epoch_{epoch + 1}.pt"
            save_phase4_checkpoint(model, epoch_checkpoint_path, metadata=epoch_metadata)
            tqdm.write(f"Saved Phase 4 epoch checkpoint to {epoch_checkpoint_path}")

    if tracker.count == 0:
        raise ValueError("No Phase 4 pairs were trained; inspect skip reasons in the trace/config.")

    metadata = {
        "phase": "phase4_counterfactual",
        "trace_path": str(trace_path),
        "num_pairs": len(pairs),
        "trained_pairs": tracker.count,
        "skipped_missing_examples": skipped_missing,
        "loaded_controller_checkpoint": loaded_controller,
        "negative_type_probs": negative_type_probs,
        "controller_lr": controller_lr,
        "lora_lr": lora_lr,
        "train_controller": train_controller,
        "context_full_probability": context_full_probability,
        "rank_margin": rank_margin,
        "positive_ce_weight": positive_ce_weight,
        "rank_weight": rank_weight,
        "controller_loss_weight": controller_loss_weight,
        "noise_scale": noise_scale,
        "summary": tracker.summary(),
        "seed": seed,
    }

    checkpoint_path = output_dir / "counterfactual.pt"
    save_phase4_checkpoint(model, checkpoint_path, metadata=metadata)
    with open(output_dir / "counterfactual_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved Phase 4 checkpoint to {checkpoint_path}")
    print(f"Saved Phase 4 summary to {output_dir / 'counterfactual_summary.json'}")


if __name__ == "__main__":
    main()
