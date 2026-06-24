import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/train_controller_sft.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.controller_sft import (
    PHASE3_V2_TYPE_LOSS_WEIGHTS,
    build_example_index,
    load_mined_trace_rows,
    replay_controller_sft_loss,
    save_controller_sft_checkpoint,
    set_controller_sft_trainable,
    set_seed,
)
from lvar.dataset import build_dataset
from lvar.grpo_training import load_vlm_lora_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import ACTION_NAMES_NO_GLOBAL, add_model_loading_args, apply_model_loading_overrides


def load_config(config_path: str):
    """Load YAML config values shared across scripts."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_weighted_means(totals, counts, means, mean_counts) -> None:
    """Accumulate weighted scalar means from per-example metric dictionaries."""
    for key, value in means.items():
        count = int(mean_counts.get(key, 0))
        if count <= 0:
            continue
        totals[key] += float(value) * count
        counts[key] += count


def finalize_weighted_means(totals, counts):
    """Return weighted means for JSON/logging."""
    return {key: totals[key] / max(1, counts[key]) for key in totals}


def main() -> None:
    """Train the Phase 3 controller from Phase 2 mined traces."""
    parser = argparse.ArgumentParser(description="Train the Phase 3 controller with SFT over mined traces.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--trace-jsonl", default=None, help="Override phase3.trace_path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit mined trace rows used for SFT.")
    parser.add_argument("--seed", type=int, default=None, help="Override phase3.seed.")
    parser.add_argument("--output-dir", default=None, help="Override phase3.output_dir.")
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    phase3_cfg = config.get("phase3", {})
    dataset_cfg = config["dataset"]
    model_cfg = apply_model_loading_overrides(config["model"], args)

    if "controller_max_steps" in phase3_cfg:
        model_cfg["controller_max_steps"] = int(phase3_cfg["controller_max_steps"])
    phase3_v2 = bool(phase3_cfg.get("phase3_v2", phase3_cfg.get("remove_global", False)))
    if phase3_v2:
        model_cfg["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
        model_cfg["mask_immediate_repeats"] = bool(phase3_cfg.get("mask_immediate_repeats", True))

    seed = int(args.seed if args.seed is not None else phase3_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)

    trace_path = Path(args.trace_jsonl or phase3_cfg.get("trace_path", "outputs/phase2_m3cot_traces.jsonl"))
    output_dir = Path(args.output_dir or phase3_cfg.get("output_dir", "outputs/controller_sft_m3cot"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_limit = args.limit if args.limit is not None else phase3_cfg.get("max_examples")
    rows = load_mined_trace_rows(trace_path, limit=trace_limit)
    if not rows:
        raise ValueError(f"No mined trace rows found in {trace_path}.")

    dataset_options = dict(dataset_cfg)
    dataset_partition = phase3_cfg.get("dataset_partition")
    dataset_limit = phase3_cfg.get("dataset_limit", dataset_cfg.get("limit"))
    dataset = build_dataset(dataset_options, limit=dataset_limit, partition=dataset_partition)
    example_index = build_example_index(dataset)

    model = QwenLVAR(model_cfg)
    phase4_vlm_checkpoint_path = phase3_cfg.get("phase4_vlm_checkpoint_path")
    loaded_phase4_vlm = False
    if phase4_vlm_checkpoint_path:
        loaded_phase4_vlm = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        print(
            f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}"
            if loaded_phase4_vlm
            else f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}"
        )
    model.train()
    trainable_params = set_controller_sft_trainable(model)
    if not trainable_params:
        raise ValueError("No trainable controller parameters were found.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(phase3_cfg.get("controller_lr", phase3_cfg.get("learning_rate", 1e-4))),
        weight_decay=float(phase3_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(phase3_cfg.get("num_epochs", 1))
    grad_clip_norm = float(phase3_cfg.get("grad_clip_norm", 1.0))
    log_every = int(phase3_cfg.get("log_every", 10))
    image_size = phase3_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    full_context_probability = float(phase3_cfg.get("full_context_probability", 0.1))
    if full_context_probability < 0.0 or full_context_probability > 1.0:
        raise ValueError("phase3.full_context_probability must be in [0, 1].")
    decision_block_normalized = bool(phase3_cfg.get("decision_block_normalized", phase3_v2))
    use_type_loss_weights = bool(phase3_cfg.get("use_type_loss_weights", phase3_v2))
    type_loss_weights = {}
    if use_type_loss_weights:
        type_loss_weights = dict(PHASE3_V2_TYPE_LOSS_WEIGHTS if phase3_v2 else {})
        type_loss_weights.update(phase3_cfg.get("type_loss_weights", {}) or {})
    visual_or_region_min_improvement = float(phase3_cfg.get("visual_or_region_min_improvement", 0.05))
    think_min_improvement = float(phase3_cfg.get("think_min_improvement", 0.03))
    max_decision_blocks = int(phase3_cfg.get("max_decision_blocks_per_example", 6))
    max_primitive_actions = int(phase3_cfg.get("max_primitive_actions_per_example", 8))
    no_op_stop_ce_threshold = float(phase3_cfg.get("no_op_stop_ce_threshold", 0.05))
    remove_global = bool(phase3_cfg.get("remove_global", phase3_v2))
    visual_block_dropout_p = float(phase3_cfg.get("visual_block_dropout_p", 0.0))
    multi_hot_patch_labels = bool(phase3_cfg.get("multi_hot_patch_labels", False))
    multi_hot_patch_target_mode = str(phase3_cfg.get("multi_hot_patch_target_mode", "binary"))
    multi_hot_patch_order_decay = float(phase3_cfg.get("multi_hot_patch_order_decay", 0.5))
    context_rng = random.Random(seed)

    global_step = 0
    skipped_missing = 0
    action_totals: Counter[str] = Counter()
    initial_visual_totals: Counter[str] = Counter()
    loss_component_totals: Counter[str] = Counter()
    loss_component_counts: Counter[str] = Counter()
    action_loss_totals: Counter[str] = Counter()
    action_loss_counts: Counter[str] = Counter()
    logit_stat_totals: Counter[str] = Counter()
    logit_stat_counts: Counter[str] = Counter()
    transform_totals: Counter[str] = Counter()
    skipped_block_totals: Counter[str] = Counter()
    multi_hot_totals: Counter[str] = Counter()
    loss_total = 0.0
    trained_examples = 0
    loss_history_path = output_dir / "controller_sft_losses.jsonl"
    # Line buffering preserves completed-step losses even if a long run is interrupted.
    loss_history_handle = open(loss_history_path, "w", encoding="utf-8", buffering=1)

    for epoch in range(num_epochs):
        progress = tqdm(
            rows,
            total=len(rows),
            desc=f"Phase 3 SFT epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
        )
        for row in progress:
            example_id = str(row.get("example_id"))
            source_example = example_index.get(example_id)
            if source_example is None:
                skipped_missing += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = replay_controller_sft_loss(
                model,
                row,
                source_example,
                image_size=image_size,
                full_context_probability=full_context_probability,
                rng=context_rng,
                decision_block_normalized=decision_block_normalized,
                type_loss_weights=type_loss_weights,
                phase3_v2=phase3_v2,
                visual_or_region_min_improvement=visual_or_region_min_improvement,
                think_min_improvement=think_min_improvement,
                max_decision_blocks_per_example=max_decision_blocks,
                max_primitive_actions_per_example=max_primitive_actions,
                no_op_stop_ce_threshold=no_op_stop_ce_threshold,
                remove_global=remove_global,
                visual_block_dropout_p=visual_block_dropout_p,
                multi_hot_patch_labels=multi_hot_patch_labels,
                multi_hot_patch_target_mode=multi_hot_patch_target_mode,
                multi_hot_patch_order_decay=multi_hot_patch_order_decay,
            )
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()

            global_step += 1
            trained_examples += 1
            loss_value = float(loss.detach().item())
            loss_total += loss_value
            loss_components = metrics.get("loss_components", {})
            loss_history_handle.write(
                json.dumps(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "example_id": example_id,
                        "total_loss": loss_value,
                        "action_loss": float(loss_components.get("type_loss", 0.0)),
                        "region_loss": float(loss_components.get("region_loss", 0.0)),
                        "patch_loss": float(loss_components.get("patch_loss", 0.0)),
                    }
                )
                + "\n"
            )
            action_totals.update(metrics["action_counts"])
            initial_visual_totals.update([metrics["initial_visual_mode"]])
            update_weighted_means(
                loss_component_totals,
                loss_component_counts,
                metrics.get("loss_components", {}),
                metrics.get("loss_component_counts", {}),
            )
            update_weighted_means(
                action_loss_totals,
                action_loss_counts,
                metrics.get("action_loss_means", {}),
                metrics.get("action_loss_counts", {}),
            )
            update_weighted_means(
                logit_stat_totals,
                logit_stat_counts,
                metrics.get("logit_stats", {}),
                metrics.get("logit_stat_counts", {}),
            )
            transform = metrics.get("transform", {}) or {}
            for key in ("candidate_blocks", "kept_blocks", "kept_non_stop_blocks", "kept_primitives", "converted_noop_to_stop"):
                transform_totals[key] += int(transform.get(key, 0))
            transform_totals["dropped_visual_blocks"] += int(metrics.get("dropped_visual_blocks", 0))
            skipped_block_totals.update(transform.get("skipped_blocks", {}) or {})
            multi_hot_totals["patch_blocks"] += int(metrics.get("multi_hot_patch_blocks", 0))
            multi_hot_totals["patch_indices"] += int(metrics.get("multi_hot_patch_indices", 0))
            mean_components = finalize_weighted_means(loss_component_totals, loss_component_counts)
            mean_loss = loss_total / max(1, trained_examples)
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                mean_loss=f"{mean_loss:.4f}",
                type=f"{mean_components.get('type_loss', 0.0):.4f}",
                patch=f"{mean_components.get('patch_loss', 0.0):.4f}",
                region=f"{mean_components.get('region_loss', 0.0):.4f}",
                targets=metrics["num_targets"],
                kept=transform.get("kept_blocks", 0),
                skipped=skipped_missing,
            )

            if log_every > 0 and global_step % log_every == 0:
                tqdm.write(
                    f"step={global_step} epoch={epoch + 1}/{num_epochs} "
                    f"loss={loss_value:.4f} mean_loss={mean_loss:.4f} "
                    f"type_loss={mean_components.get('type_loss', 0.0):.4f} "
                    f"patch_loss={mean_components.get('patch_loss', 0.0):.4f} "
                    f"region_loss={mean_components.get('region_loss', 0.0):.4f} "
                    f"blocks={metrics['num_targets']} primitives={metrics.get('num_primitive_targets', 0)} "
                    f"kept_blocks={transform.get('kept_blocks', 0)} "
                    f"noop_stop={transform.get('converted_noop_to_stop', 0)} "
                    f"initial_visual={metrics['initial_visual_mode']} example_id={example_id}"
                )

        if trained_examples > 0:
            epoch_metadata = {
                "phase": "phase3_controller_sft",
                "epoch": epoch + 1,
                "trace_path": str(trace_path),
                "num_trace_rows": len(rows),
                "trained_examples": trained_examples,
                "skipped_missing_examples": skipped_missing,
                "mean_loss": loss_total / max(1, trained_examples),
                "loss_components": finalize_weighted_means(loss_component_totals, loss_component_counts),
                "action_loss_means": finalize_weighted_means(action_loss_totals, action_loss_counts),
                "logit_stats": finalize_weighted_means(logit_stat_totals, logit_stat_counts),
                "action_counts": dict(action_totals),
                "initial_visual_counts": dict(initial_visual_totals),
                "full_context_probability": full_context_probability,
                "loaded_phase4_vlm": loaded_phase4_vlm,
                "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
                "phase3_v2": phase3_v2,
                "decision_block_normalized": decision_block_normalized,
                "use_type_loss_weights": use_type_loss_weights,
                "type_loss_weights": type_loss_weights,
                "loss_history_path": str(loss_history_path),
                "visual_block_dropout_p": visual_block_dropout_p,
                "multi_hot_patch_labels": multi_hot_patch_labels,
                "multi_hot_patch_target_mode": multi_hot_patch_target_mode,
                "multi_hot_patch_order_decay": multi_hot_patch_order_decay,
                "multi_hot_totals": dict(multi_hot_totals),
                "transform_totals": dict(transform_totals),
                "skipped_block_totals": dict(skipped_block_totals),
                "action_names": model.action_names,
                "controller_context_window": model.controller_num_states,
                "controller_max_steps": model.step_embedding.num_embeddings,
                "seed": seed,
            }
            epoch_checkpoint_path = output_dir / f"controller_sft_epoch_{epoch + 1}.pt"
            save_controller_sft_checkpoint(model, epoch_checkpoint_path, metadata=epoch_metadata)
            tqdm.write(f"Saved Phase 3 epoch checkpoint to {epoch_checkpoint_path}")

    if trained_examples == 0:
        loss_history_handle.close()
        raise ValueError(
            "No mined rows matched the source dataset by example_id; "
            "check phase3.dataset_partition/dataset_limit and the trace file."
        )

    metadata = {
        "phase": "phase3_controller_sft",
        "trace_path": str(trace_path),
        "num_trace_rows": len(rows),
        "trained_examples": trained_examples,
        "skipped_missing_examples": skipped_missing,
        "mean_loss": loss_total / max(1, trained_examples),
        "loss_components": finalize_weighted_means(loss_component_totals, loss_component_counts),
        "action_loss_means": finalize_weighted_means(action_loss_totals, action_loss_counts),
        "logit_stats": finalize_weighted_means(logit_stat_totals, logit_stat_counts),
        "action_counts": dict(action_totals),
        "initial_visual_counts": dict(initial_visual_totals),
        "full_context_probability": full_context_probability,
        "loaded_phase4_vlm": loaded_phase4_vlm,
        "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
        "phase3_v2": phase3_v2,
        "decision_block_normalized": decision_block_normalized,
        "use_type_loss_weights": use_type_loss_weights,
        "type_loss_weights": type_loss_weights,
        "loss_history_path": str(loss_history_path),
        "visual_block_dropout_p": visual_block_dropout_p,
        "multi_hot_patch_labels": multi_hot_patch_labels,
        "multi_hot_patch_target_mode": multi_hot_patch_target_mode,
        "multi_hot_patch_order_decay": multi_hot_patch_order_decay,
        "multi_hot_totals": dict(multi_hot_totals),
        "transform_totals": dict(transform_totals),
        "skipped_block_totals": dict(skipped_block_totals),
        "action_names": model.action_names,
        "controller_context_window": model.controller_num_states,
        "controller_max_steps": model.step_embedding.num_embeddings,
        "seed": seed,
    }

    checkpoint_path = output_dir / "controller_sft.pt"
    save_controller_sft_checkpoint(model, checkpoint_path, metadata=metadata)
    with open(output_dir / "controller_sft_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    loss_history_handle.close()
    print(f"Saved Phase 3 controller SFT checkpoint to {checkpoint_path}")
    print(f"Saved Phase 3 summary to {output_dir / 'controller_sft_summary.json'}")
    print(f"Saved Phase 3 loss history to {loss_history_path}")


if __name__ == "__main__":
    main()
