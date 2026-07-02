import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_vlm_lora_checkpoint
from lvar.latent_depth import (
    BUCKET_PROMPT,
    build_latent_depth_supervision,
    label_initial_positions,
    load_fixed_think_rows,
)
from lvar.latent_depth_controller import LatentDepthController
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides


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


def normalize_context_mode(context: str) -> str:
    mode = str(context).strip().lower()
    if mode in {"global", "full", "full_image", "full_context"}:
        return "full_context"
    if mode in {"coarse", "coarse_context", "global_mean", "global_token"}:
        return "global_mean"
    raise ValueError("context must be one of: global, coarse, full_context, global_mean.")


def group_rows_by_example(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(row)
    return grouped


def split_example_ids(
    example_ids: List[str],
    validation_fraction: float,
    seed: int,
) -> Tuple[set[str], set[str]]:
    ids = list(example_ids)
    random.Random(seed).shuffle(ids)
    val_count = int(round(len(ids) * validation_fraction))
    val_ids = set(ids[:val_count])
    train_ids = set(ids[val_count:])
    if not train_ids and val_ids:
        moved = sorted(val_ids)[0]
        val_ids.remove(moved)
        train_ids.add(moved)
    return train_ids, val_ids


def apply_one_fixed_think_update(model: QwenLVAR, state: Dict[str, Any]) -> torch.Tensor:
    """Return the latent hidden vector used for the update, then mutate state."""
    last_hidden, state_hidden, act_hidden = model._read_current_hidden(state)
    latent_hidden = state_hidden if model.use_control_tokens and not model.think_append_hidden else last_hidden
    if model.think_append_hidden:
        model._append_hidden_token(state, last_hidden, track_as_think=True)
    elif model.use_control_tokens:
        state["inputs_embeds"] = model._write_recurrent_tokens(
            state["inputs_embeds"],
            state["latent_pos"],
            state["act_pos"],
            state_hidden,
            act_hidden,
        )
    else:
        updated_embeds = state["inputs_embeds"].clone()
        updated_embeds[:, -1, :] = last_hidden.to(updated_embeds.dtype)
        state["inputs_embeds"] = updated_embeds
    return latent_hidden.detach().squeeze(0).cpu()


def extract_features_for_example(
    model: QwenLVAR,
    source_example: Dict[str, Any],
    question: str,
    image_size: Optional[int],
    context_mode: str,
    max_depth: int,
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    with torch.no_grad():
        batch = model.prepare_inputs(
            source_example["image"],
            question,
            add_answer_instruction=False,
            image_size=image_size,
        )
        image_tokens = model.get_projected_image_tokens(batch)
        batch["projected_image_tokens"] = image_tokens
        bank = model.build_visual_bank(image_tokens)
        state = model.build_coarse_initial_state(batch, bank) if context_mode == "global_mean" else model.build_initial_state(batch)
        labels = label_initial_positions(model, batch, bank, context_mode=context_mode)
        if len(labels) != int(state["inputs_embeds"].size(1)):
            raise ValueError(
                f"Initial label length {len(labels)} does not match sequence length {state['inputs_embeds'].size(1)}."
            )
        prompt_indices = [idx for idx, label in enumerate(labels) if label == BUCKET_PROMPT]
        prompt_indices = prompt_indices[-max_prompt_tokens:]
        if not prompt_indices:
            raise ValueError("No prompt tokens were available for latent-depth controller features.")
        prompt_tokens = state["inputs_embeds"][0, prompt_indices, :].detach().cpu()
        visual_token = bank["global"][0].detach().cpu()
        latent_vectors: List[torch.Tensor] = []
        for _ in range(int(max_depth)):
            latent_vectors.append(apply_one_fixed_think_update(model, state))
    return {
        "visual_token": visual_token,
        "prompt_tokens": prompt_tokens,
        "latent_vectors": latent_vectors,
    }


def latent_tokens_for_depth(features: Dict[str, Any], depth: int, hidden_size: int) -> torch.Tensor:
    vectors = features["latent_vectors"][: int(depth)]
    if not vectors:
        return torch.zeros((0, hidden_size), dtype=features["visual_token"].dtype)
    return torch.stack(vectors, dim=0)


def controller_loss_for_row(
    controller: LatentDepthController,
    features: Dict[str, Any],
    row: Dict[str, Any],
    device: torch.device,
    hidden_size: int,
) -> torch.Tensor:
    visual = features["visual_token"].to(device).unsqueeze(0)
    prompt = features["prompt_tokens"].to(device).unsqueeze(0)
    latents = latent_tokens_for_depth(features, int(row["depth"]), hidden_size).to(device).unsqueeze(0)
    target = torch.tensor([float(row["target_stop"])], device=device)
    logit = controller(visual, prompt, latents)
    return F.binary_cross_entropy_with_logits(logit.float(), target.float())


@torch.no_grad()
def evaluate_controller(
    controller: LatentDepthController,
    model: QwenLVAR,
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    example_index: Dict[str, Dict[str, Any]],
    image_size: Optional[int],
    context_mode: str,
    max_depth: int,
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    controller.eval()
    device = next(controller.parameters()).device
    losses: List[float] = []
    correct = 0
    total = 0
    predicted_depths: Counter[int] = Counter()
    target_depths: Counter[int] = Counter()
    skipped_missing = 0
    for example_id, rows in grouped_rows.items():
        source = example_index.get(str(example_id))
        if source is None:
            skipped_missing += 1
            continue
        rows = sorted(rows, key=lambda item: int(item["depth"]))
        features = extract_features_for_example(
            model,
            source,
            question=rows[0].get("question") or source["question"],
            image_size=image_size,
            context_mode=context_mode,
            max_depth=max_depth,
            max_prompt_tokens=max_prompt_tokens,
        )
        first_predicted_stop: Optional[int] = None
        for row in rows:
            visual = features["visual_token"].to(device).unsqueeze(0)
            prompt = features["prompt_tokens"].to(device).unsqueeze(0)
            latents = latent_tokens_for_depth(features, int(row["depth"]), model.hidden_size).to(device).unsqueeze(0)
            target = torch.tensor([float(row["target_stop"])], device=device)
            logit = controller(visual, prompt, latents)
            loss = F.binary_cross_entropy_with_logits(logit.float(), target.float())
            probability = float(torch.sigmoid(logit).item())
            predicted_stop = probability >= 0.5
            correct += int(predicted_stop == bool(row["target_stop"]))
            total += 1
            losses.append(float(loss.item()))
            if predicted_stop and first_predicted_stop is None:
                first_predicted_stop = int(row["depth"])
        predicted_depths[int(first_predicted_stop if first_predicted_stop is not None else max_depth)] += 1
        target_depths[int(rows[0]["earliest_correct_depth"])] += 1
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "binary_accuracy": correct / total if total else None,
        "num_rows": total,
        "skipped_missing_examples": skipped_missing,
        "predicted_depth_distribution": dict(predicted_depths),
        "target_earliest_depth_distribution": dict(target_depths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary latent-depth STOP/CONTINUE controller.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--fixed-think-jsonl", action="append", default=[])
    parser.add_argument("--fixed-think-glob", action="append", default=[])
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--target-policy", choices=["earliest_correct", "all_correct"], default="earliest_correct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-partition", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--context", default="global", choices=["global", "coarse", "full_context", "global_mean"])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--phase4-vlm-checkpoint-path", default=None)
    parser.add_argument("--controller-hidden-size", type=int, default=512)
    parser.add_argument("--controller-layers", type=int, default=2)
    parser.add_argument("--controller-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    add_model_loading_args(parser)
    args = parser.parse_args()

    if not args.fixed_think_jsonl and not args.fixed_think_glob:
        raise ValueError("Provide at least one --fixed-think-jsonl or --fixed-think-glob.")
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")

    config = load_config(args.config)
    model_cfg = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = dict(config["dataset"])
    train_cfg = config.get("train", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows, load_summary = load_fixed_think_rows(args.fixed_think_jsonl, args.fixed_think_glob)
    supervision, supervision_summary = build_latent_depth_supervision(
        rows,
        max_depth=args.max_depth,
        target_policy=args.target_policy,
    )
    if not supervision:
        raise ValueError("No latent-depth supervision rows were built from the provided fixed-THINK files.")

    dataset = build_dataset(dataset_cfg, limit=args.limit, partition=args.dataset_partition)
    example_index = {str(example.get("id")): example for example in dataset}
    train_ids, val_ids = split_example_ids(
        sorted({str(row["example_id"]) for row in supervision}),
        validation_fraction=float(args.validation_fraction),
        seed=seed,
    )
    train_rows = [row for row in supervision if str(row["example_id"]) in train_ids]
    val_rows = [row for row in supervision if str(row["example_id"]) in val_ids]
    train_grouped = group_rows_by_example(train_rows)
    val_grouped = group_rows_by_example(val_rows)

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

    controller = LatentDepthController(
        input_hidden_size=model.hidden_size,
        controller_hidden_size=args.controller_hidden_size,
        num_layers=args.controller_layers,
        num_heads=args.controller_heads,
        dropout=args.dropout,
        max_prompt_tokens=args.max_prompt_tokens,
        max_latent_steps=args.max_depth,
    ).to(model.device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_history_path = output_dir / "latent_depth_controller_losses.jsonl"
    context_mode = normalize_context_mode(args.context)
    image_size = args.image_size
    if image_size is None:
        image_size = config.get("inference", {}).get("image_size", config.get("phase2", {}).get("image_size", 280))

    loss_rows: List[Dict[str, Any]] = []
    global_step = 0
    skipped_missing = 0
    for epoch in range(int(args.num_epochs)):
        controller.train()
        example_items = list(train_grouped.items())
        random.Random(seed + epoch).shuffle(example_items)
        progress = tqdm(example_items, desc=f"Latent-depth controller epoch {epoch + 1}/{args.num_epochs}")
        epoch_losses: List[float] = []
        for example_id, rows_for_example in progress:
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
                max_depth=args.max_depth,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            losses = [
                controller_loss_for_row(controller, features, row, model.device, model.hidden_size)
                for row in rows_for_example
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(args.grad_clip_norm))
            optimizer.step()

            global_step += 1
            loss_value = float(loss.detach().item())
            epoch_losses.append(loss_value)
            loss_row = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "example_id": example_id,
                "loss": loss_value,
                "num_targets": len(rows_for_example),
                "earliest_correct_depth": rows_for_example[0]["earliest_correct_depth"],
                "correct_depths": rows_for_example[0]["correct_depths"],
            }
            loss_rows.append(loss_row)
            if args.log_every > 0 and global_step % args.log_every == 0:
                tqdm.write(
                    f"step={global_step} epoch={epoch + 1} loss={loss_value:.4f} "
                    f"example_id={example_id}"
                )
            progress.set_postfix(loss=f"{loss_value:.4f}", mean=f"{sum(epoch_losses) / len(epoch_losses):.4f}")

    write_jsonl(loss_history_path, loss_rows)
    val_metrics = evaluate_controller(
        controller,
        model,
        val_grouped,
        example_index,
        image_size=image_size,
        context_mode=context_mode,
        max_depth=args.max_depth,
        max_prompt_tokens=args.max_prompt_tokens,
    ) if val_grouped else {}
    train_metrics = evaluate_controller(
        controller,
        model,
        train_grouped,
        example_index,
        image_size=image_size,
        context_mode=context_mode,
        max_depth=args.max_depth,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    checkpoint_path = output_dir / "latent_depth_controller.pt"
    metadata = {
        "phase": "latent_depth_binary_controller",
        "config": args.config,
        "max_depth": args.max_depth,
        "target_policy": args.target_policy,
        "context": args.context,
        "context_mode": context_mode,
        "image_size": image_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "hidden_size": model.hidden_size,
        "controller_hidden_size": args.controller_hidden_size,
        "controller_layers": args.controller_layers,
        "controller_heads": args.controller_heads,
        "seed": seed,
    }
    torch.save({"state_dict": controller.state_dict(), "metadata": metadata}, checkpoint_path)

    summary = {
        **metadata,
        "checkpoint_path": str(checkpoint_path),
        "loss_history_path": str(loss_history_path),
        "fixed_think_load_summary": load_summary,
        "supervision_summary": supervision_summary,
        "dataset_partition": args.dataset_partition,
        "num_dataset_examples": len(dataset),
        "num_train_examples": len(train_grouped),
        "num_validation_examples": len(val_grouped),
        "num_train_rows": len(train_rows),
        "num_validation_rows": len(val_rows),
        "skipped_missing_training_examples": skipped_missing,
        "loaded_phase4_vlm": loaded_phase4_vlm,
        "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    summary_path = output_dir / "latent_depth_controller_summary.json"
    write_json(summary_path, summary)

    print(f"Saved latent-depth controller to {checkpoint_path}")
    print(f"Wrote loss history to {loss_history_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
