import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

# Allow running as a script: `python scripts/train_grpo.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import (
    Phase5MetricTracker,
    clipped_grpo_loss,
    clipped_grpo_diagnostics,
    compute_phase5_reward,
    load_controller_checkpoint,
    load_vlm_lora_checkpoint,
    normalize_group_rewards as phase5_normalize_group_rewards,
    recompute_action_log_probs,
    rollout_phase5,
    save_phase5_checkpoint,
    set_phase5_trainable,
)
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import correctness_reward
from lvar.utils import ACTION_NAMES_NO_GLOBAL, add_model_loading_args, apply_model_loading_overrides

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised in environments without HF deps
    Accelerator = None


def load_config(config_path: str):
    """Load YAML config values shared across scripts."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set Python/Torch seeds for reproducible rollouts and optimization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """
    Group-wise reward normalization used by GRPO-style advantage estimation.

    This matches the "relative within prompt group" intuition while guarding
    against zero-variance groups.
    """
    return phase5_normalize_group_rewards(rewards, epsilon=1e-8)

def asymmetric_baseline_weight(
        baseline_score: float,
        rollout_score: float,
        improve_weight: float = 1.5,
        miss_weight: float = 1.0,
        already_correct_weight: float = 0.5,
        regression_weight: float = 1.5,
    ) -> float:
        baseline_correct = baseline_score > 0.5
        rollout_correct = rollout_score > 0.5

        if not baseline_correct and rollout_correct:
            return improve_weight

        if not baseline_correct and not rollout_correct:
            return miss_weight

        if baseline_correct and rollout_correct:
            return already_correct_weight

        # baseline correct, rollout wrong
        return regression_weight


def trainable_state_dict(model: torch.nn.Module) -> dict:
    """Return only trainable LVAR/controller-facing parameters, excluding frozen backbone weights."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_controller_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Save a controller-only checkpoint to the requested path."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model), checkpoint_path)
    print(f"Saved controller checkpoint to {checkpoint_path}")


def append_jsonl_row(path: Path, row: dict) -> None:
    """Append one JSON-serializable row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def compute_correctness_only_reward(correctness_score: float, rollout: dict) -> dict:
    """Return reward components for pure correctness RL."""
    actions = rollout.get("actions", [])
    think_count = sum(1 for action in actions if str(action.get("type", "")).upper() == "THINK")
    visual_count = len(rollout.get("selected_visual_actions") or [])
    return {
        "reward": float(correctness_score),
        "r_correct": float(correctness_score),
        "r_logp": 0.0,
        "r_cf": 0.0,
        "r_stop": 0.0,
        "r_visual": 0.0,
        "r_early_stop": 0.0,
        "r_think_once": 0.0,
        "think_count": float(think_count),
        "visual_count": float(visual_count),
        "num_steps": float(rollout.get("num_steps", len(actions))),
    }


def build_constant_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR schedule with optional linear warmup."""
    warmup_steps = int(max(0, round(float(total_steps) * float(warmup_ratio))))

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_grpo_policy_loss(advantages: torch.Tensor, rollout_outputs: list) -> torch.Tensor | None:
    """Build the policy-gradient loss from rollout action log-prob tensors."""
    loss_terms = []
    for advantage, rollout in zip(advantages, rollout_outputs):
        action_log_prob_sum = rollout.get("action_log_prob_sum")

        if action_log_prob_sum is None:
            continue

        action_loss = action_log_prob_sum / max(1, len(rollout["action_log_probs"]))
        loss_terms.append(-advantage.detach() * action_loss)

    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def compute_clipped_grpo_policy_loss(
    advantages: torch.Tensor,
    rollout_outputs: list,
    current_log_probs: list,
    clip_epsilon: float = 0.2,
) -> torch.Tensor | None:
    """Build the clipped Phase 5 GRPO loss."""
    return clipped_grpo_loss(advantages, rollout_outputs, current_log_probs, clip_epsilon=clip_epsilon)


def main() -> None:
    """Train Phase 5 controller refinement with clipped GRPO."""
    parser = argparse.ArgumentParser(description="Train Phase 5 GRPO controller refinement.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save intermediate epoch checkpoints every N epochs. Use 0 to disable epoch checkpoints.",
    )
    add_model_loading_args(parser)
    args = parser.parse_args()

    if Accelerator is None:
        raise ImportError("accelerate is required for train_grpo.py. Install the requirements first.")

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    train_cfg = config.get("phase5", config.get("train", {}))
    dataset_cfg = config["dataset"]
    if "controller_max_steps" in train_cfg:
        config["model"]["controller_max_steps"] = int(train_cfg["controller_max_steps"])
        config["model"]["max_steps"] = int(train_cfg["controller_max_steps"])
    phase3_cfg = config.get("phase3", {})
    phase3_v2_cfg = config.get("phase3_v2", {})
    phase3_v2_enabled = bool(phase3_cfg.get("phase3_v2", phase3_v2_cfg.get("enabled", False)))
    phase3_v2_removes_global = bool(phase3_v2_cfg.get("remove_global", phase3_cfg.get("remove_global", True)))
    if phase3_v2_enabled and phase3_v2_removes_global:
        config["model"]["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
    if "mask_immediate_repeats" in config.get("inference", {}):
        config["model"]["mask_immediate_repeats"] = bool(config["inference"]["mask_immediate_repeats"])
    if "temperature" in train_cfg or "rollout_temperature" in train_cfg:
        config["model"]["controller_temperature"] = float(
            train_cfg.get("temperature", train_cfg.get("rollout_temperature"))
        )
    dataset_partition = train_cfg.get("dataset_partition", "validation")
    split_seed = int(train_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(train_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    set_seed(int(train_cfg.get("seed", 42)))
    accelerator = Accelerator()
    model = QwenLVAR(config["model"]).to(accelerator.device)
    phase4_vlm_checkpoint_path = train_cfg.get("phase4_vlm_checkpoint_path", train_cfg.get("phase4_checkpoint_path"))
    controller_checkpoint_path = train_cfg.get("controller_checkpoint_path")
    loaded_phase4_vlm = False
    loaded_controller = False
    if phase4_vlm_checkpoint_path:
        loaded_phase4_vlm = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        if accelerator.is_local_main_process:
            print(
                f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}"
                if loaded_phase4_vlm
                else f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}"
            )
    if controller_checkpoint_path:
        loaded_controller = load_controller_checkpoint(model, controller_checkpoint_path)
        if accelerator.is_local_main_process:
            print(
                f"Loaded controller checkpoint: {controller_checkpoint_path}"
                if loaded_controller
                else f"Controller checkpoint not found: {controller_checkpoint_path}"
            )
    model.train()
    trainable_params = set_phase5_trainable(model)

    optimizer_name = str(train_cfg.get("optimizer", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported phase5.optimizer: {train_cfg.get('optimizer')}. Only AdamW is supported.")
    betas = train_cfg.get("betas", [0.9, 0.999])
    if len(betas) != 2:
        raise ValueError("phase5.betas must contain exactly two values.")
    learning_rate = float(train_cfg.get("learning_rate", train_cfg.get("controller_lr", 5e-5)))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(
        dataset_options,
        limit=train_cfg.get("max_examples", dataset_cfg.get("limit")),
        partition=dataset_partition,
    )
    example_index = {str(example.get("id")): example for example in dataset}

    output_dir = Path(train_cfg.get("output_dir", "outputs/grpo_phase5_m3cot"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(train_cfg.get("metrics_path", output_dir / "grpo_training_metrics.jsonl"))
    summary_path = Path(train_cfg.get("summary_path", output_dir / "grpo_training_summary.json"))
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    append_metrics = bool(train_cfg.get("append_metrics", False))
    if accelerator.is_local_main_process and metrics_path.exists() and not append_metrics:
        metrics_path.unlink()

    num_epochs = int(train_cfg.get("dataset_epochs", train_cfg.get("num_epochs", 1)))
    checkpoint_every = int(
        args.checkpoint_every
        if args.checkpoint_every is not None
        else train_cfg.get("checkpoint_every", 1)
    )
    if checkpoint_every < 0:
        raise ValueError("phase5.checkpoint_every / --checkpoint-every must be >= 0.")
    group_size = int(train_cfg.get("num_rollouts_per_prompt", train_cfg.get("group_size", 6)))
    grad_clip_norm = float(train_cfg.get("max_grad_norm", train_cfg.get("grad_clip_norm", 1.0)))
    log_every = int(train_cfg.get("log_every", 10))
    max_controller_steps = int(train_cfg.get("max_controller_steps", train_cfg.get("controller_max_steps", 20)))
    rollout_temperature = float(train_cfg.get("temperature", train_cfg.get("rollout_temperature", 1.5)))
    clip_epsilon = float(train_cfg.get("ppo_clip_range", train_cfg.get("clip_epsilon", 0.2)))
    advantage_epsilon = float(train_cfg.get("advantage_epsilon", 1e-6))
    update_epochs = int(train_cfg.get("update_epochs", 1))
    image_size = train_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    logp_weight = float(train_cfg.get("logp_weight", 0.2))
    counterfactual_weight = float(train_cfg.get("counterfactual_weight", 0.3))
    use_counterfactual_reward = bool(train_cfg.get("use_counterfactual_reward", True))
    cf_random_image_probability = float(train_cfg.get("cf_random_image_probability", 0.35))
    no_stop_penalty = float(train_cfg.get("no_stop_penalty", 0.2))
    think_once_bonus = float(train_cfg.get("think_once_bonus", 0.05))
    no_visual_penalty = float(train_cfg.get("no_visual_penalty", 0.3))
    early_stop_penalty = float(train_cfg.get("early_stop_penalty", 0.3))
    min_controller_actions_before_stop = int(train_cfg.get("min_controller_actions_before_stop", 2))
    min_visual_actions_before_stop = int(train_cfg.get("min_visual_actions_before_stop", 1))
    reward_mode = str(train_cfg.get("reward_mode", "shaped")).lower()
    if reward_mode not in {"shaped", "correctness_only"}:
        raise ValueError("phase5.reward_mode must be 'shaped' or 'correctness_only'.")
    use_baseline_advantage_weighting = bool(
        train_cfg.get("use_baseline_advantage_weighting", reward_mode != "correctness_only")
    )
    lr_schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
    if lr_schedule != "constant":
        raise ValueError("phase5.lr_schedule currently supports only 'constant'.")
    total_update_steps = max(1, num_epochs * len(dataset) * update_epochs)
    scheduler = build_constant_with_warmup_scheduler(
        optimizer,
        total_steps=total_update_steps,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
    )
    rng = random.Random(int(train_cfg.get("seed", 42)))
    metric_tracker = Phase5MetricTracker()

    global_step = 0
    prompt_step = 0
    skipped_zero_advantage = 0
    skipped_no_loss = 0

    for epoch in range(num_epochs):
        for example in dataset:
            prompt_step += 1
            # ------------------------------------------------------------
            # 1. Optionally compute no-latent baseline correctness once.
            # ------------------------------------------------------------
            baseline_score_float = 0.0
            if use_baseline_advantage_weighting:
                with torch.no_grad():
                    baseline_output = model.baseline_forward(
                        example["image"],
                        example["question"],
                    )
                    baseline_score = correctness_reward(
                        baseline_output["answer"],
                        example["gold_answer"],
                    )
                baseline_score_float = float(baseline_score)

            # ------------------------------------------------------------
            # 2. Sample grouped LVAR trajectories for the same prompt.
            # ------------------------------------------------------------
            rollout_outputs = []
            rewards = []
            rollout_scores = []
            reward_components = []

            for _ in range(group_size):
                rollout = rollout_phase5(
                    model,
                    example["image"],
                    example["question"],
                    max_controller_steps=max_controller_steps,
                    temperature=rollout_temperature,
                    image_size=image_size,
                    min_controller_actions_before_stop=min_controller_actions_before_stop,
                    min_visual_actions_before_stop=min_visual_actions_before_stop,
                )
                rollout_outputs.append(rollout)

                rollout_score = correctness_reward(
                    rollout["answer"],
                    example["gold_answer"],
                )
                rollout_scores.append(float(rollout_score))
                if reward_mode == "correctness_only":
                    components = compute_correctness_only_reward(float(rollout_score), rollout)
                else:
                    components = compute_phase5_reward(
                        model,
                        rollout,
                        example,
                        example_index,
                        correctness_score=float(rollout_score),
                        rng=rng,
                        logp_weight=logp_weight,
                        counterfactual_weight=counterfactual_weight,
                        use_counterfactual_reward=use_counterfactual_reward,
                        cf_random_image_probability=cf_random_image_probability,
                        no_stop_penalty=no_stop_penalty,
                        think_once_bonus=think_once_bonus,
                        no_visual_penalty=no_visual_penalty,
                        early_stop_penalty=early_stop_penalty,
                        min_controller_actions_before_stop=min_controller_actions_before_stop,
                        min_visual_actions_before_stop=min_visual_actions_before_stop,
                        image_size=image_size,
                    )
                reward_components.append(components)
                rewards.append(float(components["reward"]))
                metric_tracker.update(components)

            # ------------------------------------------------------------
            # 3. Convert rewards into group-normalized advantages.
            #    Do NOT subtract baseline here because it cancels under
            #    per-prompt group normalization.
            # ------------------------------------------------------------
            reward_tensor = torch.tensor(
                rewards,
                device=accelerator.device,
                dtype=torch.float32,
            )

            advantages = phase5_normalize_group_rewards(reward_tensor, epsilon=advantage_epsilon)

            if use_baseline_advantage_weighting:
                # Apply no-latent baseline as prompt-level advantage weight.
                weights = torch.tensor([
                    asymmetric_baseline_weight(
                        baseline_score=baseline_score_float,
                        rollout_score=float(r),
                        improve_weight=float(train_cfg.get("improve_weight", 1.5)),
                        miss_weight=float(train_cfg.get("miss_weight", 1.0)),
                        already_correct_weight=float(train_cfg.get("already_correct_weight", 0.5)),
                        regression_weight=float(train_cfg.get("regression_weight", 1.5)),
                    )
                    for r in rollout_scores
                    ],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                advantages = advantages * weights
            reward_mean = float(reward_tensor.mean().item())
            reward_std = float(reward_tensor.std(unbiased=False).item())
            reward_min = float(reward_tensor.min().item())
            reward_max = float(reward_tensor.max().item())
            correct_rollouts = int(sum(1 for score in rollout_scores if float(score) > 0.5))
            zero_advantage = float(advantages.abs().sum().detach().item()) == 0.0
            base_log_row = {
                "event": "prompt",
                "epoch": epoch + 1,
                "prompt_step": prompt_step,
                "global_step": global_step,
                "example_id": example.get("id"),
                "dataset_partition": dataset_partition,
                "reward_mode": reward_mode,
                "num_rollouts": group_size,
                "correct_rollouts": correct_rollouts,
                "incorrect_rollouts": group_size - correct_rollouts,
                "rollout_accuracy": correct_rollouts / max(1, group_size),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "zero_advantage": zero_advantage,
                "skipped_zero_advantage_total": skipped_zero_advantage,
                "skipped_no_loss_total": skipped_no_loss,
                "baseline_score": baseline_score_float,
                "adv_mean": float(advantages.detach().float().mean().item()),
                "adv_std": float(advantages.detach().float().std(unbiased=False).item()),
                "adv_abs_mean": float(advantages.detach().float().abs().mean().item()),
                "learning_rate": scheduler.get_last_lr()[0],
                "temperature": rollout_temperature,
                "clip_epsilon": clip_epsilon,
                "max_grad_norm": grad_clip_norm,
            }
            if float(advantages.abs().sum().detach().item()) == 0.0:
                skipped_zero_advantage += 1
                base_log_row["event"] = "skip_zero_advantage"
                base_log_row["skipped_zero_advantage_total"] = skipped_zero_advantage
                if accelerator.is_local_main_process:
                    append_jsonl_row(metrics_path, base_log_row)
                continue

            loss = None
            last_diagnostics = {
                "adv_mean": float(advantages.detach().float().mean().item()),
                "adv_std": float(advantages.detach().float().std(unbiased=False).item()),
                "adv_abs_mean": float(advantages.detach().float().abs().mean().item()),
                "mean_ratio": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
            }
            for _ in range(update_epochs):
                current_log_probs = [
                    recompute_action_log_probs(
                        model,
                        example["image"],
                        example["question"],
                        rollout["actions"],
                        temperature=rollout_temperature,
                        image_size=image_size,
                        min_controller_actions_before_stop=min_controller_actions_before_stop,
                        min_visual_actions_before_stop=min_visual_actions_before_stop,
                    )
                    for rollout in rollout_outputs
                ]
                loss = compute_clipped_grpo_policy_loss(
                    advantages,
                    rollout_outputs,
                    current_log_probs,
                    clip_epsilon=clip_epsilon,
                )
                if loss is None:
                    continue

                loss_value = float(loss.detach().item())
                last_diagnostics = clipped_grpo_diagnostics(
                    advantages,
                    rollout_outputs,
                    current_log_probs,
                    clip_epsilon=clip_epsilon,
                )

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                last_diagnostics["grad_norm"] = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                optimizer.step()
                scheduler.step()

            if loss is None:
                skipped_no_loss += 1
                base_log_row["event"] = "skip_no_loss"
                base_log_row["skipped_no_loss_total"] = skipped_no_loss
                if accelerator.is_local_main_process:
                    append_jsonl_row(metrics_path, base_log_row)
                continue

            del rollout_outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            global_step += 1
            update_log_row = {
                **base_log_row,
                "event": "update",
                "global_step": global_step,
                "loss": loss_value,
                "mean_ratio": last_diagnostics["mean_ratio"],
                "clip_fraction": last_diagnostics["clip_fraction"],
                "grad_norm": last_diagnostics["grad_norm"],
                "learning_rate": scheduler.get_last_lr()[0],
                "skipped_zero_advantage_total": skipped_zero_advantage,
                "skipped_no_loss_total": skipped_no_loss,
                "metrics": metric_tracker.summary(),
            }
            if accelerator.is_local_main_process:
                append_jsonl_row(metrics_path, update_log_row)

            if global_step % log_every == 0 and accelerator.is_local_main_process:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_value:.4f} "
                    f"reward_mean={float(reward_tensor.mean().item()):.4f} "
                    f"reward_std={float(reward_tensor.std(unbiased=False).item()):.4f} "
                    f"reward_mode={reward_mode} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    f"baseline_score={baseline_score_float:.1f} "
                    f"adv_mean={last_diagnostics['adv_mean']:.4f} "
                    f"adv_std={last_diagnostics['adv_std']:.4f} "
                    f"adv_abs_mean={last_diagnostics['adv_abs_mean']:.4f} "
                    f"mean_ratio={last_diagnostics['mean_ratio']:.4f} "
                    f"clip_fraction={last_diagnostics['clip_fraction']:.4f} "
                    f"grad_norm={last_diagnostics['grad_norm']:.4f} "
                    f"metrics={metric_tracker.summary()} "
                )
        should_checkpoint_epoch = checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0
        if should_checkpoint_epoch and accelerator.is_local_main_process:
            epoch_checkpoint_path = output_dir / f"phase5_controller_epoch_{epoch + 1}.pt"
            save_phase5_checkpoint(
                model,
                epoch_checkpoint_path,
                metadata={
                    "phase": "phase5_grpo",
                    "epoch": epoch + 1,
                    "loaded_phase4_vlm": loaded_phase4_vlm,
                    "loaded_controller": loaded_controller,
                    "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
                    "controller_checkpoint_path": controller_checkpoint_path,
                    "dataset_partition": dataset_partition,
                    "metrics": metric_tracker.summary(),
                    "checkpoint_every": checkpoint_every,
                    "metrics_path": str(metrics_path),
                    "summary_path": str(summary_path),
                    "skipped_zero_advantage": skipped_zero_advantage,
                    "skipped_no_loss": skipped_no_loss,
                },
            )

    # Save final weights from the main process only.
    if accelerator.is_local_main_process:
        checkpoint_path = output_dir / "phase5_controller.pt"
        save_phase5_checkpoint(
            model,
            checkpoint_path,
            metadata={
                "phase": "phase5_grpo",
                "loaded_phase4_vlm": loaded_phase4_vlm,
                "loaded_controller": loaded_controller,
                "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
                "controller_checkpoint_path": controller_checkpoint_path,
                "dataset_partition": dataset_partition,
                "metrics": metric_tracker.summary(),
                "checkpoint_every": checkpoint_every,
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path),
                "skipped_zero_advantage": skipped_zero_advantage,
                "skipped_no_loss": skipped_no_loss,
            },
        )
        write_json(
            summary_path,
            {
                "phase": "phase5_grpo",
                "dataset_partition": dataset_partition,
                "reward_mode": reward_mode,
                "num_epochs": num_epochs,
                "num_prompts_seen": prompt_step,
                "num_updates": global_step,
                "num_rollouts_per_prompt": group_size,
                "temperature": rollout_temperature,
                "learning_rate": learning_rate,
                "optimizer": train_cfg.get("optimizer", "AdamW"),
                "betas": [float(betas[0]), float(betas[1])],
                "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
                "lr_schedule": lr_schedule,
                "ppo_clip_range": clip_epsilon,
                "max_grad_norm": grad_clip_norm,
                "use_baseline_advantage_weighting": use_baseline_advantage_weighting,
                "skipped_zero_advantage": skipped_zero_advantage,
                "skipped_no_loss": skipped_no_loss,
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
                "phase4_vlm_checkpoint_path": phase4_vlm_checkpoint_path,
                "controller_checkpoint_path": controller_checkpoint_path,
                "metrics": metric_tracker.summary(),
            },
        )
        print(f"Wrote GRPO training metrics to {metrics_path}")
        print(f"Wrote GRPO training summary to {summary_path}")


if __name__ == "__main__":
    main()
