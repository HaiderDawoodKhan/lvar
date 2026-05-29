import argparse
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
    compute_phase5_reward,
    load_trainable_checkpoint,
    normalize_group_rewards as phase5_normalize_group_rewards,
    recompute_action_log_probs,
    rollout_phase5,
    save_phase5_checkpoint,
    set_phase5_trainable,
)
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import correctness_reward
from lvar.utils import add_model_loading_args, apply_model_loading_overrides

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
    if "rollout_temperature" in train_cfg:
        config["model"]["controller_temperature"] = float(train_cfg["rollout_temperature"])
    dataset_partition = train_cfg.get("dataset_partition", "validation")
    split_seed = int(train_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(train_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    set_seed(int(train_cfg.get("seed", 42)))
    accelerator = Accelerator()
    model = QwenLVAR(config["model"]).to(accelerator.device)
    phase4_checkpoint_path = train_cfg.get("phase4_checkpoint_path")
    loaded_phase4 = False
    if phase4_checkpoint_path:
        loaded_phase4 = load_trainable_checkpoint(model, phase4_checkpoint_path)
        if accelerator.is_local_main_process:
            print(f"Loaded Phase 4 checkpoint: {phase4_checkpoint_path}" if loaded_phase4 else f"Phase 4 checkpoint not found: {phase4_checkpoint_path}")
    model.train()
    trainable_params = set_phase5_trainable(model)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("controller_lr", train_cfg.get("learning_rate", 5e-5))),
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

    num_epochs = int(train_cfg.get("num_epochs", 1))
    group_size = int(train_cfg.get("group_size", 6))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 10))
    max_controller_steps = int(train_cfg.get("max_controller_steps", train_cfg.get("controller_max_steps", 20)))
    rollout_temperature = float(train_cfg.get("rollout_temperature", 1.5))
    clip_epsilon = float(train_cfg.get("clip_epsilon", 0.2))
    advantage_epsilon = float(train_cfg.get("advantage_epsilon", 1e-6))
    update_epochs = int(train_cfg.get("update_epochs", 1))
    image_size = train_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    logp_weight = float(train_cfg.get("logp_weight", 0.2))
    counterfactual_weight = float(train_cfg.get("counterfactual_weight", 0.3))
    use_counterfactual_reward = bool(train_cfg.get("use_counterfactual_reward", True))
    cf_random_image_probability = float(train_cfg.get("cf_random_image_probability", 0.35))
    no_stop_penalty = float(train_cfg.get("no_stop_penalty", 0.2))
    rng = random.Random(int(train_cfg.get("seed", 42)))
    metric_tracker = Phase5MetricTracker()

    global_step = 0

    for epoch in range(num_epochs):
        for example in dataset:
            # ------------------------------------------------------------
            # 1. Compute no-latent baseline correctness once for this prompt.
            # ------------------------------------------------------------
            with torch.no_grad():
                baseline_output = model.baseline_forward(
                    example["image"],
                    example["question"],
                )
                baseline_score = correctness_reward(
                    baseline_output["answer"],
                    example["gold_answer"],
                )

            # Convert to a scalar Python float for prompt-level weighting.
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
                )
                rollout_outputs.append(rollout)

                rollout_score = correctness_reward(
                    rollout["answer"],
                    example["gold_answer"],
                )
                rollout_scores.append(float(rollout_score))
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

            # Apply no-latent baseline as prompt-level advantage weight.
            # If baseline is wrong, amplify the preference signal.
            # If baseline is correct, dampen the preference signal.
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
            if float(advantages.abs().sum().detach().item()) == 0.0:
                continue

            loss = None
            for _ in range(update_epochs):
                current_log_probs = [
                    recompute_action_log_probs(
                        model,
                        example["image"],
                        example["question"],
                        rollout["actions"],
                        temperature=rollout_temperature,
                        image_size=image_size,
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

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)

                optimizer.step()

            if loss is None:
                continue

            del rollout_outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            global_step += 1

            if global_step % log_every == 0 and accelerator.is_local_main_process:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_value:.4f} "
                    f"reward_mean={float(reward_tensor.mean().item()):.4f} "
                    f"reward_std={float(reward_tensor.std(unbiased=False).item()):.4f} "
                    f"baseline_score={baseline_score_float:.1f} "
                    f"metrics={metric_tracker.summary()} "
                )
        if accelerator.is_local_main_process:
            epoch_checkpoint_path = output_dir / f"phase5_controller_epoch_{epoch + 1}.pt"
            save_phase5_checkpoint(
                model,
                epoch_checkpoint_path,
                metadata={
                    "phase": "phase5_grpo",
                    "epoch": epoch + 1,
                    "loaded_phase4": loaded_phase4,
                    "dataset_partition": dataset_partition,
                    "metrics": metric_tracker.summary(),
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
                "loaded_phase4": loaded_phase4,
                "dataset_partition": dataset_partition,
                "metrics": metric_tracker.summary(),
            },
        )


if __name__ == "__main__":
    main()
