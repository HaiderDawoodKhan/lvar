import copy
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from lvar.counterfactual_training import (
    apply_counterfactual_actions,
    build_negative_actions,
    differentiable_state_ce,
)
from lvar.utils import (
    ACTION_GLOBAL,
    ACTION_NAMES,
    ACTION_PATCH,
    ACTION_REGION,
    ACTION_STOP,
    ACTION_THINK,
    normalize_action_names,
)


VISUAL_ACTIONS = {"GLOBAL", "REGION", "PATCH"}


def normalize_group_rewards(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Group-normalize rewards, returning zeros if the group has no preference signal."""
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std(unbiased=False)
    if float(std.detach().item()) < float(epsilon):
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + float(epsilon))


def apply_asymmetric_advantage_weights(
    advantages: torch.Tensor,
    baseline_score: float,
    rollout_scores: Sequence[float],
    weight_fn,
    **weight_kwargs,
) -> torch.Tensor:
    """Apply baseline-vs-rollout asymmetric scaling after group normalization."""
    weights = torch.tensor(
        [
            weight_fn(
                baseline_score=baseline_score,
                rollout_score=float(score),
                **weight_kwargs,
            )
            for score in rollout_scores
        ],
        device=advantages.device,
        dtype=advantages.dtype,
    )
    return advantages * weights


def set_phase5_trainable(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Freeze VLM/LoRA and train only controller-facing parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_modules = [model.controller, model.step_embedding]
    controller_state_norm = getattr(model, "controller_state_norm", None)
    if controller_state_norm is not None:
        trainable_modules.append(controller_state_norm)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    if hasattr(model, "backbone"):
        model.backbone.eval()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def phase5_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return trainable Phase 5 controller-facing parameters."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_phase5_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save a Phase 5 controller checkpoint."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": phase5_state_dict(model), "metadata": metadata or {}}, path)


def load_trainable_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> bool:
    """Load Phase 4/5 trainable-state checkpoint if present."""
    path = Path(checkpoint_path)
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    checkpoint_actions = metadata.get("action_names")
    model_actions = getattr(model, "action_names", ACTION_NAMES)
    if checkpoint_actions is not None:
        checkpoint_actions = normalize_action_names(checkpoint_actions)
        model_actions = normalize_action_names(model_actions)
        if checkpoint_actions != model_actions:
            raise ValueError(
                f"Checkpoint action_names {checkpoint_actions} do not match model action_names {model_actions}."
            )
    model.load_state_dict(state_dict, strict=False)
    return True


def _load_checkpoint_payload(checkpoint_path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"state_dict": payload, "metadata": {}}


def _check_action_names(model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
    checkpoint_actions = metadata.get("action_names")
    if checkpoint_actions is None:
        return
    checkpoint_actions = normalize_action_names(checkpoint_actions)
    model_actions = normalize_action_names(getattr(model, "action_names", ACTION_NAMES))
    if checkpoint_actions != model_actions:
        raise ValueError(
            f"Checkpoint action_names {checkpoint_actions} do not match model action_names {model_actions}."
        )


def load_vlm_lora_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> bool:
    """Load only VLM LoRA weights from a Phase 4 checkpoint."""
    payload = _load_checkpoint_payload(checkpoint_path)
    if payload is None:
        return False
    state_dict = payload.get("state_dict", payload)
    lora_state = {name: tensor for name, tensor in state_dict.items() if "lora_" in name}
    if not lora_state:
        raise ValueError(f"No LoRA weights found in VLM checkpoint: {checkpoint_path}")
    model.load_state_dict(lora_state, strict=False)
    return True


def load_controller_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> bool:
    """Load only controller-facing weights from a controller checkpoint."""
    payload = _load_checkpoint_payload(checkpoint_path)
    if payload is None:
        return False
    metadata = payload.get("metadata", {})
    _check_action_names(model, metadata)
    state_dict = payload.get("state_dict", payload)
    controller_prefixes = ("controller.", "step_embedding.", "controller_state_norm.")
    controller_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name.startswith(controller_prefixes)
    }
    if not controller_state:
        raise ValueError(f"No controller weights found in controller checkpoint: {checkpoint_path}")
    model.load_state_dict(controller_state, strict=False)
    return True


def _scaled_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-8)


def _masked_patch_logits(patch_logits: torch.Tensor, selected_patches: set[int]) -> torch.Tensor:
    if not selected_patches:
        return patch_logits
    masked = patch_logits.clone()
    valid_indices = [idx for idx in selected_patches if 0 <= idx < masked.size(-1)]
    if len(valid_indices) >= masked.size(-1):
        return patch_logits
    if valid_indices:
        masked[:, valid_indices] = torch.finfo(masked.dtype).min
    return masked


def _masked_stop_logits(
    type_logits: torch.Tensor,
    model: torch.nn.Module,
    step_idx: int,
    selected_visual_count: int,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
) -> torch.Tensor:
    """Mask STOP until the rollout has enough actions/visual evidence."""
    should_mask_stop = (
        step_idx < int(min_controller_actions_before_stop)
        or selected_visual_count < int(min_visual_actions_before_stop)
    )
    if not should_mask_stop:
        return type_logits
    action_name_to_id = getattr(model, "action_name_to_id", {name: idx for idx, name in ACTION_NAMES.items()})
    stop_id = action_name_to_id.get("STOP")
    if stop_id is None or type_logits.size(-1) <= 1:
        return type_logits
    masked = type_logits.clone()
    masked[:, int(stop_id)] = torch.finfo(masked.dtype).min
    return masked


def _sample_from_logits(logits: torch.Tensor, sample: bool = True) -> Tuple[int, torch.Tensor]:
    distribution = Categorical(logits=logits)
    tensor = distribution.sample() if sample else torch.argmax(logits, dim=-1)
    return int(tensor.item()), distribution.log_prob(tensor).squeeze(0)


def _action_from_selection(
    action_id: int,
    action_names: Optional[Dict[int, str]] = None,
    region_idx: Optional[int] = None,
    patch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    names = action_names or ACTION_NAMES
    action_name = names[action_id]
    action = {"type": action_name}
    if action_name == "REGION":
        action["region_idx"] = int(region_idx)
    elif action_name == "PATCH":
        action["patch_idx"] = int(patch_idx)
    return action


def prepare_full_context_state_and_bank(
    model: torch.nn.Module,
    image: Any,
    question: str,
    image_size: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Build Phase 5 full-image initial state and visual bank."""
    with torch.no_grad():
        batch = model.prepare_inputs(image, question, image_size=image_size)
        image_tokens = model.get_projected_image_tokens(batch)
        batch["projected_image_tokens"] = image_tokens
        bank = model.build_visual_bank(image_tokens)
        state = model.build_initial_state(batch)
    return state, bank


def select_controller_action(
    model: torch.nn.Module,
    state: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    step_idx: int,
    temperature: float,
    selected_patches: set[int],
    selected_visual_count: int = 0,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
    sample: bool = True,
) -> Tuple[Dict[str, Any], torch.Tensor, Dict[str, torch.Tensor]]:
    """Select one controller action with optional patch masking."""
    type_logits, region_logits, patch_logits = model.controller_logits_from_state(state, bank, step_idx)
    type_logits = _masked_stop_logits(
        type_logits,
        model,
        step_idx,
        selected_visual_count,
        min_controller_actions_before_stop=min_controller_actions_before_stop,
        min_visual_actions_before_stop=min_visual_actions_before_stop,
    )
    scaled_type = _scaled_logits(type_logits, temperature)
    scaled_region = _scaled_logits(region_logits, temperature)
    scaled_patch = _scaled_logits(_masked_patch_logits(patch_logits, selected_patches), temperature)
    action_id, log_prob = _sample_from_logits(scaled_type, sample=sample)
    action_name = getattr(model, "action_names", ACTION_NAMES)[action_id]
    region_idx = None
    patch_idx = None
    if action_name == "REGION":
        region_idx, region_log_prob = _sample_from_logits(scaled_region, sample=sample)
        log_prob = log_prob + region_log_prob
    elif action_name == "PATCH":
        patch_idx, patch_log_prob = _sample_from_logits(scaled_patch, sample=sample)
        log_prob = log_prob + patch_log_prob
    return _action_from_selection(action_id, getattr(model, "action_names", ACTION_NAMES), region_idx, patch_idx), log_prob, {
        "type_logits": scaled_type,
        "region_logits": scaled_region,
        "patch_logits": scaled_patch,
    }


def rollout_phase5(
    model: torch.nn.Module,
    image: Any,
    question: str,
    max_controller_steps: int = 20,
    temperature: float = 1.5,
    image_size: Optional[int] = None,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
) -> Dict[str, Any]:
    """Collect one stochastic Phase 5 rollout from full image context."""
    state, bank = prepare_full_context_state_and_bank(model, image, question, image_size=image_size)
    actions: List[Dict[str, Any]] = []
    old_log_probs: List[torch.Tensor] = []
    selected_patches: set[int] = set()
    selected_visual_actions: List[Dict[str, Any]] = []
    selected_visual_count = 0
    stopped = False

    for step_idx in range(max_controller_steps):
        with torch.no_grad():
            action, log_prob, _ = select_controller_action(
                model,
                state,
                bank,
                step_idx,
                temperature,
                selected_patches,
                selected_visual_count=selected_visual_count,
                min_controller_actions_before_stop=min_controller_actions_before_stop,
                min_visual_actions_before_stop=min_visual_actions_before_stop,
                sample=True,
            )
        action_type = action["type"]
        actions.append(action)
        old_log_probs.append(log_prob.detach())
        if action_type == "PATCH":
            selected_patches.add(int(action["patch_idx"]))
            selected_visual_actions.append(copy.deepcopy(action))
            selected_visual_count += 1
        elif action_type in {"REGION", "GLOBAL"}:
            selected_visual_actions.append(copy.deepcopy(action))
            selected_visual_count += 1
        if action_type == "STOP":
            stopped = True
            break
        with torch.no_grad():
            model.apply_mined_actions(state, bank, [action])

    decode_state = state
    if getattr(model, "use_control_tokens", False):
        decode_state = model.drop_act_token(model.clone_state(state))
    with torch.no_grad():
        decoded = model.decode_answer(model._build_decode_state(decode_state))

    return {
        "actions": actions,
        "old_log_probs": old_log_probs,
        "stopped": stopped,
        "final_state": state,
        "bank": bank,
        "answer": decoded["answer"],
        "generated_text": decoded["generated_text"],
        "selected_visual_actions": selected_visual_actions,
        "num_steps": len(actions),
    }


def action_log_prob_for_replay(
    model: torch.nn.Module,
    state: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    action: Dict[str, Any],
    step_idx: int,
    temperature: float,
    selected_patches: set[int],
    selected_visual_count: int = 0,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
) -> torch.Tensor:
    """Compute current log-prob for a stored action under replay state."""
    type_logits, region_logits, patch_logits = model.controller_logits_from_state(state, bank, step_idx)
    type_logits = _masked_stop_logits(
        type_logits,
        model,
        step_idx,
        selected_visual_count,
        min_controller_actions_before_stop=min_controller_actions_before_stop,
        min_visual_actions_before_stop=min_visual_actions_before_stop,
    )
    scaled_type = _scaled_logits(type_logits, temperature)
    action_name_to_id = getattr(model, "action_name_to_id", {name: idx for idx, name in ACTION_NAMES.items()})
    action_id = action_name_to_id[str(action["type"]).upper()]
    action_tensor = torch.tensor([action_id], device=scaled_type.device, dtype=torch.long)
    log_prob = Categorical(logits=scaled_type).log_prob(action_tensor).squeeze(0)
    action_name = str(action["type"]).upper()
    if action_name == "REGION":
        scaled_region = _scaled_logits(region_logits, temperature)
        region_tensor = torch.tensor([int(action["region_idx"])], device=scaled_region.device, dtype=torch.long)
        log_prob = log_prob + Categorical(logits=scaled_region).log_prob(region_tensor).squeeze(0)
    elif action_name == "PATCH":
        scaled_patch = _scaled_logits(_masked_patch_logits(patch_logits, selected_patches), temperature)
        patch_tensor = torch.tensor([int(action["patch_idx"])], device=scaled_patch.device, dtype=torch.long)
        log_prob = log_prob + Categorical(logits=scaled_patch).log_prob(patch_tensor).squeeze(0)
    return log_prob


def recompute_action_log_probs(
    model: torch.nn.Module,
    image: Any,
    question: str,
    actions: Sequence[Dict[str, Any]],
    temperature: float,
    image_size: Optional[int] = None,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
) -> List[torch.Tensor]:
    """Replay stored actions and recompute current-policy log-probs."""
    state, bank = prepare_full_context_state_and_bank(model, image, question, image_size=image_size)
    selected_patches: set[int] = set()
    selected_visual_count = 0
    log_probs: List[torch.Tensor] = []
    for step_idx, action in enumerate(actions):
        log_probs.append(
            action_log_prob_for_replay(
                model,
                state,
                bank,
                action,
                step_idx,
                temperature,
                selected_patches,
                selected_visual_count=selected_visual_count,
                min_controller_actions_before_stop=min_controller_actions_before_stop,
                min_visual_actions_before_stop=min_visual_actions_before_stop,
            )
        )
        action_type = str(action["type"]).upper()
        if action_type == "PATCH":
            selected_patches.add(int(action["patch_idx"]))
            selected_visual_count += 1
        elif action_type in {"REGION", "GLOBAL"}:
            selected_visual_count += 1
        if action_type == "STOP":
            break
        with torch.no_grad():
            model.apply_mined_actions(state, bank, [action])
    return log_probs


def clipped_grpo_loss(
    advantages: torch.Tensor,
    rollouts: Sequence[Dict[str, Any]],
    current_log_probs: Sequence[Sequence[torch.Tensor]],
    clip_epsilon: float = 0.2,
) -> Optional[torch.Tensor]:
    """PPO/GRPO clipped policy loss over stored rollouts."""
    loss_terms: List[torch.Tensor] = []
    for advantage, rollout, current_steps in zip(advantages, rollouts, current_log_probs):
        old_steps = rollout.get("old_log_probs") or []
        if not old_steps or not current_steps:
            continue
        step_terms = []
        for current_log_prob, old_log_prob in zip(current_steps, old_steps):
            ratio = torch.exp(current_log_prob - old_log_prob.to(current_log_prob.device))
            clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
            step_terms.append(torch.minimum(ratio * advantage.detach(), clipped * advantage.detach()))
        if step_terms:
            loss_terms.append(-torch.stack(step_terms).mean())
    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def clipped_grpo_diagnostics(
    advantages: torch.Tensor,
    rollouts: Sequence[Dict[str, Any]],
    current_log_probs: Sequence[Sequence[torch.Tensor]],
    clip_epsilon: float = 0.2,
) -> Dict[str, float]:
    """Return ratio/clip diagnostics for PPO-style Phase 5 updates."""
    ratios: List[torch.Tensor] = []
    clipped_flags: List[torch.Tensor] = []
    for rollout, current_steps in zip(rollouts, current_log_probs):
        old_steps = rollout.get("old_log_probs") or []
        for current_log_prob, old_log_prob in zip(current_steps, old_steps):
            ratio = torch.exp(current_log_prob.detach() - old_log_prob.to(current_log_prob.device).detach())
            ratios.append(ratio.float())
            clipped_flags.append((torch.abs(ratio - 1.0) > float(clip_epsilon)).float())
    if not ratios:
        return {
            "adv_mean": float(advantages.detach().float().mean().item()) if advantages.numel() else 0.0,
            "adv_std": float(advantages.detach().float().std(unbiased=False).item()) if advantages.numel() else 0.0,
            "adv_abs_mean": float(advantages.detach().float().abs().mean().item()) if advantages.numel() else 0.0,
            "mean_ratio": 0.0,
            "clip_fraction": 0.0,
        }
    ratio_tensor = torch.stack(ratios)
    clipped_tensor = torch.stack(clipped_flags)
    return {
        "adv_mean": float(advantages.detach().float().mean().item()),
        "adv_std": float(advantages.detach().float().std(unbiased=False).item()),
        "adv_abs_mean": float(advantages.detach().float().abs().mean().item()),
        "mean_ratio": float(ratio_tensor.mean().item()),
        "clip_fraction": float(clipped_tensor.mean().item()),
    }


def target_logprob(model: torch.nn.Module, state: Dict[str, Any], target_text: str) -> torch.Tensor:
    """Length-normalized average log-prob for target text."""
    return -differentiable_state_ce(model, state, target_text)


def _wrong_same_image_actions(actions: Sequence[Dict[str, Any]], bank: Dict[str, torch.Tensor], rng: random.Random) -> List[Dict[str, Any]]:
    pair = {"example_id": "rollout", "positive_actions": [action for action in actions if action["type"] != "STOP"]}
    negative_actions, reason = build_negative_actions(pair, bank, "same_image_wrong", rng)
    if negative_actions is None:
        return [copy.deepcopy(action) for action in pair["positive_actions"]]
    return negative_actions


def counterfactual_logprob_reward(
    model: torch.nn.Module,
    rollout: Dict[str, Any],
    source_example: Dict[str, Any],
    example_index: Dict[str, Dict[str, Any]],
    rng: random.Random,
    gold_target: str,
    random_image_probability: float = 0.35,
    image_size: Optional[int] = None,
) -> float:
    """Compute R_cf by corrupting rollout visual evidence with same/random-image evidence."""
    visual_actions = rollout.get("selected_visual_actions") or []
    if not visual_actions:
        return 0.0

    final_state = rollout["final_state"]
    with torch.no_grad():
        logp_real = target_logprob(model, final_state, gold_target)

    actions = [action for action in rollout["actions"] if action["type"] != "STOP"]
    pair = {
        "example_id": source_example.get("id"),
        "question": source_example.get("question", ""),
        "positive_actions": actions,
        "prefix_trace": [],
    }
    negative_type = "different_image_random" if rng.random() < random_image_probability else "same_image_wrong"
    negative_actions, reason = build_negative_actions(
        pair,
        rollout["bank"],
        negative_type,
        rng,
        example_index=example_index,
        model=model,
        image_size=image_size,
        negative_bank_cache={},
    )
    if negative_actions is None:
        negative_actions = _wrong_same_image_actions(actions, rollout["bank"], rng)

    corrupt_state, corrupt_bank = prepare_full_context_state_and_bank(
        model,
        source_example["image"],
        source_example["question"],
        image_size=image_size,
    )
    with torch.no_grad():
        apply_counterfactual_actions(model, corrupt_state, corrupt_bank, negative_actions)
        logp_corrupt = target_logprob(model, corrupt_state, gold_target)
    del reason
    return float((logp_real - logp_corrupt).detach().cpu().item())


def compute_phase5_reward(
    model: torch.nn.Module,
    rollout: Dict[str, Any],
    example: Dict[str, Any],
    example_index: Dict[str, Dict[str, Any]],
    correctness_score: float,
    rng: random.Random,
    logp_weight: float = 0.2,
    counterfactual_weight: float = 0.3,
    use_counterfactual_reward: bool = True,
    cf_random_image_probability: float = 0.35,
    no_stop_penalty: float = 0.2,
    think_once_bonus: float = 0.05,
    no_visual_penalty: float = 0.3,
    early_stop_penalty: float = 0.3,
    min_controller_actions_before_stop: int = 0,
    min_visual_actions_before_stop: int = 0,
    image_size: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Phase 5 compact reward and components."""
    gold_target = str(example.get("answer") or example.get("gold_answer") or "")
    with torch.no_grad():
        r_logp = float(target_logprob(model, rollout["final_state"], gold_target).detach().cpu().item())
    r_cf = 0.0
    if use_counterfactual_reward:
        r_cf = counterfactual_logprob_reward(
            model,
            rollout,
            example,
            example_index,
            rng,
            gold_target,
            random_image_probability=cf_random_image_probability,
            image_size=image_size,
        )
    num_steps = int(rollout.get("num_steps", len(rollout.get("actions", []))))
    visual_count = len(rollout.get("selected_visual_actions") or [])
    r_stop = 0.0 if rollout.get("stopped") else -float(no_stop_penalty)
    r_visual = -float(no_visual_penalty) if visual_count < int(min_visual_actions_before_stop) else 0.0
    r_early_stop = 0.0
    if rollout.get("stopped") and num_steps <= int(min_controller_actions_before_stop):
        r_early_stop = -float(early_stop_penalty)
    think_count = sum(1 for action in rollout.get("actions", []) if str(action.get("type", "")).upper() == "THINK")
    r_think_once = float(think_once_bonus) if think_count == 1 else 0.0
    reward = (
        float(correctness_score)
        + float(logp_weight) * r_logp
        + float(counterfactual_weight) * r_cf
        + r_stop
        + r_visual
        + r_early_stop
        + r_think_once
    )
    return {
        "reward": reward,
        "r_correct": float(correctness_score),
        "r_logp": r_logp,
        "r_cf": r_cf,
        "r_stop": r_stop,
        "r_visual": r_visual,
        "r_early_stop": r_early_stop,
        "r_think_once": r_think_once,
        "think_count": float(think_count),
        "visual_count": float(visual_count),
        "num_steps": float(num_steps),
    }


class Phase5MetricTracker:
    """Track compact Phase 5 training metrics."""

    def __init__(self) -> None:
        self.count = 0
        self.values: Counter[str] = Counter()

    def update(self, metrics: Dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            self.values[key] += float(value)

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0}
        summary = {key: value / self.count for key, value in self.values.items()}
        summary["count"] = self.count
        return summary
