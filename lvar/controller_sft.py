import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lvar.utils import (
    ACTION_GLOBAL,
    ACTION_NAME_TO_ID,
    ACTION_PATCH,
    ACTION_REGION,
    ACTION_STOP,
    ACTION_THINK,
)


ACTION_TYPE_IDS = {
    "THINK": ACTION_THINK,
    "GLOBAL": ACTION_GLOBAL,
    "REGION": ACTION_REGION,
    "PATCH": ACTION_PATCH,
    "STOP": ACTION_STOP,
}


def load_mined_trace_rows(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load Phase 2 mined JSONL rows."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_example_index(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index source dataset examples by stringified id for mined-row image lookup."""
    index: Dict[str, Dict[str, Any]] = {}
    for example in dataset:
        index[str(example.get("id"))] = example
    return index


def flatten_supervised_actions(decisions: Iterable[Dict[str, Any]], include_stop: bool = True) -> List[Dict[str, Any]]:
    """Flatten non-empty mined decision actions and optionally append STOP."""
    actions: List[Dict[str, Any]] = []
    for decision in decisions:
        decision_actions = decision.get("actions") or []
        actions.extend(dict(action) for action in decision_actions)
    if include_stop:
        actions.append({"type": "STOP"})
    return actions


def action_type(action: Dict[str, Any]) -> str:
    """Return a normalized primitive action type."""
    action_name = str(action.get("type", "")).upper()
    if action_name == "NO_OP":
        raise ValueError("NO_OP is not a Phase 3 controller target.")
    if action_name not in ACTION_TYPE_IDS:
        raise ValueError(f"Unsupported Phase 3 action type: {action_name}")
    return action_name


def compute_action_loss(
    type_logits: torch.Tensor,
    region_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    action: Dict[str, Any],
) -> torch.Tensor:
    """Compute controller SFT loss for one primitive action."""
    name = action_type(action)
    device = type_logits.device
    target_type = torch.tensor([ACTION_NAME_TO_ID[name]], device=device, dtype=torch.long)
    loss = F.cross_entropy(type_logits, target_type)

    if name == "PATCH":
        patch_idx = int(action["patch_idx"])
        target_patch = torch.tensor([patch_idx], device=device, dtype=torch.long)
        loss = loss + F.cross_entropy(patch_logits, target_patch)
    elif name == "REGION":
        region_idx = int(action["region_idx"])
        target_region = torch.tensor([region_idx], device=device, dtype=torch.long)
        loss = loss + F.cross_entropy(region_logits, target_region)

    return loss


def set_controller_sft_trainable(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Freeze the VLM and visual-bank modules, leaving only controller-facing SFT params."""
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


def controller_sft_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return the controller-only checkpoint payload."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_controller_sft_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save controller SFT parameters and lightweight metadata."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": controller_sft_state_dict(model),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def prepare_controller_sft_state(
    model: torch.nn.Module,
    source_example: Dict[str, Any],
    mined_row: Dict[str, Any],
    image_size: Optional[int] = None,
    use_full_context: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Build initial replay state and visual bank for one mined example."""
    with torch.no_grad():
        batch = model.prepare_inputs(
            source_example["image"],
            mined_row.get("question", source_example["question"]),
            add_answer_instruction=False,
            image_size=image_size,
        )
        image_tokens = model.get_projected_image_tokens(batch)
        bank = model.build_visual_bank(image_tokens)
        if use_full_context:
            state = model.build_initial_state(batch)
        else:
            state = model.build_coarse_initial_state(batch, bank)
    return state, bank


def replay_controller_sft_loss(
    model: torch.nn.Module,
    mined_row: Dict[str, Any],
    source_example: Dict[str, Any],
    image_size: Optional[int] = None,
    full_context_probability: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Replay mined decisions and return the mean controller SFT loss."""
    if full_context_probability < 0.0 or full_context_probability > 1.0:
        raise ValueError("full_context_probability must be in [0, 1].")
    sampler = rng or random
    use_full_context = sampler.random() < full_context_probability
    state, bank = prepare_controller_sft_state(
        model,
        source_example,
        mined_row,
        image_size=image_size,
        use_full_context=use_full_context,
    )
    losses: List[torch.Tensor] = []
    action_counts: Counter[str] = Counter()
    skipped_noop = 0
    controller_step = 0

    for decision in mined_row.get("decisions", []):
        actions = decision.get("actions") or []
        if not actions:
            skipped_noop += 1
            continue

        for action in actions:
            type_logits, region_logits, patch_logits = model.controller_logits_from_state(
                state,
                bank,
                controller_step,
            )
            losses.append(compute_action_loss(type_logits, region_logits, patch_logits, action))
            action_counts[action_type(action)] += 1
            with torch.no_grad():
                model.apply_mined_actions(state, bank, [action])
            controller_step += 1

    stop_action = {"type": "STOP"}
    type_logits, region_logits, patch_logits = model.controller_logits_from_state(
        state,
        bank,
        controller_step,
    )
    losses.append(compute_action_loss(type_logits, region_logits, patch_logits, stop_action))
    action_counts["STOP"] += 1

    metrics = {
        "example_id": mined_row.get("example_id"),
        "num_targets": len(losses),
        "num_controller_steps": controller_step + 1,
        "skipped_noop_decisions": skipped_noop,
        "action_counts": dict(action_counts),
        "initial_visual_mode": "full_context" if use_full_context else "global_mean",
        "used_full_context": use_full_context,
    }
    return torch.stack(losses).mean(), metrics


def set_seed(seed: int) -> None:
    """Set Python/Torch seeds for reproducible controller SFT."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
