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

DEFAULT_TYPE_LOSS_WEIGHTS = {
    "PATCH": 1.0,
    "REGION": 1.0,
    "THINK": 1.0,
    "GLOBAL": 1.0,
    "STOP": 1.0,
}

PHASE3_V2_TYPE_LOSS_WEIGHTS = {
    "PATCH": 1.0,
    "REGION": 2.5,
    "THINK": 5.0,
    "STOP": 5.0,
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


def action_type(action: Dict[str, Any], action_name_to_id: Optional[Dict[str, int]] = None) -> str:
    """Return a normalized primitive action type."""
    action_name = str(action.get("type", "")).upper()
    if action_name == "NO_OP":
        raise ValueError("NO_OP is not a Phase 3 controller target.")
    valid_actions = action_name_to_id if action_name_to_id is not None else ACTION_TYPE_IDS
    if action_name not in valid_actions:
        raise ValueError(f"Unsupported Phase 3 action type: {action_name}")
    return action_name


def compute_action_loss(
    type_logits: torch.Tensor,
    region_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    action: Dict[str, Any],
    return_components: bool = False,
    type_loss_weights: Optional[Dict[str, float]] = None,
    action_name_to_id: Optional[Dict[str, int]] = None,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | str]]:
    """Compute controller SFT loss for one primitive action."""
    name = action_type(action, action_name_to_id=action_name_to_id)
    device = type_logits.device
    action_ids = action_name_to_id if action_name_to_id is not None else ACTION_NAME_TO_ID
    target_type = torch.tensor([action_ids[name]], device=device, dtype=torch.long)
    raw_type_loss = F.cross_entropy(type_logits, target_type)
    type_weight = float((type_loss_weights or DEFAULT_TYPE_LOSS_WEIGHTS).get(name, 1.0))
    type_loss = raw_type_loss * type_weight
    patch_loss = torch.zeros((), device=device, dtype=type_loss.dtype)
    region_loss = torch.zeros((), device=device, dtype=type_loss.dtype)
    loss = type_loss

    if name == "PATCH":
        patch_idx = int(action["patch_idx"])
        target_patch = torch.tensor([patch_idx], device=device, dtype=torch.long)
        patch_loss = F.cross_entropy(patch_logits, target_patch)
        loss = loss + patch_loss
    elif name == "REGION":
        region_idx = int(action["region_idx"])
        target_region = torch.tensor([region_idx], device=device, dtype=torch.long)
        region_loss = F.cross_entropy(region_logits, target_region)
        loss = loss + region_loss

    if return_components:
        return loss, {
            "action_type": name,
            "total_loss": loss,
            "type_loss": type_loss,
            "raw_type_loss": raw_type_loss,
            "patch_loss": patch_loss,
            "region_loss": region_loss,
        }
    return loss


def _decision_improvement(decision: Dict[str, Any]) -> float:
    if "improvement" in decision:
        return float(decision.get("improvement") or 0.0)
    ce_noop = decision.get("ce_noop")
    ce_selected = decision.get("ce_selected")
    if ce_noop is None or ce_selected is None:
        return 0.0
    return float(ce_noop) - float(ce_selected)


def _clean_decision_actions(actions: Iterable[Dict[str, Any]], remove_global: bool) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for action in actions:
        action_name = str(action.get("type", "")).upper()
        if remove_global and action_name == "GLOBAL":
            continue
        copied = dict(action)
        copied["type"] = action_name
        cleaned.append(copied)
    return cleaned


def _is_visual_block(actions: List[Dict[str, Any]]) -> bool:
    return any(str(action.get("type", "")).upper() in {"PATCH", "REGION", "GLOBAL"} for action in actions)


def transform_phase3_v2_decision_blocks(
    decisions: Iterable[Dict[str, Any]],
    visual_or_region_min_improvement: float = 0.05,
    think_min_improvement: float = 0.03,
    max_decision_blocks_per_example: int = 6,
    max_primitive_actions_per_example: int = 8,
    no_op_stop_ce_threshold: float = 0.05,
    remove_global: bool = True,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """Clean mined decisions into capped Phase 3 v2 supervision blocks."""
    candidates: List[Tuple[int, float, List[Dict[str, Any]]]] = []
    skipped = Counter()
    terminal_stop_from_noop = False

    for index, decision in enumerate(decisions):
        raw_actions = decision.get("actions") or []
        selected = str(decision.get("selected", "")).upper()
        if not raw_actions:
            if selected == "NO_OP" and float(decision.get("ce_noop", 1.0)) <= no_op_stop_ce_threshold:
                candidates.append((index, float("inf"), [{"type": "STOP"}]))
                terminal_stop_from_noop = True
                break
            skipped["noop"] += 1
            continue

        actions = _clean_decision_actions(raw_actions, remove_global=remove_global)
        if not actions:
            skipped["global"] += 1
            continue

        improvement = _decision_improvement(decision)
        action_names = {str(action.get("type", "")).upper() for action in actions}
        if action_names == {"THINK"}:
            if improvement < think_min_improvement:
                skipped["weak_think"] += 1
                continue
        elif _is_visual_block(actions):
            if improvement < visual_or_region_min_improvement:
                skipped["weak_visual"] += 1
                continue
        else:
            skipped["unsupported"] += 1
            continue

        candidates.append((index, improvement, actions))

    selected_blocks: List[Tuple[int, float, List[Dict[str, Any]]]] = []
    primitive_count = 0
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        actions = candidate[2]
        if len(selected_blocks) >= max_decision_blocks_per_example:
            skipped["cap_blocks"] += 1
            continue
        if primitive_count + len(actions) > max_primitive_actions_per_example:
            skipped["cap_primitives"] += 1
            continue
        selected_blocks.append(candidate)
        primitive_count += len(actions)

    selected_blocks.sort(key=lambda item: item[0])
    blocks = [actions for _, _, actions in selected_blocks]
    if not terminal_stop_from_noop:
        blocks.append([{"type": "STOP"}])

    metrics = {
        "candidate_blocks": len(candidates),
        "kept_blocks": len(blocks),
        "kept_non_stop_blocks": len(selected_blocks),
        "kept_primitives": sum(len(block) for block in blocks),
        "skipped_blocks": dict(skipped),
        "converted_noop_to_stop": int(terminal_stop_from_noop),
    }
    return blocks, metrics


def _update_logit_stats(prefix: str, logits: torch.Tensor, totals: Dict[str, float], counts: Dict[str, int]) -> None:
    detached = logits.detach().float()
    for stat_name, value in {
        "min": detached.min().item(),
        "max": detached.max().item(),
        "mean": detached.mean().item(),
    }.items():
        key = f"{prefix}_{stat_name}"
        totals[key] = totals.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def _mean_dict(totals: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {key: totals[key] / max(1, counts.get(key, 0)) for key in totals}


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
    decision_block_normalized: bool = False,
    type_loss_weights: Optional[Dict[str, float]] = None,
    phase3_v2: bool = False,
    visual_or_region_min_improvement: float = 0.05,
    think_min_improvement: float = 0.03,
    max_decision_blocks_per_example: int = 6,
    max_primitive_actions_per_example: int = 8,
    no_op_stop_ce_threshold: float = 0.05,
    remove_global: bool = False,
    visual_block_dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Replay mined decisions and return the mean controller SFT loss."""
    if full_context_probability < 0.0 or full_context_probability > 1.0:
        raise ValueError("full_context_probability must be in [0, 1].")
    if visual_block_dropout_p < 0.0 or visual_block_dropout_p > 1.0:
        raise ValueError("visual_block_dropout_p must be in [0, 1].")
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
    component_loss_totals: Dict[str, float] = {}
    component_loss_counts: Dict[str, int] = {}
    action_loss_totals: Dict[str, float] = {}
    action_loss_counts: Dict[str, int] = {}
    logit_stat_totals: Dict[str, float] = {}
    logit_stat_counts: Dict[str, int] = {}
    skipped_noop = 0
    dropped_visual_blocks = 0
    controller_step = 0
    action_name_to_id = getattr(model, "action_name_to_id", None)
    transform_metrics: Dict[str, Any] = {}

    def record_step_metrics(
        type_logits: torch.Tensor,
        region_logits: torch.Tensor,
        patch_logits: torch.Tensor,
        components: Dict[str, torch.Tensor | str],
    ) -> None:
        _update_logit_stats("type_logits", type_logits, logit_stat_totals, logit_stat_counts)
        _update_logit_stats("region_logits", region_logits, logit_stat_totals, logit_stat_counts)
        _update_logit_stats("patch_logits", patch_logits, logit_stat_totals, logit_stat_counts)
        action_name = str(components["action_type"])
        total_value = float(components["total_loss"].detach().item())  # type: ignore[union-attr]
        action_loss_totals[action_name] = action_loss_totals.get(action_name, 0.0) + total_value
        action_loss_counts[action_name] = action_loss_counts.get(action_name, 0) + 1
        for key in ("total_loss", "type_loss", "raw_type_loss", "patch_loss", "region_loss"):
            value = components[key]
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"patch_loss", "region_loss"} and float(value.detach().item()) == 0.0:
                continue
            component_loss_totals[key] = component_loss_totals.get(key, 0.0) + float(value.detach().item())
            component_loss_counts[key] = component_loss_counts.get(key, 0) + 1

    if phase3_v2:
        blocks, transform_metrics = transform_phase3_v2_decision_blocks(
            mined_row.get("decisions", []),
            visual_or_region_min_improvement=visual_or_region_min_improvement,
            think_min_improvement=think_min_improvement,
            max_decision_blocks_per_example=max_decision_blocks_per_example,
            max_primitive_actions_per_example=max_primitive_actions_per_example,
            no_op_stop_ce_threshold=no_op_stop_ce_threshold,
            remove_global=remove_global,
        )
        skipped_noop = int(transform_metrics.get("skipped_blocks", {}).get("noop", 0))
    else:
        blocks = []
        for decision in mined_row.get("decisions", []):
            actions = decision.get("actions") or []
            if not actions:
                skipped_noop += 1
                continue
            blocks.append([dict(action) for action in actions])
        blocks.append([{"type": "STOP"}])

    def apply_visual_dropout(block: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        if visual_block_dropout_p <= 0.0 or not _is_visual_block(block):
            return block, False
        if sampler.random() >= visual_block_dropout_p:
            return block, False
        kept = [action for action in block if str(action.get("type", "")).upper() == "THINK"]
        return kept, True

    for block in blocks:
        actions, dropped_block = apply_visual_dropout(block)
        if dropped_block:
            dropped_visual_blocks += 1
        if not actions:
            continue
        block_losses: List[torch.Tensor] = []
        for action in actions:
            type_logits, region_logits, patch_logits = model.controller_logits_from_state(
                state,
                bank,
                controller_step,
            )
            loss, components = compute_action_loss(
                type_logits,
                region_logits,
                patch_logits,
                action,
                return_components=True,
                type_loss_weights=type_loss_weights,
                action_name_to_id=action_name_to_id,
            )
            block_losses.append(loss)
            action_counts[str(components["action_type"])] += 1
            record_step_metrics(type_logits, region_logits, patch_logits, components)
            if str(components["action_type"]) != "STOP":
                with torch.no_grad():
                    model.apply_mined_actions(state, bank, [action])
                controller_step += 1
        if block_losses:
            if decision_block_normalized:
                losses.append(torch.stack(block_losses).mean())
            else:
                losses.extend(block_losses)

    if not losses:
        raise ValueError(f"No Phase 3 controller targets remained for example {mined_row.get('example_id')}.")

    metrics = {
        "example_id": mined_row.get("example_id"),
        "num_targets": len(losses),
        "num_primitive_targets": sum(action_counts.values()),
        "num_controller_steps": sum(action_counts.values()),
        "skipped_noop_decisions": skipped_noop,
        "dropped_visual_blocks": dropped_visual_blocks,
        "action_counts": dict(action_counts),
        "loss_components": _mean_dict(component_loss_totals, component_loss_counts),
        "loss_component_counts": dict(component_loss_counts),
        "action_loss_means": _mean_dict(action_loss_totals, action_loss_counts),
        "action_loss_counts": dict(action_loss_counts),
        "logit_stats": _mean_dict(logit_stat_totals, logit_stat_counts),
        "logit_stat_counts": dict(logit_stat_counts),
        "initial_visual_mode": "full_context" if use_full_context else "global_mean",
        "used_full_context": use_full_context,
        "decision_block_normalized": decision_block_normalized,
        "transform": transform_metrics,
    }
    return torch.stack(losses).mean(), metrics


def set_seed(seed: int) -> None:
    """Set Python/Torch seeds for reproducible controller SFT."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
