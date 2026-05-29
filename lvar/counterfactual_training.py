import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from lvar.controller_sft import action_type, compute_action_loss


NEGATIVE_TYPES = ("same_image_wrong", "different_image_random", "same_image_noisy")
CONTEXT_GLOBAL = "global_mean"
CONTEXT_FULL = "full_context"


def load_counterfactual_pairs(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Flatten Phase 2 mined JSONL rows into Phase 4 counterfactual pair records."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            for pair in row.get("counterfactual_pairs") or []:
                pairs.append(
                    {
                        "example_id": row.get("example_id"),
                        "question": row.get("question"),
                        "prefix_trace": copy.deepcopy(pair.get("prefix_trace") or []),
                        "positive_actions": copy.deepcopy(pair.get("positive_actions") or []),
                        "target_text": pair.get("target_text", ""),
                    }
                )
                if limit is not None and len(pairs) >= limit:
                    return pairs
    return pairs


def validate_negative_type_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Validate and normalize negative type probabilities."""
    normalized = {name: float(probs.get(name, 0.0)) for name in NEGATIVE_TYPES}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("negative_type_probs cannot contain negative values.")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("negative_type_probs must have positive total mass.")
    return {name: value / total for name, value in normalized.items()}


def sample_negative_type(probs: Dict[str, float], rng: random.Random) -> str:
    """Sample one configured negative type."""
    normalized = validate_negative_type_probs(probs)
    draw = rng.random()
    cumulative = 0.0
    for name in NEGATIVE_TYPES:
        cumulative += normalized[name]
        if draw <= cumulative:
            return name
    return NEGATIVE_TYPES[-1]


def sample_context_mode(full_probability: float, rng: random.Random) -> str:
    """Sample the initial visual context mode for both paths in a pair."""
    if full_probability < 0.0 or full_probability > 1.0:
        raise ValueError("context_full_probability must be in [0, 1].")
    return CONTEXT_FULL if rng.random() < full_probability else CONTEXT_GLOBAL


def _tokenize_target(model: torch.nn.Module, target_text: str) -> torch.Tensor:
    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("The processor must expose a tokenizer for CE scoring.")
    encoded = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos = torch.tensor([[int(eos_token_id)]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, eos], dim=1)
    return input_ids.to(model.device)


def differentiable_state_ce(model: torch.nn.Module, state: Dict[str, Any], target_text: str) -> torch.Tensor:
    """Compute differentiable CE(target_text | state) over target tokens only."""
    target_ids = _tokenize_target(model, target_text)
    if target_ids.numel() == 0:
        raise ValueError("Cannot score an empty target.")
    target_embeds = model._embed_input_ids(target_ids)
    prefix_embeds = state["inputs_embeds"]
    prefix_mask = state["attention_mask"]
    prefix_len = prefix_embeds.size(1)

    if target_ids.size(1) > 1:
        input_embeds = torch.cat([prefix_embeds, target_embeds[:, :-1, :]], dim=1)
        target_mask = torch.ones(
            (prefix_mask.size(0), target_ids.size(1) - 1),
            device=model.device,
            dtype=prefix_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
    else:
        input_embeds = prefix_embeds
        attention_mask = prefix_mask

    outputs = model.backbone(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )
    logits = outputs.logits[:, prefix_len - 1 : prefix_len - 1 + target_ids.size(1), :]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target_ids.reshape(-1),
        reduction="mean",
    )


def prepare_phase4_state_and_bank(
    model: torch.nn.Module,
    source_example: Dict[str, Any],
    question: str,
    image_size: Optional[int],
    context_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Build a full or coarse replay state and visual bank for a source image."""
    with torch.no_grad():
        batch = model.prepare_inputs(
            source_example["image"],
            question,
            add_answer_instruction=False,
            image_size=image_size,
        )
        image_tokens = model.get_projected_image_tokens(batch)
        bank = model.build_visual_bank(image_tokens)
        if context_mode == CONTEXT_FULL:
            state = model.build_initial_state(batch)
        elif context_mode == CONTEXT_GLOBAL:
            state = model.build_coarse_initial_state(batch, bank)
        else:
            raise ValueError(f"Unsupported context mode: {context_mode}")
    return state, bank


def _insert_evidence_tensor(model: torch.nn.Module, state: Dict[str, Any], evidence_tokens: torch.Tensor) -> None:
    model._insert_evidence_token(state, evidence_tokens.to(model.device))


def apply_counterfactual_actions(
    model: torch.nn.Module,
    state: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    actions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply normal or Phase 4 evidence-override actions to a recurrent state."""
    for action in actions:
        action_name = str(action.get("type", "")).upper()
        evidence_tokens = action.get("evidence_tokens")
        if evidence_tokens is not None and action_name in {"GLOBAL", "REGION", "PATCH"}:
            _insert_evidence_tensor(model, state, evidence_tokens)
            continue
        model.apply_mined_actions(state, bank, [action])
    return state


def _visual_ids(actions: Iterable[Dict[str, Any]], visual_type: str) -> List[int]:
    key = "patch_idx" if visual_type == "PATCH" else "region_idx"
    ids: List[int] = []
    for action in actions:
        if str(action.get("type", "")).upper() == visual_type and key in action:
            ids.append(int(action[key]))
    return ids


def _sample_wrong_mapping(
    positive_ids: Sequence[int],
    num_choices: int,
    avoid_ids: Sequence[int],
    rng: random.Random,
) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    preferred = [idx for idx in range(num_choices) if idx not in set(avoid_ids)]
    for pos_id in positive_ids:
        if pos_id in mapping:
            continue
        candidates = [idx for idx in preferred if idx != pos_id]
        if not candidates:
            candidates = [idx for idx in range(num_choices) if idx != pos_id]
        if not candidates:
            return None
        mapping[pos_id] = int(rng.choice(candidates))
    return mapping


def _make_noisy(tokens: torch.Tensor, noise_scale: float) -> torch.Tensor:
    original = tokens.detach()
    std = original.float().std(unbiased=False)
    if float(std.item()) == 0.0:
        std = torch.ones((), device=original.device, dtype=original.dtype)
    noisy = original + float(noise_scale) * std.to(original.dtype) * torch.randn_like(original)
    original_norm = torch.linalg.vector_norm(original.float()).clamp_min(1e-8)
    noisy_norm = torch.linalg.vector_norm(noisy.float()).clamp_min(1e-8)
    return (noisy * (original_norm / noisy_norm).to(noisy.dtype)).to(tokens.dtype)


def _sample_negative_example(
    example_id: str,
    example_index: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    candidates = [example for key, example in example_index.items() if str(key) != str(example_id)]
    if not candidates:
        return None
    return rng.choice(candidates)


def _bank_for_example(
    model: torch.nn.Module,
    example: Dict[str, Any],
    question: str,
    image_size: Optional[int],
    cache: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    example_id = str(example.get("id"))
    if example_id in cache:
        return cache[example_id]
    with torch.no_grad():
        batch = model.prepare_inputs(
            example["image"],
            question,
            add_answer_instruction=False,
            image_size=image_size,
        )
        image_tokens = model.get_projected_image_tokens(batch)
        bank = model.build_visual_bank(image_tokens)
    cache[example_id] = bank
    return bank


def build_negative_actions(
    pair: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    negative_type: str,
    rng: random.Random,
    example_index: Optional[Dict[str, Dict[str, Any]]] = None,
    model: Optional[torch.nn.Module] = None,
    image_size: Optional[int] = None,
    negative_bank_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    noise_scale: float = 0.5,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Regenerate negative actions while preserving positive action structure."""
    if negative_type not in NEGATIVE_TYPES:
        raise ValueError(f"Unsupported negative type: {negative_type}")

    positive_actions = pair.get("positive_actions") or []
    prefix_trace = pair.get("prefix_trace") or []
    all_trace_actions = list(prefix_trace) + list(positive_actions)

    patch_ids = _visual_ids(positive_actions, "PATCH")
    region_ids = _visual_ids(positive_actions, "REGION")
    has_global = any(str(action.get("type", "")).upper() == "GLOBAL" for action in positive_actions)

    patch_mapping: Dict[int, int] = {}
    region_mapping: Dict[int, int] = {}
    negative_bank: Optional[Dict[str, torch.Tensor]] = None
    negative_example_id = None

    if negative_type == "same_image_wrong":
        if has_global:
            return None, "global_same_image_wrong_undefined"
        if patch_ids:
            mapping = _sample_wrong_mapping(
                patch_ids,
                int(bank["patches"].size(0)),
                _visual_ids(all_trace_actions, "PATCH"),
                rng,
            )
            if mapping is None:
                return None, "patch_no_wrong_choice"
            patch_mapping = mapping
        if region_ids:
            mapping = _sample_wrong_mapping(
                region_ids,
                int(bank["raw_regions"].size(0)),
                _visual_ids(all_trace_actions, "REGION"),
                rng,
            )
            if mapping is None:
                return None, "region_no_wrong_choice"
            region_mapping = mapping

    elif negative_type == "different_image_random":
        if example_index is None or model is None:
            return None, "missing_negative_image_context"
        negative_example = _sample_negative_example(str(pair.get("example_id")), example_index, rng)
        if negative_example is None:
            return None, "no_different_image"
        negative_example_id = str(negative_example.get("id"))
        negative_bank = _bank_for_example(
            model,
            negative_example,
            str(pair.get("question") or ""),
            image_size,
            negative_bank_cache if negative_bank_cache is not None else {},
        )
        if patch_ids:
            patch_mapping = {
                pos_id: int(rng.randrange(int(negative_bank["patches"].size(0))))
                for pos_id in set(patch_ids)
            }
        if region_ids:
            region_mapping = {
                pos_id: int(rng.randrange(int(negative_bank["raw_regions"].size(0))))
                for pos_id in set(region_ids)
            }

    negative_actions: List[Dict[str, Any]] = []
    for action in positive_actions:
        action_name = str(action.get("type", "")).upper()
        if action_name == "PATCH":
            pos_idx = int(action["patch_idx"])
            if negative_type == "same_image_wrong":
                neg_idx = patch_mapping[pos_idx]
                negative_actions.append({"type": "PATCH", "patch_idx": neg_idx, "negative_type": negative_type})
            elif negative_type == "different_image_random":
                if negative_bank is None:
                    return None, "missing_negative_bank"
                neg_idx = patch_mapping[pos_idx]
                negative_actions.append(
                    {
                        "type": "PATCH",
                        "patch_idx": neg_idx,
                        "negative_type": negative_type,
                        "negative_example_id": negative_example_id,
                        "evidence_tokens": negative_bank["patches"][neg_idx].unsqueeze(0),
                    }
                )
            else:
                negative_actions.append(
                    {
                        "type": "PATCH",
                        "patch_idx": pos_idx,
                        "negative_type": negative_type,
                        "evidence_tokens": _make_noisy(bank["patches"][pos_idx].unsqueeze(0), noise_scale),
                    }
                )
        elif action_name == "REGION":
            pos_idx = int(action["region_idx"])
            if negative_type == "same_image_wrong":
                neg_idx = region_mapping[pos_idx]
                negative_actions.append({"type": "REGION", "region_idx": neg_idx, "negative_type": negative_type})
            elif negative_type == "different_image_random":
                if negative_bank is None:
                    return None, "missing_negative_bank"
                neg_idx = region_mapping[pos_idx]
                negative_actions.append(
                    {
                        "type": "REGION",
                        "region_idx": neg_idx,
                        "negative_type": negative_type,
                        "negative_example_id": negative_example_id,
                        "evidence_tokens": negative_bank["raw_regions"][neg_idx],
                    }
                )
            else:
                negative_actions.append(
                    {
                        "type": "REGION",
                        "region_idx": pos_idx,
                        "negative_type": negative_type,
                        "evidence_tokens": _make_noisy(bank["raw_regions"][pos_idx], noise_scale),
                    }
                )
        elif action_name == "GLOBAL":
            if negative_type == "same_image_wrong":
                return None, "global_same_image_wrong_undefined"
            if negative_type == "different_image_random":
                if negative_bank is None:
                    return None, "missing_negative_bank"
                negative_actions.append(
                    {
                        "type": "GLOBAL",
                        "negative_type": negative_type,
                        "negative_example_id": negative_example_id,
                        "evidence_tokens": negative_bank["global"],
                    }
                )
            else:
                negative_actions.append(
                    {
                        "type": "GLOBAL",
                        "negative_type": negative_type,
                        "evidence_tokens": _make_noisy(bank["global"], noise_scale),
                    }
                )
        else:
            negative_actions.append(copy.deepcopy(action))

    return negative_actions, None


def compute_controller_loss_for_actions(
    model: torch.nn.Module,
    state: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    actions: Sequence[Dict[str, Any]],
    start_step: int = 0,
) -> Tuple[torch.Tensor, int, Counter[str]]:
    """Compute controller SFT loss for positive actions while replaying them."""
    losses: List[torch.Tensor] = []
    action_counts: Counter[str] = Counter()
    controller_step = int(start_step)
    for action in actions:
        name = action_type(action)
        type_logits, region_logits, patch_logits = model.controller_logits_from_state(state, bank, controller_step)
        losses.append(compute_action_loss(type_logits, region_logits, patch_logits, action))
        action_counts[name] += 1
        with torch.no_grad():
            model.apply_mined_actions(state, bank, [action])
        controller_step += 1
    if not losses:
        zero = torch.zeros((), device=model.device, dtype=next(model.parameters()).dtype)
        return zero, controller_step, action_counts
    return torch.stack(losses).mean(), controller_step, action_counts


def replay_counterfactual_pair_loss(
    model: torch.nn.Module,
    pair: Dict[str, Any],
    source_example: Dict[str, Any],
    example_index: Dict[str, Dict[str, Any]],
    rng: random.Random,
    negative_type_probs: Dict[str, float],
    context_full_probability: float = 0.2,
    image_size: Optional[int] = 280,
    rank_margin: float = 0.1,
    positive_ce_weight: float = 0.2,
    rank_weight: float = 0.4,
    noise_scale: float = 0.5,
    negative_bank_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Replay one Phase 4 pair and return its full loss plus metrics."""
    context_mode = sample_context_mode(context_full_probability, rng)
    negative_type = sample_negative_type(negative_type_probs, rng)
    question = str(pair.get("question") or source_example.get("question") or "")
    base_state, bank = prepare_phase4_state_and_bank(model, source_example, question, image_size, context_mode)

    negative_actions, skip_reason = build_negative_actions(
        pair,
        bank,
        negative_type,
        rng,
        example_index=example_index,
        model=model,
        image_size=image_size,
        negative_bank_cache=negative_bank_cache,
        noise_scale=noise_scale,
    )
    if negative_actions is None:
        return None, {
            "example_id": pair.get("example_id"),
            "negative_type": negative_type,
            "context_mode": context_mode,
            "skip_reason": skip_reason or "negative_construction_failed",
        }

    prefix_trace = pair.get("prefix_trace") or []
    positive_actions = pair.get("positive_actions") or []
    target_text = str(pair.get("target_text") or "")

    pos_state = model.clone_state(base_state)
    neg_state = model.clone_state(base_state)
    model.apply_mined_actions(pos_state, bank, prefix_trace)
    model.apply_mined_actions(neg_state, bank, prefix_trace)

    ctrl_state = model.clone_state(pos_state)
    l_ctrl, _, action_counts = compute_controller_loss_for_actions(model, ctrl_state, bank, positive_actions)

    apply_counterfactual_actions(model, pos_state, bank, positive_actions)
    apply_counterfactual_actions(model, neg_state, bank, negative_actions)

    ce_pos = differentiable_state_ce(model, pos_state, target_text)
    ce_neg = differentiable_state_ce(model, neg_state, target_text)
    rank_loss = torch.relu(torch.tensor(float(rank_margin), device=ce_pos.device, dtype=ce_pos.dtype) + ce_pos - ce_neg)
    total_loss = l_ctrl + float(positive_ce_weight) * ce_pos + float(rank_weight) * rank_loss
    margin = ce_neg.detach() - ce_pos.detach()

    metrics = {
        "example_id": pair.get("example_id"),
        "negative_type": negative_type,
        "context_mode": context_mode,
        "ce_pos": float(ce_pos.detach().cpu().item()),
        "ce_neg": float(ce_neg.detach().cpu().item()),
        "margin": float(margin.cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "satisfied": bool((ce_pos.detach() + float(rank_margin) < ce_neg.detach()).cpu().item()),
        "l_ctrl": float(l_ctrl.detach().cpu().item()),
        "loss": float(total_loss.detach().cpu().item()),
        "action_counts": dict(action_counts),
    }
    return total_loss, metrics


def phase4_parameter_groups(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """Return separate Phase 4 parameter groups for controller and LoRA params."""
    controller_params: List[torch.nn.Parameter] = []
    lora_params: List[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(parameter)
        else:
            controller_params.append(parameter)
    return {"controller": controller_params, "lora": lora_params}


def set_phase4_trainable(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Train controller-facing parameters and LLM LoRA adapters only."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_modules = [model.controller, model.step_embedding]
    controller_state_norm = getattr(model, "controller_state_norm", None)
    if controller_state_norm is not None:
        trainable_modules.append(controller_state_norm)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True

    if hasattr(model, "backbone"):
        model.backbone.eval()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def phase4_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return trainable Phase 4 parameters."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_phase4_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a Phase 4 trainable-parameter checkpoint."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": phase4_state_dict(model), "metadata": metadata or {}}, path)


class CounterfactualMetricTracker:
    """Accumulate Phase 4 metrics globally and by group."""

    def __init__(self) -> None:
        self.count = 0
        self.skips: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.global_values: Dict[str, float] = defaultdict(float)
        self.by_negative_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.by_context_mode: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, metrics: Dict[str, Any]) -> None:
        if "skip_reason" in metrics:
            self.skips[str(metrics["skip_reason"])] += 1
            return
        self.count += 1
        self.action_counts.update(metrics.get("action_counts", {}))
        numeric_keys = ["ce_pos", "ce_neg", "margin", "rank_loss", "satisfied", "l_ctrl", "loss"]
        for key in numeric_keys:
            value = float(metrics[key])
            self.global_values[key] += value
            self.by_negative_type[str(metrics["negative_type"])][key] += value
            self.by_context_mode[str(metrics["context_mode"])][key] += value
        self.by_negative_type[str(metrics["negative_type"])]["count"] += 1
        self.by_context_mode[str(metrics["context_mode"])]["count"] += 1

    def _averages(self, values: Dict[str, float]) -> Dict[str, float]:
        count = int(values.get("count", self.count))
        if count <= 0:
            return {}
        return {
            key: float(value) / count
            for key, value in values.items()
            if key != "count"
        } | {"count": count}

    def summary(self) -> Dict[str, Any]:
        global_values = dict(self.global_values)
        global_values["count"] = self.count
        return {
            "count": self.count,
            "global": self._averages(global_values),
            "by_negative_type": {
                key: self._averages(dict(values))
                for key, values in self.by_negative_type.items()
            },
            "by_context_mode": {
                key: self._averages(dict(values))
                for key, values in self.by_context_mode.items()
            },
            "skips": dict(self.skips),
            "action_counts": dict(self.action_counts),
        }
