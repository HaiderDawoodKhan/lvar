import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from lvar.utils import extract_tagged_answer


ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


@dataclass
class Candidate:
    """One scored oracle action candidate."""

    name: str
    actions: List[Dict[str, Any]]
    ce: float


def split_rationale_into_sentences(text: str) -> List[str]:
    """Split an M3CoT rationale into sentence-like blocks."""
    cleaned = ANSWER_TAG_RE.sub("", text or "").strip()
    if not cleaned:
        return []
    normalized = re.sub(r"\s+", " ", cleaned)
    units = re.split(r"(?<=[.!?])\s+", normalized)
    units = [unit.strip() for unit in units if unit.strip()]
    if len(units) <= 1:
        units = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return units


def group_steps_to_max(units: Sequence[str], max_steps: int) -> List[str]:
    """Group sentence steps evenly so the result has at most max_steps blocks."""
    if not units:
        return []
    if len(units) <= max_steps:
        return [str(unit).strip() for unit in units if str(unit).strip()]
    num_blocks = max(1, min(max_steps, len(units)))
    merged = []
    for block_idx in range(num_blocks):
        start = math.floor(block_idx * len(units) / num_blocks)
        end = math.floor((block_idx + 1) * len(units) / num_blocks)
        merged.append(" ".join(units[start:end]).strip())
    return [block for block in merged if block]


def preprocess_reasoning_steps(
    example: Dict[str, Any],
    max_steps: int = 8,
) -> List[str]:
    """Match the Phase 1 M3CoT rationale sentence splitting and max-step grouping."""
    explicit_steps = example.get("steps")
    if explicit_steps:
        steps = [str(step).strip() for step in explicit_steps if str(step).strip()]
        return group_steps_to_max(steps, max_steps)

    rationale = str(example.get("rationale") or "")
    if not rationale:
        solution = str(example.get("solution") or "")
        rationale = ANSWER_TAG_RE.sub("", solution).strip()
    units = split_rationale_into_sentences(rationale)
    return group_steps_to_max(units, max_steps)


def build_step_target(steps: Sequence[str], step_idx: int, answer: str) -> str:
    """Build Y_t using the same explicit CoT + final answer text as Phase 1."""
    future_steps = "".join(f"{step.rstrip()}\n" for step in steps[step_idx + 1 :] if step)
    return f"{future_steps}Therefore, the answer is {answer}"


def contains_visual_action(actions: Sequence[Dict[str, Any]]) -> bool:
    return any(action.get("type") in {"GLOBAL", "REGION", "PATCH"} for action in actions)


class OracleTraceMiner:
    """Mine Phase 2 supervised controller traces from fixed reasoning targets."""

    def __init__(
        self,
        model: Any,
        selection_delta: float = 0.03,
        patch_k_choices: Optional[Sequence[int]] = None,
        max_steps: int = 8,
        rng: Optional[random.Random] = None,
        initial_visual_mode: str = "global_mean",
        image_size: Optional[int] = 280,
        counterfactual_negative_source: str = "same_image",
    ) -> None:
        self.model = model
        self.selection_delta = float(selection_delta)
        self.patch_k_choices = list(patch_k_choices or [1, 2, 3, 4])
        self.max_steps = int(max_steps)
        self.rng = rng or random.Random()
        visual_mode = str(initial_visual_mode).strip().lower()
        if visual_mode in {"global", "full", "full_image"}:
            visual_mode = "full_context"
        if visual_mode in {"coarse", "coarse_context", "global_token"}:
            visual_mode = "global_mean"
        if visual_mode not in {"global_mean", "full_context"}:
            raise ValueError("initial_visual_mode must be one of: global_mean, full_context.")
        self.initial_visual_mode = visual_mode
        self.image_size = image_size
        negative_source = str(counterfactual_negative_source).strip().lower()
        if negative_source not in {"same_image", "different_image"}:
            raise ValueError("counterfactual_negative_source must be 'same_image' or 'different_image'.")
        self.counterfactual_negative_source = negative_source
        self.summary = {
            "num_examples": 0,
            "num_decisions": 0,
            "mean_selected_improvement": 0.0,
            "action_counts": {},
            "counterfactual_pairs": 0,
            "counterfactual_skips": {},
        }

    def _tokenize_target(self, text: str) -> torch.Tensor:
        tokenizer = getattr(self.model.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("The processor must expose a tokenizer for CE scoring.")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos = torch.tensor([[int(eos_token_id)]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, eos], dim=1)
        return input_ids.to(self.model.device)

    def score_state_ce(self, state: Dict[str, Any], target_text: str) -> float:
        """Compute CE(target_text | state) over target tokens only."""
        target_ids = self._tokenize_target(target_text)
        if target_ids.numel() == 0:
            raise ValueError("Cannot score an empty target.")
        target_embeds = self.model._embed_input_ids(target_ids)
        prefix_embeds = state["inputs_embeds"]
        prefix_mask = state["attention_mask"]
        prefix_len = prefix_embeds.size(1)
        if target_ids.size(1) > 1:
            input_embeds = torch.cat([prefix_embeds, target_embeds[:, :-1, :]], dim=1)
            target_mask = torch.ones(
                (prefix_mask.size(0), target_ids.size(1) - 1),
                device=self.model.device,
                dtype=prefix_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
        else:
            input_embeds = prefix_embeds
            attention_mask = prefix_mask

        with torch.no_grad():
            outputs = self.model.backbone(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            logits = outputs.logits[:, prefix_len - 1 : prefix_len - 1 + target_ids.size(1), :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return float(loss.detach().cpu().item())

    def score_actions(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        target_text: str,
        name: str,
        actions: Sequence[Dict[str, Any]],
    ) -> Candidate:
        candidate_state = self.model.clone_state(state)
        self.model.apply_mined_actions(candidate_state, bank, list(actions))
        return Candidate(name=name, actions=copy.deepcopy(list(actions)), ce=self.score_state_ce(candidate_state, target_text))

    def _best_global_candidate(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        target_text: str,
    ) -> Candidate:
        global_only = self.score_actions(state, bank, target_text, "GLOBAL", [{"type": "GLOBAL"}])
        global_think = self.score_actions(
            state,
            bank,
            target_text,
            "GLOBAL_THINK",
            [{"type": "GLOBAL"}, {"type": "THINK"}],
        )
        return min([global_only, global_think], key=lambda candidate: candidate.ce)

    def _best_region_candidate(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        target_text: str,
    ) -> Candidate:
        best_region = None
        for region_idx in range(int(bank["raw_regions"].size(0))):
            candidate = self.score_actions(
                state,
                bank,
                target_text,
                "REGION",
                [{"type": "REGION", "region_idx": region_idx}],
            )
            if best_region is None or candidate.ce < best_region.ce:
                best_region = candidate
        if best_region is None:
            raise ValueError("Visual bank contains no regions.")
        region_think = self.score_actions(
            state,
            bank,
            target_text,
            "REGION_THINK",
            best_region.actions + [{"type": "THINK"}],
        )
        return min([best_region, region_think], key=lambda candidate: candidate.ce)

    def _best_patch_candidate(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        target_text: str,
    ) -> Candidate:
        patch_scores = []
        for patch_idx in range(int(bank["patches"].size(0))):
            candidate = self.score_actions(
                state,
                bank,
                target_text,
                "PATCH",
                [{"type": "PATCH", "patch_idx": patch_idx}],
            )
            patch_scores.append(candidate)
        if not patch_scores:
            raise ValueError("Visual bank contains no patches.")
        ranked = sorted(patch_scores, key=lambda candidate: candidate.ce)
        k = int(self.rng.choice(self.patch_k_choices))
        k = max(1, min(k, len(ranked)))
        patch_actions = [copy.deepcopy(candidate.actions[0]) for candidate in ranked[:k]]
        patch_seq = self.score_actions(state, bank, target_text, "PATCH_SEQ", patch_actions)
        patch_seq_think = self.score_actions(
            state,
            bank,
            target_text,
            "PATCH_SEQ_THINK",
            patch_actions + [{"type": "THINK"}],
        )
        return min([patch_seq, patch_seq_think], key=lambda candidate: candidate.ce)

    def score_step_candidates(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        target_text: str,
    ) -> Dict[str, Candidate]:
        ce_noop = self.score_state_ce(state, target_text)
        candidates = {
            "NO_OP": Candidate("NO_OP", [], ce_noop),
            "THINK": self.score_actions(state, bank, target_text, "THINK", [{"type": "THINK"}]),
            "GLOBAL": self._best_global_candidate(state, bank, target_text),
            "REGION": self._best_region_candidate(state, bank, target_text),
            "PATCH": self._best_patch_candidate(state, bank, target_text),
        }
        return candidates

    def select_candidate(self, candidates: Dict[str, Candidate]) -> Candidate:
        noop = candidates["NO_OP"]
        best_nonempty = min(
            [candidate for name, candidate in candidates.items() if name != "NO_OP"],
            key=lambda candidate: candidate.ce,
        )
        if noop.ce - best_nonempty.ce > self.selection_delta:
            return best_nonempty
        return noop

    def build_counterfactual_pair(
        self,
        example_id: Any,
        step_idx: int,
        prefix_trace: Sequence[Dict[str, Any]],
        positive_actions: Sequence[Dict[str, Any]],
        target_text: str,
        bank: Dict[str, torch.Tensor],
        negative_global_example_ids: Optional[Sequence[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not contains_visual_action(positive_actions):
            return None
        negative_actions = []
        positive_patch_indices = {
            int(action["patch_idx"]) for action in positive_actions if action.get("type") == "PATCH"
        }
        patch_pool = [idx for idx in range(int(bank["patches"].size(0))) if idx not in positive_patch_indices]
        used_negative_patches = set()
        cross_image_pool = [candidate_id for candidate_id in (negative_global_example_ids or []) if candidate_id != example_id]

        for action in positive_actions:
            action_type = action.get("type")
            if action_type == "PATCH":
                if self.counterfactual_negative_source == "different_image":
                    if not cross_image_pool:
                        self._record_counterfactual_skip("patch_no_negative_example")
                        return None
                    negative_actions.append(
                        {
                            "type": "PATCH",
                            "patch_idx": int(action["patch_idx"]),
                            "source_example_id": self.rng.choice(list(cross_image_pool)),
                        }
                    )
                    continue
                available = [idx for idx in patch_pool if idx not in used_negative_patches]
                if not available:
                    available = patch_pool
                if not available:
                    self._record_counterfactual_skip("patch_no_negative")
                    return None
                wrong_idx = int(self.rng.choice(available))
                used_negative_patches.add(wrong_idx)
                negative_actions.append({"type": "PATCH", "patch_idx": wrong_idx})
            elif action_type == "REGION":
                if self.counterfactual_negative_source == "different_image":
                    if not cross_image_pool:
                        self._record_counterfactual_skip("region_no_negative_example")
                        return None
                    negative_actions.append(
                        {
                            "type": "REGION",
                            "region_idx": int(action["region_idx"]),
                            "source_example_id": self.rng.choice(list(cross_image_pool)),
                        }
                    )
                    continue
                pool = [idx for idx in range(int(bank["raw_regions"].size(0))) if idx != int(action["region_idx"])]
                if not pool:
                    self._record_counterfactual_skip("region_no_negative")
                    return None
                negative_actions.append({"type": "REGION", "region_idx": int(self.rng.choice(pool))})
            elif action_type == "GLOBAL":
                if not cross_image_pool:
                    self._record_counterfactual_skip("global_no_negative_example")
                    return None
                negative_actions.append({"type": "GLOBAL", "source_example_id": self.rng.choice(list(cross_image_pool))})
            else:
                negative_actions.append(copy.deepcopy(action))

        return {
            "step_idx": int(step_idx),
            "prefix_trace": copy.deepcopy(list(prefix_trace)),
            "positive_actions": copy.deepcopy(list(positive_actions)),
            "negative_actions": negative_actions,
            "target_text": target_text,
        }

    def _record_counterfactual_skip(self, reason: str) -> None:
        skips = self.summary["counterfactual_skips"]
        skips[reason] = int(skips.get(reason, 0)) + 1

    def _record_decision(self, selected: Candidate, improvement: float) -> None:
        self.summary["num_decisions"] += 1
        counts = self.summary["action_counts"]
        counts[selected.name] = int(counts.get(selected.name, 0)) + 1
        old_mean = float(self.summary["mean_selected_improvement"])
        count = int(self.summary["num_decisions"])
        self.summary["mean_selected_improvement"] = old_mean + (float(improvement) - old_mean) / count

    def mine_example(
        self,
        example: Dict[str, Any],
        negative_global_example_ids: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        image = example.get("image")
        question = str(example.get("question") or "")
        prepared = self.model.prepare_inputs(
            image,
            question,
            add_answer_instruction=False,
            image_size=self.image_size,
        )
        image_tokens = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        bank = self.model.build_visual_bank(image_tokens)
        if self.initial_visual_mode == "full_context":
            state = self.model.build_initial_state(prepared)
        else:
            state = self.model.build_coarse_initial_state(prepared, bank)

        steps = preprocess_reasoning_steps(example, max_steps=self.max_steps)
        answer = extract_tagged_answer(str(example.get("solution") or ""))
        if not answer:
            answer = str(example.get("answer") or example.get("gold_answer") or "").strip()

        trace: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        counterfactual_pairs: List[Dict[str, Any]] = []

        for step_idx in range(len(steps)):
            target_text = build_step_target(steps, step_idx, answer)
            prefix_trace = copy.deepcopy(trace)
            candidates = self.score_step_candidates(state, bank, target_text)
            selected = self.select_candidate(candidates)
            ce_noop = candidates["NO_OP"].ce
            improvement = ce_noop - selected.ce
            decisions.append(
                {
                    "step_idx": int(step_idx),
                    "selected": selected.name,
                    "actions": copy.deepcopy(selected.actions),
                    "ce_noop": float(ce_noop),
                    "ce_selected": float(selected.ce),
                    "improvement": float(improvement),
                }
            )
            self._record_decision(selected, improvement)

            pair = self.build_counterfactual_pair(
                example.get("id"),
                step_idx,
                prefix_trace,
                selected.actions,
                target_text,
                bank,
                negative_global_example_ids=negative_global_example_ids,
            )
            if pair is not None:
                counterfactual_pairs.append(pair)
                self.summary["counterfactual_pairs"] = int(self.summary["counterfactual_pairs"]) + 1

            if selected.actions:
                self.model.apply_mined_actions(state, bank, selected.actions)
                trace.extend(copy.deepcopy(selected.actions))

        trace.append({"type": "STOP"})
        self.summary["num_examples"] = int(self.summary["num_examples"]) + 1
        return {
            "example_id": example.get("id"),
            "initial_visual_mode": self.initial_visual_mode,
            "question": question,
            "answer": answer,
            "steps": steps,
            "trace": trace,
            "decisions": decisions,
            "counterfactual_pairs": counterfactual_pairs,
        }

    def get_summary(self) -> Dict[str, Any]:
        summary = copy.deepcopy(self.summary)
        summary["initial_visual_mode"] = self.initial_visual_mode
        summary["counterfactual_negative_source"] = self.counterfactual_negative_source
        return summary


def action_type_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for action in row.get("trace", []):
            action_type = str(action.get("type"))
            counts[action_type] = counts.get(action_type, 0) + 1
    return counts
