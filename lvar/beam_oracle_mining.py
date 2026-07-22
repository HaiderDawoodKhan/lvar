"""Optional beam-search oracle mining for PATCH/THINK controller traces."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lvar.oracle_mining import OracleTraceMiner, preprocess_reasoning_steps
from lvar.utils import extract_tagged_answer


@dataclass
class BeamTrajectory:
    """Mutable search state for one oracle beam."""

    state: Dict[str, Any]
    trace: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    ce_rationale: float = 0.0
    ce_answer: float = 0.0
    stage_actions: List[Dict[str, Any]] = field(default_factory=list)
    stage_events: List[Dict[str, Any]] = field(default_factory=list)
    stage_attended_patch_indices: List[int] = field(default_factory=list)
    stage_initial_score: float = 0.0


class BeamOracleTraceMiner(OracleTraceMiner):
    """Mine up to ``beam_width`` PATCH/THINK-only oracle trajectories.

    The greedy ``OracleTraceMiner`` remains deliberately untouched.  This
    implementation uses full visual context because attention can only rank
    image-patch positions that are present in the current backbone prefix.
    """

    def __init__(
        self,
        model: Any,
        beam_width: int = 3,
        max_steps: int = 5,
        max_actions_per_stage: int = 4,
        patch_top_k: int = 32,
        ce_improvement_threshold: float = 0.01,
        rationale_ce_weight: float = 0.4,
        image_size: Optional[int] = 280,
    ) -> None:
        super().__init__(
            model=model,
            max_steps=max_steps,
            initial_visual_mode="full_context",
            image_size=image_size,
        )
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1.")
        if max_actions_per_stage < 1:
            raise ValueError("max_actions_per_stage must be >= 1.")
        if patch_top_k < 1:
            raise ValueError("patch_top_k must be >= 1.")
        if ce_improvement_threshold < 0.0:
            raise ValueError("ce_improvement_threshold must be non-negative.")
        if not 0.0 <= rationale_ce_weight <= 1.0:
            raise ValueError("rationale_ce_weight must be in [0, 1].")
        self.beam_width = int(beam_width)
        self.max_actions_per_stage = int(max_actions_per_stage)
        self.patch_top_k = int(patch_top_k)
        self.ce_improvement_threshold = float(ce_improvement_threshold)
        self.rationale_ce_weight = float(rationale_ce_weight)
        self.summary.update(
            {
                "beam_width": self.beam_width,
                "max_actions_per_stage": self.max_actions_per_stage,
                "patch_top_k": self.patch_top_k,
                "ce_improvement_threshold": self.ce_improvement_threshold,
                "rationale_ce_weight": self.rationale_ce_weight,
                "num_surviving_trajectories": 0,
                "num_pruned_trajectories": 0,
                "num_early_stopped_buckets": 0,
                "mean_weighted_ce": 0.0,
                "mean_rationale_ce": 0.0,
                "mean_answer_ce": 0.0,
            }
        )

    def _score_components(self, state: Dict[str, Any], rationale_text: str, answer_text: str) -> Tuple[float, float, float]:
        rationale_ce = self.score_state_ce(state, rationale_text) if rationale_text.strip() else 0.0
        answer_ce = self.score_state_ce(state, answer_text)
        score = self.rationale_ce_weight * rationale_ce + (1.0 - self.rationale_ce_weight) * answer_ce
        return float(rationale_ce), float(answer_ce), float(score)

    def _image_positions(self, prepared: Dict[str, Any], num_patches: int) -> List[int]:
        input_ids = prepared.get("input_ids")
        if input_ids is None:
            raise ValueError("Prepared inputs must include input_ids to locate image patches.")
        image_token_id = getattr(self.model, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Model does not expose image_token_id for attention-based patch ranking.")
        positions = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten().tolist()
        if len(positions) != num_patches:
            raise ValueError(
                f"Expected {num_patches} image placeholder positions, found {len(positions)}."
            )
        return [int(position) for position in positions]

    def _attended_patch_indices(
        self,
        trajectory: BeamTrajectory,
        bank: Dict[str, torch.Tensor],
        image_positions: Sequence[int],
    ) -> List[int]:
        """Rank patch keys by final-token attention, with a deterministic fallback."""
        state = trajectory.state
        kwargs = {"inputs_embeds": state["inputs_embeds"], "attention_mask": state["attention_mask"], "return_dict": True, "use_cache": False}
        if hasattr(self.model, "_state_position_kwargs"):
            kwargs.update(self.model._state_position_kwargs(state))
        try:
            with torch.no_grad():
                outputs = self.model.backbone(**kwargs, output_attentions=True)
            attentions = getattr(outputs, "attentions", None)
        except (TypeError, RuntimeError):
            attentions = None

        if attentions:
            valid_positions = [int(position) for position in image_positions if int(position) < state["inputs_embeds"].size(1)]
            if len(valid_positions) == int(bank["patches"].size(0)):
                position_tensor = torch.tensor(valid_positions, device=state["inputs_embeds"].device, dtype=torch.long)
                layers = []
                query_pos = state["inputs_embeds"].size(1) - 1
                for attention in attentions:
                    if attention is None or attention.size(-1) <= query_pos:
                        continue
                    layers.append(attention[0, :, query_pos, :].index_select(-1, position_tensor).mean(dim=0))
                if layers:
                    scores = torch.stack(layers, dim=0).mean(dim=0)
                    count = min(self.patch_top_k, int(scores.numel()))
                    return [int(index) for index in torch.topk(scores, k=count, largest=True, sorted=True).indices.cpu().tolist()]

        # Small test doubles and some attention backends do not expose attention
        # tensors.  Use the current final hidden state as a stable proxy instead.
        with torch.no_grad():
            outputs = self.model.backbone(**kwargs, output_hidden_states=True)
        final_hidden = self.model._extract_final_hidden(outputs) if hasattr(self.model, "_extract_final_hidden") else outputs.hidden_states[-1]
        query = final_hidden[:, -1, :]
        scores = torch.matmul(bank["patches"].to(query.dtype), query[0].to(bank["patches"].dtype))
        count = min(self.patch_top_k, int(scores.numel()))
        return [int(index) for index in torch.topk(scores, k=count, largest=True, sorted=True).indices.cpu().tolist()]

    def _clone_trajectory(self, trajectory: BeamTrajectory) -> BeamTrajectory:
        return BeamTrajectory(
            state=self.model.clone_state(trajectory.state),
            trace=copy.deepcopy(trajectory.trace),
            decisions=copy.deepcopy(trajectory.decisions),
            score=float(trajectory.score),
            ce_rationale=float(trajectory.ce_rationale),
            ce_answer=float(trajectory.ce_answer),
            stage_actions=copy.deepcopy(trajectory.stage_actions),
            stage_events=copy.deepcopy(trajectory.stage_events),
            stage_attended_patch_indices=list(trajectory.stage_attended_patch_indices),
            stage_initial_score=float(trajectory.stage_initial_score),
        )

    @staticmethod
    def _trajectory_key(trajectory: BeamTrajectory) -> Tuple[Tuple[str, Optional[int]], ...]:
        return tuple((str(action.get("type", "")).upper(), action.get("patch_idx")) for action in trajectory.trace + trajectory.stage_actions)

    def _keep_best(self, candidates: Sequence[BeamTrajectory]) -> List[BeamTrajectory]:
        unique: Dict[Tuple[Tuple[str, Optional[int]], ...], BeamTrajectory] = {}
        for candidate in candidates:
            key = self._trajectory_key(candidate)
            if key not in unique or candidate.score < unique[key].score:
                unique[key] = candidate
        ranked = sorted(unique.values(), key=lambda item: (item.score, self._trajectory_key(item)))
        self.summary["num_pruned_trajectories"] += max(0, len(ranked) - self.beam_width)
        return ranked[: self.beam_width]

    def _search_stage(
        self,
        beams: Sequence[BeamTrajectory],
        stage_idx: int,
        rationale_text: str,
        answer_text: str,
        bank: Dict[str, torch.Tensor],
        image_positions: Sequence[int],
    ) -> List[BeamTrajectory]:
        frontier = [self._clone_trajectory(beam) for beam in beams]
        for beam in frontier:
            # Each stage has a different teacher-forced future target.  Re-score
            # the unchanged prefix before comparing insertion improvements.
            rationale_ce, answer_ce, score = self._score_components(beam.state, rationale_text, answer_text)
            beam.ce_rationale = rationale_ce
            beam.ce_answer = answer_ce
            beam.score = score
            beam.stage_actions = []
            beam.stage_events = []
            beam.stage_attended_patch_indices = []
            beam.stage_initial_score = beam.score

        for _ in range(self.max_actions_per_stage):
            candidates = list(frontier)  # Retain the option to end this bucket now.
            expansions: List[BeamTrajectory] = []
            for beam in frontier:
                patches = self._attended_patch_indices(beam, bank, image_positions)
                beam.stage_attended_patch_indices = list(patches)
                actions = [{"type": "PATCH", "patch_idx": patch_idx} for patch_idx in patches] + [{"type": "THINK"}]
                best_improvement = float("-inf")
                for action in actions:
                    child = self._clone_trajectory(beam)
                    self.model.apply_mined_actions(child.state, bank, [action])
                    rationale_ce, answer_ce, score = self._score_components(child.state, rationale_text, answer_text)
                    improvement = beam.score - score
                    best_improvement = max(best_improvement, improvement)
                    if improvement < self.ce_improvement_threshold:
                        continue
                    child.score = score
                    child.ce_rationale = rationale_ce
                    child.ce_answer = answer_ce
                    child.stage_actions.append(copy.deepcopy(action))
                    child.stage_events.append(
                        {
                            "action": copy.deepcopy(action),
                            "attended_patch_indices": patches,
                            "ce_rationale": rationale_ce,
                            "ce_answer": answer_ce,
                            "weighted_ce": score,
                            "improvement": improvement,
                        }
                    )
                    expansions.append(child)
                if best_improvement < self.ce_improvement_threshold:
                    self.summary["num_early_stopped_buckets"] += 1
            if not expansions:
                break
            frontier = self._keep_best(candidates + expansions)

        finalized: List[BeamTrajectory] = []
        for beam in frontier:
            finalized_beam = self._clone_trajectory(beam)
            improvement = finalized_beam.stage_initial_score - finalized_beam.score
            finalized_beam.decisions.append(
                {
                    "step_idx": int(stage_idx),
                    "selected": "BEAM_BUCKET" if finalized_beam.stage_actions else "NO_OP",
                    "actions": copy.deepcopy(finalized_beam.stage_actions),
                    "ce_before": float(finalized_beam.stage_initial_score),
                    "ce_selected": float(finalized_beam.score),
                    "ce_rationale": float(finalized_beam.ce_rationale),
                    "ce_answer": float(finalized_beam.ce_answer),
                    "weighted_ce": float(finalized_beam.score),
                    "improvement": float(improvement),
                    "attended_patch_indices": list(finalized_beam.stage_attended_patch_indices),
                    "insertions": copy.deepcopy(finalized_beam.stage_events),
                }
            )
            finalized_beam.trace.extend(copy.deepcopy(finalized_beam.stage_actions))
            finalized_beam.stage_actions = []
            finalized_beam.stage_events = []
            finalized_beam.stage_attended_patch_indices = []
            finalized.append(finalized_beam)
        return self._keep_best(finalized)

    def mine_example(self, example: Dict[str, Any], negative_global_example_ids: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        del negative_global_example_ids
        prepared = self.model.prepare_inputs(example.get("image"), str(example.get("question") or ""), add_answer_instruction=False, image_size=self.image_size)
        image_tokens = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        bank = self.model.build_visual_bank(image_tokens)
        state = self.model.build_initial_state(prepared)
        image_positions = self._image_positions(prepared, int(bank["patches"].size(0)))
        steps = preprocess_reasoning_steps(example, max_steps=self.max_steps)
        answer = extract_tagged_answer(str(example.get("solution") or "")) or str(example.get("answer") or example.get("gold_answer") or "").strip()
        answer_text = f"Therefore, the answer is {answer}"
        initial_rationale = "".join(f"{step.rstrip()}\n" for step in steps)
        rationale_ce, answer_ce, score = self._score_components(state, initial_rationale, answer_text)
        beams = [BeamTrajectory(state=state, score=score, ce_rationale=rationale_ce, ce_answer=answer_ce)]

        for stage_idx in range(len(steps)):
            remaining_rationale = "".join(f"{step.rstrip()}\n" for step in steps[stage_idx + 1 :])
            beams = self._search_stage(beams, stage_idx, remaining_rationale, answer_text, bank, image_positions)

        beams = self._keep_best(beams)
        serialized = []
        for rank, beam in enumerate(beams, start=1):
            trace = copy.deepcopy(beam.trace) + [{"type": "STOP"}]
            serialized.append(
                {
                    "rank": rank,
                    "trace": trace,
                    "decisions": copy.deepcopy(beam.decisions),
                    "ce_rationale": float(beam.ce_rationale),
                    "ce_answer": float(beam.ce_answer),
                    "weighted_ce": float(beam.score),
                }
            )
            self.summary["num_surviving_trajectories"] += 1
            for action in beam.trace:
                action_type = str(action.get("type", "")).upper()
                counts = self.summary["action_counts"]
                counts[action_type] = int(counts.get(action_type, 0)) + 1
        best = serialized[0]
        self.summary["num_examples"] += 1
        count = int(self.summary["num_examples"])
        for key, value in (("mean_weighted_ce", best["weighted_ce"]), ("mean_rationale_ce", best["ce_rationale"]), ("mean_answer_ce", best["ce_answer"])):
            self.summary[key] += (float(value) - float(self.summary[key])) / count
        return {
            "example_id": example.get("id"),
            "initial_visual_mode": "full_context",
            "question": str(example.get("question") or ""),
            "answer": answer,
            "steps": steps,
            "trace": copy.deepcopy(best["trace"]),
            "decisions": copy.deepcopy(best["decisions"]),
            "beam_trajectories": serialized,
            "counterfactual_pairs": [],
        }


def summarize_beam_trace_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Return lightweight trajectory/action counts for persisted beam rows."""
    row_list = list(rows)
    return {
        "num_examples": len(row_list),
        "num_trajectories": sum(len(row.get("beam_trajectories") or []) for row in row_list),
        "num_patch_actions": sum(
            sum(str(action.get("type", "")).upper() == "PATCH" for action in trajectory.get("trace") or [])
            for row in row_list for trajectory in row.get("beam_trajectories") or []
        ),
        "num_think_actions": sum(
            sum(str(action.get("type", "")).upper() == "THINK" for action in trajectory.get("trace") or [])
            for row in row_list for trajectory in row.get("beam_trajectories") or []
        ),
    }
