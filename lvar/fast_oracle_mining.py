"""Rank-once fast variants of the PATCH/THINK oracle search."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from lvar.beam_oracle_mining import BeamOracleTraceMiner, BeamTrajectory


@dataclass
class RankedAction:
    """One action scored once from a stage's unmodified parent state."""

    action: Dict[str, Any]
    ce_rationale: float
    ce_answer: float
    weighted_ce: float
    improvement: float
    rank: int = 0


@dataclass
class FastStageNode:
    """One started or no-op stage trajectory plus its fixed proposal ranking."""

    trajectory: BeamTrajectory
    ranking: List[RankedAction]
    attempted_keys: set[tuple[str, int | None]]
    proposals_tried: int
    started: bool


class FastOracleTraceMiner(BeamOracleTraceMiner):
    """One-shot candidate ranking with thresholded cumulative-prefix checks.

    Every active beam scores the attended patches plus THINK once at the start
    of a reasoning stage.  Subsequent insertions only test prefixes drawn from
    that fixed ranking; they never re-score the entire action set.
    """

    def __init__(self, *args: Any, proposal_count: int = 6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if proposal_count < 1:
            raise ValueError("proposal_count must be >= 1.")
        self.proposal_count = int(proposal_count)
        self.summary.update(
            {
                "search_mode": "rank_once_fast",
                "proposal_count": self.proposal_count,
                "num_initial_candidate_scores": 0,
                "num_prefix_scores": 0,
                "num_rejected_proposals": 0,
            }
        )

    @staticmethod
    def _action_key(action: Dict[str, Any]) -> tuple[str, int | None]:
        return str(action.get("type", "")).upper(), action.get("patch_idx")

    def _prepare_stage(
        self,
        beam: BeamTrajectory,
        rationale_text: str,
        answer_text: str,
    ) -> BeamTrajectory:
        prepared = self._clone_trajectory(beam)
        rationale_ce, answer_ce, score = self._score_components(prepared.state, rationale_text, answer_text)
        prepared.score = score
        prepared.ce_rationale = rationale_ce
        prepared.ce_answer = answer_ce
        prepared.stage_actions = []
        prepared.stage_events = []
        prepared.stage_attended_patch_indices = []
        prepared.stage_initial_score = score
        return prepared

    def _rank_actions_once(
        self,
        beam: BeamTrajectory,
        rationale_text: str,
        answer_text: str,
        bank: Dict[str, torch.Tensor],
        image_positions: Sequence[int],
    ) -> List[RankedAction]:
        patches = self._attended_patch_indices(beam, bank, image_positions)
        beam.stage_attended_patch_indices = list(patches)
        actions = [{"type": "PATCH", "patch_idx": patch_idx} for patch_idx in patches] + [{"type": "THINK"}]
        ranking: List[RankedAction] = []
        for action in actions:
            candidate_state = self.model.clone_state(beam.state)
            self.model.apply_mined_actions(candidate_state, bank, [action])
            rationale_ce, answer_ce, score = self._score_components(candidate_state, rationale_text, answer_text)
            self.summary["num_initial_candidate_scores"] += 1
            ranking.append(
                RankedAction(
                    action=copy.deepcopy(action),
                    ce_rationale=rationale_ce,
                    ce_answer=answer_ce,
                    weighted_ce=score,
                    improvement=beam.score - score,
                )
            )
        ranking.sort(key=lambda item: (item.weighted_ce, self._action_key(item.action)))
        for rank, item in enumerate(ranking, start=1):
            item.rank = rank
        return ranking

    def _seed_node(
        self,
        parent: BeamTrajectory,
        ranking: List[RankedAction],
        first: RankedAction,
        bank: Dict[str, torch.Tensor],
    ) -> FastStageNode:
        child = self._clone_trajectory(parent)
        self.model.apply_mined_actions(child.state, bank, [first.action])
        child.score = first.weighted_ce
        child.ce_rationale = first.ce_rationale
        child.ce_answer = first.ce_answer
        child.stage_actions.append(copy.deepcopy(first.action))
        child.stage_events.append(
            {
                "action": copy.deepcopy(first.action),
                "proposal_rank": first.rank,
                "attended_patch_indices": list(parent.stage_attended_patch_indices),
                "ce_rationale": first.ce_rationale,
                "ce_answer": first.ce_answer,
                "weighted_ce": first.weighted_ce,
                "improvement": first.improvement,
            }
        )
        return FastStageNode(
            trajectory=child,
            ranking=ranking,
            attempted_keys={self._action_key(first.action)},
            proposals_tried=1,
            started=True,
        )

    def _keep_best_nodes(self, nodes: Sequence[FastStageNode]) -> List[FastStageNode]:
        unique: Dict[tuple[tuple[str, int | None], ...], FastStageNode] = {}
        for node in nodes:
            key = self._trajectory_key(node.trajectory)
            current = unique.get(key)
            if current is None or node.trajectory.score < current.trajectory.score:
                unique[key] = node
        ranked = sorted(unique.values(), key=lambda node: (node.trajectory.score, self._trajectory_key(node.trajectory)))
        self.summary["num_pruned_trajectories"] += max(0, len(ranked) - self.beam_width)
        return ranked[: self.beam_width]

    def _append_fixed_ranked_proposals(
        self,
        node: FastStageNode,
        rationale_text: str,
        answer_text: str,
        bank: Dict[str, torch.Tensor],
    ) -> None:
        """Accept/reject remaining top proposals without changing their ranking."""
        beam = node.trajectory
        for proposal in node.ranking:
            if len(beam.stage_actions) >= self.max_actions_per_stage:
                break
            if node.proposals_tried >= self.proposal_count:
                break
            key = self._action_key(proposal.action)
            if key in node.attempted_keys:
                continue
            node.attempted_keys.add(key)
            node.proposals_tried += 1
            candidate_state = self.model.clone_state(beam.state)
            self.model.apply_mined_actions(candidate_state, bank, [proposal.action])
            rationale_ce, answer_ce, score = self._score_components(candidate_state, rationale_text, answer_text)
            self.summary["num_prefix_scores"] += 1
            improvement = beam.score - score
            if improvement < self.ce_improvement_threshold:
                self.summary["num_rejected_proposals"] += 1
                continue
            beam.state = candidate_state
            beam.score = score
            beam.ce_rationale = rationale_ce
            beam.ce_answer = answer_ce
            beam.stage_actions.append(copy.deepcopy(proposal.action))
            beam.stage_events.append(
                {
                    "action": copy.deepcopy(proposal.action),
                    "proposal_rank": proposal.rank,
                    "attended_patch_indices": list(beam.stage_attended_patch_indices),
                    "ce_rationale": rationale_ce,
                    "ce_answer": answer_ce,
                    "weighted_ce": score,
                    "improvement": improvement,
                }
            )
        if len(beam.stage_actions) < self.max_actions_per_stage and node.proposals_tried >= self.proposal_count:
            self.summary["num_early_stopped_buckets"] += 1

    def _finalize_stage(self, node: FastStageNode, stage_idx: int) -> BeamTrajectory:
        beam = self._clone_trajectory(node.trajectory)
        improvement = beam.stage_initial_score - beam.score
        ranked_candidates = [
            {
                "action": copy.deepcopy(item.action),
                "proposal_rank": item.rank,
                "ce_rationale": item.ce_rationale,
                "ce_answer": item.ce_answer,
                "weighted_ce": item.weighted_ce,
                "improvement": item.improvement,
            }
            for item in node.ranking[: self.proposal_count]
        ]
        beam.decisions.append(
            {
                "step_idx": int(stage_idx),
                "selected": "FAST_BUCKET" if beam.stage_actions else "NO_OP",
                "actions": copy.deepcopy(beam.stage_actions),
                "ce_before": float(beam.stage_initial_score),
                "ce_selected": float(beam.score),
                "ce_rationale": float(beam.ce_rationale),
                "ce_answer": float(beam.ce_answer),
                "weighted_ce": float(beam.score),
                "improvement": float(improvement),
                "attended_patch_indices": list(beam.stage_attended_patch_indices),
                "ranked_candidates": ranked_candidates,
                "insertions": copy.deepcopy(beam.stage_events),
            }
        )
        beam.trace.extend(copy.deepcopy(beam.stage_actions))
        beam.stage_actions = []
        beam.stage_events = []
        beam.stage_attended_patch_indices = []
        return beam

    def _search_stage(
        self,
        beams: Sequence[BeamTrajectory],
        stage_idx: int,
        rationale_text: str,
        answer_text: str,
        bank: Dict[str, torch.Tensor],
        image_positions: Sequence[int],
    ) -> List[BeamTrajectory]:
        prepared = [self._prepare_stage(beam, rationale_text, answer_text) for beam in beams]
        seed_nodes: List[FastStageNode] = []
        for parent in prepared:
            ranking = self._rank_actions_once(parent, rationale_text, answer_text, bank, image_positions)
            proposals = ranking[: self.proposal_count]
            # Retain a no-op path, just as the adaptive beam search does.
            seed_nodes.append(FastStageNode(parent, ranking, set(), 0, False))
            for first in proposals:
                if first.improvement >= self.ce_improvement_threshold:
                    seed_nodes.append(self._seed_node(parent, ranking, first, bank))

        survivors = self._keep_best_nodes(seed_nodes)
        for node in survivors:
            if node.started:
                self._append_fixed_ranked_proposals(node, rationale_text, answer_text, bank)
            elif not node.ranking or node.ranking[0].improvement < self.ce_improvement_threshold:
                self.summary["num_early_stopped_buckets"] += 1
        finalized = [self._finalize_stage(node, stage_idx) for node in survivors]
        return self._keep_best(finalized)


class GreedyFastOracleTraceMiner(FastOracleTraceMiner):
    """Fast rank-once oracle search with exactly one surviving trajectory."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(model=model, beam_width=1, **kwargs)
        self.summary["search_strategy"] = "greedy_fast"


class BeamSearchFastOracleTraceMiner(FastOracleTraceMiner):
    """Fast rank-once oracle search retaining multiple first-action beams."""

    def __init__(self, model: Any, beam_width: int = 3, **kwargs: Any) -> None:
        super().__init__(model=model, beam_width=beam_width, **kwargs)
        self.summary["search_strategy"] = "beam_search_fast"
