import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lvar.oracle_mining import preprocess_reasoning_steps
from lvar.utils import extract_tagged_answer


def select_top_k_patches(
    step_hidden: torch.Tensor,
    patch_tokens: torch.Tensor,
    top_k: int = 3,
) -> Tuple[List[int], List[float]]:
    """Rank projected patch vectors by cosine similarity to one step hidden state."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    patches = patch_tokens.squeeze(0) if patch_tokens.dim() == 3 else patch_tokens
    if patches.dim() != 2:
        raise ValueError("patch_tokens must have shape [num_patches, hidden_size].")
    if patches.size(0) < top_k:
        raise ValueError(f"Patch bank contains {patches.size(0)} patches, fewer than requested top_k={top_k}.")
    hidden = step_hidden.squeeze()
    if hidden.dim() != 1 or hidden.size(0) != patches.size(-1):
        raise ValueError(
            "step_hidden and patch_tokens must share hidden size; "
            f"got {tuple(hidden.shape)} and {tuple(patches.shape)}."
        )

    similarities = F.cosine_similarity(
        patches.float(),
        hidden.float().unsqueeze(0),
        dim=-1,
    )
    scores, indices = torch.topk(similarities, k=top_k, largest=True, sorted=True)
    return (
        [int(index) for index in indices.detach().cpu().tolist()],
        [float(score) for score in scores.detach().cpu().tolist()],
    )


def summarize_cosine_trace_rows(
    rows: Iterable[Dict[str, Any]],
    top_k: int = 3,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """Build cumulative mining counts from all rows currently stored in a trace JSONL."""
    row_list = list(rows)
    decisions = [decision for row in row_list for decision in (row.get("decisions") or [])]
    actions = [action for row in row_list for action in (row.get("trace") or [])]
    cosine_scores = [
        float(score)
        for decision in decisions
        for score in (decision.get("cosine_similarities") or [])
    ]
    return {
        "mining_method": "step_hidden_patch_cosine",
        "top_k": int(top_k),
        "max_steps": int(max_steps),
        "initial_visual_mode": "full_context",
        "num_examples": len(row_list),
        "num_decisions": len(decisions),
        "num_patch_actions": sum(str(action.get("type", "")).upper() == "PATCH" for action in actions),
        "num_think_actions": sum(str(action.get("type", "")).upper() == "THINK" for action in actions),
        "mean_selected_cosine_similarity": (
            sum(cosine_scores) / len(cosine_scores) if cosine_scores else None
        ),
    }


class CosineSimilarityTraceMiner:
    """Mine fixed top-k patch traces from cumulative gold-rationale hidden states."""

    def __init__(
        self,
        model: Any,
        top_k: int = 3,
        max_steps: int = 8,
        image_size: Optional[int] = 280,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        self.model = model
        self.top_k = int(top_k)
        self.max_steps = int(max_steps)
        self.image_size = image_size
        self.summary: Dict[str, Any] = {
            "mining_method": "step_hidden_patch_cosine",
            "top_k": self.top_k,
            "max_steps": self.max_steps,
            "initial_visual_mode": "full_context",
            "num_examples": 0,
            "num_decisions": 0,
            "num_patch_actions": 0,
            "num_think_actions": 0,
            "cosine_similarity_sum": 0.0,
            "cosine_similarity_count": 0,
        }

    def _tokenize_step(self, step: str) -> torch.Tensor:
        tokenizer = getattr(self.model.processor, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("The processor must expose a callable tokenizer for rationale steps.")
        text = str(step).strip()
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None or input_ids.numel() == 0:
            raise ValueError("A rationale step tokenized to an empty sequence.")
        return input_ids.to(device=self.model.device, dtype=torch.long)

    def _append_embeddings(
        self,
        state: Dict[str, Any],
        embeddings: torch.Tensor,
        track_as_visual: bool = False,
    ) -> None:
        tokens = embeddings
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 3 or tokens.size(0) != state["inputs_embeds"].size(0):
            raise ValueError("Appended embeddings must have shape [batch, tokens, hidden_size].")
        start = int(state["inputs_embeds"].size(1))
        tokens = tokens.to(device=self.model.device, dtype=state["inputs_embeds"].dtype)
        state["inputs_embeds"] = torch.cat([state["inputs_embeds"], tokens], dim=1)
        new_mask = torch.ones(
            (state["attention_mask"].size(0), tokens.size(1)),
            device=self.model.device,
            dtype=state["attention_mask"].dtype,
        )
        state["attention_mask"] = torch.cat([state["attention_mask"], new_mask], dim=1)
        self.model._append_position_ids(state, num_tokens=int(tokens.size(1)))
        if track_as_visual:
            positions = list(range(start, start + int(tokens.size(1))))
            state.setdefault("trace_all_positions", []).extend(positions)
            state.setdefault("trace_visual_positions", []).extend(positions)

    def _append_rationale_step(self, state: Dict[str, Any], step: str) -> int:
        input_ids = self._tokenize_step(step)
        step_embeddings = self.model._embed_input_ids(input_ids)
        self._append_embeddings(state, step_embeddings)
        return int(input_ids.size(1))

    def _read_last_hidden(self, state: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model.backbone(
                inputs_embeds=state["inputs_embeds"],
                attention_mask=state["attention_mask"],
                **self.model._state_position_kwargs(state),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return self.model._extract_final_hidden(outputs)[:, -1, :]

    def mine_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prepared = self.model.prepare_inputs(
            example.get("image"),
            str(example.get("question") or ""),
            add_answer_instruction=False,
            image_size=self.image_size,
        )
        image_tokens = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        bank = self.model.build_visual_bank(image_tokens)
        if int(bank["patches"].size(0)) < self.top_k:
            raise ValueError(
                f"Patch bank contains {bank['patches'].size(0)} patches, fewer than requested top_k={self.top_k}."
            )
        state = self.model.build_initial_state(prepared)

        steps = preprocess_reasoning_steps(example, max_steps=self.max_steps)
        answer = extract_tagged_answer(str(example.get("solution") or ""))
        if not answer:
            answer = str(example.get("answer") or example.get("gold_answer") or "").strip()

        trace: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        for step_idx, step in enumerate(steps):
            before_step = int(state["inputs_embeds"].size(1))
            step_token_count = self._append_rationale_step(state, step)
            after_step = int(state["inputs_embeds"].size(1))
            step_hidden = self._read_last_hidden(state)
            patch_indices, cosine_scores = select_top_k_patches(
                step_hidden,
                bank["patches"],
                top_k=self.top_k,
            )
            patch_actions = [
                {
                    "type": "PATCH",
                    "patch_idx": patch_idx,
                    "cosine_similarity": cosine_score,
                }
                for patch_idx, cosine_score in zip(patch_indices, cosine_scores)
            ]
            actions = patch_actions + [{"type": "THINK"}]

            selected_patches = bank["patches"].index_select(
                0,
                torch.tensor(patch_indices, device=bank["patches"].device, dtype=torch.long),
            )
            self._append_embeddings(state, selected_patches, track_as_visual=True)
            think_hidden = self._read_last_hidden(state)
            self.model._append_hidden_token(state, think_hidden, track_as_think=True)
            after_actions = int(state["inputs_embeds"].size(1))

            decisions.append(
                {
                    "step_idx": int(step_idx),
                    "selected": "PATCH_SEQ_THINK",
                    "actions": copy.deepcopy(actions),
                    "patch_indices": patch_indices,
                    "cosine_similarities": cosine_scores,
                    "step_token_count": step_token_count,
                    "sequence_length_before_step": before_step,
                    "sequence_length_after_step": after_step,
                    "sequence_length_after_actions": after_actions,
                }
            )
            trace.extend(copy.deepcopy(actions))
            self.summary["num_decisions"] += 1
            self.summary["num_patch_actions"] += self.top_k
            self.summary["num_think_actions"] += 1
            self.summary["cosine_similarity_sum"] += sum(cosine_scores)
            self.summary["cosine_similarity_count"] += len(cosine_scores)

        trace.append({"type": "STOP"})
        self.summary["num_examples"] += 1
        return {
            "example_id": example.get("id"),
            "mining_method": "step_hidden_patch_cosine",
            "initial_visual_mode": "full_context",
            "top_k": self.top_k,
            "max_steps": self.max_steps,
            "question": str(example.get("question") or ""),
            "answer": answer,
            "steps": steps,
            "trace": trace,
            "decisions": decisions,
            "counterfactual_pairs": [],
        }

    def get_summary(self) -> Dict[str, Any]:
        summary = copy.deepcopy(self.summary)
        count = int(summary.pop("cosine_similarity_count"))
        total = float(summary.pop("cosine_similarity_sum"))
        summary["mean_selected_cosine_similarity"] = total / count if count else None
        return summary
