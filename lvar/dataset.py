import random
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from lvar.oracle_mining import group_steps_to_max, split_rationale_into_sentences
from lvar.utils import extract_tagged_answer, normalize_answer_text

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised in environments without HF deps
    load_dataset = None


class CLEVRCoGenTDataset(Dataset):
    """Thin Dataset wrapper that exposes CLEVR CoGenT rows in LVAR-ready format."""

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        dataset_name: str = "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex",
        partition: str = "all",
        test_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        """
        Load a Hugging Face split and optionally truncate it for quick experiments.

        Attributes:
            self.dataset: The underlying HF dataset split used for indexing.
        """
        if load_dataset is None:
            raise ImportError(
                "datasets is required to load MMInstruction/Clevr_CoGenT_TrainA_70K_Complex. "
                "Install the requirements first."
            )
        self.dataset = load_dataset(dataset_name, split=split)

        # Deterministic partitioning ensures reproducible, non-overlapping train/test subsets.
        if partition not in {"all", "train", "test"}:
            raise ValueError("partition must be one of: all, train, test")
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be in the open interval (0, 1)")

        if partition != "all":
            total_size = len(self.dataset)
            all_indices = list(range(total_size))
            rng = random.Random(split_seed)
            rng.shuffle(all_indices)

            test_size = int(total_size * test_fraction)
            test_size = max(1, min(test_size, total_size - 1))
            test_indices = all_indices[:test_size]
            train_indices = all_indices[test_size:]
            selected_indices = train_indices if partition == "train" else test_indices
            self.dataset = self.dataset.select(selected_indices)

        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def __len__(self) -> int:
        """Return number of examples available after optional truncation."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one example with normalized answer fields used by inference/training.

        Keys:
            id: Stable row id (or index fallback)
            image: PIL image object from the dataset
            question: CLEVR problem text
            solution: Raw solution text containing XML-like tags
            gold_answer: Normalized text extracted from <answer>...</answer>
        """
        row = self.dataset[index]
        solution = row["solution"]
        gold_answer = normalize_answer_text(extract_tagged_answer(solution))
        return {
            "id": row.get("id", index),
            "image": row["image"],
            "question": row["problem"],
            "solution": solution,
            "gold_answer": gold_answer,
        }


class M3CoTDataset(Dataset):
    """Dataset wrapper that exposes M3CoT rows in the LVAR-ready format."""

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        dataset_name: str = "LightChen2333/M3CoT",
        require_image: bool = True,
        max_latent_stage: int = 8,
    ) -> None:
        if load_dataset is None:
            raise ImportError(
                "datasets is required to load LightChen2333/M3CoT. Install the requirements first."
            )
        self.dataset = load_dataset(dataset_name, split=split)
        if require_image:
            self.dataset = self.dataset.filter(lambda row: row.get("image") is not None)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.max_latent_stage = int(max_latent_stage)

    def __len__(self) -> int:
        """Return number of examples available after optional truncation."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one M3CoT example using the shared LVAR example contract.

        M3CoT answers are multiple-choice labels (A/B/C/...), so the choices are
        folded into the question text and the gold answer is the normalized label.
        """
        row = self.dataset[index]
        choices = list(row.get("choices") or [])
        choices_str = "[Options]:\n" + "\n".join(
            [
                f"({chr(65 + choice_index)}).{{{str(choice).strip()}}}"
                for choice_index, choice in enumerate(choices)
            ]
        )
        question_with_braces = f"{{{str(row['question']).strip()}}}"
        formatted_question = f"[Question]:{question_with_braces}\n{choices_str}\nAnswer:\n"
        answer = str(row["answer"]).strip()
        rationale = str(row.get("rationale") or "").strip()
        steps = group_steps_to_max(split_rationale_into_sentences(rationale), self.max_latent_stage)
        solution = f"{rationale}\n<answer>{answer}</answer>" if rationale else f"<answer>{answer}</answer>"
        return {
            "id": row.get("id", index),
            "image": row["image"],
            "question": formatted_question,
            "steps": steps,
            "answer": answer,
            "solution": solution,
            "gold_answer": normalize_answer_text(answer),
            "domain": row.get("domain"),
            "topic": row.get("topic"),
            "idx": int(index),
        }


class ScienceQADataset(Dataset):
    """Dataset wrapper that exposes ScienceQA rows in the LVAR-ready format."""

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        dataset_name: str = "derek-thomas/ScienceQA",
        require_image: bool = True,
        max_latent_stage: int = 8,
    ) -> None:
        if load_dataset is None:
            raise ImportError("datasets is required to load derek-thomas/ScienceQA. Install the requirements first.")
        self.dataset = load_dataset(dataset_name, split=split)
        if require_image:
            self.dataset = self.dataset.filter(lambda row: row.get("image") is not None)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.max_latent_stage = int(max_latent_stage)

    def __len__(self) -> int:
        """Return number of examples available after optional filtering/truncation."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one ScienceQA example using the shared LVAR example contract.

        ScienceQA stores the answer as a choice index. We render prompts with
        the same multiple-choice format as M3CoT and evaluate against the
        corresponding normalized letter.
        """
        row = self.dataset[index]
        choices = list(row.get("choices") or [])
        answer_index = int(row["answer"])
        answer_label = chr(ord("A") + answer_index)
        question_with_braces = f"{{{str(row['question']).strip()}}}"
        if choices:
            choices_str = "[Options]:\n" + "\n".join(
                [
                    f"({chr(65 + choice_index)}).{{{str(choice).strip()}}}"
                    for choice_index, choice in enumerate(choices)
                ]
            )
            formatted_question = f"[Question]:{question_with_braces}\n{choices_str}\nAnswer:\n"
        else:
            formatted_question = f"[Question]:{question_with_braces}\nAnswer:\n"

        solution_text = str(row.get("solution") or "").strip()
        steps = group_steps_to_max(split_rationale_into_sentences(solution_text), self.max_latent_stage)
        solution = (
            f"{solution_text}\n<answer>{answer_label}</answer>"
            if solution_text
            else f"<answer>{answer_label}</answer>"
        )
        return {
            "id": row.get("id", index),
            "image": row["image"],
            "question": formatted_question,
            "steps": steps,
            "answer": answer_label,
            "solution": solution,
            "gold_answer": normalize_answer_text(answer_label),
            "choices": choices,
            "answer_index": answer_index,
            "raw_answer": str(row["answer"]).strip(),
            "hint": row.get("hint"),
            "task": row.get("task"),
            "grade": row.get("grade"),
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "category": row.get("category"),
            "skill": row.get("skill"),
        }


def build_dataset(dataset_cfg: Dict[str, Any], limit: Optional[int] = None, partition: Optional[str] = None) -> Dataset:
    """Instantiate the configured dataset behind a shared script-facing API."""
    dataset_type = str(dataset_cfg.get("type", "clevr")).strip().lower()
    dataset_limit = dataset_cfg.get("limit") if limit is None else limit
    if dataset_type in {"clevr", "clevr_cogent", "clevr-cogent"}:
        return CLEVRCoGenTDataset(
            split=dataset_cfg.get("split", "train"),
            limit=dataset_limit,
            dataset_name=dataset_cfg.get("name", "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"),
            partition=partition or dataset_cfg.get("partition", "all"),
            test_fraction=float(dataset_cfg.get("test_fraction", 0.1)),
            split_seed=int(dataset_cfg.get("split_seed", 42)),
        )
    if dataset_type in {"m3cot", "m3-cot"}:
        split = partition if partition in {"train", "validation", "test"} else dataset_cfg.get("split", "train")
        return M3CoTDataset(
            split=split,
            limit=dataset_limit,
            dataset_name=dataset_cfg.get("name", "LightChen2333/M3CoT"),
            require_image=bool(dataset_cfg.get("require_image", True)),
            max_latent_stage=int(dataset_cfg.get("max_latent_stage", 8)),
        )
    if dataset_type in {"scienceqa", "science-qa", "sqa"}:
        split = partition if partition in {"train", "validation", "test"} else dataset_cfg.get("split", "train")
        return ScienceQADataset(
            split=split,
            limit=dataset_limit,
            dataset_name=dataset_cfg.get("name", "derek-thomas/ScienceQA"),
            require_image=bool(dataset_cfg.get("require_image", True)),
            max_latent_stage=int(dataset_cfg.get("max_latent_stage", 8)),
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
