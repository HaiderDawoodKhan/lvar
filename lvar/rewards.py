import re

from lvar.utils import normalize_answer_text


def normalize_answer(answer: str) -> str:
    """Shared normalization entrypoint used by all reward calculations."""
    return normalize_answer_text(answer)


def _normalize_gold_choice(gold_answer: str) -> str:
    gt_answer = str(gold_answer or "").strip().upper()
    if gt_answer.isdigit():
        digit = int(gt_answer)
        if 0 <= digit <= 3:
            return chr(ord("A") + digit)
    return gt_answer


def extract_choice_candidates(generated_text: str) -> set[str]:
    """Extract multiple-choice answer letters from free-form model output."""
    cleaned_text = re.sub(
        r"(?<=answer:)\s*(\n+\s*)?assistant\b",
        "",
        str(generated_text or ""),
        flags=re.IGNORECASE,
    )
    candidates = {
        match.group(1).upper()
        for match in re.finditer(
            r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])",
            cleaned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    for match in re.finditer(
        r"(?:the\s+answer\s+is|Answer:)\s*[\n\s]*(\d)",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        digit = int(match.group(1))
        if 0 <= digit <= 3:
            candidates.add(chr(ord("A") + digit))
    return candidates


def verify_choice_output(generated_text: str, gold_answer: str) -> bool:
    """Verifier used by M3CoT inference and GRPO correctness rewards."""
    gt_answer = _normalize_gold_choice(gold_answer)
    if not gt_answer:
        return False
    candidates = extract_choice_candidates(generated_text)
    if candidates:
        return gt_answer in candidates
    return normalize_answer(generated_text) == normalize_answer(gold_answer)


def correctness_reward(prediction: str, gold_answer: str) -> float:
    """Return 1.0 when free-form multiple-choice output matches gold."""
    return float(verify_choice_output(prediction, gold_answer))


def baseline_correctness_reward(prediction: str, gold_answer: str) -> float:
    """Baseline correctness mirror kept separate for clarity in delta reward code."""
    return correctness_reward(prediction, gold_answer)


def delta_reward(lvar_prediction: str, baseline_prediction: str, gold_answer: str) -> float:
    """Compute R_delta = R_lvar - R_base used by controller policy optimization."""
    lvar_score = correctness_reward(lvar_prediction, gold_answer)
    baseline_score = baseline_correctness_reward(baseline_prediction, gold_answer)
    return lvar_score - baseline_score
