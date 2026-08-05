"""Shared gold-rationale preprocessing used by mining and GRPO rewards."""

import math
import re
from typing import List, Sequence


ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


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
    """Group contiguous rationale units evenly into at most ``max_steps`` blocks."""
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
