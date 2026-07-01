import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


BUCKET_IMAGE = "image"
BUCKET_PROMPT = "prompt"
BUCKET_LATENT = "latent"


def _safe_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def label_initial_positions(
    model: Any,
    batch: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    context_mode: str,
) -> List[str]:
    """
    Label initial embedding positions as image, prompt, or latent/control.

    Full-context states keep Qwen's image placeholder span. Coarse/global-mean
    states replace that span with bank["global"], so labels must mirror the
    replacement sequence rather than raw input_ids length.
    """
    input_ids = batch["input_ids"][0]
    image_token_id = getattr(model, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Backbone config does not expose image_token_id; cannot label image positions.")

    mode = str(context_mode).strip().lower()
    coarse = mode in {"coarse", "coarse_context", "global_mean", "global_token"}
    if not coarse:
        labels = [
            BUCKET_IMAGE if int(token_id.item()) == int(image_token_id) else BUCKET_PROMPT
            for token_id in input_ids
        ]
    else:
        image_positions = torch.nonzero(input_ids == int(image_token_id), as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise ValueError("No image placeholder positions were found.")
        start = int(image_positions[0].item())
        end = int(image_positions[-1].item()) + 1
        labels = [BUCKET_PROMPT] * start
        labels.extend([BUCKET_IMAGE] * int(bank["global"].size(0)))
        labels.extend([BUCKET_PROMPT] * (int(input_ids.numel()) - end))

    if getattr(model, "use_control_tokens", False):
        labels.extend([BUCKET_LATENT, BUCKET_LATENT])
    return labels


def append_latent_label(labels: List[str], state_before_len: int, state_after_len: int) -> None:
    """Extend labels after an appended fixed-THINK latent token."""
    delta = int(state_after_len) - int(state_before_len)
    if delta <= 0:
        return
    labels.extend([BUCKET_LATENT] * delta)


def aggregate_attention_by_bucket(
    attentions: Iterable[torch.Tensor],
    labels: Sequence[str],
    query_pos: int,
    buckets: Sequence[str] = (BUCKET_IMAGE, BUCKET_PROMPT, BUCKET_LATENT),
    exclude_query_from_latent: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate attention mass from one query position to labeled key buckets.

    Returns per-layer head statistics plus an overall summary across all
    layer-head values. Empty buckets are represented as zero mass.
    """
    labels = list(labels)
    query_pos = int(query_pos)
    per_layer: List[Dict[str, Any]] = []
    bucket_values: Dict[str, List[float]] = {bucket: [] for bucket in buckets}

    for layer_idx, layer_attention in enumerate(attentions):
        attention = layer_attention.detach().float()
        if attention.dim() == 4:
            attention = attention[0]
        if attention.dim() != 3:
            raise ValueError(f"Expected attention tensor with shape [heads, q, k], got {tuple(attention.shape)}.")
        if query_pos < 0 or query_pos >= attention.size(1):
            raise ValueError(f"query_pos={query_pos} is outside attention query length {attention.size(1)}.")

        key_len = int(attention.size(-1))
        layer_row: Dict[str, Any] = {"layer_idx": layer_idx}
        for bucket in buckets:
            indices = [
                idx
                for idx, label in enumerate(labels[:key_len])
                if label == bucket and not (exclude_query_from_latent and bucket == BUCKET_LATENT and idx == query_pos)
            ]
            if indices:
                index_tensor = torch.tensor(indices, device=attention.device, dtype=torch.long)
                masses = attention[:, query_pos, index_tensor].sum(dim=-1).cpu().tolist()
                masses = [float(value) for value in masses]
            else:
                masses = [0.0 for _ in range(int(attention.size(0)))]
            stats = _safe_stats(masses)
            stats["num_tokens"] = len(indices)
            layer_row[bucket] = stats
            bucket_values[bucket].extend(masses)
        per_layer.append(layer_row)

    summary = {bucket: _safe_stats(values) for bucket, values in bucket_values.items()}
    return {"per_layer": per_layer, "summary": summary}


def compute_hidden_step_metrics(hidden_vectors: Sequence[torch.Tensor]) -> List[Dict[str, Optional[float]]]:
    """Compute norm and consecutive-step deltas for latent hidden vectors."""
    metrics: List[Dict[str, Optional[float]]] = []
    previous_vector: Optional[torch.Tensor] = None
    previous_norm: Optional[float] = None
    for step_idx, vector in enumerate(hidden_vectors):
        current = vector.detach().float().reshape(-1).cpu()
        norm = float(torch.linalg.vector_norm(current).item())
        norm_delta = None if previous_norm is None else float(norm - previous_norm)
        delta_norm = None
        if previous_vector is not None:
            delta_norm = float(torch.linalg.vector_norm(current - previous_vector).item())
        metrics.append(
            {
                "step_idx": int(step_idx),
                "hidden_norm": norm,
                "hidden_norm_delta": norm_delta,
                "hidden_delta_norm": delta_norm,
            }
        )
        previous_vector = current
        previous_norm = norm
    return metrics


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def resolve_fixed_think_paths(
    paths: Optional[Iterable[str | Path]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
) -> List[Path]:
    resolved: List[Path] = []
    for raw_path in paths or []:
        path = Path(raw_path)
        if path.exists():
            resolved.append(path)
    for pattern in glob_patterns or []:
        resolved.extend(Path(match) for match in glob.glob(str(pattern)))
    return sorted({path.resolve() for path in resolved})


def infer_depth_from_row_or_path(row: Dict[str, Any], path: str | Path) -> int:
    if row.get("num_think_steps") is not None:
        return int(row["num_think_steps"])
    if row.get("num_steps") is not None and str(row.get("trace_variant", "")).startswith("fixed_think"):
        return int(row["num_steps"])
    path_text = str(path)
    matches = re.findall(r"(?:fixed[_-]?think[_-]?steps?|latent[_-]?steps?)[_-]?(\d+)", path_text)
    if matches:
        return int(matches[-1])
    raise ValueError(f"Could not infer latent depth from row or path: {path}")


def load_fixed_think_rows(
    paths: Optional[Iterable[str | Path]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load fixed-THINK JSONL rows and attach inferred latent depth."""
    resolved = resolve_fixed_think_paths(paths, glob_patterns)
    rows: List[Dict[str, Any]] = []
    for path in resolved:
        for row in read_jsonl(path):
            copied = dict(row)
            copied["latent_depth"] = infer_depth_from_row_or_path(copied, path)
            copied["_source_path"] = str(path)
            rows.append(copied)
    return rows, {"paths": [str(path) for path in resolved], "num_rows": len(rows)}


def build_latent_depth_supervision(
    rows: Iterable[Dict[str, Any]],
    max_depth: int = 10,
    target_policy: str = "earliest_correct",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build binary STOP/CONTINUE supervision from fixed-depth correctness.

    earliest_correct: CONTINUE for depths before the first correct depth, STOP
    at the first correct depth.
    all_correct: STOP at every correct depth up to the last correct depth,
    CONTINUE at non-correct depths that still have a future correct depth.
    """
    policy = str(target_policy).strip().lower()
    if policy not in {"earliest_correct", "all_correct"}:
        raise ValueError("target_policy must be 'earliest_correct' or 'all_correct'.")
    max_depth = int(max_depth)
    by_example: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        example_id = str(row.get("example_id"))
        if example_id in {"", "None"}:
            continue
        depth = int(row.get("latent_depth", row.get("num_think_steps", -1)))
        if depth < 0 or depth > max_depth:
            continue
        by_example[example_id][depth] = row

    supervision: List[Dict[str, Any]] = []
    skipped_no_correct: List[str] = []
    missing_depths: Dict[str, List[int]] = {}
    for example_id, depth_rows in sorted(by_example.items()):
        expected_depths = set(range(max_depth + 1))
        missing = sorted(expected_depths - set(depth_rows))
        if missing:
            missing_depths[example_id] = missing
        correct_depths = sorted(depth for depth, row in depth_rows.items() if bool(row.get("correct")))
        if not correct_depths:
            skipped_no_correct.append(example_id)
            continue

        if policy == "earliest_correct":
            stop_depths = [correct_depths[0]]
            train_depths = [depth for depth in range(stop_depths[0] + 1) if depth in depth_rows]
        else:
            stop_depths = correct_depths
            train_depths = [depth for depth in range(correct_depths[-1] + 1) if depth in depth_rows]

        for depth in train_depths:
            source_row = depth_rows[depth]
            supervision.append(
                {
                    "example_id": example_id,
                    "depth": int(depth),
                    "target_stop": 1.0 if depth in stop_depths else 0.0,
                    "target_action": "STOP" if depth in stop_depths else "CONTINUE",
                    "correct_depths": correct_depths,
                    "earliest_correct_depth": correct_depths[0],
                    "gold_answer": source_row.get("gold_answer"),
                    "question": source_row.get("question"),
                    "source_path": source_row.get("_source_path"),
                }
            )

    summary = {
        "num_examples": len(by_example),
        "num_supervision_rows": len(supervision),
        "num_skipped_no_correct": len(skipped_no_correct),
        "skipped_no_correct_example_ids": skipped_no_correct,
        "num_examples_with_missing_depths": len(missing_depths),
        "missing_depths": missing_depths,
        "max_depth": max_depth,
        "target_policy": policy,
    }
    return supervision, summary
