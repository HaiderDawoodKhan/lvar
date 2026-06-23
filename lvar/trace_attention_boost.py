"""Inference-time pre-softmax attention boosting for LVAR trace tokens."""

from __future__ import annotations

import re
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set

import torch
import torch.nn.functional as F


TRACE_BOOST_TARGETS = {"trace_all", "trace_visual"}
TRACE_BOOST_LAYER_MODES = {"all", "latter_half"}
TRACE_BOOST_APPLY_STAGES = {"answer_only"}


@dataclass(frozen=True)
class TraceBoostConfig:
    """Configuration for answer-stage LVAR trace attention boosting."""

    enabled: bool = False
    target: str = "trace_visual"
    layer_mode: str = "latter_half"
    alpha: float = 0.2
    apply_stage: str = "answer_only"

    def __post_init__(self) -> None:
        if self.target not in TRACE_BOOST_TARGETS:
            raise ValueError(
                f"Unknown trace boost target: {self.target}. "
                f"Expected one of: {sorted(TRACE_BOOST_TARGETS)}."
            )
        if self.layer_mode not in TRACE_BOOST_LAYER_MODES:
            raise ValueError(
                f"Unknown trace boost layer_mode: {self.layer_mode}. "
                f"Expected one of: {sorted(TRACE_BOOST_LAYER_MODES)}."
            )
        if not torch.isfinite(torch.tensor(float(self.alpha))) or float(self.alpha) < 0.0:
            raise ValueError("Trace boost alpha must be a finite non-negative value.")
        if self.apply_stage not in TRACE_BOOST_APPLY_STAGES:
            raise ValueError(
                f"Unknown trace boost apply_stage: {self.apply_stage}. "
                f"Expected one of: {sorted(TRACE_BOOST_APPLY_STAGES)}."
            )

    @classmethod
    def from_value(cls, value: Optional[Any]) -> "TraceBoostConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("trace_boost must be a TraceBoostConfig or mapping.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_boost_layers(num_layers: int, layer_mode: str) -> Set[int]:
    """Resolve decoder-layer indices selected by a trace boost mode."""
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative.")
    if layer_mode == "all":
        return set(range(num_layers))
    if layer_mode == "latter_half":
        return set(range(num_layers // 2, num_layers))
    raise ValueError(f"Unknown layer_mode: {layer_mode}")


def _valid_position_tensor(
    positions: Optional[Sequence[int]],
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    if not positions:
        return torch.empty(0, device=device, dtype=torch.long)
    valid = sorted({int(position) for position in positions if 0 <= int(position) < key_length})
    return torch.tensor(valid, device=device, dtype=torch.long)


def apply_trace_attention_boost(
    attn_scores: torch.Tensor,
    boost_positions: Optional[Sequence[int]],
    alpha: float,
    query_start: Optional[int] = None,
) -> torch.Tensor:
    """Boost selected keys only for answer-stage query rows.

    Masked entries are left untouched. This is important when a full causal
    sequence is decoded without a KV cache: an otherwise-valid trace key may
    still be in the future for an earlier query row and therefore equal -inf.

    ``query_start`` is the first query row allowed to change. When omitted, only
    the final query row is boosted. This keeps prompt and trace representations
    unchanged while allowing all answer-stage rows to attend more strongly to
    the reasoning trace during full-prefix, no-cache decoding.
    """
    if attn_scores.ndim < 2 or not boost_positions or float(alpha) == 0.0:
        return attn_scores
    indices = _valid_position_tensor(boost_positions, attn_scores.size(-1), attn_scores.device)
    if indices.numel() == 0:
        return attn_scores
    query_length = attn_scores.size(-2)
    first_query = query_length - 1 if query_start is None else int(query_start)
    if first_query < 0:
        first_query += query_length
    first_query = max(0, first_query)
    if first_query >= query_length:
        return attn_scores

    answer_rows = attn_scores[..., first_query:, :]
    selected = answer_rows.index_select(-1, indices)
    boosted_selected = torch.where(
        torch.isfinite(selected),
        selected + float(alpha) * selected.abs(),
        selected,
    )
    boosted = attn_scores.clone()
    boosted_answer_rows = answer_rows.clone()
    boosted_answer_rows.index_copy_(-1, indices, boosted_selected)
    boosted[..., first_query:, :] = boosted_answer_rows
    return boosted


class TraceAttentionBoostRuntime:
    """PEFT-safe Qwen eager-attention interception and mass aggregation.

    Qwen2-VL eager attention calls a softmax after applying its causal mask.
    Rather than copying a version-specific Hugging Face attention forward, this
    runtime wraps language self-attention modules to expose the current layer and
    temporarily intercepts eager softmax only while answer decoding is active.
    """

    _LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")

    def __init__(self, config: TraceBoostConfig) -> None:
        self.config = config
        self.num_layers = 0
        self.boost_layers: Set[int] = set()
        self._wrapped_module_ids: Set[int] = set()
        self._current_layer: Optional[int] = None
        self._active = False
        self._trace_all_positions: List[int] = []
        self._trace_visual_positions: List[int] = []
        self._boost_positions: List[int] = []
        self._answer_query_start: Optional[int] = None
        self._mass_sums = {"trace": 0.0, "visual_trace": 0.0, "think": 0.0}
        self._mass_count = 0
        self._softmax_hits = 0

    def install(self, backbone: torch.nn.Module) -> None:
        """Wrap language self-attention modules and assign stable layer indices."""
        if not self.config.enabled:
            return
        modules_by_layer: Dict[int, torch.nn.Module] = {}
        for name, module in backbone.named_modules():
            match = self._LAYER_PATTERN.search(name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            modules_by_layer.setdefault(layer_idx, module)

        if not modules_by_layer:
            raise ValueError(
                "Trace boosting is enabled, but no Qwen language self-attention layers were found. "
                "Expected module names ending in layers.<index>.self_attn."
            )
        expected_layers = list(range(max(modules_by_layer) + 1))
        if sorted(modules_by_layer) != expected_layers:
            raise ValueError(
                "Trace boosting found a non-contiguous set of language attention layers: "
                f"{sorted(modules_by_layer)}."
            )

        self.num_layers = len(modules_by_layer)
        self.boost_layers = get_boost_layers(self.num_layers, self.config.layer_mode)
        for layer_idx, module in sorted(modules_by_layer.items()):
            self._wrap_attention_module(module, layer_idx)

    def _wrap_attention_module(self, module: torch.nn.Module, layer_idx: int) -> None:
        if id(module) in self._wrapped_module_ids:
            return
        original_forward = module.forward
        runtime = self

        def wrapped_forward(module_self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            del module_self
            previous_layer = runtime._current_layer
            runtime._current_layer = layer_idx
            try:
                return original_forward(*args, **kwargs)
            finally:
                runtime._current_layer = previous_layer

        module.forward = types.MethodType(wrapped_forward, module)
        self._wrapped_module_ids.add(id(module))

    def _reset_measurements(self) -> None:
        self._mass_sums = {"trace": 0.0, "visual_trace": 0.0, "think": 0.0}
        self._mass_count = 0
        self._softmax_hits = 0

    def _should_intercept(self, input_tensor: torch.Tensor, dim: Optional[int]) -> bool:
        normalized_dim = -1 if dim is None else int(dim)
        if normalized_dim < 0:
            normalized_dim += input_tensor.ndim
        return (
            self._active
            and self._current_layer in self.boost_layers
            and input_tensor.ndim == 4
            and normalized_dim == input_tensor.ndim - 1
        )

    def _record_attention_masses(self, probabilities: torch.Tensor) -> None:
        # The last row is the query whose logits predict the next answer token.
        query_probabilities = probabilities[..., -1, :].detach().float()
        key_length = query_probabilities.size(-1)
        trace_indices = _valid_position_tensor(
            self._trace_all_positions, key_length, query_probabilities.device
        )
        visual_indices = _valid_position_tensor(
            self._trace_visual_positions, key_length, query_probabilities.device
        )
        visual_set = set(self._trace_visual_positions)
        think_positions = [position for position in self._trace_all_positions if position not in visual_set]
        think_indices = _valid_position_tensor(think_positions, key_length, query_probabilities.device)

        def mass(indices: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return torch.zeros(query_probabilities.shape[:-1], device=query_probabilities.device)
            return query_probabilities.index_select(-1, indices).sum(dim=-1)

        trace_mass = mass(trace_indices)
        visual_mass = mass(visual_indices)
        think_mass = mass(think_indices)
        self._mass_sums["trace"] += float(trace_mass.sum().cpu().item())
        self._mass_sums["visual_trace"] += float(visual_mass.sum().cpu().item())
        self._mass_sums["think"] += float(think_mass.sum().cpu().item())
        self._mass_count += int(trace_mass.numel())

    def _functional_softmax(
        self,
        original_softmax: Any,
        input_tensor: torch.Tensor,
        dim: Optional[int],
        stacklevel: int,
        dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        if not self._should_intercept(input_tensor, dim):
            return original_softmax(input_tensor, dim=dim, _stacklevel=stacklevel, dtype=dtype)
        boosted = apply_trace_attention_boost(
            input_tensor,
            self._boost_positions,
            self.config.alpha,
            query_start=self._answer_query_start,
        )
        probabilities = original_softmax(boosted, dim=dim, _stacklevel=stacklevel, dtype=dtype)
        self._softmax_hits += 1
        self._record_attention_masses(probabilities)
        return probabilities

    def _torch_softmax(
        self,
        original_softmax: Any,
        input_tensor: torch.Tensor,
        dim: int,
        dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        if not self._should_intercept(input_tensor, dim):
            return original_softmax(input_tensor, dim=dim, dtype=dtype)
        boosted = apply_trace_attention_boost(
            input_tensor,
            self._boost_positions,
            self.config.alpha,
            query_start=self._answer_query_start,
        )
        probabilities = original_softmax(boosted, dim=dim, dtype=dtype)
        self._softmax_hits += 1
        self._record_attention_masses(probabilities)
        return probabilities

    @contextmanager
    def answer_decode(
        self,
        trace_all_positions: Optional[Iterable[int]],
        trace_visual_positions: Optional[Iterable[int]],
        answer_query_start: Optional[int] = None,
    ) -> Iterator[None]:
        """Activate boosting and attention logging for one answer decode."""
        if not self.config.enabled:
            self._reset_measurements()
            yield
            return

        self._trace_all_positions = sorted({int(position) for position in trace_all_positions or []})
        self._trace_visual_positions = sorted(
            {int(position) for position in trace_visual_positions or []}
        )
        self._boost_positions = (
            self._trace_all_positions
            if self.config.target == "trace_all"
            else self._trace_visual_positions
        )
        self._answer_query_start = answer_query_start
        self._reset_measurements()
        original_functional_softmax = F.softmax
        original_torch_softmax = torch.softmax
        runtime = self

        def functional_softmax(
            input_tensor: torch.Tensor,
            dim: Optional[int] = None,
            _stacklevel: int = 3,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            return runtime._functional_softmax(
                original_functional_softmax,
                input_tensor,
                dim,
                _stacklevel,
                dtype,
            )

        def torch_softmax(
            input_tensor: torch.Tensor,
            dim: int,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            return runtime._torch_softmax(
                original_torch_softmax,
                input_tensor,
                dim,
                dtype,
            )

        self._active = True
        F.softmax = functional_softmax
        torch.softmax = torch_softmax
        try:
            yield
            if self._softmax_hits == 0:
                raise RuntimeError(
                    "Trace boosting was enabled, but no eager Qwen attention softmax was intercepted. "
                    "Confirm that the installed Transformers Qwen2-VL implementation uses eager attention."
                )
        finally:
            self._active = False
            self._current_layer = None
            self._answer_query_start = None
            F.softmax = original_functional_softmax
            torch.softmax = original_torch_softmax

    def attention_mass_summary(self) -> Dict[str, Any]:
        denominator = self._mass_count
        return {
            "trace_attention_mass": self._mass_sums["trace"] / denominator if denominator else None,
            "visual_trace_attention_mass": (
                self._mass_sums["visual_trace"] / denominator if denominator else None
            ),
            "think_attention_mass": self._mass_sums["think"] / denominator if denominator else None,
            "trace_boost_attention_observations": denominator,
            "trace_boost_softmax_hits": self._softmax_hits,
        }
