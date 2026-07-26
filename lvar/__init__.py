from .dataset import CLEVRCoGenTDataset, M3CoTDataset, ScienceQADataset, build_dataset
from .cosine_mining import CosineSimilarityTraceMiner, select_top_k_patches, summarize_cosine_trace_rows
from .oracle_mining import (
    OracleTraceMiner,
    build_step_target,
    group_steps_to_max,
    preprocess_reasoning_steps,
    split_rationale_into_sentences,
)
from .beam_oracle_mining import BeamOracleTraceMiner
from .fast_oracle_mining import BeamSearchFastOracleTraceMiner, GreedyFastOracleTraceMiner
from .qwen_lvar import QwenLVAR
from .trace_attention_boost import TraceBoostConfig
from .rewards import (
    baseline_correctness_reward,
    correctness_reward,
    delta_reward,
    normalize_answer,
)

__all__ = [
    "CLEVRCoGenTDataset",
    "M3CoTDataset",
    "ScienceQADataset",
    "build_dataset",
    "CosineSimilarityTraceMiner",
    "select_top_k_patches",
    "summarize_cosine_trace_rows",
    "OracleTraceMiner",
    "BeamOracleTraceMiner",
    "GreedyFastOracleTraceMiner",
    "BeamSearchFastOracleTraceMiner",
    "build_step_target",
    "group_steps_to_max",
    "preprocess_reasoning_steps",
    "split_rationale_into_sentences",
    "QwenLVAR",
    "TraceBoostConfig",
    "baseline_correctness_reward",
    "correctness_reward",
    "delta_reward",
    "normalize_answer",
]
