import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.latent_depth import (
    BUCKET_IMAGE,
    BUCKET_LATENT,
    BUCKET_PROMPT,
    aggregate_attention_by_bucket,
    append_latent_label,
    compute_hidden_step_metrics,
    label_initial_positions,
)
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import verify_choice_output
from lvar.utils import (
    add_model_loading_args,
    add_trace_boost_args,
    apply_model_loading_overrides,
    apply_trace_boost_overrides,
    boosted_output_path,
    trace_boost_is_enabled,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def entropy_tracking_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_entropy_tracking.json")


def normalize_context_mode(context: str) -> str:
    mode = str(context).strip().lower()
    if mode in {"global", "full", "full_image", "full_context"}:
        return "full_context"
    if mode in {"coarse", "coarse_context", "global_mean", "global_token"}:
        return "global_mean"
    raise ValueError("context must be one of: global, coarse, full_context, global_mean.")


def apply_fixed_think_steps(
    model: QwenLVAR,
    state: Dict[str, Any],
    num_think_steps: int,
    labels: Optional[List[str]] = None,
    track_latent_depth_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Apply a fixed number of THINK updates without invoking the controller.

    This intentionally mirrors the THINK branch in QwenLVAR.forward_reasoning_step:
    read the current hidden state, then either append that hidden state as a token
    or update the recurrent control token slots, depending on model config.
    """
    latent_hidden_vectors: List[torch.Tensor] = []
    latent_step_attention: List[Dict[str, Any]] = []
    for step_idx in range(int(num_think_steps)):
        sequence_length_before = int(state["inputs_embeds"].size(1))
        if track_latent_depth_metrics:
            with torch.no_grad():
                outputs = model.backbone(
                    inputs_embeds=state["inputs_embeds"],
                    attention_mask=state["attention_mask"],
                    **model._state_position_kwargs(state),
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
            final_hidden = model._extract_final_hidden(outputs)
            last_hidden = final_hidden[:, -1, :]
            if model.use_control_tokens:
                state_hidden = final_hidden[:, state["latent_pos"], :]
                act_hidden = final_hidden[:, state["act_pos"], :]
            else:
                state_hidden = last_hidden
                act_hidden = None
            attentions = getattr(outputs, "attentions", None)
            if not attentions or attentions[-1] is None:
                raise ValueError(
                    "Backbone did not return attentions. Set --attn-implementation eager "
                    "when using --track-latent-depth-metrics."
                )
            if labels is None:
                raise ValueError("labels are required when tracking latent-depth metrics.")
            query_pos = int(state["latent_pos"]) if model.use_control_tokens and not model.think_append_hidden else sequence_length_before - 1
            attention_metrics = aggregate_attention_by_bucket(
                attentions,
                labels=labels,
                query_pos=query_pos,
            )
            latent_step_attention.append(
                {
                    "step_idx": step_idx,
                    "query_pos": query_pos,
                    "sequence_length": sequence_length_before,
                    **attention_metrics,
                }
            )
        else:
            last_hidden, state_hidden, act_hidden = model._read_current_hidden(state)
        latent_hidden = state_hidden if model.use_control_tokens and not model.think_append_hidden else last_hidden
        latent_hidden_vectors.append(latent_hidden.detach())

        if model.think_append_hidden:
            model._append_hidden_token(state, last_hidden, track_as_think=True)
        elif model.use_control_tokens:
            state["inputs_embeds"] = model._write_recurrent_tokens(
                state["inputs_embeds"],
                state["latent_pos"],
                state["act_pos"],
                state_hidden,
                act_hidden,
            )
        else:
            updated_embeds = state["inputs_embeds"].clone()
            updated_embeds[:, -1, :] = last_hidden.to(updated_embeds.dtype)
            state["inputs_embeds"] = updated_embeds

        state.setdefault("trace", []).append(
            {
                "step_idx": step_idx,
                "action_id": None,
                "action": "THINK",
                "action_source": "fixed",
                "should_stop": False,
                "sequence_length_before": sequence_length_before,
                "sequence_length_after": int(state["inputs_embeds"].size(1)),
            }
        )
        if labels is not None:
            append_latent_label(labels, sequence_length_before, int(state["inputs_embeds"].size(1)))
    if track_latent_depth_metrics:
        state["latent_step_attention"] = latent_step_attention
        state["latent_step_attention_summary"] = [
            {
                "step_idx": row["step_idx"],
                "image_mass": row["summary"][BUCKET_IMAGE]["mean"],
                "prompt_mass": row["summary"][BUCKET_PROMPT]["mean"],
                "latent_mass": row["summary"][BUCKET_LATENT]["mean"],
            }
            for row in latent_step_attention
        ]
        state["latent_step_hidden_metrics"] = compute_hidden_step_metrics(latent_hidden_vectors)
    return state


def fixed_think_decode(
    model: QwenLVAR,
    example: Dict[str, Any],
    num_think_steps: int,
    context_mode: str,
    image_size: Optional[int],
    add_answer_instruction: bool,
    track_latent_depth_metrics: bool = False,
) -> Dict[str, Any]:
    prepared = model.prepare_inputs(
        example["image"],
        example["question"],
        add_answer_instruction=add_answer_instruction,
        image_size=image_size,
    )
    image_tokens = model.get_projected_image_tokens(prepared)
    prepared["projected_image_tokens"] = image_tokens
    bank = model.build_visual_bank(image_tokens)
    if context_mode == "global_mean":
        state = model.build_coarse_initial_state(prepared, bank)
    else:
        state = model.build_initial_state(prepared)
    labels = None
    if track_latent_depth_metrics:
        labels = label_initial_positions(model, prepared, bank, context_mode=context_mode)
        if len(labels) != int(state["inputs_embeds"].size(1)):
            raise ValueError(
                f"Initial latent-depth label length {len(labels)} does not match "
                f"sequence length {state['inputs_embeds'].size(1)}."
            )
    state["sample_actions"] = False
    state = apply_fixed_think_steps(
        model,
        state,
        num_think_steps=num_think_steps,
        labels=labels,
        track_latent_depth_metrics=track_latent_depth_metrics,
    )
    if model.use_control_tokens:
        state = model.drop_act_token(state)
    decoded = model.decode_answer(model._build_decode_state(state))
    return {
        **decoded,
        "trace": state["trace"],
        "num_steps": len(state["trace"]),
        "latent_step_attention": state.get("latent_step_attention"),
        "latent_step_attention_summary": state.get("latent_step_attention_summary"),
        "latent_step_hidden_metrics": state.get("latent_step_hidden_metrics"),
    }


def build_entropy_tracking_row(
    example: Dict[str, Any],
    decoded: Dict[str, Any],
    is_correct: bool,
    num_think_steps: int,
    context: str,
    context_mode: str,
) -> Dict[str, Any]:
    option_entropy = decoded.get("answer_option_entropy") or {}
    return {
        "example_id": example["id"],
        "correct": is_correct,
        "gold_answer": example["gold_answer"],
        "raw_answer": example.get("answer", ""),
        "decoded_answer": decoded.get("answer"),
        "context": context,
        "context_mode": context_mode,
        "trace_variant": f"fixed_think_{num_think_steps}",
        "num_think_steps": num_think_steps,
        "num_trace_actions": len(decoded["trace"]),
        "num_output_tokens": len(decoded["generated_ids"]),
        "answer_option_entropy": option_entropy.get("entropy"),
        "answer_option_probabilities": option_entropy.get("softmax_option_probabilities"),
        "answer_option_raw_probabilities": option_entropy.get("raw_option_probabilities"),
        "answer_option_token_ids": option_entropy.get("option_token_ids"),
        "answer_option_selected_option": option_entropy.get("selected_option"),
        "answer_option_selected_token_id": option_entropy.get("selected_token_id"),
        "answer_option_decoded_token_index": option_entropy.get("decoded_token_index"),
        "decoded_token_entropies": decoded["token_entropies"],
        "decoded_token_entropy_mean": decoded["token_entropy_mean"],
        "decoded_token_entropy_median": decoded["token_entropy_median"],
        "decoded_token_entropy_max": decoded["token_entropy_max"],
        "trace_attention_mass": decoded.get("trace_attention_mass"),
        "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
        "think_attention_mass": decoded.get("think_attention_mass"),
        "latent_step_attention_summary": decoded.get("latent_step_attention_summary"),
        "latent_step_hidden_metrics": decoded.get("latent_step_hidden_metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run M3CoT inference with a fixed number of THINK steps, without "
            "controller decisions and without replayed/mined traces."
        )
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--dataset-partition", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-think-steps", type=int, default=2)
    parser.add_argument(
        "--context",
        default="global",
        choices=["global", "coarse", "full_context", "global_mean"],
        help="'global'/'full_context' uses the full image-token prompt; 'coarse'/'global_mean' uses one pooled visual token.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--add-answer-instruction", action="store_true", default=False)
    parser.add_argument("--track-latent-depth-metrics", action="store_true", default=False)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Backbone attention implementation. Use eager when tracking latent-depth metrics.",
    )
    add_model_loading_args(parser)
    add_trace_boost_args(parser)
    args = parser.parse_args()

    if args.num_think_steps < 0:
        raise ValueError("--num-think-steps must be non-negative.")

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    config["model"] = apply_trace_boost_overrides(config["model"], args)
    if args.attn_implementation:
        config["model"]["attn_implementation"] = args.attn_implementation
    elif args.track_latent_depth_metrics:
        config["model"]["attn_implementation"] = "eager"
    dataset_cfg = config["dataset"]
    phase2_cfg = config.get("phase2", {})
    inference_cfg = config.get("inference", {})
    image_size = args.image_size
    if image_size is None:
        image_size = inference_cfg.get("image_size", phase2_cfg.get("image_size", 280))

    seed = int(args.seed if args.seed is not None else phase2_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    context_mode = normalize_context_mode(args.context)
    dataset = build_dataset(dataset_cfg, limit=args.limit, partition=args.dataset_partition)
    model = QwenLVAR(config["model"])
    model.eval()

    output_path = Path(
        boosted_output_path(
            args.output,
            enabled=trace_boost_is_enabled(config["model"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entropy_path = entropy_tracking_path(output_path)
    entropy_rows: List[Dict[str, Any]] = []

    total = 0
    correct = 0
    total_think_tokens = 0
    total_output_tokens = 0
    attention_mass_values = {
        "trace_attention_mass": [],
        "visual_trace_attention_mass": [],
        "think_attention_mass": [],
    }
    latent_depth_metric_values: Dict[int, Dict[str, List[float]]] = {}

    with open(output_path, "w", encoding="utf-8") as handle:
        for index in tqdm(range(len(dataset)), desc=f"Fixed THINK x{args.num_think_steps}"):
            example = dataset[index]
            with torch.no_grad():
                decoded = fixed_think_decode(
                    model=model,
                    example=example,
                    num_think_steps=args.num_think_steps,
                    context_mode=context_mode,
                    image_size=image_size,
                    add_answer_instruction=bool(args.add_answer_instruction),
                    track_latent_depth_metrics=bool(args.track_latent_depth_metrics),
                )

            generated_text = decoded["generated_text"]
            is_correct = verify_choice_output(generated_text, example["gold_answer"])
            num_think_tokens = sum(
                int(step["sequence_length_after"]) - int(step["sequence_length_before"])
                for step in decoded["trace"]
            )
            num_output_tokens = len(decoded["generated_ids"])

            total += 1
            correct += int(is_correct)
            total_think_tokens += num_think_tokens
            total_output_tokens += num_output_tokens
            for key in attention_mass_values:
                value = decoded.get(key)
                if value is not None:
                    attention_mass_values[key].append(float(value))
            for step_row in decoded.get("latent_step_attention_summary") or []:
                step_idx = int(step_row["step_idx"])
                metrics = latent_depth_metric_values.setdefault(
                    step_idx,
                    {
                        "image_mass": [],
                        "prompt_mass": [],
                        "latent_mass": [],
                        "hidden_norm": [],
                        "hidden_norm_delta": [],
                        "hidden_delta_norm": [],
                    },
                )
                for key in ("image_mass", "prompt_mass", "latent_mass"):
                    if step_row.get(key) is not None:
                        metrics[key].append(float(step_row[key]))
            for step_row in decoded.get("latent_step_hidden_metrics") or []:
                step_idx = int(step_row["step_idx"])
                metrics = latent_depth_metric_values.setdefault(
                    step_idx,
                    {
                        "image_mass": [],
                        "prompt_mass": [],
                        "latent_mass": [],
                        "hidden_norm": [],
                        "hidden_norm_delta": [],
                        "hidden_delta_norm": [],
                    },
                )
                for key in ("hidden_norm", "hidden_norm_delta", "hidden_delta_norm"):
                    if step_row.get(key) is not None:
                        metrics[key].append(float(step_row[key]))

            row = {
                "example_id": example["id"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "raw_answer": example.get("answer", ""),
                "domain": example.get("domain"),
                "topic": example.get("topic"),
                "context": args.context,
                "context_mode": context_mode,
                "trace_variant": f"fixed_think_{args.num_think_steps}",
                "correct": is_correct,
                "num_steps": len(decoded["trace"]),
                "num_think_steps": args.num_think_steps,
                "num_think_tokens": num_think_tokens,
                "num_output_tokens": num_output_tokens,
                "num_total_tokens": num_think_tokens + num_output_tokens,
                "generated_text": generated_text,
                "decoded_answer": decoded["answer"],
                "trace": decoded["trace"],
                "trace_attention_mass": decoded.get("trace_attention_mass"),
                "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
                "think_attention_mass": decoded.get("think_attention_mass"),
                "latent_step_attention": decoded.get("latent_step_attention"),
                "latent_step_attention_summary": decoded.get("latent_step_attention_summary"),
                "latent_step_hidden_metrics": decoded.get("latent_step_hidden_metrics"),
            }
            write_jsonl_row(handle, row)
            entropy_rows.append(
                build_entropy_tracking_row(
                    example=example,
                    decoded=decoded,
                    is_correct=is_correct,
                    num_think_steps=args.num_think_steps,
                    context=args.context,
                    context_mode=context_mode,
                )
            )

    accuracy = correct / total if total else 0.0
    write_json(entropy_path, entropy_rows)
    summary = {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_name": dataset_cfg.get("name"),
        "dataset_partition": args.dataset_partition,
        "output_path": str(output_path),
        "entropy_tracking_path": str(entropy_path),
        "checkpoint_path": config["model"].get("checkpoint_path"),
        "use_checkpoint": config["model"].get("use_checkpoint"),
        "trace_boost": model.trace_boost_config.to_dict(),
        "context": args.context,
        "context_mode": context_mode,
        "num_think_steps": args.num_think_steps,
        "add_answer_instruction": bool(args.add_answer_instruction),
        "image_size": image_size,
        "seed": seed,
        "track_latent_depth_metrics": bool(args.track_latent_depth_metrics),
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_think_tokens": round(total_think_tokens / total, 2) if total else 0.0,
            "avg_output_tokens": round(total_output_tokens / total, 2) if total else 0.0,
            "avg_total_tokens": round((total_think_tokens + total_output_tokens) / total, 2) if total else 0.0,
            "trace_attention_mass": (
                sum(attention_mass_values["trace_attention_mass"])
                / len(attention_mass_values["trace_attention_mass"])
                if attention_mass_values["trace_attention_mass"] else None
            ),
            "visual_trace_attention_mass": (
                sum(attention_mass_values["visual_trace_attention_mass"])
                / len(attention_mass_values["visual_trace_attention_mass"])
                if attention_mass_values["visual_trace_attention_mass"] else None
            ),
            "think_attention_mass": (
                sum(attention_mass_values["think_attention_mass"])
                / len(attention_mass_values["think_attention_mass"])
                if attention_mass_values["think_attention_mass"] else None
            ),
        },
        "latent_depth_metrics_by_step": [
            {
                "step_idx": step_idx,
                **{
                    key: (sum(values) / len(values) if values else None)
                    for key, values in metrics.items()
                },
            }
            for step_idx, metrics in sorted(latent_depth_metric_values.items())
        ],
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)

    print(f"Wrote {total} fixed-THINK predictions to {output_path}")
    print(f"Wrote entropy tracking to {entropy_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
