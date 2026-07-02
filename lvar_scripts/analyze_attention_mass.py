import argparse
import json
import math
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
from lvar.grpo_training import load_controller_checkpoint, load_vlm_lora_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import ACTION_NAMES_NO_GLOBAL, add_model_loading_args, apply_model_loading_overrides


BUCKET_IMAGE = "image"
BUCKET_TEXT = "text"
BUCKET_REASONING = "reasoning_latent"
BUCKET_OUTPUT = "output"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def initial_position_labels(
    model: QwenLVAR,
    batch: Dict[str, Any],
    bank: Dict[str, torch.Tensor],
    use_coarse_context: bool,
) -> List[str]:
    input_ids = batch["input_ids"][0]
    image_token_id = model.image_token_id
    if image_token_id is None:
        raise ValueError("Backbone config does not expose image_token_id; cannot label image-token positions.")

    if not use_coarse_context:
        labels = [
            BUCKET_IMAGE if int(token_id.item()) == int(image_token_id) else BUCKET_TEXT
            for token_id in input_ids
        ]
    else:
        image_positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise ValueError("No image placeholder positions were found.")
        start = int(image_positions[0].item())
        end = int(image_positions[-1].item()) + 1
        num_visual_tokens = int(bank["global"].size(0))
        labels = [BUCKET_TEXT] * start
        labels.extend([BUCKET_IMAGE] * num_visual_tokens)
        labels.extend([BUCKET_TEXT] * (input_ids.numel() - end))

    if model.use_control_tokens:
        labels.extend([BUCKET_REASONING, BUCKET_REASONING])
    return labels


def insert_reasoning_labels_for_step(
    labels: List[str],
    step_trace: Dict[str, Any],
    model: QwenLVAR,
) -> None:
    before = int(step_trace["sequence_length_before"])
    after = int(step_trace["sequence_length_after"])
    delta = after - before
    if delta <= 0:
        return

    action = str(step_trace.get("action", "")).upper()
    if action == "THINK" and model.think_append_hidden:
        labels.extend([BUCKET_REASONING] * delta)
        return

    insert_pos = before - 2 if model.use_control_tokens else before - 1
    if insert_pos < 0 or insert_pos > len(labels):
        raise ValueError(f"Cannot insert reasoning labels at position {insert_pos} for label length {len(labels)}.")
    labels[insert_pos:insert_pos] = [BUCKET_REASONING] * delta


def drop_act_label_if_needed(labels: List[str], state: Dict[str, Any]) -> None:
    act_pos = state.get("act_pos")
    if act_pos is None:
        return
    del labels[int(act_pos)]


def run_controller_and_decode(
    model: QwenLVAR,
    example: Dict[str, Any],
    image_size: Optional[int],
    use_coarse_context: bool,
    sample_actions: Optional[bool],
) -> Dict[str, Any]:
    prepared = model.prepare_inputs(
        example["image"],
        example["question"],
        add_answer_instruction=False,
        image_size=image_size,
    )
    image_tokens = model.get_projected_image_tokens(prepared)
    prepared["projected_image_tokens"] = image_tokens
    bank = model.build_visual_bank(image_tokens)
    state = model.build_coarse_initial_state(prepared, bank) if use_coarse_context else model.build_initial_state(prepared)
    labels = initial_position_labels(model, prepared, bank, use_coarse_context)
    if len(labels) != state["inputs_embeds"].size(1):
        raise ValueError(f"Initial label length {len(labels)} does not match sequence length {state['inputs_embeds'].size(1)}.")

    if sample_actions is None:
        state["sample_actions"] = model.training or model._inference_uses_sampling()
    else:
        state["sample_actions"] = bool(sample_actions)

    stopped = False
    for step_idx in range(model.max_steps):
        state, _, stopped, step_trace = model.forward_reasoning_step(state, bank, step_idx)
        insert_reasoning_labels_for_step(labels, step_trace, model)
        if len(labels) != state["inputs_embeds"].size(1):
            raise ValueError(
                f"Label length {len(labels)} does not match sequence length {state['inputs_embeds'].size(1)} after step {step_idx}."
            )
        if stopped:
            break

    if model.use_control_tokens:
        drop_act_label_if_needed(labels, state)
        state = model.drop_act_token(state)

    with torch.no_grad():
        decoded = model.decode_answer(model._build_decode_state(state))

    generated_count = len(decoded["generated_ids"])
    final_labels = labels + ([BUCKET_OUTPUT] * generated_count)
    if len(final_labels) != int(decoded["final_sequence_length"]):
        raise ValueError(
            f"Final label length {len(final_labels)} does not match final sequence length {decoded['final_sequence_length']}."
        )
    return {
        "decoded": decoded,
        "labels": final_labels,
        "trace": state["trace"],
        "stopped": stopped,
    }


def last_layer_attention_masses(
    model: QwenLVAR,
    decoded: Dict[str, Any],
    labels: List[str],
    normalize_over_buckets: bool,
) -> Optional[Dict[str, float]]:
    generated_count = len(decoded["generated_ids"])
    if generated_count == 0:
        return None

    with torch.no_grad():
        outputs = model.backbone(
            inputs_embeds=decoded["final_inputs_embeds"],
            attention_mask=decoded["final_attention_mask"],
            **(
                {"position_ids": decoded["final_position_ids"]}
                if model.use_mrope_position_ids and decoded.get("final_position_ids") is not None
                else {}
            ),
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
    attentions = getattr(outputs, "attentions", None)
    if not attentions or attentions[-1] is None:
        raise ValueError(
            "Backbone did not return attentions. Set model.attn_implementation=eager or pass "
            "--attn-implementation eager."
        )

    last_attention = attentions[-1].detach().float()[0].mean(dim=0)
    output_start = int(decoded["decode_prefix_length"])
    output_end = int(decoded["final_sequence_length"])
    bucket_indices = {
        BUCKET_IMAGE: [idx for idx, label in enumerate(labels) if label == BUCKET_IMAGE],
        BUCKET_TEXT: [idx for idx, label in enumerate(labels) if label == BUCKET_TEXT],
        BUCKET_REASONING: [idx for idx, label in enumerate(labels) if label == BUCKET_REASONING],
    }
    per_token = {BUCKET_IMAGE: [], BUCKET_TEXT: [], BUCKET_REASONING: []}
    for query_pos in range(output_start, output_end):
        masses = {}
        for bucket, indices in bucket_indices.items():
            if not indices:
                masses[bucket] = 0.0
            else:
                index_tensor = torch.tensor(indices, device=last_attention.device, dtype=torch.long)
                masses[bucket] = float(last_attention[query_pos, index_tensor].sum().item())
        if normalize_over_buckets:
            denom = sum(masses.values())
            if denom > 0:
                masses = {bucket: value / denom for bucket, value in masses.items()}
        for bucket, value in masses.items():
            per_token[bucket].append(value)

    return {
        "image_mass": sum(per_token[BUCKET_IMAGE]) / len(per_token[BUCKET_IMAGE]),
        "text_mass": sum(per_token[BUCKET_TEXT]) / len(per_token[BUCKET_TEXT]),
        "reasoning_latent_mass": sum(per_token[BUCKET_REASONING]) / len(per_token[BUCKET_REASONING]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze last-layer output-token attention mass over image, prompt text, and controller trace tokens."
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", default="outputs/attention_mass_summary.json")
    parser.add_argument("--phase4-vlm-checkpoint-path", default=None)
    parser.add_argument("--controller-checkpoint-path", default=None)
    parser.add_argument("--dataset-partition", default=None)
    parser.add_argument("--use-coarse-context", action="store_true", default=False)
    parser.add_argument("--sample-actions", action="store_true", default=False)
    parser.add_argument("--raw-total-mass", action="store_true", default=False)
    parser.add_argument("--attn-implementation", default="eager")
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = apply_model_loading_overrides(config["model"], args)
    if args.attn_implementation:
        model_cfg["attn_implementation"] = args.attn_implementation

    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})
    dataset_cfg = config["dataset"]
    if "action_selection" in inference_cfg:
        model_cfg["action_selection"] = inference_cfg["action_selection"]
    phase3_cfg = config.get("phase3", {})
    phase3_v2_cfg = config.get("phase3_v2", {})
    phase3_v2_enabled = bool(phase3_cfg.get("phase3_v2", phase3_v2_cfg.get("enabled", False)))
    phase3_v2_removes_global = bool(phase3_v2_cfg.get("remove_global", phase3_cfg.get("remove_global", True)))
    if phase3_v2_enabled and phase3_v2_removes_global:
        model_cfg["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
    if "mask_immediate_repeats" in inference_cfg:
        model_cfg["mask_immediate_repeats"] = bool(inference_cfg["mask_immediate_repeats"])

    dataset_partition = args.dataset_partition or inference_cfg.get("dataset_partition", "test")
    split_seed = int(inference_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(inference_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))
    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(dataset_options, limit=args.limit, partition=dataset_partition)
    print(f"Loaded {len(dataset)} examples from partition '{dataset_partition}'")

    model = QwenLVAR(model_cfg)
    phase4_vlm_checkpoint_path = args.phase4_vlm_checkpoint_path or inference_cfg.get(
        "phase4_vlm_checkpoint_path",
        config.get("phase5", {}).get("phase4_vlm_checkpoint_path", ""),
    )
    if phase4_vlm_checkpoint_path:
        loaded = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        print(f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}" if loaded else f"Missing Phase 4 checkpoint: {phase4_vlm_checkpoint_path}")

    controller_checkpoint_path = args.controller_checkpoint_path or inference_cfg.get(
        "controller_checkpoint_path",
        config.get("phase5", {}).get("controller_checkpoint_path", ""),
    )
    if controller_checkpoint_path:
        loaded = load_controller_checkpoint(model, controller_checkpoint_path)
        print(f"Loaded controller checkpoint: {controller_checkpoint_path}" if loaded else f"Missing controller checkpoint: {controller_checkpoint_path}")

    model.eval()
    image_size = inference_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280))
    normalize_over_buckets = not args.raw_total_mass
    sample_rows = []
    image_values: List[float] = []
    text_values: List[float] = []
    reasoning_values: List[float] = []
    skipped_no_output = 0

    for example in tqdm(dataset, total=len(dataset), desc="Analyzing attention"):
        result = run_controller_and_decode(
            model,
            example,
            image_size=image_size,
            use_coarse_context=args.use_coarse_context,
            sample_actions=args.sample_actions,
        )
        masses = last_layer_attention_masses(
            model,
            result["decoded"],
            result["labels"],
            normalize_over_buckets=normalize_over_buckets,
        )
        if masses is None:
            skipped_no_output += 1
            continue
        image_values.append(masses["image_mass"])
        text_values.append(masses["text_mass"])
        reasoning_values.append(masses["reasoning_latent_mass"])
        sample_rows.append(
            {
                "example_id": example.get("id"),
                "generated_text": result["decoded"]["generated_text"],
                "num_steps": len(result["trace"]),
                "num_output_tokens": len(result["decoded"]["generated_ids"]),
                **masses,
            }
        )

    summary = {
        "num_requested": int(args.limit),
        "num_loaded": len(dataset),
        "num_analyzed": len(sample_rows),
        "skipped_no_output": skipped_no_output,
        "dataset_partition": dataset_partition,
        "image_size": image_size,
        "use_coarse_context": bool(args.use_coarse_context),
        "normalize_over_buckets": normalize_over_buckets,
        "phase4_vlm_checkpoint": phase4_vlm_checkpoint_path,
        "controller_checkpoint": controller_checkpoint_path,
        "average_attention": {
            "image_mass": mean_std(image_values),
            "text_mass": mean_std(text_values),
            "reasoning_latent_mass": mean_std(reasoning_values),
        },
        "samples": sample_rows,
    }
    output_path = Path(args.output)
    write_json(output_path, summary)

    avg = summary["average_attention"]
    print("\naverage attention:")
    print(f"image_mass mean/std: {avg['image_mass']['mean']:.6f} / {avg['image_mass']['std']:.6f}")
    print(f"text_mass  mean/std: {avg['text_mass']['mean']:.6f} / {avg['text_mass']['std']:.6f}")
    print(
        "reasoning_latent_mass mean/std: "
        f"{avg['reasoning_latent_mass']['mean']:.6f} / {avg['reasoning_latent_mass']['std']:.6f}"
    )
    print(f"Wrote attention summary to {output_path}")


if __name__ == "__main__":
    main()
