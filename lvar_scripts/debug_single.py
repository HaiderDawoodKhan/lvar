import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

# Allow running as a script: `python scripts/debug_single.py ...`.
# Direct execution otherwise adds only `scripts/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.grpo_training import load_controller_checkpoint, load_vlm_lora_checkpoint
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import ACTION_NAMES_NO_GLOBAL, add_model_loading_args, apply_model_loading_overrides, format_trace_step


def load_config(config_path: str):
    """Load YAML config file used by all script entrypoints."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resize_image(image, image_size: int):
    """Resize PIL-like images to the fixed debug/mining resolution."""
    if image is not None and hasattr(image, "resize"):
        return image.resize((int(image_size), int(image_size)))
    return image


def print_visual_token_stats(model: QwenLVAR, image, question: str, image_size: int) -> None:
    """Print visual-bank sizes before running the answer generation path."""
    with torch.no_grad():
        prepared = model.prepare_inputs(
            image,
            question,
            image_size=image_size,
        )
        image_tokens = model.get_projected_image_tokens(prepared)
        bank = model.build_visual_bank(image_tokens)

    patch_tokens = image_tokens.squeeze(0) if image_tokens.dim() == 3 else image_tokens
    regions = bank["regions"]
    global_tokens = bank["global"]

    print(f"Image size: {image_size}x{image_size}")
    print(f"Image tokens: {patch_tokens.size(0)} tokens, dim {patch_tokens.size(-1)}")
    print(f"Regions: {regions.size(0)} regions, dim {regions.size(-1)}")
    print(f"Global tokens: {global_tokens.size(0)} tokens, dim {global_tokens.size(-1)}")
    if model._current_postmerge_grid is not None:
        print(f"Post-merge image grid: {model._current_postmerge_grid}")


def main() -> None:
    """Run one example and print a human-readable LVAR reasoning trace."""
    # CLI inputs: config path plus optional explicit example index override.
    parser = argparse.ArgumentParser(description="Run a single LVAR debug example.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="If set, run 6 stochastic LVAR rollouts for the same question.",
    )
    parser.add_argument(
        "--phase4-vlm-checkpoint-path",
        default=None,
        help="Override debug/phase5/phase3 Phase 4 VLM LoRA checkpoint path.",
    )
    parser.add_argument(
        "--controller-checkpoint-path",
        default=None,
        help="Override debug/phase5/phase3 controller checkpoint path.",
    )
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    dataset_cfg = config["dataset"]
    debug_cfg = config.get("debug", {})
    phase3_cfg = config.get("phase3", {})
    phase3_v2_cfg = config.get("phase3_v2", {})
    phase5_cfg = config.get("phase5", {})
    inference_cfg = config.get("inference", {})

    phase3_v2_enabled = bool(phase3_cfg.get("phase3_v2", phase3_v2_cfg.get("enabled", False)))
    phase3_v2_removes_global = bool(phase3_v2_cfg.get("remove_global", phase3_cfg.get("remove_global", True)))
    use_no_global_controller = phase3_v2_enabled and phase3_v2_removes_global
    if use_no_global_controller:
        config["model"]["controller_action_names"] = list(ACTION_NAMES_NO_GLOBAL.values())
    if "mask_immediate_repeats" in inference_cfg:
        config["model"]["mask_immediate_repeats"] = bool(inference_cfg["mask_immediate_repeats"])
    if "action_selection" in inference_cfg:
        config["model"]["action_selection"] = inference_cfg["action_selection"]

    phase4_vlm_checkpoint_path = (
        args.phase4_vlm_checkpoint_path
        or debug_cfg.get("phase4_vlm_checkpoint_path")
        or phase5_cfg.get("phase4_vlm_checkpoint_path")
        or phase3_cfg.get("phase4_vlm_checkpoint_path")
    )
    controller_checkpoint_path = (
        args.controller_checkpoint_path
        or debug_cfg.get("controller_checkpoint_path")
        or phase5_cfg.get("controller_checkpoint_path")
        or phase3_cfg.get("controller_checkpoint_path")
        or inference_cfg.get("controller_checkpoint_path")
    )

    # Build dataset and model exactly like main inference path so debug reflects real behavior.
    dataset = build_dataset(dataset_cfg, limit=dataset_cfg.get("limit"))
    index = args.index if args.index is not None else random.randrange(len(dataset))
    example = dataset[index]

    model = QwenLVAR(config["model"])
    if phase4_vlm_checkpoint_path:
        loaded = load_vlm_lora_checkpoint(model, phase4_vlm_checkpoint_path)
        print(
            f"Loaded Phase 4 VLM LoRA checkpoint: {phase4_vlm_checkpoint_path}"
            if loaded
            else f"Phase 4 VLM LoRA checkpoint not found: {phase4_vlm_checkpoint_path}"
        )
    if controller_checkpoint_path:
        loaded = load_controller_checkpoint(model, controller_checkpoint_path)
        print(
            f"Loaded controller checkpoint: {controller_checkpoint_path}"
            if loaded
            else f"Controller checkpoint not found: {controller_checkpoint_path}"
        )
    image_size = int(debug_cfg.get("image_size", config.get("phase2", {}).get("image_size", 280)))
    image = resize_image(example["image"], image_size)
    print_visual_token_stats(model, image, example["question"], image_size)

    uses_sampling = model._inference_uses_sampling() if not args.rollout else True
    lvar_output = None
    rollout_outputs = []
    if args.rollout:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(6):
                rollout_outputs.append(
                    model.forward(
                        image,
                        example["question"],
                        sample_actions=True,
                        image_size=image_size,
                    )
                )
        model.train(was_training)
    else:
        lvar_output = model.generate_lvar(image, example["question"], image_size=image_size)
    # baseline_output = model.generate_baseline(example["image"], example["question"])

    # Print structured trace for quick inspection of action choices and loop length.
    print(f"Example ID: {example['id']}")
    print(
        "Sampling strategy: "
        f"{model.action_selection} "
        f"({'stochastic sampling' if uses_sampling else 'deterministic argmax'})"
    )
    print(f"Question: {example['question']}")
    print(f"Gold answer: {example['gold_answer']}")
    if args.rollout:
        for rollout_idx, rollout in enumerate(rollout_outputs, start=1):
            print(f"Rollout {rollout_idx}/6:")
            print("Reasoning trace:")
            for step in rollout["trace"]:
                print(f"  {format_trace_step(step)}")
            print(f"LVAR answer: {rollout['answer']}")
            print(f"Generated token IDs: {rollout['generated_ids']}")
            print(f"Num steps: {rollout['num_steps']}")
            print(f"Stopped: {rollout['stopped']}")
    else:
        print("Reasoning trace:")
        for step in lvar_output["trace"]:
            print(f"  {format_trace_step(step)}")
        print(f"LVAR answer: {lvar_output['prediction']}")
        print(f"Generated token IDs: {lvar_output['generated_ids']}")
    # print(f"Baseline answer: {baseline_output['prediction']}")
    # print(f"Reasoning steps: {lvar_output['num_steps']}")


if __name__ == "__main__":
    main()
