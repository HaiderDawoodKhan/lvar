import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

# Allow running as a script: `python lvar_scripts/diagnose_qwen_lvar_vs_reference.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import (
    add_model_loading_args,
    apply_model_loading_overrides,
    extract_tagged_answer,
    normalize_answer_text,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - optional diagnostic dependency
    process_vision_info = None


TENSOR_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tensor_model_inputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Keep only tensors accepted by the default Qwen generate/forward path."""
    return {
        key: value
        for key, value in batch.items()
        if key in TENSOR_INPUT_KEYS and isinstance(value, torch.Tensor)
    }


def keep_grid_tensors_on_cpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Keep Qwen grid metadata on CPU for the reference path.

    Some Torch/CUDA installs fail on tiny CUDA integer reductions such as
    ``image_grid_thw.prod(-1).tolist()`` inside the HF Qwen2-VL forward path
    when CUDA JIT linker libraries are missing. Qwen only needs these grids as
    metadata, so CPU long tensors are both sufficient and closer to how many
    reference snippets pass them around.
    """
    updated = dict(batch)
    for key in ("image_grid_thw", "video_grid_thw"):
        value = updated.get(key)
        if isinstance(value, torch.Tensor):
            updated[key] = value.detach().cpu().to(dtype=torch.long)
    return updated


def tensor_summary(value: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    summary: Dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
    }
    if value.numel() and value.numel() <= 20:
        summary["values"] = value.detach().cpu().tolist()
    return summary


def tensors_equal(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[bool]:
    if left is None or right is None:
        return None
    if left.shape != right.shape:
        return False
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


def build_messages(
    image: Any,
    question: str,
    add_answer_instruction: bool,
    image_size: Optional[int],
    use_resized_image_metadata: bool,
) -> List[Dict[str, Any]]:
    prompt = str(question)
    if add_answer_instruction:
        prompt = f"{prompt}\nReturn only the final answer inside <answer>...</answer>."

    image_content: Dict[str, Any] = {"type": "image", "image": image}
    if use_resized_image_metadata and image_size is not None:
        image_content["resized_height"] = int(image_size)
        image_content["resized_width"] = int(image_size)
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_reference_inputs(
    model: QwenLVAR,
    image: Any,
    question: str,
    add_answer_instruction: bool,
    image_size: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, str]:
    """
    Build the canonical Qwen reference inputs.

    If qwen-vl-utils is installed, this follows the usual
    apply_chat_template -> process_vision_info -> processor(..., padding=True)
    path. Otherwise it falls back to the processor-only equivalent.
    """
    messages = build_messages(
        image=image,
        question=question,
        add_answer_instruction=add_answer_instruction,
        image_size=image_size,
        use_resized_image_metadata=process_vision_info is not None,
    )
    text = model.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        batch = model.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        source = "qwen_vl_utils.process_vision_info"
    else:
        reference_image = image
        if image_size is not None and reference_image is not None and hasattr(reference_image, "resize"):
            reference_image = reference_image.resize((int(image_size), int(image_size)))
        batch = model.processor(
            text=[text],
            images=[reference_image] if reference_image is not None else None,
            padding=True,
            return_tensors="pt",
        )
        source = "processor_fallback_no_qwen_vl_utils"

    moved = model._move_batch_to_device(dict(batch))
    return moved, messages, text, source


def first_difference(left: Sequence[int], right: Sequence[int]) -> Optional[Dict[str, int]]:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return {"index": index, "left": int(left_id), "right": int(right_id)}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left": int(left[min(len(left), len(right))]) if len(left) > len(right) else -1,
            "right": int(right[min(len(left), len(right))]) if len(right) > len(left) else -1,
        }
    return None


def trim_at_eos(token_ids: Sequence[int], eos_token_id: Optional[int]) -> List[int]:
    trimmed: List[int] = []
    for token_id in token_ids:
        if eos_token_id is not None and int(token_id) == int(eos_token_id):
            break
        trimmed.append(int(token_id))
    return trimmed


def decode_ids(model: QwenLVAR, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    tensor = torch.tensor(list(token_ids), dtype=torch.long)
    return model._decode_ids(tensor)


def unwrap_candidates(module: Any) -> Iterable[Any]:
    """Yield likely PEFT/HF wrapper layers that may expose Qwen helper methods."""
    yielded: set[int] = set()
    stack = [module]
    paths = [
        ("base_model",),
        ("base_model", "model"),
        ("base_model", "model", "model"),
        ("model",),
        ("model", "model"),
        ("language_model",),
    ]
    for path in paths:
        current = module
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            stack.append(current)
    for candidate in stack:
        candidate_id = id(candidate)
        if candidate_id not in yielded:
            yielded.add(candidate_id)
            yield candidate


def compute_qwen_position_ids(
    backbone: Any,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """Best-effort access to Qwen2-VL's MRoPE position-id constructor."""
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    image_grid_thw = batch.get("image_grid_thw")
    video_grid_thw = batch.get("video_grid_thw")
    if input_ids is None:
        return None, None, "input_ids missing"

    errors = []
    for candidate in unwrap_candidates(backbone):
        get_rope_index = getattr(candidate, "get_rope_index", None)
        if get_rope_index is None:
            continue
        try:
            result = get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        except TypeError:
            try:
                result = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            except Exception as exc:  # pragma: no cover - version dependent
                errors.append(f"{type(candidate).__name__}: {type(exc).__name__}: {exc}")
                continue
        except Exception as exc:  # pragma: no cover - version dependent
            errors.append(f"{type(candidate).__name__}: {type(exc).__name__}: {exc}")
            continue

        if isinstance(result, tuple):
            position_ids = result[0]
            rope_deltas = result[1] if len(result) > 1 else None
        else:
            position_ids = result
            rope_deltas = None
        return position_ids, rope_deltas, None

    if errors:
        return None, None, "; ".join(errors)
    return None, None, "no get_rope_index method found on backbone/wrappers"


def top_tokens(model: QwenLVAR, logits: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=min(k, logits.size(-1)), dim=-1)
    probs = torch.softmax(logits.float(), dim=-1).gather(-1, indices)
    rows = []
    for token_id, logit, prob in zip(indices[0].tolist(), values[0].tolist(), probs[0].tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "text": decode_ids(model, [int(token_id)]),
                "logit": float(logit),
                "prob": float(prob),
            }
        )
    return rows


def compare_next_token_logits(
    model: QwenLVAR,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    top_k: int,
) -> Dict[str, Any]:
    reference = reference_logits[:, -1, :].detach().float()
    candidate = candidate_logits[:, -1, :].detach().float()
    delta = (reference - candidate).abs()
    reference_top1 = int(torch.argmax(reference, dim=-1).item())
    candidate_top1 = int(torch.argmax(candidate, dim=-1).item())
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "reference_top1": {
            "token_id": reference_top1,
            "text": decode_ids(model, [reference_top1]),
        },
        "candidate_top1": {
            "token_id": candidate_top1,
            "text": decode_ids(model, [candidate_top1]),
        },
        "top1_match": reference_top1 == candidate_top1,
        "reference_top_tokens": top_tokens(model, reference, top_k),
        "candidate_top_tokens": top_tokens(model, candidate, top_k),
    }


def shorten_exception_message(exc: BaseException, max_chars: int = 1200) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"\n{3,}", "\n\n", message).strip()
    if len(message) > max_chars:
        message = message[:max_chars].rstrip() + "\n... [truncated]"
    return message


def run_reference_forward(model: QwenLVAR, model_inputs: Dict[str, torch.Tensor]) -> Any:
    try:
        return model.backbone(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    except Exception as exc:
        short = shorten_exception_message(exc)
        hint = ""
        if "libnvJitLink" in short or "nvJitLink" in short:
            hint = (
                "\n\nHint: this is an environment/CUDA library issue, not a model-output "
                "difference. The diagnostic now keeps image_grid_thw on CPU to avoid the "
                "common Qwen metadata CUDA reduction, but if the visual tower itself still "
                "triggers CUDA JIT, fix the CUDA/PyTorch install or run on a node/env with "
                "the matching libnvJitLink available."
            )
        raise RuntimeError(f"Default Qwen reference forward failed.\n{short}{hint}") from exc


def run_reference_generate(
    model: QwenLVAR,
    model_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
) -> torch.Tensor:
    try:
        return model.backbone.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        short = shorten_exception_message(exc)
        hint = ""
        if "libnvJitLink" in short or "nvJitLink" in short:
            hint = (
                "\n\nHint: this is an environment/CUDA library issue during default "
                "Qwen generation. Try the patched script first; if it still fails, "
                "the active PyTorch/CUDA stack cannot find the matching libnvJitLink."
            )
        raise RuntimeError(f"Default Qwen reference generate failed.\n{short}{hint}") from exc


def load_example(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    if args.image_path or args.prompt:
        if not args.image_path or args.prompt is None:
            raise ValueError("--image-path and --prompt must be provided together for an explicit example.")
        image = Image.open(args.image_path).convert("RGB")
        return {
            "id": args.example_id or Path(args.image_path).stem,
            "image": image,
            "question": args.prompt,
            "gold_answer": normalize_answer_text(args.gold_answer),
        }

    dataset_cfg = dict(config["dataset"])
    if args.dataset_limit is not None:
        dataset_cfg["limit"] = args.dataset_limit
    dataset = build_dataset(dataset_cfg, partition=args.partition)
    example = dataset[int(args.index)]
    return {
        "id": example.get("id", args.index),
        "image": example["image"],
        "question": example["question"],
        "gold_answer": example.get("gold_answer", ""),
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose divergence between the LVAR embedding-prefix decode path and "
            "the default Qwen2-VL reference generate path using the same loaded LVAR weights."
        )
    )
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--partition", default=None, help="Dataset partition/split override, e.g. validation or test.")
    parser.add_argument("--index", type=int, default=0, help="Dataset row index when --image-path/--prompt are not used.")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--image-path", default=None, help="Optional explicit image path instead of loading a dataset row.")
    parser.add_argument("--prompt", default=None, help="Optional explicit prompt paired with --image-path.")
    parser.add_argument("--example-id", default=None)
    parser.add_argument("--gold-answer", default="")
    parser.add_argument("--image-size", type=int, default=None, help="Resize image before both paths, matching LVAR mining/eval.")
    parser.add_argument(
        "--add-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the LVAR tagged-answer instruction to the user prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default=None, help="Optional path for the full diagnostic report.")
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    if args.max_new_tokens is not None:
        config["model"]["max_answer_tokens"] = int(args.max_new_tokens)

    example = load_example(args, config)
    model = QwenLVAR(config["model"])
    model.eval()

    with torch.no_grad():
        lvar_prepared = model.prepare_inputs(
            example["image"],
            example["question"],
            add_answer_instruction=bool(args.add_answer_instruction),
            image_size=args.image_size,
        )
        lvar_rendered_chat_template = model.processor.apply_chat_template(
            lvar_prepared["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        reference_prepared, reference_messages, reference_rendered_chat_template, reference_preprocess_source = (
            prepare_reference_inputs(
                model,
                image=example["image"],
                question=example["question"],
                add_answer_instruction=bool(args.add_answer_instruction),
                image_size=args.image_size,
            )
        )
        reference_model_inputs = keep_grid_tensors_on_cpu(tensor_model_inputs(reference_prepared))

        image_tokens = model.get_projected_image_tokens(lvar_prepared)
        lvar_prepared_with_tokens = dict(lvar_prepared)
        lvar_prepared_with_tokens["projected_image_tokens"] = image_tokens
        lvar_inputs_embeds, lvar_attention_mask = model._build_multimodal_embeddings(lvar_prepared_with_tokens)
        lvar_state = {
            "inputs_embeds": lvar_inputs_embeds,
            "attention_mask": lvar_attention_mask,
            "latent_pos": None,
            "act_pos": None,
            "trace_all_positions": [],
            "trace_visual_positions": [],
        }

        reference_forward = run_reference_forward(model, reference_model_inputs)
        lvar_forward = model.backbone(
            inputs_embeds=lvar_inputs_embeds,
            attention_mask=lvar_attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )

        position_ids, rope_deltas, position_error = compute_qwen_position_ids(model.backbone, reference_model_inputs)
        lvar_with_reference_positions_comparison = None
        if position_ids is not None and position_ids.shape[-1] == lvar_inputs_embeds.shape[1]:
            lvar_positioned_forward = model.backbone(
                inputs_embeds=lvar_inputs_embeds,
                attention_mask=lvar_attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            lvar_with_reference_positions_comparison = compare_next_token_logits(
                model,
                reference_forward.logits,
                lvar_positioned_forward.logits,
                top_k=args.top_k,
            )

        lvar_decoded = model.decode_answer(lvar_state)
        generated = run_reference_generate(
            model,
            reference_model_inputs,
            max_new_tokens=model.max_answer_tokens,
        )

    prompt_length = int(reference_prepared["input_ids"].shape[1])
    reference_generated_ids_raw = [int(token_id) for token_id in generated[0, prompt_length:].detach().cpu().tolist()]
    reference_generated_ids = trim_at_eos(reference_generated_ids_raw, model.eos_token_id)
    lvar_generated_ids = [int(token_id) for token_id in lvar_decoded["generated_ids"]]

    lvar_vs_reference_logits = compare_next_token_logits(
        model,
        reference_forward.logits,
        lvar_forward.logits,
        top_k=args.top_k,
    )

    report: Dict[str, Any] = {
        "example": {
            "id": example["id"],
            "question": example["question"],
            "gold_answer": example.get("gold_answer", ""),
        },
        "preprocessing": {
            "add_answer_instruction": bool(args.add_answer_instruction),
            "image_size": args.image_size,
            "reference_source": reference_preprocess_source,
            "reference_grid_tensors_kept_on_cpu": True,
            "lvar_rendered_chat_template": lvar_rendered_chat_template,
            "reference_rendered_chat_template": reference_rendered_chat_template,
            "rendered_chat_template_equal": lvar_rendered_chat_template == reference_rendered_chat_template,
            "messages_equal": lvar_prepared.get("messages") == reference_messages,
            "input_ids_equal": tensors_equal(lvar_prepared.get("input_ids"), reference_prepared.get("input_ids")),
            "attention_mask_equal": tensors_equal(
                lvar_prepared.get("attention_mask"),
                reference_prepared.get("attention_mask"),
            ),
            "image_grid_thw_equal": tensors_equal(
                lvar_prepared.get("image_grid_thw"),
                reference_prepared.get("image_grid_thw"),
            ),
            "lvar_input_ids": tensor_summary(lvar_prepared.get("input_ids")),
            "reference_input_ids": tensor_summary(reference_prepared.get("input_ids")),
            "lvar_attention_mask": tensor_summary(lvar_prepared.get("attention_mask")),
            "reference_attention_mask": tensor_summary(reference_prepared.get("attention_mask")),
            "lvar_pixel_values": tensor_summary(lvar_prepared.get("pixel_values")),
            "reference_pixel_values": tensor_summary(reference_prepared.get("pixel_values")),
            "lvar_image_grid_thw": tensor_summary(lvar_prepared.get("image_grid_thw")),
            "reference_image_grid_thw": tensor_summary(reference_prepared.get("image_grid_thw")),
            "lvar_inputs_embeds": tensor_summary(lvar_inputs_embeds),
            "lvar_embedding_attention_mask": tensor_summary(lvar_attention_mask),
            "projected_image_tokens": tensor_summary(image_tokens),
            "premerge_grid": model._current_premerge_grid,
            "postmerge_grid": model._current_postmerge_grid,
        },
        "position_probe": {
            "position_ids": tensor_summary(position_ids),
            "rope_deltas": tensor_summary(rope_deltas),
            "error": position_error,
            "lvar_with_reference_position_ids_next_token": lvar_with_reference_positions_comparison,
        },
        "next_token_comparisons": {
            "reference_vs_lvar_embedding_prefix": lvar_vs_reference_logits,
        },
        "generation": {
            "reference": {
                "generated_ids_raw": reference_generated_ids_raw,
                "generated_ids": reference_generated_ids,
                "generated_text": decode_ids(model, reference_generated_ids),
                "answer": extract_tagged_answer(decode_ids(model, reference_generated_ids)),
            },
            "lvar_embedding_prefix": {
                "generated_ids": lvar_generated_ids,
                "generated_text": lvar_decoded["generated_text"],
                "answer": lvar_decoded["answer"],
            },
            "first_token_difference": first_difference(reference_generated_ids, lvar_generated_ids),
            "exact_token_match": reference_generated_ids == lvar_generated_ids,
            "exact_text_match": decode_ids(model, reference_generated_ids) == lvar_decoded["generated_text"],
        },
    }

    print_section("Example")
    print(f"ID: {report['example']['id']}")
    print(f"Gold answer: {report['example']['gold_answer']}")
    print(report["example"]["question"])

    print_section("Preprocessing")
    print(f"reference source: {report['preprocessing']['reference_source']}")
    print(f"chat template equal: {report['preprocessing']['rendered_chat_template_equal']}")
    print(f"messages equal: {report['preprocessing']['messages_equal']}")
    print(f"input_ids equal: {report['preprocessing']['input_ids_equal']}")
    print(f"attention_mask equal: {report['preprocessing']['attention_mask_equal']}")
    print(f"image_grid_thw equal: {report['preprocessing']['image_grid_thw_equal']}")
    print(f"LVAR input_ids: {report['preprocessing']['lvar_input_ids']}")
    print(f"Reference input_ids: {report['preprocessing']['reference_input_ids']}")
    print(f"LVAR pixel_values: {report['preprocessing']['lvar_pixel_values']}")
    print(f"Reference pixel_values: {report['preprocessing']['reference_pixel_values']}")
    print(f"LVAR image_grid_thw: {report['preprocessing']['lvar_image_grid_thw']}")
    print(f"Reference image_grid_thw: {report['preprocessing']['reference_image_grid_thw']}")
    print(f"LVAR inputs_embeds: {report['preprocessing']['lvar_inputs_embeds']}")
    print(f"projected_image_tokens: {report['preprocessing']['projected_image_tokens']}")
    print(f"pre/post merge grid: {model._current_premerge_grid} -> {model._current_postmerge_grid}")

    print_section("Next-token logits")
    print(json.dumps(report["next_token_comparisons"], indent=2))
    if lvar_with_reference_positions_comparison is not None:
        print_section("LVAR prefix with Qwen reference position_ids")
        print(json.dumps(lvar_with_reference_positions_comparison, indent=2))
    elif position_error:
        print_section("Position probe")
        print(f"Could not compute/apply Qwen position_ids: {position_error}")

    print_section("Generation")
    print("Reference generated text:")
    print(report["generation"]["reference"]["generated_text"])
    print(f"Reference answer: {report['generation']['reference']['answer']}")
    print("\nLVAR embedding-prefix generated text:")
    print(report["generation"]["lvar_embedding_prefix"]["generated_text"])
    print(f"LVAR answer: {report['generation']['lvar_embedding_prefix']['answer']}")
    print(f"\nExact token match: {report['generation']['exact_token_match']}")
    print(f"First token difference: {report['generation']['first_token_difference']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nSaved diagnostic report to: {output_path}")


if __name__ == "__main__":
    main()
