import math
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from lvar.utils import (
    ACTION_GLOBAL,
    ACTION_NAMES,
    ACTION_PATCH,
    ACTION_REGION,
    ACTION_STOP,
    ACTION_THINK,
    extract_tagged_answer,
    normalize_action_names,
)
from lvar.trace_attention_boost import TraceAttentionBoostRuntime, TraceBoostConfig

try:
    from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover - exercised in environments without HF deps
    AutoProcessor = None
    AutoTokenizer = None
    Qwen2VLForConditionalGeneration = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - exercised in environments without PEFT
    LoraConfig = None
    get_peft_model = None


class ControllerHead(nn.Module):
    """Small policy head that scores action type and fixed visual-unit indices."""

    def __init__(
        self,
        hidden_size: int,
        num_actions: int,
        use_control_tokens: bool = False,
        controller_num_states: int = 1,
        num_regions: int = 25,
        num_patches: int = 100,
    ) -> None:
        """
        Args:
            hidden_size: Backbone hidden width used by latent/act states.
            num_actions: Number of high-level controller actions.
            use_control_tokens: Whether latent/act control tokens are enabled.
            controller_num_states: Number of hidden-state positions read by the
                controller in tokenless mode.
            num_regions: Fixed number of region classes.
            num_patches: Fixed number of patch classes.

        Attributes:
            fuse: MLP combining controller state embeddings.
            type_head: Produces logits for THINK/STOP/GLOBAL/REGION/PATCH.
            region_head: Produces logits over fixed region indices.
            patch_head: Produces logits over fixed patch indices.
        """
        super().__init__()
        self.use_control_tokens = use_control_tokens
        self.num_regions = int(num_regions)
        self.num_patches = int(num_patches)
        if self.num_regions <= 0:
            raise ValueError("num_regions must be greater than 0.")
        if self.num_patches <= 0:
            raise ValueError("num_patches must be greater than 0.")
        input_factor = 3 if use_control_tokens else controller_num_states + 1
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * input_factor, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.type_head = nn.Linear(hidden_size, num_actions)
        self.region_head = nn.Linear(hidden_size, self.num_regions)
        self.patch_head = nn.Linear(hidden_size, self.num_patches)

    def forward(
        self,
        state_hidden: torch.Tensor,
        step_hidden: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        act_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action-type logits and index logits for regions and patches."""
        # Fuse controller context from current reasoning state and recurrent step id.
        if self.use_control_tokens:
            if act_hidden is None:
                raise ValueError("act_hidden is required when control tokens are enabled.")
            controller_inputs = [state_hidden, act_hidden, step_hidden]
        else:
            controller_inputs = [state_hidden, step_hidden]
        controller_hidden = self.fuse(torch.cat(controller_inputs, dim=-1))
        type_logits = self.type_head(controller_hidden)
        return type_logits, self.region_head(controller_hidden), self.patch_head(controller_hidden)


class QwenLVAR(nn.Module):
    """Minimal LVAR wrapper around Qwen2-VL with recurrent controller actions."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        backbone: Optional[nn.Module] = None,
        processor: Optional[Any] = None,
    ) -> None:
        """
        Initialize frozen Qwen2-VL plus trainable LVAR-specific parameters.

        Attributes:
            cfg/model_id/device/dtype: Runtime config values.
            max_steps: Hard upper bound on recurrent reasoning iterations.
            region_window: Non-overlapping region pooling window over patch grid.
            max_answer_tokens: Greedy decode length cap.
            action_selection: Stored config knob; runtime path uses sample flag.
            backbone: Frozen Qwen2-VL (or injected stub backbone in tests).
            processor: HF processor used to build multimodal prompts.
            hidden_size: LM hidden width shared by added parameters.
            image_token_id/eos_token_id: Backbone token ids used in embedding/decode logic.
            latent_token/act_token: Learned recurrent control tokens.
            global_pool/region_pool: Attention scorers for visual-bank pooling.
            controller: Policy head over action type + region/patch indices.
            step_embedding: Embedding table for recurrent step index.
            _current_image_grid: Cached (H, W) grid read from processor outputs.
        """
        super().__init__()
        # Basic runtime config.
        self.cfg = cfg
        self.trace_boost_config = TraceBoostConfig.from_value(cfg.get("trace_boost"))
        self.model_id = cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self.dtype = self._resolve_dtype(cfg.get("dtype", "auto"))
        self.max_steps = int(cfg.get("max_steps", 4))
        self.region_window = cfg.get("region_window", 2)
        self.max_answer_tokens = int(cfg.get("max_answer_tokens", 16))
        self.action_selection = cfg.get("action_selection", "argmax")
        self.controller_temperature = float(cfg.get("controller_temperature", 1.0))
        self.controller_num_regions = int(cfg.get("controller_num_regions", 25))
        self.controller_num_patches = int(cfg.get("controller_num_patches", 100))
        self.action_names = normalize_action_names(cfg.get("controller_action_names"))
        self.action_name_to_id = {name: idx for idx, name in self.action_names.items()}
        self.mask_immediate_repeats = bool(cfg.get("mask_immediate_repeats", False))
        self.pooling = self._resolve_pooling(cfg.get("pooling", "mean"))
        self.use_control_tokens = bool(cfg.get("use_control_tokens", False))
        self.think_append_hidden = bool(cfg.get("think_append_hidden", True))
        self.controller_num_states = int(cfg.get("controller_context_window", 3))
        self.controller_max_steps = int(cfg.get("controller_max_steps", self.max_steps))
        self.checkpoint_path = cfg.get("checkpoint_path") or cfg.get("ivtlr_checkpoint_path")
        self.use_checkpoint = bool(cfg.get("use_checkpoint", bool(self.checkpoint_path)))

        if self.controller_temperature <= 0.0:
            raise ValueError("controller_temperature must be greater than 0.")
        if self.controller_num_regions <= 0:
            raise ValueError("controller_num_regions must be greater than 0.")
        if self.controller_num_patches <= 0:
            raise ValueError("controller_num_patches must be greater than 0.")
        if self.controller_num_states <= 0:
            raise ValueError("controller_context_window must be greater than 0.")
        if self.controller_max_steps <= 0:
            raise ValueError("controller_max_steps must be greater than 0.")
        if self.use_checkpoint and not self.checkpoint_path:
            raise ValueError("use_checkpoint is true but no checkpoint_path was provided.")

        # Load real HF components unless tests inject a stub backbone/processor pair.
        if backbone is None:
            if Qwen2VLForConditionalGeneration is None or AutoProcessor is None:
                raise ImportError(
                    "transformers is required to instantiate the real Qwen2-VL backbone. "
                    "Install the requirements first."
            )
            self.processor = self._load_processor()
            self.backbone = self._load_backbone()
            self._maybe_resize_token_embeddings()
            self._maybe_apply_lora()
            self._maybe_load_backbone_checkpoint()
        else:
            self.backbone = backbone
            self.processor = processor
            if self.processor is None:
                raise ValueError("A processor must be provided when injecting a custom backbone.")

        # Read backbone metadata needed by custom multimodal and decode paths.
        if hasattr(self.backbone, "to"):
            self.backbone.to(self.device)
        self.hidden_size = getattr(
            self.backbone.config,
            "hidden_size",
            self.backbone.get_input_embeddings().embedding_dim,
        )
        self.image_token_id = getattr(self.backbone.config, "image_token_id", None)
        self.eos_token_id = getattr(self.backbone.config, "eos_token_id", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if self.eos_token_id is None and tokenizer is not None:
            self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

        # Trainable LVAR additions.
        self.latent_token = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.act_token = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.global_pool = nn.Linear(self.hidden_size, 1)
        self.region_pool = nn.Linear(self.hidden_size, 1)
        self.controller_state_norm = nn.LayerNorm(self.hidden_size)
        self.controller = ControllerHead(
            self.hidden_size,
            len(self.action_names),
            use_control_tokens=self.use_control_tokens,
            controller_num_states=self.controller_num_states,
            num_regions=self.controller_num_regions,
            num_patches=self.controller_num_patches,
        )
        self.step_embedding = nn.Embedding(self.controller_max_steps, self.hidden_size)

        # Freeze backbone to keep training focused on controller-driven reasoning behavior.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self._current_premerge_grid: Optional[Tuple[int, int]] = None
        self._current_postmerge_grid: Optional[Tuple[int, int]] = None
        self._current_image_grid: Optional[Tuple[int, int]] = None
        self.trace_boost_runtime = TraceAttentionBoostRuntime(self.trace_boost_config)
        self.trace_boost_runtime.install(self.backbone)
        self.to(self.device)

    def _load_processor(self) -> Any:
        """Load processor/tokenizer and register latent special tokens if requested."""
        processor = AutoProcessor.from_pretrained(self.model_id)
        if not bool(self.cfg.get("add_latent_special_tokens", self.use_checkpoint)):
            return processor
        if AutoTokenizer is None:
            raise ImportError("transformers AutoTokenizer is required to add latent special tokens.")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=False,
            trust_remote_code=True,
            padding_side="right",
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|start-latent|>",
                    "<|end-latent|>",
                    "<|latent|>",
                ]
            }
        )
        processor.tokenizer = tokenizer
        return processor

    def _load_backbone(self) -> nn.Module:
        """Load the Qwen2-VL backbone with config-compatible HF kwargs."""
        backbone_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(self.cfg.get("trust_remote_code", True)),
        }
        attn_implementation = self.cfg.get("attn_implementation")
        if self.trace_boost_config.enabled:
            attn_implementation = "eager"
        if attn_implementation is not None:
            backbone_kwargs["attn_implementation"] = attn_implementation
        if self.device.type == "cuda":
            backbone_kwargs["torch_dtype"] = self.dtype
            if bool(self.cfg.get("device_map_cuda", False)):
                backbone_kwargs["device_map"] = "cuda"
        return Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **backbone_kwargs,
        )

    def _maybe_apply_lora(self) -> None:
        """Wrap the backbone with LoRA adapters when loading an IVTLR/PEFT checkpoint."""
        lora_cfg = self.cfg.get("lora", {})
        use_lora = bool(lora_cfg.get("enabled", self.use_checkpoint))
        if not use_lora:
            return
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required to load LoRA/IVTLR checkpoints. Install peft first.")
        config = LoraConfig(
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 16))),
            lora_dropout=float(lora_cfg.get("lora_dropout", lora_cfg.get("dropout", 0.05))),
            bias=lora_cfg.get("bias", "none"),
            inference_mode=bool(lora_cfg.get("inference_mode", False)),
        )
        self.backbone = get_peft_model(self.backbone, config)

    def _get_backbone_child(self, module: nn.Module, name: str) -> Optional[nn.Module]:
        """Safely fetch a child module across PEFT/wrapper implementations."""
        try:
            child = getattr(module, name)
        except AttributeError:
            return None
        return child if isinstance(child, nn.Module) else None

    def _resolve_visual_encoder(self, required: bool = True) -> Optional[nn.Module]:
        """
        Locate Qwen2-VL's vision encoder across HF/PEFT version differences.

        Some Transformers releases expose the encoder as ``backbone.visual``;
        others place it under nested modules such as ``backbone.model.visual``.
        PEFT wrappers add another layer, so mining utilities should not assume a
        single attribute path.
        """
        cached = getattr(self, "_visual_encoder_cache", None)
        if cached is not None:
            return cached

        queue: list[tuple[nn.Module, int]] = [(self.backbone, 0)]
        seen: set[int] = set()
        visual_names = ("visual", "vision_tower", "vision_model")
        wrapper_names = ("base_model", "model", "module")

        while queue:
            module, depth = queue.pop(0)
            module_id = id(module)
            if module_id in seen or depth > 4:
                continue
            seen.add(module_id)

            for name in visual_names:
                child = self._get_backbone_child(module, name)
                if child is not None:
                    self._visual_encoder_cache = child
                    return child

            for name in wrapper_names:
                child = self._get_backbone_child(module, name)
                if child is not None:
                    queue.append((child, depth + 1))

        try:
            for name, child in self.backbone.named_modules():
                if name.endswith(".visual") or name == "visual":
                    self._visual_encoder_cache = child
                    return child
        except Exception:
            pass

        if required:
            raise ValueError(
                "Could not locate the Qwen2-VL visual encoder. Checked common paths "
                "including backbone.visual, backbone.model.visual, and PEFT base_model wrappers."
            )
        return None

    def _resolve_spatial_merge_size(self) -> int:
        visual = self._resolve_visual_encoder(required=False)
        if visual is not None and hasattr(visual, "spatial_merge_size"):
            return int(getattr(visual, "spatial_merge_size", 1))

        config = getattr(self.backbone, "config", None)
        vision_config = getattr(config, "vision_config", None)
        for owner in (vision_config, config):
            if owner is not None and hasattr(owner, "spatial_merge_size"):
                return int(getattr(owner, "spatial_merge_size", 1))
        return 1

    def _maybe_resize_token_embeddings(self) -> None:
        """Resize embeddings after special tokens are attached to the processor tokenizer."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(self.backbone, "resize_token_embeddings"):
            return
        self.backbone.resize_token_embeddings(len(tokenizer))

    def _clean_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove wrappers used by DDP/IVTLR so keys match the PEFT-wrapped backbone."""
        clean_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "base_causallm."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            clean_state_dict[new_key] = value
        return clean_state_dict

    def _align_checkpoint_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Align known Qwen/PEFT wrapper-depth differences against target keys."""
        aligned_state_dict = {}
        target_keys = set(target_state_dict)
        for key, value in state_dict.items():
            candidate_keys = [key]
            if key.startswith("base_model.model."):
                candidate_keys.append("base_model.model.model." + key[len("base_model.model.") :])
            if key.startswith("model."):
                candidate_keys.append("base_model.model.model." + key[len("model.") :])
            for prefix in ("base_model.model.model.", "base_model.model."):
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    if suffix.startswith(("embed_tokens.", "layers.", "norm.", "rotary_emb.")):
                        candidate_keys.append(prefix + "language_model." + suffix)

            aligned_key = key
            for candidate_key in candidate_keys:
                if candidate_key in target_keys:
                    aligned_key = candidate_key
                    break
            aligned_state_dict[aligned_key] = value
        return aligned_state_dict

    def _maybe_load_backbone_checkpoint(self) -> None:
        """Load an IVTLR/PEFT checkpoint into the backbone before LVAR modules are added."""
        if not self.use_checkpoint:
            return
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clean_state_dict = self._clean_checkpoint_state_dict(state_dict)
        aligned_state_dict = self._align_checkpoint_state_dict(clean_state_dict, self.backbone.state_dict())
        missing, unexpected = self.backbone.load_state_dict(aligned_state_dict, strict=False)
        print(f"Loaded backbone checkpoint: {self.checkpoint_path}")
        print("Missing backbone keys:", len(missing))
        print("Unexpected backbone keys:", len(unexpected))
        print("First missing backbone keys:", missing[:20])
        print("First unexpected backbone keys:", unexpected[:20])
        if bool(self.cfg.get("merge_lora", False)):
            if not hasattr(self.backbone, "merge_and_unload"):
                raise ValueError("merge_lora is true, but the backbone is not a mergeable PEFT model.")
            self.backbone = self.backbone.merge_and_unload()

    def train(self, mode: bool = True) -> "QwenLVAR":
        """
        Keep backbone in eval mode even during training.

        We still allow gradients through custom parameters, but backbone weights
        remain frozen and should not switch to dropout/training behavior.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def _resolve_device(self, device_name: str) -> torch.device:
        """Resolve user config to an actual torch.device."""
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        """Map string dtype aliases from config into torch dtypes."""
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(str(dtype_name).lower(), torch.float32)

    def _resolve_pooling(self, pooling: str) -> str:
        """Validate and normalize the visual-bank pooling mode."""
        mode = str(pooling).strip().lower()
        if mode not in {"attention", "mean", "max"}:
            raise ValueError("pooling must be one of: attention, mean, max.")
        return mode

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor fields to model device while keeping metadata untouched."""
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings with model-compatible dtype/device."""
        embeddings = self.backbone.get_input_embeddings()(input_ids.to(self.device))
        return embeddings.to(dtype=self.latent_token.dtype)

    def _attention_pool(self, tokens: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        """Single-query attention pooling used for global/region visual summaries."""
        weights = torch.softmax(scorer(tokens).squeeze(-1), dim=0)
        return torch.sum(weights.unsqueeze(-1) * tokens, dim=0)

    def _pool_tokens(
        self,
        tokens: torch.Tensor,
        scorer: Optional[nn.Linear] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Pool a token set into one vector using attention, mean, or max pooling."""
        pool_mode = self.pooling if mode is None else self._resolve_pooling(mode)
        if pool_mode == "attention":
            if scorer is None:
                raise ValueError("attention pooling requires a scorer.")
            return self._attention_pool(tokens, scorer)
        if pool_mode == "mean":
            return tokens.mean(dim=0)
        return tokens.max(dim=0).values

    def _pad_patch_grid_for_regions(
        self,
        patch_grid: torch.Tensor,
        region_h: int,
        region_w: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Pad grid by duplicating border patches so region windows tile exactly."""
        grid_h, grid_w, _ = patch_grid.shape
        pad_h = (-grid_h) % region_h
        pad_w = (-grid_w) % region_w
        if pad_h == 0 and pad_w == 0:
            return patch_grid, grid_h, grid_w

        padded_grid = patch_grid
        if pad_h > 0:
            padded_rows = padded_grid[-1:, :, :].expand(pad_h, -1, -1)
            padded_grid = torch.cat([padded_grid, padded_rows], dim=0)
        if pad_w > 0:
            padded_cols = padded_grid[:, -1:, :].expand(-1, pad_w, -1)
            padded_grid = torch.cat([padded_grid, padded_cols], dim=1)
        return padded_grid, grid_h + pad_h, grid_w + pad_w

    def _resolve_image_grids(self, image_grid_thw: torch.Tensor) -> tuple[tuple[int,int], tuple[int,int]]:
        if image_grid_thw.dim() == 2:
            H = int(image_grid_thw[0, -2].item())
            W = int(image_grid_thw[0, -1].item())
        else:
            H = int(image_grid_thw[-2].item())
            W = int(image_grid_thw[-1].item())

        merge = self._resolve_spatial_merge_size()
        if H % merge != 0 or W % merge != 0:
            raise ValueError(f"Pre-merge grid {(H,W)} not divisible by merge size {merge}.")

        return (H, W), (H // merge, W // merge)

    def _build_multimodal_embeddings(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build embedding-space prefix where image placeholder tokens are replaced
        by projected image vectors from the vision encoder.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        embeddings = self._embed_input_ids(input_ids)
        image_tokens = batch.get("projected_image_tokens")
        if image_tokens is not None:
            if self.image_token_id is None:
                raise ValueError("The backbone config does not expose image_token_id.")
            image_tokens = image_tokens.squeeze(0) if image_tokens.dim() == 3 else image_tokens
            image_mask = input_ids == self.image_token_id
            num_image_slots = int(image_mask.sum().item())
            if num_image_slots != image_tokens.size(0):
                raise ValueError(
                    f"Expected {num_image_slots} projected image tokens but received {image_tokens.size(0)}."
                )
            # Clone before in-place replacement so callers that reuse embeddings stay safe.
            embeddings = embeddings.clone()
            embeddings[image_mask] = image_tokens.to(embeddings.dtype)
        return embeddings, attention_mask.to(self.device)

    def _build_pooled_multimodal_embeddings(
        self,
        batch: Dict[str, Any],
        pooling: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a decode-only prefix with all image placeholder tokens replaced by
        one pooled projected-image embedding.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(self.device)
        if input_ids.size(0) != 1:
            raise ValueError("Pooled decode baselines currently support batch size 1.")
        image_tokens = batch.get("projected_image_tokens")
        if image_tokens is None:
            raise ValueError("projected_image_tokens are required for pooled decode baselines.")
        if self.image_token_id is None:
            raise ValueError("The backbone config does not expose image_token_id.")

        image_tokens = image_tokens.squeeze(0) if image_tokens.dim() == 3 else image_tokens
        pooled_token = self._pool_tokens(image_tokens, mode=pooling).view(1, 1, -1)
        return self._build_visual_token_multimodal_embeddings(batch, pooled_token)

    def _build_visual_token_multimodal_embeddings(
        self,
        batch: Dict[str, Any],
        visual_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace the image placeholder span with a custom visual-token sequence."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(self.device)
        if input_ids.size(0) != 1:
            raise ValueError("Custom visual-token decode baselines currently support batch size 1.")
        if self.image_token_id is None:
            raise ValueError("The backbone config does not expose image_token_id.")

        visual_tokens = visual_tokens.squeeze(0) if visual_tokens.dim() == 3 else visual_tokens
        image_positions = torch.nonzero(input_ids[0] == self.image_token_id, as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise ValueError("No image placeholder tokens were found.")

        start = int(image_positions[0].item())
        end = int(image_positions[-1].item()) + 1
        embeddings = self._embed_input_ids(input_ids)
        inputs_embeds = torch.cat(
            [
                embeddings[:, :start, :],
                visual_tokens.unsqueeze(0).to(embeddings.dtype),
                embeddings[:, end:, :],
            ],
            dim=1,
        )
        visual_mask = torch.ones((1, visual_tokens.size(0)), device=self.device, dtype=attention_mask.dtype)
        pooled_attention_mask = torch.cat(
            [
                attention_mask[:, :start],
                visual_mask,
                attention_mask[:, end:],
            ],
            dim=1,
        )
        return inputs_embeds, pooled_attention_mask

    def _decode_ids(self, generated_ids: torch.Tensor) -> str:
        """Decode token ids using processor or tokenizer fallback chain."""
        token_ids = generated_ids.tolist()
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode([token_ids], skip_special_tokens=True)[0]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        return " ".join(str(token_id) for token_id in token_ids)

    def _encode_text_ids(self, text: str) -> torch.Tensor:
        """Tokenize plain text without special tokens and return a 1D id tensor."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("The processor tokenizer is required for entropy tracking.")
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None:
            raise ValueError("Tokenizer output did not include input_ids.")
        return input_ids.squeeze(0).to(device=self.device, dtype=torch.long)

    def _first_token_id(self, text: str) -> int:
        token_ids = self._encode_text_ids(text)
        if token_ids.numel() == 0:
            raise ValueError(f"Could not tokenize option text: {text!r}")
        return int(token_ids[0].item())

    def _entropy_from_logits(self, logits: torch.Tensor) -> float:
        """Compute natural-log entropy over a categorical distribution."""
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
        return float(entropy.squeeze().detach().cpu().item())

    def _aggregate_entropies(self, entropies: List[float]) -> Dict[str, Optional[float]]:
        if not entropies:
            return {"mean": None, "median": None, "max": None}
        return {
            "mean": float(mean(entropies)),
            "median": float(median(entropies)),
            "max": float(max(entropies)),
        }

    def _answer_option_token_ids(self) -> Dict[str, List[int]]:
        option_token_ids: Dict[str, List[int]] = {}
        for option in ("A", "B", "C", "D"):
            variants = (option, f" {option}", f"{option}.", f" {option}.")
            ids = {self._first_token_id(variant) for variant in variants}
            option_token_ids[option] = sorted(ids)
        return option_token_ids

    def _matched_answer_option(self, token_id: int, option_token_ids: Dict[str, List[int]]) -> Optional[str]:
        for option, token_ids in option_token_ids.items():
            if token_id in token_ids:
                return option
        return None

    def _answer_option_entropy_from_logits(
        self,
        logits: torch.Tensor,
        option_token_ids: Dict[str, List[int]],
        selected_token_id: int,
        decoded_token_index: int,
    ) -> Dict[str, Any]:
        """
        Measure entropy over A/B/C/D from an existing decode-step distribution.

        Per request, option probabilities are extracted from the current token
        distribution, softmaxed across A-D, and then used for entropy.
        """
        selected_option = self._matched_answer_option(selected_token_id, option_token_ids)
        vocab_probs = torch.softmax(logits.float(), dim=-1).squeeze(0)
        raw_option_probs = torch.tensor(
            [
                float(vocab_probs[token_ids].sum().detach().cpu().item())
                for token_ids in option_token_ids.values()
            ],
            device=self.device,
            dtype=torch.float32,
        )
        normalized_option_probs = torch.softmax(raw_option_probs, dim=-1)
        entropy = -(
            normalized_option_probs * torch.log(normalized_option_probs.clamp_min(1e-12))
        ).sum()
        return {
            "decoded_token_index": decoded_token_index,
            "selected_token_id": selected_token_id,
            "selected_option": selected_option,
            "option_token_ids": option_token_ids,
            "raw_option_probabilities": {
                option: float(probability)
                for option, probability in zip(option_token_ids.keys(), raw_option_probs.detach().cpu().tolist())
            },
            "softmax_option_probabilities": {
                option: float(probability)
                for option, probability in zip(option_token_ids.keys(), normalized_option_probs.detach().cpu().tolist())
            },
            "entropy": float(entropy.detach().cpu().item()),
        }

    def _select_action(self, type_logits: torch.Tensor, sample_actions: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select hard action ids and return the chosen-action log-prob."""
        distribution = Categorical(logits=type_logits)
        if sample_actions:
            action_tensor = distribution.sample()
        else:
            action_tensor = torch.argmax(type_logits, dim=-1)
        return action_tensor, distribution.log_prob(action_tensor)

    def _select_index(self, index_logits: torch.Tensor, sample_actions: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a region/patch index from logits and return its log-prob."""
        distribution = Categorical(logits=index_logits)
        if sample_actions:
            index_tensor = distribution.sample()
        else:
            index_tensor = torch.argmax(index_logits, dim=-1)
        return index_tensor, distribution.log_prob(index_tensor)

    def _distribution_to_list(self, logits: torch.Tensor) -> list:
        """Convert controller logits into a plain Python probability list."""
        probabilities = torch.softmax(logits, dim=-1)
        squeezed = probabilities.squeeze(0)
        return [float(value) for value in squeezed.detach().cpu().tolist()]

    def _controller_head_entropies(
        self,
        type_logits: torch.Tensor,
        region_logits: torch.Tensor,
        patch_logits: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure head and hierarchical joint entropy from selection logits."""
        action_probs = torch.softmax(type_logits.float(), dim=-1).squeeze(0)
        action_entropy = self._entropy_from_logits(type_logits)
        region_entropy = self._entropy_from_logits(region_logits)
        patch_entropy = self._entropy_from_logits(patch_logits)
        region_probability = (
            float(action_probs[self.action_names.index("REGION")].detach().cpu().item())
            if "REGION" in self.action_names
            else 0.0
        )
        patch_probability = (
            float(action_probs[self.action_names.index("PATCH")].detach().cpu().item())
            if "PATCH" in self.action_names
            else 0.0
        )
        controller_entropy = (
            action_entropy
            + region_probability * region_entropy
            + patch_probability * patch_entropy
        )
        return {
            "controller_action_entropy": action_entropy,
            "controller_region_entropy": region_entropy,
            "controller_patch_entropy": patch_entropy,
            "controller_entropy": float(controller_entropy),
        }

    def _controller_entropy_tracking(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect controller-step entropies and aggregate each series."""
        field_to_prefix = {
            "controller_action_entropy": "controller_action_entropy",
            "controller_region_entropy": "controller_region_entropy",
            "controller_patch_entropy": "controller_patch_entropy",
            "controller_entropy": "controller_entropy",
        }
        tracking: Dict[str, Any] = {}
        for field, prefix in field_to_prefix.items():
            values = [float(step[field]) for step in trace if step.get(field) is not None]
            summary = self._aggregate_entropies(values)
            tracking[f"{prefix}_values"] = values
            tracking[f"{prefix}_mean"] = summary["mean"]
            tracking[f"{prefix}_median"] = summary["median"]
            tracking[f"{prefix}_max"] = summary["max"]
        return tracking

    def _scale_controller_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling so larger values produce flatter controller distributions."""
        return logits / self.controller_temperature

    def _inference_uses_sampling(self) -> bool:
        """Resolve config action_selection into inference sampling mode."""
        if isinstance(self.action_selection, bool):
            return self.action_selection
        mode = str(self.action_selection).strip().lower()
        if mode in {"sample", "sampling", "stochastic"}:
            return True
        if mode in {"argmax", "greedy", "deterministic"}:
            return False
        return False

    def _write_recurrent_tokens(
        self,
        inputs_embeds: torch.Tensor,
        latent_pos: int,
        act_pos: int,
        latent_hidden: torch.Tensor,
        act_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the pure recurrent THINK update.

        The next-step input embedding for each control token is set directly to
        that token's current output hidden state. There is no extra projection:
        THINK is just one more recurrent pass of the same hidden state.
        """
        updated_embeds = inputs_embeds.clone()
        updated_embeds[:, latent_pos, :] = latent_hidden.to(updated_embeds.dtype)
        updated_embeds[:, act_pos, :] = act_hidden.to(updated_embeds.dtype)
        return updated_embeds

    def _append_hidden_token(
        self,
        state: Dict[str, Any],
        hidden: torch.Tensor,
        track_as_think: bool = True,
    ) -> None:
        """Append a recurrent hidden-state token and extend the attention mask."""
        append_pos = int(state["inputs_embeds"].size(1))
        new_embed = hidden.unsqueeze(1).to(state["inputs_embeds"].dtype)
        state["inputs_embeds"] = torch.cat([state["inputs_embeds"], new_embed], dim=1)
        new_mask = torch.ones(
            (state["attention_mask"].size(0), 1),
            device=self.device,
            dtype=state["attention_mask"].dtype,
        )
        state["attention_mask"] = torch.cat([state["attention_mask"], new_mask], dim=1)
        if track_as_think:
            state.setdefault("trace_all_positions", []).append(append_pos)

    def _shift_trace_positions_for_insert(
        self,
        state: Dict[str, Any],
        insert_pos: int,
        num_tokens: int,
    ) -> None:
        """Shift tracked absolute positions when tokens are inserted mid-sequence."""
        for key in ("trace_all_positions", "trace_visual_positions"):
            positions = state.setdefault(key, [])
            state[key] = [
                int(position) + num_tokens if int(position) >= insert_pos else int(position)
                for position in positions
            ]

    def _shift_trace_positions_for_drop(self, state: Dict[str, Any], drop_pos: int) -> None:
        """Remove a dropped position and shift all later trace positions left."""
        for key in ("trace_all_positions", "trace_visual_positions"):
            shifted = []
            for position in state.setdefault(key, []):
                position = int(position)
                if position == drop_pos:
                    continue
                shifted.append(position - 1 if position > drop_pos else position)
            state[key] = shifted

    def _insert_evidence_token(
        self,
        state: Dict[str, Any],
        evidence_tokens: torch.Tensor,
        track_as_visual: bool = True,
    ) -> None:
        """Insert projected evidence tokens before the latent token or current final token."""
        projected = evidence_tokens.unsqueeze(0).to(state["inputs_embeds"].dtype)
        num_tokens = projected.size(1)
        insert_pos = state["latent_pos"] if self.use_control_tokens else state["inputs_embeds"].size(1) - 1
        insert_pos = int(insert_pos)
        self._shift_trace_positions_for_insert(state, insert_pos, num_tokens)
        prefix = state["inputs_embeds"][:, :insert_pos, :]
        suffix = state["inputs_embeds"][:, insert_pos:, :]
        state["inputs_embeds"] = torch.cat([prefix, projected, suffix], dim=1)

        prefix_mask = state["attention_mask"][:, :insert_pos]
        suffix_mask = state["attention_mask"][:, insert_pos:]
        new_mask = torch.ones(
            (state["attention_mask"].size(0), num_tokens),
            device=self.device,
            dtype=state["attention_mask"].dtype,
        )
        state["attention_mask"] = torch.cat([prefix_mask, new_mask, suffix_mask], dim=1)

        if self.use_control_tokens:
            state["latent_pos"] += num_tokens
            state["act_pos"] += num_tokens
        if track_as_visual:
            inserted_positions = list(range(insert_pos, insert_pos + num_tokens))
            state.setdefault("trace_all_positions", []).extend(inserted_positions)
            state.setdefault("trace_visual_positions", []).extend(inserted_positions)
            state["trace_all_positions"] = sorted(set(state["trace_all_positions"]))
            state["trace_visual_positions"] = sorted(set(state["trace_visual_positions"]))

    def prepare_inputs(
        self,
        images: Any,
        questions: Any,
        add_answer_instruction: bool = True,
        image_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build one multimodal prompt for image + question and tokenize via processor.

        Normal inference adds a tagged-answer instruction; Phase 2 mining can
        disable that suffix to match the Phase 1 M3CoT collator prompt.
        """
        image = images[0] if isinstance(images, (list, tuple)) else images
        question = questions[0] if isinstance(questions, (list, tuple)) else questions
        if image_size is not None and image is not None and hasattr(image, "resize"):
            image = image.resize((int(image_size), int(image_size)))
        prompt = str(question)
        if add_answer_instruction:
            prompt = f"{prompt}\nReturn only the final answer inside <answer>...</answer>."
        content: list = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("The processor must implement apply_chat_template for this prototype.")
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        processor_kwargs: dict = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            processor_kwargs["images"] = [image]
        batch = self.processor(**processor_kwargs)
        batch = self._move_batch_to_device(dict(batch))
        batch["messages"] = messages
        batch["question"] = question
        return batch

    def get_projected_image_tokens(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Extract LM-space projected image tokens from Qwen2-VL visual encoder.

        This is the single method where model-internal visual calls are allowed,
        because HF public APIs do not directly expose this projected token bank.
        """
        if "projected_image_tokens" in batch:
            return batch["projected_image_tokens"]
        
        
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        if pixel_values is None:
            raise ValueError("pixel_values are required to extract projected image tokens.")
        
        if image_grid_thw is not None:
            pre_grid, post_grid = self._resolve_image_grids(image_grid_thw)
            self._current_premerge_grid = pre_grid
            self._current_postmerge_grid = post_grid
            self._current_image_grid = post_grid
        else:
            self._current_premerge_grid = None
            self._current_postmerge_grid = None
            self._current_image_grid = None
            
        visual = self._resolve_visual_encoder(required=True)
        # Support minor signature differences across backbone/test doubles.
        try:
            image_tokens = visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            try:
                image_tokens = visual(pixel_values, image_grid_thw)
            except TypeError:
                image_tokens = visual(pixel_values)
        if image_tokens.dim() == 3:
            image_tokens = image_tokens[0]
            
        if self._current_postmerge_grid is not None:
            expected = self._current_postmerge_grid[0] * self._current_postmerge_grid[1]
            if image_tokens.size(0) != expected:
                raise ValueError(
                    f"Expected {expected} post-merge tokens for grid {self._current_postmerge_grid}, "
                    f"got {image_tokens.size(0)}."
                )
        return image_tokens.to(self.device, dtype=self.latent_token.dtype)

    def build_visual_bank(self, image_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Construct:
            patches: raw projected image tokens
            regions: pooled non-overlapping windows over patch grid
            global: pooled summary over all patches
        """
        patch_tokens = image_tokens.squeeze(0) if image_tokens.dim() == 3 else image_tokens
        regions, raw_regions = self.build_region_tokens(patch_tokens, pooling=self.pooling)
        global_token = self._pool_tokens(patch_tokens, self.global_pool).unsqueeze(0)

        return {
            "global": global_token,
            "regions": regions,
            "raw_regions": raw_regions,
            "patches": patch_tokens,
        }

    def build_region_tokens(self, image_tokens: torch.Tensor, pooling: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool local image-token windows into region tokens and also return raw windows."""
        patch_tokens = image_tokens.squeeze(0) if image_tokens.dim() == 3 else image_tokens
        grid = self._current_postmerge_grid or self._current_image_grid
        if grid is None:
            raise ValueError("Current image grid is unknown; call get_projected_image_tokens first.")
        grid_h, grid_w = grid
        if grid_h * grid_w != patch_tokens.size(0):
            raise ValueError("Projected image tokens do not match the expected patch grid.")
        region_window = self.region_window
        if isinstance(region_window, int):
            region_window = (region_window, region_window)
        region_h, region_w = region_window

        # Convert flat patch list into grid for non-overlapping region windows.
        patch_grid = patch_tokens.view(grid_h, grid_w, self.hidden_size)
        patch_grid, region_grid_h, region_grid_w = self._pad_patch_grid_for_regions(
            patch_grid,
            region_h,
            region_w,
        )

        pooled_tokens = []
        raw_windows = []
        for row in range(0, region_grid_h, region_h):
            for col in range(0, region_grid_w, region_w):
                window = patch_grid[row : row + region_h, col : col + region_w, :].reshape(-1, self.hidden_size)
                pooled_tokens.append(self._pool_tokens(window, self.region_pool, mode=pooling))
                raw_windows.append(window)
        return torch.stack(pooled_tokens, dim=0), torch.stack(raw_windows, dim=0)

    def build_initial_state(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create recurrent state dict with optional latent/act control tokens.

        The state object is intentionally simple and mutable so each reasoning
        step can update it in-place.
        """
        inputs_embeds, attention_mask = self._build_multimodal_embeddings(batch)
        latent_pos = None
        act_pos = None
        if self.use_control_tokens:
            latent = self.latent_token.view(1, 1, -1).to(inputs_embeds.dtype)
            act = self.act_token.view(1, 1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, latent, act], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), 2), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            latent_pos = inputs_embeds.size(1) - 2
            act_pos = inputs_embeds.size(1) - 1
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "latent_pos": latent_pos,
            "act_pos": act_pos,
            "trace_all_positions": [],
            "trace_visual_positions": [],
            "trace": [],
            "action_log_probs": [],
            "question": batch.get("question"),
            "sample_actions": False,
        }

    def build_coarse_initial_state(
        self,
        batch: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Create a mining state with the image span replaced by one global token.

        This is used by Phase 2 oracle mining. Normal inference and training still
        use build_initial_state, which keeps the full image-token prompt.
        """
        inputs_embeds, attention_mask = self._build_visual_token_multimodal_embeddings(batch, bank["global"])
        latent_pos = None
        act_pos = None
        if self.use_control_tokens:
            latent = self.latent_token.view(1, 1, -1).to(inputs_embeds.dtype)
            act = self.act_token.view(1, 1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, latent, act], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), 2), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            latent_pos = inputs_embeds.size(1) - 2
            act_pos = inputs_embeds.size(1) - 1
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "latent_pos": latent_pos,
            "act_pos": act_pos,
            "trace_all_positions": [],
            "trace_visual_positions": [],
            "trace": [],
            "action_log_probs": [],
            "question": batch.get("question"),
            "sample_actions": False,
        }

    def clone_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Clone a recurrent state so oracle candidate scoring cannot mutate it."""
        cloned: Dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    def _extract_final_hidden(self, outputs: Any) -> torch.Tensor:
        """Read the final sequence hidden state from HF causal/output variants."""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        raise AttributeError(
            "Backbone output did not include hidden_states or last_hidden_state; "
            "call the backbone with output_hidden_states=True when hidden tokens are needed."
        )

    def _read_current_hidden(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the frozen backbone and return hidden states needed for explicit THINK."""
        with torch.no_grad():
            outputs = self.backbone(
                inputs_embeds=state["inputs_embeds"],
                attention_mask=state["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        final_hidden = self._extract_final_hidden(outputs)
        last_hidden = final_hidden[:, -1, :]
        if self.use_control_tokens:
            state_hidden = final_hidden[:, state["latent_pos"], :]
            act_hidden = final_hidden[:, state["act_pos"], :]
        else:
            state_hidden = last_hidden
            act_hidden = None
        return last_hidden, state_hidden, act_hidden

    def _controller_step_hidden(self, step_idx: int) -> torch.Tensor:
        """Embed the primitive controller step used by SFT/rollout policy calls."""
        if step_idx < 0 or step_idx >= self.step_embedding.num_embeddings:
            raise ValueError(
                f"controller step {step_idx} is outside step_embedding capacity "
                f"{self.step_embedding.num_embeddings}; increase model.controller_max_steps."
            )
        return self.step_embedding(torch.tensor([step_idx], device=self.device, dtype=torch.long))

    def _build_controller_state_hidden(
        self,
        final_hidden: torch.Tensor,
        state: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build normalized controller context from the final hidden-state sequence."""
        if self.use_control_tokens:
            state_hidden = self.controller_state_norm(final_hidden[:, state["latent_pos"], :])
            act_hidden = self.controller_state_norm(final_hidden[:, state["act_pos"], :])
            return state_hidden, act_hidden

        n_states = self.controller_num_states
        if final_hidden.size(1) < n_states:
            raise ValueError(
                f"controller_context_window={n_states} requires at least {n_states} hidden states, "
                f"got sequence length {final_hidden.size(1)}."
            )
        last_n = final_hidden[:, -n_states:, :]
        normalized = self.controller_state_norm(last_n)
        state_hidden = normalized.reshape(normalized.size(0), -1)
        return state_hidden, None

    def controller_logits_from_state(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw controller logits for the current recurrent state."""
        with torch.no_grad():
            outputs = self.backbone(
                inputs_embeds=state["inputs_embeds"],
                attention_mask=state["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        final_hidden = self._extract_final_hidden(outputs)
        state_hidden, act_hidden = self._build_controller_state_hidden(final_hidden, state)
        step_hidden = self._controller_step_hidden(step_idx)
        logits = self.controller(state_hidden, step_hidden, bank, act_hidden=act_hidden)
        self._validate_fixed_index_logits(logits[1], logits[2], bank)
        return logits

    def _validate_fixed_index_logits(
        self,
        region_logits: torch.Tensor,
        patch_logits: torch.Tensor,
        bank: Dict[str, torch.Tensor],
    ) -> None:
        """Ensure fixed classifier heads match the current visual bank size."""
        num_regions = bank["regions"].size(0)
        num_patches = bank["patches"].size(0)
        if region_logits.size(-1) != num_regions:
            raise ValueError(
                f"Controller region head has {region_logits.size(-1)} classes, but visual bank has {num_regions} regions."
            )
        if patch_logits.size(-1) != num_patches:
            raise ValueError(
                f"Controller patch head has {patch_logits.size(-1)} classes, but visual bank has {num_patches} patches."
            )

    def apply_mined_actions(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        actions: list,
    ) -> Dict[str, Any]:
        """Apply an explicit Phase 2 action sequence to a recurrent state."""
        for action in actions:
            action_type = str(action.get("type", "")).upper()
            if action_type in {"NO_OP", "STOP"}:
                continue
            if action_type == "THINK":
                last_hidden, state_hidden, act_hidden = self._read_current_hidden(state)
                if self.think_append_hidden:
                    self._append_hidden_token(state, last_hidden)
                elif self.use_control_tokens:
                    state["inputs_embeds"] = self._write_recurrent_tokens(
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
            elif action_type == "GLOBAL":
                self._insert_evidence_token(state, bank["global"])
            elif action_type == "REGION":
                self._insert_evidence_token(state, bank["raw_regions"][int(action["region_idx"])])
            elif action_type == "PATCH":
                patch = bank["patches"][int(action["patch_idx"])].unsqueeze(0)
                self._insert_evidence_token(state, patch)
            else:
                raise ValueError(f"Unsupported mined action type: {action_type}")
        return state

    def forward_reasoning_step(
        self,
        state: Dict[str, Any],
        bank: Dict[str, torch.Tensor],
        step_idx: int,
    ) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        Run one LVAR controller step:
        1) backbone pass, 2) controller action, 3) optional evidence insertion.
        """
        # Run current sequence through the LM and read the latest hidden states.
        # Backbone is frozen — run under no_grad so its computation graph is
        # discarded immediately, saving memory across recurrent steps and rollouts.
        with torch.no_grad():
            outputs = self.backbone(
                inputs_embeds=state["inputs_embeds"],
                attention_mask=state["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        final_hidden = self._extract_final_hidden(outputs)
        last_hidden = final_hidden[:, -1, :]
        if self.use_control_tokens:
            recurrent_state_hidden = final_hidden[:, state["latent_pos"], :]
            recurrent_act_hidden = final_hidden[:, state["act_pos"], :]
        else:
            recurrent_state_hidden = last_hidden
            recurrent_act_hidden = None
        state_hidden, act_hidden = self._build_controller_state_hidden(final_hidden, state)
        # Step embedding gives the controller explicit notion of iteration depth.
        step_hidden = self._controller_step_hidden(step_idx)
        type_logits, region_logits, patch_logits = self.controller(
            state_hidden,
            step_hidden,
            bank,
            act_hidden=act_hidden,
        )
        self._validate_fixed_index_logits(region_logits, patch_logits, bank)
        scaled_type_logits = self._scale_controller_logits(type_logits)
        scaled_region_logits = self._scale_controller_logits(region_logits)
        scaled_patch_logits = self._scale_controller_logits(patch_logits)
        if self.mask_immediate_repeats and state.get("last_action") == "REGION":
            last_region = state.get("last_region_index")
            if isinstance(last_region, int) and scaled_region_logits.size(-1) > 1:
                scaled_region_logits = scaled_region_logits.clone()
                if 0 <= last_region < scaled_region_logits.size(-1):
                    scaled_region_logits[:, last_region] = torch.finfo(scaled_region_logits.dtype).min
        if self.mask_immediate_repeats and state.get("last_action") == "PATCH":
            last_patch = state.get("last_patch_index")
            if isinstance(last_patch, int) and scaled_patch_logits.size(-1) > 1:
                scaled_patch_logits = scaled_patch_logits.clone()
                if 0 <= last_patch < scaled_patch_logits.size(-1):
                    scaled_patch_logits[:, last_patch] = torch.finfo(scaled_patch_logits.dtype).min
        action_probs = self._distribution_to_list(scaled_type_logits)
        region_probs = self._distribution_to_list(scaled_region_logits)
        patch_probs = self._distribution_to_list(scaled_patch_logits)
        controller_entropies = self._controller_head_entropies(
            scaled_type_logits,
            scaled_region_logits,
            scaled_patch_logits,
        )
        action_tensor, action_log_prob = self._select_action(
            scaled_type_logits,
            state.get("sample_actions", False),
        )
        action_id = int(action_tensor.item())
        action_name = self.action_names[action_id]
        should_stop = action_name == "STOP"

        # Map action to evidence token selection. THINK and STOP add no evidence.
        region_index = None
        patch_index = None
        evidence_token = None
        if action_name == "GLOBAL":
            evidence_token = bank["global"][0].unsqueeze(0)
        elif action_name == "REGION":
            region_tensor, region_log_prob = self._select_index(
                scaled_region_logits,
                state.get("sample_actions", False),
            )
            region_index = int(region_tensor.item())
            action_log_prob = action_log_prob + region_log_prob
            evidence_token = bank["raw_regions"][region_index]
        elif action_name == "PATCH":
            patch_tensor, patch_log_prob = self._select_index(
                scaled_patch_logits,
                state.get("sample_actions", False),
            )
            patch_index = int(patch_tensor.item())
            action_log_prob = action_log_prob + patch_log_prob
            evidence_token = bank["patches"][patch_index].unsqueeze(0)

        sequence_length_before = state["inputs_embeds"].size(1)
        # THINK is the only action that performs a pure recurrent hidden-state update.
        if action_name == "THINK":
            if self.think_append_hidden:
                self._append_hidden_token(state, last_hidden)
            elif self.use_control_tokens:
                state["inputs_embeds"] = self._write_recurrent_tokens(
                    state["inputs_embeds"],
                    state["latent_pos"],
                    state["act_pos"],
                    recurrent_state_hidden,
                    recurrent_act_hidden,
                )
            else:
                updated_embeds = state["inputs_embeds"].clone()
                updated_embeds[:, -1, :] = last_hidden.to(updated_embeds.dtype)
                state["inputs_embeds"] = updated_embeds

        # Then insert chosen evidence before the latent token or current final token.
        if evidence_token is not None:
            self._insert_evidence_token(state, evidence_token)

        # Persist step-level metadata for debug and policy-gradient training.
        step_trace = {
            "step_idx": step_idx,
            "action_id": action_id,
            "action": action_name,
            "action_names": self.action_names,
            "should_stop": should_stop,
            "action_probs": action_probs,
            "region_probs": region_probs,
            "patch_probs": patch_probs,
            **controller_entropies,
            "controller_temperature": self.controller_temperature,
            "region_index": region_index,
            "patch_index": patch_index,
            "sequence_length_before": sequence_length_before,
            "sequence_length_after": state["inputs_embeds"].size(1),
            "action_log_prob": float(action_log_prob.detach().item()),
        }
        state["trace"].append(step_trace)
        state["action_log_probs"].append(action_log_prob)
        state["last_action"] = action_name
        state["last_region_index"] = region_index
        state["last_patch_index"] = patch_index
        return state, action_id, should_stop, step_trace

    def drop_act_token(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove act token only for final decoding.

        Important: this uses tensor slicing + concat on existing tensors, so
        autograd stays connected to all prior recurrent computations.
        """
        act_pos = state["act_pos"]
        if act_pos is None:
            return state
        # Keep the graph connected by slicing the existing tensor and concatenating
        # the surviving pieces. We intentionally do not detach or rebuild from scratch.
        before = state["inputs_embeds"][:, :act_pos, :]
        after = state["inputs_embeds"][:, act_pos + 1 :, :]
        state["inputs_embeds"] = torch.cat([before, after], dim=1)
        state["attention_mask"] = torch.cat(
            [state["attention_mask"][:, :act_pos], state["attention_mask"][:, act_pos + 1 :]],
            dim=1,
        )
        self._shift_trace_positions_for_drop(state, int(act_pos))
        state["act_pos"] = None
        return state

    def decode_answer(self, state: Dict[str, Any], labels: Optional[Any] = None) -> Dict[str, Any]:
        """
        Greedy autoregressive decoding from a custom embedding prefix.

        We avoid model.generate() here because the prefix has custom token-level
        surgery (latent/evidence tokens and act-token removal) that we need to
        control explicitly.
        """
        del labels
        decode_prefix_length = state["inputs_embeds"].size(1)
        current_embeds = state["inputs_embeds"]
        current_mask = state["attention_mask"]
        generated_ids = []
        token_entropies = []
        option_token_ids = self._answer_option_token_ids()
        option_entropy = None

        # Decode token-by-token with argmax to keep inference deterministic.
        with self.trace_boost_runtime.answer_decode(
            state.get("trace_all_positions", []),
            state.get("trace_visual_positions", []),
            answer_query_start=decode_prefix_length - 1,
        ):
            for decoded_token_index in range(self.max_answer_tokens):
                outputs = self.backbone(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                token_id = int(next_token.item())
                if self.eos_token_id is not None and token_id == self.eos_token_id:
                    break
                token_entropies.append(self._entropy_from_logits(next_token_logits))
                if option_entropy is None and self._matched_answer_option(token_id, option_token_ids) is not None:
                    option_entropy = self._answer_option_entropy_from_logits(
                        next_token_logits,
                        option_token_ids,
                        selected_token_id=token_id,
                        decoded_token_index=decoded_token_index,
                    )
                generated_ids.append(token_id)
                # Feed generated token back into the running prefix.
                next_embed = self._embed_input_ids(next_token.unsqueeze(1))
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                current_mask = torch.cat(
                    [
                        current_mask,
                        torch.ones((1, 1), device=self.device, dtype=current_mask.dtype),
                    ],
                    dim=1,
                )

        generated_tensor = torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        generated_text = self._decode_ids(generated_tensor.cpu()) if generated_ids else ""
        token_entropy_summary = self._aggregate_entropies(token_entropies)
        attention_mass_summary = self.trace_boost_runtime.attention_mass_summary()
        return {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "answer": extract_tagged_answer(generated_text),
            "token_entropies": token_entropies,
            "token_entropy_mean": token_entropy_summary["mean"],
            "token_entropy_median": token_entropy_summary["median"],
            "token_entropy_max": token_entropy_summary["max"],
            "answer_option_entropy": option_entropy,
            **attention_mass_summary,
            "decode_prefix_length": decode_prefix_length,
            "final_sequence_length": current_embeds.size(1),
            "final_inputs_embeds": current_embeds,
            "final_attention_mask": current_mask,
        }

    def _build_decode_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a detached decode-only view of recurrent state.

        Training rewards only need decoded text, so final generation should not
        retain a full autoregressive graph through the frozen backbone.
        """
        return {
            "inputs_embeds": state["inputs_embeds"].detach(),
            "attention_mask": state["attention_mask"].detach(),
            "latent_pos": state.get("latent_pos"),
            "act_pos": state.get("act_pos"),
            "trace_all_positions": list(state.get("trace_all_positions", [])),
            "trace_visual_positions": list(state.get("trace_visual_positions", [])),
        }

    def forward(
        self,
        images: Any,
        questions: Any,
        labels: Optional[Any] = None,
        sample_actions: Optional[bool] = None,
        add_answer_instruction: bool = True,
        use_coarse_context: bool = False,
        image_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main LVAR path: prepare -> visual bank -> recurrent loop -> act drop -> decode.
        """
        # Build model-ready inputs and visual candidate bank.
        prepared = self.prepare_inputs(
            images,
            questions,
            add_answer_instruction=add_answer_instruction,
            image_size=image_size,
        )
        image_tokens = self.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        bank = self.build_visual_bank(image_tokens)
        if use_coarse_context:
            state = self.build_coarse_initial_state(prepared, bank)
        else:
            state = self.build_initial_state(prepared)
        if sample_actions is None:
            # During training we always sample; during inference defer to config.
            state["sample_actions"] = self.training or self._inference_uses_sampling()
        else:
            state["sample_actions"] = sample_actions

        # Controller loop runs until STOP or max_steps.
        stopped = False
        for step_idx in range(self.max_steps):
            state, _, stopped, _ = self.forward_reasoning_step(state, bank, step_idx)
            if stopped:
                break

        # Final decode excludes act token only in the legacy control-token path.
        if self.use_control_tokens:
            state = self.drop_act_token(state)
        with torch.no_grad():
            decoded = self.decode_answer(self._build_decode_state(state), labels=labels)
        action_log_prob_sum = None
        if state["action_log_probs"]:
            action_log_prob_sum = torch.stack(state["action_log_probs"]).sum()
        controller_entropy_tracking = self._controller_entropy_tracking(state["trace"])
        return {
            "answer": decoded["answer"],
            "generated_text": decoded["generated_text"],
            "generated_ids": decoded["generated_ids"],
            "token_entropies": decoded.get("token_entropies", []),
            "token_entropy_mean": decoded.get("token_entropy_mean"),
            "token_entropy_median": decoded.get("token_entropy_median"),
            "token_entropy_max": decoded.get("token_entropy_max"),
            "answer_option_entropy": decoded.get("answer_option_entropy"),
            "trace_attention_mass": decoded.get("trace_attention_mass"),
            "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
            "think_attention_mass": decoded.get("think_attention_mass"),
            "trace_boost_attention_observations": decoded.get("trace_boost_attention_observations", 0),
            "trace_boost_softmax_hits": decoded.get("trace_boost_softmax_hits", 0),
            "trace": state["trace"],
            "num_steps": len(state["trace"]),
            "stopped": stopped,
            "decode_prefix_length": decoded["decode_prefix_length"],
            "final_sequence_length": decoded["final_sequence_length"],
            "action_log_probs": state["action_log_probs"],
            "action_log_prob_sum": action_log_prob_sum,
            **controller_entropy_tracking,
        }

    def baseline_forward(
        self,
        images: Any,
        questions: Any,
        labels: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Reference no-latent path used for delta reward comparisons."""
        prepared = self.prepare_inputs(images, questions)
        image_tokens = self.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        inputs_embeds, attention_mask = self._build_multimodal_embeddings(prepared)
        state = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "latent_pos": None,
            "act_pos": None,
            "trace_all_positions": [],
            "trace_visual_positions": [],
        }
        decoded = self.decode_answer(state, labels=labels)
        return {
            "answer": decoded["answer"],
            "generated_text": decoded["generated_text"],
            "generated_ids": decoded["generated_ids"],
            "token_entropies": decoded.get("token_entropies", []),
            "token_entropy_mean": decoded.get("token_entropy_mean"),
            "token_entropy_median": decoded.get("token_entropy_median"),
            "token_entropy_max": decoded.get("token_entropy_max"),
            "answer_option_entropy": decoded.get("answer_option_entropy"),
            "trace_attention_mass": decoded.get("trace_attention_mass"),
            "visual_trace_attention_mass": decoded.get("visual_trace_attention_mass"),
            "think_attention_mass": decoded.get("think_attention_mass"),
            "trace": [],
            "num_steps": 0,
            "decode_prefix_length": decoded["decode_prefix_length"],
            "final_sequence_length": decoded["final_sequence_length"],
        }

    def pooled_baseline_forward(
        self,
        images: Any,
        questions: Any,
        pooling: str,
        labels: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Decode from a prompt where the full image-token span is one pooled visual token."""
        prepared = self.prepare_inputs(images, questions)
        image_tokens = self.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        inputs_embeds, attention_mask = self._build_pooled_multimodal_embeddings(prepared, pooling)
        state = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "latent_pos": None,
            "act_pos": None,
        }
        decoded = self.decode_answer(state, labels=labels)
        return {
            "answer": decoded["answer"],
            "generated_text": decoded["generated_text"],
            "generated_ids": decoded["generated_ids"],
            "trace": [],
            "num_steps": 0,
            "decode_prefix_length": decoded["decode_prefix_length"],
            "final_sequence_length": decoded["final_sequence_length"],
        }

    def region_baseline_forward(
        self,
        images: Any,
        questions: Any,
        pooling: str,
        labels: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Decode from a prompt where image tokens are replaced by pooled region tokens."""
        prepared = self.prepare_inputs(images, questions)
        image_tokens = self.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = image_tokens
        region_tokens, _ = self.build_region_tokens(image_tokens, pooling=pooling)
        inputs_embeds, attention_mask = self._build_visual_token_multimodal_embeddings(prepared, region_tokens)
        state = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "latent_pos": None,
            "act_pos": None,
        }
        decoded = self.decode_answer(state, labels=labels)
        return {
            "answer": decoded["answer"],
            "generated_text": decoded["generated_text"],
            "generated_ids": decoded["generated_ids"],
            "trace": [],
            "num_steps": 0,
            "decode_prefix_length": decoded["decode_prefix_length"],
            "final_sequence_length": decoded["final_sequence_length"],
            "num_region_tokens": region_tokens.size(0),
        }

    def generate_lvar(self, images: Any, questions: Any, image_size: Optional[int] = None) -> Dict[str, Any]:
        """Inference wrapper for LVAR with deterministic (argmax) controller behavior."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(
                images,
                questions,
                sample_actions=self._inference_uses_sampling(),
                image_size=image_size,
            )
        self.train(was_training)
        return {
            "prediction": output["answer"],
            "trace": output["trace"],
            "num_steps": output["num_steps"],
            "generated_text": output["generated_text"],
            "generated_ids": output["generated_ids"],
            "token_entropies": output.get("token_entropies", []),
            "token_entropy_mean": output.get("token_entropy_mean"),
            "token_entropy_median": output.get("token_entropy_median"),
            "token_entropy_max": output.get("token_entropy_max"),
            "answer_option_entropy": output.get("answer_option_entropy"),
            **{
                key: value
                for key, value in output.items()
                if key.startswith("controller_") and "temperature" not in key
            },
        }

    def generate_baseline(self, images: Any, questions: Any) -> Dict[str, Any]:
        """Inference wrapper for baseline path."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.baseline_forward(images, questions)
        self.train(was_training)
        return {
            "prediction": output["answer"],
            "generated_text": output["generated_text"],
            "generated_ids": output["generated_ids"],
        }

    def generate_pooled_baseline(self, images: Any, questions: Any, pooling: str) -> Dict[str, Any]:
        """Inference wrapper for a decode-only pooled-image baseline."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.pooled_baseline_forward(images, questions, pooling=pooling)
        self.train(was_training)
        return {
            "prediction": output["answer"],
            "generated_text": output["generated_text"],
            "generated_ids": output["generated_ids"],
            "decode_prefix_length": output["decode_prefix_length"],
        }

    def generate_region_baseline(self, images: Any, questions: Any, pooling: str) -> Dict[str, Any]:
        """Inference wrapper for a decode-only region-token baseline."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.region_baseline_forward(images, questions, pooling=pooling)
        self.train(was_training)
        return {
            "prediction": output["answer"],
            "generated_text": output["generated_text"],
            "generated_ids": output["generated_ids"],
            "decode_prefix_length": output["decode_prefix_length"],
            "num_region_tokens": output["num_region_tokens"],
        }
