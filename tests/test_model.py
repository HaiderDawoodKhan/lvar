import types
import unittest

import torch
import torch.nn.functional as F

from lvar.controller_sft import set_controller_sft_trainable
from lvar.qwen_lvar import ControllerHead, QwenLVAR
from lvar.utils import (
    ACTION_GLOBAL,
    ACTION_PATCH,
    ACTION_REGION,
    ACTION_STOP,
    ACTION_THINK,
)


class DummyTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        del return_tensors, add_special_tokens
        token_ids = []
        for piece in str(text).split():
            token_ids.append((sum(ord(char) for char in piece) % 3) + 1)
        if not token_ids:
            token_ids = [1]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        pieces = []
        for token_id in token_ids:
            if token_id == 1:
                pieces.append("<answer>yes</answer>")
            elif token_id == 0:
                pieces.append("</s>")
            else:
                pieces.append(f"tok{token_id}")
        return "".join(pieces)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        del messages, add_generation_prompt
        if tokenize:
            return {
                "input_ids": torch.tensor([[5, 999, 999, 999, 999, 6]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
                "pixel_values": torch.tensor([[0.0]], dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }
        return "<dummy text>"

    def __call__(self, text=None, images=None, return_tensors=None):
        return {
            "input_ids": torch.tensor([[5, 999, 999, 999, 999, 6]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.tensor([[0.0]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        del skip_special_tokens
        return [self.tokenizer.decode(sequence) for sequence in sequences]


class DummyVision(torch.nn.Module):
    def __init__(self, spatial_merge_size=1):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.register_buffer(
            "image_tokens",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, pixel_values, grid_thw=None):
        del pixel_values, grid_thw
        return self.image_tokens.clone()


class DummyMergedGridProcessor(DummyProcessor):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        batch = super().apply_chat_template(messages, add_generation_prompt, tokenize)
        if isinstance(batch, dict):
            batch["image_grid_thw"] = torch.tensor([[1, 4, 4]], dtype=torch.long)
        return batch

    def __call__(self, text=None, images=None, return_tensors=None):
        batch = super().__call__(text=text, images=images, return_tensors=return_tensors)
        batch["image_grid_thw"] = torch.tensor([[1, 4, 4]], dtype=torch.long)
        return batch


class DummyBackbone(torch.nn.Module):
    def __init__(self, spatial_merge_size=1):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4, image_token_id=999, eos_token_id=0)
        self.embedding = torch.nn.Embedding(1100, 4)
        self.visual = DummyVision(spatial_merge_size=spatial_merge_size)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[5] = torch.tensor([0.5, 0.5, 0.0, 0.0])
            self.embedding.weight[6] = torch.tensor([0.0, 0.5, 0.5, 0.0])
            self.embedding.weight[1] = torch.tensor([0.0, 0.0, 0.5, 0.5])

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
        **kwargs,
    ):
        del input_ids, attention_mask, output_hidden_states, return_dict, use_cache, kwargs
        if inputs_embeds is None:
            raise ValueError("DummyBackbone expects inputs_embeds in these tests.")
        seq_len = inputs_embeds.size(1)
        positions = torch.arange(seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, seq_len, 1)
        hidden = inputs_embeds + positions * 0.01
        logits = torch.zeros((1, seq_len, 4), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        logits[:, -1, 1] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=[inputs_embeds, hidden],
            last_hidden_state=hidden,
        )


class DummyNestedVisualBackbone(DummyBackbone):
    def __init__(self, spatial_merge_size=1):
        super().__init__(spatial_merge_size=spatial_merge_size)
        visual = self.visual
        del self.visual
        self.model = torch.nn.Module()
        self.model.visual = visual


def build_model(**overrides):
    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 3,
        "region_window": 1,
        "max_answer_tokens": 1,
        "action_selection": "argmax",
        "controller_context_window": 1,
    }
    cfg.update(overrides)
    return QwenLVAR(cfg, backbone=DummyBackbone(), processor=DummyProcessor())


def build_merged_grid_model():
    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 3,
        "region_window": 1,
        "max_answer_tokens": 1,
        "action_selection": "argmax",
        "controller_context_window": 1,
    }
    return QwenLVAR(
        cfg,
        backbone=DummyBackbone(spatial_merge_size=2),
        processor=DummyMergedGridProcessor(),
    )


def build_nested_visual_model():
    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 3,
        "region_window": 1,
        "max_answer_tokens": 1,
        "action_selection": "argmax",
        "controller_context_window": 1,
    }
    return QwenLVAR(
        cfg,
        backbone=DummyNestedVisualBackbone(spatial_merge_size=2),
        processor=DummyMergedGridProcessor(),
    )


class QwenLVARTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        prepared = self.model.prepare_inputs("image", "question")
        projected = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        self.prepared = prepared
        self.projected = projected
        self.bank = self.model.build_visual_bank(projected)

    def _set_controller_action(self, action_id, capture=None):
        def forward(state_hidden, step_hidden, bank, act_hidden=None):
            if capture is not None:
                capture["state_hidden"] = state_hidden.detach().clone()
                capture["step_hidden"] = step_hidden.detach().clone()
                capture["act_hidden"] = act_hidden
            type_logits = torch.full((1, 5), -10.0)
            type_logits[0, action_id] = 10.0
            region_logits = torch.arange(bank["regions"].size(0), dtype=torch.float32).unsqueeze(0)
            patch_logits = torch.arange(bank["patches"].size(0), dtype=torch.float32).unsqueeze(0)
            return type_logits, region_logits, patch_logits

        self.model.controller.forward = forward

    def test_pool_tokens_supports_attention_mean_and_max(self):
        tokens = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 1.0, 8.0]])
        with torch.no_grad():
            self.model.global_pool.weight.zero_()
            self.model.global_pool.bias.zero_()

        self.assertTrue(torch.allclose(self.model._pool_tokens(tokens, self.model.global_pool, "mean"), tokens.mean(0)))
        self.assertTrue(torch.allclose(self.model._pool_tokens(tokens, self.model.global_pool, "max"), tokens.max(0).values))
        self.assertTrue(
            torch.allclose(
                self.model._pool_tokens(tokens, self.model.global_pool, "attention"),
                tokens.mean(0),
            )
        )

    def test_checkpoint_cleanup_strips_ddp_and_ivtlr_prefixes(self):
        state_dict = {
            "module.base_causallm.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(1),
            "base_causallm.model.layers.0.self_attn.k_proj.lora_A.default.weight": torch.ones(1),
            "model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.ones(1),
        }

        cleaned = self.model._clean_checkpoint_state_dict(state_dict)

        self.assertIn("model.layers.0.self_attn.q_proj.lora_A.default.weight", cleaned)
        self.assertIn("model.layers.0.self_attn.k_proj.lora_A.default.weight", cleaned)
        self.assertIn("model.layers.0.self_attn.v_proj.lora_A.default.weight", cleaned)

    def test_checkpoint_alignment_matches_qwen_peft_wrapper_depth(self):
        state_dict = {
            "base_model.model.visual.patch_embed.proj.weight": torch.ones(1),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(1),
            "embedding.weight": torch.ones(1),
        }
        target_state_dict = {
            "base_model.model.model.visual.patch_embed.proj.weight": torch.zeros(1),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(1),
        }

        aligned = self.model._align_checkpoint_state_dict(state_dict, target_state_dict)

        self.assertIn("base_model.model.model.visual.patch_embed.proj.weight", aligned)
        self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight", aligned)
        self.assertIn("embedding.weight", aligned)

    def test_checkpoint_alignment_matches_qwen_language_model_submodule(self):
        state_dict = {
            "base_model.model.model.embed_tokens.weight": torch.ones(1),
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.ones(1),
            "base_model.model.model.norm.weight": torch.ones(1),
            "base_model.model.model.visual.patch_embed.proj.weight": torch.ones(1),
        }
        target_state_dict = {
            "base_model.model.model.language_model.embed_tokens.weight": torch.zeros(1),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros(1),
            "base_model.model.model.language_model.norm.weight": torch.zeros(1),
            "base_model.model.model.visual.patch_embed.proj.weight": torch.zeros(1),
        }

        aligned = self.model._align_checkpoint_state_dict(state_dict, target_state_dict)

        self.assertIn("base_model.model.model.language_model.embed_tokens.weight", aligned)
        self.assertIn("base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight", aligned)
        self.assertIn("base_model.model.model.language_model.norm.weight", aligned)
        self.assertIn("base_model.model.model.visual.patch_embed.proj.weight", aligned)

    def test_checkpoint_loading_can_be_disabled_with_path_present(self):
        model = build_model(checkpoint_path="/tmp/unused.pt", use_checkpoint=False)

        self.assertFalse(model.use_checkpoint)
        self.assertEqual(model.checkpoint_path, "/tmp/unused.pt")

    def test_checkpoint_loading_requires_path_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "no checkpoint_path"):
            build_model(use_checkpoint=True)

    def test_build_visual_bank_shapes(self):
        self.assertEqual(tuple(self.bank["patches"].shape), (4, 4))
        self.assertEqual(tuple(self.bank["regions"].shape), (4, 4))
        self.assertEqual(tuple(self.bank["raw_regions"].shape), (4, 1, 4))
        self.assertEqual(tuple(self.bank["global"].shape), (1, 4))
        self.assertTrue(torch.allclose(self.bank["global"][0], torch.tensor([0.25, 0.25, 0.25, 0.25])))

    def test_build_visual_bank_infers_merged_patch_grid(self):
        model = build_merged_grid_model()
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        bank = model.build_visual_bank(projected)
        self.assertEqual(tuple(model._current_postmerge_grid), (2, 2))
        self.assertEqual(tuple(bank["patches"].shape), (4, 4))

    def test_projected_image_tokens_resolve_nested_visual_encoder(self):
        model = build_nested_visual_model()
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        bank = model.build_visual_bank(projected)
        self.assertEqual(tuple(model._current_postmerge_grid), (2, 2))
        self.assertEqual(tuple(bank["patches"].shape), (4, 4))

    def test_build_visual_bank_pads_non_divisible_grid(self):
        model = build_model()
        model.region_window = 2
        model._current_postmerge_grid = (3, 5)
        projected = torch.arange(60, dtype=torch.float32).view(15, 4)

        bank = model.build_visual_bank(projected)

        self.assertEqual(tuple(bank["patches"].shape), (15, 4))
        self.assertEqual(tuple(bank["regions"].shape), (6, 4))
        self.assertEqual(tuple(bank["raw_regions"].shape), (6, 4, 4))
        self.assertEqual(tuple(bank["global"].shape), (1, 4))

    def test_default_initial_state_has_no_control_tokens(self):
        state = self.model.build_initial_state(self.prepared)
        self.assertEqual(state["inputs_embeds"].size(1), self.prepared["input_ids"].size(1))
        self.assertIsNone(state["latent_pos"])
        self.assertIsNone(state["act_pos"])

    def test_coarse_initial_state_replaces_image_span_with_global_token(self):
        state = self.model.build_coarse_initial_state(self.prepared, self.bank)

        self.assertEqual(state["inputs_embeds"].size(1), 3)
        self.assertTrue(torch.allclose(state["inputs_embeds"][:, 1, :], self.bank["global"]))
        self.assertIsNone(state["latent_pos"])
        self.assertIsNone(state["act_pos"])

    def test_apply_mined_actions_supports_multi_patch_and_think(self):
        state = self.model.build_coarse_initial_state(self.prepared, self.bank)
        initial_length = state["inputs_embeds"].size(1)

        updated = self.model.apply_mined_actions(
            state,
            self.bank,
            [
                {"type": "PATCH", "patch_idx": 0},
                {"type": "PATCH", "patch_idx": 1},
                {"type": "THINK"},
            ],
        )

        self.assertEqual(updated["inputs_embeds"].size(1), initial_length + 3)
        self.assertTrue(torch.allclose(updated["inputs_embeds"][:, 1, :], self.bank["global"]))
        self.assertTrue(torch.allclose(updated["inputs_embeds"][:, 2, :], self.bank["patches"][0].view(1, -1)))
        self.assertTrue(torch.allclose(updated["inputs_embeds"][:, 3, :], self.bank["patches"][1].view(1, -1)))

    def test_apply_mined_region_inserts_raw_region_tokens(self):
        model = build_model(controller_context_window=1, region_window=2)
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        bank = model.build_visual_bank(projected)
        state = model.build_coarse_initial_state(prepared, bank)

        updated = model.apply_mined_actions(state, bank, [{"type": "REGION", "region_idx": 0}])

        self.assertEqual(updated["inputs_embeds"].size(1), 7)
        self.assertTrue(torch.allclose(updated["inputs_embeds"][:, 2:6, :], bank["raw_regions"][0].unsqueeze(0)))

    def test_tokenless_controller_uses_last_hidden_state_and_step(self):
        state = self.model.build_initial_state(self.prepared)
        capture = {}
        self._set_controller_action(ACTION_STOP, capture=capture)

        self.model.forward_reasoning_step(state, self.bank, 0)

        expected = F.layer_norm(state["inputs_embeds"][:, -1, :] + 0.05, (self.model.hidden_size,))
        self.assertTrue(torch.allclose(capture["state_hidden"], expected))
        self.assertIsNone(capture["act_hidden"])
        self.assertEqual(tuple(capture["step_hidden"].shape), (1, 4))

    def test_reasoning_step_supports_causal_outputs_without_last_hidden_state(self):
        model = build_model()
        calls = []

        def forward(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        ):
            del input_ids, attention_mask, return_dict, use_cache, kwargs
            calls.append(output_hidden_states)
            seq_len = inputs_embeds.size(1)
            positions = torch.arange(seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, seq_len, 1)
            hidden = inputs_embeds + positions * 0.01
            logits = torch.zeros((1, seq_len, 4), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            return types.SimpleNamespace(logits=logits, hidden_states=[inputs_embeds, hidden])

        model.backbone.forward = forward
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        bank = model.build_visual_bank(projected)
        state = model.build_initial_state(prepared)
        capture = {}

        def controller_forward(state_hidden, step_hidden, bank, act_hidden=None):
            capture["state_hidden"] = state_hidden.detach().clone()
            type_logits = torch.full((1, 5), -10.0)
            type_logits[0, ACTION_STOP] = 10.0
            region_logits = torch.zeros(1, bank["regions"].size(0))
            patch_logits = torch.zeros(1, bank["patches"].size(0))
            return type_logits, region_logits, patch_logits

        model.controller.forward = controller_forward
        model.forward_reasoning_step(state, bank, 0)

        self.assertEqual(calls, [True])
        expected = F.layer_norm(state["inputs_embeds"][:, -1, :] + 0.05, (model.hidden_size,))
        self.assertTrue(torch.allclose(capture["state_hidden"], expected))

    def test_forward_reasoning_actions_tokenless(self):
        for action_id in [ACTION_THINK, ACTION_STOP, ACTION_GLOBAL, ACTION_REGION, ACTION_PATCH]:
            with self.subTest(action_id=action_id):
                state = self.model.build_initial_state(self.prepared)
                initial_length = state["inputs_embeds"].size(1)
                initial_final_embed = state["inputs_embeds"][:, -1, :].clone()
                self._set_controller_action(action_id)
                updated_state, selected_action, should_stop, step_trace = self.model.forward_reasoning_step(
                    state, self.bank, 0
                )
                self.assertEqual(selected_action, action_id)
                if action_id == ACTION_THINK:
                    self.assertEqual(step_trace["sequence_length_after"], initial_length + 1)
                    appended = updated_state["inputs_embeds"][:, -1, :]
                    self.assertTrue(torch.allclose(appended, initial_final_embed + 0.05))
                elif action_id in [ACTION_GLOBAL, ACTION_REGION, ACTION_PATCH]:
                    self.assertEqual(step_trace["sequence_length_after"], initial_length + 1)
                    self.assertTrue(torch.allclose(updated_state["inputs_embeds"][:, -1, :], initial_final_embed))
                    self.assertIsNone(updated_state["latent_pos"])
                    self.assertIsNone(updated_state["act_pos"])
                else:
                    self.assertEqual(step_trace["sequence_length_after"], initial_length)
                    self.assertTrue(torch.allclose(updated_state["inputs_embeds"][:, -1, :], initial_final_embed))
                self.assertEqual(should_stop, action_id == ACTION_STOP)

    def test_controller_temperature_flattens_action_distribution(self):
        cold_model = build_model(controller_temperature=0.5)
        hot_model = build_model(controller_temperature=2.0)

        def forward(state_hidden, step_hidden, bank, act_hidden=None):
            del state_hidden, step_hidden, bank, act_hidden
            type_logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
            region_logits = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            patch_logits = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            return type_logits, region_logits, patch_logits

        cold_model.controller.forward = forward
        hot_model.controller.forward = forward

        cold_state = cold_model.build_initial_state(self.prepared)
        hot_state = hot_model.build_initial_state(self.prepared)
        cold_model._current_postmerge_grid = self.model._current_postmerge_grid
        hot_model._current_postmerge_grid = self.model._current_postmerge_grid

        _, _, _, cold_trace = cold_model.forward_reasoning_step(cold_state, cold_model.build_visual_bank(self.projected), 0)
        _, _, _, hot_trace = hot_model.forward_reasoning_step(hot_state, hot_model.build_visual_bank(self.projected), 0)

        self.assertGreater(cold_trace["action_probs"][0], hot_trace["action_probs"][0])
        self.assertLess(cold_trace["action_probs"][1], hot_trace["action_probs"][1])
        self.assertEqual(cold_trace["controller_temperature"], 0.5)
        self.assertEqual(hot_trace["controller_temperature"], 2.0)

    def test_training_forward_decodes_detached_state_without_grad(self):
        captured = {}

        def fake_decode(state, labels=None):
            del labels
            captured["requires_grad"] = state["inputs_embeds"].requires_grad
            captured["grad_enabled"] = torch.is_grad_enabled()
            return {
                "answer": "yes",
                "generated_text": "<answer>yes</answer>",
                "generated_ids": [1],
                "decode_prefix_length": state["inputs_embeds"].size(1),
                "final_sequence_length": state["inputs_embeds"].size(1),
            }

        self.model.train()
        self.model.max_steps = 1
        self._set_controller_action(ACTION_GLOBAL)
        self.model.decode_answer = fake_decode

        output = self.model.forward("image", "question", sample_actions=True)

        self.assertEqual(output["answer"], "yes")
        self.assertFalse(captured["requires_grad"])
        self.assertFalse(captured["grad_enabled"])

    def test_legacy_control_tokens_still_drop_act_token(self):
        model = build_model(use_control_tokens=True, think_append_hidden=False)
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        state = model.build_initial_state(prepared)

        self.assertEqual(state["inputs_embeds"].size(1), prepared["input_ids"].size(1) + 2)
        self.assertIsNotNone(state["latent_pos"])
        self.assertIsNotNone(state["act_pos"])

        dropped = model.drop_act_token(state)
        self.assertTrue(dropped["inputs_embeds"].requires_grad)
        self.assertIsNotNone(dropped["inputs_embeds"].grad_fn)
        self.assertIsNone(dropped["act_pos"])

    def test_pooled_baseline_reduces_image_span_to_one_embedding(self):
        mean_output = self.model.pooled_baseline_forward("image", "question", pooling="mean")
        max_output = self.model.pooled_baseline_forward("image", "question", pooling="max")

        self.assertEqual(mean_output["decode_prefix_length"], 3)
        self.assertEqual(max_output["decode_prefix_length"], 3)
        self.assertEqual(mean_output["num_steps"], 0)
        self.assertEqual(max_output["trace"], [])

    def test_region_baseline_replaces_image_span_with_region_tokens(self):
        model = build_model()
        model.region_window = 2

        mean_output = model.region_baseline_forward("image", "question", pooling="mean")
        max_output = model.region_baseline_forward("image", "question", pooling="max")

        self.assertEqual(mean_output["num_region_tokens"], 1)
        self.assertEqual(max_output["num_region_tokens"], 1)
        self.assertEqual(mean_output["decode_prefix_length"], 3)
        self.assertEqual(max_output["decode_prefix_length"], 3)
        self.assertEqual(mean_output["trace"], [])

    def test_baseline_excludes_latent_and_act_tokens(self):
        baseline = self.model.baseline_forward("image", "question")
        self.assertEqual(baseline["decode_prefix_length"], self.prepared["input_ids"].size(1))
        self.assertEqual(baseline["num_steps"], 0)
        self.assertEqual(baseline["trace"], [])

    def test_generate_lvar_respects_action_selection(self):
        model = build_model()
        captured = {}

        def fake_forward(images, questions, labels=None, sample_actions=None):
            del images, questions, labels
            captured["sample_actions"] = sample_actions
            return {
                "answer": "yes",
                "trace": [],
                "num_steps": 0,
                "generated_text": "<answer>yes</answer>",
                "generated_ids": [1],
            }

        model.forward = fake_forward

        model.action_selection = "sample"
        model.generate_lvar("image", "question")
        self.assertTrue(captured["sample_actions"])

        model.action_selection = "argmax"
        model.generate_lvar("image", "question")
        self.assertFalse(captured["sample_actions"])

    def test_controller_reads_multiple_hidden_states(self):
        model = build_model(controller_context_window=3)
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        bank = model.build_visual_bank(projected)
        state = model.build_initial_state(prepared)
        capture = {}

        def forward(state_hidden, step_hidden, bank, act_hidden=None):
            capture["state_hidden"] = state_hidden.detach().clone()
            capture["step_hidden"] = step_hidden.detach().clone()
            type_logits = torch.full((1, 5), -10.0)
            type_logits[0, ACTION_STOP] = 10.0
            region_logits = torch.zeros(1, bank["regions"].size(0))
            patch_logits = torch.zeros(1, bank["patches"].size(0))
            return type_logits, region_logits, patch_logits

        model.controller.forward = forward
        model.forward_reasoning_step(state, bank, 0)

        self.assertEqual(tuple(capture["state_hidden"].shape), (1, 12))
        expected_hidden = state["inputs_embeds"] + torch.arange(
            state["inputs_embeds"].size(1),
            dtype=state["inputs_embeds"].dtype,
        ).view(1, -1, 1) * 0.01
        expected = F.layer_norm(expected_hidden[:, -3:, :], (model.hidden_size,)).reshape(1, -1)
        self.assertTrue(torch.allclose(capture["state_hidden"], expected))

    def test_phase3_trainable_filter_keeps_only_controller_facing_params(self):
        model = build_model(controller_context_window=3, controller_max_steps=16)

        trainable = set_controller_sft_trainable(model)
        trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertTrue(trainable)
        self.assertIn("controller_state_norm.weight", trainable_names)
        self.assertIn("controller_state_norm.bias", trainable_names)
        self.assertTrue(any(name.startswith("controller.") for name in trainable_names))
        self.assertIn("step_embedding.weight", trainable_names)
        self.assertNotIn("latent_token", trainable_names)
        self.assertNotIn("act_token", trainable_names)
        self.assertFalse(any(name.startswith("global_pool.") for name in trainable_names))
        self.assertFalse(any(name.startswith("region_pool.") for name in trainable_names))
        self.assertFalse(any(name.startswith("backbone.") for name in trainable_names))

    def test_controller_index_logits_are_normalized_and_finite(self):
        head = ControllerHead(
            hidden_size=4,
            num_actions=5,
            controller_num_states=1,
            index_temperature=0.07,
        )
        state_hidden = torch.randn(1, 4) * 1000.0
        step_hidden = torch.randn(1, 4) * 1000.0
        bank = {
            "regions": torch.randn(3, 4) * 1000.0,
            "patches": torch.randn(4, 4) * 1000.0,
        }

        _, region_logits, patch_logits = head(state_hidden, step_hidden, bank)

        self.assertTrue(torch.isfinite(region_logits).all())
        self.assertTrue(torch.isfinite(patch_logits).all())
        self.assertLessEqual(float(region_logits.abs().max()), (1.0 / 0.07) + 1e-5)
        self.assertLessEqual(float(patch_logits.abs().max()), (1.0 / 0.07) + 1e-5)

    def test_region_inserts_raw_patches(self):
        model = build_model(controller_context_window=1, region_window=2)
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        bank = model.build_visual_bank(projected)
        state = model.build_initial_state(prepared)
        initial_length = state["inputs_embeds"].size(1)

        def forward(state_hidden, step_hidden, bank, act_hidden=None):
            type_logits = torch.full((1, 5), -10.0)
            type_logits[0, ACTION_REGION] = 10.0
            region_logits = torch.zeros(1, bank["regions"].size(0))
            region_logits[0, 0] = 10.0
            patch_logits = torch.zeros(1, bank["patches"].size(0))
            return type_logits, region_logits, patch_logits

        model.controller.forward = forward
        updated_state, action_id, should_stop, step_trace = model.forward_reasoning_step(
            state, bank, 0
        )

        self.assertEqual(action_id, ACTION_REGION)
        self.assertEqual(step_trace["sequence_length_after"], initial_length + 4)
        self.assertIn("raw_regions", bank)
        self.assertEqual(tuple(bank["raw_regions"].shape), (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
