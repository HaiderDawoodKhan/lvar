import random
import types
import unittest

import torch

from lvar.counterfactual_training import (
    CONTEXT_FULL,
    CONTEXT_GLOBAL,
    build_negative_actions,
    differentiable_state_ce,
    load_counterfactual_pairs,
    phase4_parameter_groups,
    replay_counterfactual_pair_loss,
    sample_context_mode,
    set_phase4_trainable,
    validate_negative_type_probs,
)


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        del text, return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([[1, 0]], dtype=torch.long)}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.lora_adapter = torch.nn.Parameter(torch.tensor(0.1))
        self.base_weight = torch.nn.Parameter(torch.tensor(0.2))
        self.visual = torch.nn.Identity()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, attention_mask=None, return_dict=True, use_cache=False, **kwargs):
        del attention_mask, return_dict, use_cache, kwargs
        batch, seq_len, _ = inputs_embeds.shape
        logits = torch.zeros(batch, seq_len, 8, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        signal = inputs_embeds.sum(dim=-1) + self.lora_adapter
        logits[..., 1] = signal
        logits[..., 0] = -signal
        return types.SimpleNamespace(logits=logits)


class TinyPhase4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.processor = types.SimpleNamespace(tokenizer=TinyTokenizer())
        self.backbone = TinyBackbone()
        self.controller = torch.nn.Linear(3, 5)
        self.step_embedding = torch.nn.Embedding(8, 3)
        self.controller_state_norm = torch.nn.LayerNorm(3)
        self.latent_token = torch.nn.Parameter(torch.ones(3))
        self.global_pool = torch.nn.Linear(3, 1)
        self.region_pool = torch.nn.Linear(3, 1)

    def _embed_input_ids(self, input_ids):
        return self.backbone.get_input_embeddings()(input_ids)

    def prepare_inputs(self, image, question, add_answer_instruction=False, image_size=None):
        return {"image": image, "question": question, "image_size": image_size}

    def get_projected_image_tokens(self, batch):
        offset = 10.0 if batch["image"] == "neg-image" else 0.0
        return torch.arange(12, dtype=torch.float32).view(4, 3) + offset

    def build_visual_bank(self, image_tokens):
        return {
            "global": image_tokens.mean(0, keepdim=True),
            "patches": image_tokens,
            "regions": image_tokens[:2],
            "raw_regions": torch.stack([image_tokens[:2], image_tokens[2:4]], dim=0),
        }

    def build_initial_state(self, batch):
        del batch
        return {
            "inputs_embeds": torch.ones(1, 4, 3),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "latent_pos": None,
            "act_pos": None,
        }

    def build_coarse_initial_state(self, batch, bank):
        del batch, bank
        return {
            "inputs_embeds": torch.zeros(1, 2, 3),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "latent_pos": None,
            "act_pos": None,
        }

    def clone_state(self, state):
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

    def _insert_evidence_token(self, state, evidence_tokens):
        projected = evidence_tokens.unsqueeze(0)
        state["inputs_embeds"] = torch.cat([state["inputs_embeds"], projected], dim=1)
        state["attention_mask"] = torch.cat(
            [state["attention_mask"], torch.ones(1, projected.size(1), dtype=torch.long)],
            dim=1,
        )

    def apply_mined_actions(self, state, bank, actions):
        for action in actions:
            action_type = action["type"].upper()
            if action_type == "PATCH":
                self._insert_evidence_token(state, bank["patches"][int(action["patch_idx"])].unsqueeze(0))
            elif action_type == "REGION":
                self._insert_evidence_token(state, bank["raw_regions"][int(action["region_idx"])])
            elif action_type == "GLOBAL":
                self._insert_evidence_token(state, bank["global"])
            elif action_type == "THINK":
                self._insert_evidence_token(state, torch.zeros(1, 3))
        return state

    def controller_logits_from_state(self, state, bank, step_idx):
        del state, step_idx
        hidden = torch.zeros(1, 3)
        type_logits = self.controller(hidden)
        patch_logits = torch.zeros(1, bank["patches"].size(0))
        region_logits = torch.zeros(1, bank["raw_regions"].size(0))
        return type_logits, region_logits, patch_logits


class CounterfactualTrainingTests(unittest.TestCase):
    def test_load_counterfactual_pairs_flattens_mined_rows(self):
        path = "/tmp/test_phase4_pairs.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"example_id":"ex","question":"q","counterfactual_pairs":[{"prefix_trace":[],"positive_actions":[{"type":"PATCH","patch_idx":1}],"target_text":"target"}]}\n'
            )

        pairs = load_counterfactual_pairs(path)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["example_id"], "ex")
        self.assertEqual(pairs[0]["positive_actions"][0]["patch_idx"], 1)

    def test_context_sampler_uses_one_mode_per_pair(self):
        self.assertEqual(sample_context_mode(1.0, random.Random(0)), CONTEXT_FULL)
        self.assertEqual(sample_context_mode(0.0, random.Random(0)), CONTEXT_GLOBAL)

    def test_negative_probabilities_normalize_50_35_15(self):
        probs = validate_negative_type_probs(
            {"same_image_wrong": 0.5, "different_image_random": 0.35, "same_image_noisy": 0.15}
        )

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs["same_image_wrong"], 0.5)
        self.assertAlmostEqual(probs["different_image_random"], 0.35)
        self.assertAlmostEqual(probs["same_image_noisy"], 0.15)

    def test_same_image_negatives_preserve_structure_and_avoid_positive_ids(self):
        bank = TinyPhase4Model().build_visual_bank(torch.arange(12, dtype=torch.float32).view(4, 3))
        pair = {
            "example_id": "ex",
            "prefix_trace": [{"type": "PATCH", "patch_idx": 0}],
            "positive_actions": [
                {"type": "PATCH", "patch_idx": 1},
                {"type": "PATCH", "patch_idx": 1},
                {"type": "THINK"},
                {"type": "REGION", "region_idx": 0},
            ],
        }

        actions, reason = build_negative_actions(pair, bank, "same_image_wrong", random.Random(2))

        self.assertIsNone(reason)
        self.assertEqual([action["type"] for action in actions], ["PATCH", "PATCH", "THINK", "REGION"])
        self.assertEqual(actions[0]["patch_idx"], actions[1]["patch_idx"])
        self.assertNotEqual(actions[0]["patch_idx"], 1)
        self.assertNotEqual(actions[3]["region_idx"], 0)

    def test_different_image_uses_one_negative_source_for_whole_sequence(self):
        model = TinyPhase4Model()
        source_bank = model.build_visual_bank(torch.arange(12, dtype=torch.float32).view(4, 3))
        pair = {
            "example_id": "pos",
            "question": "q",
            "positive_actions": [
                {"type": "PATCH", "patch_idx": 1},
                {"type": "REGION", "region_idx": 0},
            ],
        }
        example_index = {
            "pos": {"id": "pos", "image": "pos-image", "question": "q"},
            "neg": {"id": "neg", "image": "neg-image", "question": "q"},
        }

        actions, reason = build_negative_actions(
            pair,
            source_bank,
            "different_image_random",
            random.Random(1),
            example_index=example_index,
            model=model,
            image_size=280,
            negative_bank_cache={},
        )

        self.assertIsNone(reason)
        self.assertEqual(actions[0]["negative_example_id"], "neg")
        self.assertEqual(actions[1]["negative_example_id"], "neg")
        self.assertIn("evidence_tokens", actions[0])
        self.assertIn("evidence_tokens", actions[1])

    def test_noisy_negative_preserves_ids_and_rescales_norm(self):
        bank = TinyPhase4Model().build_visual_bank(torch.arange(12, dtype=torch.float32).view(4, 3))
        pair = {
            "example_id": "ex",
            "positive_actions": [{"type": "PATCH", "patch_idx": 2}],
        }

        actions, reason = build_negative_actions(pair, bank, "same_image_noisy", random.Random(3), noise_scale=0.5)

        self.assertIsNone(reason)
        self.assertEqual(actions[0]["patch_idx"], 2)
        original_norm = torch.linalg.vector_norm(bank["patches"][2].unsqueeze(0).float())
        noisy_norm = torch.linalg.vector_norm(actions[0]["evidence_tokens"].float())
        self.assertTrue(torch.allclose(original_norm, noisy_norm, atol=1e-5))

    def test_differentiable_ce_backprops_to_lora_parameter(self):
        model = TinyPhase4Model()
        state = model.build_coarse_initial_state({}, {})

        loss = differentiable_state_ce(model, state, "target")
        loss.backward()

        self.assertIsNotNone(model.backbone.lora_adapter.grad)

    def test_replay_loss_matches_weighted_formula_and_preserves_shared_context(self):
        model = TinyPhase4Model()
        pair = {
            "example_id": "pos",
            "question": "q",
            "prefix_trace": [{"type": "PATCH", "patch_idx": 0}],
            "positive_actions": [{"type": "PATCH", "patch_idx": 1}, {"type": "THINK"}],
            "target_text": "target",
        }
        example_index = {
            "pos": {"id": "pos", "image": "pos-image", "question": "q"},
            "neg": {"id": "neg", "image": "neg-image", "question": "q"},
        }

        loss, metrics = replay_counterfactual_pair_loss(
            model,
            pair,
            example_index["pos"],
            example_index,
            random.Random(0),
            {"same_image_wrong": 1.0, "different_image_random": 0.0, "same_image_noisy": 0.0},
            context_full_probability=1.0,
            positive_ce_weight=0.2,
            rank_weight=0.4,
            rank_margin=0.1,
        )

        expected = metrics["l_ctrl"] + 0.2 * metrics["ce_pos"] + 0.4 * metrics["rank_loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach().item()), expected, places=5)
        self.assertEqual(metrics["context_mode"], CONTEXT_FULL)
        self.assertEqual(metrics["action_counts"]["PATCH"], 1)
        self.assertEqual(metrics["action_counts"]["THINK"], 1)

    def test_phase4_trainable_filter_keeps_controller_and_lora_only(self):
        model = TinyPhase4Model()

        set_phase4_trainable(model)
        trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertIn("backbone.lora_adapter", trainable_names)
        self.assertTrue(any(name.startswith("controller.") for name in trainable_names))
        self.assertIn("step_embedding.weight", trainable_names)
        self.assertIn("controller_state_norm.weight", trainable_names)
        self.assertNotIn("latent_token", trainable_names)
        self.assertFalse(any(name.startswith("global_pool.") for name in trainable_names))
        self.assertFalse(any(name.startswith("region_pool.") for name in trainable_names))
        self.assertNotIn("backbone.base_weight", trainable_names)

    def test_phase4_parameter_groups_split_controller_and_lora(self):
        model = TinyPhase4Model()

        set_phase4_trainable(model)
        groups = phase4_parameter_groups(model)
        lora_ids = {id(parameter) for parameter in groups["lora"]}
        controller_ids = {id(parameter) for parameter in groups["controller"]}

        self.assertIn(id(model.backbone.lora_adapter), lora_ids)
        self.assertIn(id(model.controller.weight), controller_ids)
        self.assertIn(id(model.step_embedding.weight), controller_ids)
        self.assertNotIn(id(model.backbone.lora_adapter), controller_ids)


if __name__ == "__main__":
    unittest.main()
