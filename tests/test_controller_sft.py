import unittest

import torch
import torch.nn.functional as F

from lvar.controller_sft import (
    compute_action_loss,
    flatten_supervised_actions,
    replay_controller_sft_loss,
)
from lvar.utils import ACTION_GLOBAL, ACTION_PATCH, ACTION_REGION, ACTION_STOP, ACTION_THINK


class TinyReplayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.seen_steps = []
        self.applied_actions = []
        self.initial_modes = []

    def prepare_inputs(self, images, questions, add_answer_instruction=True, image_size=None):
        return {
            "image": images,
            "question": questions,
            "add_answer_instruction": add_answer_instruction,
            "image_size": image_size,
        }

    def get_projected_image_tokens(self, batch):
        del batch
        return torch.arange(16, dtype=torch.float32).view(4, 4)

    def build_visual_bank(self, image_tokens):
        return {
            "global": image_tokens.mean(0, keepdim=True),
            "regions": image_tokens[:3],
            "patches": image_tokens,
            "raw_regions": image_tokens[:3].unsqueeze(1),
        }

    def build_coarse_initial_state(self, batch, bank):
        del bank
        self.initial_modes.append("global_mean")
        return {"batch": batch, "applied": []}

    def build_initial_state(self, batch):
        self.initial_modes.append("full_context")
        return {"batch": batch, "applied": []}

    def controller_logits_from_state(self, state, bank, step_idx):
        del state, bank
        self.seen_steps.append(step_idx)
        type_logits = torch.zeros((1, 5), dtype=torch.float32)
        region_logits = torch.zeros((1, 3), dtype=torch.float32)
        patch_logits = torch.zeros((1, 4), dtype=torch.float32)
        return type_logits, region_logits, patch_logits

    def apply_mined_actions(self, state, bank, actions):
        del bank
        for action in actions:
            action_type = action["type"].upper()
            state["applied"].append(action_type)
            self.applied_actions.append(action_type)
        return state


class ControllerSFTTests(unittest.TestCase):
    def test_flatten_actions_skips_noop_decisions_and_appends_stop(self):
        decisions = [
            {"selected": "PATCH_SEQ", "actions": [{"type": "PATCH", "patch_idx": 2}, {"type": "PATCH", "patch_idx": 1}]},
            {"selected": "NO_OP", "actions": []},
            {"selected": "THINK", "actions": [{"type": "THINK"}]},
        ]

        actions = flatten_supervised_actions(decisions)

        self.assertEqual([action["type"] for action in actions], ["PATCH", "PATCH", "THINK", "STOP"])
        self.assertEqual(actions[0]["patch_idx"], 2)
        self.assertEqual(actions[1]["patch_idx"], 1)

    def test_action_loss_uses_type_only_for_non_indexed_actions(self):
        type_logits = torch.tensor([[0.0, 2.0, 1.0, -1.0, -2.0]])
        region_logits = torch.tensor([[10.0, -10.0]])
        patch_logits = torch.tensor([[10.0, -10.0]])

        for action_type, action_id in [("THINK", ACTION_THINK), ("GLOBAL", ACTION_GLOBAL), ("STOP", ACTION_STOP)]:
            with self.subTest(action_type=action_type):
                loss = compute_action_loss(type_logits, region_logits, patch_logits, {"type": action_type})
                expected = F.cross_entropy(type_logits, torch.tensor([action_id]))
                self.assertTrue(torch.allclose(loss, expected))

                component_loss, components = compute_action_loss(
                    type_logits,
                    region_logits,
                    patch_logits,
                    {"type": action_type},
                    return_components=True,
                )
                self.assertTrue(torch.allclose(component_loss, expected))
                self.assertTrue(torch.allclose(components["type_loss"], expected))
                self.assertEqual(components["action_type"], action_type)

    def test_patch_and_region_losses_include_index_heads(self):
        type_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        region_logits = torch.tensor([[0.0, 2.0, -1.0]])
        patch_logits = torch.tensor([[1.0, -1.0, 3.0, 0.0]])

        patch_loss = compute_action_loss(type_logits, region_logits, patch_logits, {"type": "PATCH", "patch_idx": 2})
        expected_patch = F.cross_entropy(type_logits, torch.tensor([ACTION_PATCH])) + F.cross_entropy(
            patch_logits,
            torch.tensor([2]),
        )
        self.assertTrue(torch.allclose(patch_loss, expected_patch))

        region_loss = compute_action_loss(type_logits, region_logits, patch_logits, {"type": "REGION", "region_idx": 1})
        expected_region = F.cross_entropy(type_logits, torch.tensor([ACTION_REGION])) + F.cross_entropy(
            region_logits,
            torch.tensor([1]),
        )
        self.assertTrue(torch.allclose(region_loss, expected_region))

    def test_replay_skips_noop_without_incrementing_step_and_adds_stop(self):
        model = TinyReplayModel()
        mined_row = {
            "example_id": "ex-1",
            "question": "formatted question",
            "decisions": [
                {"selected": "PATCH_SEQ", "actions": [{"type": "PATCH", "patch_idx": 2}]},
                {"selected": "NO_OP", "actions": []},
                {"selected": "THINK", "actions": [{"type": "THINK"}]},
                {"selected": "REGION", "actions": [{"type": "REGION", "region_idx": 1}]},
            ],
        }
        source_example = {"id": "ex-1", "image": "image", "question": "source question"}

        loss, metrics = replay_controller_sft_loss(model, mined_row, source_example, image_size=280)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.seen_steps, [0, 1, 2, 3])
        self.assertEqual(model.applied_actions, ["PATCH", "THINK", "REGION"])
        self.assertEqual(metrics["num_targets"], 4)
        self.assertEqual(metrics["skipped_noop_decisions"], 1)
        self.assertEqual(metrics["action_counts"]["STOP"], 1)
        self.assertIn("type_loss", metrics["loss_components"])
        self.assertIn("patch_loss", metrics["loss_components"])
        self.assertIn("region_loss", metrics["loss_components"])
        self.assertIn("PATCH", metrics["action_loss_means"])
        self.assertIn("type_logits_max", metrics["logit_stats"])
        self.assertIn("patch_logits_mean", metrics["logit_stats"])
        self.assertEqual(metrics["initial_visual_mode"], "global_mean")

    def test_replay_supports_31_actions_plus_stop_step(self):
        model = TinyReplayModel()
        mined_row = {
            "example_id": "ex-1",
            "question": "formatted question",
            "decisions": [{"selected": "THINK", "actions": [{"type": "THINK"}]} for _ in range(31)],
        }
        source_example = {"id": "ex-1", "image": "image", "question": "source question"}

        loss, metrics = replay_controller_sft_loss(model, mined_row, source_example, image_size=280)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.seen_steps, list(range(32)))
        self.assertEqual(metrics["num_targets"], 32)
        self.assertEqual(metrics["num_controller_steps"], 32)

    def test_replay_can_start_from_full_context_by_probability(self):
        model = TinyReplayModel()
        mined_row = {
            "example_id": "ex-1",
            "question": "formatted question",
            "decisions": [{"selected": "NO_OP", "actions": []}],
        }
        source_example = {"id": "ex-1", "image": "image", "question": "source question"}

        _, metrics = replay_controller_sft_loss(
            model,
            mined_row,
            source_example,
            full_context_probability=1.0,
        )

        self.assertEqual(model.initial_modes, ["full_context"])
        self.assertEqual(metrics["initial_visual_mode"], "full_context")
        self.assertTrue(metrics["used_full_context"])

    def test_replay_rejects_invalid_full_context_probability(self):
        model = TinyReplayModel()
        mined_row = {"example_id": "ex-1", "decisions": []}
        source_example = {"id": "ex-1", "image": "image", "question": "source question"}

        with self.assertRaisesRegex(ValueError, "full_context_probability"):
            replay_controller_sft_loss(
                model,
                mined_row,
                source_example,
                full_context_probability=1.5,
            )


if __name__ == "__main__":
    unittest.main()
