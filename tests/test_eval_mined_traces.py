import random
import unittest
from types import SimpleNamespace

import torch

from lvar_scripts.eval_mined_traces_m3cot import (
    build_replay_blocks,
    build_replay_trace,
    flatten_replay_blocks,
    next_token_entropy_from_state,
    rewrite_visual_action_index,
)


class ReplayVisualIndexModeTests(unittest.TestCase):
    def test_original_mode_preserves_action_and_does_not_mutate_it(self):
        action = {"type": "PATCH", "patch_idx": 10, "metadata": "kept"}

        rewritten = rewrite_visual_action_index(action, "original", num_regions=25, num_patches=100)

        self.assertEqual(rewritten, action)
        self.assertIsNot(rewritten, action)

    def test_last_mode_uses_final_patch_and_region_indices(self):
        patch = rewrite_visual_action_index(
            {"type": "PATCH", "patch_idx": 10}, "last", num_regions=25, num_patches=100
        )
        region = rewrite_visual_action_index(
            {"type": "REGION", "region_idx": 13}, "last", num_regions=25, num_patches=100
        )

        self.assertEqual(patch["patch_idx"], 99)
        self.assertEqual(region["region_idx"], 24)

    def test_random_mode_samples_each_visual_action_from_its_bank(self):
        rng = random.Random(7)
        actions = [
            {"type": "THINK"},
            {"type": "GLOBAL"},
            {"type": "PATCH", "patch_idx": 10},
            {"type": "PATCH", "patch_idx": 88},
            {"type": "REGION", "region_idx": 13},
            {"type": "STOP"},
        ]

        rewritten = [
            rewrite_visual_action_index(action, "random", num_regions=25, num_patches=100, rng=rng)
            for action in actions
        ]

        self.assertEqual([action["type"] for action in rewritten], [action["type"] for action in actions])
        self.assertEqual([rewritten[2]["patch_idx"], rewritten[3]["patch_idx"]], [41, 19])
        self.assertEqual(rewritten[4]["region_idx"], 12)
        self.assertEqual(rewritten[0], actions[0])
        self.assertEqual(rewritten[1], actions[1])
        self.assertEqual(rewritten[5], actions[5])


class ReplayTraceVariantTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "trace": [
                {"type": "THINK"},
                {"type": "PATCH", "patch_idx": 10},
                {"type": "PATCH", "patch_idx": 11},
                {"type": "REGION", "region_idx": 3},
                {"type": "GLOBAL"},
                {"type": "THINK"},
                {"type": "STOP"},
            ],
            "decisions": [
                {"selected": "THINK", "actions": [{"type": "THINK"}]},
                {
                    "selected": "PATCH_SEQ",
                    "actions": [{"type": "PATCH", "patch_idx": 10}, {"type": "PATCH", "patch_idx": 11}],
                },
                {"selected": "REGION", "actions": [{"type": "REGION", "region_idx": 3}]},
                {"selected": "GLOBAL", "actions": [{"type": "GLOBAL"}]},
                {"selected": "THINK", "actions": [{"type": "THINK"}]},
            ],
        }

    def build(self, variant, seed=7):
        return build_replay_trace(
            self.row,
            trace_variant=variant,
            visual_or_region_min_improvement=0.0,
            think_min_improvement=0.0,
            max_decision_blocks_per_example=99,
            max_primitive_actions_per_example=99,
            rng=random.Random(seed),
        )

    def test_no_visual_keeps_only_reasoning_and_terminal_stop(self):
        trace, metrics = self.build("no_visual")

        self.assertEqual([action["type"] for action in trace], ["THINK", "THINK", "STOP"])
        self.assertEqual(metrics["removed_visual_actions"], 4)

    def test_no_reasoning_has_an_empty_replay_trace(self):
        trace, metrics = self.build("no_reasoning")

        self.assertEqual(trace, [])
        self.assertTrue(metrics["no_reasoning"])

    def test_shuffle_is_deterministic_and_keeps_compound_blocks_and_stop(self):
        trace, _ = self.build("shuffled", seed=19)
        blocks = build_replay_blocks(
            self.row,
            replay_trace=trace,
            trace_variant="shuffled",
            rng=random.Random(19),
        )

        self.assertEqual(flatten_replay_blocks(blocks), trace)
        self.assertEqual(blocks[-1]["label"], "STOP")
        patch_blocks = [block for block in blocks if block["label"] == "PATCH_SEQ"]
        self.assertEqual(len(patch_blocks), 1)
        self.assertEqual(len(patch_blocks[0]["actions"]), 2)


class StepEntropyTests(unittest.TestCase):
    def test_top_k_entropy_is_renormalized_and_reports_retained_mass(self):
        logits = torch.tensor([[[4.0, 3.0, 1.0, -2.0]]])

        class FakeBackbone:
            def __call__(self, **kwargs):
                del kwargs
                return SimpleNamespace(logits=logits)

        class FakeModel:
            backbone = FakeBackbone()

            @staticmethod
            def _entropy_from_logits(values):
                probabilities = torch.softmax(values.float(), dim=-1)
                return float((-(probabilities * probabilities.log()).sum()).item())

        state = {"inputs_embeds": torch.zeros(1, 2, 3), "attention_mask": torch.ones(1, 2)}
        result = next_token_entropy_from_state(FakeModel(), state, top_k=2)

        self.assertEqual(result["top_k"], 2)
        self.assertEqual(result["vocab_size"], 4)
        self.assertGreater(result["retained_probability_mass"], 0.9)
        self.assertLessEqual(result["entropy"], float(torch.log(torch.tensor(2.0))))


if __name__ == "__main__":
    unittest.main()
