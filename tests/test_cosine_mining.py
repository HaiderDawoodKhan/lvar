import json
import tempfile
import unittest
from pathlib import Path

import torch

from lvar.cosine_mining import CosineSimilarityTraceMiner, select_top_k_patches, summarize_cosine_trace_rows
from lvar_scripts.mine_cosine_similarity import pending_dataset_indices, read_jsonl_rows
from tests.test_model import build_model


class CosineSimilarityMiningTests(unittest.TestCase):
    def test_top_k_patch_ranking_is_descending_and_distinct(self):
        patches = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )

        indices, scores = select_top_k_patches(torch.tensor([1.0, 0.0]), patches, top_k=3)

        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(len(set(indices)), 3)
        self.assertGreaterEqual(scores[0], scores[1])
        self.assertGreaterEqual(scores[1], scores[2])

    def test_top_k_rejects_a_patch_bank_smaller_than_k(self):
        with self.assertRaisesRegex(ValueError, "contains 2 patches"):
            select_top_k_patches(torch.tensor([1.0, 0.0]), torch.eye(2), top_k=3)

    def test_miner_builds_cumulative_patch_think_blocks_and_terminal_stop(self):
        model = build_model(think_append_hidden=True)
        miner = CosineSimilarityTraceMiner(model, top_k=3, max_steps=8, image_size=None, mining_mode="sequential")
        row = miner.mine_example(
            {
                "id": "ex-1",
                "image": "image",
                "question": "question",
                "steps": ["Inspect the object.", "Choose the matching option."],
                "solution": "Inspect the object. Choose the matching option.\n<answer>D</answer>",
            }
        )

        self.assertEqual(row["mining_method"], "step_hidden_patch_cosine")
        self.assertEqual(row["top_k"], 3)
        self.assertEqual(row["answer"], "D")
        self.assertEqual(len(row["steps"]), 2)
        self.assertEqual(len(row["decisions"]), 2)
        self.assertEqual(
            [action["type"] for action in row["decisions"][0]["actions"]],
            ["PATCH", "PATCH", "PATCH", "THINK"],
        )
        self.assertEqual([action["type"] for action in row["trace"][-2:]], ["THINK", "STOP"])

        for decision in row["decisions"]:
            self.assertEqual(len(decision["patch_indices"]), 3)
            self.assertEqual(len(set(decision["patch_indices"])), 3)
            self.assertEqual(len(decision["cosine_similarities"]), 3)
            self.assertEqual(
                decision["patch_indices"],
                [action["patch_idx"] for action in decision["actions"][:3]],
            )
            self.assertGreater(decision["sequence_length_after_step"], decision["sequence_length_before_step"])
            self.assertGreater(decision["sequence_length_after_actions"], decision["sequence_length_after_step"])

        self.assertGreater(
            row["decisions"][1]["sequence_length_before_step"],
            row["decisions"][0]["sequence_length_before_step"],
        )
        self.assertNotIn("D", " ".join(row["steps"]))
        self.assertEqual(miner.get_summary()["num_patch_actions"], 6)
        self.assertEqual(miner.get_summary()["num_think_actions"], 2)

    def test_single_pass_is_default_and_does_not_insert_patches_or_think(self):
        model = build_model(think_append_hidden=True)
        forward_calls = {"count": 0}
        original_forward = model.backbone.forward

        def counting_forward(*args, **kwargs):
            forward_calls["count"] += 1
            return original_forward(*args, **kwargs)

        model.backbone.forward = counting_forward
        miner = CosineSimilarityTraceMiner(model, top_k=3, max_steps=8, image_size=None)
        row = miner.mine_example(
            {
                "id": "ex-1",
                "image": "image",
                "question": "question",
                "steps": ["Inspect the object.", "Choose the matching option."],
                "solution": "Inspect the object. Choose the matching option.\n<answer>D</answer>",
            }
        )

        self.assertEqual(miner.mining_mode, "single_pass")
        self.assertEqual(forward_calls["count"], 1)
        self.assertEqual(
            [action["type"] for action in row["trace"]],
            ["PATCH", "PATCH", "PATCH", "PATCH", "PATCH", "PATCH", "STOP"],
        )
        self.assertEqual(row["mining_mode"], "single_pass")
        self.assertEqual(row["decisions"][0]["sequence_length_after_actions"], row["decisions"][0]["sequence_length_after_step"])

    def test_pending_indices_support_resume_and_reverse_order(self):
        dataset = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        indices, pending = pending_dataset_indices(dataset, {"b"}, start_from_end=True)

        self.assertEqual(indices, [2, 1, 0])
        self.assertEqual(pending, [2, 0])

    def test_summary_aggregates_all_rows_after_resume(self):
        rows = [
            {
                "decisions": [
                    {
                        "cosine_similarities": [0.9, 0.8, 0.7],
                        "actions": [
                            {"type": "PATCH", "patch_idx": 0},
                            {"type": "PATCH", "patch_idx": 1},
                            {"type": "PATCH", "patch_idx": 2},
                            {"type": "THINK"},
                        ],
                    }
                ],
                "trace": [
                    {"type": "PATCH", "patch_idx": 0},
                    {"type": "PATCH", "patch_idx": 1},
                    {"type": "PATCH", "patch_idx": 2},
                    {"type": "THINK"},
                    {"type": "STOP"},
                ],
            },
            {"decisions": [], "trace": [{"type": "STOP"}]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = summarize_cosine_trace_rows(read_jsonl_rows(path), top_k=3, max_steps=8)

        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["num_decisions"], 1)
        self.assertEqual(summary["num_patch_actions"], 3)
        self.assertEqual(summary["num_think_actions"], 1)
        self.assertAlmostEqual(summary["mean_selected_cosine_similarity"], 0.8)


if __name__ == "__main__":
    unittest.main()
