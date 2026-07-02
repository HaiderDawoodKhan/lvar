import tempfile
import unittest
from pathlib import Path

import torch

from lvar.latent_depth import (
    BUCKET_IMAGE,
    BUCKET_LATENT,
    BUCKET_PROMPT,
    aggregate_attention_by_bucket,
    build_latent_depth_supervision,
    compute_hidden_step_metrics,
    load_fixed_think_rows,
)
from lvar.latent_depth_controller import LatentDepthController


class LatentDepthHelperTests(unittest.TestCase):
    def test_aggregate_attention_by_bucket(self):
        attention = torch.zeros(1, 2, 4, 4)
        attention[0, 0, 3, :] = torch.tensor([0.10, 0.20, 0.30, 0.40])
        attention[0, 1, 3, :] = torch.tensor([0.25, 0.25, 0.25, 0.25])
        labels = [BUCKET_IMAGE, BUCKET_PROMPT, BUCKET_PROMPT, BUCKET_LATENT]

        result = aggregate_attention_by_bucket([attention], labels=labels, query_pos=3)

        layer = result["per_layer"][0]
        self.assertAlmostEqual(layer[BUCKET_IMAGE]["mean"], 0.175)
        self.assertAlmostEqual(layer[BUCKET_PROMPT]["mean"], 0.5)
        self.assertAlmostEqual(layer[BUCKET_LATENT]["mean"], 0.0)
        self.assertEqual(layer[BUCKET_LATENT]["num_tokens"], 0)
        self.assertAlmostEqual(result["summary"][BUCKET_IMAGE]["max"], 0.25)

    def test_compute_hidden_step_metrics(self):
        metrics = compute_hidden_step_metrics(
            [
                torch.tensor([[3.0, 4.0]]),
                torch.tensor([[6.0, 8.0]]),
            ]
        )

        self.assertEqual(metrics[0]["step_idx"], 0)
        self.assertAlmostEqual(metrics[0]["hidden_norm"], 5.0)
        self.assertIsNone(metrics[0]["hidden_norm_delta"])
        self.assertAlmostEqual(metrics[1]["hidden_norm"], 10.0)
        self.assertAlmostEqual(metrics[1]["hidden_norm_delta"], 5.0)
        self.assertAlmostEqual(metrics[1]["hidden_delta_norm"], 5.0)

    def test_build_latent_depth_supervision_earliest_correct(self):
        rows = [
            {"example_id": "a", "latent_depth": 0, "correct": False},
            {"example_id": "a", "latent_depth": 1, "correct": True},
            {"example_id": "a", "latent_depth": 2, "correct": True},
            {"example_id": "b", "latent_depth": 0, "correct": False},
        ]

        supervision, summary = build_latent_depth_supervision(rows, max_depth=2)

        self.assertEqual(summary["num_skipped_no_correct"], 1)
        self.assertIn("b", summary["missing_depths"])
        self.assertEqual(summary["missing_depths"]["b"], [1, 2])
        self.assertEqual([(row["depth"], row["target_stop"]) for row in supervision], [(0, 0.0), (1, 1.0)])
        self.assertEqual(supervision[0]["correct_depths"], [1, 2])
        self.assertEqual(supervision[0]["earliest_correct_depth"], 1)

    def test_build_latent_depth_supervision_depth_zero(self):
        supervision, _ = build_latent_depth_supervision(
            [{"example_id": "a", "latent_depth": 0, "correct": True}],
            max_depth=2,
        )

        self.assertEqual(len(supervision), 1)
        self.assertEqual(supervision[0]["depth"], 0)
        self.assertEqual(supervision[0]["target_stop"], 1.0)

    def test_load_fixed_think_rows_infers_depth_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixed_think_steps_3" / "predictions.jsonl"
            path.parent.mkdir()
            path.write_text('{"example_id": "a", "correct": true}\n', encoding="utf-8")

            rows, summary = load_fixed_think_rows([path])

        self.assertEqual(summary["num_rows"], 1)
        self.assertEqual(rows[0]["latent_depth"], 3)


class LatentDepthControllerTests(unittest.TestCase):
    def test_forward_supports_variable_latents_and_padding(self):
        controller = LatentDepthController(
            input_hidden_size=4,
            controller_hidden_size=8,
            num_layers=1,
            num_heads=2,
            max_prompt_tokens=10,
            max_latent_steps=3,
        )
        visual = torch.randn(2, 4)
        prompt = torch.randn(2, 10, 4)
        latent = torch.randn(2, 3, 4)
        prompt_mask = torch.ones(2, 10, dtype=torch.bool)
        latent_mask = torch.tensor([[True, True, False], [True, False, False]])

        logits = controller(visual, prompt, latent, prompt_mask=prompt_mask, latent_mask=latent_mask)

        self.assertEqual(tuple(logits.shape), (2,))

    def test_forward_allows_zero_latents(self):
        controller = LatentDepthController(
            input_hidden_size=4,
            controller_hidden_size=8,
            num_layers=1,
            num_heads=2,
            max_prompt_tokens=10,
            max_latent_steps=3,
        )

        logits = controller(torch.randn(1, 4), torch.randn(1, 5, 4))

        self.assertEqual(tuple(logits.shape), (1,))


if __name__ == "__main__":
    unittest.main()
