import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from lvar.trace_attention_boost import (
    TraceAttentionBoostRuntime,
    TraceBoostConfig,
    apply_trace_attention_boost,
    get_boost_layers,
)
from lvar.utils import apply_trace_boost_overrides, boosted_output_path


class FakeAttention(torch.nn.Module):
    def forward(self, scores):
        return F.softmax(scores, dim=-1)


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeBackbone(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer() for _ in range(num_layers)])


class TraceAttentionBoostTests(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        config = TraceBoostConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.target, "trace_visual")
        self.assertEqual(config.layer_mode, "latter_half")
        self.assertEqual(config.alpha, 0.2)
        with self.assertRaises(ValueError):
            TraceBoostConfig(target="image")
        with self.assertRaises(ValueError):
            TraceBoostConfig(layer_mode="first_half")
        with self.assertRaises(ValueError):
            TraceBoostConfig(alpha=-0.1)

    def test_layer_selection(self):
        self.assertEqual(get_boost_layers(5, "all"), {0, 1, 2, 3, 4})
        self.assertEqual(get_boost_layers(5, "latter_half"), {2, 3, 4})
        with self.assertRaises(ValueError):
            get_boost_layers(4, "middle")

    def test_pre_softmax_boost_is_exact_and_preserves_masks(self):
        scores = torch.tensor([[[[-2.0, 3.0, float("-inf"), 4.0]]]])
        boosted = apply_trace_attention_boost(scores, [0, 1, 2], alpha=0.2)
        expected = torch.tensor([[[[-1.6, 3.6, float("-inf"), 4.0]]]])
        self.assertTrue(torch.equal(torch.isneginf(boosted), torch.isneginf(expected)))
        self.assertTrue(torch.allclose(boosted[torch.isfinite(boosted)], expected[torch.isfinite(expected)]))
        self.assertIs(apply_trace_attention_boost(scores, [], alpha=0.2), scores)

    def test_boost_changes_only_answer_stage_query_rows(self):
        scores = torch.tensor(
            [[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]]]
        )
        boosted = apply_trace_attention_boost(
            scores,
            boost_positions=[1],
            alpha=0.5,
            query_start=2,
        )

        self.assertTrue(torch.equal(boosted[..., :2, :], scores[..., :2, :]))
        self.assertTrue(torch.equal(boosted[..., 2:, 0], scores[..., 2:, 0]))
        self.assertTrue(torch.equal(boosted[..., 2:, 2], scores[..., 2:, 2]))
        self.assertTrue(
            torch.allclose(boosted[..., 2:, 1], scores[..., 2:, 1] + 0.5 * scores[..., 2:, 1].abs())
        )

    def test_default_boost_scope_is_only_final_query_row(self):
        scores = torch.tensor([[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]])
        boosted = apply_trace_attention_boost(scores, [0], alpha=0.25)

        self.assertTrue(torch.equal(boosted[..., :-1, :], scores[..., :-1, :]))
        self.assertAlmostEqual(float(boosted[..., -1, 0]), 5.0)

    def test_runtime_only_boosts_selected_answer_layers_and_logs_masses(self):
        config = TraceBoostConfig(
            enabled=True,
            target="trace_visual",
            layer_mode="latter_half",
            alpha=0.5,
        )
        runtime = TraceAttentionBoostRuntime(config)
        backbone = FakeBackbone(num_layers=4)
        runtime.install(backbone)
        scores = torch.tensor([[[[0.0, 1.0, 2.0]]]])

        unboosted = backbone.model.layers[2].self_attn(scores)
        expected_unboosted = torch.softmax(scores, dim=-1)
        self.assertTrue(torch.allclose(unboosted, expected_unboosted))

        with runtime.answer_decode(
            trace_all_positions=[1, 2],
            trace_visual_positions=[1],
            answer_query_start=0,
        ):
            early = backbone.model.layers[0].self_attn(scores)
            boosted = backbone.model.layers[2].self_attn(scores)

        expected_scores = torch.tensor([[[[0.0, 1.5, 2.0]]]])
        expected_boosted = torch.softmax(expected_scores, dim=-1)
        self.assertTrue(torch.allclose(early, expected_unboosted))
        self.assertTrue(torch.allclose(boosted, expected_boosted))

        summary = runtime.attention_mass_summary()
        self.assertAlmostEqual(
            summary["trace_attention_mass"],
            float(expected_boosted[..., [1, 2]].sum()),
        )
        self.assertAlmostEqual(
            summary["visual_trace_attention_mass"],
            float(expected_boosted[..., 1].sum()),
        )
        self.assertAlmostEqual(
            summary["think_attention_mass"],
            float(expected_boosted[..., 2].sum()),
        )
        self.assertEqual(summary["trace_boost_softmax_hits"], 1)

    def test_known_output_directories_are_rewritten_once(self):
        self.assertEqual(
            boosted_output_path(
                "outputs/inference/current_lvar_model/predictions.jsonl",
                enabled=True,
            ),
            "outputs/inference/current_lvar_model_boosted/predictions.jsonl",
        )
        self.assertEqual(
            boosted_output_path(
                "outputs/inference/test_oracle_boosted/predictions.jsonl",
                enabled=True,
            ),
            "outputs/inference/test_oracle_boosted/predictions.jsonl",
        )
        self.assertEqual(
            boosted_output_path("outputs/custom/predictions.jsonl", enabled=True),
            "outputs/custom/predictions.jsonl",
        )

    def test_cli_values_override_nested_trace_boost_config(self):
        args = Namespace(
            trace_boost=True,
            trace_boost_target="trace_all",
            trace_boost_layer_mode="all",
            trace_boost_alpha=0.3,
        )
        updated = apply_trace_boost_overrides(
            {"trace_boost": {"enabled": False, "target": "trace_visual"}},
            args,
        )
        self.assertEqual(
            updated["trace_boost"],
            {
                "enabled": True,
                "target": "trace_all",
                "layer_mode": "all",
                "alpha": 0.3,
            },
        )


if __name__ == "__main__":
    unittest.main()
