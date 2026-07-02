import random
import unittest

from lvar_scripts.infer_lvar_m3cot_rollouts import (
    answer_key,
    build_variant_row,
    select_variant_rollouts,
)


class RolloutAggregationTests(unittest.TestCase):
    def setUp(self):
        self.example = {"id": "ex-1", "gold_answer": "A", "answer": "A"}
        self.rollouts = [
            {
                "rollout_idx": 0,
                "answer_key": "A",
                "correct": True,
                "token_entropy_mean": 0.2,
                "answer_option_entropy": {"entropy": 0.3},
                "hidden_step_entropy_mean": 0.4,
                "controller_entropy_mean": 0.5,
            },
            {
                "rollout_idx": 1,
                "answer_key": "B",
                "correct": False,
                "token_entropy_mean": 1.2,
                "answer_option_entropy": {"entropy": 1.3},
                "hidden_step_entropy_mean": 1.4,
                "controller_entropy_mean": 1.5,
            },
            {
                "rollout_idx": 2,
                "answer_key": "A",
                "correct": True,
                "token_entropy_mean": 0.4,
                "answer_option_entropy": {"entropy": 0.7},
                "hidden_step_entropy_mean": 0.6,
                "controller_entropy_mean": 0.8,
            },
        ]

    def test_answer_key_prefers_choice_candidates(self):
        self.assertEqual(answer_key("The answer is B", "<answer>A</answer>"), "B")

    def test_best_of_n_selects_most_common_answer_rollouts(self):
        correct, selected, extra = select_variant_rollouts(
            self.rollouts,
            variant="best_of_n",
            rng=random.Random(0),
        )
        row = build_variant_row(self.example, "best_of_n", correct, selected, extra)

        self.assertTrue(correct)
        self.assertEqual(extra["selected_answer_key"], "A")
        self.assertEqual(row["selected_rollout_indices"], [0, 2])
        self.assertAlmostEqual(row["decoded_token_entropy_mean"], 0.3)
        self.assertAlmostEqual(row["answer_option_entropy_mean"], 0.5)
        self.assertAlmostEqual(row["hidden_step_entropy_mean"], 0.5)

    def test_oracle_uses_correct_rollouts_when_present(self):
        correct, selected, extra = select_variant_rollouts(
            self.rollouts,
            variant="oracle",
            rng=random.Random(0),
        )

        self.assertTrue(correct)
        self.assertTrue(extra["oracle_found_correct"])
        self.assertEqual([rollout["rollout_idx"] for rollout in selected], [0, 2])

    def test_random_selects_one_rollout(self):
        correct, selected, extra = select_variant_rollouts(
            self.rollouts,
            variant="random",
            rng=random.Random(1),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(extra["selected_rollout_idx"], selected[0]["rollout_idx"])
        self.assertEqual(correct, selected[0]["correct"])


if __name__ == "__main__":
    unittest.main()
