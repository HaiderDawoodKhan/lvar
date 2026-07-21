import unittest

from lvar_scripts.infer_lvar_m3cot import apply_max_controller_steps_override


class InferenceControllerStepOverrideTests(unittest.TestCase):
    def test_override_sets_rollout_and_embedding_capacity_without_mutating_input(self):
        original = {"max_steps": 8, "controller_max_steps": 16, "other": True}

        updated = apply_max_controller_steps_override(original, 33)

        self.assertEqual(updated["max_steps"], 33)
        self.assertEqual(updated["controller_max_steps"], 33)
        self.assertEqual(original["max_steps"], 8)

    def test_override_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            apply_max_controller_steps_override({}, 0)


if __name__ == "__main__":
    unittest.main()
