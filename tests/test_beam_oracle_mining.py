import json
import tempfile
import unittest
from pathlib import Path

from lvar.beam_oracle_mining import BeamOracleTraceMiner, BeamTrajectory
from lvar.controller_sft import load_mined_trace_rows
from lvar_scripts.eval_mined_traces_m3cot import select_best_beam_trajectory
from test_model import build_model


class BeamOracleMiningTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        prepared = self.model.prepare_inputs("image", "question", add_answer_instruction=False)
        projected = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        self.prepared = prepared
        self.bank = self.model.build_visual_bank(projected)

    def test_stage_search_is_patch_think_only_and_caps_bucket(self):
        miner = BeamOracleTraceMiner(
            self.model,
            beam_width=3,
            max_actions_per_stage=4,
            patch_top_k=2,
            ce_improvement_threshold=0.01,
        )
        state = self.model.build_initial_state(self.prepared)
        base_length = state["inputs_embeds"].size(1)
        miner._attended_patch_indices = lambda trajectory, bank, image_positions: [0, 1]
        miner._score_components = lambda candidate_state, rationale, answer: (
            10.0 - 0.02 * (candidate_state["inputs_embeds"].size(1) - base_length),
            10.0 - 0.02 * (candidate_state["inputs_embeds"].size(1) - base_length),
            10.0 - 0.02 * (candidate_state["inputs_embeds"].size(1) - base_length),
        )
        initial = BeamTrajectory(state=state, score=10.0, ce_rationale=10.0, ce_answer=10.0)
        beams = miner._search_stage([initial], 0, "later rationale", "answer", self.bank, [1, 2, 3, 4])

        self.assertLessEqual(len(beams), 3)
        self.assertGreaterEqual(len(beams), 1)
        for beam in beams:
            decision = beam.decisions[-1]
            self.assertLessEqual(len(decision["actions"]), 4)
            self.assertTrue(all(action["type"] in {"PATCH", "THINK"} for action in decision["actions"]))
            for insertion in decision["insertions"]:
                self.assertEqual(insertion["attended_patch_indices"], [0, 1])
                self.assertGreaterEqual(insertion["improvement"], 0.01)

    def test_phase3_loader_expands_beam_trajectories(self):
        row = {
            "example_id": "example-1",
            "trace": [{"type": "STOP"}],
            "decisions": [],
            "beam_trajectories": [
                {"rank": 1, "weighted_ce": 1.0, "trace": [{"type": "PATCH", "patch_idx": 0}, {"type": "STOP"}], "decisions": [{"actions": [{"type": "PATCH", "patch_idx": 0}]}]},
                {"rank": 2, "weighted_ce": 1.1, "trace": [{"type": "THINK"}, {"type": "STOP"}], "decisions": [{"actions": [{"type": "THINK"}]}]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows = load_mined_trace_rows(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["beam_rank"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["decisions"][0]["actions"][0]["type"], "PATCH")

    def test_replay_selects_rank_one_beam_even_when_top_level_trace_differs(self):
        row = {
            "trace": [{"type": "THINK"}, {"type": "STOP"}],
            "beam_trajectories": [
                {"rank": 2, "weighted_ce": 0.1, "trace": [{"type": "THINK"}, {"type": "STOP"}], "decisions": []},
                {"rank": 1, "weighted_ce": 0.2, "trace": [{"type": "PATCH", "patch_idx": 3}, {"type": "STOP"}], "decisions": []},
            ],
        }
        selected = select_best_beam_trajectory(row)

        self.assertEqual(selected["selected_beam_rank"], 1)
        self.assertEqual(selected["trace"][0], {"type": "PATCH", "patch_idx": 3})


if __name__ == "__main__":
    unittest.main()
