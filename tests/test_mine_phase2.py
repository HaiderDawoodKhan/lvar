import json
import tempfile
import unittest
from pathlib import Path

from lvar_scripts.mine_phase2 import iter_dataset_indices, read_completed_example_ids


class MinePhase2ScriptTests(unittest.TestCase):
    def test_iter_dataset_indices_can_start_from_end(self):
        self.assertEqual(iter_dataset_indices(4, start_from_end=False), [0, 1, 2, 3])
        self.assertEqual(iter_dataset_indices(4, start_from_end=True), [3, 2, 1, 0])

    def test_read_completed_example_ids_from_existing_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"example_id": "a"}) + "\n")
                handle.write("\n")
                handle.write(json.dumps({"example_id": "b"}) + "\n")

            self.assertEqual(read_completed_example_ids(path), {"a", "b"})

    def test_read_completed_example_ids_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.jsonl"

            self.assertEqual(read_completed_example_ids(path), set())


if __name__ == "__main__":
    unittest.main()
