import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from experience import SuccessMemory
from plastic import PlasticPolicy
from rehearse import run


class RehearsalTests(unittest.TestCase):
    def test_replay_changes_candidate_keeps_source_and_counts_zero_new_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training/latest.npz"
            policy = PlasticPolicy()
            policy.save(source, policy.graph_hash, dict(steps=512, iteration=2))
            original = source.read_bytes()
            memory = SuccessMemory(root / "training/success-memory", policy.graph_hash)
            records = [
                dict(
                    observation=np.linspace(-1, 1, 156),
                    action=np.array([2, 1, 1, 0, 0, 0, 4]),
                    reset=i == 0,
                    frames=12,
                )
                for i in range(8)
            ]
            self.assertTrue(memory.add(records, 0, "a", "b"))
            with patch("rehearse.watch_stop"):
                output = run(root, "cpu", max_updates=1)
            self.assertEqual(source.read_bytes(), original)
            candidate = PlasticPolicy()
            metadata = candidate.load(output, candidate.graph_hash)
            self.assertEqual(metadata["steps"], 512)
            self.assertEqual(metadata["offline_replay_updates"], 1)
            self.assertGreater(np.max(np.abs(candidate.actor - policy.actor)), 0)
            self.assertGreater(np.max(np.abs(candidate.theta - policy.theta)), 0)
            report = json.loads((root / "training/offline/report.json").read_text())
            self.assertEqual(report["fresh_game_frames"], 0)
            self.assertFalse(report["source_modified"])
            self.assertEqual(output, root / "training/offline/latest.npz")

    def test_no_successes_reports_the_requirement_without_changing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training/latest.npz"
            policy = PlasticPolicy()
            policy.save(source, policy.graph_hash, {})
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "No verified successful routes"):
                run(root, "cpu", max_updates=1)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root / "training/offline/latest.npz").exists())


if __name__ == "__main__":
    unittest.main()
