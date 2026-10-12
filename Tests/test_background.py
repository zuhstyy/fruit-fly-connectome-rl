from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from runtime import RunLock
from verify import assess, main


class BackgroundTests(unittest.TestCase):
    def test_exclusive_lock_blocks_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            with RunLock(d):
                with self.assertRaises(RuntimeError):
                    with RunLock(d):
                        pass
            with RunLock(d):
                pass

    def test_verify_does_not_call_changed_weights_improvement(self):
        rows = [
            dict(
                run_id="x",
                brain_updates=i,
                learned_edges=3326,
                rooms_this_run=2,
                completed_chapters=[],
                worker_rewards=[dict(best_route_rank=1)],
            )
            for i in range(3)
        ]
        self.assertIn("No demonstrated", assess(rows)["verdict"])
        rows[-1]["worker_rewards"][0]["best_route_rank"] = 2
        self.assertIn("Observed forward", assess(rows)["verdict"])
        rows[-1]["completed_chapters"] = [0]
        self.assertIn("game-reported chapter", assess(rows)["verdict"])

    def test_report_handles_empty_and_partial_log_without_touching_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            saved = root / "training"
            saved.mkdir()
            (saved / "latest.npz").write_bytes(b"do not modify")
            (saved / "progress.jsonl").write_text('{"run_id":"x"}\n{"unfinished":')
            result = main(root)
            self.assertIn("Not enough evidence", result["verdict"])
            self.assertEqual((saved / "latest.npz").read_bytes(), b"do not modify")
            self.assertTrue((saved / "verification.html").exists())
