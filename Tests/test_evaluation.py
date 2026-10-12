"""Exercise campaign proof guards with synthetic chapter outcomes, not Celeste."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_workers import FakeGame, config
from protocol import ControlPaused
from plastic import PlasticPolicy
import evaluate


class EvaluationTests(unittest.TestCase):
    def exercise(self, interrupt):
        class Game(FakeGame):
            def __init__(self):
                super().__init__()
                self.interrupt = interrupt
                self.practice = []

            def practice_mode(self, enabled):
                self.practice.append(enabled)

            def wait_until_on(self, stop):
                pass

            def step(self, action, frames):
                if self.interrupt:
                    self.interrupt = False
                    raise ControlPaused("Synthetic manual interruption")
                return super().step(action, frames)

        game = Game()

        class Worker:
            def __init__(self, *args):
                self.closed = False

            def launch(self):
                return game

            def report(self, **fields):
                pass

            def shutdown(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "training/latest.npz"
            policy = PlasticPolicy()
            policy.save(
                checkpoint,
                policy.graph_hash,
                dict(chapters=list(range(8)), iteration=1),
            )
            before = checkpoint.read_bytes()
            with (
                patch("evaluate.ROOT", root),
                patch("evaluate.Worker", Worker),
                patch("evaluate.validate_game", return_value=root / "game"),
                patch("evaluate.repair_original"),
                patch("evaluate.build", return_value=root / "mod.zip"),
                patch("evaluate.movement_check"),
                patch("evaluate.emit"),
            ):
                evaluate.run(dict(config(), game=str(root / "game")))
            result = json.loads((root / "training/evaluation.json").read_text())
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(game.practice, [False])
            self.assertFalse(result["weights_updated"])
            self.assertEqual(result["teacher_actions"], 0)
            self.assertEqual(result["practice_restores"], 0)
            self.assertEqual(result["completed_chapters"], list(range(8)))
            self.assertEqual(len(result["checkpoint_sha256"]), 64)
            return result

    def test_fixed_policy_synthetic_campaign_records_all_eight_native_events(self):
        self.assertTrue(self.exercise(False)["full_main_story_clear"])

    def test_manual_interruption_invalidates_the_synthetic_full_clear(self):
        result = self.exercise(True)
        self.assertEqual(result["manual_interruptions"], 1)
        self.assertFalse(result["full_main_story_clear"])
