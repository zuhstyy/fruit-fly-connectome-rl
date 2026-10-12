"""End-to-end Python coordinator check using 64 synthetic bridge clients.
This exercises scheduling, rollout/update, checkpoint, telemetry and shutdown;
it does not claim to run Celeste or validate Windows rendering.
"""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
import train
from plastic import PlasticPolicy
from test_learning import state
from test_workers import FakeGame, config


class RunIntegrationTests(unittest.TestCase):
    def test_64_real_worker_slots_collect_update_save_and_shutdown(self):
        constructed = []

        class Game(FakeGame):
            def step(self, action, frames):
                self.frame += frames
                return state(frame=self.frame, action_frames=frames)

        class Worker:
            def __init__(self, index, game, mod, stop, log, ticks):
                self.index, self.stop, self.log = index, stop, log
                self.client = Game()
                self.telemetry = None
                self.restarts = 0
                self.closed = False
                self.launch_info = {"bridge_pid": 1000 + index}
                constructed.append(self)

            def report(self, **fields):
                if self.telemetry:
                    self.telemetry.update(self.index, **fields)

            def launch(self):
                self.report(pid=1000 + self.index, process_created=1, phase="preflight")
                return self.client

            def shutdown(self):
                self.closed = True
                self.report(pid=None, phase="stopped")

        update = PlasticPolicy.update_rollouts

        def one_update(policy, batches, **settings):
            result = update(policy, batches, **settings)
            constructed[0].stop.set()
            return result

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = config()
            c.update(workers=64, game=str(root / "game"), background=True, rollout=128)
            with (
                patch("train.ROOT", root),
                patch("train.Worker", Worker),
                patch("train.validate_game", return_value=root / "game"),
                patch("train.build", return_value=root / "mod.zip"),
                patch("train.repair_original"),
                patch("train.emit"),
                patch(
                    "train.movement_check",
                    side_effect=lambda client, chapter, stop: client.start(chapter),
                ),
                patch(
                    "runtime.process_info", return_value={"alive": True, "created": 1}
                ),
                patch.object(PlasticPolicy, "update_rollouts", one_update),
            ):
                train.run(c)
            self.assertEqual(len(constructed), 64)
            self.assertTrue(all(w.closed for w in constructed))
            saved = json.loads((root / "training/status.json").read_text())
            self.assertEqual(saved["requested_workers"], 64)
            self.assertEqual(saved["live_game_processes"], 64)
            self.assertEqual(saved["ready_workers"], 64)
            self.assertEqual(saved["steps"], 2048)
            self.assertEqual(saved["brain_updates"], 1)
            self.assertEqual(len(saved["worker_rewards"]), 64)
            self.assertEqual(
                json.loads((root / "training/live.json").read_text())["phase"],
                "stopped",
            )
            p = PlasticPolicy()
            meta = p.load(root / "training/latest.npz", p.graph_hash)
            self.assertEqual(meta["steps"], 2048)
