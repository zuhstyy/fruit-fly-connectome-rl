from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from performance import throughput_summary
from protocol import Client, BridgeError
from search_teacher import SearchTeacher
from test_protocol import server
from test_practice import PracticeGame, room_state

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PerformanceTests(unittest.TestCase):
    def test_experience_time_is_simulation_frames_not_decisions(self):
        report = throughput_summary(60 * 86400, 86400, {"learning": 1, "collection": 9})
        self.assertEqual(report["aggregate_game_days"], 1)
        self.assertEqual(report["game_speed"], 1)
        self.assertAlmostEqual(report["projected_game_years_per_wall_day"], 1 / 365)
        self.assertEqual(report["largest_phase"], "collection")

    def test_static_geometry_is_reused_only_with_same_room_and_revision(self):
        def reply(request):
            number = request["id"]
            return dict(
                type="state",
                id=number,
                chapter=0,
                room="a",
                ready=True,
                world_revision=1 if number <= 2 else 2,
                map={"tiles": "..."} if number == 1 else None,
                portals=[{"room": "b"}] if number == 1 else None,
                route_graph=[{"name": "a"}] if number == 1 else None,
            )

        with server(reply) as port:
            client = Client(port, "secret")
            first = client.request("status")
            cached = client.request("status")
            self.assertEqual(first["map"], cached["map"])
            self.assertEqual(first["portals"], cached["portals"])
            with self.assertRaisesRegex(BridgeError, "Missing room geometry"):
                client.request("status")
            client.close(False)

    def test_room_transition_cannot_reuse_the_old_rooms_map(self):
        def reply(request):
            return dict(
                type="state",
                id=request["id"],
                chapter=0,
                room="a" if request["id"] == 1 else "b",
                world_revision=1,
                ready=True,
                map={"tiles": "..."} if request["id"] == 1 else None,
            )

        with server(reply) as port:
            client = Client(port, "secret")
            client.request("status")
            with self.assertRaises(BridgeError):
                client.request("status")
            client.close(False)

    def test_missing_config_has_an_actionable_message_and_spaces_stay_one_argument(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="folder with spaces ") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Save settings"):
                launcher.command_for(root, "python", "benchmark")
            (root / "config.json").write_text("{}")
            command = launcher.command_for(root, "python with spaces.exe", "benchmark")
            self.assertEqual(command[0], "python with spaces.exe")
            self.assertEqual(command[-1], str(root / "config.json"))
            self.assertIn("gpu-check", [mode for _, mode in launcher.TOOLS])

    def test_search_budget_restores_root_and_stops_testing_more_actions(self):
        game = PracticeGame()
        original = game.value["x"]
        teacher = SearchTeacher()
        with patch("search_teacher.time.monotonic", return_value=10):
            action = teacher.choose(
                game, room_state(), np.random.default_rng(17), deadline=5
            )
        self.assertEqual(teacher.trials, 1)
        self.assertEqual(game.value["x"], original)
        self.assertEqual(len(action), 7)


if __name__ == "__main__":
    unittest.main()
