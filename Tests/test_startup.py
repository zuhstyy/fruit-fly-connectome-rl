from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from workers import Worker, exit_description


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def is_set(self):
        return False

    def wait(self, seconds):
        self.now += seconds
        return False


class StartupTests(unittest.TestCase):
    def make_worker(self, directory, code=0):
        root = Path(directory)
        clock = Clock()
        worker = Worker(0, root / "game", root / "mod.zip", clock, lambda m: None)
        worker.dest = root / "worker copy"
        worker.dest.mkdir()
        worker.port = 32123
        worker.process = Mock(pid=1234)
        worker.process.poll.return_value = code
        worker.process.wait.return_value = code
        saves = worker.dest / "training-data/Saves"
        saves.mkdir(parents=True)
        client = Mock()
        client.request.return_value = dict(
            save_path=str(saves), worker_path=str(worker.dest), worker_pid=5678
        )
        return worker, saves, client, clock

    def test_exited_launcher_with_delayed_child_bridge_is_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            w, saves, child, clock = self.make_worker(t)
            with (
                patch("workers.time.monotonic", clock.monotonic),
                patch(
                    "workers.Client",
                    side_effect=[
                        ConnectionRefusedError(),
                        ConnectionRefusedError(),
                        child,
                    ],
                ),
            ):
                self.assertIs(w._connect(saves), child)
            self.assertTrue(w.launch_info["bridge_connected"])
            self.assertEqual(w.launch_info["launcher_exit_code"], 0)
            self.assertEqual(w.launch_info["bridge_pid"], 5678)

    def test_real_exit_reports_numeric_code_even_when_logs_are_empty(self):
        with tempfile.TemporaryDirectory() as t:
            w, saves, _, clock = self.make_worker(t, -1073741701)
            (w.dest / "flybrain-process.log").touch()
            with (
                patch("workers.time.monotonic", clock.monotonic),
                patch("workers.Client", side_effect=ConnectionRefusedError()),
            ):
                with self.assertRaises(RuntimeError) as error:
                    w._connect(saves, exit_grace=0.5)
            self.assertIn("0xC000007B", str(error.exception))
            self.assertIn("(empty)", str(error.exception))
            self.assertIn("flybrain-launch.json", str(error.exception))

    def test_normal_game_bridge_is_rejected_even_if_save_path_matches(self):
        with tempfile.TemporaryDirectory() as t:
            w, saves, client, _ = self.make_worker(t)
            client.request.return_value["worker_path"] = str(w.game)
            with patch("workers.Client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "not the isolated worker"):
                    w._connect(saves)
            client.close.assert_called_once_with(False)
            self.assertIsNone(w.client)

    def test_stop_sends_quit_to_child_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as t:
            w, _, child, _ = self.make_worker(t)
            parent = w.process
            w.client = child
            w.shutdown()
            child.close.assert_called_once_with(quit_game=True)
            parent.terminate.assert_not_called()
            parent.kill.assert_not_called()

    def test_launch_records_command_and_sets_worker_only_steam_and_trace_env(self):
        with tempfile.TemporaryDirectory() as t:
            w, saves, client, _ = self.make_worker(t)
            process = w.process
            with (
                patch("workers.prepare_copy", return_value=saves),
                patch("workers.subprocess.Popen", return_value=process) as popen,
                patch.object(w, "_connect", return_value=client),
            ):
                self.assertIs(w.launch(), client)
            args, kwargs = popen.call_args
            self.assertEqual(args[0], [str(w.dest / "Celeste.exe")])
            self.assertEqual(kwargs["cwd"], w.dest)
            self.assertEqual(kwargs["env"]["SteamAppId"], "504230")
            self.assertEqual(kwargs["env"]["EVEREST_SAVEPATH"], str(saves.parent))
            self.assertEqual(kwargs["env"]["COREHOST_TRACE"], "1")
            info = (w.dest / "flybrain-launch.json").read_text()
            self.assertNotIn(w.token, info)
            self.assertEqual(json.loads(info)["launcher_pid"], 1234)
            w.handle.close()

    def test_exit_code_format_preserves_windows_status(self):
        self.assertEqual(exit_description(None), "still running")
        self.assertEqual(exit_description(0), "0 (0x00000000)")
        self.assertIn("0xC000007B", exit_description(-1073741701))


if __name__ == "__main__":
    unittest.main()
