from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from runtime import (
    LiveStatus,
    RunLock,
    set_visibility,
    desired_visibility,
    capacity_reason,
    read_json,
    change_windows,
)
from windows import apply_windows, GIB, owned_process
from scheduling import launch_workers, rollout_size


class MonitoringTests(unittest.TestCase):
    def test_show_is_explicit_and_idempotent_with_selected_override(self):
        with tempfile.TemporaryDirectory() as d:
            set_visibility(d, True)
            set_visibility(d, True)
            self.assertTrue(desired_visibility(d, 1))
            self.assertTrue(desired_visibility(d, 64))
            set_visibility(d, False, [2])
            self.assertFalse(desired_visibility(d, 2))
            self.assertTrue(desired_visibility(d, 1))
            set_visibility(d, False)
            self.assertFalse(desired_visibility(d, 64))
            set_visibility(d, True, [64])
            self.assertTrue(desired_visibility(d, 64))
            self.assertFalse(desired_visibility(d, 1))

    def test_show_restores_both_hidden_and_minimized_windows(self):
        api = Mock()
        api.enumerate.return_value = [
            dict(handle=11, visible=False, minimized=False),
            dict(handle=12, visible=True, minimized=True),
            dict(handle=13, visible=True, minimized=False),
        ]
        result = apply_windows(700, True, worker=1, api=api)
        self.assertEqual(result["windows"], 3)
        self.assertEqual(api.restore.call_count, 2)
        api.restore.assert_any_call(11, False)
        api.restore.assert_any_call(12, False)
        api.hide.assert_not_called()
        api.reset_mock()
        apply_windows(700, False, api=api)
        self.assertEqual(api.hide.call_count, 2)
        api.restore.assert_not_called()

    def test_monitor_counts_only_live_verified_processes_not_requested_slots(self):
        with tempfile.TemporaryDirectory() as d:
            live = LiveStatus(d, 64, "test")
            for i in range(3):
                live.update(i, pid=100 + i, process_created=1, phase="running")
            with (
                patch(
                    "runtime.process_info",
                    side_effect=lambda pid: dict(
                        alive=pid != 101, created=2 if pid == 102 else 1
                    ),
                ),
                patch("runtime.memory_status", return_value=None),
            ):
                view = live.snapshot()
            self.assertEqual(view["requested_workers"], 64)
            self.assertEqual(view["live_game_processes"], 1)
            self.assertEqual(view["ready_workers"], 1)
            self.assertEqual(len(view["workers"]), 64)

    def test_process_reuse_or_wrong_path_cannot_control_another_window(self):
        with tempfile.TemporaryDirectory() as d:
            row = dict(pid=1, process_created=200, worker_path=d)
            with patch(
                "windows.process_info",
                return_value=dict(
                    alive=True, created=201, path=str(Path(d) / "Celeste.exe")
                ),
            ):
                self.assertFalse(owned_process(row))
            with patch(
                "windows.process_info",
                return_value=dict(alive=True, created=200, path="/another/Celeste.exe"),
            ):
                self.assertFalse(owned_process(row))
            with patch(
                "windows.process_info",
                return_value=dict(
                    alive=True, created=200, path=str(Path(d) / "Celeste.exe")
                ),
            ):
                self.assertTrue(owned_process(row))

    def test_atomic_live_status_has_startup_rows_before_any_update(self):
        with tempfile.TemporaryDirectory() as d:
            live = LiveStatus(d, 64, "startup")
            live.start()
            self.assertEqual(
                len(read_json(Path(d) / "training/live.json")["workers"]), 64
            )
            live.close("error", "A concrete startup failure")
            self.assertIn(
                "concrete", read_json(Path(d) / "training/live.json")["message"]
            )

    def test_stale_status_cannot_restore_windows(self):
        with tempfile.TemporaryDirectory() as d:
            with (
                patch("runtime.is_running", return_value=False),
                patch("runtime.apply_windows") as apply,
            ):
                text = change_windows(d, True)
                self.assertIn("No active learner", text)
                apply.assert_not_called()

    def test_64_workers_schedule_64_distinct_game_startups(self):
        stop = threading.Event()
        calls = []
        lock = threading.Lock()
        workers = [
            SimpleNamespace(
                index=i, launch_info={}, report=Mock(), log=Mock(), shutdown=Mock()
            )
            for i in range(64)
        ]

        def start(w):
            with lock:
                calls.append(w.index)
            w.launch_info["bridge_pid"] = 1000 + w.index
            return SimpleNamespace(worker=w)

        ready, reason = launch_workers(
            workers, start, stop, 4, memory=lambda: None, inspect=lambda pid: None
        )
        self.assertEqual(len(ready), 64)
        self.assertEqual(set(calls), set(range(64)))
        self.assertEqual(reason, "")
        self.assertEqual(len({r.worker.launch_info["bridge_pid"] for r in ready}), 64)

    def test_capacity_limit_is_explicit_and_does_not_fake_remaining_games(self):
        workers = [
            SimpleNamespace(
                index=i, launch_info={}, report=Mock(), log=Mock(), shutdown=Mock()
            )
            for i in range(64)
        ]
        state = {"started": 0}

        def start(w):
            state["started"] += 1
            w.launch_info["bridge_pid"] = w.index + 10
            return SimpleNamespace(worker=w)

        def memory():
            return dict(
                total=16 * GIB,
                available=(12 if state["started"] == 0 else 1) * GIB,
                commit_available=20 * GIB,
            )

        ready, reason = launch_workers(
            workers,
            start,
            threading.Event(),
            2,
            memory=memory,
            inspect=lambda pid: None,
        )
        self.assertEqual(len(ready), 1)
        self.assertIn("Memory limit", reason)
        self.assertEqual(workers[63].report.call_args.kwargs["phase"], "blocked")
        self.assertEqual(rollout_size(dict(rollout=128), 64), 32)

    def test_reused_bridge_pid_is_rejected(self):
        workers = [
            SimpleNamespace(
                index=i, launch_info={}, report=Mock(), log=Mock(), shutdown=Mock()
            )
            for i in range(2)
        ]

        def start(w):
            w.launch_info["bridge_pid"] = 123
            return SimpleNamespace(worker=w)

        ready, _ = launch_workers(
            workers,
            start,
            threading.Event(),
            1,
            memory=lambda: None,
            inspect=lambda pid: None,
        )
        self.assertEqual(len(ready), 1)
        self.assertEqual(workers[1].report.call_args.kwargs["phase"], "failed")
