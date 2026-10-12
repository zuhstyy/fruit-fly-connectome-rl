from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from learning import Policy
from plastic import PlasticPolicy
from train import Runner, validate_config
from protocol import BridgeError, ControlPaused, ControlStopped
from test_learning import state


class FakeGame:
    def __init__(self):
        self.frame = 0
        self.chapter = 0

    def set_feedback(self, total, last, distance, score=0.0, event="", immediate=True):
        self.feedback = (total, last, distance, score, event)
        if not hasattr(self, "feedback_history"):
            self.feedback_history = []
        self.feedback_history.append(self.feedback)

    def start(self, chapter, stop=None):
        self.chapter = chapter
        self.frame = 0
        return state(chapter=chapter)

    def step(self, action, frames):
        self.frame += frames
        # Known artificial terminal at 12 frames tests campaign handling, not game skill.
        return state(
            chapter=self.chapter,
            x=10 + self.frame,
            frame=self.frame,
            completed=self.frame >= 12,
        )

    def retry(self, force=False):
        return self.start(self.chapter)


class FakeWorker:
    def __init__(self, index):
        self.index = index
        self.stop = threading.Event()
        self.client = FakeGame()

    def recover(self, chapter):
        return self.client.start(chapter)


def config():
    return dict(
        seed=17,
        workers=2,
        ticks=1,
        frames=4,
        rollout=32,
        episode_steps=500,
        stuck_steps=240,
        hours=1,
        chapters=list(range(8)),
    )


class WorkerTests(unittest.TestCase):
    def test_plastic_worker_keeps_learning_without_timer_deaths(self):
        class StalledGame(FakeGame):
            def step(self, action, frames):
                self.frame += frames
                return state(
                    frame=self.frame,
                    route_rank=1,
                    exit_forward=True,
                    exit_id="0:1",
                    exit_distance=100,
                )

            def retry(self, force=False):
                raise AssertionError(
                    "A live stalled character must never be killed/retried"
                )

        w = FakeWorker(0)
        w.client = StalledGame()
        c = config()
        c.update(stuck_steps=30, episode_steps=100)
        p = PlasticPolicy()
        r = Runner(
            w,
            p,
            state(route_rank=1, exit_forward=True, exit_id="0:1", exit_distance=100),
            c,
        )
        first = r.roll(128)
        self.assertEqual(len(first["actions"]), 128)
        self.assertFalse(w.stop.is_set())
        self.assertEqual(r.deaths, 0)
        self.assertGreater(float(first["temperatures"].max()), 1)
        self.assertGreater(r.practice_resets, 0)
        metrics = p.update_rollouts([first])
        r.publish_feedback()
        second = r.roll(32)
        self.assertEqual(len(second["actions"]), 32)
        self.assertEqual(w.client.learning_info["brain_updates"], 1)
        self.assertGreater(w.client.learning_info["learned_edges"], 0)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_plastic_worker_retries_real_death_and_resets_memory(self):
        class DyingGame(FakeGame):
            retries = 0

            def step(self, action, frames):
                self.frame += frames
                return state(frame=self.frame, dead=self.frame == 8)

            def retry(self, force=False):
                if force:
                    raise AssertionError("No forced death allowed")
                self.retries += 1
                self.frame = 0
                return state()

        w = FakeWorker(0)
        w.client = DyingGame()
        p = PlasticPolicy()
        p.actor[:, 14:] = 0
        p.actor[-1, 14:] = -100
        p.actor[-1, 16] = 0
        r = Runner(w, p, state(), config())
        batch = r.roll(3)
        self.assertEqual(w.client.retries, 1)
        self.assertEqual(r.deaths, 1)
        np.testing.assert_array_equal(batch["resets"], [True, False, True])
        self.assertTrue(np.isfinite(batch["advantages"]).all())

    def test_runner_does_not_guess_backward_penalties_from_room_rank(self):
        class BacktrackingGame(FakeGame):
            def step(self, action, frames):
                self.frame += frames
                return state(
                    room="b" if self.frame == frames else "a",
                    route_rank=1 if self.frame == frames else 0,
                    frame=self.frame,
                )

        w = FakeWorker(0)
        w.client = BacktrackingGame()
        r = Runner(w, Policy(), state(room="a", route_rank=0), config())
        batch = r.roll(2)
        total, last, distance, score, event = w.client.feedback
        self.assertLess(score, 0)
        self.assertLess(total, 0)
        self.assertAlmostEqual(float(batch[4][-1]), last, places=5)
        self.assertNotIn("BACKTRACK", event)
        self.assertEqual(len(w.client.feedback_history), 3)

    def test_manual_interruption_is_excluded_and_chapter_restarts(self):
        class PausingGame(FakeGame):
            calls = 0
            starts = 0
            manual_waits = 0

            def step(self, action, frames):
                self.calls += 1
                if self.calls == 2:
                    raise ControlPaused("OFF")
                return super().step(action, frames)

            def wait_until_on(self, stop):
                self.manual_waits += 1
                self.frame = 9000  # The human moves while control is OFF.

            def start(self, chapter, stop=None):
                self.starts += 1
                return super().start(chapter, stop)

        w = FakeWorker(0)
        w.client = PausingGame()
        r = Runner(w, Policy(), state(), config())
        batch = r.roll(3)
        self.assertEqual(len(batch[0]), 2)
        self.assertEqual(r.total_steps, 2)
        self.assertEqual(w.client.manual_waits, 1)
        self.assertEqual(w.client.starts, 1)
        self.assertEqual(r.state["frame"], 4)
        self.assertEqual(r.completed, [])
        self.assertTrue(np.isfinite(batch[4]).all())

    def test_emergency_stop_does_not_restart_the_worker(self):
        class StoppedGame(FakeGame):
            def step(self, a, f):
                raise ControlStopped("F10")

        w = FakeWorker(0)
        w.client = StoppedGame()

        def forbidden(chapter):
            self.fail("A stopped worker must not be recovered")

        w.recover = forbidden
        r = Runner(w, Policy(), state(), config())
        self.assertIsNone(r.roll(3))
        self.assertTrue(w.stop.is_set())

    def test_parallel_rollouts_and_full_campaign_bookkeeping(self):
        p = Policy()
        runners = [Runner(FakeWorker(i), p, state(), config()) for i in range(2)]
        with ThreadPoolExecutor(2) as pool:
            batches = list(pool.map(lambda r: r.roll(32), runners))
        for r, b in zip(runners, batches):
            self.assertEqual(set(r.completed), set(range(8)))
            self.assertFalse(r.worker.stop.is_set())
            self.assertEqual(len(b[0]), 32)
            self.assertTrue(np.isfinite(b[4]).all())
        self.assertFalse(np.array_equal(batches[0][1], batches[1][1]))

    def test_bad_worker_is_recovered_without_false_completion(self):
        class BrokenOnce(FakeGame):
            def step(self, a, f):
                raise BridgeError("injected socket failure")

        w = FakeWorker(0)
        w.client = BrokenOnce()

        def recover(chapter):
            w.client = FakeGame()
            return w.client.start(chapter)

        w.recover = recover
        r = Runner(w, Policy(), state(), config())
        b = r.roll(4)
        self.assertEqual(r.completed, [0])
        self.assertEqual(len(b[0]), 3)

    def test_configuration_bounds(self):
        validate_config(config())
        c = config()
        c["workers"] = 100
        with self.assertRaises(ValueError):
            validate_config(c)


if __name__ == "__main__":
    unittest.main()
