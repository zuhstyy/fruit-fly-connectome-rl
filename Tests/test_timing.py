from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from protocol import DURATIONS, HEADS, LEGACY_SCHEMA, action_fields
from plastic import PlasticPolicy, PlasticState, LEGACY_PLASTIC_SCHEMA
from learning import gae, Reward, softmax, Policy
from train import prepare_checkpoint, Runner
from test_learning import state
from test_workers import FakeWorker, FakeGame, config
from test_plastic import rollout


class TimingTests(unittest.TestCase):
    def test_duration_head_controls_wire_frames_and_calibration_still_works(self):
        for choice, frames in enumerate(DURATIONS):
            value = action_fields([2, 1, 1, 1, 1, 0, choice], frames=4)
            self.assertEqual(value["frames"], frames)
            self.assertTrue(value["jump"] and value["dash"] and value["grab"])
        self.assertEqual(action_fields([2, 1, 0, 0, 0, 0], 8)["frames"], 8)
        with self.assertRaises(ValueError):
            action_fields([2, 1, 0, 0, 0, 0, 99])
        with self.assertRaises(ValueError):
            action_fields([2, 1, 0, 0, 0, 0], 31)

    def test_time_scaled_gae_and_terminal_boundary(self):
        adv, ret = gae(
            [1, 2], [3, 4], [False, False], 5, gamma=0.9, lam=0.8, durations=[2, 12]
        )
        last = 2 + 0.9**3 * 5 - 4
        first = 1 + 0.9**0.5 * 4 - 3 + (0.9 * 0.8) ** 0.5 * last
        np.testing.assert_allclose(adv, [first, last], rtol=1e-6)
        np.testing.assert_allclose(ret, [first + 3, last + 4], rtol=1e-6)
        adv, _ = gae([1, 100], [0, 0], [True, False], 100, durations=[30, 1])
        self.assertEqual(adv[0], 1)

    def test_time_penalty_cannot_be_evaded_by_selecting_long_holds(self):
        s = state()
        short = Reward()
        long = Reward()
        short.reset(s)
        long.reset(s)
        parts = sum(short.step(s, s, 1) for _ in range(30))
        self.assertAlmostEqual(parts, long.step(s, s, 30))
        self.assertAlmostEqual(
            short.steps_without_progress, long.steps_without_progress
        )

    def test_runner_uses_learned_duration_and_reports_truncated_action_time(self):
        class Game(FakeGame):
            def step(self, action, frames):
                self.last_frames = frames
                self.frame += 17
                return state(frame=self.frame, action_frames=17)

        w = FakeWorker(0)
        w.client = Game()
        p = PlasticPolicy()
        p.act = lambda f, rng, temperature=1.0: (
            np.array([2, 1, 1, 1, 1, 0, 6]),
            0.0,
            0.0,
        )
        r = Runner(w, p, state(), config())
        batch = r.roll(4)
        self.assertEqual(w.client.last_frames, 30)
        np.testing.assert_array_equal(batch["durations"], [17] * 4)
        self.assertEqual(r.longest_holds["right"], 68)
        self.assertEqual(r.longest_holds["jump"], 68)
        self.assertEqual(r.snapshot()["requested_frames"], 30)
        self.assertEqual(r.duration_counts["30"], 4)
        second = r.roll(2)
        self.assertFalse(
            second["resets"][0]
        )  # No artificial stream-switch memory reset.

    def test_30_checkpoint_migration_preserves_every_existing_parameter_and_moment(
        self,
    ):
        p = PlasticPolicy()
        p.update_rollouts([rollout(p, np.ones((16, 92)))])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "latest.npz"
            p.save(path, p.graph_hash, {"steps": 123, "chapters": list(range(8))})
            with np.load(path, allow_pickle=False) as f:
                old = {k: f[k].copy() for k in f.files}
            for key in ("actor", "m0", "v0"):
                old[key] = old[key][:, :14]
            old["schema"] = np.array(LEGACY_SCHEMA)
            old["plastic_schema"] = np.array(LEGACY_PLASTIC_SCHEMA)
            np.savez(path, **old)
            original = path.read_bytes()
            q = PlasticPolicy(seed=44)
            metadata = prepare_checkpoint(q, path, lambda s: None)
            self.assertEqual(metadata["migrated_from"], "3.0.0")
            self.assertEqual(metadata["steps"], 123)
            np.testing.assert_array_equal(q.actor[:, :14], old["actor"])
            np.testing.assert_array_equal(q.value, old["value"])
            np.testing.assert_array_equal(q.theta, old["theta"])
            for i in range(3):
                np.testing.assert_array_equal(
                    q.m[i][:, :14] if i == 0 else q.m[i], old["m" + str(i)]
                )
                np.testing.assert_array_equal(
                    q.v[i][:, :14] if i == 0 else q.v[i], old["v" + str(i)]
                )
            self.assertEqual(q.t, int(old["t"]))
            self.assertEqual(q.updates, int(old["updates"]))
            self.assertEqual(path.read_bytes(), original)
            backups = list((path.parent / "legacy-models").glob("*.npz"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            q.save(path, q.graph_hash, metadata)
            r = PlasticPolicy()
            r.load(path, r.graph_hash)
            np.testing.assert_array_equal(q.actor, r.actor)

    def test_batched_loss_matches_independent_sparse_forward_for_ragged_resets(self):
        p = PlasticPolicy()
        rng = np.random.default_rng(6)
        segments = []
        all_features = []
        for n in (2, 7, 16):
            b = rollout(p, rng.normal(size=(n, 92)))
            s = {k: v for k, v in b.items() if k != "before"}
            s["initial"] = b["before"][0]
            s["resets"][n // 2] = True
            segments.append(s)
            h = s["initial"].copy()
            for obs, reset in zip(s["observations"], s["resets"]):
                if reset:
                    h.fill(0)
                f, h, _ = p.forward(obs, h)
                all_features.append(f)
        cat = lambda k: np.concatenate([s[k] for s in segments])
        expected = Policy.gradients(
            p,
            np.asarray(all_features),
            cat("actions"),
            cat("old_logp"),
            cat("advantages"),
            cat("returns"),
            entropy=0.02,
            temperatures=cat("temperatures"),
        )
        actual = p.sequence_gradients(segments)
        self.assertAlmostEqual(expected[0], actual[0], places=12)
        np.testing.assert_allclose(expected[1], actual[1][0], atol=1e-12)
        np.testing.assert_allclose(expected[2], actual[1][1], atol=1e-12)
        ix = np.unravel_index(
            np.argmax(abs(actual[1][0][:, 14:])), actual[1][0][:, 14:].shape
        )
        ix = (ix[0], ix[1] + 14)
        saved = p.actor[ix]
        eps = 1e-5
        p.actor[ix] = saved + eps
        plus = p.sequence_gradients(segments)[0]
        p.actor[ix] = saved - eps
        minus = p.sequence_gradients(segments)[0]
        p.actor[ix] = saved
        self.assertAlmostEqual((plus - minus) / (2 * eps), actual[1][0][ix], places=6)

    def test_duration_policy_learns_sensor_dependent_short_and_long_holds(self):
        p = PlasticPolicy(seed=6)
        rng = np.random.default_rng(10)

        def score():
            values = []
            for sign, choice in [(-1, 1), (1, 4)]:
                obs = np.zeros(92)
                obs[0] = sign
                f, _, _ = p.forward(obs, np.zeros(p.nodes))
                values.append(softmax((f @ p.actor)[14:])[choice])
            return float(np.mean(values))

        before = score()
        for _ in range(40):
            brain = PlasticState(p)
            rows = {
                k: []
                for k in (
                    "observations",
                    "before",
                    "resets",
                    "actions",
                    "old_logp",
                    "advantages",
                    "returns",
                    "temperatures",
                )
            }
            for i in range(96):
                obs = np.zeros(92)
                obs[0] = rng.choice([-1.0, 1.0])
                f = brain.features(obs)
                a, lp, value = p.act(f, rng, 1.5)
                reward = 1.0 if a[6] == (1 if obs[0] < 0 else 4) else -0.1
                for key, item in dict(
                    observations=obs,
                    before=brain.last_before,
                    resets=brain.last_reset,
                    actions=a,
                    old_logp=lp,
                    advantages=reward - value,
                    returns=reward,
                    temperatures=1.5,
                ).items():
                    rows[key].append(item)
            p.update_rollouts(
                [{k: np.asarray(v) for k, v in rows.items()}], epochs=2, rate=0.008
            )
        self.assertGreater(score(), before + 0.2)
        self.assertGreater(p.diagnostics()["learned_edges"], 100)
