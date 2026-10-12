import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from learning import Policy, Reservoir, Reward, gae
from protocol import HEADS, action_fields, sensors


def state(**kw):
    s = dict(
        chapter=0,
        room="start",
        x=10.0,
        y=30.0,
        vx=0.0,
        vy=0.0,
        grounded=True,
        wall_left=False,
        wall_right=False,
        dashes=1,
        stamina=110.0,
        goal_dx=100.0,
        goal_dy=0.0,
        bounds_x=0,
        bounds_y=0,
        bounds_width=320,
        bounds_height=180,
        grid=[0.0] * 81,
        dead=False,
        completed=False,
        frame=0,
    )
    s.update(kw)
    return s


class LearningTests(unittest.TestCase):
    def test_real_graph_preserved_and_deterministic(self):
        a, b = Reservoir(), Reservoir()
        self.assertEqual(len(a.root_ids), 256)
        self.assertEqual(len(a.edge_source), 3326)
        self.assertTrue(
            np.array_equal(a.features(sensors(state())), b.features(sensors(state())))
        )
        a.reset()
        b.reset()
        self.assertTrue(np.isfinite(a.features(np.ones(92))).all())
        with self.assertRaises(ValueError):
            a.features(np.zeros(91))

    def test_all_action_combinations(self):
        from itertools import product

        items = [action_fields(a) for a in product(*(range(n) for n in HEADS))]
        self.assertEqual(len(items), 1008)
        self.assertEqual(len({json.dumps(a, sort_keys=True) for a in items}), 1008)
        with self.assertRaises(ValueError):
            action_fields([3, 1, 0, 0, 0, 0])

    def test_gae_does_not_cross_deaths(self):
        a, r = gae([1, 2, 100], [0, 0, 0], [False, True, False], 10, gamma=1, lam=1)
        np.testing.assert_allclose(r, [3, 2, 110])

    def test_ppo_gradient_finite_difference(self):
        p = Policy(features=3, seed=3)
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 3)).astype(np.float64)
        p.actor = p.actor.astype(np.float64)
        p.value = p.value.astype(np.float64)
        chosen = [p.act(f, rng) for f in x]
        a = np.array([v[0] for v in chosen])
        lp = np.array([v[1] for v in chosen])
        adv = rng.normal(size=12)
        ret = rng.normal(size=12)
        args = (x, a, lp, adv, ret)
        _, ga, gv, _, _ = p.gradients(*args)
        for matrix, grad, indices in [
            (p.actor, ga, [(0, 0), (2, 4), (1, 9)]),
            (p.value, gv, [(0,), (2,)]),
        ]:
            for ix in indices:
                old = matrix[ix]
                eps = 1e-5
                matrix[ix] = old + eps
                plus = p.gradients(*args)[0]
                matrix[ix] = old - eps
                minus = p.gradients(*args)[0]
                matrix[ix] = old
                self.assertAlmostEqual(
                    (plus - minus) / (2 * eps), float(grad[ix]), places=5
                )

    def test_clipped_surrogate_gradient(self):
        p = Policy(features=2)
        x = np.ones((1, 2))
        a = np.zeros((1, len(HEADS)), dtype=int)
        lp = sum(np.log(1 / n) for n in HEADS)
        # Ratio far above 1.2: positive-advantage actor gradient is zero.
        _, g, _, _, _ = p.gradients(
            x, a, np.array([lp - 3]), np.array([1.0]), np.array([0.0]), entropy=0
        )
        self.assertLess(abs(g).max(), 1e-10)

    def test_checkpoint_roundtrip_and_graph_guard(self):
        p = Policy()
        p.t = 12
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "latest.npz"
            p.save(path, "graph", {"steps": 42})
            q = Policy(seed=99)
            self.assertEqual(q.load(path, "graph")["steps"], 42)
            np.testing.assert_array_equal(p.actor, q.actor)
            self.assertEqual(q.t, 12)
            with self.assertRaises(ValueError):
                q.load(path, "different")
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_learning_improves_simple_reward(self):
        # Behavioral check: can the same PPO implementation learn rewarded rightward motion?
        p = Policy(features=2)
        rng = np.random.default_rng(18)
        x = np.ones((512, 2), dtype=np.float32)
        before = sum(p.act(x[0], rng)[0][0] == 2 for _ in range(1000)) / 1000
        for _ in range(20):
            samples = [p.act(f, rng) for f in x]
            a = np.array([s[0] for s in samples])
            lp = np.array([s[1] for s in samples])
            v = np.array([s[2] for s in samples])
            reward = (a[:, 0] == 2).astype(float)
            p.update((x, a, lp, reward - v, reward), rate=0.02, epochs=2)
        after = sum(p.act(x[0], rng)[0][0] == 2 for _ in range(1000)) / 1000
        self.assertGreater(after, 0.85)
        self.assertGreater(after, before + 0.4)

    def test_reward_no_spawn_or_oscillation_farming(self):
        rw = Reward()
        s = state()
        rw.reset(s)
        self.assertLess(rw.step(s, s), 0)
        t = state(x=40)
        first = rw.step(s, t)
        rw.step(t, s)
        second = rw.step(s, t)
        self.assertLess(first, 0)
        self.assertLess(second, 0)
        rw.reset(s)
        self.assertLess(rw.step(s, s), 0)

    def test_sensor_validation(self):
        self.assertEqual(sensors(state()).shape, (156,))
        with self.assertRaises(Exception):
            sensors(state(x=float("nan")))

    def test_backward_entry_penalizes_but_preserves_earned_progress(self):
        rw = Reward()
        a = state(
            room="start",
            route_rank=1,
            exit_forward=True,
            exit_id="start:next",
            exit_distance=200,
        )
        b = state(
            room="start",
            route_rank=1,
            exit_forward=True,
            exit_id="start:next",
            exit_distance=50,
        )
        rw.reset(a)
        earned = rw.step(a, b)
        self.assertGreater(rw.score, 0)
        # The car was never visited during this run. Static rank still detects it.
        loss = rw.step(b, state(room="car", route_rank=0))
        self.assertLess(rw.score, 0)
        self.assertAlmostEqual(rw.score, earned + loss)
        self.assertIn("BACKTRACK", rw.event)
        rw.reset(state(room="car", route_rank=0))
        self.assertEqual(rw.score, 0)
        self.assertEqual(rw.last, 0)

    def test_no_positive_reward_for_wrong_exit_or_new_ground(self):
        rw = Reward()
        a = state(exit_forward=False, exit_id="back", exit_distance=100)
        b = state(x=-100, exit_forward=False, exit_id="back", exit_distance=0)
        rw.reset(a)
        self.assertLess(rw.step(a, b), 0)
        self.assertEqual(rw.distance_reward, 0)

    def test_distance_reward_signed_and_no_oscillation_gain(self):
        a = state(exit_forward=True, exit_id="a:b", exit_distance=100)
        b = state(exit_forward=True, exit_id="a:b", exit_distance=60)
        rw = Reward()
        rw.reset(a)
        rw.step(a, b)
        self.assertAlmostEqual(rw.distance_reward, 0.8)
        rw.step(b, a)
        self.assertAlmostEqual(rw.distance_reward, -0.8)
        self.assertLess(rw.step(a, b) + rw.step(b, a), 0)

    def test_distance_reward_ignores_discontinuities(self):
        a = state(exit_forward=True, exit_id="a:b", exit_distance=100)
        for b in [
            state(exit_id="a:c", exit_distance=0),
            state(room="b", exit_forward=True, exit_id="a:b", exit_distance=0),
            state(dead=True, exit_forward=True, exit_id="a:b", exit_distance=0),
            state(completed=True, exit_forward=True, exit_id="a:b", exit_distance=0),
            state(exit_forward=True, exit_id="a:b", exit_distance=float("nan")),
        ]:
            rw = Reward()
            rw.reset(a)
            rw.step(a, b)
            self.assertEqual(rw.distance_reward, 0)

    def test_backtracking_penalty_and_no_room_reward_farming(self):
        rw = Reward()
        a = state(room="a")
        b = state(room="b")
        rw.reset(a)
        self.assertGreater(rw.step(a, b), 2)
        self.assertLessEqual(rw.step(b, state(room="a", x=80)), -1)
        self.assertLess(rw.step(a, state(room="a", x=120)), 0)
        self.assertLess(rw.step(a, b), 0)
        rw.reset(b)  # Death/retry does not forget which room was behind.
        self.assertLessEqual(rw.step(b, a), -1)
        rw.reset(a)  # Respawn itself is not a backward action.
        self.assertAlmostEqual(rw.step(a, a), -0.01)

    def test_backtracking_death_and_chapter_boundaries(self):
        rw = Reward()
        a = state(room="a")
        b = state(room="b")
        rw.reset(a)
        rw.step(a, b)
        self.assertAlmostEqual(rw.step(b, state(room="a", dead=True)), -2.01)
        c = state(chapter=1, room="a")
        rw.reset(c)
        self.assertAlmostEqual(rw.step(c, c), -0.01)
        self.assertGreater(rw.step(c, state(chapter=1, room="b")), 2)


if __name__ == "__main__":
    unittest.main()
