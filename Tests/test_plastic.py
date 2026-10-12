from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from plastic import PlasticPolicy, PlasticState


def rollout(policy, observations, seed=31, temperature=1.5):
    rng = np.random.default_rng(seed)
    brain = PlasticState(policy)
    rows = {
        k: []
        for k in [
            "observations",
            "before",
            "resets",
            "actions",
            "old_logp",
            "advantages",
            "returns",
            "temperatures",
        ]
    }
    for obs in observations:
        f = brain.features(obs)
        action, logp, value = policy.act(f, rng, temperature)
        reward = 1.0 if action[0] == (2 if obs[0] > 0 else 0) else -1.0
        for key, item in dict(
            observations=obs,
            before=brain.last_before,
            resets=brain.last_reset,
            actions=action,
            old_logp=logp,
            advantages=reward - value,
            returns=reward,
            temperatures=temperature,
        ).items():
            rows[key].append(item)
    return {k: np.asarray(v) for k, v in rows.items()}


class PlasticTests(unittest.TestCase):
    def test_recurrent_and_actor_gradients_match_finite_difference(self):
        p = PlasticPolicy(seed=8)
        b = rollout(p, np.random.default_rng(9).normal(size=(4, 92)))
        s = {k: v for k, v in b.items() if k != "before"}
        s["initial"] = b["before"][0]
        loss, grads, _, _ = p.sequence_gradients([s])
        for param, grad in zip(p.parameters, grads):
            flat = int(np.argmax(np.abs(grad)))
            ix = np.unravel_index(flat, param.shape)
            old = param[ix]
            eps = 1e-5
            param[ix] = old + eps
            plus = p.sequence_gradients([s])[0]
            param[ix] = old - eps
            minus = p.sequence_gradients([s])[0]
            param[ix] = old
            self.assertGreater(abs(grad[ix]), 1e-9)
            self.assertAlmostEqual(
                (plus - minus) / (2 * eps), float(grad[ix]), places=6
            )

    def test_bptt_does_not_cross_reset(self):
        p = PlasticPolicy(seed=8)
        b = rollout(p, np.random.default_rng(10).normal(size=(4, 92)))
        b["resets"][2] = True
        s = {k: v for k, v in b.items() if k != "before"}
        s["initial"] = b["before"][0]
        _, grads, _, _ = p.sequence_gradients([s])
        ix = int(np.argmax(np.abs(grads[2])))
        old = p.theta[ix]
        eps = 1e-5
        p.theta[ix] = old + eps
        plus = p.sequence_gradients([s])[0]
        p.theta[ix] = old - eps
        minus = p.sequence_gradients([s])[0]
        p.theta[ix] = old
        self.assertAlmostEqual((plus - minus) / (2 * eps), grads[2][ix], places=6)

    def test_real_edges_change_and_preserve_topology_sign_and_stability(self):
        p = PlasticPolicy()
        b = rollout(p, np.random.default_rng(11).normal(size=(32, 92)))
        initial = p.edge_weights.copy()
        p.update_rollouts([b])
        self.assertGreater(p.diagnostics()["learned_edges"], 0)
        self.assertFalse(np.array_equal(initial, p.edge_weights))
        self.assertEqual(len(p.theta), 3326)
        np.testing.assert_array_equal(np.sign(initial), np.sign(p.edge_weights))
        sums = np.bincount(p.target, weights=np.abs(p.edge_weights), minlength=p.nodes)
        self.assertLessEqual(sums.max(), 0.935001)

    def test_full_checkpoint_roundtrip_and_graph_guard(self):
        p = PlasticPolicy()
        p.update_rollouts([rollout(p, np.ones((16, 92)))])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "latest.npz"
            p.save(path, p.graph_hash, {"steps": 16})
            q = PlasticPolicy(seed=9)
            self.assertEqual(q.load(path, p.graph_hash)["steps"], 16)
            for a, b in zip(p.parameters, q.parameters):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(p.m + p.v, q.m + q.v):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(p.updates, q.updates)
            with self.assertRaises(ValueError):
                q.load(path, "wrong graph")

    def test_legacy_model_is_backed_up_before_fresh_training(self):
        from learning import Policy
        from train import prepare_checkpoint

        p = PlasticPolicy()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "latest.npz"
            Policy().save(path, p.graph_hash, {"steps": 123})
            original = path.read_bytes()
            self.assertEqual(prepare_checkpoint(p, path, lambda m: None), {})
            backups = list((path.parent / "legacy-models").glob("*.npz"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(path.read_bytes(), original)

    def test_reencode_bootstrap_does_not_advance_memory_twice(self):
        p = PlasticPolicy()
        s = PlasticState(p)
        s.features(np.ones(92))
        p.theta += 0.001
        expected, activity, _ = p.forward(s.last_observation, s.last_before)
        np.testing.assert_allclose(s.refresh(), expected)
        np.testing.assert_allclose(s.activity, activity)

    def test_learning_improves_sensor_dependent_action(self):
        p = PlasticPolicy(seed=6)
        rng = np.random.default_rng(3)

        def accuracy():
            scores = []
            for sign in [-1.0, 1.0]:
                obs = np.zeros(92)
                obs[0] = sign
                f, _, _ = p.forward(obs, np.zeros(p.nodes))
                logits = (f @ p.actor)[:3]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                scores.append(probs[2 if sign > 0 else 0])
            return float(np.mean(scores))

        before = accuracy()
        for i in range(22):
            obs = np.zeros((96, 92))
            obs[:, 0] = rng.choice([-1.0, 1.0], size=96)
            b = rollout(p, obs, seed=i, temperature=1.0)
            p.update_rollouts([b], rate=0.008, epochs=2)
        self.assertGreater(accuracy(), before + 0.15)
        self.assertGreater(p.diagnostics()["learned_edges"], 100)


if __name__ == "__main__":
    unittest.main()
