from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from accelerator import TorchGradients, choose_backend, probe_segments
from plastic import PlasticPolicy


class BackendSelectionTests(unittest.TestCase):
    def test_cpu_selection_preserves_parameters_optimizer_and_rng(self):
        policy = PlasticPolicy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "before.npz"
            policy.save(path, policy.graph_hash, {})
            before = path.read_bytes()
            report = choose_backend(policy, "cpu", log=lambda _: None)
            policy.save(path, policy.graph_hash, {})
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(report["selected"], "numpy")
            self.assertIsNone(policy.gradient_backend)


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Optional PyTorch is not installed"
)
class TorchGradientTests(unittest.TestCase):
    def segments(self, policy):
        segments = probe_segments(policy, 2, 6)
        segments[0]["resets"][3] = True
        for key in [
            "observations",
            "resets",
            "actions",
            "old_logp",
            "advantages",
            "returns",
            "temperatures",
        ]:
            segments[1][key] = segments[1][key][:4]
        segments[1]["observations"] = segments[1]["observations"][:, :92]
        segments[0]["temperatures"][:] = 1.7
        policy.value[:] = 0.03
        return segments

    def test_ppo_matches_numpy_with_resets_padding_and_temperatures(self):
        policy = PlasticPolicy()
        segments = self.segments(policy)
        reference = policy.numpy_sequence_gradients(segments)
        for precision, tolerance in [("float64", 1e-10), ("float32", 2e-6)]:
            actual = TorchGradients(policy, "cpu", precision).gradients(segments)
            self.assertAlmostEqual(reference[0], actual[0], places=5)
            for expected, gradient in zip(reference[1], actual[1]):
                np.testing.assert_allclose(
                    expected, gradient, atol=tolerance, rtol=tolerance
                )

    def test_imitation_gradients_match_and_value_is_unchanged(self):
        policy = PlasticPolicy()
        segments = self.segments(policy)
        expected = policy.numpy_sequence_gradients(segments, objective="imitation")
        backend = TorchGradients(policy, "cpu", "float64")
        actual = backend.gradients(segments, objective="imitation")
        for a, b in zip(expected[1], actual[1]):
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(actual[1][1], np.zeros_like(policy.value))

    def test_tensor_update_changes_real_edges_and_reloads_numpy_checkpoint(self):
        policy = PlasticPolicy()
        segments = self.segments(policy)
        initial = policy.edge_weights.copy()
        policy.gradient_backend = TorchGradients(policy, "cpu", "float32")
        loss, gradients, _, _ = policy.sequence_gradients(segments)
        policy.apply_gradients(gradients, 0.001)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(np.max(np.abs(initial - policy.edge_weights)), 0)
        np.testing.assert_array_equal(np.sign(initial), np.sign(policy.edge_weights))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            policy.save(path, policy.graph_hash, {"steps": 123})
            restored = PlasticPolicy()
            restored.load(path, restored.graph_hash)
            for a, b in zip(
                policy.parameters + policy.m + policy.v,
                restored.parameters + restored.m + restored.v,
            ):
                np.testing.assert_array_equal(a, b)
            self.assertIsNone(restored.gradient_backend)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_cuda_parity_on_available_hardware(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("No CUDA GPU in this environment")
        policy = PlasticPolicy()
        segments = self.segments(policy)
        expected = policy.numpy_sequence_gradients(segments)
        actual = TorchGradients(policy, "cuda").gradients(segments)
        for a, b in zip(expected[1], actual[1]):
            np.testing.assert_allclose(a, b, rtol=0.004, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
