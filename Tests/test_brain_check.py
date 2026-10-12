"""The evidence check must be useful without mutating a running model."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from check_brain import GRAPH, inspect_graph, run_checks, wiring_experiment
from plastic import PlasticPolicy


class BrainCheckTests(unittest.TestCase):
    def test_fly_connections_affect_action_probabilities(self):
        policy = PlasticPolicy()
        before = [array.copy() for array in policy.parameters]
        result = wiring_experiment(policy)
        self.assertTrue(result["wiring_changes_action_probabilities"])
        self.assertGreater(result["mean_neuron_activity_difference"], 0)
        for original, current in zip(before, policy.parameters):
            np.testing.assert_array_equal(original, current)

    def test_wiring_report_does_not_claim_an_effect_if_actor_ignores_neurons(self):
        policy = PlasticPolicy()
        policy.actor[:-1] = 0
        result = wiring_experiment(policy)
        self.assertFalse(result["wiring_changes_action_probabilities"])
        self.assertGreater(result["mean_neuron_activity_difference"], 0)

    def test_full_check_preserves_checkpoint_bytes_and_reports_saved_history(self):
        policy = PlasticPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.npz"
            policy.save(checkpoint, policy.graph_hash, {"steps": 123})
            before = checkpoint.read_bytes()
            result = run_checks(checkpoint)
            self.assertTrue(result["passed"])
            self.assertEqual(result["saved_decisions"], 123)
            self.assertEqual(result["saved_changed_edges"], 0)
            self.assertGreater(
                result["learning_experiment"]["changed_recurrent_edges"], 0
            )
            self.assertTrue(all(result["memory_experiment"].values()))
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse(result["sources"]["original_exports_checked"])

    def test_wrong_graph_checkpoint_is_rejected(self):
        policy = PlasticPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "bad.npz"
            policy.save(checkpoint, policy.graph_hash, {})
            with np.load(checkpoint, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            arrays["graph_hash"] = np.array("wrong graph")
            np.savez(checkpoint, **arrays)
            with self.assertRaises(ValueError):
                run_checks(checkpoint)

    def test_unlisted_recurrent_edge_is_rejected(self):
        with np.load(GRAPH, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        row, column = np.argwhere(arrays["recurrent"] == 0)[0]
        arrays["recurrent"][row, column] = 0.1
        with tempfile.TemporaryDirectory() as temporary:
            graph = Path(temporary) / "graph.npz"
            np.savez(graph, **arrays)
            graph.with_suffix(".json").write_text(
                GRAPH.with_suffix(".json").read_text()
            )
            with self.assertRaisesRegex(ValueError, "exactly the listed edges"):
                inspect_graph(graph)

    def test_original_export_hash_mismatch_is_rejected(self):
        from check_brain import verify_original_sources

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "neurons.csv.gz").write_bytes(b"not the original export")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_original_sources(temporary)

    def test_checkpoint_save_recovers_from_a_brief_reader_lock(self):
        policy = PlasticPolicy()
        replace = Path.replace
        attempts = []

        def reader_lock_once(path, destination):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError("temporary reader lock")
            return replace(path, destination)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.npz"
            with (
                patch.object(Path, "replace", reader_lock_once),
                patch("learning.time.sleep"),
            ):
                policy.save(checkpoint, policy.graph_hash, {})
            restored = PlasticPolicy()
            restored.load(checkpoint, policy.graph_hash)
            self.assertEqual(len(attempts), 2)
            for before, after in zip(policy.parameters, restored.parameters):
                np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
