"""Check fly wiring, its effect on actions, learning, and saved memory.

This is a small offline experiment. It never opens Celeste or updates the
trainer's checkpoint. Its synthetic inputs cannot prove that a room is solved.
"""

import argparse
from collections import defaultdict
import copy
import html
import json
from pathlib import Path
import tempfile

import numpy as np

from celeste_connectome import rows, sha256
from learning import softmax
from plastic import PlasticPolicy, PlasticState
from protocol import HEADS, VERSION, SENSOR_COUNT

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "Python/data/mcns_subnetwork.npz"


def inspect_graph(graph=GRAPH):
    """Check that the bundled edge list and recurrent matrix agree."""
    graph = Path(graph)
    metadata = json.loads(graph.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(graph, allow_pickle=False) as data:
        neuron_ids = data["root_ids"]
        source = data["edge_source"]
        target = data["edge_target"]
        recurrent = data["recurrent"]
        size = len(neuron_ids)
        if size != metadata["subnetwork_neurons"] or len(set(neuron_ids)) != size:
            raise ValueError("Neuron IDs do not match the graph metadata")
        if (
            source.shape != target.shape
            or len(source) != metadata["subnetwork_directed_edges"]
        ):
            raise ValueError("Edge count does not match the graph metadata")
        if source.dtype.kind not in "iu" or target.dtype.kind not in "iu":
            raise ValueError("Edge indices must be integers")
        if (
            (source < 0).any()
            or (target < 0).any()
            or (source >= size).any()
            or (target >= size).any()
        ):
            raise ValueError("An edge points outside the selected neurons")
        if len(set(zip(source.tolist(), target.tolist()))) != len(source):
            raise ValueError("The graph contains duplicate directed edges")
        if recurrent.shape != (size, size) or not np.isfinite(recurrent).all():
            raise ValueError("Invalid recurrent matrix")
        expected = np.zeros_like(recurrent)
        expected[target, source] = recurrent[target, source]
        if not np.array_equal(expected, recurrent) or np.count_nonzero(
            recurrent
        ) != len(source):
            raise ValueError("Recurrent matrix does not use exactly the listed edges")
        if (
            data["input_weights"].shape != (size, 92)
            or not np.isfinite(data["input_weights"]).all()
        ):
            raise ValueError("Invalid game-to-neuron input projection")
    return {
        "dataset": metadata["dataset"],
        "neurons": size,
        "directed_edges": len(source),
        "graph_sha256": sha256(graph),
        "matrix_matches_edge_list": True,
    }


def verify_original_sources(data_directory, graph=GRAPH):
    """Match the selected neuron IDs and every edge count to the original CSVs.

    This optional check reads the large connection export once. The usual
    CHECK_BRAIN command does not require downloading the original dataset.
    """
    graph = Path(graph)
    metadata = json.loads(graph.with_suffix(".json").read_text(encoding="utf-8"))
    data_directory = Path(data_directory)
    neurons_file = data_directory / "neurons.csv.gz"
    connections_file = data_directory / "connections_princeton.csv.gz"
    for file, key in (
        (neurons_file, "neurons_sha256"),
        (connections_file, "connections_sha256"),
    ):
        if not file.is_file() or sha256(file) != metadata[key]:
            raise ValueError(
                "Original export missing or checksum mismatch: " + file.name
            )
    with np.load(graph, allow_pickle=False) as data:
        selected = data["root_ids"].tolist()
        indices = {root: index for index, root in enumerate(selected)}
        found = set()
        for row in rows(neurons_file):
            root = row["Root ID"]
            if root in indices:
                index = indices[root]
                neurotransmitter = row["Verified NT type"] or row["Predicted NT type"]
                if (
                    row["Super Class"] != data["classes"][index]
                    or neurotransmitter != data["neurotransmitters"][index]
                ):
                    raise ValueError("Neuron annotation differs from original: " + root)
                found.add(root)
        if found != set(selected):
            raise ValueError("Selected neuron ID missing from original table")
        counts = defaultdict(int)
        row_count = 0
        for row in rows(connections_file):
            row_count += 1
            pre, post = row["pre_root_id"], row["post_root_id"]
            count = int(row["syn_count"])
            if (
                pre in indices
                and post in indices
                and count >= metadata["min_synapses_per_neuropil_row"]
            ):
                counts[(indices[pre], indices[post])] += count
        expected = dict(
            zip(
                zip(data["edge_source"].tolist(), data["edge_target"].tolist()),
                data["edge_synapses"].tolist(),
            )
        )
        if counts != expected or row_count != metadata["source_connection_rows"]:
            raise ValueError("Selected connections differ from the original export")
        # Rebuild the simplified signs and normalization, not just the edge IDs.
        signs = np.array(
            [
                -1.0 if nt in {"GABA", "HIST"} else 1.0
                for nt in data["neurotransmitters"]
            ]
        )
        recurrent = np.zeros_like(data["recurrent"])
        for (pre, post), count in counts.items():
            recurrent[post, pre] = np.log1p(count) * signs[pre]
        recurrent *= 0.85 / np.maximum(
            np.sum(np.abs(recurrent), axis=1, keepdims=True), 1.0
        )
        if not np.array_equal(recurrent, data["recurrent"]):
            raise ValueError(
                "Normalized recurrent weights differ from the original counts"
            )
    return {
        "original_exports_checked": True,
        "matched_neuron_ids": len(found),
        "matched_directed_edges": len(counts),
        "connection_rows_read": row_count,
        "normalized_weights_match": True,
        "neurons_sha256": metadata["neurons_sha256"],
        "connections_sha256": metadata["connections_sha256"],
    }


def wiring_experiment(policy):
    """Compare identical inputs with real recurrent weights and zero weights."""
    rng = np.random.default_rng(61)
    observations = rng.uniform(-0.5, 0.5, size=(16, SENSOR_COUNT))
    connected_activity = np.zeros(policy.nodes)
    disconnected_activity = np.zeros(policy.nodes)
    disconnected_weights = np.zeros_like(policy.edge_weights)
    activity_changes = []
    probability_changes = []
    for observation in observations:
        connected, connected_activity, _ = policy.forward(
            observation, connected_activity
        )
        disconnected, disconnected_activity, _ = policy.forward(
            observation, disconnected_activity, weights=disconnected_weights
        )
        activity_changes.extend(np.abs(connected_activity - disconnected_activity))
        connected_logits = connected @ policy.actor
        disconnected_logits = disconnected @ policy.actor
        offset = 0
        for choices in HEADS:
            end = offset + choices
            connected_probabilities = softmax(connected_logits[offset:end])
            disconnected_probabilities = softmax(disconnected_logits[offset:end])
            probability_changes.extend(
                np.abs(connected_probabilities - disconnected_probabilities)
            )
            offset = end
    maximum = float(np.max(probability_changes))
    return {
        "synthetic_observations": len(observations),
        "mean_neuron_activity_difference": float(np.mean(activity_changes)),
        "maximum_action_probability_difference": maximum,
        "wiring_changes_action_probabilities": maximum > 1e-12,
        "interpretation": "A nonzero difference demonstrates computational use of recurrent edges. It does not show that this wiring is better than random wiring or that Celeste rooms clear.",
    }


def learning_experiment(policy):
    """Run one genuine PPO update on a copy of the model using a toy task."""
    probe = copy.deepcopy(policy)
    rng = np.random.default_rng(72)
    brain = PlasticState(probe)
    batch = {
        key: []
        for key in (
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
    for observation in rng.uniform(-0.5, 0.5, size=(32, SENSOR_COUNT)):
        features = brain.features(observation)
        action, log_probability, predicted_reward = probe.act(
            features, rng, temperature=1.5
        )
        desired_direction = 2 if observation[0] >= 0 else 0
        reward = 1.0 if action[0] == desired_direction else -1.0
        values = {
            "observations": observation,
            "before": brain.last_before,
            "resets": brain.last_reset,
            "actions": action,
            "old_logp": log_probability,
            "advantages": reward - predicted_reward,
            "returns": reward,
            "temperatures": 1.5,
        }
        for key, value in values.items():
            batch[key].append(value)
    batch = {key: np.asarray(values) for key, values in batch.items()}
    original_weights = probe.edge_weights.copy()
    original_actor = probe.actor.copy()
    original_value = probe.value.copy()
    metrics = probe.update_rollouts([batch], epochs=1)
    changed_edges = int(
        np.count_nonzero(np.abs(probe.edge_weights - original_weights) > 1e-12)
    )
    result = {
        "synthetic_decisions": 32,
        "changed_recurrent_edges": changed_edges,
        "actor_changed": not np.array_equal(original_actor, probe.actor),
        "value_changed": not np.array_equal(original_value, probe.value),
        "edge_signs_preserved": bool(
            np.array_equal(np.sign(original_weights), np.sign(probe.edge_weights))
        ),
        "metrics": metrics,
        "interpretation": "This exercises the same optimizer used by LEARN on a disposable model copy. A changed weight verifies training, not an improved game clear rate.",
    }
    return result, probe


def memory_experiment(policy, graph=GRAPH):
    """Save and reload the disposable model and compare its learned arrays."""
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "probe.npz"
        policy.save(checkpoint, policy.graph_hash, {"experiment": "disposable"})
        restored = PlasticPolicy(seed=99, graph=graph)
        restored.load(checkpoint, policy.graph_hash)
        pairs = zip(
            policy.parameters + policy.m + policy.v,
            restored.parameters + restored.m + restored.v,
        )
        exact_arrays = all(np.array_equal(before, after) for before, after in pairs)
        observation = np.random.default_rng(83).uniform(-0.5, 0.5, SENSOR_COUNT)
        original_features = policy.forward(observation, np.zeros(policy.nodes))[0]
        restored_features = restored.forward(observation, np.zeros(policy.nodes))[0]
        same_decision_scores = np.array_equal(
            original_features @ policy.actor, restored_features @ restored.actor
        )
        exact_rng = policy.rng.bit_generator.state == restored.rng.bit_generator.state
    return {
        "parameters_and_optimizer_restored_exactly": exact_arrays,
        "decision_scores_restored_exactly": bool(same_decision_scores),
        "optimizer_step_restored": policy.t == restored.t,
        "model_updates_restored": policy.updates == restored.updates,
        "random_generator_restored": exact_rng,
    }


def run_checks(checkpoint=None, graph=GRAPH, source_data=None):
    """Read a stable saved model and check a copy, leaving training untouched."""
    graph_result = inspect_graph(graph)
    policy = PlasticPolicy(graph=graph)
    metadata = {}
    checkpoint_hash = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        # Copy one committed checkpoint snapshot. Atomic trainer saves can safely
        # replace latest.npz after this copy; this experiment uses the snapshot.
        snapshot = checkpoint.read_bytes()
        import hashlib

        checkpoint_hash = hashlib.sha256(snapshot).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            copy_path = Path(temporary) / "snapshot.npz"
            copy_path.write_bytes(snapshot)
            metadata = policy.load(copy_path, policy.graph_hash)
    wiring = wiring_experiment(policy)
    learning, probe = learning_experiment(policy)
    memory = memory_experiment(probe, graph)
    sources = {
        "original_exports_checked": False,
        "note": "Run with --source-data to compare the original CSV exports; this check alone validates the bundled graph's internal consistency.",
    }
    if source_data is not None:
        sources = verify_original_sources(source_data, graph)
    passed = (
        wiring["wiring_changes_action_probabilities"]
        and learning["changed_recurrent_edges"] > 0
        and learning["actor_changed"]
        and learning["value_changed"]
        and learning["edge_signs_preserved"]
        and all(memory.values())
    )
    return {
        "version": VERSION,
        "passed": bool(passed),
        "model": (
            "saved checkpoint snapshot"
            if checkpoint is not None
            else "fresh seed-17 model"
        ),
        "checkpoint_sha256": checkpoint_hash,
        "saved_decisions": metadata.get("steps", 0),
        "saved_model_updates": policy.updates,
        "saved_changed_edges": policy.diagnostics()["learned_edges"],
        "graph": graph_result,
        "sources": sources,
        "wiring_experiment": wiring,
        "learning_experiment": learning,
        "memory_experiment": memory,
        "gameplay_claim": "No Celeste room clear, biological fidelity, or connectome advantage is established by this offline check.",
    }


def report_text(result):
    """Explain the measured checks in plain language."""
    wiring = result["wiring_experiment"]
    learning = result["learning_experiment"]
    source = result["sources"]
    return "\n".join(
        [
            "Fly Brain Celeste — offline brain check",
            "Result: "
            + (
                "PASS"
                if result["passed"]
                else "CHECK FAILED — inspect the JSON results"
            ),
            "Model checked: " + result["model"],
            f"Graph: {result['graph']['neurons']} neurons / {result['graph']['directed_edges']} directed MCNS edges",
            f"Original CSV exports checked: {source['original_exports_checked']}",
            f"Maximum action probability change with recurrent edges removed: {wiring['maximum_action_probability_difference']:.8f}",
            f"Fly connections changed by a disposable PPO update: {learning['changed_recurrent_edges']}",
            f"Actor changed: {learning['actor_changed']}; value layer changed: {learning['value_changed']}",
            f"Exact save/reload checks: {all(result['memory_experiment'].values())}",
            f"Saved model: {result['saved_decisions']} decisions / {result['saved_model_updates']} updates / {result['saved_changed_edges']} changed edges",
            "",
            "The trainer's checkpoint was read only. Test learning happened on a copy.",
            "A running trainer can continue saving newer checkpoints during this check.",
            wiring["interpretation"],
            result["gameplay_claim"],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Check a new model even if latest.npz exists",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        help="Directory containing the original MCNS CSV exports",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "training/brain-check")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if (
        not args.fresh
        and checkpoint is None
        and (ROOT / "training/latest.npz").is_file()
    ):
        checkpoint = ROOT / "training/latest.npz"
    result = run_checks(checkpoint, source_data=args.source_data)
    args.output.mkdir(parents=True, exist_ok=True)
    text = report_text(result)
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "report.txt").write_text(text + "\n", encoding="utf-8")
    page = args.output / "report.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>Fly brain check</title><style>body{font:18px/1.6 system-ui;max-width:900px;margin:40px auto;padding:24px}pre{white-space:pre-wrap}</style><h1>Offline fly brain check</h1><pre>'
        + html.escape(text)
        + "</pre>",
        encoding="utf-8",
    )
    print(text)
    print("\nReports: " + str(args.output))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
