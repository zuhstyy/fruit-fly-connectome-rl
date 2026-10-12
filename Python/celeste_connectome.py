"""Select real MCNS connections from CSV exports and load the resulting graph.

The trainer uses this graph through PlasticPolicy. The older FlyBrain.choose
method is retained as a historical fixed-readout example; LEARN does not use it.
"""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(data_dir, output, size=256, min_synapses=5, seed=17):
    neurons_file = data_dir / "neurons.csv.gz"
    edges_file = data_dir / "connections_princeton.csv.gz"
    if not neurons_file.exists() or not edges_file.exists():
        raise FileNotFoundError(
            "Expected neurons.csv.gz and connections_princeton.csv.gz in --data"
        )
    neurons = {}
    for row in rows(neurons_file):
        if "Root ID" not in row:
            raise ValueError(
                "Expected the MCNS Codex neuron table with a 'Root ID' column"
            )
        neurons[row["Root ID"]] = row
    if not 32 <= size <= len(neurons):
        raise ValueError("Choose at least 32 nodes, at most the dataset size")
    if len(neurons) != 166700:
        raise ValueError(
            "This release expects the chosen 166,700-neuron MCNS v1.0 table"
        )
    print(
        f"Loaded {len(neurons):,} MCNS neurons. Pass 1/3: rank descending neurons.",
        flush=True,
    )
    # Descending neurons link brain processing to the nerve cord. They are anchors,
    # not proven Celeste motor neurons: the game mapping is an engineering choice.
    descending = {
        root
        for root, row in neurons.items()
        if row["Super Class"] == "descending_neuron"
    }
    scores = defaultdict(int)
    total_edges = 0
    for row in rows(edges_file):
        total_edges += 1
        if not {
            "pre_root_id",
            "post_root_id",
            "syn_count",
            "neuropil",
            "nt_type",
        }.issubset(row):
            raise ValueError("Wrong MCNS connection table columns")
        count = int(row["syn_count"])
        if count < min_synapses:
            continue
        pre, post = row["pre_root_id"], row["post_root_id"]
        if pre in descending:
            scores[pre] += count
        if post in descending:
            scores[post] += count
    if not scores:
        raise ValueError("No descending neuron connectivity found")
    # One anchor gives a connected star; fill with its strongest real neighbors.
    anchor = sorted(scores, key=lambda root: (-scores[root], int(root)))[0]
    neighbors = defaultdict(int)
    print(f"Pass 2/3: gather neighbors of descending neuron {anchor}.", flush=True)
    for row in rows(edges_file):
        count = int(row["syn_count"])
        if count < min_synapses:
            continue
        pre, post = row["pre_root_id"], row["post_root_id"]
        if pre == anchor and post in neurons and post != anchor:
            neighbors[post] += count
        if post == anchor and pre in neurons and pre != anchor:
            neighbors[pre] += count
    if len(neighbors) < size - 1:
        raise ValueError(
            f"Anchor has only {len(neighbors)} eligible neighbors; lower --size"
        )
    selected = [anchor] + sorted(
        neighbors, key=lambda root: (-neighbors[root], int(root))
    )[: size - 1]
    index = {root: i for i, root in enumerate(selected)}
    synapses = defaultdict(int)
    print("Pass 3/3: keep real edges inside the selected subnetwork.", flush=True)
    for row in rows(edges_file):
        count = int(row["syn_count"])
        pre, post = row["pre_root_id"], row["post_root_id"]
        if count >= min_synapses and pre in index and post in index:
            synapses[(index[pre], index[post])] += count
    classes = [neurons[root]["Super Class"] for root in selected]
    nts = [
        neurons[root]["Verified NT type"] or neurons[root]["Predicted NT type"]
        for root in selected
    ]
    # Toy sign model: GABA / histamine are negative; other/unknown NTs positive.
    # Glutamate and modulators are not assigned a universal biological effect.
    # This is a simplified rate network, not a calibrated fly brain simulation.
    signs = np.array([-1.0 if nt in {"GABA", "HIST"} else 1.0 for nt in nts])
    recurrent = np.zeros((size, size), dtype=np.float32)
    for (pre, post), count in synapses.items():
        recurrent[post, pre] = np.log1p(count) * signs[pre]
    recurrent *= 0.85 / np.maximum(
        np.sum(np.abs(recurrent), axis=1, keepdims=True), 1.0
    )
    rng = np.random.default_rng(seed)
    input_nodes = sorted(
        range(size),
        key=lambda i: (
            "sensory" not in classes[i] and "ascending" not in classes[i],
            -float(np.sum(np.abs(recurrent[i]))),
            int(selected[i]),
        ),
    )
    input_weights = np.zeros((size, 92), dtype=np.float32)
    for sensor in range(92):
        chosen = rng.choice(input_nodes[: max(32, size // 2)], size=4, replace=False)
        input_weights[chosen, sensor] = rng.choice([-0.65, 0.65], size=4)
    output_nodes = sorted(
        range(size),
        key=lambda i: (
            classes[i] not in {"descending_neuron", "vnc_motor", "cb_motor"},
            -float(np.sum(np.abs(recurrent[:, i]))),
            int(selected[i]),
        ),
    )[:32]
    source, target = zip(*sorted(synapses))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        recurrent=recurrent,
        input_weights=input_weights,
        output_nodes=np.array(output_nodes),
        root_ids=np.array(selected),
        edge_source=np.array(source),
        edge_target=np.array(target),
        edge_synapses=np.array([synapses[edge] for edge in sorted(synapses)]),
        classes=np.array(classes),
        neurotransmitters=np.array(nts),
    )
    metadata = {
        "dataset": "MCNS v1.0",
        "sex": "male",
        "scope": "brain and nerve cord",
        "source_neurons": len(neurons),
        "source_connection_rows": total_edges,
        "subnetwork_neurons": size,
        "subnetwork_directed_edges": len(synapses),
        "anchor_root_id": anchor,
        "min_synapses_per_neuropil_row": min_synapses,
        "seed": seed,
        "neurons_sha256": sha256(neurons_file),
        "connections_sha256": sha256(edges_file),
        "selection": "highest weighted-degree descending anchor and its strongest direct neighbors",
        "note": "Real induced MCNS topology; synthetic game sensors/actions and simplified rate dynamics.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {size} neurons / {len(synapses)} directed edges to {output}", flush=True
    )
    return metadata


class FlyBrain:
    def __init__(self, graph=ROOT / "data" / "mcns_subnetwork.npz", seed=17):
        with np.load(graph, allow_pickle=False) as data:
            self.recurrent = data["recurrent"].copy()
            self.inputs = data["input_weights"].copy()
            self.outputs = data["output_nodes"].copy()
            self.root_ids = data["root_ids"].copy()
            self.edge_source = data["edge_source"].copy()
            self.edge_target = data["edge_target"].copy()
        self.edge_weights = self.recurrent[self.edge_target, self.edge_source]
        self.graph_hash = sha256(Path(graph))
        self.activity = np.zeros(len(self.root_ids), dtype=np.float32)
        self.weights = (
            np.random.default_rng(seed)
            .normal(0, 0.15, (20, len(self.outputs) + 1))
            .astype(np.float32)
        )

    def reset(self):
        self.activity.fill(0)

    def choose(self, observation):
        sensors = np.asarray(observation, dtype=np.float32)
        if sensors.shape != (92,):
            raise ValueError("Expected 92 sensors")
        drive = np.einsum("ij,j->i", self.inputs, sensors, optimize=False)
        for _ in range(3):
            messages = np.bincount(
                self.edge_target,
                weights=self.edge_weights * self.activity[self.edge_source],
                minlength=len(self.activity),
            ).astype(np.float32)
            self.activity = 0.65 * self.activity + 0.35 * np.tanh(messages + drive)
        motor = np.concatenate(
            [self.activity[self.outputs], np.ones(1, dtype=np.float32)]
        )
        return int(np.argmax(np.einsum("ij,j->i", self.weights, motor, optimize=False)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                weights=self.weights,
                graph_sha256=np.array(self.graph_hash),
                protocol=np.array(1),
            )
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["graph_sha256"]) != self.graph_hash:
                raise ValueError("Model was trained with a different MCNS graph")
            if (
                data["weights"].shape != self.weights.shape
                or not np.isfinite(data["weights"]).all()
            ):
                raise ValueError("Invalid readout weights")
            self.weights = data["weights"].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "mcns_subnetwork.npz"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min-synapses", type=int, default=5)
    args = parser.parse_args()
    prepare(args.data, args.output, args.size, args.min_synapses)
