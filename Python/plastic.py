"""PPO with truncated backpropagation through actual MCNS recurrent edges.

NumPy with optional CUDA gradients. The biological graph is unchanged; every existing edge has a
trainable log-gain. No new recurrent connections or sign flips are introduced.
The game input mapping is synthetic and fixed. This is not a biological brain
simulation: it is a trainable recurrent model constrained by its connectome.
"""

from pathlib import Path
import json
import numpy as np
from celeste_connectome import FlyBrain
from learning import Policy, softmax, replace_checkpoint
from protocol import (
    HEADS,
    SCHEMA,
    LEGACY_SCHEMA,
    DURATIONS,
    SENSOR_COUNT,
    LEGACY_SENSORS,
)

LEGACY_PLASTIC_SCHEMA = "mcns-plastic-v1-all256-loggain-bptt16"
PREVIOUS_PLASTIC_SCHEMA = "mcns-plastic-v2-duration7-bptt16"
PLASTIC_SCHEMA = "mcns-plastic-v3-context156-imitation-bptt32"


class PlasticPolicy(Policy):
    """Turn game observations into actions using the selected fly wiring.

    ``actor`` chooses buttons, ``value`` predicts future reward, and ``theta``
    adjusts the strength of each existing fly connection. Those three arrays
    learn together. A worker's temporary neuron activity lives in PlasticState.
    """

    def __init__(self, seed=17, graph=None):
        brain = FlyBrain(seed=seed) if graph is None else FlyBrain(graph, seed)
        # The original 92-sensor mapping is retained. Extra state/object
        # sensors enter the SAME fly neurons through a deterministic projection.
        extra_rng = np.random.default_rng(400017)
        extra = extra_rng.normal(
            0, 0.05, (len(brain.root_ids), SENSOR_COUNT - LEGACY_SENSORS)
        )
        self.inputs = np.concatenate([brain.inputs.astype(np.float64), extra], axis=1)
        self.source, self.target = brain.edge_source.copy(), brain.edge_target.copy()
        self.base = brain.edge_weights.astype(np.float64)
        self.nodes = len(brain.root_ids)
        self.graph_hash = brain.graph_hash
        super().__init__(features=self.nodes + 1, seed=seed)
        self.actor = self.actor.astype(np.float64)
        self.value = self.value.astype(np.float64)
        # Rare dash/grab/talk avoids random button spam; direction is not scripted.
        self.actor[-1, 6:8] = [0.0, -1.1]
        self.actor[-1, 8:10] = [0.0, -3.0]
        self.actor[-1, 10:12] = [0.0, -2.0]
        self.actor[-1, 12:14] = [0.0, -4.0]
        self.initialize_duration()
        self.theta = np.zeros(len(self.base), dtype=np.float64)
        self.m = [np.zeros_like(p) for p in self.parameters]
        self.v = [np.zeros_like(p) for p in self.parameters]
        self.updates = 0
        self.gradient_backend = None

    def initialize_duration(self):
        # Begin close to the successful four-frame controller; PPO can learn
        # shorter taps and longer runs/holds without discarding its movement.
        self.actor[:, 14:] = 0.0
        self.actor[-1, 14:] = [-2.0, -1.0, 1.5, -0.5, -1.5, -2.5, -3.0]

    @property
    def parameters(self):
        return [self.actor, self.value, self.theta]

    @property
    def edge_weights(self):
        return self.base * np.exp(self.theta)

    def forward(self, observation, activity, weights=None):
        """Run one observation through three recurrent neuron updates.

        The optional weights argument is used by CHECK_BRAIN to compare the
        same model with its recurrent connections temporarily disconnected.
        It does not replace or modify the model's stored weights.
        """
        observation = np.asarray(observation, dtype=np.float64)
        observation = self.pad_observations(observation)
        neuron_activity = np.asarray(activity, dtype=np.float64).copy()
        if (
            neuron_activity.shape != (self.nodes,)
            or not np.isfinite(neuron_activity).all()
        ):
            raise ValueError("Invalid recurrent activity")
        connection_weights = self.edge_weights if weights is None else weights
        sensor_input = np.einsum("ij,j->i", self.inputs, observation, optimize=False)
        trace = []
        for _ in range(3):
            # Each source neuron sends its activity along the real edge list.
            messages = connection_weights * neuron_activity[self.source]
            incoming = np.bincount(self.target, weights=messages, minlength=self.nodes)
            activated = np.tanh(incoming + sensor_input)
            trace.append((neuron_activity, activated))
            # Retain some previous activity: this is the short-term memory.
            neuron_activity = 0.65 * neuron_activity + 0.35 * activated
        # The final 1 is a bias input for the action and reward-prediction layers.
        features = np.append(neuron_activity, 1.0)
        return features, neuron_activity, trace

    @staticmethod
    def pad_observations(observation):
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape[-1] == LEGACY_SENSORS:
            padding = [(0, 0)] * observation.ndim
            padding[-1] = (0, SENSOR_COUNT - LEGACY_SENSORS)
            observation = np.pad(observation, padding)
        if observation.shape[-1] != SENSOR_COUNT or not np.isfinite(observation).all():
            raise ValueError(
                "Expected 156 finite sensors (or a compatible 92-sensor record)"
            )
        return observation

    def act(self, f, rng, temperature=1.0):
        """Sample six button choices and one duration from neuron activity."""
        if not np.isfinite(temperature) or not 1 <= temperature <= 3:
            raise ValueError("Invalid exploration temperature")
        logits = f @ self.actor / temperature
        actions = []
        log_probability = 0.0
        offset = 0
        for choices in HEADS:
            probabilities = softmax(logits[offset : offset + choices])
            choice = int(rng.choice(choices, p=probabilities / probabilities.sum()))
            actions.append(choice)
            log_probability += float(np.log(max(probabilities[choice], 1e-20)))
            offset += choices
        predicted_reward = float(f @ self.value)
        return np.asarray(actions, dtype=np.int64), log_probability, predicted_reward

    def sequence_gradients(self, segments, entropy=0.02, objective="ppo"):
        if self.gradient_backend is not None:
            return self.gradient_backend.gradients(segments, entropy, objective)
        return self.numpy_sequence_gradients(segments, entropy, objective)

    def numpy_sequence_gradients(self, segments, entropy=0.02, objective="ppo"):
        """Find how each trainable array contributed to the PPO loss.

        Time runs forward to collect neuron states, then backward to assign
        credit. Reset markers prevent credit from crossing a death or restart.
        Several independent sequences are batched for speed; they never share
        their neuron activity. See docs/LEARNING.md for the equations.
        """
        # Vectorize independent recurrent sequences while retaining the time axis
        # and every reset boundary. This computes the same truncated BPTT loss.
        weights = self.edge_weights
        count = len(segments)
        length = max(len(s["observations"]) for s in segments)
        observations = np.zeros((count, length, SENSOR_COUNT))
        resets = np.ones((count, length), dtype=bool)
        valid = np.zeros((count, length), dtype=bool)
        for i, s in enumerate(segments):
            n = len(s["observations"])
            observations[i, :n] = self.pad_observations(s["observations"])
            resets[i, :n] = s["resets"]
            valid[i, :n] = True
        h = np.asarray([s["initial"] for s in segments]).copy()
        features = np.zeros((count, length, self.nodes + 1))
        traces = []
        matrix = np.zeros((self.nodes, self.nodes))
        np.add.at(matrix, (self.target, self.source), weights)
        for t in range(length):
            h[resets[:, t]] = 0
            drive = observations[:, t] @ self.inputs.T
            local = []
            for _ in range(3):
                z = h @ matrix.T + drive
                activated = np.tanh(z)
                local.append((h, activated))
                h = 0.65 * h + 0.35 * activated
            features[:, t, : self.nodes] = h
            features[:, t, -1] = 1
            traces.append(local)

        def concatenate(key):
            return np.concatenate([segment[key] for segment in segments])

        if objective == "imitation":
            loss, ga, gv, kl, ent, dx = self.imitation_gradients(
                features[valid], concatenate("actions")
            )
        else:
            loss, ga, gv, kl, ent, dx = super().gradients(
                features[valid],
                concatenate("actions"),
                concatenate("old_logp"),
                concatenate("advantages"),
                concatenate("returns"),
                entropy=entropy,
                temperatures=concatenate("temperatures"),
                feature_gradient=True,
            )
        # ga/gv: actor/value gradients; dx: gradient arriving at neuron outputs.
        derivatives = np.zeros((count, length, self.nodes))
        derivatives[valid] = dx[:, : self.nodes]
        gt = np.zeros_like(self.theta)
        dh = np.zeros_like(h)
        # gt accumulates the gradient for each fly connection's log-gain.
        for t in range(length - 1, -1, -1):
            dh += derivatives[:, t]
            for before, activated in reversed(traces[t]):
                dz = 0.35 * dh * (1 - activated**2)
                gt += np.sum(
                    dz[:, self.target] * before[:, self.source] * weights, axis=0
                )
                dh = 0.65 * dh + dz @ matrix
            dh[resets[:, t]] = 0
        return loss, [ga, gv, gt], kl, ent

    def update_rollouts(
        self, batches, epochs=2, sequence_length=32, sequences_per_batch=8, rate=1e-3
    ):
        """Learn from completed worker decisions without changing the wiring."""
        advantages = np.concatenate([b["advantages"] for b in batches])
        mean = float(advantages.mean())
        scale = max(float(advantages.std()), 1e-6)
        segments = []
        for b in batches:
            for start in range(0, len(b["actions"]), sequence_length):
                end = start + sequence_length
                s = {
                    k: np.asarray(b[k][start:end])
                    for k in [
                        "observations",
                        "resets",
                        "actions",
                        "old_logp",
                        "advantages",
                        "returns",
                        "temperatures",
                    ]
                }
                s["advantages"] = (s["advantages"] - mean) / scale
                s["initial"] = b["before"][start].copy()
                segments.append(s)
        metrics = []
        for _ in range(epochs):
            order = self.rng.permutation(len(segments))
            for start in range(0, len(order), sequences_per_batch):
                sample = [
                    segments[i] for i in order[start : start + sequences_per_batch]
                ]
                loss, grads, kl, ent = self.sequence_gradients(sample)
                if not np.isfinite(loss) or any(
                    not np.isfinite(g).all() for g in grads
                ):
                    raise FloatingPointError(
                        "Invalid recurrent gradient; last saved checkpoint retained"
                    )
                self.apply_gradients(grads, rate)
                metrics.append((loss, kl, ent))
                if kl > 0.03:
                    break
            if metrics[-1][1] > 0.03:
                break
        self.updates += 1
        return dict(
            zip(("loss", "kl", "entropy"), map(float, np.mean(metrics, axis=0)))
        )

    def apply_gradients(self, grads, rate):
        norm = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
        factor = min(1.0, 0.5 / max(norm, 1e-12))
        self.t += 1
        for j, (parameter, gradient) in enumerate(zip(self.parameters, grads)):
            gradient = gradient * factor
            self.m[j] = 0.9 * self.m[j] + 0.1 * gradient
            self.v[j] = 0.999 * self.v[j] + 0.001 * gradient * gradient
            parameter -= (
                rate
                * (self.m[j] / (1 - 0.9**self.t))
                / (np.sqrt(self.v[j] / (1 - 0.999**self.t)) + 1e-8)
            )
        np.clip(self.theta, np.log(0.25), np.log(1.1), out=self.theta)

    def imitation_gradients(self, features, actions):
        """Supervise successful inputs without misusing PPO's old log-probs."""
        logits = features @ self.actor
        derivative = np.zeros_like(logits)
        loss, entropy, offset = 0.0, 0.0, 0
        for head, choices in enumerate(HEADS):
            probability = softmax(logits[:, offset : offset + choices])
            logs = np.log(np.maximum(probability, 1e-20))
            labels = actions[:, head]
            loss -= float(logs[np.arange(len(features)), labels].mean()) / len(HEADS)
            entropy -= float((probability * logs).sum(axis=1).mean())
            probability[np.arange(len(features)), labels] -= 1
            derivative[:, offset : offset + choices] = probability / (
                len(features) * len(HEADS)
            )
            offset += choices
        return (
            loss,
            features.T @ derivative,
            np.zeros_like(self.value),
            0.0,
            entropy,
            derivative @ self.actor.T,
        )

    def imitate(self, trajectories, decisions=128, rate=3e-4):
        """Reconstruct current recurrent activity, then learn from successes.

        Each training fragment gets its actual preceding activity under the
        current weights. Old hidden states are not blindly replayed after updates.
        """
        if decisions <= 0:
            return dict(imitation_loss=0.0, imitation_decisions=0)
        segments = []
        remaining = decisions
        for trajectory in trajectories:
            observations = self.pad_observations(trajectory["observations"])
            actions = trajectory["actions"]
            resets = trajectory["resets"]
            if not len(actions):
                continue
            length = min(32, remaining, len(actions))
            start = int(self.rng.integers(0, len(actions) - length + 1))
            activity = np.zeros(self.nodes)
            for i in range(start):
                if resets[i]:
                    activity[:] = 0
                _, activity, _ = self.forward(observations[i], activity)
            segments.append(
                dict(
                    observations=observations[start : start + length],
                    actions=actions[start : start + length],
                    resets=resets[start : start + length],
                    initial=activity,
                )
            )
            remaining -= length
            if remaining <= 0:
                break
        if not segments:
            return dict(imitation_loss=0.0, imitation_decisions=0)
        loss, grads, _, _ = self.sequence_gradients(segments, objective="imitation")
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads):
            raise FloatingPointError("Invalid imitation gradient")
        self.apply_gradients(grads, rate)
        return dict(
            imitation_loss=loss,
            imitation_decisions=sum(len(s["actions"]) for s in segments),
        )

    def diagnostics(self):
        change = np.abs(np.expm1(self.theta))
        return dict(
            brain_updates=self.updates,
            learned_edges=int(np.count_nonzero(change > 1e-7)),
            brain_change=float(change.mean()),
        )

    def save(self, path, graph_hash, metadata):
        if graph_hash != self.graph_hash:
            raise ValueError("Wrong graph")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                actor=self.actor,
                value=self.value,
                theta=self.theta,
                m0=self.m[0],
                m1=self.m[1],
                m2=self.m[2],
                v0=self.v[0],
                v1=self.v[1],
                v2=self.v[2],
                t=self.t,
                updates=self.updates,
                graph_hash=np.array(graph_hash),
                schema=np.array(SCHEMA),
                plastic_schema=np.array(PLASTIC_SCHEMA),
                metadata=np.array(json.dumps(metadata)),
                rng=np.array(json.dumps(self.rng.bit_generator.state)),
            )
        replace_checkpoint(temp, path)

    def load(self, path, graph_hash):
        with np.load(path, allow_pickle=False) as d:
            if "plastic_schema" not in d:
                raise ValueError(
                    "This is an old fixed-network model. Press LEARN to back it up and start trainable connections."
                )
            migrating = (
                str(d["schema"]) == LEGACY_SCHEMA
                and str(d["plastic_schema"]) == LEGACY_PLASTIC_SCHEMA
            )
            context_migration = (
                str(d["plastic_schema"]) == PREVIOUS_PLASTIC_SCHEMA
                and str(d["schema"]) == SCHEMA
            )
            if (
                str(d["graph_hash"]) != graph_hash
                or graph_hash != self.graph_hash
                or (
                    not migrating
                    and not context_migration
                    and (
                        str(d["schema"]) != SCHEMA
                        or str(d["plastic_schema"]) != PLASTIC_SCHEMA
                    )
                )
            ):
                raise ValueError("Incompatible recurrent checkpoint")
            loaded = []
            for key, p in zip(["actor", "value", "theta"], self.parameters):
                shape = (
                    (self.nodes + 1, 14) if migrating and key == "actor" else p.shape
                )
                if d[key].shape != shape or not np.isfinite(d[key]).all():
                    raise ValueError("Invalid checkpoint " + key)
                if migrating and key == "actor":
                    expanded = self.actor.copy()
                    expanded[:, :14] = d[key]
                    loaded.append(expanded)
                else:
                    loaded.append(d[key].copy())
            moments = []
            for prefix in ["m", "v"]:
                arrays = []
                for j, p in enumerate(self.parameters):
                    a = d[prefix + str(j)]
                    shape = (self.nodes + 1, 14) if migrating and j == 0 else p.shape
                    if (
                        a.shape != shape
                        or not np.isfinite(a).all()
                        or (prefix == "v" and (a < 0).any())
                    ):
                        raise ValueError("Invalid optimizer checkpoint")
                    if migrating and j == 0:
                        expanded = np.zeros_like(p)
                        expanded[:, :14] = a
                        arrays.append(expanded)
                    else:
                        arrays.append(a.copy())
                moments.append(arrays)
            if np.any(loaded[2] < np.log(0.25) - 1e-8) or np.any(
                loaded[2] > np.log(1.1) + 1e-8
            ):
                raise ValueError("Recurrent gain outside stable bounds")
            self.actor, self.value, self.theta = loaded
            self.m, self.v = moments
            self.t = int(d["t"])
            self.updates = int(d["updates"])
            self.rng.bit_generator.state = json.loads(str(d["rng"]))
            metadata = json.loads(str(d["metadata"]))
            if migrating:
                metadata["migrated_from"] = "3.0.0"
            if context_migration:
                metadata["migrated_from"] = "3.1/3.2"
            return metadata


class PlasticState:
    """Per-worker activity; learned parameters are shared and updated between rollouts."""

    def __init__(self, policy):
        self.policy = policy
        self.outputs = np.arange(policy.nodes)
        self.graph_hash = policy.graph_hash
        self.reset()

    def reset(self):
        self.activity = np.zeros(self.policy.nodes)
        self.reset_pending = True

    def features(self, observation):
        self.last_observation = np.asarray(observation, dtype=np.float64).copy()
        self.last_before = self.activity.copy()
        self.last_reset = self.reset_pending
        self.reset_pending = False
        f, self.activity, _ = self.policy.forward(
            self.last_observation, self.last_before
        )
        return f

    def refresh(self):
        # The bootstrap observation was encoded before the policy update.
        # Re-encode it once with the new weights, not twice into the memory state.
        f, self.activity, _ = self.policy.forward(
            self.last_observation, self.last_before
        )
        return f
