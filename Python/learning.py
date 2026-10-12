"""PPO objective, reward and legacy-readout utilities.

The active learner is PlasticPolicy in plastic.py, which trains recurrent MCNS
connection gains with truncated BPTT as well as its actor/value layers.
"""

from pathlib import Path
import json
import time
import numpy as np
from celeste_connectome import FlyBrain
from protocol import HEADS, SCHEMA


def replace_checkpoint(temporary, destination):
    """Commit a completed save, retrying a brief Windows reader lock."""
    for attempt in range(5):
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Reservoir(FlyBrain):
    """Old fixed-connection encoder, retained for compatibility tests only.

    Current training uses PlasticPolicy and PlasticState, not this class.
    """

    def features(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape == (156,):
            observation = observation[:92]
        if observation.shape != (92,) or not np.isfinite(observation).all():
            raise ValueError("Expected 92 finite sensors")
        drive = np.einsum("ij,j->i", self.inputs, observation, optimize=False)
        for _ in range(3):
            messages = np.bincount(
                self.edge_target,
                weights=self.edge_weights * self.activity[self.edge_source],
                minlength=len(self.activity),
            ).astype(np.float32)
            self.activity = 0.65 * self.activity + 0.35 * np.tanh(messages + drive)
        return np.append(self.activity[self.outputs], np.float32(1)).astype(np.float32)


class Policy:
    def __init__(self, features=33, seed=17):
        self.rng = np.random.default_rng(seed)
        self.actor = self.rng.normal(0, 0.02, (features, sum(HEADS))).astype(np.float32)
        self.value = np.zeros(features, dtype=np.float32)
        self.m = [np.zeros_like(self.actor), np.zeros_like(self.value)]
        self.v = [np.zeros_like(self.actor), np.zeros_like(self.value)]
        self.t = 0

    def act(self, f, rng):
        a, logp, offset = [], 0.0, 0
        logits = f @ self.actor
        for n in HEADS:
            p = softmax(logits[offset : offset + n])
            choice = int(rng.choice(n, p=p.astype(float) / p.astype(float).sum()))
            a.append(choice)
            logp += float(np.log(max(p[choice], 1e-20)))
            offset += n
        return np.array(a, dtype=np.int64), logp, float(f @ self.value)

    def gradients(
        self,
        x,
        actions,
        old_logp,
        advantages,
        returns,
        clip=0.2,
        entropy=0.01,
        temperatures=None,
        feature_gradient=False,
    ):
        temperature = (
            np.ones((len(x), 1))
            if temperatures is None
            else np.asarray(temperatures).reshape(-1, 1)
        )
        logits = (x @ self.actor) / temperature
        parts, offset = [], 0
        logp = np.zeros(len(x), dtype=np.float64)
        ent = np.zeros(len(x), dtype=np.float64)
        for h, n in enumerate(HEADS):
            p = softmax(logits[:, offset : offset + n])
            logs = np.log(np.maximum(p, 1e-20))
            logp += logs[np.arange(len(x)), actions[:, h]]
            ent -= np.sum(p * logs, axis=1)
            parts.append((p, logs, h, offset, n))
            offset += n
        ratio = np.exp(np.clip(logp - old_logp, -20, 20))
        clipped = np.clip(ratio, 1 - clip, 1 + clip)
        # Stop the surrogate gradient only on the clipped side of the objective.
        enabled = ~(
            ((advantages >= 0) & (ratio > 1 + clip))
            | ((advantages < 0) & (ratio < 1 - clip))
        )
        factor = -advantages * ratio * enabled / len(x)
        dl = np.zeros_like(logits)
        for p, logs, h, off, n in parts:
            one = np.zeros_like(p)
            one[np.arange(len(x)), actions[:, h]] = 1
            dl[:, off : off + n] = factor[:, None] * (one - p)
            # d(-entropy)/d(logits) = p*(log(p)-E[log(p)]).
            dl[:, off : off + n] += (
                entropy / len(x) * p * (logs - (p * logs).sum(axis=1, keepdims=True))
            )
        dl /= temperature
        prediction = x @ self.value
        diff = prediction - returns
        ga = x.T @ dl
        gv = x.T @ diff / len(x)
        loss = (
            -np.minimum(ratio * advantages, clipped * advantages).mean()
            - entropy * ent.mean()
            + 0.5 * np.mean(diff**2)
        )
        result = (
            float(loss),
            ga,
            gv,
            float(np.mean(old_logp - logp)),
            float(ent.mean()),
        )
        if feature_gradient:
            return (
                *result,
                dl @ self.actor.T + diff[:, None] * self.value[None, :] / len(x),
            )
        return result

    def update(self, data, epochs=4, batch_size=256, rate=3e-4):
        x, a, lp, adv, ret = data
        adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
        metrics = []
        for _ in range(epochs):
            ids = self.rng.permutation(len(x))
            for i in range(0, len(x), batch_size):
                ix = ids[i : i + batch_size]
                loss, ga, gv, kl, ent = self.gradients(
                    x[ix], a[ix], lp[ix], adv[ix], ret[ix]
                )
                if (
                    not np.isfinite(loss)
                    or not np.isfinite(ga).all()
                    or not np.isfinite(gv).all()
                ):
                    raise FloatingPointError(
                        "Non-finite PPO update; last good checkpoint retained"
                    )
                norm = float(np.sqrt((ga * ga).sum() + (gv * gv).sum()))
                scale = min(1.0, 0.5 / max(norm, 1e-9))
                self.t += 1
                for j, (parameter, gradient) in enumerate(
                    [(self.actor, ga), (self.value, gv)]
                ):
                    gradient = gradient * scale
                    self.m[j] = 0.9 * self.m[j] + 0.1 * gradient
                    self.v[j] = 0.999 * self.v[j] + 0.001 * gradient**2
                    parameter -= (
                        rate
                        * (self.m[j] / (1 - 0.9**self.t))
                        / (np.sqrt(self.v[j] / (1 - 0.999**self.t)) + 1e-8)
                    )
                metrics.append((loss, kl, ent))
            if metrics[-1][1] > 0.03:
                break
        return dict(
            zip(("loss", "kl", "entropy"), map(float, np.mean(metrics, axis=0)))
        )

    def save(self, path, graph_hash, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(
                f,
                actor=self.actor,
                value=self.value,
                m0=self.m[0],
                m1=self.m[1],
                v0=self.v[0],
                v1=self.v[1],
                t=self.t,
                graph_hash=np.array(graph_hash),
                schema=np.array(SCHEMA),
                metadata=np.array(json.dumps(metadata)),
                rng=np.array(json.dumps(self.rng.bit_generator.state)),
            )
        replace_checkpoint(temp, path)

    def load(self, path, graph_hash):
        with np.load(path, allow_pickle=False) as d:
            if str(d["graph_hash"]) != graph_hash or str(d["schema"]) != SCHEMA:
                raise ValueError(
                    "Checkpoint graph or action schema mismatch; refusing to resume"
                )
            shapes = dict(
                actor=self.actor.shape,
                value=self.value.shape,
                m0=self.actor.shape,
                m1=self.value.shape,
                v0=self.actor.shape,
                v1=self.value.shape,
            )
            for k, shape in shapes.items():
                if d[k].shape != shape or not np.isfinite(d[k]).all():
                    raise ValueError("Invalid checkpoint array: " + k)
            self.actor, self.value = d["actor"].copy(), d["value"].copy()
            self.m = [d["m0"].copy(), d["m1"].copy()]
            self.v = [d["v0"].copy(), d["v1"].copy()]
            self.t = int(d["t"])
            self.rng.bit_generator.state = json.loads(str(d["rng"]))
            return json.loads(str(d["metadata"]))


def gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95, durations=None):
    """Estimate whether an action turned out better than the model expected.

    Later rewards can credit earlier actions. A terminal marker stops that
    credit, and longer actions receive more elapsed-time discounting.
    """
    # Semi-Markov credit assignment: discounts use actual elapsed action time.
    # The old four-frame decision is the reference unit, preserving its horizon.
    elapsed = (
        np.ones(len(rewards))
        if durations is None
        else np.asarray(durations, dtype=float) / 4.0
    )
    if (
        elapsed.shape != (len(rewards),)
        or not np.isfinite(elapsed).all()
        or (elapsed < 0).any()
    ):
        raise ValueError("Invalid elapsed action frames")
    adv = np.zeros(len(rewards), dtype=np.float32)
    tail = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nxt = bootstrap if t == len(rewards) - 1 else values[t + 1]
        alive = 1 - float(dones[t])
        discount, trace = gamma ** elapsed[t], lam ** elapsed[t]
        delta = rewards[t] + discount * nxt * alive - values[t]
        tail = delta + discount * trace * alive * tail
        adv[t] = tail
    return adv, adv + np.asarray(values, dtype=np.float32)


class Reward:
    """Score progress toward the exit and real game-reported transitions.

    Score is the current attempt's actual rewards. Death/backtracking penalties
    are flat: earned progress remains distinguishable from standing still.
    A real respawn begins a new zero score. Stalling never forces a death.
    """

    REVISION = 8

    def __init__(self):
        self.chapter = None
        self.room_order = {}
        self.awarded_rooms = set()
        self.reset()

    def track_room(self, room):
        if room[0] != self.chapter:
            self.chapter = room[0]
            self.room_order = {}
            self.awarded_rooms = set()
        if room not in self.room_order:
            self.room_order[room] = len(self.room_order)

    def reset(self, state=None, fresh_attempt=True):
        if state is not None:
            room = (int(state["chapter"]), state["room"])
            self.track_room(room)
            if state.get("navigation_revision") == 1 and fresh_attempt:
                self.awarded_rooms = {room}
            else:
                self.awarded_rooms.add(room)
        self.steps_without_progress = 0
        self.distance_reward = 0.0
        self.score = 0.0
        self.last = 0.0
        self.event = "Attempt reset to zero"
        self.best_distance = {}

    def finish(self, reward, extra=0.0):
        # Record the exact reward used by PPO, so the HUD shows the same value.
        self.score += reward + extra
        self.last = float(reward + extra)
        return self.last

    def potential(self, state):
        if (
            state.get("dead")
            or state.get("completed")
            or not state.get("navigation_valid")
        ):
            return 0.0
        distance = max(0.0, float(state["navigation_distance"]))
        return 4.0 / (1.0 + distance / 80.0)

    def route_step(self, old, new, frames):
        """Discounted potential shaping; idle and closed loops cannot farm it.

        Positive potential makes an unchanged state receive a small negative
        shaping reward. Death and completion have zero terminal potential.
        Artificial practice resets are episode boundaries, never reward events.
        """
        self.steps_without_progress += frames / 4.0
        self.distance_reward = 0.0
        self.event = "Time cost / no new progress"
        room = (int(new["chapter"]), new["room"])
        changed = (old["chapter"], old["room"]) != room
        reward = -0.01 * frames / 4.0
        # This remains the same potential function across genuine room changes.
        self.distance_reward = 0.99 ** (frames / 4.0) * self.potential(
            new
        ) - self.potential(old)
        reward += self.distance_reward
        if new["completed"]:
            self.event = "Game-reported chapter completion +25"
            return self.finish(reward + 25)
        if new["dead"]:
            self.event = "Game-reported death -2"
            return self.finish(reward - 2)
        forward = (
            changed and old.get("exit_forward") and old.get("exit_room") == new["room"]
        )
        if forward:
            self.steps_without_progress = 0
            if room not in self.awarded_rooms:
                self.awarded_rooms.add(room)
                reward += 5
                self.event = "Verified target-room transition +5"
            else:
                self.event = "Target room revisited / no loop bonus"
        elif changed:
            self.event = "Room change; no guessed backward penalty"
        elif old.get("navigation_id") == new.get("navigation_id") and new.get(
            "navigation_valid"
        ):
            distance = new["navigation_distance"]
            best = self.best_distance.get(
                new["navigation_id"], old.get("navigation_distance", distance)
            )
            if distance < best - 0.5:
                self.steps_without_progress = 0
                self.best_distance[new["navigation_id"]] = distance
                self.event = "New progress along static route hint"
        self.track_room(room)
        return self.finish(reward)

    def step(self, old, new, frames=4):
        if old.get("navigation_revision") == 1:
            return self.route_step(old, new, frames)
        self.distance_reward = 0.0
        self.event = "No forward progress"
        self.steps_without_progress += frames / 4.0
        old_room = (int(old["chapter"]), old["room"])
        room = (int(new["chapter"]), new["room"])
        self.track_room(old_room)
        self.track_room(room)
        old_rank = old.get("route_rank", self.room_order.get(old_room, -1))
        rank = new.get("route_rank", self.room_order.get(room, -1))
        changed = old_room != room
        known = old_room[0] == room[0] and old_rank >= 0 and rank >= 0
        # Time cost is independent of how many decisions split the same hold.
        r = -0.01 * frames / 4.0
        if new["completed"]:
            self.event = "Chapter completed +25"
            return self.finish(r + 25)
        if new["dead"]:
            self.event = "Game death -2 (progress retained)"
            return self.finish(r - 2)
        if not old["dead"] and changed and known and rank < old_rank:
            self.event = "BACKTRACK -5 (progress retained)"
            return self.finish(r - 5)
        if not old["dead"] and changed and known and rank > old_rank:
            self.steps_without_progress = 0
            if room not in self.awarded_rooms:
                self.awarded_rooms.add(room)
                r += 5
                self.event = "New forward room +5"
            else:
                self.event = "Forward room revisited (no repeat bonus)"
        if (
            not old["dead"]
            and not changed
            and old.get("exit_forward", False)
            and new.get("exit_forward", False)
            and old.get("exit_id")
            and old.get("exit_id") == new.get("exit_id")
        ):
            before, after = old.get("exit_distance"), new.get("exit_distance")
            if (
                before is not None
                and after is not None
                and np.isfinite([before, after]).all()
            ):
                before = min(max(float(before), 0), 2000)
                after = min(max(float(after), 0), 2000)
                self.distance_reward = (before - after) * 0.02
                r += self.distance_reward
                if self.distance_reward > 0.0001:
                    self.event = "Closer to forward exit"
                elif self.distance_reward < -0.0001:
                    self.event = "Moving AWAY from forward exit"
                key = (room, new["exit_id"])
                best = self.best_distance.get(key, before)
                if after < best - 0.5:
                    self.steps_without_progress = 0
                    self.best_distance[key] = after
                else:
                    self.best_distance.setdefault(key, best)
        return self.finish(r)
