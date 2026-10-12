"""Run a fixed saved policy from Prologue, recording actual game completions."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
from learning import Reward
from plastic import PlasticPolicy, PlasticState
from protocol import HEADS, sensors, action_frames, ControlPaused, ControlStopped
from train import validate_config, emit
from navigation import Navigation
from runtime import LiveStatus, atomic_json
from workers import ROOT, Worker, validate_game, build, movement_check, repair_original


def run(config):
    from runtime import RunLock

    with RunLock(ROOT):
        from controls import Controls

        Controls(ROOT).new_run()
        _run(config)


def _run(config):
    validate_config(config)
    from runtime import set_visibility

    set_visibility(ROOT, True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    policy = PlasticPolicy()
    brain = PlasticState(policy)
    checkpoint = ROOT / "training/latest.npz"
    metadata = policy.load(checkpoint, brain.graph_hash)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if metadata["chapters"] != config["chapters"]:
        raise ValueError("Campaign mismatch")
    game = validate_game(config["game"])
    repair_original(game, emit)
    mod = build(game, emit, config.get("rebuild", False))

    def watch_stop():
        while not stop.wait(0.25):
            if (ROOT / "STOP").exists():
                stop.set()

    threading.Thread(target=watch_stop, daemon=True).start()
    worker = Worker(0, game, mod, stop, emit, config["ticks"])
    live = LiveStatus(ROOT, 1, str(time.time_ns()))
    live.start()
    worker.telemetry = live
    worker.background = False
    navigation = Navigation()
    rng = np.random.default_rng(config.get("seed", 17) + 2026)
    record = dict(
        checkpoint_iteration=metadata["iteration"],
        completed_chapters=[],
        deaths=0,
        decisions=0,
        full_main_story_clear=False,
        weights_updated=False,
        practice_restores=0,
        teacher_actions=0,
        manual_interruptions=0,
        action_selection=config.get("evaluation_action", "sample"),
        graph_sha256=policy.graph_hash,
        checkpoint_sha256=checkpoint_hash,
        started_unix=time.time(),
    )
    try:
        c = worker.launch()
        c.learning_info = policy.diagnostics()
        c.practice_mode(False)
        while True:
            try:
                movement_check(c, 0, stop)
                break
            except ControlPaused:
                c.wait_until_on(stop)
        state = c.start(0, stop)
        index = 0
        reward = Reward()
        reward.reset(state)
        total_reward = 0.0
        c.set_feedback(0, 0, 0, 0, "Evaluation: attempt reset")
        end = time.monotonic() + config["hours"] * 3600
        emit("Evaluating fixed policy from Prologue. No weights will change.")
        while (
            time.monotonic() < end
            and not (ROOT / "STOP").exists()
            and not stop.is_set()
        ):
            state = navigation.annotate(state)
            f = brain.features(sensors(state))
            logits = f @ policy.actor
            a = []
            off = 0
            for n in HEADS:
                if config.get("evaluation_action", "sample") == "greedy":
                    a.append(int(np.argmax(logits[off : off + n])))
                else:
                    from learning import softmax

                    probabilities = softmax(logits[off : off + n])
                    a.append(int(rng.choice(n, p=probabilities / probabilities.sum())))
                off += n
            try:
                old = state
                state = c.step(a, action_frames(a))
            except ControlPaused:
                c.wait_until_on(stop)
                state = c.start(config["chapters"][index], stop)
                brain.reset()
                navigation.reset()
                state = navigation.annotate(state)
                record["manual_interruptions"] = (
                    record.get("manual_interruptions", 0) + 1
                )
                reward.reset(state)
                c.set_feedback(total_reward, 0, 0, 0, "Evaluation: attempt reset")
                continue
            state = navigation.annotate(state)
            total_reward += reward.step(
                old, state, state.get("action_frames", action_frames(a))
            )
            c.set_feedback(
                total_reward,
                reward.last,
                reward.distance_reward,
                reward.score,
                reward.event,
            )
            record["decisions"] += 1
            worker.report(
                phase="evaluating",
                chapter=state["chapter"],
                room=state["room"],
                action=a,
                action_frames=state.get("action_frames", action_frames(a)),
                requested_frames=action_frames(a),
                last_response=time.time(),
                score=reward.score,
                actions=record["decisions"],
                event=reward.event,
                objective=state.get("objective"),
                actual_vx=state.get("vx"),
                actual_vy=state.get("vy"),
            )
            live.update(
                phase="evaluating",
                decisions=record["decisions"],
                completed_chapters=record["completed_chapters"],
                message="Frozen model from Prologue; no teacher or practice resets",
            )
            if state["completed"]:
                record["completed_chapters"].append(config["chapters"][index])
                index += 1
                emit("Evaluation chapter completion: " + str(state["chapter"]))
                if index == len(config["chapters"]):
                    record["full_main_story_clear"] = record[
                        "manual_interruptions"
                    ] == 0 and record["completed_chapters"] == list(range(8))
                    break
                state = c.start(config["chapters"][index], stop)
                brain.reset()
                navigation.reset()
                state = navigation.annotate(state)
                reward.reset(state)
                c.set_feedback(total_reward, 0, 0, 0, "Evaluation: attempt reset")
            elif state["dead"]:
                record["deaths"] += 1
                try:
                    state = c.retry()
                except ControlPaused:
                    c.wait_until_on(stop)
                    state = c.start(config["chapters"][index], stop)
                    record["manual_interruptions"] = (
                        record.get("manual_interruptions", 0) + 1
                    )
                brain.reset()
                navigation.reset()
                state = navigation.annotate(state)
                reward.reset(state)
                c.set_feedback(total_reward, 0, 0, 0, "Evaluation: attempt reset")
            elif state["room"] != old["room"]:
                brain.reset()
                navigation.reset()
                reward.reset(state, fresh_attempt=False)
    except (ControlStopped, InterruptedError):
        emit("Evaluation stopped.")
    finally:
        stop.set()
        worker.shutdown()
        record["finished_unix"] = time.time()
        target = ROOT / "training/evaluation.json"
        target.parent.mkdir(exist_ok=True)
        atomic_json(target, record)
        live.close(
            "stopped", "Frozen-policy evaluation finished; see training/evaluation.json"
        )
        emit("Evaluation recorded: " + json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
