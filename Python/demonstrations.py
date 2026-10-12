"""Verify an optional recorded input sequence in Celeste before teaching it.

Default LEARN needs no recording: its search teacher generates examples. This
tool adds an expert chapter recording only after Celeste confirms completion.
"""

import argparse
import json
from pathlib import Path
import threading
import numpy as np
from protocol import action_fields, sensors, action_frames
from plastic import PlasticPolicy
from navigation import Navigation
from experience import SuccessMemory
from runtime import RunLock, watch_stop
from controls import Controls
from train import validate_config
from workers import ROOT, Worker, validate_game, build, repair_original


def validate_recording(value):
    if not isinstance(value, dict):
        raise ValueError("Recording must be a JSON object")
    if type(value.get("chapter")) is not int or not 0 <= value["chapter"] <= 7:
        raise ValueError("Recording chapter must be 0 through 7")
    actions = value.get("actions", [])
    if not isinstance(actions, list) or not 1 <= len(actions) <= 100000:
        raise ValueError("Recording must contain 1 to 100000 button/duration arrays")
    for action in actions:
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError("Every action needs seven choices, including duration")
        action_fields(action)
    return actions


def run(config, recording):
    validate_config(config)
    actions = validate_recording(recording)
    with RunLock(ROOT):
        Controls(ROOT).new_run()
        stop = threading.Event()
        watch_stop(ROOT, stop)
        game = validate_game(config["game"])
        repair_original(game)
        worker = Worker(0, game, build(game), stop, print, config["ticks"])
        pending, records = [], []
        navigation = Navigation()
        policy = PlasticPolicy()
        try:
            client = worker.launch()
            client.practice_mode(False)
            state = client.start(recording["chapter"], worker.stop)
            reset = True
            for action in actions:
                if stop.is_set():
                    raise InterruptedError(
                        "Demonstration stopped; no incomplete chapter was accepted"
                    )
                old = navigation.annotate(state)
                new = client.step(action)
                records.append(
                    dict(
                        observation=sensors(old),
                        action=np.asarray(action),
                        reset=reset,
                        frames=new.get("action_frames", action_frames(action)),
                        teacher=True,
                    )
                )
                reset = False
                if new["completed"] or new["room"] != old["room"]:
                    pending.append(
                        (
                            records,
                            old["chapter"],
                            old["room"],
                            new["room"],
                            new["completed"],
                        )
                    )
                    records, reset = [], True
                    navigation.reset()
                if new["completed"]:
                    memory = SuccessMemory(
                        ROOT / "training/success-memory", policy.graph_hash
                    )
                    added = sum(
                        memory.add(
                            rows,
                            chapter,
                            room,
                            target,
                            completed=completed,
                            source="verified_demonstration",
                        )
                        for rows, chapter, room, target, completed in pending
                    )
                    print(
                        f"Celeste confirmed chapter completion. Added {added} verified demonstration routes.",
                        flush=True,
                    )
                    return
                if new["dead"]:
                    state = client.retry()
                    records, reset = [], True
                    navigation.reset()
                else:
                    state = new
            raise ValueError(
                "Recording ended without game-reported chapter completion. No demonstration was accepted."
            )
        finally:
            stop.set()
            worker.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.file is None:
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        filename = filedialog.askopenfilename(
            title="Choose a recorded chapter input JSON", filetypes=[("JSON", "*.json")]
        )
        window.destroy()
        if not filename:
            raise SystemExit(0)
        args.file = Path(filename)
    run(json.loads(args.config.read_text()), json.loads(args.file.read_text()))
