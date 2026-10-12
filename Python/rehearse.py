"""Learn from saved successes without Celeste, locally or on a GPU host.

Writes a separate candidate. Replays are not fresh frames or new room clears.
"""

import argparse
import hashlib
from pathlib import Path
import threading
import time
from accelerator import choose_backend
from experience import SuccessMemory
from plastic import PlasticPolicy
from runtime import RunLock, atomic_json, watch_stop
from controls import Controls

ROOT = Path(__file__).resolve().parents[1]


def run(root=ROOT, device="auto", hours=1.0, max_updates=0, stop=None):
    root = Path(root)
    if device not in {"auto", "cpu", "cuda"} or not 0 < hours <= 720 or max_updates < 0:
        raise ValueError("Invalid rehearsal settings")
    source = root / "training/latest.npz"
    if not source.exists():
        raise ValueError("Train first: no training/latest.npz exists")
    with RunLock(root):
        Controls(root).new_run()
        stop = stop or threading.Event()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        policy = PlasticPolicy()
        metadata = policy.load(source, policy.graph_hash)
        memory = SuccessMemory(root / "training/success-memory", policy.graph_hash)
        if not memory.entries:
            raise ValueError(
                "No verified successful routes yet. Rehearsal needs successes; it cannot discover rooms offline."
            )
        destination = root / "training/offline"
        destination.mkdir(exist_ok=True)
        backend = choose_backend(policy, device, count=32)
        watch_stop(root, stop)
        started, updates, replayed = time.monotonic(), 0, 0
        report = {}
        try:
            while not stop.is_set() and time.monotonic() - started < hours * 3600:
                routes = memory.sample(policy.rng, count=32)
                routes = [routes[i % len(routes)] for i in range(32)]
                metrics = policy.imitate(routes, decisions=1024)
                replayed += metrics["imitation_decisions"]
                policy.updates += 1
                updates += 1
                candidate = dict(
                    metadata,
                    imitation_decisions_total=metadata.get(
                        "imitation_decisions_total", 0
                    )
                    + replayed,
                    offline_source_sha256=source_hash,
                    offline_replay_updates=updates,
                    offline_replayed_decisions=replayed,
                    saved_unix=time.time(),
                )
                if (
                    updates == 1
                    or updates % 20 == 0
                    or (max_updates and updates >= max_updates)
                ):
                    policy.save(
                        destination / "latest.npz", policy.graph_hash, candidate
                    )
                    report = dict(
                        updates=updates,
                        replayed_decisions=replayed,
                        fresh_game_decisions=0,
                        fresh_game_frames=0,
                        native_game_opened=False,
                        source_sha256=source_hash,
                        output="training/offline/latest.npz",
                        source_modified=hashlib.sha256(source.read_bytes()).hexdigest()
                        != source_hash,
                        backend=backend,
                        seconds=time.monotonic() - started,
                        **policy.diagnostics(),
                        **metrics,
                    )
                    atomic_json(destination / "report.json", report)
                    print(
                        f"Rehearsal {updates}: {replayed:,} replayed actions; zero new game frames; saved separate candidate",
                        flush=True,
                    )
                if max_updates and updates >= max_updates:
                    break
            if updates:
                policy.save(destination / "latest.npz", policy.graph_hash, candidate)
                report.update(
                    updates=updates,
                    replayed_decisions=replayed,
                    seconds=time.monotonic() - started,
                )
                atomic_json(destination / "report.json", report)
        finally:
            stop.set()
        return destination / "latest.npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args.root, args.device, args.hours, args.max_updates)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
