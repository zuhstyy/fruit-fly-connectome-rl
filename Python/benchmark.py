"""Compare native frame pacing and real process counts without changing weights."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
from runtime import RunLock, LiveStatus, atomic_json, set_visibility, watch_stop
from controls import Controls
from train import validate_config
from workers import ROOT, Worker, validate_game, build, movement_check, repair_original
from scheduling import launch_workers
from performance import throughput_summary


def measure(worker, decisions=120):
    client = worker.client
    frames, completed = 0, 0
    latencies = []
    for i in range(decisions):
        if worker.stop.is_set():
            break
        started = time.perf_counter()
        # Include real jumps and runs; no room-clear or learning claim is made.
        state = client.step([2, 1, int(i % 12 < 6), 0, 0, 0, 2])
        latencies.append(1000 * (time.perf_counter() - started))
        completed += 1
        frames += state.get("action_frames", 0)
        if state["dead"]:
            client.retry()
        if state["completed"]:
            client.start(0, worker.stop)
    return dict(
        frames=frames,
        decisions=completed,
        p50_step_ms=float(np.median(latencies)) if latencies else 0,
        p95_step_ms=float(np.percentile(latencies, 95)) if latencies else 0,
        bridge=client.performance(),
    )


def run(config):
    validate_config(config)
    with RunLock(ROOT):
        Controls(ROOT).new_run()
        stop = threading.Event()
        watch_stop(ROOT, stop)
        game = validate_game(config["game"])
        repair_original(game)
        mod = build(game)
        set_visibility(ROOT, False)
        rows = []
        workloads = [
            (1, False),
            *[(n, True) for n in (1, 2, 4, 8) if n <= config["workers"]],
        ]
        for count, fast in workloads:
            if stop.is_set():
                break
            live = LiveStatus(ROOT, count, str(time.time_ns()))
            live.start()
            workers = [
                Worker(i, game, mod, stop, print, 30 if fast else 4)
                for i in range(count)
            ]
            for worker in workers:
                worker.telemetry, worker.background, worker.fast_simulation = (
                    live,
                    True,
                    fast,
                )

            def start(worker):
                client = worker.launch()
                movement_check(client, 0, stop)
                status = client.request("status")
                if status.get("fast_simulation") != fast:
                    raise RuntimeError(
                        "Worker did not confirm the requested pacing mode"
                    )
                return worker

            try:
                ready, message = launch_workers(
                    workers, start, stop, config.get("startup_parallelism", 2)
                )
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=max(1, len(ready))) as pool:
                    measurements = list(pool.map(measure, ready))
                seconds = time.monotonic() - started
                total = sum(m["frames"] for m in measurements)
                row = dict(
                    requested=count,
                    ready=len(ready),
                    fast_simulation=fast,
                    measurements=measurements,
                    pids=[w.launch_info.get("bridge_pid") for w in ready],
                    frames=total,
                    seconds=seconds,
                    **throughput_summary(total, seconds),
                    startup_message=message,
                )
                rows.append(row)
                print(
                    f"{'FAST' if fast else 'DISPLAY-PACED BASELINE'} | {len(ready)}/{count} games | {row['frames_per_second']:.0f} frames/sec",
                    flush=True,
                )
            finally:
                for worker in workers:
                    worker.shutdown()
                live.close("stopped", "Throughput benchmark completed")
        complete = [
            r
            for r in rows
            if r["fast_simulation"] and r["ready"] == r["requested"] and r["frames"] > 0
        ]
        baseline = next(
            (
                r
                for r in rows
                if not r["fast_simulation"] and r["ready"] == 1 and r["frames"] > 0
            ),
            None,
        )
        single = next((r for r in complete if r["requested"] == 1), None)
        result = dict(
            rows=rows,
            weights_updated=False,
            version="4.1.0",
            single_worker_speedup=(
                single["frames_per_second"] / baseline["frames_per_second"]
                if single and baseline
                else None
            ),
            recommended_tested_workers=(
                max(complete, key=lambda r: r["frames_per_second"])["requested"]
                if complete
                else None
            ),
            note="Short scripted test of actual game processes. It excludes startup time; includes step/respawn time. It is not learning speed or overnight stability.",
        )
        atomic_json(ROOT / "training/throughput.json", result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2),
            flush=True,
        )
        print("Saved training/throughput.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit(
            "Open START.cmd and Save settings first. No config.json exists yet."
        )
    run(json.loads(args.config.read_text()))
