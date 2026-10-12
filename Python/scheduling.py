"""Start real workers in bounded batches; never substitute RNG streams for games."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from windows import memory_status, process_info
from runtime import capacity_reason


def launch_workers(
    workers, start_one, stop, parallelism=2, memory=memory_status, inspect=process_info
):
    ready = []
    pending = {}
    next_index = 0
    reason = ""
    failures = 0
    measured = 512 * 1024**2
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        while (next_index < len(workers) or pending) and not stop.is_set():
            # Measure one full startup before allowing concurrent launches.
            limit = parallelism if ready else 1
            while next_index < len(workers) and len(pending) < limit and not reason:
                blocked = capacity_reason(memory(), measured, len(pending))
                if blocked:
                    if not pending:
                        reason = blocked
                    break
                worker = workers[next_index]
                next_index += 1
                pending[pool.submit(start_one, worker)] = worker
            if not pending:
                break
            done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                worker = pending.pop(future)
                try:
                    runner = future.result()
                    pid = worker.launch_info.get("bridge_pid")
                    if not isinstance(pid, int) or pid <= 0:
                        raise RuntimeError("Game did not report a valid process ID")
                    if any(
                        r.worker.launch_info.get("bridge_pid") == pid for r in ready
                    ):
                        raise RuntimeError(
                            "Two workers reported the same PID; they are not independent games"
                        )
                    ready.append(runner)
                    failures = 0
                    info = inspect(pid) or {}
                    measured = max(
                        measured,
                        info.get("working_bytes", 0),
                        info.get("private_bytes", 0),
                    )
                except Exception as e:
                    try:
                        worker.shutdown()
                    except Exception:
                        pass
                    worker.report(phase="failed", event=str(e))
                    worker.log(f"Worker {worker.index+1} failed: {e}")
                    failures += 1
                    if not ready or failures >= 3:
                        reason = "Startup stopped after repeated/initial failure. See the failed worker and run.log."
            if reason and not pending:
                break
        for worker in workers[next_index:]:
            worker.report(
                phase="blocked", event=reason or "Run was stopped before launch"
            )
    if stop.is_set():
        return [], reason or "Stopped during startup"
    return sorted(ready, key=lambda r: r.worker.index), reason


def rollout_size(config, worker_count):
    # A 64-game rollout should not delay every update by collecting 64 times
    # the one-game budget. Keep bounded batches while preserving BPTT length.
    return max(
        32,
        min(
            config["rollout"],
            config.get("batch_decisions", 2048) // max(1, worker_count),
        ),
    )
