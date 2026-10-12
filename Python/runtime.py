"""Owned-run locking, atomic live status and explicit window controls."""

from pathlib import Path
import json
import os
import tempfile
import threading
import time
from windows import memory_status, process_info, apply_windows, owned_process, GIB


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, allow_nan=False)
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


class RunLock:
    def __init__(self, root):
        self.path = Path(root) / "training/run.lock"
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError(
                "Training is already running. Open MONITOR.cmd to view it."
            )
        return self

    def __exit__(self, *args):
        if self.file:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()
            self.file = None


def is_running(root):
    try:
        with RunLock(root):
            pass
        return False
    except RuntimeError:
        return True


def set_visibility(root, show, workers=None):
    path = Path(root) / "training/visibility.json"
    value = read_json(path, {"default": False, "workers": {}})
    if workers is None:
        value = dict(default=bool(show), workers={})
    else:
        for worker in workers:
            value.setdefault("workers", {})[str(worker)] = bool(show)
    value["generation"] = time.time_ns()
    atomic_json(path, value)
    return value


def desired_visibility(root, worker, default=False):
    value = read_json(Path(root) / "training/visibility.json")
    return bool(
        value.get("workers", {}).get(str(worker), value.get("default", default))
    )


def change_windows(root, show, workers=None):
    set_visibility(root, show, workers)
    live = read_json(Path(root) / "training/live.json")
    matched = 0
    errors = []
    if not is_running(root) or time.time() - live.get("heartbeat", 0) > 10:
        return "No active learner with fresh status in this folder. Open the monitor to check startup/errors."
    for row in live.get("workers", []):
        if workers is not None and row["worker"] not in workers:
            continue
        if not owned_process(row):
            continue
        try:
            info = apply_windows(
                row["pid"],
                show,
                row["worker"],
                foreground=workers is not None,
                force=True,
            )
            matched += info["windows"]
        except OSError as e:
            errors.append(f'Worker {row["worker"]}: {e}')
    return f'{"Show" if show else "Hide"} requested for {matched} game windows. ' + (
        "; ".join(errors)
        if errors
        else "The monitor shows each worker and its last response."
    )


class WindowVisibility:
    def __init__(
        self, pid, root, stop, worker=1, default=False, on_status=None, identity=None
    ):
        self.pid, self.root, self.stop, self.worker = pid, Path(root), stop, worker
        self.default, self.on_status, self.identity = default, on_status, identity

    def start(self):
        if os.name != "nt":
            return
        threading.Thread(
            target=self.run, daemon=True, name=f"windows-{self.worker}"
        ).start()

    def run(self):
        last = None
        while not self.stop.is_set():
            try:
                if self.identity and not owned_process(self.identity):
                    break
                value = apply_windows(
                    self.pid,
                    desired_visibility(self.root, self.worker, self.default),
                    self.worker,
                )
            except Exception as e:
                value = dict(windows=0, visible=0, message=str(e))
            if value != last and self.on_status:
                self.on_status(value)
            last = value
            self.stop.wait(0.5)


def watch_stop(root, stop):
    """Share the launcher's STOP button with auxiliary native-game tools."""
    import signal

    signal.signal(signal.SIGINT, lambda *args: stop.set())
    signal.signal(signal.SIGTERM, lambda *args: stop.set())

    def watch():
        while not stop.wait(0.25):
            if (Path(root) / "STOP").exists():
                stop.set()

    threading.Thread(target=watch, daemon=True).start()


class LiveStatus:
    def __init__(self, root, count, run_id):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.data = dict(
            version="4.1.0",
            run_id=run_id,
            trainer_pid=os.getpid(),
            phase="starting",
            requested_workers=count,
            message="Preparing real game processes",
            iteration=0,
            decisions=0,
        )
        self.rows = {
            i: dict(worker=i + 1, phase="queued", pid=None, actions=0)
            for i in range(count)
        }
        self.thread = None

    def update(self, index=None, **fields):
        with self.lock:
            if index is None:
                self.data.update(fields)
            else:
                self.rows[index].update(fields)

    def snapshot(self):
        with self.lock:
            value = {**self.data, "workers": [dict(row) for row in self.rows.values()]}
        for row in value["workers"]:
            if row.get("pid"):
                info = process_info(row["pid"])
                row["process_alive"] = bool(
                    info
                    and info.get("alive")
                    and info.get("created") == row.get("process_created")
                )
                if info:
                    row["memory_mb"] = round(info.get("working_bytes", 0) / 1024**2, 1)
            else:
                row["process_alive"] = False
        value["live_game_processes"] = sum(w["process_alive"] for w in value["workers"])
        value["ready_workers"] = sum(
            w["process_alive"]
            and w["phase"]
            in ("running", "learning", "ready", "paused", "searching", "evaluating")
            for w in value["workers"]
        )
        value["memory"] = memory_status()
        value["heartbeat"] = time.time()
        return value

    def flush(self):
        atomic_json(self.root / "training/live.json", self.snapshot())

    def start(self):
        self.flush()

        def loop():
            while not self.stop.wait(1):
                try:
                    self.flush()
                except OSError:
                    pass  # Never kill training because a report is momentarily locked.

        self.thread = threading.Thread(target=loop, daemon=True, name="live-status")
        self.thread.start()

    def close(self, phase, message):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.update(phase=phase, message=message)
        self.flush()


def capacity_reason(memory, estimated_worker_bytes, pending=0):
    if memory is None:
        return None
    reserve = max(2 * GIB, int(memory["total"] * 0.1))
    need = max(512 * 1024**2, int(estimated_worker_bytes * 1.3)) * (pending + 1)
    if memory["available"] < need + reserve or memory["commit_available"] < need + GIB:
        return (
            f'Memory limit: {memory["available"]/GIB:.1f} GiB available; '
            f"next worker/startups need about {need/GIB:.1f} GiB plus {reserve/GIB:.1f} GiB reserve. "
            "Queued games were not launched; existing workers can continue learning."
        )
    return None
