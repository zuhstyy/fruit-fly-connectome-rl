"""Start/stop background training without keeping the launcher open."""

from pathlib import Path
import json
import os
import subprocess
import sys
from controls import Controls
from runtime import RunLock

ROOT = Path(__file__).resolve().parents[1]


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "start"
    if action == "stop":
        Controls(ROOT).stop()
        print("Stop requested. The completed update will be saved.")
        return
    if action in ("show", "hide"):
        from runtime import change_windows

        print(change_windows(ROOT, action == "show"))
        return
    config = ROOT / "config.json"
    if not config.exists():
        raise SystemExit(
            "Open START.cmd and Save settings with your Celeste folder first."
        )
    from train import validate_config
    from workers import validate_game

    c = json.loads(config.read_text())
    if action == "start64":
        c["workers"] = 64
        config.write_text(json.dumps(c, indent=2))
    validate_config(c)
    validate_game(c["game"])
    with RunLock(ROOT):
        pass
    Controls(ROOT).new_run()
    saved = ROOT / "training"
    saved.mkdir(exist_ok=True)
    with (saved / "run.log").open("ab") as log:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(ROOT / "Python/train.py"),
                "--config",
                str(config),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
    print(
        f"Background learner started (PID {process.pid}). This window may close. MONITOR.cmd shows status and games; STOP.cmd stops cleanly."
    )

    subprocess.Popen([sys.executable, str(ROOT / "Python/monitor.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
