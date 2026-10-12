"""Start the app or a maintenance tool in the project's private environment."""

from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
CONFIG_MODES = {"evaluate", "benchmark", "demonstrations", "start", "start64"}
TOOLS = [
    ("Verify recorded progress", "verify"),
    ("Check the fly model", "brain"),
    ("Benchmark game speed and worker counts", "benchmark"),
    ("Evaluate the saved model", "evaluate"),
    ("Install NVIDIA GPU learning support", "install-gpu"),
    ("Compare CPU and GPU learning speed", "gpu-check"),
    ("Rehearse saved successes without opening games", "rehearse"),
    ("Import a verified demonstration JSON", "demonstrations"),
    ("Show game windows", "show"),
    ("Hide game windows", "hide"),
    ("Start training with saved settings", "start"),
]


def command_for(root, python, mode, extra=()):
    root, python = Path(root), str(python)
    if mode in CONFIG_MODES and not (root / "config.json").is_file():
        raise ValueError(
            "No saved setup yet. Open START.cmd, choose Celeste, and press Save settings first."
        )
    if mode in {"verify", "brain", "gpu-check", "rehearse"}:
        script = {
            "verify": "verify",
            "brain": "check_brain",
            "gpu-check": "gpu_check",
            "rehearse": "rehearse",
        }[mode]
        return [python, "-u", str(root / f"Python/{script}.py"), *extra]
    if mode in {"evaluate", "benchmark", "demonstrations"}:
        return [
            python,
            "-u",
            str(root / f"Python/{mode}.py"),
            "--config",
            str(root / "config.json"),
            *extra,
        ]
    if mode in {"monitor", "show"}:
        return [
            python,
            str(root / "Python/monitor.py"),
            *(["--show"] if mode == "show" else []),
        ]
    if mode in {"start", "start64", "stop", "hide"}:
        return [python, "-u", str(root / "Python/background.py"), mode]
    return [python, str(root / "app.py")]


def main():
    if os.name != "nt":
        raise SystemExit(
            "The game launcher supports Windows. Model tests and offline GPU checks also run on Linux."
        )
    if sys.version_info < (3, 11):
        raise SystemExit("Use Python 3.11 or newer.")
    mode = sys.argv[1] if len(sys.argv) > 1 else "ui"
    if mode == "stop":
        sys.path.insert(0, str(ROOT / "Python"))
        from controls import Controls

        Controls(ROOT).stop()
        print("Stop requested. Waiting for in-flight work and the next completed save.")
        return 0
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    python = ROOT / ".venv-learn/Scripts/python.exe"
    if not python.exists():
        print("Creating the learning environment...", flush=True)
        venv.create(ROOT / ".venv-learn", with_pip=True)
    if subprocess.run(
        [str(python), "-c", "import numpy"], capture_output=True
    ).returncode:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            check=True,
        )
    if mode == "tools":
        print(
            "FlyBrain tools — stop training before benchmark, evaluation or demonstration import.\n"
        )
        for number, (label, _) in enumerate(TOOLS, 1):
            print(f"{number}. {label}")
        choice = input("\nNumber (Enter closes): ").strip()
        if not choice:
            return 0
        if not choice.isdigit() or not 1 <= int(choice) <= len(TOOLS):
            raise ValueError("Choose a listed number.")
        mode = TOOLS[int(choice) - 1][1]
    if mode == "install-gpu":
        print(
            "Installing optional PyTorch CUDA support. This is a large download; the NVIDIA driver is still required.",
            flush=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "torch==2.13.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ],
            check=True,
        )
        mode = "gpu-check"
    command = command_for(ROOT, python, mode, sys.argv[2:])
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), flush=True)
        raise SystemExit(1)
