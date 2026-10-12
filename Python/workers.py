"""Build and run owned, isolated copies of the user's Everest installation."""

from pathlib import Path
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import time
import zipfile
from protocol import Client, BridgeError, VERSION
from mod_install import find_flybrain, move_to_backup, repair_original, module_names

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "saves",
    "backups",
    "mods",
    "orig",
    ".git",
    "logs",
    "log.txt",
    "error_log.txt",
    "everest-env.txt",
    "everest-launch.txt",
    "flybrain-disabled",
    "disabled-mods",
    "training-data",
    "nextlaunchisvanilla.txt",
    "flybrain-process.log",
    "flybrain-host-trace.txt",
    "flybrain-launch.json",
}
COPY_REVISION = 3


def exit_description(code):
    return "still running" if code is None else f"{code} (0x{code & 0xffffffff:08X})"


def validate_game(path):
    path = Path(path).resolve()
    for name in ("Celeste.exe", "Celeste.dll", "FNA.dll", "MMHOOK_Celeste.dll"):
        if not (path / name).is_file():
            raise ValueError(f"Choose the Everest folder containing {name}.")
    if path == ROOT or path in ROOT.parents or ROOT in path.parents:
        raise ValueError(
            "Extract LEARN outside the Celeste installation to avoid recursive copying."
        )
    return path


def build(game, log=print, force_rebuild=False):
    bundled = ROOT / "prebuilt/FlyBrainCeleste.zip"
    if bundled.is_file() and not force_rebuild:
        with zipfile.ZipFile(bundled) as z:
            manifest = z.read("everest.yaml").decode("utf-8")
            if (
                f"Version: {VERSION}" not in manifest
                or z.read("bin/FlyBrainCeleste.dll")[:2] != b"MZ"
            ):
                raise RuntimeError(
                    "The bundled mod is damaged or the wrong version. Extract the complete ZIP again."
                )
        log(f"Using bundled compiled mod {VERSION}. No SDK build required.")
        return bundled
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise RuntimeError(
            ".NET SDK 8 or newer is required to build against your game."
        )
    sdks = subprocess.check_output([dotnet, "--list-sdks"], text=True)
    if not any(
        s.split(".")[0].isdigit() and int(s.split(".")[0]) >= 8
        for s in sdks.splitlines()
    ):
        raise RuntimeError("Install the .NET SDK (8 or newer), not just Runtime.")
    log("Building the new mod against your Everest installation...")
    output = subprocess.run(
        [
            dotnet,
            "build",
            str(ROOT / "Source/FlyBrainCeleste.csproj"),
            "-c",
            "Release",
            f"-p:CelestePath={game}",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
    )
    log(output.stdout)
    if output.returncode:
        raise RuntimeError(
            "C# build failed. Copy the build output above; no training workers were launched."
        )
    dll = ROOT / "bin/FlyBrainCeleste.dll"
    if not dll.is_file():
        raise RuntimeError("Build did not produce FlyBrainCeleste.dll")
    target = ROOT / "build/FlyBrainCeleste.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "Source/everest.yaml", "everest.yaml")
        z.write(dll, "bin/FlyBrainCeleste.dll")
    return target


def ignore(directory, names):
    return [
        n for n in names if n.lower() in EXCLUDED or Path(directory, n).is_symlink()
    ]


def prepare_copy(game, dest, mod, log=print):
    dest = Path(dest)
    marker = dest / "flybrain-worker.json"
    boot_files = (
        "Celeste.exe",
        "Celeste.dll",
        "Celeste.runtimeconfig.json",
        "Celeste.deps.json",
        "piton-runtime.yaml",
    )
    identity = {
        "source": str(game),
        "copy_revision": COPY_REVISION,
        "boot_files": {
            name: [(game / name).stat().st_size, (game / name).stat().st_mtime_ns]
            for name in boot_files
            if (game / name).is_file()
        },
    }
    if dest.exists() and not marker.exists():
        raise RuntimeError(f"Refusing to overwrite unowned folder {dest}")
    current = json.loads(marker.read_text()) if marker.exists() else None
    if current != identity:
        if current is not None and current.get("source") != str(game):
            raise RuntimeError(
                "Worker belongs to a different game path; choose a fresh LEARN folder."
            )
        log(f"Preparing isolated game copy: {dest.name} (first run can take a while)")
        # Mutable game files are real copies, never hardlinks into the normal install.
        shutil.copytree(game, dest, dirs_exist_ok=True, ignore=ignore)
        marker.write_text(json.dumps(identity))
    mods = dest / "Mods"
    mods.mkdir(exist_ok=True)
    # A reused worker may contain an older ZIP, a renamed download, or an
    # extracted source folder. Move every non-core package outside Mods first.
    old = [
        p
        for p in mods.iterdir()
        if p.name not in {"FlyBrainCeleste.zip", "SpeedrunTool.zip"}
        and p.stem.lower() not in {"everest", "everestcore", "_everest"}
    ]
    old += [
        p
        for p in mods.iterdir()
        if p.stem.lower() in {"everest", "everestcore", "_everest"}
        and "FlyBrainCeleste" in module_names(p)
    ]
    move_to_backup(old, dest / "disabled-mods", log)
    # Keep only packaged Everest core content, not unrelated gameplay mods.
    if (game / "Mods").exists():
        for p in (game / "Mods").iterdir():
            if p.stem.lower() in {"everest", "everestcore", "_everest"}:
                if "FlyBrainCeleste" in module_names(p):
                    continue
                if p.is_dir():
                    shutil.copytree(p, mods / p.name, dirs_exist_ok=True)
                elif p.is_file():
                    shutil.copy2(p, mods / p.name)
    dependency = ROOT / "prebuilt/SpeedrunTool.zip"
    info_path = ROOT / "prebuilt/SpeedrunTool-INFO.json"
    if not dependency.is_file() or not info_path.is_file():
        raise RuntimeError(
            "Missing bundled SpeedrunTool. Extract the complete 4.1 package."
        )
    info = json.loads(info_path.read_text())
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError("Bundled practice dependency failed its SHA256 check.")
    shutil.copy2(dependency, mods / "SpeedrunTool.zip")
    shutil.copy2(mod, mods / "FlyBrainCeleste.zip")
    if find_flybrain(dest) != [mods / "FlyBrainCeleste.zip"]:
        raise RuntimeError(
            "Worker must have exactly one FlyBrainCeleste manifest. Rebuild before launching."
        )
    # EVEREST_SAVEPATH is the parent of Saves and Backups, not Saves itself.
    saves = dest / "training-data" / "Saves"
    saves.mkdir(parents=True, exist_ok=True)
    (saves.parent / "Backups").mkdir(exist_ok=True)
    # A copied Steam build otherwise asks Steam to restart the original install
    # before Everest/mod logging begins, losing this worker's environment.
    # This is Steam's development launch mechanism; Steam still checks ownership.
    (dest / "steam_appid.txt").write_text("504230\n", encoding="ascii")
    # Never inherit a pending "restart into vanilla" request in an owned worker.
    (dest / "nextLaunchIsVanilla.txt").unlink(missing_ok=True)
    return saves


class Worker:
    def __init__(self, index, game, mod, stop, log=print, ticks=1):
        self.index, self.game, self.mod, self.stop, self.log, self.ticks = (
            index,
            game,
            mod,
            stop,
            log,
            ticks,
        )
        self.dest = ROOT / "workers" / f"worker-{index+1:02d}"
        self.token = secrets.token_hex(32)
        self.process = self.client = self.handle = None
        self.restarts = 0
        self.launch_info = {}
        self.telemetry = None

    def report(self, **fields):
        if self.telemetry:
            self.telemetry.update(self.index, **fields)

    def launch(self):
        self.report(phase="copying", pid=None, event="Preparing isolated game files")
        saves = prepare_copy(self.game, self.dest, self.mod, self.log)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        env = os.environ.copy()
        env.update(
            FLYBRAIN_WORKER="1",
            FLYBRAIN_WORKER_ID=str(self.index),
            FLYBRAIN_PORT=str(self.port),
            FLYBRAIN_TOKEN=self.token,
            FLYBRAIN_TICKS=str(self.ticks),
            FLYBRAIN_FAST="1" if getattr(self, "fast_simulation", True) else "0",
            FLYBRAIN_VISIBILITY_FILE=str(ROOT / "training/visibility.json"),
            EVEREST_SAVEPATH=str(saves.parent.resolve()),
            FLYBRAIN_PAUSE_FILE=str(ROOT / "PAUSE"),
            FLYBRAIN_STOP_FILE=str(ROOT / "STOP"),
            SteamAppId="504230",
            SteamGameId="504230",
            EVEREST_LOG_FILENAME="flybrain-game.txt",
            COREHOST_TRACE="1",
            COREHOST_TRACEFILE=str(self.dest / "flybrain-host-trace.txt"),
            DOTNET_HOST_TRACE="1",
            DOTNET_HOST_TRACEFILE=str(self.dest / "flybrain-host-trace.txt"),
        )
        command = [str(self.dest / "Celeste.exe")]
        self.launch_info = dict(
            command=command,
            working_directory=str(self.dest),
            save_directory=str(saves),
            source=str(self.game),
            steam_app_id="504230",
            started_unix=time.time(),
            files={},
        )
        for name in (
            "Celeste.exe",
            "Celeste.dll",
            "Celeste.runtimeconfig.json",
            "Celeste.deps.json",
            "piton-runtime.yaml",
            "lib64-win-x64",
            "lib64-win-x86",
            "piton-runtime",
            "steam_appid.txt",
        ):
            path = self.dest / name
            self.launch_info["files"][name] = dict(
                exists=path.exists(),
                bytes=path.stat().st_size if path.is_file() else None,
            )
        self._save_launch_info()
        # Keep only this attempt in active diagnostic files, so an older error
        # cannot be mistaken for the current failure. Previous attempt tails are
        # already in training/run.log when recovery was requested.
        for name in ("flybrain-host-trace.txt", "flybrain-game.txt"):
            (self.dest / name).write_text("", encoding="utf-8")
        self.handle = (self.dest / "flybrain-process.log").open("wb")
        self.log(
            f"Worker {self.index+1}: launch command: {subprocess.list2cmdline(command)}"
        )
        self.log(
            f"Worker {self.index+1}: Steam app ID configured for the worker copy. Keep Steam open for the Steam edition."
        )
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.dest,
                env=env,
                stdout=self.handle,
                stderr=subprocess.STDOUT,
            )
            self.launch_info["launcher_pid"] = self.process.pid
            self._save_launch_info()
        except OSError as e:
            self.launch_info["launch_error"] = str(e)
            self._save_launch_info()
            raise RuntimeError(
                f"Windows could not start the worker: {e}\n" + self.failure_details()
            ) from e
        self.log(
            f"Worker {self.index+1}: starting Celeste; waiting for the mod bridge..."
        )
        self.report(phase="starting", event="Waiting for authenticated mod bridge")
        client = self._connect(saves)
        if self.launch_info.get("bridge_pid"):
            from runtime import WindowVisibility
            from windows import process_info

            info = process_info(self.launch_info["bridge_pid"]) or {}
            identity = dict(
                pid=self.launch_info["bridge_pid"],
                process_created=info.get("created"),
                worker_path=str(self.dest),
            )
            self.report(
                phase="preflight",
                **identity,
                event="Checking isolated save and movement",
            )
            self.visibility_stop = __import__("threading").Event()
            WindowVisibility(
                self.launch_info["bridge_pid"],
                ROOT,
                self.visibility_stop,
                worker=self.index + 1,
                default=not getattr(self, "background", False),
                identity=identity,
                on_status=lambda value: self.report(window=value),
            ).start()
        return client

    def _save_launch_info(self):
        (self.dest / "flybrain-launch.json").write_text(
            json.dumps(self.launch_info, indent=2), encoding="utf-8"
        )

    def _connect(self, saves, timeout=120, exit_grace=8):
        deadline = time.monotonic() + timeout
        exited_at = None
        while time.monotonic() < deadline and not self.stop.is_set():
            candidate = None
            try:
                # The bridge, not the first process handle, proves a worker is
                # alive. A launcher can exit after handing off to a child.
                candidate = Client(self.port, self.token, timeout=5, log=self.log)
                s = candidate.request("status")
                if Path(s.get("save_path", "")).resolve() != saves.resolve():
                    raise RuntimeError("Worker save isolation check failed")
                if Path(s.get("worker_path", "")).resolve() != self.dest.resolve():
                    raise RuntimeError("Connected game is not the isolated worker copy")
                candidate.socket.settimeout(130)
                self.client = candidate
                self.launch_info["bridge_pid"] = s.get("worker_pid")
                self.launch_info["launcher_exit_code"] = self.process.poll()
                self.launch_info["bridge_connected"] = True
                self._save_launch_info()
                return self.client
            except (ConnectionRefusedError, TimeoutError, OSError):
                if candidate:
                    candidate.close(False)
            except Exception as e:
                if candidate:
                    candidate.close(False)
                raise RuntimeError(
                    f"Worker {self.index+1} bridge verification failed: {e}\n"
                    + self.failure_details()
                ) from e
            code = self.process.poll()
            if code is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                    self.launch_info["launcher_exit_code"] = code
                    self._save_launch_info()
                    self.log(
                        f"Worker {self.index+1}: launch process ended with {exit_description(code)}; checking for a handed-off game process..."
                    )
                if time.monotonic() - exited_at >= exit_grace:
                    raise RuntimeError(
                        f"Worker {self.index+1} never connected after the launch process exited with {exit_description(code)}.\n"
                        "The failure is before training-save entry. Keep Steam running if this is the Steam edition.\n"
                        + self.failure_details()
                    )
            self.stop.wait(0.25)
        if self.stop.is_set():
            raise InterruptedError("Stopped during worker startup")
        raise RuntimeError(
            f"Worker {self.index+1} failed to connect.\n" + self.failure_details()
        )

    def failure_details(self):
        parts = []
        for name in (
            "flybrain-launch.json",
            "error_log.txt",
            "flybrain-process.log",
            "flybrain-game.txt",
            "log.txt",
            "flybrain-host-trace.txt",
        ):
            path = self.dest / name
            if path.is_file():
                if path.stat().st_mtime < self.launch_info.get("started_unix", 0) - 2:
                    continue  # A log from a previous attempt is not this failure.
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 10000))
                    content = stream.read().decode("utf-8", errors="replace")
                    parts.append(
                        f"{path}:\n" + (content if content.strip() else "(empty)")
                    )
        return "\n".join(parts) or f"Logs are in {self.dest}"

    def shutdown(self):
        had_process = self.process is not None or self.client is not None
        if getattr(self, "visibility_stop", None) is not None:
            self.visibility_stop.set()
        if self.client:
            # Authenticated quit also reaches a handed-off child after its
            # launcher has exited. Never kill another Celeste/Steam by name.
            self.client.close(quit_game=True)
            self.client = None
        if self.process:
            # Only terminate the process tree created by this Worker object.
            if os.name == "nt" and self.process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.handle:
            self.handle.close()
            self.handle = None
        if had_process:
            self.report(phase="stopped", pid=None)

    def recover(self, chapter):
        self.shutdown()
        self.restarts += 1
        if self.restarts > 3:
            raise RuntimeError(
                f"Worker {self.index+1} failed repeatedly; stopped after 3 recoveries."
            )
        self.log(f"Recovering worker {self.index+1}, attempt {self.restarts}/3")
        c = self.launch()
        return c.start(chapter, self.stop)


def movement_check(client, chapter, stop):
    state = client.start(chapter, stop)
    directions = set()
    # Explicitly scripted installation test. This is not training or a learned clear.
    for move in (2, 2, 2, 0, 0, 0):
        if stop.is_set():
            raise InterruptedError("Stopped")
        s = client.step([move, 1, 0, 0, 0, 0], 8)
        if not s["dead"] and (s["x"] - state["x"]) * (move - 1) > 0.2:
            directions.add(move)
        state = client.retry() if s["dead"] else s
    if directions != {0, 2}:
        raise RuntimeError(
            "Live movement check failed: left/right movement was not confirmed in both directions. Training was not started."
        )
    # Discard the probe and start a fresh attempt before collecting any training data.
    return client.start(chapter, stop)
