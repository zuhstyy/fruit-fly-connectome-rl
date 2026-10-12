"""Local LEARN button. No website, account, or cloud training needed."""

from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Python"))
from controls import Controls
from mod_install import repair_original


class App:
    def __init__(self, root):
        self.root, self.process = root, None
        self.events = queue.Queue()
        root.title("FlyBrain Celeste 4.1 — LEARN")
        root.geometry("960x800")
        root.minsize(880, 720)
        self.controls = Controls(ROOT)
        self.stopping = False
        try:
            config_path = ROOT / "config.json"
            if not config_path.exists():
                config_path = ROOT / "config.example.json"
            config = json.loads(config_path.read_text())
        except (OSError, ValueError):
            config = {}
        self.game = tk.StringVar(value=config.get("game", ""))
        self.workers = tk.IntVar(value=config.get("workers", 2))
        self.background = tk.BooleanVar(value=config.get("background", True))
        self.ticks = tk.IntVar(value=30)
        self.backend = tk.StringVar(value=config.get("backend", "auto"))
        self.fast = tk.BooleanVar(value=config.get("fast_simulation", True))
        self.hours = tk.DoubleVar(value=config.get("hours", 8))
        self.frames = tk.IntVar(value=config.get("frames", 4))
        self.rebuild = tk.BooleanVar(value=False)
        self.enabled = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="OFF — no training run")
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Fly Brain / Celeste", font=("Segoe UI", 22)).pack(
            anchor="w"
        )
        ttk.Label(
            frame,
            text="Practice + automatic search + fly learning • Prologue through Summit",
        ).pack(anchor="w", pady=(4, 16))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.game).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose Celeste folder", command=self.browse).pack(
            side="right", padx=(8, 0)
        )
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=12)
        for title, var, lo, hi in [
            ("Live games", self.workers, 1, 64),
            ("Hours", self.hours, 0.1, 720),
        ]:
            ttk.Label(row, text=title).pack(side="left", padx=(0, 6))
            ttk.Spinbox(row, textvariable=var, from_=lo, to=hi, width=5).pack(
                side="left", padx=(0, 18)
            )
        ttk.Label(row, text="Learner").pack(side="left", padx=6)
        ttk.Combobox(
            row,
            textvariable=self.backend,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=8,
        ).pack(side="left")
        ttk.Label(
            frame,
            text="The model learns how long to hold: 1, 2, 4, 8, 12, 20 or 30 game frames. Consecutive holds stay pressed.",
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Live games is the actual requested process count. Startup is staged; MONITOR shows ready / requested.",
        ).pack(anchor="w", pady=4)
        ttk.Checkbutton(
            frame, text="Hide game windows (background)", variable=self.background
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame,
            text="Fast simulation (fixed 60 Hz physics, uncapped worker loop)",
            variable=self.fast,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="64 full game processes require enough memory/CPU/GPU. Resource or startup limits are reported explicitly.",
        ).pack(anchor="w")
        control_box = ttk.LabelFrame(frame, text="Learning ON / OFF", padding=10)
        control_box.pack(fill="x", pady=(12, 0))
        self.toggle = ttk.Checkbutton(
            control_box,
            text="Bot control ON",
            variable=self.enabled,
            command=self.toggle_control,
            state="disabled",
        )
        self.toggle.pack(side="left")
        ttk.Label(control_box, textvariable=self.mode).pack(side="left", padx=16)
        ttk.Label(
            frame,
            text="OFF = manual play. ON resumes from this chapter’s start. In the game: F9 toggles, F10 stops all.",
        ).pack(anchor="w", pady=6)
        monitor_row = ttk.Frame(frame)
        monitor_row.pack(fill="x", pady=8)
        ttk.Button(
            monitor_row, text="LIVE MONITOR / show a game", command=self.monitor
        ).pack(side="left")
        ttk.Button(
            monitor_row, text="SHOW ALL GAMES", command=lambda: self.monitor(True)
        ).pack(side="left", padx=8)
        ttk.Button(
            monitor_row, text="TOOLS / GPU / benchmark", command=self.tools
        ).pack(side="left", padx=8)
        ttk.Button(monitor_row, text="Save settings", command=self.save_settings).pack(
            side="right"
        )
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=12)
        self.learn = ttk.Button(buttons, text="LEARN", command=self.start)
        self.learn.pack(side="left", ipadx=35, ipady=8)
        self.stop_button = ttk.Button(
            buttons, text="STOP & SAVE", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=8, ipady=8)
        ttk.Button(buttons, text="VERIFY progress", command=self.verify).pack(
            side="right", padx=6
        )
        ttk.Button(buttons, text="Open results", command=self.results).pack(
            side="right", ipady=8
        )
        self.evaluate = ttk.Button(
            buttons, text="EVALUATE saved model", command=lambda: self.start(True)
        )
        self.evaluate.pack(side="left", padx=8, ipady=8)
        repair_row = ttk.Frame(frame)
        repair_row.pack(fill="x", pady=(0, 8))
        self.repair_button = ttk.Button(
            repair_row, text="Repair old / duplicate FlyBrain mods", command=self.repair
        )
        self.repair_button.pack(side="left")
        self.play_button = ttk.Button(
            repair_row, text="PLAY normally (AI off)", command=self.play_normal
        )
        self.play_button.pack(side="left", padx=8)
        ttk.Label(
            frame,
            text="Repair moves old FlyBrain packages to Celeste/FlyBrain-disabled. LEARN also runs this repair.",
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame,
            text="Advanced: rebuild C# from source (requires .NET SDK 8+)",
            variable=self.rebuild,
        ).pack(anchor="w", pady=(4, 8))
        self.output = tk.Text(
            frame, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self.output.pack(fill="both", expand=True)
        self.append(
            "Version 4.1 includes the compiled C# bridge. Choose Everest Celeste, then LEARN.\nStart with 2 hidden games. Fast simulation removes the display frame limit while keeping 1/60-second physics.\nGPU support is optional under TOOLS. Auto measures it and retains CPU when faster.\nCopy your old training folder here to resume. Extract outside Celeste; do not put this outer ZIP in Mods.\n"
        )
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def browse(self):
        folder = filedialog.askdirectory(
            title="Select the folder containing Celeste.exe and Celeste.dll"
        )
        if folder:
            self.game.set(folder)

    def append(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        if int(self.output.index("end-1c").split(".")[0]) > 5000:
            self.output.delete("1.0", "1000.0")
        self.output.see("end")
        self.output.configure(state="disabled")

    def save_settings(self):
        try:
            config = self.make_config()
            from train import validate_config
            from workers import validate_game

            validate_config(config)
            validate_game(config["game"])
            (ROOT / "config.json").write_text(json.dumps(config, indent=2))
            self.append(
                "Settings saved. Benchmark and evaluation can now run from TOOLS.\n"
            )
        except Exception as error:
            messagebox.showerror("Cannot save", str(error))

    def tools(self):
        subprocess.Popen(
            [sys.executable, str(ROOT / "launch.py"), "tools"],
            cwd=ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )

    def make_config(self):
        return dict(
            game=self.game.get(),
            workers=self.workers.get(),
            ticks=self.ticks.get(),
            hours=self.hours.get(),
            frames=4,
            background=self.background.get(),
            startup_parallelism=2,
            batch_decisions=2048,
            practice=True,
            teacher_interval=4,
            teacher_decisions=2,
            teacher_seconds=3.0,
            imitation_decisions=128,
            evaluation_interval=25,
            evaluation_frames=900,
            rollout=256,
            sequences_per_batch=16,
            backend=self.backend.get(),
            fast_simulation=self.fast.get(),
            episode_steps=3600,
            stuck_steps=240,
            seed=17,
            chapters=list(range(8)),
            rebuild=self.rebuild.get(),
        )

    def start(self, evaluate=False):
        if self.process is not None:
            return
        try:
            config = self.make_config()
            from train import validate_config
            from workers import validate_game

            validate_config(config)
            validate_game(config["game"])
            (ROOT / "config.json").write_text(json.dumps(config, indent=2))
            from runtime import RunLock

            with RunLock(ROOT):
                pass
            self.controls.new_run()
            self.stopping = False
            command = [
                sys.executable,
                "-u",
                str(ROOT / ("Python/evaluate.py" if evaluate else "Python/train.py")),
                "--config",
                str(ROOT / "config.json"),
            ]
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            def collect(p):
                (ROOT / "training").mkdir(exist_ok=True)
                with (ROOT / "training/run.log").open("a", encoding="utf-8") as log:
                    for line in p.stdout:
                        log.write(line)
                        log.flush()
                        self.events.put(line)
                p.wait()
                self.events.put(("exit", p.returncode))

            threading.Thread(target=collect, args=(self.process,), daemon=True).start()
            if not evaluate:
                self.monitor()
            self.learn.configure(state="disabled")
            self.evaluate.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.toggle.configure(state="normal")
            self.enabled.set(True)
            self.mode.set("ON — opening training save / checking movement")
            self.repair_button.configure(state="disabled")
            self.play_button.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Cannot start", str(e))

    def stop(self):
        self.controls.stop()
        self.stopping = True
        self.enabled.set(False)
        self.mode.set("OFF — stopping and saving")
        self.toggle.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.append(
            "Stopping after in-flight game requests; saving the completed update.\n"
        )

    def toggle_control(self):
        if self.process is None or self.stopping:
            self.enabled.set(False)
            return
        if self.enabled.get():
            self.controls.resume()
            self.mode.set("ON — restarting current chapter / learning")
            self.append(
                "Bot ON. Resuming from the current chapter’s start; learned weights are retained.\n"
            )
        else:
            self.controls.pause()
            self.mode.set("OFF — manual keyboard/controller control")
            self.append(
                "Bot OFF. The worker stays open for manual play; manual movement is excluded from learning.\n"
            )

    def repair(self):
        try:
            from workers import validate_game

            game = validate_game(self.game.get())
            backup = repair_original(game, lambda line: self.append(line + "\n"))
            self.append("Repair complete. Your save files were not changed.\n")
            return game
        except Exception as e:
            messagebox.showerror("Repair failed", str(e))
            return None

    def play_normal(self):
        game = self.repair()
        if game is None:
            return
        try:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("FLYBRAIN_") and k != "EVEREST_SAVEPATH"
            }
            subprocess.Popen([str(game / "Celeste.exe")], cwd=game, env=env)
            self.append(
                "Opened the normal game with AI off. Choose your usual save there.\n"
            )
        except OSError as e:
            messagebox.showerror("Could not open Celeste", str(e))

    def poll(self):
        while not self.events.empty():
            value = self.events.get_nowait()
            if isinstance(value, tuple):
                self.append(f"Run exited with code {value[1]}.\n")
                self.process = None
                self.stopping = False
                self.learn.configure(state="normal")
                self.evaluate.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.toggle.configure(state="disabled")
                self.enabled.set(False)
                self.mode.set(
                    "OFF — stopped" if value[1] == 0 else "OFF — error; see log below"
                )
                self.repair_button.configure(state="normal")
                self.play_button.configure(state="normal")
            else:
                self.append(value)
        if self.process is not None and not self.stopping:
            paused = self.controls.paused
            if self.enabled.get() == paused:
                self.enabled.set(not paused)
                self.mode.set(
                    "OFF — manual keyboard/controller control"
                    if paused
                    else "ON — restarting current chapter / learning"
                )
        self.root.after(100, self.poll)

    def monitor(self, show=False):
        subprocess.Popen(
            [sys.executable, str(ROOT / "Python/monitor.py")]
            + (["--show"] if show else []),
            cwd=ROOT,
        )

    def verify(self):
        subprocess.Popen([sys.executable, str(ROOT / "Python/verify.py")], cwd=ROOT)

    def results(self):
        path = ROOT / "training"
        path.mkdir(exist_ok=True)
        os.startfile(path)

    def close(self):
        if self.process is not None:
            self.stop()
            self.append("Wait for workers to stop, then close this window.\n")
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
