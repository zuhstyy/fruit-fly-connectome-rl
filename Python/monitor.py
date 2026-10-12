"""Live desktop monitor; independent of launcher and game-window visibility."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk
from controls import Controls
from runtime import read_json, is_running, change_windows

ROOT = Path(__file__).resolve().parents[1]


def display_status(root, now=None):
    value = read_json(Path(root) / "training/live.json")
    age = (time.time() if now is None else now) - value.get("heartbeat", 0)
    value["fresh"] = bool(value and age <= 10 and is_running(root))
    if not value["fresh"]:
        value["live_game_processes"] = 0
        value["ready_workers"] = 0
    return value


def action_text(row):
    a = row.get("action", [])
    if len(a) < 6:
        return "waiting"
    names = ["left", "neutral", "right"]
    vertical = ["up", "neutral", "down"]
    buttons = [
        name for name, held in zip(("jump", "dash", "grab", "interact"), a[2:6]) if held
    ]
    return " + ".join([names[a[0]], vertical[a[1]], *buttons])


class Monitor:
    def __init__(self, root, folder=ROOT, show=False):
        self.root, self.folder = root, Path(folder)
        root.title("FlyBrain Celeste 4.1 — live monitor")
        root.geometry("1250x780")
        root.minsize(1000, 600)
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        box = ttk.Frame(root, padding=18)
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="Fly Brain / Live monitor", font=("Segoe UI", 22)).pack(
            anchor="w"
        )
        self.heading = tk.StringVar(value="Reading live worker status…")
        ttk.Label(box, textvariable=self.heading, font=("Segoe UI", 14)).pack(
            anchor="w", pady=(8, 4)
        )
        self.details = tk.StringVar()
        ttk.Label(box, textvariable=self.details, wraplength=1160).pack(anchor="w")
        self.notice = tk.StringVar()
        ttk.Label(
            box, textvariable=self.notice, wraplength=1160, foreground="#a64016"
        ).pack(anchor="w", pady=6)
        controls = ttk.Frame(box)
        controls.pack(fill="x", pady=10)
        for label, command in [
            ("Show selected", lambda: self.visibility(True, True)),
            ("Hide selected", lambda: self.visibility(False, True)),
            ("SHOW ALL GAMES", lambda: self.visibility(True)),
            ("HIDE ALL", lambda: self.visibility(False)),
            ("VERIFY progress", self.verify),
            ("Open logs", self.logs),
            ("STOP & SAVE", self.stop),
        ]:
            ttk.Button(controls, text=label, command=command).pack(
                side="left", padx=(0, 6)
            )
        ttk.Label(
            box,
            text="Each row is one requested Celeste process. Double-click a row to restore that game. Closing this monitor keeps training running.",
        ).pack(anchor="w", pady=(0, 8))
        table = ttk.Frame(box)
        table.pack(fill="both", expand=True)
        columns = (
            "worker",
            "pid",
            "phase",
            "chapter",
            "room",
            "points",
            "actions",
            "hold",
            "response",
            "window",
            "memory",
        )
        self.tree = ttk.Treeview(
            table, columns=columns, show="headings", selectmode="extended"
        )
        widths = (55, 70, 90, 65, 75, 85, 85, 100, 90, 85, 75)
        titles = (
            "Game",
            "PID",
            "State",
            "Chapter",
            "Room",
            "Points",
            "Decisions",
            "Hold actual/asked",
            "Response age",
            "Visible / found",
            "RAM MB",
        )
        for col, width, title in zip(columns, widths, titles):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, minwidth=45)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda event: self.visibility(True, True))
        self.tree.bind("<<TreeviewSelect>>", lambda event: self.selection())
        self.tree.tag_configure("failed", foreground="#b43821")
        self.tree.tag_configure("blocked", foreground="#965d15")
        self.selected = tk.StringVar(
            value="Select a worker to inspect its action, target and latest event."
        )
        ttk.Label(box, textvariable=self.selected, wraplength=1160).pack(
            anchor="w", pady=12
        )
        self.feedback = tk.StringVar()
        ttk.Label(box, textvariable=self.feedback, wraplength=1160).pack(anchor="w")
        self.rows = {}
        self.refresh()
        if show:
            root.after(250, lambda: self.visibility(True))

    def visibility(self, show, selected=False):
        workers = [int(i) for i in self.tree.selection()] if selected else None
        if selected and not workers:
            self.feedback.set("Select one or more worker rows first.")
            return
        try:
            self.feedback.set(change_windows(self.folder, show, workers))
        except Exception as e:
            self.feedback.set("Window command failed: " + str(e))

    def stop(self):
        Controls(self.folder).stop()
        self.feedback.set(
            "Stopping after in-flight requests; saving the completed update."
        )

    def verify(self):
        subprocess.Popen(
            [sys.executable, str(self.folder / "Python/verify.py")], cwd=self.folder
        )

    def logs(self):
        path = self.folder / "training"
        path.mkdir(exist_ok=True)
        if os.name == "nt":
            os.startfile(path)

    def selection(self):
        selected = self.tree.selection()
        if not selected:
            return
        row = self.rows.get(int(selected[0]), {})
        target = row.get("exit_room") or "not found"
        distance = row.get("navigation_distance")
        self.selected.set(
            f'Worker {row.get("worker")}: {action_text(row)} | target {target}'
            + (f" / {distance:.0f} px" if isinstance(distance, (int, float)) else "")
            + f' | {row.get("event","")} | Window: {row.get("window",{}).get("message","waiting")}'
            + f' | Objective: {row.get("objective","unknown")} | actual velocity {row.get("actual_vx",0):.1f}, {row.get("actual_vy",0):.1f}'
            + f' | {row.get("practice_message","")} | clears {row.get("room_clears",0)} | practice resets {row.get("practice_resets",0)}'
        )

    def refresh(self):
        value = display_status(self.folder)
        requested = value.get(
            "requested_workers",
            read_json(self.folder / "config.json").get("workers", 64),
        )
        alive = value.get("live_game_processes", 0)
        ready = value.get("ready_workers", 0)
        phase = value.get("phase", "not started")
        self.heading.set(
            f"{alive} / {requested} real games alive  •  {ready} ready  •  {phase.upper()}"
            if value["fresh"]
            else "No active learner with fresh status — showing last recorded rows"
        )
        self.details.set(
            f'Update {value.get("iteration",0)}  |  {value.get("decisions",0):,} decisions  |  '
            f'{value.get("decisions_per_second",0):.1f} decisions/sec  |  '
            f'{value.get("learned_edges",0)}/3326 fly connections changed  |  '
            f'Last optimizer update: {value.get("update_seconds",0):.3f}s | '
            f'{value.get("success_routes",0)} saved solutions | '
            f'{value.get("curriculum",{}).get("evaluation_clears",0)}/{value.get("curriculum",{}).get("evaluation_attempts",0)} room tests passed'
            f' | {value.get("performance",{}).get("frames_per_second",0):.0f} controlled frames/sec'
            f' | bottleneck: {value.get("performance",{}).get("largest_phase","measuring")}'
            f' | backend: {value.get("backend",{}).get("selected","CPU")}'
        )
        self.notice.set(
            value.get("capacity_message")
            or value.get(
                "message",
                "Open START.cmd or TOOLS > Start with saved settings from this same folder.",
            )
        )
        self.rows = {r["worker"]: r for r in value.get("workers", [])}
        for index, row in self.rows.items():
            age = time.time() - row.get("last_response", time.time())
            window = row.get("window", {})
            values = (
                index,
                row.get("pid") or "—",
                row.get("phase"),
                row.get("chapter", "—"),
                row.get("room", "—"),
                f'{row.get("score",0):+.2f}',
                row.get("actions", 0),
                f'{row.get("action_frames",0)}/{row.get("requested_frames",0)} f',
                f"{max(0,age):.1f}s" if row.get("last_response") else "—",
                f'{window.get("visible",0)}/{window.get("windows",0)}',
                row.get("memory_mb", "—"),
            )
            key = str(index)
            tags = (row.get("phase", ""),)
            if self.tree.exists(key):
                self.tree.item(key, values=values, tags=tags)
            else:
                self.tree.insert("", "end", iid=key, values=values, tags=tags)
        for key in self.tree.get_children():
            if int(key) not in self.rows:
                self.tree.delete(key)
        self.selection()
        self.root.after(1000, self.refresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    Monitor(root, show=args.show)
    root.mainloop()
