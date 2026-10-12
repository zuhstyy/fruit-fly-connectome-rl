"""Bounded, authenticated, localhost-only protocol for one Celeste worker."""

import json
import socket
import time
import numpy as np

BUTTON_HEADS = (3, 3, 2, 2, 2, 2)
DURATIONS = (1, 2, 4, 8, 12, 20, 30)  # simulation frames, at 60 Hz
HEADS = (*BUTTON_HEADS, len(DURATIONS))
LEGACY_SCHEMA = "mcns-ppo-v2-92-3x3x2x2x2x2"
SCHEMA = "mcns-ppo-v3-92-buttons-duration7"
VERSION = "4.1.0"
PROTOCOL = 6


class BridgeError(RuntimeError):
    pass


class ControlPaused(BridgeError):
    pass


class ControlStopped(BridgeError):
    pass


LEGACY_SENSORS = 92
SENSOR_COUNT = 156


def sensors(s):
    v = [
        s["goal_dx"] / 320,
        s["goal_dy"] / 180,
        s["vx"] / 240,
        s["vy"] / 240,
        float(s["grounded"]),
        float(s["wall_left"]),
        float(s["wall_right"]),
        s["dashes"] / 2,
        s["stamina"] / 110,
        (s["x"] - s["bounds_x"]) / max(1, s["bounds_width"]),
        (s["y"] - s["bounds_y"]) / max(1, s["bounds_height"]),
        *s["grid"],
    ]
    a = np.asarray(v, dtype=np.float32)
    if a.shape != (92,) or not np.isfinite(a).all():
        raise BridgeError("Invalid game observation")
    extra = np.zeros(64, dtype=np.float32)
    player_state = int(s.get("player_state", -1))
    if 0 <= player_state < 24:
        extra[player_state] = 1
    extra[24:32] = [
        float(s.get("in_control", True)),
        float(s.get("holding", False)),
        float(s.get("jump_held", False)),
        float(s.get("dash_held", False)),
        float(s.get("grab_held", False)),
        s.get("wind_x", 0) / 240,
        s.get("wind_y", 0) / 240,
        min(s.get("keys", 0), 2) / 2,
    ]
    # Six nearest relevant objects: position, velocity, and semantic class.
    objects = sorted(
        s.get("objects", []),
        key=lambda e: (e["x"] - s["x"]) ** 2 + (e["y"] - s["y"]) ** 2,
    )
    kinds = {
        "hazard": -1,
        "platform": 0.2,
        "refill": 0.4,
        "booster": 0.6,
        "key": 0.8,
        "switch": 0.9,
        "carry": 1,
        "door": -0.2,
        "goal": -0.4,
    }
    for i, e in enumerate(objects[:6]):
        extra[32 + i * 5 : 37 + i * 5] = [
            (e["x"] - s["x"]) / 320,
            (e["y"] - s["y"]) / 180,
            e.get("vx", 0) / 240,
            e.get("vy", 0) / 240,
            kinds.get(e.get("kind"), 0),
        ]
    extra[62] = float(s.get("navigation_valid", False))
    extra[63] = min(s.get("navigation_distance", 0), 2000) / 2000
    result = np.concatenate([np.clip(a, -1, 1), np.clip(extra, -1, 1)])
    if not np.isfinite(result).all():
        raise BridgeError("Invalid extended game observation")
    return result


def action_frames(a, fallback=4):
    return DURATIONS[int(a[6])] if len(a) == len(HEADS) else fallback


def action_fields(a, frames=None):
    if len(a) not in (6, 7) or any(
        not isinstance(x, (int, np.integer)) or not 0 <= x < n for x, n in zip(a, HEADS)
    ):
        raise ValueError("Invalid action")
    # Six-button actions are reserved for the scripted installation movement check.
    # A policy's seventh head always determines how long its buttons stay held.
    duration = action_frames(a, 4 if frames is None else frames)
    if type(duration) is not int or not 1 <= duration <= 30:
        raise ValueError("Invalid action duration")
    return dict(
        move_x=int(a[0]) - 1,
        move_y=int(a[1]) - 1,
        jump=bool(a[2]),
        dash=bool(a[3]),
        grab=bool(a[4]),
        interact=bool(a[5]),
        frames=duration,
    )


class Client:
    def __init__(self, port, token, timeout=130, log=print):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stream = self.socket.makefile("rwb")
        self.token, self.sequence, self.log = token, 0, log
        self.feedback = {}
        self.learning_info = {}
        self.world_cache = {}
        self.timings = {}
        try:
            h = self.read()
            if (
                h.get("type") != "hello"
                or h.get("protocol") != PROTOCOL
                or h.get("version") != VERSION
            ):
                raise BridgeError(
                    f'Wrong mod version: {h.get("version")}. Use the complete {VERSION} package.'
                )
        except BaseException:
            self.close(False)
            raise

    def read(self):
        line = self.stream.readline(65537)
        if not line or len(line) > 65536:
            raise BridgeError("Worker disconnected or sent an oversized response")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BridgeError("Invalid JSON response")
        if value.get("type") == "state" and "world_revision" in value:
            key = (value.get("chapter"), value.get("room"), value["world_revision"])
            fields = ("map", "route_graph", "portals")
            if value.get("map") is not None:
                self.world_cache[key] = {name: value.get(name) for name in fields}
                while len(self.world_cache) > 16:
                    self.world_cache.pop(next(iter(self.world_cache)))
            elif key in self.world_cache:
                value.update(self.world_cache[key])
            elif value.get("ready"):
                raise BridgeError("Missing room geometry for this world revision")
        return value

    def request(self, kind, **fields):
        self.sequence += 1
        packet = dict(
            type=kind,
            id=self.sequence,
            token=self.token,
            **self.feedback,
            **self.learning_info,
            **fields,
        )
        started = time.perf_counter()
        self.stream.write(
            (json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n").encode()
        )
        self.stream.flush()
        result = self.read()
        row = self.timings.setdefault(kind, dict(calls=0, seconds=0.0, max_ms=0.0))
        elapsed = time.perf_counter() - started
        row["calls"] += 1
        row["seconds"] += elapsed
        row["max_ms"] = max(row["max_ms"], elapsed * 1000)
        if result.get("id") != self.sequence:
            raise BridgeError("Request/response mismatch")
        if result.get("type") == "error":
            if result.get("code") == "control_paused":
                raise ControlPaused(result.get("message", "Bot control is OFF"))
            if result.get("code") == "control_stopped":
                raise ControlStopped(result.get("message", "Stopped from the game"))
            raise BridgeError(result.get("message", "Unknown game error"))
        return result

    def performance(self):
        return {
            name: {**row, "mean_ms": row["seconds"] * 1000 / max(1, row["calls"])}
            for name, row in self.timings.items()
        }

    def start(self, chapter, stop=None):
        deadline = time.monotonic() + 120
        last_scene = None
        self.log(f"Opening training save and chapter {chapter} automatically...")
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise InterruptedError("Stopped")
            try:
                status = self.request("status")
                scene = status.get("scene")
                if scene != last_scene:
                    self.log(f'Game startup: {scene or "initializing"}')
                    last_scene = scene
                return self.request("start", chapter=chapter)
            except ControlPaused:
                self.wait_until_on(stop)
                deadline = time.monotonic() + 120
            except BridgeError as e:
                if "not_ready:" not in str(e):
                    raise
                if stop is not None:
                    stop.wait(0.5)
                else:
                    time.sleep(0.5)
        raise BridgeError(
            f"Worker did not finish loading in 120 seconds (last scene: {last_scene}). See the worker log."
        )

    def wait_until_on(self, stop=None):
        self.log(
            "Bot control OFF. Manual inputs work. Turn ON to restart this chapter and resume learning."
        )
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("Stopped while bot control was OFF")
            status = self.request("status")
            if status.get("emergency_stop"):
                raise ControlStopped("Emergency stop from game")
            if not status.get("control_paused", False):
                return
            if stop is not None:
                stop.wait(0.25)
            else:
                time.sleep(0.25)

    def set_feedback(self, total, last, distance, score=0.0, event="", immediate=True):
        self.feedback = dict(
            reward_ready=True,
            reward_total=float(total),
            reward_last=float(last),
            reward_distance=float(distance),
            reward_score=float(score),
            reward_event=str(event)[:100],
        )
        if immediate:
            self.request("feedback")

    def step(self, action, frames=None):
        return self.request("step", **action_fields(action, frames))

    def capture(self, slot=0):
        return self.request("capture", slot=slot)

    def restore(self, slot=0):
        return self.request("restore", slot=slot)

    def clear(self, slot=0):
        return self.request("clear", slot=slot)

    def practice_mode(self, enabled):
        return self.request("practice", enabled=bool(enabled))

    def retry(self, force=False):
        return self.request("retry", force=force)

    def close(self, release=True, quit_game=False):
        if self.stream is None:
            return
        try:
            if quit_game or release:
                self.socket.settimeout(0.5 if quit_game else 0.2)
                self.request("quit" if quit_game else "release")
        except Exception:
            pass
        finally:
            try:
                self.stream.close()
            except OSError:
                pass
            finally:
                self.socket.close()
                self.stream = None
