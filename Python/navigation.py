"""Turn the game's collision map into a route hint and a stable objective.

The field goes around static walls. It does not claim that every free-cell path
is physically reachable by jumping. Moving objects remain separate sensors.
"""

from collections import deque
import math
import numpy as np


class Navigation:
    def __init__(self):
        self.cache_key = None
        self.field = None
        self.objective = None
        self.destination = None
        self.portal_point = None
        self.failed_targets = {}
        self.visited = set()
        self.chapter = None

    def reset(self):
        self.cache_key = self.field = self.objective = None
        self.destination = self.portal_point = None

    def note_failure(self, state):
        key = (state["chapter"], state["room"], self.destination)
        self.failed_targets[key] = self.failed_targets.get(key, 0) + 1
        self.destination = self.portal_point = None
        self.cache_key = None

    def portal_objective(self, state):
        if state["chapter"] != self.chapter:
            self.chapter = state["chapter"]
            self.visited.clear()
            self.failed_targets.clear()
        self.visited.add(state["room"])
        portals = state.get("portals") or []
        if not portals:
            return None
        if self.destination is None or not any(
            p["room"] == self.destination for p in portals
        ):
            graph = {r["name"]: r["neighbors"] for r in state.get("route_graph") or []}
            terminals = [
                r["name"]
                for r in state.get("route_graph") or []
                if r.get("terminal_hint")
            ]
            distances = {}
            queue = deque((name, 0) for name in terminals)
            while queue:
                name, distance = queue.popleft()
                if name in distances:
                    continue
                distances[name] = distance
                queue.extend(
                    (n, distance + 1) for n in graph.get(name, []) if n not in distances
                )

            def cost(portal):
                room = portal["room"]
                failures = self.failed_targets.get(
                    (state["chapter"], state["room"], room), 0
                )
                # The Prologue car is a known earlier room, not a new frontier.
                car = 1000 if state["chapter"] == 0 and room == "-1" else 0
                route = distances.get(room, 10 if room in self.visited else 0)
                return (
                    car
                    + route
                    + min(failures, 10) * 4
                    + (0.5 if room in self.visited else 0)
                )

            chosen = min(portals, key=cost)
            self.destination = chosen["room"]
            self.portal_point = chosen["x"], chosen["y"]
        state["exit_room"] = self.destination
        state["exit_forward"] = True  # Selected task, not a claim about map order.
        return "portal:" + self.destination, *self.portal_point

    def choose_objective(self, state):
        """Prefer a visible required key before a nearby locked door."""
        objects = state.get("objects", [])
        carried = [
            e
            for e in objects
            if e.get("kind") == "carry" and e.get("name") == "TheoCrystal"
        ]
        if carried and not state.get("holding"):
            return "carry:TheoCrystal", carried[0]["x"], carried[0]["y"]
        keys = [e for e in objects if e.get("kind") == "key"]
        doors = [
            e
            for e in objects
            if e.get("kind") == "door" and "Lock" in e.get("name", "")
        ]
        if keys and doors and not state.get("keys", 0):
            key = min(
                keys,
                key=lambda e: (e["x"] - state["x"]) ** 2 + (e["y"] - state["y"]) ** 2,
            )
            return "key", key["x"], key["y"]
        portal = self.portal_objective(state)
        if portal is not None:
            return portal
        if state.get("exit_forward") and state.get("exit_id"):
            return (
                state["exit_id"],
                state["x"] + state.get("raw_goal_dx", state["goal_dx"]),
                state["y"] + state.get("raw_goal_dy", state["goal_dy"]),
            )
        # A terminal room can have no geometric exit. Expose a visible final
        # object as a hint, but only Celeste's completion event counts as a win.
        goals = [e for e in objects if e.get("kind") == "goal"]
        if goals:
            goal = min(
                goals,
                key=lambda e: (e["x"] - state["x"]) ** 2 + (e["y"] - state["y"]) ** 2,
            )
            return "object:" + goal["name"], goal["x"], goal["y"]
        return None

    def annotate(self, state):
        state = dict(state)
        state.setdefault("raw_goal_dx", state.get("goal_dx", 0))
        state.setdefault("raw_goal_dy", state.get("goal_dy", 0))
        state["navigation_revision"] = 1
        state["navigation_valid"] = False
        state["objective"] = "unknown"
        if not state.get("ready", True) or state.get("dead") or state.get("completed"):
            return state
        objective = self.choose_objective(state)
        room_map = state.get("map")
        if objective is None or not room_map:
            return state
        name, gx, gy = objective
        state["objective_x"], state["objective_y"] = gx, gy
        cell, width, height = room_map["cell"], room_map["width"], room_map["height"]
        if not (
            cell >= 1
            and 1 <= width <= 128
            and 1 <= height <= 128
            and len(room_map["tiles"]) == width * height
        ):
            raise ValueError("Invalid collision map")
        tx = min(width - 1, max(0, int((gx - room_map["x"]) / cell)))
        ty = min(height - 1, max(0, int((gy - room_map["y"]) / cell)))
        cache_key = (state["chapter"], state["room"], name, tx, ty, room_map["tiles"])
        if cache_key != self.cache_key:
            blocked = np.asarray(list(room_map["tiles"])).reshape(height, width) != "."
            self.field = distance_field(blocked, (tx, ty))
            self.cache_key = cache_key
        x = min(width - 1, max(0, int((state["x"] - room_map["x"]) / cell)))
        y = min(height - 1, max(0, int((state["y"] - room_map["y"]) / cell)))
        distance = float(self.field[y, x])
        if not math.isfinite(distance):
            state["objective"] = "unreachable static hint: " + name
            return state
        wx, wy = x, y
        for _ in range(3):
            candidates = [
                (self.field[ny, nx], nx, ny)
                for nx, ny in neighbors(wx, wy, width, height)
            ]
            best = min(candidates, default=(self.field[wy, wx], wx, wy))
            if best[0] >= self.field[wy, wx]:
                break
            _, wx, wy = best
        # Preserve continuous progress within a cell; a coarse field alone
        # would give no feedback until the player crossed its cell boundary.
        center_x = room_map["x"] + (x + 0.5) * cell
        center_y = room_map["y"] + (y + 0.5) * cell
        waypoint_x = room_map["x"] + (wx + 0.5) * cell
        waypoint_y = room_map["y"] + (wy + 0.5) * cell
        offset = math.hypot(
            state["x"] - waypoint_x, state["y"] - waypoint_y
        ) - math.hypot(center_x - waypoint_x, center_y - waypoint_y)
        state.update(
            navigation_valid=True,
            navigation_distance=max(0, distance * cell + offset),
            navigation_id=f'{state["chapter"]}:{state["room"]}:{name}:{tx}:{ty}',
            objective=name,
            goal_dx=waypoint_x - state["x"],
            goal_dy=waypoint_y - state["y"],
        )
        return state


def neighbors(x, y, width, height):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def distance_field(blocked, target):
    height, width = blocked.shape
    tx, ty = target
    # Exit centers may lie just inside a boundary tile. Use the nearest free
    # cell within two tiles, rather than treating arbitrary far cells as exits.
    candidates = [
        (abs(x - tx) + abs(y - ty), x, y)
        for y in range(max(0, ty - 2), min(height, ty + 3))
        for x in range(max(0, tx - 2), min(width, tx + 3))
        if not blocked[y, x]
    ]
    result = np.full((height, width), np.inf)
    if not candidates:
        return result
    _, tx, ty = min(candidates)
    result[ty, tx] = 0
    queue = deque([(tx, ty)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in neighbors(x, y, width, height):
            if not blocked[ny, nx] and not math.isfinite(result[ny, nx]):
                result[ny, nx] = result[y, x] + 1
                queue.append((nx, ny))
    return result
