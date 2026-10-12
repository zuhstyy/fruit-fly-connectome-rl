"""Share room practice results and schedule useful work between rollouts."""

import threading
import time
from collections import deque
from runtime import atomic_json, read_json


class Curriculum:
    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.RLock()
        previous = read_json(folder / "curriculum.json")
        self.rooms = previous.get("rooms", {})
        self.events = deque(maxlen=200)

    @staticmethod
    def key(chapter, room):
        return str(chapter) + ":" + room

    def record(self, chapter, room, success, frames, mode="practice"):
        key = self.key(chapter, room)
        with self.lock:
            row = self.rooms.setdefault(
                key,
                dict(
                    chapter=chapter,
                    room=room,
                    attempts=0,
                    clears=0,
                    evaluation_attempts=0,
                    evaluation_clears=0,
                    best_frames=None,
                ),
            )
            if mode == "evaluation":
                row["evaluation_attempts"] += 1
                row["evaluation_clears"] += int(success)
            else:
                row["attempts"] += 1
                row["clears"] += int(success)
            if success:
                row["best_frames"] = min(row["best_frames"] or frames, frames)
            row["last_unix"] = time.time()
            self.events.append(
                dict(
                    chapter=chapter,
                    room=room,
                    success=success,
                    mode=mode,
                    frames=frames,
                )
            )

    def choose(self, worker, chapters, current, rng):
        """Most workers explore the frontier; some rehearse earlier chapters."""
        with self.lock:
            rows = list(self.rooms.values())
        if not rows or worker % 4 != 3:
            return current
        candidates = [r for r in rows if r["chapter"] <= current]
        if not candidates:
            return current
        # Failures and low clear rates get more practice. Keep every stage a
        # possible choice, so learned movement is periodically rehearsed.
        weights = [
            1
            + (r["attempts"] - r["clears"]) / max(1, r["attempts"])
            + 1 / (1 + r["clears"])
            for r in candidates
        ]
        import numpy as np

        chosen = candidates[
            int(rng.choice(len(candidates), p=np.asarray(weights) / sum(weights)))
        ]
        return (
            chapters.index(chosen["chapter"])
            if chosen["chapter"] in chapters
            else current
        )

    def summary(self):
        with self.lock:
            rows = [dict(r) for r in self.rooms.values()]
        return dict(
            rooms=rows,
            attempted_rooms=len(rows),
            room_attempts=sum(r["attempts"] for r in rows),
            room_clears=sum(r["clears"] for r in rows),
            evaluation_attempts=sum(r["evaluation_attempts"] for r in rows),
            evaluation_clears=sum(r["evaluation_clears"] for r in rows),
        )

    def save(self):
        with self.lock:
            atomic_json(
                self.folder / "curriculum.json",
                dict(rooms=self.rooms, saved_unix=time.time()),
            )
