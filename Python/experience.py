"""Bounded, persistent memory of actual successful room attempts."""

from pathlib import Path
import hashlib
import json
import threading
import time
import numpy as np
from learning import replace_checkpoint
from protocol import HEADS, SENSOR_COUNT


class SuccessMemory:
    """Keep up to three successful routes per room; never store game binaries."""

    def __init__(self, folder, graph_hash, limit=256):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.graph_hash, self.limit = graph_hash, limit
        self.lock = threading.RLock()
        self.entries = {}
        self.rejected = 0
        for path in sorted(self.folder.glob("*.npz"))[-limit:]:
            try:
                trajectory = self.read(path)
                self.entries[path.stem] = trajectory
            except (OSError, ValueError, KeyError):
                self.rejected += 1

    def read(self, path):
        with np.load(path, allow_pickle=False) as record:
            metadata = json.loads(str(record["metadata"]))
            observations = record["observations"].copy()
            actions = record["actions"].copy()
            resets = record["resets"].copy()
        if metadata.get("graph_hash") != self.graph_hash or metadata.get(
            "source"
        ) not in {"agent_room_clear", "search_room_clear", "verified_demonstration"}:
            raise ValueError("Unverified or incompatible success memory")
        if not (
            1 <= len(actions) <= 2048
            and observations.shape == (len(actions), SENSOR_COUNT)
            and actions.shape == (len(actions), len(HEADS))
            and resets.shape == (len(actions),)
            and resets.dtype == np.bool_
            and resets[0]
            and np.isfinite(observations).all()
            and np.issubdtype(actions.dtype, np.integer)
        ):
            raise ValueError("Invalid success trajectory")
        for head, choices in enumerate(HEADS):
            if ((actions[:, head] < 0) | (actions[:, head] >= choices)).any():
                raise ValueError("Invalid demonstration action")
        return dict(
            metadata=metadata, observations=observations, actions=actions, resets=resets
        )

    def add(
        self,
        records,
        chapter,
        room,
        destination,
        completed=False,
        worker=0,
        source=None,
    ):
        if not 1 <= len(records) <= 2048 or not records[0]["reset"]:
            return False
        observations = np.asarray([r["observation"] for r in records], dtype=np.float32)
        actions = np.asarray([r["action"] for r in records], dtype=np.int64)
        resets = np.asarray([r["reset"] for r in records], dtype=bool)
        frames = sum(r["frames"] for r in records)
        source = source or (
            "search_room_clear"
            if any(r.get("teacher") for r in records)
            else "agent_room_clear"
        )
        if source not in {
            "agent_room_clear",
            "search_room_clear",
            "verified_demonstration",
        }:
            raise ValueError("Unverified success source")
        metadata = dict(
            source=source,
            chapter=int(chapter),
            room=room,
            destination=destination,
            chapter_completed=bool(completed),
            frames=frames,
            worker=worker,
            graph_hash=self.graph_hash,
            recorded_unix=time.time(),
            observation_schema=SENSOR_COUNT,
        )
        key = hashlib.sha256(observations.tobytes() + actions.tobytes()).hexdigest()[
            :24
        ]
        trajectory = dict(
            metadata=metadata, observations=observations, actions=actions, resets=resets
        )
        with self.lock:
            if key in self.entries:
                return False
            peers = [
                (k, t)
                for k, t in self.entries.items()
                if (t["metadata"]["chapter"], t["metadata"]["room"]) == (chapter, room)
            ]
            if len(peers) >= 3:
                worst_key, worst = max(
                    peers, key=lambda pair: pair[1]["metadata"]["frames"]
                )
                if frames >= worst["metadata"]["frames"]:
                    return False
                self.remove(worst_key)
            if len(self.entries) >= self.limit:
                oldest = min(
                    self.entries,
                    key=lambda k: self.entries[k]["metadata"]["recorded_unix"],
                )
                self.remove(oldest)
            path = self.folder / (key + ".npz")
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    observations=observations,
                    actions=actions,
                    resets=resets,
                    metadata=np.array(json.dumps(metadata)),
                )
            replace_checkpoint(temp, path)
            self.entries[key] = trajectory
            return True

    def remove(self, key):
        self.entries.pop(key, None)
        (self.folder / (key + ".npz")).unlink(missing_ok=True)

    def sample(self, rng, count=4):
        with self.lock:
            values = list(self.entries.values())
            if not values:
                return []
            selected = rng.choice(
                len(values), size=min(count, len(values)), replace=False
            )
            return [values[int(i)] for i in selected]

    def summary(self):
        with self.lock:
            return dict(
                success_routes=len(self.entries),
                successful_rooms=len(
                    {
                        (t["metadata"]["chapter"], t["metadata"]["room"])
                        for t in self.entries.values()
                    }
                ),
                rejected_memory_files=self.rejected,
            )


def state_signature(state):
    """Check observable restore invariants; upstream clones the full level."""
    fields = (
        "chapter",
        "room",
        "x",
        "y",
        "vx",
        "vy",
        "dashes",
        "stamina",
        "player_state",
        "grounded",
        "wall_left",
        "wall_right",
        "holding",
        "keys",
        "jump_held",
        "dash_held",
        "grab_held",
    )
    result = {
        k: round(state[k], 4) if isinstance(state.get(k), float) else state.get(k)
        for k in fields
    }
    result["objects"] = sorted(
        (e.get("name"), round(e["x"], 3), round(e["y"], 3))
        for e in state.get("objects", [])
    )
    return result
