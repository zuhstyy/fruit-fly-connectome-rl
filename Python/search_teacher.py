"""A small automatic teacher that tests inputs in the REAL game.

It discovers examples with bounded two-step lookahead and native savestates.
Search actions are never mixed into PPO rollouts. The fly later learns from
verified room solutions, and evaluation disables this teacher completely.
"""

import numpy as np
import time
from protocol import DURATIONS, action_frames, BridgeError
from navigation import Navigation
from experience import state_signature


def movement_candidates(state, rng):
    choices = []
    # This is a movement vocabulary, not a prerecorded route. Every candidate
    # has to survive actual Celeste physics before the teacher can choose it.
    for horizontal in (0, 2):
        for jump, grab in ((0, 0), (1, 0), (1, 1)):
            choices.append(np.array([horizontal, 1, jump, 0, grab, 0, 4]))
    choices += [
        np.array([1, 0, 0, 0, 1, 0, 3]),
        np.array([1, 2, 0, 0, 1, 0, 3]),
        np.array([1, 1, 0, 0, 0, 1, 2]),
    ]
    if state.get("player_state") in (3, 19):  # Swimming or feather flight.
        for horizontal, vertical in ((1, 0), (1, 2), (0, 0), (2, 0), (0, 2), (2, 2)):
            choices.append(np.array([horizontal, vertical, 0, 0, 0, 0, 3]))
    if state.get("dashes", 0) and not state.get("dash_held"):
        for horizontal, vertical in (
            (2, 1),
            (0, 1),
            (1, 0),
            (2, 0),
            (0, 0),
            (1, 2),
            (2, 2),
            (0, 2),
        ):
            choices.append(np.array([horizontal, vertical, 0, 1, 0, 0, 3]))
    # Long jumps are represented too. Random duration variations avoid fixing
    # the teacher to one timing for every obstacle.
    for action in choices[:6]:
        copy = action.copy()
        copy[6] = int(rng.choice([2, 3, 5, 6]))
        choices.append(copy)
    # Equal scores must not systematically favor the first (left) candidate.
    rng.shuffle(choices)
    return choices


def quality(start, end, navigation, visited):
    if end.get("dead"):
        return -100.0
    if end.get("completed"):
        return 100.0
    if end["room"] != start["room"]:
        return (
            50.0
            if start.get("exit_forward") and start.get("exit_room") == end["room"]
            else -20.0
        )
    end = navigation.annotate(end)
    if not end.get("navigation_valid"):
        return -5.0
    distance = end["navigation_distance"]
    novelty = (int(end["x"] // 8), int(end["y"] // 8), int(end.get("player_state", 0)))
    # A little novelty breaks ties at an obstacle; it is bounded and does not
    # become the agent's reward function or a room-completion claim.
    return -distance / 80 + (0.1 if novelty not in visited else 0)


class SearchTeacher:
    def __init__(self):
        self.trials = self.trial_deaths = self.executed = 0
        self.visited = set()

    def choose(self, client, state, rng, deadline=None, stop=None):
        root = client.capture(2)
        candidates = movement_candidates(state, rng)
        first = []
        navigation = Navigation()
        if state.get("objective", "").startswith("portal:"):
            navigation.destination = state["exit_room"]
            navigation.portal_point = state["objective_x"], state["objective_y"]
        try:
            # Check each first action, keeping only the top three live branches.
            for action in candidates:
                if first and (
                    (deadline is not None and time.monotonic() >= deadline)
                    or (stop is not None and stop.is_set())
                ):
                    break
                self.restore(client, 2, root)
                outcome = client.step(action)
                self.trials += 1
                self.trial_deaths += int(outcome["dead"])
                first.append(
                    (quality(state, outcome, navigation, self.visited), action, outcome)
                )
            first.sort(key=lambda row: row[0], reverse=True)
            best_score, best_action, _ = first[0]
            # Look one action further to discover jumps that first move away
            # from an exit. Native states retain hazards and button history.
            for branch, (score, action, outcome) in enumerate(first[:3]):
                if (deadline is not None and time.monotonic() >= deadline) or (
                    stop is not None and stop.is_set()
                ):
                    break
                if (
                    outcome["dead"]
                    or outcome["completed"]
                    or outcome["room"] != state["room"]
                ):
                    continue
                self.restore(client, 2, root)
                intermediate = client.step(action)
                checkpoint = client.capture(3)
                followups = movement_candidates(intermediate, rng)
                # Limit work per decision; include a neutral release to allow
                # subsequent jumps/dashes to create a fresh button press.
                selected = rng.choice(
                    len(followups), size=min(8, len(followups)), replace=False
                )
                followups = [
                    np.array([1, 1, 0, 0, 0, 0, 0]),
                    *(followups[int(i)] for i in selected),
                ]
                for following in followups:
                    if (deadline is not None and time.monotonic() >= deadline) or (
                        stop is not None and stop.is_set()
                    ):
                        break
                    self.restore(client, 3, checkpoint)
                    end = client.step(following)
                    self.trials += 1
                    self.trial_deaths += int(end["dead"])
                    combined = quality(state, end, navigation, self.visited) - 0.01
                    if combined > best_score:
                        best_score, best_action = combined, action
            return best_action
        finally:
            self.restore(client, 2, root)

    @staticmethod
    def restore(client, slot, expected):
        restored = client.restore(slot)
        if state_signature(restored) != state_signature(expected):
            raise BridgeError("Search restore invariant failed")

    def remember(self, state):
        self.executed += 1
        self.visited.add(
            (
                int(state["x"] // 8),
                int(state["y"] // 8),
                int(state.get("player_state", 0)),
            )
        )
        if len(self.visited) > 4096:
            self.visited.clear()

    def summary(self):
        return dict(
            teacher_trials=self.trials,
            teacher_trial_deaths=self.trial_deaths,
            teacher_actions=self.executed,
        )
