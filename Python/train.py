"""Unattended, synchronous multi-worker PPO with atomic checkpoints."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import signal
import threading
import time
import numpy as np
from protocol import (
    sensors,
    BridgeError,
    ControlPaused,
    ControlStopped,
    action_frames,
    DURATIONS,
)
from learning import Reservoir, Policy, Reward, gae
from plastic import PlasticPolicy, PlasticState, PLASTIC_SCHEMA
from runtime import RunLock, LiveStatus, set_visibility, atomic_json
from workers import ROOT, Worker, validate_game, build, movement_check, repair_original
from navigation import Navigation
from experience import SuccessMemory, state_signature
from curriculum import Curriculum
from search_teacher import SearchTeacher


def emit(message):
    print(message, flush=True)


class Runner:
    """One real game and its private activity, action RNG, and reward history.

    Every Runner shares the same policy weights. Only the main training loop
    changes those weights, after all current rollouts have finished.
    """

    def __init__(
        self,
        worker,
        policy,
        state,
        config,
        chapter_index=0,
        memory=None,
        curriculum=None,
    ):
        self.navigation = Navigation()
        raw_state = state
        state = self.navigation.annotate(state)
        self.worker, self.policy, self.state, self.config = (
            worker,
            policy,
            state,
            config,
        )
        self.chapter_index = self.highest_chapter_index = chapter_index
        self.rng = np.random.default_rng(config["seed"] + worker.index * 1009)
        self.plastic = isinstance(policy, PlasticPolicy)
        self.brain = (
            PlasticState(policy) if self.plastic else Reservoir(seed=config["seed"])
        )
        self.reward = Reward()
        self.reward.reset(state)
        self.current = self.brain.features(sensors(state))
        self.episode_steps = self.total_steps = self.deaths = 0
        self.completed = []
        self.rooms = set()
        self.total_reward = 0.0
        self.measured_frames = 0
        self.actions_seen = self.left_actions = self.jump_actions = self.left_jumps = 0
        self.backtracks = 0
        self.agent_id = worker.index
        self.best_rank = state.get("route_rank", -1)
        self.duration_counts = {str(n): 0 for n in DURATIONS}
        self.last_action = []
        self.last_frames = 0
        self.last_requested = 0
        self.last_response = time.time()
        self.last_live = 0.0
        self.held_frames = {
            k: 0 for k in ("left", "right", "jump", "dash", "grab", "interact")
        }
        self.longest_holds = self.held_frames.copy()
        self.memory, self.curriculum = memory, curriculum
        self.attempt_records = []
        self.attempt_frames = 0
        self.room_clears = self.practice_resets = self.restore_failures = 0
        self.learner_room_clears = self.assisted_room_clears = 0
        self.checkpoints = {}
        self.practice_available = bool(
            config.get("practice", True) and state.get("practice_available", False)
        )
        self.practice_message = "Practice unavailable; ordinary PPO only"
        self.teacher = SearchTeacher()
        self.last_checkpoint_frame = 0
        self.requested_left_actual_right = self.requested_right_actual_left = 0
        self.needs_entry_capture = self.practice_available
        if self.practice_available:
            worker.client.practice_mode(True)
            self.capture_checkpoint(0, raw_state)
            if 0 in self.checkpoints:
                self.restore_checkpoint(0)
            self.needs_entry_capture = not self.checkpoints
        self.publish_feedback()
        self.publish_live("ready")

    def publish_feedback(self, immediate=True):
        self.worker.client.learning_info = dict(
            **(self.policy.diagnostics() if self.plastic else {}),
            exploration=self.temperature(),
            objective=self.state.get("objective", "unknown")[:160],
            navigation_distance=self.state.get("navigation_distance") or 0,
            training_mode="Learner",
            room_clears=self.room_clears,
            practice_resets=self.practice_resets,
            **self.teacher.summary(),
        )
        self.worker.client.set_feedback(
            self.total_reward,
            self.reward.last,
            self.reward.distance_reward,
            self.reward.score,
            self.reward.event,
            immediate=immediate,
        )

    def publish_live(self, phase="running"):
        now = time.monotonic()
        if phase == "running" and now - self.last_live < 0.1:
            return
        self.last_live = now
        if hasattr(self.worker, "report"):
            self.worker.report(phase=phase, **self.snapshot())

    def snapshot(self):
        return dict(
            bridge=(
                self.worker.client.performance()
                if hasattr(self.worker.client, "performance")
                else {}
            ),
            actions=self.actions_seen,
            left_actions=self.left_actions,
            jump_actions=self.jump_actions,
            left_jump_actions=self.left_jumps,
            backtracks=self.backtracks,
            best_route_rank=self.best_rank,
            chapter=self.state["chapter"],
            room=self.state["room"],
            score=self.reward.score,
            last=self.reward.last,
            distance=self.reward.distance_reward,
            event=self.reward.event,
            exit_room=self.state.get("exit_room"),
            exit_distance=self.state.get("exit_distance"),
            deaths=self.deaths,
            action=self.last_action,
            requested_frames=self.last_requested,
            action_frames=self.last_frames,
            duration_counts=dict(self.duration_counts),
            longest_holds=dict(self.longest_holds),
            last_response=self.last_response,
            room_clears=self.room_clears,
            practice_resets=self.practice_resets,
            learner_room_clears=self.learner_room_clears,
            assisted_room_clears=self.assisted_room_clears,
            practice_available=self.practice_available,
            practice_message=self.practice_message,
            restore_failures=self.restore_failures,
            objective=self.state.get("objective", "unknown"),
            navigation_distance=self.state.get("navigation_distance"),
            player_state=self.state.get("player_state", -1),
            actual_vx=self.state.get("vx", 0),
            actual_vy=self.state.get("vy", 0),
            requested_left_actual_right=self.requested_left_actual_right,
            requested_right_actual_left=self.requested_right_actual_left,
            **self.teacher.summary(),
        )

    def capture_checkpoint(self, slot, raw_state):
        if not self.practice_available or not raw_state.get("ready", True):
            return
        try:
            captured = self.worker.client.capture(slot)
            if state_signature(captured) != state_signature(raw_state):
                raise BridgeError("Capture changed observable game state")
            self.checkpoints[slot] = dict(
                state=captured,
                records=list(self.attempt_records),
                frames=self.attempt_frames,
            )
            self.last_checkpoint_frame = self.attempt_frames
            self.practice_message = (
                "Room entry saved" if slot == 0 else "New safe frontier saved"
            )
        except (ControlPaused, ControlStopped):
            raise
        except BridgeError as error:
            self.practice_message = str(error)

    def restore_checkpoint(self, slot):
        checkpoint = self.checkpoints.get(slot)
        if checkpoint is None:
            return False
        try:
            restored = self.worker.client.restore(slot)
            if state_signature(restored) != state_signature(checkpoint["state"]):
                raise BridgeError("Restore mismatch: practice disabled for this worker")
        except (ControlPaused, ControlStopped):
            raise
        except BridgeError as error:
            self.restore_failures += 1
            self.practice_message = str(error)
            self.practice_available = False
            self.checkpoints.clear()
            self.worker.client.practice_mode(False)
            return False
        self.practice_resets += 1
        self.attempt_records = list(checkpoint["records"])
        self.attempt_frames = checkpoint["frames"]
        self.state = self.navigation.annotate(restored)
        self.brain.reset()
        # Reconstruct checkpoint memory with CURRENT learned weights, including
        # all preceding observations. A stale saved activity is never reused.
        for record in self.attempt_records:
            if record["reset"]:
                self.brain.reset()
            self.brain.features(record["observation"])
        self.current = self.brain.features(sensors(self.state))
        self.reward.reset(self.state)
        self.held_frames = {key: 0 for key in self.held_frames}
        self.practice_message = (
            "Practicing saved frontier" if slot else "Practicing room entry"
        )
        self.last_response = time.time()
        return True

    def finish_attempt(self, success, old, new):
        if self.curriculum:
            self.curriculum.record(
                old["chapter"], old["room"], success, self.attempt_frames
            )
        if success:
            self.room_clears += 1
            if any(record.get("teacher") for record in self.attempt_records):
                self.assisted_room_clears += 1
            else:
                self.learner_room_clears += 1
            if self.memory:
                self.memory.add(
                    self.attempt_records,
                    old["chapter"],
                    old["room"],
                    new["room"],
                    completed=new["completed"],
                    worker=self.worker.index + 1,
                )

    def reset_room(self, state, fresh_attempt=True):
        self.navigation.reset()
        self.state = self.navigation.annotate(state)
        self.brain.reset()
        self.reward.reset(self.state, fresh_attempt=fresh_attempt)
        self.attempt_records = []
        self.attempt_frames = 0
        self.checkpoints.clear()
        self.needs_entry_capture = self.practice_available
        self.current = self.brain.features(sensors(self.state))

    def evaluate_room(self, budget=900):
        """A fixed current policy gets one room attempt, without frontier resets.

        Entry/return restores are scaffolding around the attempt. This is a
        room test, never described as a complete playthrough from the title.
        """
        if (
            not self.practice_available
            or 0 not in self.checkpoints
            or not self.state.get("ready", True)
        ):
            return None
        raw = self.worker.client.capture(7)
        original_state = self.state
        brain = PlasticState(self.policy)
        observed = self.worker.client.restore(0)
        if state_signature(observed) != state_signature(self.checkpoints[0]["state"]):
            self.worker.client.restore(7)
            self.restore_failures += 1
            return None
        navigation = Navigation()
        chapter, room = observed["chapter"], observed["room"]
        frames, success = 0, False
        try:
            for _ in range(1200):
                if frames >= budget or self.worker.stop.is_set():
                    break
                observed = navigation.annotate(observed)
                features = brain.features(sensors(observed))
                logits, offset, action = features @ self.policy.actor, 0, []
                from protocol import HEADS

                for choices in HEADS:
                    action.append(int(np.argmax(logits[offset : offset + choices])))
                    offset += choices
                new = self.worker.client.step(action)
                frames += new.get("action_frames", action_frames(action))
                if new["completed"] or (
                    new["room"] != room
                    and observed.get("exit_room") == new["room"]
                    and observed.get("exit_forward")
                ):
                    success = True
                    break
                if new["dead"] or new["room"] != room:
                    break
                observed = new
        finally:
            returned = self.worker.client.restore(7)
            if state_signature(returned) != state_signature(raw):
                raise BridgeError("Evaluation return-state mismatch")
            self.worker.client.clear(7)
            self.state = original_state
            self.navigation.reset()
        if self.curriculum:
            self.curriculum.record(chapter, room, success, frames, "evaluation")
        return dict(
            chapter=chapter,
            room=room,
            clear=success,
            frames=frames,
            entry_signature=state_signature(self.checkpoints[0]["state"]),
            weights_updated=False,
            frontier_resets=0,
            start="saved_room_entry",
            full_main_story_clear=False,
            saved_unix=time.time(),
        )

    def temperature(self):
        # More exploration when stalled. Practice restores are separate controls.
        return min(
            2.5,
            1.0
            + max(
                0.0, self.reward.steps_without_progress / self.config["stuck_steps"] - 1
            )
            * 0.5,
        )

    def teach(self, count=4):
        """Discover real solutions between PPO rollouts; never forge policy data."""
        if not self.practice_available or not self.state.get("ready", True):
            return
        self.publish_live("searching")
        deadline = time.monotonic() + self.config.get("teacher_seconds", 3.0)
        for _ in range(count):
            if self.worker.stop.is_set() or time.monotonic() >= deadline:
                break
            old = self.state
            self.worker.client.learning_info.update(
                training_mode="Search teacher", **self.teacher.summary()
            )
            self.worker.client.set_feedback(
                self.total_reward,
                self.reward.last,
                self.reward.distance_reward,
                self.reward.score,
                "Automatic teacher: testing real movement branches",
            )
            try:
                action = self.teacher.choose(
                    self.worker.client,
                    old,
                    self.rng,
                    deadline=deadline,
                    stop=self.worker.stop,
                )
            except (ControlPaused, ControlStopped):
                raise
            except BridgeError as error:
                self.practice_message = str(error)
                self.restore_failures += 1
                self.practice_available = False
                self.worker.client.practice_mode(False)
                self.reset_room(
                    self.worker.client.start(
                        self.config["chapters"][self.chapter_index], self.worker.stop
                    )
                )
                break
            new = self.worker.client.step(action)
            elapsed = new.get("action_frames", action_frames(action))
            self.teacher.remember(new)
            self.attempt_records.append(
                dict(
                    observation=sensors(old),
                    action=action.copy(),
                    reset=self.brain.last_reset,
                    frames=elapsed,
                    teacher=True,
                )
            )
            self.bound_attempt()
            self.attempt_frames += elapsed
            new = self.navigation.annotate(new)
            self.total_reward += self.reward.step(old, new, elapsed)
            self.last_action, self.last_requested, self.last_frames = (
                action.tolist(),
                action_frames(action),
                elapsed,
            )
            self.last_response = time.time()
            self.state = new
            changed = new["room"] != old["room"]
            forward = (
                changed
                and old.get("exit_forward")
                and old.get("exit_room") == new["room"]
            )
            if changed or new["completed"] or new["dead"]:
                self.finish_attempt(bool(forward or new["completed"]), old, new)
                if new["completed"]:
                    self.completed.append(self.config["chapters"][self.chapter_index])
                    self.chapter_index = (self.chapter_index + 1) % len(
                        self.config["chapters"]
                    )
                    self.highest_chapter_index = max(
                        self.highest_chapter_index, self.chapter_index
                    )
                    self.reset_room(
                        self.worker.client.start(
                            self.config["chapters"][self.chapter_index],
                            self.worker.stop,
                        )
                    )
                elif new["dead"]:
                    self.deaths += 1
                    if not self.restore_checkpoint(1 if 1 in self.checkpoints else 0):
                        self.reset_room(self.worker.client.retry())
                else:
                    self.reset_room(new, fresh_attempt=False)
                self.episode_steps = 0
                break
            self.current = self.brain.features(sensors(new))
            self.publish_feedback(immediate=False)
            self.publish_live("searching")
        self.publish_feedback()
        if self.practice_available:
            self.worker.client.clear(2)
            self.worker.client.clear(3)

    def bound_attempt(self):
        if len(self.attempt_records) > 2048:
            # Missing context cannot be advertised as a full successful route.
            self.attempt_records = self.attempt_records[-2048:]
            self.attempt_records[0] = dict(self.attempt_records[0], reset=False)

    def resume_after_pause(self):
        self.publish_live("paused")
        self.worker.client.wait_until_on(self.worker.stop)
        self.state = self.worker.client.start(
            self.config["chapters"][self.chapter_index], self.worker.stop
        )
        self.last_response = time.time()
        self.held_frames = {key: 0 for key in self.held_frames}
        self.brain.reset()
        self.reward.reset(self.state)
        self.episode_steps = 0
        self.reset_room(self.state)
        self.publish_feedback()

    def recover_worker(self):
        """A new process must opt into practice and validate its own saves."""
        state = self.worker.recover(self.config["chapters"][self.chapter_index])
        self.practice_available = bool(
            self.config.get("practice", True) and state.get("practice_available", False)
        )
        if self.practice_available:
            self.worker.client.practice_mode(True)
        self.reset_room(state)
        self.episode_steps = 0
        self.last_response = time.time()
        self.held_frames = {key: 0 for key in self.held_frames}
        if self.practice_available:
            self.capture_checkpoint(0, state)
            if 0 in self.checkpoints:
                self.restore_checkpoint(0)
            self.needs_entry_capture = not self.checkpoints
        self.publish_feedback()

    def roll(self, size):
        """Collect decisions, actual game outcomes, and information for PPO."""
        xs, acts, lps, vals, rewards, dones = [], [], [], [], [], []
        observations, before, resets, temperatures, durations = [], [], [], [], []
        if self.plastic:
            self.current = self.brain.refresh()
        for _ in range(size):
            if self.worker.stop.is_set():
                break
            if self.needs_entry_capture:
                try:
                    self.capture_checkpoint(0, self.state)
                except ControlPaused:
                    if dones:
                        dones[-1] = True
                    self.resume_after_pause()
                    continue
                except ControlStopped:
                    self.worker.stop.set()
                    break
                self.needs_entry_capture = 0 not in self.checkpoints
            f = self.current
            temperature = self.temperature() if self.plastic else 1.0
            action, logp, value = (
                self.policy.act(f, self.rng, temperature)
                if self.plastic
                else self.policy.act(f, self.rng)
            )
            requested = action_frames(action) if self.plastic else self.config["frames"]
            old_state = self.state
            try:
                new = self.worker.client.step(
                    action if self.plastic else action[:6], requested
                )
            except ControlPaused:
                # Manual gameplay is never attributed to a sampled policy action.
                if dones:
                    dones[-1] = True
                self.resume_after_pause()
                continue
            except ControlStopped:
                self.worker.stop.set()
                break
            except (BridgeError, OSError, ValueError) as e:
                if "Emergency stop" in str(e):
                    self.worker.stop.set()
                    raise
                emit(f"Worker {self.worker.index+1}: {e}")
                self.publish_live("recovering")
                # Truncate the previous fragment at a real interruption, not across a reset.
                if dones:
                    dones[-1] = True
                self.recover_worker()
                continue
            raw_new = new
            new = self.navigation.annotate(new)
            self.measured_frames += max(
                0, new.get("frame", 0) - self.state.get("frame", 0)
            )
            self.last_response = time.time()
            elapsed = int(new.get("action_frames", requested))
            if not 0 <= elapsed <= requested:
                raise BridgeError("Invalid elapsed action time from game")
            durations.append(elapsed)
            self.last_frames, self.last_requested, self.last_action = (
                elapsed,
                requested,
                action.tolist(),
            )
            self.duration_counts[str(requested)] = (
                self.duration_counts.get(str(requested), 0) + 1
            )
            held = dict(
                left=action[0] == 0,
                right=action[0] == 2,
                jump=bool(action[2]),
                dash=bool(action[3]),
                grab=bool(action[4]),
                interact=bool(action[5]),
            )
            for key, on in held.items():
                self.held_frames[key] = self.held_frames[key] + elapsed if on else 0
                self.longest_holds[key] = max(
                    self.longest_holds[key], self.held_frames[key]
                )
            self.episode_steps += 1
            self.total_steps += 1
            self.rooms.add((new["chapter"], new["room"]))
            self.actions_seen += 1
            self.left_actions += int(action[0] == 0)
            self.jump_actions += int(action[2] == 1)
            self.left_jumps += int(action[0] == 0 and action[2] == 1)
            self.requested_left_actual_right += int(action[0] == 0 and new["vx"] > 1)
            self.requested_right_actual_left += int(action[0] == 2 and new["vx"] < -1)
            self.best_rank = max(self.best_rank, new.get("route_rank", -1))
            if new["room"] != self.state["room"] and new.get(
                "route_rank", -1
            ) < self.state.get("route_rank", -1):
                self.backtracks += 1
            if self.total_steps % 300 == 0:
                self.worker.restarts = 0
            r = self.reward.step(self.state, new, elapsed)
            terminal = bool(new["dead"] or new["completed"])
            self.attempt_frames += elapsed
            self.attempt_records.append(
                dict(
                    observation=sensors(self.state),
                    action=action.copy(),
                    reset=(
                        bool(self.brain.last_reset)
                        if self.plastic
                        else not self.attempt_records
                    ),
                    frames=elapsed,
                )
            )
            self.bound_attempt()
            if self.plastic:
                observations.append(self.brain.last_observation.copy())
                before.append(self.brain.last_before.copy())
                resets.append(self.brain.last_reset)
                temperatures.append(temperature)
            if self.reward.steps_without_progress >= self.config["stuck_steps"]:
                self.reward.event += " | exploring; no forced death"
            xs.append(f.copy())
            acts.append(action)
            lps.append(logp)
            vals.append(value)
            rewards.append(r)
            dones.append(terminal)
            self.total_reward += r
            self.state = new
            self.publish_feedback(immediate=terminal)
            self.publish_live()
            changed = new["room"] != old_state["room"]
            forward = (
                changed
                and old_state.get("exit_forward")
                and old_state.get("exit_room") == new["room"]
            )
            if terminal or changed:
                self.finish_attempt(bool(new["completed"] or forward), old_state, new)
            if terminal:
                self.held_frames = {key: 0 for key in self.held_frames}
                try:
                    if new["completed"]:
                        self.completed.append(
                            self.config["chapters"][self.chapter_index]
                        )
                        emit(
                            f'Worker {self.worker.index+1}: GAME-REPORTED chapter {new["chapter"]} completion'
                        )
                        self.chapter_index = (self.chapter_index + 1) % len(
                            self.config["chapters"]
                        )
                        self.highest_chapter_index = max(
                            self.highest_chapter_index, self.chapter_index
                        )
                        self.state = self.worker.client.start(
                            self.config["chapters"][self.chapter_index],
                            self.worker.stop,
                        )
                    else:
                        self.deaths += 1
                        # Sometimes rehearse the whole room rather than only
                        # the frontier, so the early movement keeps improving.
                        slot = (
                            1
                            if 1 in self.checkpoints and self.rng.random() < 0.8
                            else 0
                        )
                        if self.practice_available and self.restore_checkpoint(slot):
                            self.episode_steps = 0
                            self.publish_feedback()
                            continue
                        self.state = self.worker.client.retry()
                except ControlPaused:
                    self.resume_after_pause()
                    continue
                except ControlStopped:
                    self.worker.stop.set()
                    break
                self.reset_room(self.state)
                self.episode_steps = 0
                self.publish_feedback()
                continue
            if changed:
                if forward and self.worker.index % 4 == 3 and self.rng.random() < 0.5:
                    if self.restore_checkpoint(0):
                        dones[-1] = True
                        self.episode_steps = 0
                        continue
                # Explicit room-level episode boundary; a transition bonus
                # credits the input that actually crossed the intended exit.
                dones[-1] = True
                self.reset_room(raw_new, fresh_attempt=False)
                self.held_frames = {key: 0 for key in self.held_frames}
                self.episode_steps = 0
                continue
            self.current = self.brain.features(sensors(self.state))
            checkpoint_distance = (
                self.checkpoints.get(1, self.checkpoints.get(0, {}))
                .get("state", {})
                .get("navigation_distance")
            )
            if checkpoint_distance is None and self.checkpoints:
                checkpoint_distance = self.navigation.annotate(
                    self.checkpoints.get(1, self.checkpoints.get(0))["state"]
                ).get("navigation_distance")
            if (
                self.practice_available
                and new.get("grounded")
                and new.get("navigation_valid")
                and checkpoint_distance is not None
                and new["navigation_distance"] < checkpoint_distance - 24
                and self.attempt_frames - self.last_checkpoint_frame >= 120
            ):
                self.capture_checkpoint(1, raw_new)
            stalled = self.reward.steps_without_progress >= self.config["stuck_steps"]
            timed_out = self.episode_steps >= self.config["episode_steps"]
            if stalled or timed_out:
                attempt_recorded = False
                slot = 1 if 1 in self.checkpoints and self.rng.random() < 0.75 else 0
                if self.practice_available and slot in self.checkpoints:
                    # Time-limit truncation bootstraps at the state before the
                    # artificial reset. Future rewards never cross the restore.
                    tail_value = float(self.current @ self.policy.value)
                    self.navigation.note_failure(new)
                    self.finish_attempt(False, old_state, new)
                    attempt_recorded = True
                    if self.restore_checkpoint(slot):
                        rewards[-1] += 0.99 ** (elapsed / 4.0) * tail_value
                        dones[-1] = True
                        self.episode_steps = 0
                        self.publish_feedback()
                        continue
                if timed_out:
                    if not attempt_recorded:
                        self.finish_attempt(False, old_state, new)
                    rewards[-1] += 0.99 ** (elapsed / 4.0) * float(
                        self.current @ self.policy.value
                    )
                    dones[-1] = True
                    if self.curriculum:
                        self.chapter_index = self.curriculum.choose(
                            self.worker.index,
                            self.config["chapters"],
                            self.highest_chapter_index,
                            self.rng,
                        )
                    self.reset_room(
                        self.worker.client.start(
                            self.config["chapters"][self.chapter_index],
                            self.worker.stop,
                        )
                    )
                    self.practice_resets += 1
                    self.episode_steps = 0
        if not xs:
            return None
        bootstrap = 0.0 if dones[-1] else float(self.current @ self.policy.value)
        adv, ret = gae(rewards, vals, dones, bootstrap, durations=durations)
        if self.plastic:
            return dict(
                observations=np.asarray(observations),
                before=np.asarray(before),
                resets=np.asarray(resets, dtype=bool),
                temperatures=np.asarray(temperatures),
                actions=np.asarray(acts),
                old_logp=np.asarray(lps),
                advantages=adv,
                returns=ret,
                durations=np.asarray(durations),
            )
        return np.asarray(xs), np.asarray(acts), np.asarray(lps), adv, ret


def validate_config(c):
    if type(c.get("practice", True)) is not bool:
        raise ValueError("Invalid practice toggle")
    if type(c.get("fast_simulation", True)) is not bool:
        raise ValueError("Invalid fast simulation toggle")
    if c.get("evaluation_action", "sample") not in {"sample", "greedy"}:
        raise ValueError("Invalid evaluation_action")
    for k, lo, hi in [
        ("workers", 1, 64),
        ("ticks", 1, 64),
        ("frames", 1, 12),
        ("rollout", 32, 4096),
        ("episode_steps", 100, 20000),
        ("stuck_steps", 30, 5000),
    ]:
        if type(c[k]) is not int or not lo <= c[k] <= hi:
            raise ValueError(f"Invalid {k}")
    for key, default, lo, hi in [
        ("startup_parallelism", 2, 1, 4),
        ("batch_decisions", 2048, 128, 16384),
        ("evaluation_interval", 25, 0, 10000),
        ("evaluation_frames", 900, 60, 36000),
        ("imitation_decisions", 128, 0, 1024),
        ("teacher_interval", 2, 0, 10000),
        ("teacher_decisions", 4, 1, 32),
        ("sequences_per_batch", 8, 1, 64),
    ]:
        value = c.get(key, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("Invalid " + key)
    if c.get("backend", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Invalid learner backend")
    seconds = c.get("teacher_seconds", 3.0)
    if (
        not isinstance(seconds, (float, int))
        or not math.isfinite(seconds)
        or not 0.1 <= seconds <= 60
    ):
        raise ValueError("Invalid teacher time budget")
    if not math.isfinite(c["hours"]) or not 0 < c["hours"] <= 720:
        raise ValueError("Hours must be in (0,720]")
    if c["chapters"] != list(range(8)):
        raise ValueError(
            "This release supports the main-story A-side campaign, chapter IDs 0 through 7."
        )


def prepare_checkpoint(policy, checkpoint, log=print):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return {}
    with np.load(checkpoint, allow_pickle=False) as old:
        legacy = "plastic_schema" not in old
        migrating = not legacy and str(old["plastic_schema"]) != PLASTIC_SCHEMA
    if legacy:
        backup = (
            checkpoint.parent
            / "legacy-models"
            / ("latest-before-3.0-" + str(time.time_ns()) + ".npz")
        )
        backup.parent.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, backup)
        log("Old fixed-network checkpoint backed up to " + str(backup))
        log("Starting a fresh trainable fly network; previous models are preserved.")
        return {}
    if migrating:
        backup = (
            checkpoint.parent
            / "legacy-models"
            / ("latest-before-4.0-" + str(time.time_ns()) + ".npz")
        )
        backup.parent.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, backup)
        log(
            "Saved an exact backup before extending game observations. Existing actor, value, fly weights and optimizer state are retained."
        )
    return policy.load(checkpoint, policy.graph_hash)


def run(config):
    with RunLock(ROOT):
        _run(config)


def _run(config):
    validate_config(config)
    emit(
        "Version 4.1: fast fixed-physics workers, optional CUDA gradients, bounded search and trainable MCNS edges."
    )
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    end_time = float("inf")

    def watch_stop():
        while not stop.wait(0.25):
            if (ROOT / "STOP").exists() or time.monotonic() >= end_time:
                stop.set()

    threading.Thread(target=watch_stop, daemon=True).start()
    workers, runners = [], []
    run_id = str(time.time_ns())
    saved = ROOT / "training"
    saved.mkdir(exist_ok=True)
    live = LiveStatus(ROOT, config["workers"], run_id)
    set_visibility(ROOT, not config.get("background", True))
    live.start()
    final_phase, final_message = (
        "stopped",
        "Training stopped. Last completed update remains saved.",
    )
    try:
        policy = PlasticPolicy(seed=config["seed"])
        steps, iteration, resume_index = 0, 0, 0
        checkpoint = saved / "latest.npz"
        previous = prepare_checkpoint(policy, checkpoint, emit)
        if previous:
            if previous.get("chapters") != config["chapters"]:
                raise ValueError("Checkpoint campaign mismatch")
            steps = previous.get("steps", 0)
            iteration = previous.get("iteration", 0)
            resume_index = previous.get("furthest_chapter_index", 0)
            emit(
                f'Resumed {steps:,} decisions and existing fly weights; starting chapter {config["chapters"][resume_index]}.'
            )
        from accelerator import choose_backend

        backend = choose_backend(
            policy,
            config.get("backend", "auto"),
            count=min(
                config.get("sequences_per_batch", 8),
                max(1, config["workers"] * config["rollout"] // 32),
            ),
            log=emit,
        )
        atomic_json(saved / "accelerator.json", backend)
        live.update(backend=backend)
        live.update(iteration=iteration, decisions=steps, **policy.diagnostics())
        memory = SuccessMemory(saved / "success-memory", policy.graph_hash)
        curriculum = Curriculum(saved)
        game = validate_game(config["game"])
        repair_original(game, emit)
        mod = build(game, emit, config.get("rebuild", False))
        emit(
            f'Requested {config["workers"]} REAL Celeste processes, each with its own save, state, memory and action RNG.'
        )
        for i in range(config["workers"]):
            w = Worker(i, game, mod, stop, emit, config["ticks"])
            w.telemetry = live
            w.background = config.get("background", True)
            w.fast_simulation = config.get("fast_simulation", True)
            workers.append(w)

        def start_one(w):
            emit(f'Starting real game {w.index+1}/{config["workers"]}')
            c = w.launch()
            while True:
                try:
                    state = movement_check(c, config["chapters"][resume_index], stop)
                    break
                except ControlPaused:
                    c.wait_until_on(stop)
            runner = Runner(w, policy, state, config, resume_index, memory, curriculum)
            emit(
                f'Worker {w.index+1}: PID {w.launch_info.get("bridge_pid")}, isolated save and movement checks passed.'
            )
            return runner

        from scheduling import launch_workers, rollout_size

        runners, capacity_message = launch_workers(
            workers, start_one, stop, config.get("startup_parallelism", 2)
        )
        if not runners:
            if stop.is_set():
                return
            raise RuntimeError(
                capacity_message
                or "No game passed startup; open MONITOR.cmd or training/run.log."
            )
        if len(runners) < len(workers):
            capacity_message = (
                capacity_message
                or "Some workers failed startup. Open the monitor for individual errors."
            )
            emit(
                f"ONLY {len(runners)}/{len(workers)} real games ready. {capacity_message}"
            )
        else:
            emit(f"All {len(runners)}/{len(workers)} real games passed startup.")
        live.update(
            phase="collecting",
            message=capacity_message or "All requested games passed startup",
            capacity_message=capacity_message,
        )
        start_time = time.monotonic()
        end_time = start_time + config["hours"] * 3600
        initial_steps = steps
        controlled_frames = previous.get("controlled_frames", 0)
        initial_frames = controlled_frames
        all_completed = set(previous.get("completed_chapters", []))
        furthest = resume_index
        with (
            ThreadPoolExecutor(max_workers=len(runners)) as pool,
            (saved / "metrics.csv").open("a", newline="", encoding="utf-8") as f,
        ):
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    [
                        "iteration",
                        "decisions",
                        "decisions_per_second",
                        "rooms_this_run",
                        "deaths_this_run",
                        "furthest_chapter",
                        "loss",
                        "kl",
                        "entropy",
                    ]
                )
            while runners and not stop.is_set() and time.monotonic() < end_time:
                cycle_started = time.monotonic()
                search_seconds = evaluation_seconds = 0.0
                live.update(phase="collecting")
                size = rollout_size(config, len(runners))
                futures = [(r, pool.submit(r.roll, size)) for r in runners]
                batches = []
                healthy = []
                for runner, future in futures:
                    try:
                        batch = future.result()
                        if batch is not None:
                            batches.append(batch)
                        healthy.append(runner)
                    except ControlPaused:
                        runner.resume_after_pause()
                        healthy.append(runner)
                    except (ControlStopped, InterruptedError):
                        stop.set()
                    except Exception as e:
                        emit(
                            f"Worker {runner.worker.index+1} removed after failure: {e}"
                        )
                        runner.worker.shutdown()
                        runner.worker.report(phase="failed", event=str(e))
                        capacity_message = f"{len(healthy)}+ workers remain; a worker failed. Inspect the monitor."
                        live.update(capacity_message=capacity_message)
                runners = healthy
                if len(runners) < len(workers):
                    live.update(
                        capacity_message=f"{len(runners)}/{len(workers)} workers ready. "
                        + capacity_message
                    )
                if not batches:
                    break
                collection_seconds = time.monotonic() - cycle_started
                teacher_interval = config.get("teacher_interval", 2)
                eligible = [
                    r
                    for r in runners
                    if r.practice_available and r.state.get("ready", True)
                ]
                if teacher_interval and eligible and iteration % teacher_interval == 0:
                    search_started = time.monotonic()
                    teacher_runner = eligible[
                        (iteration // teacher_interval) % len(eligible)
                    ]
                    live.update(
                        phase="searching",
                        message="Automatic movement search; separate from learner decisions",
                    )
                    try:
                        teacher_runner.teach(config.get("teacher_decisions", 4))
                    except ControlPaused:
                        teacher_runner.resume_after_pause()
                    except ControlStopped:
                        stop.set()
                    except (BridgeError, OSError, ValueError) as error:
                        emit("Search worker interrupted: " + str(error))
                        try:
                            teacher_runner.recover_worker()
                        except ControlPaused:
                            teacher_runner.resume_after_pause()
                        except ControlStopped:
                            stop.set()
                        except Exception as recovery_error:
                            teacher_runner.worker.shutdown()
                            teacher_runner.worker.report(
                                phase="failed", event=str(recovery_error)
                            )
                            runners.remove(teacher_runner)
                            capacity_message = f"{len(runners)}/{len(workers)} ready. Search worker recovery failed: {recovery_error}"
                            live.update(capacity_message=capacity_message)
                    search_seconds = time.monotonic() - search_started
                live.update(phase="learning")
                for runner in runners:
                    runner.publish_live("learning")
                update_started = time.monotonic()
                metrics = policy.update_rollouts(
                    batches, sequences_per_batch=config.get("sequences_per_batch", 8)
                )
                imitation = policy.imitate(
                    memory.sample(policy.rng),
                    decisions=config.get("imitation_decisions", 128),
                )
                update_seconds = time.monotonic() - update_started
                steps += sum(len(b["actions"]) for b in batches)
                controlled_frames += sum(
                    int(np.sum(b.get("durations", 0))) for b in batches
                )
                iteration += 1
                interval = config.get("evaluation_interval", 25)
                if interval and iteration % interval == 0:
                    candidate = next(
                        (
                            r
                            for r in runners
                            if r.practice_available and 0 in r.checkpoints
                        ),
                        None,
                    )
                    if candidate:
                        evaluation_started = time.monotonic()
                        live.update(
                            phase="evaluating",
                            message="Frozen-policy room test; no frontier restores within attempt",
                        )
                        candidate.publish_live("evaluating")
                        try:
                            evaluation = candidate.evaluate_room(
                                config.get("evaluation_frames", 900)
                            )
                            if evaluation:
                                with (saved / "room-evaluations.jsonl").open(
                                    "a", encoding="utf-8"
                                ) as report:
                                    report.write(
                                        json.dumps(
                                            dict(iteration=iteration, **evaluation)
                                        )
                                        + "\n"
                                    )
                                emit(
                                    f'Room evaluation: {evaluation["chapter"]}:{evaluation["room"]} clear={evaluation["clear"]}'
                                )
                        except ControlPaused:
                            candidate.resume_after_pause()
                        except ControlStopped:
                            stop.set()
                        except (BridgeError, OSError) as error:
                            emit("Room evaluation interrupted: " + str(error))
                            try:
                                candidate.recover_worker()
                            except ControlPaused:
                                candidate.resume_after_pause()
                            except ControlStopped:
                                stop.set()
                            except Exception as recovery_error:
                                candidate.worker.shutdown()
                                candidate.worker.report(
                                    phase="failed", event=str(recovery_error)
                                )
                                runners.remove(candidate)
                                capacity_message = f"{len(runners)}/{len(workers)} ready. Evaluation worker recovery failed: {recovery_error}"
                                live.update(capacity_message=capacity_message)
                        evaluation_seconds = time.monotonic() - evaluation_started
                curriculum.save()
                for runner in runners:
                    all_completed.update(runner.completed)
                furthest = max([furthest] + [r.highest_chapter_index for r in runners])
                metadata = dict(
                    run_id=run_id,
                    initial_completed_chapters=previous.get("completed_chapters", []),
                    steps=steps,
                    controlled_frames=controlled_frames,
                    iteration=iteration,
                    chapters=config["chapters"],
                    furthest_chapter_index=furthest,
                    completed_chapters=sorted(all_completed),
                    graph_sha256=policy.graph_hash,
                    reward_revision=Reward.REVISION,
                    **memory.summary(),
                    imitation_decisions_total=previous.get(
                        "imitation_decisions_total", 0
                    )
                    + imitation["imitation_decisions"],
                    plastic_schema=PLASTIC_SCHEMA,
                    saved_unix=time.time(),
                )
                previous["imitation_decisions_total"] = metadata[
                    "imitation_decisions_total"
                ]
                saving_started = time.monotonic()
                policy.save(checkpoint, policy.graph_hash, metadata)
                if iteration % 25 == 0:
                    policy.save(
                        saved / f"checkpoint-{iteration:06d}.npz",
                        policy.graph_hash,
                        metadata,
                    )

                # Feedback acknowledgements used to serialize one game tick per worker.
                def feedback(r):
                    try:
                        r.publish_feedback()
                    except (BridgeError, OSError, ValueError) as e:
                        r.worker.report(event="Feedback unavailable: " + str(e))

                list(pool.map(feedback, runners))
                saving_seconds = time.monotonic() - saving_started
                phases = dict(
                    collection=collection_seconds,
                    search=search_seconds,
                    learning=update_seconds,
                    evaluation=evaluation_seconds,
                    save_feedback=saving_seconds,
                )
                from performance import throughput_summary

                performance = throughput_summary(
                    controlled_frames - initial_frames,
                    time.monotonic() - start_time,
                    phases,
                )
                rate = (steps - initial_steps) / max(
                    0.001, time.monotonic() - start_time
                )
                rooms = len(set().union(*(r.rooms for r in runners))) if runners else 0
                deaths = sum(r.deaths for r in runners)
                writer.writerow(
                    [
                        iteration,
                        steps,
                        rate,
                        rooms,
                        deaths,
                        config["chapters"][furthest],
                        *metrics.values(),
                    ]
                )
                f.flush()
                worker_rewards = [
                    dict(worker=r.worker.index + 1, **r.snapshot()) for r in runners
                ]
                status = dict(
                    **metadata,
                    requested_workers=config["workers"],
                    ready_workers=len(runners),
                    live_game_processes=live.snapshot()["live_game_processes"],
                    capacity_message=capacity_message,
                    **policy.diagnostics(),
                    update_seconds=update_seconds,
                    backend=backend,
                    performance=performance,
                    worker_rewards=worker_rewards,
                    decisions_per_second=rate,
                    simulation_frames_per_second=sum(r.measured_frames for r in runners)
                    / max(0.001, time.monotonic() - start_time),
                    rooms_this_run=rooms,
                    deaths_this_run=deaths,
                    **metrics,
                    **imitation,
                    curriculum=curriculum.summary(),
                    practice_resets=sum(r.practice_resets for r in runners),
                )
                atomic_json(saved / "status.json", status)
                atomic_json(saved / "performance.json", performance)
                with (saved / "progress.jsonl").open("a", encoding="utf-8") as history:
                    history.write(json.dumps(status) + "\n")
                live.update(
                    iteration=iteration,
                    decisions=steps,
                    decisions_per_second=rate,
                    update_seconds=update_seconds,
                    performance=performance,
                    completed_chapters=sorted(all_completed),
                    **policy.diagnostics(),
                    **memory.summary(),
                    curriculum=curriculum.summary(),
                    imitation_decisions=imitation["imitation_decisions"],
                )
                emit(
                    f"Update {iteration} | {steps:,} decisions | {rate:.1f}/sec | {len(runners)}/{len(workers)} workers ready | {rooms} rooms | {deaths} deaths | {memory.summary()['success_routes']} success routes | {imitation['imitation_decisions']} imitation decisions | saved"
                )
                emit(
                    f"Speed: {performance['frames_per_second']:.0f} controlled frames/sec | {performance['game_speed']:.1f}x aggregate game time | bottleneck: {performance['largest_phase']} | collect {collection_seconds:.2f}s, search {search_seconds:.2f}s, learn {update_seconds:.2f}s"
                )
        emit(final_message)
    except BaseException as e:
        if not isinstance(e, (ControlStopped, InterruptedError, KeyboardInterrupt)):
            final_phase, final_message = "error", str(e)
        raise
    finally:
        stop.set()
        live.update(phase="stopping")
        for w in workers:
            try:
                w.shutdown()
            except Exception as e:
                emit(f"Worker {w.index+1} shutdown issue: {e}")
        live.close(final_phase, final_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text()))
    except InterruptedError:
        emit(
            "Stopped. The last completed policy update is saved, if training had begun."
        )
    except ControlStopped:
        emit("Stopped from Celeste. Manual control released; closing owned workers.")
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
