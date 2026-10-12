"""Meaningful practice/learning checks against a small synthetic game.

These tests cannot establish Celeste room clears or full-game completion.
"""

from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from experience import SuccessMemory, state_signature
from curriculum import Curriculum
from navigation import Navigation, distance_field
from learning import Reward
from plastic import PlasticPolicy, PREVIOUS_PLASTIC_SCHEMA
from protocol import sensors, HEADS
from search_teacher import SearchTeacher
from train import Runner
from protocol import ControlPaused, BridgeError
from demonstrations import validate_recording
from test_learning import state
from test_workers import FakeWorker, FakeGame, config


def room_state(**changes):
    result = state(
        x=12.0,
        y=12.0,
        room="a",
        ready=True,
        exit_forward=True,
        exit_id="a:b",
        exit_room="b",
        goal_dx=52.0,
        goal_dy=0.0,
        bounds_width=80,
        bounds_height=32,
        practice_available=True,
        map=dict(x=0, y=0, cell=8, width=10, height=4, tiles="." * 40),
        player_state=0,
        jump_held=False,
        dash_held=False,
        grab_held=False,
    )
    result.update(changes)
    return result


class PracticeGame(FakeGame):
    def __init__(self):
        super().__init__()
        self.value = room_state()
        self.slots = {}
        self.enabled = False

    def practice_mode(self, enabled):
        self.enabled = enabled

    def capture(self, slot):
        self.slots[slot] = copy.deepcopy(self.value)
        return copy.deepcopy(self.value)

    def restore(self, slot):
        self.value = copy.deepcopy(self.slots[slot])
        return copy.deepcopy(self.value)

    def clear(self, slot):
        self.slots.pop(slot, None)

    def step(self, action, frames=None):
        from protocol import action_frames

        elapsed = action_frames(action)
        self.frame += elapsed
        self.value.update(
            x=self.value["x"] + (int(action[0]) - 1) * elapsed,
            vx=(int(action[0]) - 1) * 60.0,
            frame=self.frame,
            action_frames=elapsed,
            jump_held=bool(action[2]),
            dash_held=bool(action[3]),
            grab_held=bool(action[4]),
        )
        if self.value["x"] >= 64:
            self.value["room"] = "b"
        self.value["goal_dx"] = 64 - self.value["x"]
        return copy.deepcopy(self.value)

    def start(self, chapter, stop=None):
        self.value = room_state(chapter=chapter)
        return copy.deepcopy(self.value)


class PracticeTests(unittest.TestCase):
    def test_route_field_goes_around_wall_and_rejects_unreachable_cell(self):
        wall = np.zeros((5, 7), dtype=bool)
        wall[1:, 3] = True
        field = distance_field(wall, (6, 4))
        self.assertEqual(field[4, 0], 14)
        wall[0, 3] = True
        self.assertTrue(np.isinf(distance_field(wall, (6, 4))[4, 0]))

    def test_route_can_return_to_an_earlier_room_for_an_unvisited_branch(self):
        navigation = Navigation()
        example = room_state(
            room="deadend",
            portals=[dict(room="hub", x=0, y=12)],
            route_graph=[
                dict(name="deadend", neighbors=["hub"], terminal_hint=False),
                dict(name="hub", neighbors=["deadend", "end"], terminal_hint=False),
                dict(name="end", neighbors=["hub"], terminal_hint=True),
            ],
        )
        result = navigation.annotate(example)
        self.assertEqual(result["exit_room"], "hub")
        self.assertTrue(result["navigation_valid"])

    def test_unchanged_states_and_discounted_closed_loops_cannot_farm_shaping(self):
        reward = Reward()
        a = room_state(
            navigation_revision=1,
            navigation_valid=True,
            navigation_id="a",
            navigation_distance=100.0,
        )
        b = dict(a, navigation_distance=50.0)
        self.assertLess(reward.step(a, a, 4), 0)
        r1 = reward.step(a, b, 4)
        r2 = reward.step(b, a, 4)
        self.assertGreater(r1, 0)
        self.assertLess(r1 + 0.99 * r2, 0)
        self.assertLess(reward.step(a, dict(a, dead=True), 4), -2)

    def test_room_score_reset_does_not_reaward_a_live_two_room_loop(self):
        reward = Reward()
        a = room_state(
            navigation_revision=1,
            navigation_valid=True,
            navigation_id="a",
            navigation_distance=100.0,
        )
        b = dict(a, room="b", exit_room="a")
        reward.reset(a)
        first = reward.step(a, b, 4)
        self.assertGreater(first, 4)
        reward.reset(b, fresh_attempt=False)
        self.assertLess(reward.step(b, a, 4), 0)
        reward.reset(a, fresh_attempt=False)
        self.assertLess(reward.step(a, b, 4), 0)
        # An explicit new practice attempt can earn its task completion again.
        reward.reset(a)
        self.assertGreater(reward.step(a, b, 4), 4)

    def test_success_memory_is_bounded_validated_and_survives_restart(self):
        policy = PlasticPolicy()
        records = [
            dict(
                observation=sensors(room_state()),
                action=np.array([2, 1, 1, 0, 0, 0, 3]),
                reset=True,
                frames=8,
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            memory = SuccessMemory(directory, policy.graph_hash)
            self.assertTrue(memory.add(records, 0, "a", "b"))
            self.assertFalse(memory.add(records, 0, "a", "b"))
            self.assertEqual(
                SuccessMemory(directory, policy.graph_hash).summary()["success_routes"],
                1,
            )
            self.assertEqual(
                SuccessMemory(directory, "wrong").summary()["success_routes"], 0
            )
            records[0]["reset"] = False
            self.assertFalse(memory.add(records, 0, "a", "b"))

    def test_imitation_gradients_match_finite_difference_through_fly_edges(self):
        p = PlasticPolicy()
        segment = dict(
            observations=np.random.default_rng(31).normal(size=(4, 156)),
            actions=np.tile([2, 1, 1, 0, 1, 0, 3], (4, 1)),
            resets=np.array([True, False, False, False]),
            initial=np.zeros(p.nodes),
        )
        _, gradients, _, _ = p.sequence_gradients([segment], objective="imitation")
        for parameter, gradient in zip(p.parameters, gradients):
            index = np.unravel_index(int(np.argmax(np.abs(gradient))), parameter.shape)
            old, epsilon = parameter[index], 1e-5
            parameter[index] = old + epsilon
            plus = p.sequence_gradients([segment], objective="imitation")[0]
            parameter[index] = old - epsilon
            minus = p.sequence_gradients([segment], objective="imitation")[0]
            parameter[index] = old
            self.assertAlmostEqual(
                (plus - minus) / (2 * epsilon), gradient[index], places=6
            )

    def test_imitation_learns_state_dependent_movement_and_hold_duration(self):
        p = PlasticPolicy()
        observations = np.zeros((64, 156))
        observations[:32, 0] = -1
        observations[32:, 0] = 1
        actions = np.tile([0, 1, 0, 0, 0, 0, 2], (64, 1))
        actions[32:, 0] = 2
        actions[32:, 6] = 5
        trajectory = dict(
            observations=observations, actions=actions, resets=np.ones(64, dtype=bool)
        )

        def probability():
            selected = []
            for obs, action in zip(observations[::32], actions[::32]):
                f = p.forward(obs, np.zeros(p.nodes))[0]
                from learning import softmax

                logits = f @ p.actor
                selected.append(float(softmax(logits[:3])[action[0]]))
                selected.append(float(softmax(logits[14:])[action[6]]))
            return np.asarray(selected)

        before = probability()
        for _ in range(70):
            p.imitate([trajectory] * 4, decisions=128, rate=0.02)
        after = probability()
        self.assertTrue((after > 0.85).all(), (before, after))
        self.assertGreater(p.diagnostics()["learned_edges"], 0)
        self.assertEqual(len(p.theta), 3326)

    def test_old_checkpoint_migration_preserves_parameters_and_optimizer(self):
        p = PlasticPolicy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            p.save(path, p.graph_hash, dict(chapters=list(range(8))))
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data}
            arrays["plastic_schema"] = np.array(PREVIOUS_PLASTIC_SCHEMA)
            with path.open("wb") as handle:
                np.savez(handle, **arrays)
            q = PlasticPolicy(seed=80)
            q.load(path, q.graph_hash)
            for expected, actual in zip(
                p.parameters + p.m + p.v, q.parameters + q.m + q.v
            ):
                np.testing.assert_array_equal(expected, actual)

    def test_restore_rebuilds_current_recurrent_memory_and_detects_mismatch(self):
        w = FakeWorker(0)
        w.client = PracticeGame()
        p = PlasticPolicy()
        runner = Runner(w, p, room_state(), config())
        self.assertTrue(runner.practice_available)
        self.assertTrue(runner.restore_checkpoint(0))
        self.assertTrue(runner.brain.last_reset)
        w.client.slots[0]["x"] += 1
        self.assertFalse(runner.restore_checkpoint(0))
        self.assertFalse(runner.practice_available)
        self.assertEqual(runner.restore_failures, 1)

    def test_search_uses_real_client_branches_and_restores_the_root(self):
        game = PracticeGame()
        teacher = SearchTeacher()
        start = state_signature(game.value)
        action = teacher.choose(game, room_state(), np.random.default_rng(17))
        self.assertEqual(action[0], 2)
        self.assertGreater(teacher.trials, 10)
        self.assertEqual(state_signature(game.value), start)

    def test_search_keeps_the_selected_portal_instead_of_changing_the_goal(self):
        game = PracticeGame()
        game.value.update(
            portals=[dict(room="left", x=0, y=12), dict(room="b", x=64, y=12)],
            route_graph=[
                dict(name="a", neighbors=["left", "b"], terminal_hint=False),
                dict(name="b", neighbors=["a"], terminal_hint=True),
                dict(name="left", neighbors=["a"], terminal_hint=False),
            ],
        )
        navigation = Navigation()
        navigation.destination, navigation.portal_point = "left", (0, 12)
        chosen = navigation.annotate(game.value)
        self.assertEqual(chosen["objective"], "portal:left")
        action = SearchTeacher().choose(game, chosen, np.random.default_rng(17))
        self.assertEqual(action[0], 0)

    def test_curriculum_keeps_training_and_frozen_evaluation_results_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            curriculum = Curriculum(Path(directory))
            curriculum.record(0, "a", True, 40)
            curriculum.record(0, "a", False, 80, "evaluation")
            curriculum.save()
            result = Curriculum(Path(directory)).summary()
            self.assertEqual(result["room_clears"], 1)
            self.assertEqual(result["evaluation_attempts"], 1)
            self.assertEqual(result["evaluation_clears"], 0)

    def test_teacher_clear_is_saved_and_excluded_from_ppo_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            p = PlasticPolicy()
            w = FakeWorker(0)
            w.client = PracticeGame()
            memory = SuccessMemory(Path(directory) / "memory", p.graph_hash)
            runner = Runner(w, p, room_state(), config(), memory=memory)
            runner.teach(8)
            self.assertEqual(runner.total_steps, 0)
            self.assertGreater(runner.room_clears, 0)
            self.assertEqual(memory.summary()["success_routes"], 1)
            self.assertEqual(
                next(iter(memory.entries.values()))["metadata"]["source"],
                "search_room_clear",
            )

    def test_room_evaluation_does_not_update_weights_or_claim_full_campaign(self):
        p = PlasticPolicy()
        w = FakeWorker(0)
        w.client = PracticeGame()
        with tempfile.TemporaryDirectory() as directory:
            curriculum = Curriculum(Path(directory))
            runner = Runner(w, p, room_state(), config(), curriculum=curriculum)
            parameters = [a.copy() for a in p.parameters]
            original = state_signature(w.client.value)
            result = runner.evaluate_room(60)
            self.assertFalse(result["weights_updated"])
            self.assertFalse(result["full_main_story_clear"])
            self.assertEqual(state_signature(w.client.value), original)
            for before, after in zip(parameters, p.parameters):
                np.testing.assert_array_equal(before, after)
            self.assertNotIn(7, w.client.slots)

    def test_teacher_pause_keeps_worker_and_does_not_restart_manual_play(self):
        p = PlasticPolicy()
        w = FakeWorker(0)
        w.client = PracticeGame()
        runner = Runner(w, p, room_state(), config())

        def paused(*args, **kwargs):
            raise ControlPaused("Manual control")

        runner.teacher.choose = paused
        with self.assertRaises(ControlPaused):
            runner.teach()
        self.assertTrue(runner.practice_available)
        self.assertEqual(runner.restore_failures, 0)
        self.assertEqual(runner.total_steps, 0)

    def test_recovered_process_reenables_and_checks_its_native_practice_state(self):
        w = FakeWorker(0)
        w.client = PracticeGame()
        runner = Runner(w, PlasticPolicy(), room_state(), config())
        original_client = w.client
        runner.episode_steps = 100

        def recover(chapter):
            w.client = PracticeGame()
            return w.client.start(chapter)

        w.recover = recover
        runner.recover_worker()
        self.assertIsNot(w.client, original_client)
        self.assertTrue(w.client.enabled)
        self.assertTrue(runner.practice_available)
        self.assertEqual(runner.episode_steps, 0)
        self.assertIn(0, runner.checkpoints)
        self.assertEqual(
            state_signature(w.client.value),
            state_signature(runner.checkpoints[0]["state"]),
        )

    def test_annotation_is_idempotent_for_native_exit_hint(self):
        navigation = Navigation()
        first = navigation.annotate(room_state())
        second = navigation.annotate(first)
        self.assertEqual(first, second)

    def test_recorded_inputs_reject_invalid_and_out_of_range_choices(self):
        self.assertEqual(
            validate_recording(dict(chapter=0, actions=[[2, 1, 1, 0, 1, 0, 6]])),
            [[2, 1, 1, 0, 1, 0, 6]],
        )
        for recording in (
            [],
            dict(chapter=8, actions=[[2, 1, 1, 0, 1, 0, 6]]),
            dict(chapter=0, actions=[[2, 1, 1, 0, 1, 0, 7]]),
            dict(chapter=0, actions=["bad"]),
            dict(chapter=0, actions=[]),
        ):
            with self.assertRaises((ValueError, BridgeError)):
                validate_recording(recording)


if __name__ == "__main__":
    unittest.main()
