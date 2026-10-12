# How the fly network learns

I use `PlasticPolicy` as my active controller. I pass structured game state
through a selected fly recurrent graph and sample buttons plus a duration.
I run those inputs in actual Celeste physics through the mod and collect the
result. A death, transition, or completion can interrupt a hold.

## Observations

There are **156 scaled sensors**, clipped to -1 through 1:

| Group | Count | Meaning |
| --- | --- | --- |
| Player and route | 11 | Route offset, velocity, ground/walls, dashes, stamina, room position |
| Local grid | 81 | 9 by 9 collision/entity samples around the player |
| Player state | 24 | One-hot normal, climb, dash, dream dash, feather, tutorial, etc. |
| Control and inventory | 8 | In-control, carrying, held buttons, wind, keys |
| Nearby objects | 30 | Six objects' relative positions, velocities, and semantic classes |
| Navigation | 2 | Route validity and estimated remaining distance |

The mod also reports a coarse room collision map, neighboring portals, a
geometric room graph, and up to 16 relevant objects. Python uses those to make
a stable waypoint and distance hint. They are privileged structured observations,
not screenshots. Static free-cell paths do not establish physical jump reachability.
Object selection and semantic categories are incomplete puzzle descriptions.

A chosen objective can be an exit, a visible required key, or a Theo pickup.
Returning through an earlier room can be valid. The Prologue's car room is
explicitly deprioritized. Other chapter routes are heuristic, not solved maps.

## What comes from the fly

The graph contains **256 selected MCNS neuron IDs and 3,326 directed edges**.
The original source table has 166,700 neurons. Selection anchors on a descending
neuron and its strongest eligible direct neighbors, then retains their existing
connections. This is a small selected graph, not the full nervous system.

Initial weights use `log(1 + synapse_count)`, incoming normalization, and a
simplified sign rule: GABA/histamine negative, other or unknown types positive.
That rule does not reconstruct receptor-dependent biology. The original
92-sensor mapping and a deterministic projection for 64 extra sensors are
artificial engineering choices.

| Array | Role | Trainable? |
| --- | --- | --- |
| `source`, `target` | Existing directed fly connections | No |
| `theta` | One strength adjustment per connection | Yes |
| `actor` | Button/duration probabilities from neuron activity | Yes |
| `value` | Expected future reward | Yes |
| `inputs` | Game-to-neuron projection | No |

Each edge is `base_weight * exp(theta)`, with gain bounded to 0.25 through 1.1.
Its sign and endpoints stay fixed. Incoming absolute weight sums remain bounded
by approximately 0.935. Every decision performs three rate updates:

```python
sensor_input = inputs @ observation
incoming[target] += edge_weight * activity[source]
activated = tanh(incoming + sensor_input)
activity = 0.65 * activity + 0.35 * activated
```

The actor reads all 256 resulting neuron activities. There is no observation
shortcut around the recurrent model. PPO and imitation gradients travel back
through it to the actual edge-gain array. This is a continuous rate calculation,
not biological spikes or the learning rule of a living fly.

## Buttons and duration

Seven categorical heads choose horizontal, vertical, jump, dash, grab, interact,
and **1, 2, 4, 8, 12, 20, or 30 simulation frames**. There are 1,008 combinations.
At 60 Hz, 30 frames is half a second. Consecutive decisions can keep buttons
held longer while changing other buttons. Releasing ends a hold; holding dash
does not extend Celeste's dash or cause a new press each frame.

Button history is retained across bridge requests and native restores. Jumping
against a wall can produce game-imposed motion opposite the requested direction.
The monitor reports requested buttons and actual velocity separately so this
can be diagnosed instead of assumed to be an input swap.

## Reward

The main 4.0 route reward uses a bounded potential:

```python
potential = 4 / (1 + route_distance / 80)
shaping = 0.99 ** (executed_frames / 4) * new_potential - old_potential
reward = shaping - 0.01 * executed_frames / 4
```

Potential is zero on death/completion or an invalid route hint. This discounted
difference makes unchanged states and closed loops pay a cost. It guides
movement around static walls; it does not prove a path can be jumped.

| Actual event | Additional reward |
| --- | --- |
| Selected target-room entry | +5 |
| Game-reported chapter completion | +25 |
| Real game death | -2 |
| Practice restore | No reward event; new attempt boundary |

Revisiting live rooms cannot repeatedly award their transition bonus. An
explicit new practice attempt can earn its completion bonus again. Room score
resets independently from that guard. Run return remains separate. There is no
blanket penalty for earlier room names, no forced `Player.Die`, and no completion
flag set by the trainer. The old reward branch remains for compatibility tests;
current training annotates every state with navigation revision 1.

GAE assigns credit backward using actual frame discounts, `0.99 ** (frames/4)`
and `0.95 ** (frames/4)`. Death and room-task boundaries stop credit crossing
attempts. Time-limit practice resets bootstrap from the value of the state
before the reset; they do not credit the teleported position or future attempt.
PPO clips probability changes and includes entropy exploration and value loss.

## Practice and the automatic teacher

The bundled SpeedrunTool saves the native level, rather than approximating it
by changing coordinates. FlyBrain keeps input edge history alongside each slot
and excludes its socket-owning HUD from cloning. The worker checks position,
velocity, player state, contacts, inventory, held buttons, and object positions
after restore. These observable checks are guards, not proof of every hidden
state field. A mismatch disables practice and is reported.

Each worker can save its room entry and a later grounded frontier. After real
death or a stall, it practices from these states. Some attempts start at entry
so early movement is rehearsed too. Neural activity is reconstructed from the
saved observation prefix using **current** learned weights.

Between PPO rollouts, one eligible worker rotates through bounded two-action
search. It tests run/jump/grab/climb, release, interaction, available dash
directions, and state-appropriate swimming/feather motion. Durations vary.
Actual game outcomes rank the branches using survival, chosen exits, route
hints, and bounded novelty. The root is restored before the chosen action runs.

Trial actions, trial deaths, and executed teacher actions have separate counters.
Search does not make a learned room-clear claim. It is enabled during default
training and completely disabled in independent evaluation.

## Two learning objectives

1. **PPO:** learn from fresh actions sampled by the current policy and their
   actual rewards. Normally two epochs, eight recurrent sequences per batch,
   32 decisions per sequence, learning rate 0.001, and KL early stop at 0.03.
2. **Successful-route imitation:** retain complete successful observation/action
   sequences on disk, sample them, and minimize categorical cross-entropy over
   all seven action heads. This separately trains the actor and fly gains with
   learning rate 0.0003 and up to 128 decisions per update.

Both use recurrent backpropagation and Adam with gradient norm clipping at 0.5.
Old success data and teacher actions never receive invented PPO log probabilities.
Memory retains up to 256 routes, at most three shorter variants per room and
2,048 decisions per route. Incomplete prefixes and incompatible graph/schema
files are rejected. Optional demonstrations must first complete their chapter
in the actual game before import.

This is successful-trajectory behavior cloning inspired by self-imitation.
It is not a reproduction of the published advantage-filtered SIL algorithm,
nor a complete Go-Explore archive or long-horizon planner.

## What survives a restart

| Memory | Lifetime |
| --- | --- |
| Per-worker neuron activity | Current decision history; reset/rebuilt at attempt boundaries |
| Actor, value, gains, optimizer moments, RNG | Atomic `training/latest.npz` saves |
| Successful routes | `training/success-memory/*.npz` |
| Attempt and evaluation statistics | `training/curriculum.json` |
| Full native practice states | Worker RAM only; not restored after closing the game |

A new run retains learned arrays and resumes at the start of the furthest reached
chapter. Completing Summit starts another campaign cycle during training.
Worker activity and live positions are not model files. Every fourth worker can
rehearse earlier stages; the others generally stay at the reached frontier.

## Proving progress

Automatic room tests use a frozen current policy from saved entry, with no
search, weight updates, or frontier restores within the attempt. They restore
the training state afterward. They are room tests, not full-game evidence.

EVALUATE starts a fresh Prologue and records game-reported completions through
Summit. It uses frozen weights, seeded action sampling by default, ordinary
respawns, and automatic chapter loading. Teacher/practice are disabled. Manual
interruptions invalidate its full-clear claim. Menus/cutscenes are handled by
the mod; this is structured-state game automation, not a vision-only agent.

CHECK_BRAIN establishes wiring influence, trainability, and exact model reloads
on synthetic inputs. VERIFY reports actual training/evaluation records. Neither
higher points nor changed weights alone establishes room-solving improvement.
Matched room starts, multiple attempts/seeds, checkpoint hashes, and recordings
are needed for stronger results. See [credits](CREDITS.md) and [research](RESEARCH.md).
