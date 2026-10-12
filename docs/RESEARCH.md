# Research decisions and the completion goal

My target is the main-story campaign, Prologue through Summit. Core, Farewell,
B/C sides, strawberries, and a 100% collectible run are outside this release.
A fast, verified completion remains a goal, not a promised outcome.

## Why these changes

| Finding | Implementation in 4.0 | Limit |
| --- | --- | --- |
| Celeste movement has distinct normal/climb/dash/dream/feather states | Player-state sensors, held-button history, variable durations | Sensors do not expose every internal timer |
| Random exploration can repeatedly lose useful progress | Native entry/frontier saves and bounded branch search | Two-action lookahead can miss long setup sequences |
| A successful attempt should influence later decisions | Persistent route memory and separate imitation gradients | A teacher-assisted route is not an unaided policy result |
| Straight-line distance can aim through a wall | Coarse collision distance field and waypoint | Free-cell connectivity is not physics reachability |
| Puzzles can require returning through a room | Portal graph, failure-aware target choice, key/carry hints | Not a complete key/door/switch dependency planner |
| More games can exhaust resources | Staged launch, measured PIDs/memory, optional throughput benchmark | Native capacity must be measured on the actual machine |
| Points can be mistaken for skill | Independent room/campaign evaluation and model hashes | Full-game completion still needs an actual recorded run |

I use [Noel Berry's published Player.cs](https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs)
as a movement reference for state encoding, stamina/dash observations, and press/hold distinctions.
The published controller is a reference, rather than the entirety of Celeste's
current Everest-patched mechanics. Physics remains in the actual game.

I delegate native cloning to [SpeedrunTool](https://github.com/DemoJameson/Celeste.SpeedrunTool).
The bundled version is 3.27.21, obtained through Everest's updater mirror.
Its API metadata, package identity, and SHA256 are checked. The adapter keeps
named slots, excludes the HUD, and preserves FlyBrain's held-button history.
No game code or copyrighted game binaries are copied into this repository.

The [Go-Explore paper](https://arxiv.org/abs/1901.10995) motivates returning to
useful states before further exploration. I apply that idea to a
small entry/frontier practice system. It does not implement the paper's entire
archive, exploration selection, or robustification procedure.

[Self-Imitation Learning](https://arxiv.org/abs/1806.05635) motivates reusing good
experience. I use categorical behavior cloning of successful routes, separate
from PPO, rather than the exact published SIL loss.
[Potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
informs the discounted potential differences I use. My navigation state is approximate,
objectives can change, and artificial practice/curriculum resets change the
training task; no theorem guaranteeing optimal play is claimed.

## What can still block finishing

The controller has controls and observations for dashes, stamina, moving objects,
wind, carrying, keys, and special player states. Later rooms also require
precise timing, switches, long sequences, and puzzle-dependent routing. A
geometric exit hint can still select a bad objective. Search can stall at a
local optimum, and learned skills can interfere across stages.

The current safeguards preserve good examples, rehearse earlier skills, avoid
false reward loops, and test independently. They reduce specific failure modes;
they do not establish that every room is solved. If a room remains blocked,
use its actual logs and frozen-entry results to choose the next change.

The fastest additional evidence-driven work would be:

1. Measure room-entry clears with the same checkpoint and starting conditions.
2. Inspect requested inputs, actual velocity, player state, and selected objective
   when progress stops. Fix missing observations or an incorrect objective first.
3. Add deeper adaptive search or explicit puzzle-state planning for a demonstrated
   failure that the bounded teacher cannot solve.
4. Optionally verify an expert recording, then teach that route through the same
   network. Default LEARN remains usable without one.
5. Compare multiple training seeds and alternative recurrent wiring using equal
   real-game experience. Do not infer a fly-graph advantage from one run.
6. Run independent EVALUATE from Prologue and retain the model hash and recording.

A guaranteed completion would require verified solutions for every supported
room plus reliable execution/recovery under actual game conditions. Such a
route executor would be a separate engineering guarantee; it must not be
presented as proof that a learned policy solved the game alone.
