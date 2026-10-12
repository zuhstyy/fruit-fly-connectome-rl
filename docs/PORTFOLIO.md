# Portfolio presentation

## Project summary

I started this project to explore whether real fruit-fly connection data could
be used in a learning controller. I began with a small Pygame movement/reward
experiment, chose the male MCNS dataset, and used Windows testing feedback
as I moved into Celeste. I developed the project with assistance from Codex.

I use a selected 256-neuron graph with 3,326 directed MCNS edges. I train their
strength adjustments along with button and hold-duration choices. In version
4.0, I added native practice checkpoints, automatic movement search, persistent
successful-route memory, and separate imitation training. I also separated
assisted training progress from frozen-policy evaluation.

My goal is unattended Prologue-through-Summit learning. I have not established
a complete campaign clear or a measured fly-graph advantage.

In version 4.1, I addressed display pacing, hidden presentation overhead,
repeated world data, and expensive search. I added an optional CUDA gradient
backend with measured automatic selection and controlled-frame throughput
reports. I still need actual hardware measurements of native speedups and CUDA
performance.

## Contributions

| Contribution | Attribution |
| --- | --- |
| Experiment goal and dataset choice | I chose the experiment and selected MCNS v1.0 |
| Original Pygame exploration and Windows gameplay feedback | I started the prototype and supplied testing feedback, with assistance |
| Integration, learning implementation, refactoring, checks, and packaging | I developed these with assistance from Codex |
| Neuron and connection reconstruction | Upstream MCNS research teams |
| Native practice-state machinery | SpeedrunTool maintainers, MIT license |

## Resume bullet

I developed and tested a C#/Python Celeste learning project using a trainable
256-neuron MCNS recurrent graph, variable-duration controls, native practice
states, successful-route imitation, isolated workers, and evaluation reports.

## What to show

1. Explain the original experiment and why structured game observations were used.
2. Run TOOLS → Check fly model; show graph identity, edge ablation, gradients, and exact reloads.
3. Open MONITOR and one actual game; compare requested inputs with actual velocity.
4. Show saved successful routes, practice/search counters, and frozen room tests.
5. Walk through `PlasticPolicy.forward`, `Runner.roll`, and the reward formula.
6. Discuss a concrete bug: early input initialization, backward target selection,
   live room-loop bonuses, or incorrect worker counts, with its supporting check.
7. Run TOOLS → Evaluate and report its actual result and model hash, including failures.

The strongest demonstration is a repeatable real-game result with a clear start,
frozen model, recording, and stated assistance. Training points or changed weights
are evidence of activity; they are not sufficient evidence of room mastery.

## Supported claims and future results

I preserve traceable source data and test whether recurrent edges affect
decisions and their gains train. I persist model and successful-route state.
I use offline checks for gradients, conditional movement/duration learning,
practice guards, controls, and worker coordination. I organized the repository
with setup, research, validation, licenses, tests, and CI configuration.

Full native practice fidelity, Windows visibility, 64 simultaneous live games,
overnight stability, room-success rates, and campaign completion require actual
Windows results. Do not describe the model as a whole fly brain or promise a
clear. Do not label search-assisted clears as unaided policy results.

When adding results, record the release, date, room/start state, checkpoint hash,
seed, worker count, experience budget, assistance settings, successes/attempts,
throughput, and recording. Compare alternative wiring only under matched budgets.
