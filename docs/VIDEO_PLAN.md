# Video essay: Can fly wiring learn Celeste?

## The story

I started with a yellow circle in Pygame. I wanted to use real fly connection data
in a controller and watch it learn a difficult platformer. Moving into a real game
made the hard parts visible: input timing, deceptive points, game resets, and
training speed. The payoff is the measured result of the latest frozen policy,
including what it still cannot do.

Working title: **I Used Fruit Fly Wiring to Teach an AI Celeste**.
If the final run succeeds, a clear can be the hook. If it fails, keep the same
question and show the strongest independently verified progress. Do not title it
as a completed game until a real full-story evaluation establishes that result.

## A 10–12 minute cut

| Time | Segment | Footage | Point to make |
| --- | --- | --- | --- |
| 0:00–0:25 | Cold open | Strongest actual evaluation clip, then early wall-jumping failure | What changed, and the outcome this video will explain |
| 0:25–1:20 | Why try this? | Pygame window, simple reward diagram, my original files | A beginner experiment with real connection data |
| 1:20–2:15 | What “fly brain” means | 256-node/3,326-edge graph and a few highlighted edges | A selected MCNS subgraph, artificial sensors/readout, trainable gains |
| 2:15–3:15 | Learning one action | Slow clip showing observation → fly activity → buttons/hold → outcome | Short-term activity and saved learned parameters are different memories |
| 3:15–4:35 | The first failures | My actual backward-room/wall-jump clips; requested input vs actual velocity | Points can be misleading; diagnoses need game state and independent clears |
| 4:35–5:45 | Finding useful experience | Native saved room, bounded search, successful-route replay with labels | Search-assisted experience and separate imitation training |
| 5:45–7:05 | The speed problem | Two-worker lag, bridge diagram, before/after benchmark, GPU-check report | Frame pacing and hidden presentation mattered; CUDA must be measured |
| 7:05–8:35 | Unattended experiment | Short training time-lapse, monitor PIDs, curves from recorded metrics | Actual worker count, useful sampling, wall time and assistance |
| 8:35–10:05 | The test | Continuous frozen-policy attempt from Prologue, failures included | No weight changes, teacher or practice restores; actual completion events |
| 10:05–11:30 | What I learned | Outcome table, code walkthrough, next failing room | Evidence supports specific progress; topology advantage remains an experiment |

## Lines for my narration

- “I began by getting one circle to move toward a goal. Then I tried Celeste.”
- “I selected 256 fly neurons and kept their recorded connections.”
- “The strengths of those existing connections learn along with the buttons and how long to hold them.”
- “I found that collecting points didn't always mean clearing a room, so I added a separate test.”
- “Hiding a game window wasn't enough. The framework could still present a frame.”
- “I tested the GPU instead of trusting the label. This little network can be cheaper to run on the CPU.”
- “Here's the saved policy with learning frozen. This is the result it can reproduce.”

I will explain the development assistance plainly: **“I developed this with
assistance from Codex.”** I will describe my goal, dataset choice, original
Pygame work, Windows testing, observations, and decisions I can explain. I will
credit MCNS and SpeedrunTool and tie engineering claims to code and evidence
I have reviewed and understand.

## Recording plan

1. Save a copy of the checkpoint and config before each meaningful experiment.
2. Record the monitor showing the actual release, PIDs, active process count and phase.
3. Record one visible worker; hide the others. Window visibility changes throughput,
   so capture the benchmark separately with all workers hidden.
4. Record the benchmark console and training/throughput.json. Show both worker count
   and total frames/sec; never substitute optimizer iterations for game experience.
5. Record GPU setup/check output with the selected device and measured times. If auto
   retains CPU, keep that result in the story rather than hiding it.
6. Keep two or three representative failure clips, a successful assisted attempt,
   and a frozen room test. Put assistance labels on screen.
7. STOP training and run Evaluate. Record an uninterrupted test from the Prologue.
   Preserve evaluation.json, the model hash, seed, outcome and manual interruptions.
8. For every progress chart, use actual progress.jsonl or metrics.csv. Label x-axis
   wall time or controlled decisions/frames clearly. Keep failures in the denominator.

A planning example is not a test result. Synthetic gradient tests belong in the
engineering segment and must be labeled as offline tests. Do not edit a montage
of separate assisted chapter clears into an apparent continuous autonomous run.

## On-screen result table

Fill this from actual logs; leave unavailable results as “not measured.”

| Evidence | Baseline | Latest run |
| --- | --- | --- |
| Actual games/PIDs | | |
| Controlled frames/sec | | |
| Wall hours / controlled game days | | |
| Assisted clears / attempts | | |
| Frozen room clears / attempts | | |
| Full frozen campaign, start/seed/hash | | |
| CPU/CUDA selection and update seconds | | |
| Restore failures / interruptions | | |

## Production and portfolio handoff

I will use game footage and my voice as the main visual language. Explain a single
piece of code at a time; avoid long screenshots of files. Record narration in a
quiet room, keep levels consistent and remove long pauses. Add captions for terms
such as PPO, MCNS and frozen policy. Use game/music assets only under the rights
you have; check a platform's music rules before uploading.

I will link my GitHub repository and setup guide in the description, name the exact
release, summarize the actual result and credit data/tools/assistance. Add chapters
matching the timeline. Put limitations and evaluation instructions in the repository.
Use one resume bullet with a measured result only after the evidence exists.

A defensible portfolio result can be a well-measured improvement and a thoughtful
failure analysis. A complete game clear is a valuable additional result, not a
prerequisite for telling the experiment's actual story.
