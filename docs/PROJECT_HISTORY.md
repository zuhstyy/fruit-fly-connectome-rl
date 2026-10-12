# How this project developed

This history follows the implemented files and the issues I encountered during
testing. I chose the experiment and provided Windows gameplay feedback that
shaped the changes. I developed the project with assistance from Codex.

## 1. Start with something small

My first environment was an 800 by 400 Pygame window with a yellow circle and
a green goal. I started with keyboard movement, then added random left, neutral,
and right actions held for 15 frames. Moving closer to the goal produced a
positive reward, while moving farther away produced a negative reward.

This helped me understand the observation, action, and reward loop before
introducing a full game or a trainable neural controller.

## 2. Choose real connection data

I chose the male MCNS v1.0 dataset and provided the neuron and connection
exports. I use the selection script to produce a manageable connected subset
around a descending-neuron anchor. It saves neuron IDs, edge indices, synapse
counts, annotations, normalized initial weights, and source checksums.

I retain a historical provenance report that checked all selected neurons and
edge counts against those exports. I can repeat that verification with the
optional source-data check when the originals are available. This supports the
provenance of my selected data; I am not running the entire male fly nervous
system inside the game.

## 3. Connect Python to Celeste

I use an Everest mod to expose game observations and accept frame-counted
actions through a loopback TCP bridge. In Python, I choose buttons and send
those choices back to the game. I check a per-worker token, protocol version,
action ranges, and bounded queue sizes. Celeste applies commands on its main thread.

One early startup problem came from creating virtual inputs before the engine
had initialized its input system. I moved input creation into the gameplay
lifecycle and restore real bindings after each simulated update.

I also encountered startup failures caused by duplicate FlyBrain packages.
Installation repair now identifies packages by their manifest names and moves
old copies into a backup outside the Mods folder.

## 4. Make unattended startup inspectable

I wanted unattended training to be easier to inspect and troubleshoot. I create
isolated game copies and training saves, then verify the bridge's actual worker
identity, save location, and live movement before using observations for training.

I preserve process and game logs in startup diagnostics. I configure Steam
launches for worker copies without editing my personal saves, and handle a
launcher process exiting separately from the game process it starts.

## 5. Correct misleading progress

During testing, I saw the controller repeatedly return from Prologue room 0 to
the earlier car room, numbered -1, and jump against walls. My original geometric
route calculation treated both directions from room 0 as forward. I now use
signed numeric Prologue room ordering to identify room 1 as the forward target
and room -1 as a backward transition.

I compare distance shaping against the same valid forward target, apply explicit
backtracking and death penalties, and prevent repeated discovery rewards for
revisiting forward rooms. I removed timer and stall behavior that could force
deaths. Beyond Prologue, I still rely on a geometric route heuristic, which
remains a limitation for branches and puzzles requiring backtracking.

## 6. Train the connections, not only the output

In my earlier design, I trained an output readout while keeping the fly
connections fixed. I wanted learning to reach the selected connections themselves.

I now train a bounded gain on every selected recurrent edge, along with the actor
and value layers. I use truncated recurrent backpropagation and finite-difference
tests to check that gradients reach those gains. I preserve the real edge topology
and initial signs.

## 7. Learn control duration and count real games

In version 3.1, I added a duration output, elapsed-frame reward and credit
calculations, and persistent button history. This lets the controller learn how
long to hold an input.

I also replaced exploration-stream accounting with actual separate game workers.
I use the live monitor to distinguish requested games from processes that are
alive and ready, and to show or hide worker windows. I share learned parameters
across workers while retaining their individual short-term activity.

I use batched sequences in the recurrent optimizer. A local synthetic benchmark
measured about 21% less optimizer time than the previous unbatched version. That
result measured the optimizer workload, not Celeste throughput or my Windows PC.

## 8. Make it reviewable for a portfolio

For version 3.2, I focused on making the project easier to read and review. I
expanded compressed statements, gave the central learning steps clearer names
and explanations, preserved the 3.1 checkpoint format, and added CHECK_BRAIN.

I organized documentation covering setup, learning, code-reading, data credits,
validation, and contributions. I configured CI for offline checks without
downloading or playing Celeste. I keep public example settings separate from
my local paths and training output.

## 9. Preserve progress and learn successful routes

In version 4.0, I added native entry and frontier practice through SpeedrunTool,
bounded automatic movement search, saved successful routes, and separate
imitation gradients through the same fly connections. I expanded observations
to include special movement states, held buttons, nearby objects, wind, and inventory.

I allow navigation to return through earlier rooms when necessary. I use
discounted route shaping and retained live-room bonus guards to address
repeated-point loops. I separate assisted training progress from unaided policy
results with frozen-policy room tests, independent campaign evaluation, and
search/reset counters.

I document the reasoning behind these choices and their limitations in the
research notes. I still need to establish whether the two-action search is
sufficient for later puzzles and whether the learned policy can finish a
campaign without assistance.

## What remains to establish

The newer learner appeared to behave better during my testing, but I do not have
a measured, controlled Celeste room-success rate or campaign clear in the
bundled evidence.

My next steps are to record attempts from consistent room starts, confirm actual
forward exits, and compare the saved policy against an untrained baseline.

I have implemented practice checkpoints, curriculum statistics, successful-route
training, and an optional verified demonstration importer. I still need full
native validation, deeper search or puzzle planning where necessary, reliable
independent room clears, and a recording of an actual campaign clear.

## October 11: performance rebuild

I reported that even two games were running slowly on my PC, which has an i9,
RTX 4060 Ti, and 32 GB of RAM. Inspection found that skipping scene drawing did
not suppress FNA presentation, and my worker's outer loop still followed fixed
display-time pacing.

In version 4.1, I changed both paths while retaining 1/60-second physics updates.
I reduced repeated room data, bounded search, and added measurable phase and
frame timings.

I added an optional tensor backend for the same wiring. I select CUDA
automatically only after measuring an advantage over the CPU. I consolidated
the launchers and added a video plan to the documentation.

I still need evidence from Windows testing for native performance and full-game
completion.
