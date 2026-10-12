# Speed, CUDA and remote training

I use native benchmarks and phase timings to decide which performance changes
help my training setup. I keep model-gradient speed separate from game-physics
throughput and do not infer a speedup from a GPU label.

## What changed in 4.1

I found that my old worker hook could skip Engine.Draw while FNA still ran its present step.
FNA's Game.Tick calls BeginDraw/Draw/EndDraw unless SuppressDraw is set, so this
left a display synchronization cost even when the game window was hidden.
The fixed outer loop also limited when Python commands could be consumed.

In the rebuilt worker, I switch the outer loop to variable timing only during
controlled training, supply 1/60-second GameTime to every native physics update,
and call SuppressDraw when hidden. Visible workers draw at up to 30 Hz. I restore
normal pacing when control is released. Ready workers wait on a command event,
with a short timeout to retain stop/control polling, rather than busy spinning.
Showing a worker can still incur display/presentation overhead.

Room maps, portal hints and route graphs are transmitted once per revision.
The client retains at most 16 revisions and rejects missing ready-state geometry.
World revisions change after rebuild/restore, so stale geometry cannot silently
be reused for a different room. Live reporting is limited to 10 Hz per worker. Hidden fast workers skip visual
particle aging, retaining emission calls and their random-number draws; arbitrary
entity and collision updates are preserved.
Search has a three-second default wall-time budget across its requested actions;
an in-flight native operation may take longer and must finish safely.

These changes are compiled and tested at the protocol/model level. **Native
Celeste performance and pacing fidelity still need benchmarking on my Windows PC.**

## My RTX 4060 Ti, i9 and 32 GB setup

I start with two hidden games. The exact i9 model, GPU VRAM variant, game footprint
and mod runtime matter more than the labels alone. Save 8 games without starting
learning if you want Benchmark to test 1/2/4/8. Its memory guard stops unsafe
additional launches. Choose the count with the best aggregate controlled frames/sec,
then check stability for an hour. 64 is a request limit, not a sensible default for 32 GB.

Keep frequently written training and worker files on a local SSD outside OneDrive.
Close other expensive applications and keep the PC awake. Do not change Windows
power, driver or security settings blindly; the timings should justify changes.

## Using CUDA

TOOLS → Install GPU support installs optional PyTorch 2.13.0 with the official
CUDA 12.6 wheel index. Only the private learning environment is modified. A
working compatible NVIDIA driver is required; the project does not install drivers.
The package does not require torchvision, a standalone CUDA toolkit or an SDK build.

The recurrent matrix is assembled only from the original MCNS edges and
trainable edge gains. PyTorch performs batched recurrent forward/backward gradients
for the same PPO and imitation objectives. It introduces no recurrent shortcuts.
CPU action inference stays local because each individual request is tiny.
Small gradients return once per minibatch; shared Adam arrays and NPZ saves keep
existing parameters, moments and RNG compatible across CPU/CUDA.

- **auto** checks gradient agreement, warms both backends and measures three
  end-to-end gradient runs. It selects CUDA only with at least a 10% measured win.
- **cpu** uses the NumPy reference learner.
- **cuda** forces the optional backend after availability/parity checks, and fails
  visibly if the GPU cannot run it. It does not quietly claim GPU acceleration.

CUDA uses float32; stored parameters and the CPU reference use float64. Tests
cover padded sequences, resets, exploration temperatures, both objectives,
connection updates and exact checkpoint reload. CUDA parity is an additional
hardware-gated test. Developer parity was run with PyTorch CPU; **CUDA execution
and 4060 Ti throughput were not tested here**.

The fly model has only 256 neurons. Kernel launch and transfer overhead can exceed
the arithmetic cost. Auto can correctly choose CPU even after GPU installation.
The GPU accelerates model math; it does not move C# Celeste entities or collisions
to CUDA. A full GPU-native simulator would require a separate faithful physics and
entity implementation across every chapter, followed by transfer validation.
That simulator is **not implemented** in this package.

## Measure the actual bottleneck

Training records collection, search, learning, evaluation and save/feedback times.
The largest phase is a useful first clue, not a precise profiler attribution.
The bridge also records per-command means/maxima; Benchmark records step p50/p95.

| Largest phase | Useful next experiment |
| --- | --- |
| Collection | Hide games, check fast mode, compare native worker counts and step latency |
| Search | Increase teacher_interval or reduce teacher_seconds; retain some search for successes |
| Learning | Run GPU check with the intended sequences_per_batch; compare actual update seconds |
| Evaluation | Less frequent scheduled room tests, retaining separate full-policy evaluation |
| Save/feedback | Local SSD outside syncing folders; inspect locks and available disk space |

The benchmark does not update the model. It compares one display-paced worker
with one fast worker, then fast process counts within the configured limit.
The reported speedup is measured from this test, not estimated from GPU specifications.
Its scenario includes actual movement and respawn overhead, but is not an overnight
stress test or a measure of how quickly the policy masters rooms.

## What “years in days” would mean

One 365-day game year at 60 Hz is **1,892,160,000 frames**. Summing parallel workers:

| Aggregate controlled frames/sec | Game experience in one real day | Real days for one game year |
| --- | --- | --- |
| 60 | 1 game day | 365 |
| 600 | 10 game days | 36.5 |
| 3,129 | About 52.15 game days | About 7 |
| 21,900 | 365 game days | 1 |

These are arithmetic examples, **not measured release results**. “Years of
experience” describes elapsed simulated frames across workers, not useful diversity,
biological age, reasoning ability, or a game-clear probability. Repeating the same
failed wall jump for a year adds little information. Success examples, curriculum,
exploration and independent evaluation matter as much as raw frame rate.

## Running somewhere else

The supported remote route is a **Windows desktop VM or another Windows PC** with
adequate CPU, memory, graphical-driver support and your own Celeste/Everest install.
A remote learner runs games and Python together on that machine. The bridge remains
authenticated on localhost; do not expose its ports on the internet.

1. First measure local native throughput and the largest phase.
2. Choose a remote Windows machine with more usable CPU/memory if collection is
   limiting. A larger CUDA GPU alone may not help. Obtain a current price and set
   a budget before creating a paid instance.
3. Azure NVadsA10_v5 is one documented Windows virtual-workstation family; follow
   Microsoft's supported driver instructions. Verify both game rendering and CUDA
   on the chosen instance. This is a compatibility candidate, not a tested deployment.
4. Install your owned game, Everest and Python 3.11/3.12. Extract this same release.
5. STOP locally, then copy training/ to the remote project. Select its own game path
   and Save settings. Never synchronize two writers into the same training folder.
6. Benchmark there and compare controlled frames/sec per dollar with local results.
7. Start saved settings from TOOLS, keep the desktop session available and the VM awake,
   and confirm it continues after disconnecting remote desktop. Sign-out is different
   from disconnect. Copy completed training backups back periodically.
8. STOP, copy the model/results back and run frozen evaluation locally. Stop/deallocate
   paid compute when finished; disks may still cost money.

Runpod Pods currently do not support Windows, per their documentation. This Windows
launcher is therefore not a drop-in Runpod or Colab training setup. Linux-only GPU
containers can run offline model work, but collecting native Celeste experience
would need a separate supported Linux game-launch/rendering path. No cloud account,
paid resource, distributed learner, Linux native simulator or automatic fleet is
created by this release. Do not rent an expensive GPU for an unmeasured bottleneck.

## A seven-day experiment, October 11–18, 2026

A deadline is a target, not a guarantee. Treat the first day as a go/no-go for
reliable sampling rather than letting a broken run continue for a week.

| Day | Work | Evidence to retain |
| --- | --- | --- |
| 1 | Validate resumed model, movement, save/restore, fast pacing; benchmark 1/2/4/8; compare CUDA | Throughput, timings, errors, checkpoint backup |
| 2 | Train Prologue; inspect selected exit, input holds and successful routes | Clears/attempts and separate frozen room tests |
| 3 | Rehearse early successes; investigate the first consistently failing room | Actual room clip and failure distribution |
| 4 | If search cannot discover successes, provide a verified demonstration JSON and label assistance | Accepted routes and assisted/learner counters |
| 5 | Advance curriculum with measured useful throughput; remote compute only if it improves it | Same metrics and actual remote cost |
| 6 | Freeze checkpoint; full-story evaluation from Prologue, no teacher or practice | Completion sequence, model hash, interruptions |
| 7 | Repeat evaluation; record outcome and failure cases; produce the video | Honest result, reproducible settings, evidence table |

If Prologue cannot clear independently, running more copies is not evidence that
later chapters will work. Inspect the objective, timings and examples first.
Training may clear rooms with assistance before the frozen policy can repeat them.
A supported JSON demonstration can supply useful success experience; importing an
unverified TAS file or asserting that a replay is learned skill is not supported.

## Sources checked October 11, 2026

- [CelesteTAS visual optimization rationale](https://github.com/EverestAPI/CelesteTAS-EverestInterop/blob/master/CelesteTAS-EverestInterop/Source/Gameplay/Optimization/FastForwardOptimization.cs).
- [FNA Game.cs](https://github.com/FNA-XNA/FNA/blob/master/src/Game.cs): fixed loop and SuppressDraw/present behavior.
- [Celeste Player source](https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs): native movement/controller physics.
- [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): overhead, batching and profiling.
- [Official PyTorch wheels](https://pytorch.org/get-started/previous-versions/): Windows CUDA installation.
- [Azure workstation VM family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series).
- [Azure Windows GPU drivers](https://learn.microsoft.com/en-us/azure/virtual-machines/windows/n-series-driver-setup).
- [Runpod Pods limitations](https://docs.runpod.io/pods/overview).

## GPU rehearsal without the game open

TOOLS → Rehearse saved successes runs one hour of imitation on a **copy** of the
saved model. It needs at least one previously verified successful room route.
It writes `training/offline/latest.npz` and a report, keeping `training/latest.npz`
unchanged. It counts **replayed actions and zero new game frames**. This can
strengthen already discovered movements; it cannot discover unseen rooms or prove
a game clear. Repeated examples can overfit, so evaluate the candidate before use.

This tool also works on a Linux CUDA host. Copy the project and training/latest.npz
plus training/success-memory there, install Python/NumPy and the official PyTorch
CUDA wheel, then from the project root run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python Python/rehearse.py --device auto --hours 1
```

No game, display or Windows session is needed for this **replay-only** job. Runpod
or another Linux GPU host can support this narrower workload; its benefit depends
on actual backend timings and on whether recorded successes cover your failing skills.
Do not assume a large accelerator improves this tiny graph. No paid job is launched
by the project. CUDA cloud execution itself has not been validated here.

To try the candidate, STOP both runs, back up your original training/latest.npz,
and copy training/offline/latest.npz over it. Run Evaluate from a fresh Prologue.
Keep the original model if the candidate does worse. Never merge optimizers or
write live/local and cloud runs into the same output file.
