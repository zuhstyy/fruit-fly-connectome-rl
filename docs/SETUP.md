# Setup and saved progress

I use the launcher to configure isolated workers and preserve learning between
runs. I include the compiled bridge both as `prebuilt/FlyBrainCeleste.dll` and
inside `prebuilt/FlyBrainCeleste.zip`; the launcher installs the mod ZIP.

Use Windows, Python 3.11 or 3.12, your purchased Celeste installation with Everest,
and a local folder outside the game. The compiled mod needs no .NET SDK build.

## First run

1. Extract all project files into a local folder, for example `C:\FlyBrain`.
2. Open START.cmd. Choose the folder with Celeste.exe, Celeste.dll, FNA.dll and MMHOOK_Celeste.dll.
3. Select 2 games, hidden windows, fast simulation and the auto learner.
4. Save settings. This creates config.json; training progress alone does not create it.
5. Press LEARN. Watch the movement checks and ready-process count.
6. MONITOR.cmd shows real PIDs and supports showing one game while the others train.

The first run creates a private Python environment and downloads NumPy. GPU support
is optional in TOOLS. Each worker receives a compiled FlyBrainCeleste mod and
SpeedrunTool in its own game copy; bridge source is in Source/BridgeServer.cs.
No manual save selection is needed. Leave your personal save in the normal game.

## Upgrading

1. STOP the old run and wait for its workers to close.
2. Extract 4.1 into a fresh folder. Old launcher files are intentionally consolidated.
3. Copy only your old training folder. You may copy config.json, but review the settings.
4. Open START, Save settings and LEARN. Confirm the message that existing decisions
   and fly weights were resumed.

A 4.0 model loads without a schema change. 3.1/3.2 arrays and optimizer moments are
preserved with a migration backup. Earlier fixed-network models are backed up and
start a fresh trainable model. Native practice states live in RAM; restart resumes
at the beginning of the furthest reached chapter, not an exact game position.

## Controls

The launcher and monitor provide ON/OFF and STOP. F9 pauses bot control and restores
manual input; F10 stops. Manual actions are excluded from sampled learning data.
Resuming ON restarts the chapter and retains learned parameters.

TOOLS can start saved settings in the background. Closing the monitor does not stop
training. Closing the main training app asks how to handle an active run. Keep the PC
awake. STOP retains the last completed update; an unfinished rollout may be lost.

## Common problems

| Problem | Action |
| --- | --- |
| Missing config.json | START → choose Celeste → Save settings |
| Missing launch.py | Extract the entire ZIP before opening START |
| Duplicate FlyBrain module | Stop old game windows; use launcher repair or LEARN's automatic repair |
| Missing C# source/bridge | Extract Source and prebuilt too; compiled bridge lives inside the nested mod ZIP |
| CUDA unavailable | TOOLS → Install GPU support; inspect GPU check; auto can retain CPU |
| Slow games | Hide windows, enable fast simulation, benchmark actual process counts; inspect performance.json |
| Search/restores are expensive | Search is bounded by teacher_seconds; reduce frequency in config when its measured time dominates |
| Fewer ready workers | Inspect MONITOR's memory/startup reason and worker logs |

OneDrive can add work to frequent file writes. Run the package outside synced folders.
Before running benchmark, evaluation or demonstration import, stop the learner.

## Reports

| File | Contents |
| --- | --- |
| training/latest.npz | Parameters, moments, counters, graph identity and RNG |
| training/live.json | Atomic worker PIDs, phase and heartbeat |
| training/status.json, progress.jsonl | Completed update evidence |
| training/performance.json | Actual controlled frames, wall time and phase timings |
| training/throughput.json | Display-paced/fast native benchmark and worker-count recommendation |
| training/accelerator.json, gpu-check.json | CPU/CUDA selection, checks and timings |
| training/curriculum.json | Room attempts, assisted clears and separate room tests |
| training/success-memory/ | Verified successful input routes |
| training/evaluation.json | Frozen full-story result and checkpoint hash |
| training/verification.html | Human-readable progress report |

Game logs and effective save/worker paths are recorded per worker. Share errors and
these reports for troubleshooting; do not send your game binaries or personal saves.
