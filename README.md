# FlyBrain Celeste

**A connectome-inspired reinforcement learning experiment for Celeste.**

I built FlyBrain Celeste to explore whether real fruit-fly connection data could
support a learning controller for a difficult platformer. I connect Python to
the real game through a C# Everest mod and use a selected **256-neuron,
3,326-edge Male CNS (MCNS v1.0) subgraph** to learn movement, button presses,
and input duration with PPO and imitation of verified successful routes.

I started with a small Pygame movement-and-reward experiment and expanded it
into an instrumented game-learning system, with development assistance from Codex.

## Project status

My current software release is **4.1.0**. I am working toward an unattended clear
of the main story, from Prologue through Summit. **I have not demonstrated a
full-game clear.** I use offline checks to establish model learning, checkpoint
persistence, and protocol behavior; native gameplay and throughput still require
Windows testing.

I preserve the selected fly connections while training their strengths. I use
engineered sensor mappings and neuron dynamics for this experiment; I am not
simulating an entire biological fly brain.

## Features

- **Trainable fly wiring:** fixed MCNS topology and edge signs, with bounded
  learned connection gains and actor/value layers.
- **Frame-based control:** 156 observation values, six control choices, and
  learned holds of 1, 2, 4, 8, 12, 20, or 30 game frames.
- **Parallel game workers:** isolated game copies, training saves, process
  identity checks, and an authenticated localhost TCP bridge.
- **Persistent learning:** model weights, optimizer state, RNG, successful
  routes, and progress reports survive restarts.
- **Practice and evaluation:** native save states and bounded search during
  training; separate frozen-policy evaluation with assistance disabled.
- **Performance tools:** hidden rendering, fast simulation, worker benchmarks,
  and optional CUDA gradients with measured CPU/GPU backend selection.

## Quick start

| Component | Requirement |
| --- | --- |
| Game training | Windows and your own Celeste installation with Everest |
| Python | 3.11 or 3.12, including Tkinter for the desktop launcher |
| Runtime dependency | NumPy; installed automatically on first launch |
| Optional GPU learning | NVIDIA GPU/driver and the PyTorch installer in TOOLS |
| Source builds | .NET 8 SDK and your own Everest game DLLs |

I include the compiled bridge and SpeedrunTool dependency. Training does
not require a source build. Internet access is needed for first-time dependency
installation and optional GPU setup.

1. Extract the **complete project archive** into a local folder outside Celeste
   and OneDrive, such as `C:\FlyBrain`.
2. Open **START.cmd** and select the Everest folder containing `Celeste.exe`.
3. Start with **2 games**, **Hide windows**, **Fast simulation**, and **auto** backend.
4. Press **Save settings**, then **LEARN**.
5. Open **MONITOR.cmd** to inspect workers or show one game. Use **STOP.cmd**
   to request shutdown and retain the last completed learning update.

To resume existing learning, stop the old run and copy its `training/` folder
into this project root. See [setup and upgrades](docs/SETUP.md).

**Do not place the outer project archive in Celeste's Mods folder.** The actual
mod is `prebuilt/FlyBrainCeleste.zip`, containing `bin/FlyBrainCeleste.dll`.
I also include an identical, directly accessible DLL at
[`prebuilt/FlyBrainCeleste.dll`](prebuilt/FlyBrainCeleste.dll). The launcher uses
the nested mod ZIP for installation; the loose DLL is available for inspection
or development. See the [compiled-file guide](prebuilt/README.md).

| Entry point | Purpose |
| --- | --- |
| `START.cmd` | Configure the project, train, evaluate, and toggle bot control |
| `MONITOR.cmd` | Inspect worker status, timing, inputs, and window visibility |
| `STOP.cmd` | Request a clean stop |
| `TOOLS.cmd` | Check the model, verify progress, benchmark, install GPU support, and rehearse routes |

F9 pauses bot control for manual play; F10 stops the run. Resuming control
restarts the chapter with learned weights retained.

## How it works

I use the Everest bridge to measure game state and apply frame-counted inputs
on the game thread. In Python, I turn observations into activity in the MCNS
recurrent network, then sample buttons and hold duration. I use real outcomes
as rewards and update the policy, value estimate, and connection gains with PPO
after collection. I train successful search/practice routes through a separate
imitation objective.

I keep Celeste physics on the **CPU** and use optional CUDA support for batched
model gradients. My `auto` backend checks gradient agreement and selects CUDA
only when its probe is at least 10% faster. A small model may favor the CPU.
See [architecture](docs/ARCHITECTURE.md), [learning](docs/LEARNING.md), and
[performance](docs/PERFORMANCE.md) for implementation details.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `Source/` | C# Everest bridge, observations, controls, and practice adapter |
| `Python/` | Model, trainer, workers, monitoring, benchmarks, and evaluation |
| `Python/data/` | Selected MCNS graph, provenance metadata, and data license |
| `Tests/` | Offline Python tests and C# harnesses |
| `tools/` | Developer checks and release packaging |
| `prebuilt/` | Compiled mod, bundled dependency, build identity, and upstream license |
| `docs/` | Setup, design, research, results, and publishing guidance |
| `.github/` | Automated checks and issue/pull request templates |

I exclude local settings, checkpoints, game copies, saves, logs, and environments
from Git and release archives. I keep folder names stable for launcher and
checkpoint compatibility.

## Development and validation

From the project root, in a Python environment with dependencies installed:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s Tests -v
python Python/check_brain.py --fresh
python -m black --check .
python tools/check_repository.py
python tools/check_release.py
```

See [development](docs/DEVELOPMENT.md) for virtual environments, C# builds,
and packaging. [Validation](docs/VALIDATION.md) separates historical release
evidence, current checks, and results still needing real hardware.

## Documentation and contribution

Browse the [documentation index](docs/README.md), [roadmap](docs/ROADMAP.md),
or [GitHub publishing guide](docs/PUBLISHING.md). Bug reports and proposed
changes are welcome; read [CONTRIBUTING.md](CONTRIBUTING.md). Security reports
are covered in [SECURITY.md](SECURITY.md).

## License and credits

I release my original code and documentation under the [MIT license](LICENSE). The included
MCNS-derived graph has [separate CC BY 4.0 attribution terms](Python/data/DATA_LICENSE.md).
Bundled SpeedrunTool retains its [MIT license](prebuilt/SpeedrunTool-LICENSE.txt).
Celeste game assets and binaries are not distributed. See [credits](docs/CREDITS.md)
for dataset, research, and software sources, and [CITATION.cff](CITATION.cff)
for project citation metadata.
