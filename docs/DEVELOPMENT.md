# Development

I keep my code-reading order, offline checks, build commands, and packaging
process here so the project can be reviewed and reproduced.

## Read the code in this order

| File | Start here |
| --- | --- |
| `Python/protocol.py` | `sensors` and `action_fields`: observations in, buttons out |
| `Python/plastic.py` | `PlasticPolicy.forward`: neuron activity travels along fly edges |
| `Python/learning.py` | `Reward.step` and `gae`: measure outcomes and assign credit |
| `Python/train.py` | `Runner.roll`, then `_run`: collect, update, and save |
| `Python/navigation.py` | Stable objectives and static route fields |
| `Python/search_teacher.py` | Real-game branch search, separate from the learner |
| `Python/experience.py`, `curriculum.py` | Saved routes, attempts, and independent test counts |
| `Source/PracticeStates.cs` | Native practice adapter and held-button restoration |
| `Source/ButtonHistory.cs` | Held, pressed, and released buttons |
| `Source/BotInputs.cs` | Apply a simulated input frame and restore real bindings |
| `Source/FlyBrainModule.cs` | Game lifecycle, observations, targets, and HUD |
| `Python/check_brain.py` | Verify wiring influence, training, and saved memory |
| `Python/workers.py` | Isolated copies, saves, startup checks, and recovery |

I keep the outer loop in ordinary Python. I use vectorized NumPy, or optional
CUDA tensor gradients, in the recurrent optimizer because batching is part of
the algorithm I am testing.
Read [LEARNING.md](LEARNING.md) alongside it. Short names such as `dh` in the
backward pass mean derivatives of neuron activity, not unexplained game state.
The old `Reservoir` and `FlyBrain.choose` examples are compatibility/history
code. The active entry point constructs `PlasticPolicy` and `PlasticState`.

## Offline checks

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m unittest discover -s Tests -v
.venv\Scripts\python Python/check_brain.py --fresh
.venv\Scripts\python -m black --check .
.venv\Scripts\python tools/check_repository.py
.venv\Scripts\python tools/check_release.py
.venv\Scripts\python tools/run_csharp_checks.py --celeste "C:/Games/Celeste"
```

On Linux/macOS use `.venv/bin/python`. Offline Python checks work without
Celeste. Playing/training and the provided launcher target Windows. Set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for repeatable small-model timing.
The Windows launcher sets BLAS thread defaults itself.
The final C# command requires a .NET 8 SDK/runtime and your own Everest DLLs.
It exercises button history, geometry, the real socket bridge, the practice
adapter against a fake dependency, and actual bundled dependency metadata.
It does not execute Celeste physics. Build outputs go under ignored `build`.

The GitHub Actions workflow runs offline Python checks on Ubuntu with 3.11 and
3.12, and Windows with 3.12. It
does not run Celeste or rebuild C# against game DLLs. A future green CI run means
those jobs passed on that commit; it is not a room-clear certificate.

## Optional tensor checks

Install PyTorch from its official CPU/CUDA wheel index into your development
environment, then rerun the suite. The tensor tests exercise the actual tensor
implementation on CPU when CUDA is absent; the hardware-specific CUDA test
skips explicitly. GitHub CI includes a separate CPU tensor job.

## Rebuild the mod

I supply the compiled mod in `prebuilt/FlyBrainCeleste.zip` and an identical
loose assembly at `prebuilt/FlyBrainCeleste.dll`. Developers need a .NET SDK capable
of targeting .NET 8 and their own Everest game references:

```powershell
dotnet build Source/FlyBrainCeleste.csproj -c Release -p:CelestePath="C:/Games/Celeste"
python tools/package_release.py --dll bin/FlyBrainCeleste.dll
python tools/check_release.py
```

`CelestePath` must be your actual Everest folder. Game/reference DLLs stay out
of the repository. Packaging updates `prebuilt/FlyBrainCeleste.dll` and
`prebuilt/FlyBrainCeleste.zip` from the same compiled bytes,
refreshes source/graph/file hashes, and writes `dist/FlyBrainCeleste-LEARN.zip`.
If C# has not changed, `python tools/package_release.py` can reuse the existing
compiled mod after checking its recorded source hashes. Changes to C# require
a new build. Run the relevant checks and update validation evidence for changed
behavior before making a release.

Black formats Python. `.editorconfig` and `.clang-format` describe consistent
spacing for C#. `check_repository.py` checks required public files, example
settings, and local Markdown links without contacting external sites.

## Reproduce the data selection

The original exports are optional and are not included. Obtain MCNS v1.0
`neurons.csv.gz` and `connections_princeton.csv.gz` through the credited Codex
dataset. Use a separate `data-original` folder:

```powershell
python Python/celeste_connectome.py --data data-original --output build/mcns-check.npz --size 256 --min-synapses 5
python Python/check_brain.py --fresh --source-data data-original
```

Selection scans the large connection table three times. The original export
hashes in `Python/data/mcns_subnetwork.json` identify the exact input snapshot.
Future upstream downloads can differ. Do not overwrite the graph used by an
existing checkpoint: graph identity is part of its compatibility check.
The optional source check verifies each selected ID, class, neurotransmitter,
edge count, and normalized initial weight against those original exports.

## Publish the source

See [PUBLISHING.md](PUBLISHING.md) for repository details, initial Git commands,
GitHub Desktop, and release preparation. The root README and documentation
index provide entry points for reviewers.
