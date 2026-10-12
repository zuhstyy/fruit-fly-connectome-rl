# Release 4.1.0 validation

I keep this record to separate automated project checks from actual gameplay
results. I preserve earlier evidence with its original dates and test scope.

## First-person documentation and DLL packaging — October 12, 2026

I include the compiled bridge at `prebuilt/FlyBrainCeleste.dll` as well as
`bin/FlyBrainCeleste.dll` inside the installable mod ZIP. I verify that both
copies are byte-identical and match the DLL hash in BUILD_INFO. I retain the
same compiled assembly, C# source, graph, protocol, and checkpoint formats.

The latest automated suite discovered 111 tests: 107 passed and 4 optional
tensor/CUDA tests skipped in 33.756 seconds on Linux. I checked required public
files and local documentation links, Python formatting, release hashes, nested
ZIP CRCs, citation metadata, and the archive's public text/compiled files for
personal-name references. These checks do not establish a native gameplay result.

Latest log: [documentation and DLL packaging checks](evidence/python-tests-first-person-2026-10-12.txt).

## Earlier publication cleanup checks — October 12, 2026

I kept the 4.1.0 C# source, compiled bridge, MCNS graph, and runtime
model/checkpoint formats unchanged during GitHub preparation. Checks ran on Linux
with Python 3.12.14 and NumPy 2.3.5:

| Check | Result |
| --- | --- |
| Python suite | 111 discovered, 107 passed, 4 optional tensor/CUDA tests skipped; 44.568 seconds |
| Fresh fly-model check | Passed; all 3,326 gains changed on a disposable model and exact save/reload passed |
| Python formatting | All 50 Python files passed Black 25.1.0 |
| Repository check | Required publishing files, example settings, and 61 local links across 23 Markdown files passed |
| Source/metadata syntax | Python AST, JSON, and YAML parsed successfully |
| Package checks | Public-file hashes, source/graph identity, nested ZIP CRCs, and dependency digest passed |
| Git checkout round trip | All public-file hashes retained with core.autocrlf=true; CMD files retained CRLF |

The four new regression tests cover private files in public directories,
retention of required graph/mod assets, exclusion of local root files, and
protection against archive output overwriting source. The Windows CI job is
configured but has not run here. Optional PyTorch is absent in this environment;
no new tensor/CUDA execution or native Celeste result is claimed.

Earlier log: [publication Python checks](evidence/python-tests-publication-2026-10-12.txt).
The release-build and gameplay limitations below remain applicable; retained
earlier logs are historical evidence.

## Retained release-build evidence

The compiled .NET 8 mod was rebuilt with Roslyn 4.11 / SDK 8.0.419 against genuine
stripped Celeste, FNA and MMHOOK references, with **zero errors and warnings**.
Game/reference DLLs are not included. This build environment has no playable
Windows Celeste and no CUDA GPU. Native speed and campaign results are unverified.

## Checks actually run

| Check | Result | Scope |
| --- | --- | --- |
| Python suite with optional PyTorch CPU | 107 discovered, 106 passed, 1 CUDA hardware test skipped | 11.173 seconds; model/protocol/runtime tests |
| NumPy-only Python suite | 107 discovered, 103 passed, 4 optional tensor tests skipped | 9.771 seconds |
| C# held inputs | 363 assertions passed | 120 synthetic input frames over four decisions |
| C# navigation | 11 checks passed | Synthetic room geometry |
| C# socket/control | 9 checks plus 200 correlated requests passed | Actual localhost TCP, authentication, command queue/event and shared controls |
| C# native-practice adapter | 22 checks passed | Real adapter with fake state backend; actual bundled API metadata |
| Model influence and save/reload | Passed | Fresh model; disposable learning and exact state persistence |
| 4.0 checkpoint compatibility | Exact parameter/moment match | Synthetic checkpoint produced using actual previous-release code |
| Offline rehearsal | Separate candidate learns; source unchanged; zero fresh frames | Synthetic successful route, not a new Celeste room clear |

Logs: [Python with tensors](evidence/python-tests-4.1.txt),
[NumPy-only](evidence/python-tests-numpy-4.1.txt), [C#](evidence/csharp-checks-4.1.txt).
The full suite run preceded the rehearsal cleanup `finally` adjustment; its two
focused tests were rerun afterward and both passed. C# final compilation includes
the hidden-particle update hook. No hosted GitHub CI run is claimed.

The 64-slot integration check uses **synthetic worker clients**, collects 2,048
samples, performs an actual PPO update, saves and shuts down. It verifies the
coordinator; it does not show 64 native game processes running.

## Bridge and model measurements

The 200-request raw C# socket test measured **0.0604 ms median / 0.1841 ms p95** on
this Linux build host. This excludes Celeste, FNA scheduling, rendering, physics,
world observations and learning. It is not a Windows game-throughput number.

The old NumPy model's 256-decision synthetic PPO update (one epoch) measured a
median **0.0327065 seconds** across five runs on this host. It justified prioritizing
native sampling/presentation over model-size assumptions. It is not a measured
native or GPU speedup. [Baseline workload record](evidence/model-baseline-4.0.json).

[Tensor parity](evidence/tensor-parity-4.1.json) compares the same recurrent model
and both learning objectives on PyTorch 2.5.1+cpu:

- Float64 PPO maximum gradient difference: approximately 7.8e-16.
- Float32 PPO maximum gradient difference: approximately 6.5e-7.
- Float32 imitation maximum gradient difference: approximately 4.6e-8.

Tests also cover ragged/padded sequences, reset boundaries, 92-sensor legacy
padding, exploration temperatures, real-edge changes and exact NumPy checkpoint
reload after a tensor update. PyTorch CUDA uses these same code paths, but **actual
CUDA execution and the pinned 2.13.0 Windows installation were not run here**.
Auto parity/timing selection and a hardware-gated test provide runtime checks.

## Fly wiring and memory

[The fresh brain report](evidence/brain-check-4.1.json) shows removing the recurrent
edges changes action probabilities by up to **0.01068530** on its fixed probe.
A disposable PPO update changes all 3,326 edge strengths and actor/value arrays,
preserving signs and stable gains. Parameters, Adam moments, counters, RNG and
scores reload exactly. This establishes computational influence, not superiority
to random wiring, biological fidelity, or native room mastery.

[4.0-to-4.1 compatibility](evidence/migration-4.1.json) preserves actor, value,
edge gains and all six Adam arrays with maximum error **zero**, retaining optimizer
steps and checkpoint source bytes. No migration reset is required. Earlier
[migration evidence](evidence/migration-4.0.json) is retained for 3.1/3.2 models.

The unchanged graph SHA256 is
`a06e9a6f48422b9bb47ca9ed258cb3c038095fa2bdc1ad48c467ebdaff7f3edc`.
[Historical provenance](evidence/brain-check-3.2.json) independently matched all
256 IDs and 3,326 connection counts against original MCNS exports over 9,039,608
connection rows. The fresh probe does not rescan those originals.

[The historical conditional imitation test](evidence/conditional-learning-4.0.json)
shows different sensor inputs learning opposite directions and different hold
lengths. It uses accelerated synthetic learning, not normal-rate Celeste sampling.
Its probabilities are not gameplay success rates.

## What needs real hardware evidence

The fixed outer-loop removal, FNA SuppressDraw path, revision caching and bounded
search are implemented and compiled. Hidden workers also skip visual particle
aging while retaining emission and its random draws, following the specific
CelesteTAS optimization rationale. **Native pacing/physics fidelity, render
appearance and performance must be checked on Windows.** I retain entity/collision
updates and keep C# game physics on the CPU.

Not established: native speedup multiplier, RTX 4060 Ti CUDA throughput, exact
full hidden-state restore fidelity, 64 real games, long unattended stability,
every puzzle, reliable independent room clears or full Prologue-to-Summit completion.
Runtime restore signatures are useful guards but do not inspect all hidden state.

The benchmark compares actual display-paced and fast workers and records real PIDs,
controlled frames and response latency. Frozen evaluation records a model hash,
actual chapter events and interruptions, with teacher/practice disabled. Those
tools enable evidence collection; their presence is not a successful result.

## Package identity

The rebuilt DLL, source, graph and public files are recorded in BUILD_INFO and
CONTENTS.sha256.json. ZIP CRC and hashes are checked during release verification.
The package excludes private checkpoints/settings, game/worker copies, environments,
old release dumps, reference DLLs and development downloads. Four CMD entry files
replace the former twelve. Source, tests, license credits and CI remain for review.

SpeedrunTool 3.27.21 retains its independently checked xxHash64
`13d8948e88482580` and SHA256
`c5cf070f4f69969e9166f95fca375a0a97b306e7f6c19fe22fa0928710ac07e4`.
See [development](DEVELOPMENT.md), [research](RESEARCH.md) and
[performance](PERFORMANCE.md) for reproducible commands and interpretation.
