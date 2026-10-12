# Changes

I record changes here while keeping software behavior separate from documentation
and packaging revisions.

## 2026-10-12 — first-person documentation and visible bridge DLL

- Rewrite my project history, overview, portfolio, and design introductions in first person.
- Use the project identity in citation/copyright metadata instead of a personal name.
- Include `prebuilt/FlyBrainCeleste.dll` alongside the installable mod ZIP.
- Add a compiled-file guide and verify the loose DLL against the nested DLL and BUILD_INFO.
- Preserve the bridge bytes, C# source, version 4.1.0, MCNS graph, and checkpoint formats.

## 2026-10-12 — GitHub publication cleanup

- Rewrite the README around project purpose, status, features, setup, and review.
- Add a documentation index, publishing guide, roadmap, and security reporting policy.
- Refresh contribution and issue/pull request templates, including a feature request template.
- Add offline Windows CI coverage, manual workflow dispatch, job timeouts, and local-link checks.
- Exclude local settings, environment files, keys, logs, caches, and game DLLs from public archives.
- Check the bundled SpeedrunTool digest and CRC during release verification.
- Normalize text checkout line endings across Windows/Linux while retaining CRLF for CMD scripts.
- Restrict in-project archive output to dist/ to prevent source overwrite or archive recursion.
- Retain software/bridge version 4.1.0, C# source, compiled bridge, graph, and checkpoint formats.

## 4.1.0 — simulation speed and cleanup

- Rebuild the compiled bridge with protocol 6; retain fixed 60 Hz native physics in an uncapped outer loop.
- Suppress FNA draw/presentation for hidden workers, throttle visible drawing and idle/control polling.
- Reuse room topology by revision; reject stale/missing ready-state geometry.
- Add optional PyTorch CUDA gradients for the same graph/objectives. Auto checks agreement and measures speed before selection.
- Preserve existing parameters, optimizer moments, RNG and saved-model format.
- Bound search time, reduce its default frequency and throttle live telemetry.
- Add controlled-frame/phase metrics and actual display-paced versus fast benchmarks.
- Consolidate launchers into START, MONITOR, STOP and TOOLS; add Save settings.
- Move documentation into docs and remove redundant launchers and stale release dumps.
- Add GPU/local/remote guidance, a seven-day experiment and a video essay plan.
- Native game speed, full-story completion, overnight stability and CUDA hardware remain unverified here.

## 4.0.0 — practice, discovery, and persistent successful-route learning

- Added a compiled native-practice adapter and bundled attributed SpeedrunTool
  3.27.21. Save/restore guards preserve input edges, exclude the HUD, reconstruct
  neural activity with current weights, and disable practice on mismatches.
- Added bounded automatic real-game movement search with separate trial/action
  counters. Successful routes persist and train the fly actor/gains through a
  separate imitation objective; teacher data never becomes fabricated PPO data.
- Expanded observations from 92 to 156 with player states, held buttons, nearby
  object motion, wind, inventory, and static navigation distance. Increased
  recurrent gradient sequences from 16 to 32 decisions.
- Added failure-aware portal selection, key/carry hints, earlier-stage rehearsal,
  and discount-aware shaping. Necessary backtracking is permitted; live room
  loops cannot repeatedly award their clear bonus.
- Added automatic frozen-policy room tests and independent campaign evaluation
  with checkpoint hashes. Added optional verified-demo import and native
  throughput benchmark commands.
- Added objective/velocity/practice/search HUD fields, drawing suppression for
  hidden workers, and separate learner/assisted progress reports.
- Preserved older learned arrays and optimizer state with migration backups.
  Updated setup, research, portfolio, code guide, licensing, and release evidence.
- No native full-game clear, 64-game capacity, or overnight speedup is claimed.

## 3.2.0 — source and portfolio release

- Expanded compressed Python/C# formatting and explained the central learning
  steps with clearer names and comments. Retained the working action/reward
  logic, selected graph, and 3.1 saved-model format.
- Added CHECK_BRAIN: wiring ablation, disposable recurrent PPO update, and exact
  save/reload checks. A source-data option verifies original neuron and edge
  counts; the measured release report is included.
- Added short checkpoint-save retries for a brief Windows reader lock.
- Added beginner setup, learning, code guide, project history, and portfolio
  notes with accurate assistance and dataset credits.
- Added a public example configuration, ignored local settings, runtime/dev
  dependencies, GitHub checks, contribution templates, and release scripts.
- Default settings request eight real games. The explicit 64-game command is
  retained. No room-clear or game-throughput improvement is claimed for this
  source/documentation release.

## 3.1.0 — timing and actual worker processes

- Added learned frame durations, elapsed-time rewards/GAE, and held-button
  history across requests.
- Replaced exploration streams with separate game processes and added live
  monitoring plus explicit Show/Hide controls.
- Batched recurrent sequence calculations and preserved 3.0 model state when
  adding the duration output.

## 3.0.0 — trainable fly connections and forward progress

- Added recurrent edge-gain learning alongside actor/value learning.
- Corrected signed numeric Prologue room order and forward-target feedback.
- Removed timer/stall forced deaths and improved recorded evidence reporting.
