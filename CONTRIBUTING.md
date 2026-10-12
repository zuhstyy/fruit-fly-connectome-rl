# Contributing

I welcome bug reports, documentation fixes, and focused code changes. Start
with [development](docs/DEVELOPMENT.md) and [learning](docs/LEARNING.md).

## Report an issue

Use the bug template for startup, controls, training, or monitoring problems.
Include reproduction steps, expected and actual behavior, project/Python/Everest
versions, requested and ready worker counts, and a relevant log excerpt.
For a feature request, explain the problem and how success could be measured.
Follow [SECURITY.md](SECURITY.md) for vulnerability reports.

## Propose a change

Keep changes focused and names clear. Explain the user-visible result and
preserve checkpoints and protocol compatibility where possible. Update the
corresponding documentation and changelog. Maintain the established folder
layout because launchers, tests, and packaging resolve paths from the root.

Run checks from [DEVELOPMENT.md](docs/DEVELOPMENT.md). Learning changes need
gradient and reset-boundary checks. Protocol/input changes need bridge/button
checks. Gameplay claims need real-game evidence with starting conditions,
release, attempts, successes, and assistance settings.

Before committing public-file edits, refresh the release manifest:

```sh
python tools/package_release.py
python tools/check_repository.py
python tools/check_release.py
```

C# source edits require rebuilding the mod first. The packaging command checks
source identity and refuses to reuse a stale compiled DLL. See the build command
in the development guide. Hash verification in CI expects a packaged snapshot.

## Review and attribution

Use the pull request template to explain the problem, resulting behavior,
checks, and compatibility. Separate synthetic tests from actual Celeste results.
Credit upstream code and data, preserve their licenses, and keep credentials,
personal saves, copyrighted game files, and local settings out of attachments.
