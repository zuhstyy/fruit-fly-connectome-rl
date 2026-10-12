# Compiled files

I keep the compiled bridge and its installable mod package together here.

| File | Purpose |
| --- | --- |
| [FlyBrainCeleste.dll](FlyBrainCeleste.dll) | Directly accessible compiled .NET 8 bridge |
| [FlyBrainCeleste.zip](FlyBrainCeleste.zip) | Everest mod with `everest.yaml` and the same DLL at `bin/FlyBrainCeleste.dll` |
| [BUILD_INFO.json](BUILD_INFO.json) | Software version, compiler record, and DLL/source/graph SHA256 identity |
| [SpeedrunTool.zip](SpeedrunTool.zip) | Native practice-state dependency installed in worker copies |
| [SpeedrunTool-INFO.json](SpeedrunTool-INFO.json) | Upstream version, download source, and dependency digest |
| [SpeedrunTool-LICENSE.txt](SpeedrunTool-LICENSE.txt) | Preserved upstream MIT license |

I use `FlyBrainCeleste.zip` for worker installation because Everest needs the
manifest as well as the assembly. The loose DLL is an identical copy for
inspection and development; placing it alone in Mods is not a complete install.

I regenerate both DLL copies from the same bytes when I run
`python tools/package_release.py`. `python tools/check_release.py` verifies
that the loose DLL, the nested DLL, and BUILD_INFO all agree.

The outer project archive also contains Python, launchers, source, and
documentation. Extract that archive into a project folder and use START.cmd.
See [setup](../docs/SETUP.md) and [development](../docs/DEVELOPMENT.md).
