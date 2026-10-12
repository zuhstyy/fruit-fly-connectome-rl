"""Run the C# harnesses using your own Everest references and .NET 8 SDK."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCES = {
    "HoldHarness": ["BridgeServer", "ButtonHistory"],
    "RoomRouteHarness": ["RoomRoute"],
    "BridgeControlHarness": ["BridgeServer", "WorkerControl"],
    "PracticeAdapterHarness": [
        "BridgeServer",
        "ButtonHistory",
        "PracticeStates",
        "GameAccess",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeste", type=Path, required=True)
    args = parser.parse_args()
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise SystemExit("Install the .NET 8 SDK for developer checks.")
    output = subprocess.check_output([dotnet, "--list-sdks"], text=True)
    sdks = [
        match.groups() for match in re.finditer(r"^(8\.\S+) \[(.+)\]$", output, re.M)
    ]
    if not sdks:
        raise SystemExit(
            "These harnesses use the .NET 8 SDK and runtime. Ordinary LEARN does not need an SDK."
        )
    version, sdk_folder = sdks[-1]
    dotnet_root = Path(sdk_folder).parent
    packs = sorted(
        (dotnet_root / "packs/Microsoft.NETCore.App.Ref").glob("8.*/ref/net8.0")
    )
    if not packs:
        raise SystemExit("Missing .NET 8 reference pack.")
    game = args.celeste.resolve()
    game_refs = [
        game / name for name in ("Celeste.dll", "FNA.dll", "MMHOOK_Celeste.dll")
    ]
    if not all(path.is_file() for path in game_refs):
        raise SystemExit(
            "Choose your Everest folder containing the three game/reference DLLs."
        )
    compiler = Path(sdk_folder) / version / "Roslyn/bincore/csc.dll"
    destination = ROOT / "build/csharp-checks"
    destination.mkdir(parents=True, exist_ok=True)
    for name, sources in HARNESS_SOURCES.items():
        dll = destination / (name + ".dll")
        lines = [
            "-nostdlib+",
            "-target:exe",
            "-langversion:12",
            "-nullable:enable",
            "-optimize+",
            "-out:" + str(dll),
        ]
        lines += ["-r:" + str(path) for path in [*packs[-1].glob("*.dll"), *game_refs]]
        lines += [str(ROOT / "Source/GlobalUsings.cs")]
        lines += [str(ROOT / "Source" / (source + ".cs")) for source in sources]
        lines += [str(ROOT / "Tests" / (name + ".cs"))]
        response = destination / (name + ".rsp")
        response.write_text(
            "\n".join('"' + line + '"' for line in lines) + "\n", encoding="utf-8"
        )
        subprocess.run([dotnet, str(compiler), "@" + str(response)], check=True)
        dll.with_suffix(".runtimeconfig.json").write_text(
            json.dumps(
                {
                    "runtimeOptions": {
                        "tfm": "net8.0",
                        "framework": {
                            "name": "Microsoft.NETCore.App",
                            "version": "8.0.0",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        arguments = []
        if name == "BridgeControlHarness":
            arguments = [str(destination / "controls")]
        if name == "PracticeAdapterHarness":
            arguments = [str(ROOT / "prebuilt/SpeedrunTool.zip")]
        subprocess.run([dotnet, str(dll), *arguments], check=True)
    print(
        "C# harnesses passed. These are adapter/protocol/timing checks, not Celeste playtests."
    )


if __name__ == "__main__":
    main()
