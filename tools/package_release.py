"""Package source and a compiled mod; never collect local games or saves."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

from release_files import ROOT, digest, public_files, source_hashes

sys.path.insert(0, str(ROOT / "Python"))
from plastic import PLASTIC_SCHEMA
from protocol import DURATIONS, PROTOCOL, VERSION
from learning import Reward


def package(dll=None, output=None, compiler=None):
    output = ROOT / "dist/FlyBrainCeleste-LEARN.zip" if output is None else Path(output)
    if output.resolve().is_relative_to(ROOT) and not output.resolve().is_relative_to(
        ROOT / "dist"
    ):
        raise ValueError(
            "Output inside the project must be under dist/ to protect source files"
        )
    sources = source_hashes()
    manifest = ROOT / "Source/everest.yaml"
    if f"Version: {VERSION}" not in manifest.read_text():
        raise ValueError("Python and Everest manifest versions differ")
    assembly = (ROOT / "Source/AssemblyInfo.cs").read_text()
    bridge = (ROOT / "Source/BridgeServer.cs").read_text()
    if (
        f'AssemblyFileVersion("{VERSION}.0")' not in assembly
        or f'version = "{VERSION}"' not in bridge
    ):
        raise ValueError(
            "Update the C# assembly and handshake version before packaging"
        )
    info_path = ROOT / "prebuilt/BUILD_INFO.json"
    previous = json.loads(info_path.read_text()) if info_path.exists() else {}
    mod = ROOT / "prebuilt/FlyBrainCeleste.zip"
    if dll is None:
        if (
            previous.get("version") != VERSION
            or previous.get("source_sha256") != sources
        ):
            raise ValueError(
                "C# source changed. Rebuild and pass --dll bin/FlyBrainCeleste.dll"
            )
        with zipfile.ZipFile(mod) as archive:
            binary = archive.read("bin/FlyBrainCeleste.dll")
        import hashlib

        if hashlib.sha256(binary).hexdigest() != previous["dll_sha256"]:
            raise ValueError("Existing compiled mod hash differs from BUILD_INFO")
    else:
        binary = Path(dll).read_bytes()
    if binary[:2] != b"MZ" or (VERSION + ".0").encode() not in binary:
        raise ValueError("Expected the newly compiled mod DLL for " + VERSION)
    mod.parent.mkdir(exist_ok=True)
    (mod.parent / "FlyBrainCeleste.dll").write_bytes(binary)
    with zipfile.ZipFile(mod, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("everest.yaml", manifest.read_bytes())
        archive.writestr("bin/FlyBrainCeleste.dll", binary)
    import hashlib

    validation_path = ROOT / f"docs/evidence/release-validation-{VERSION}.json"
    validation = (
        json.loads(validation_path.read_text()) if validation_path.exists() else {}
    )
    # Validation describes measured release evidence. Packaging is not a test run.
    info = {
        **validation,
        "version": VERSION,
        "protocol": PROTOCOL,
        "reward_revision": Reward.REVISION,
        "target": ".NET 8.0",
        "compiler": compiler
        or (
            previous.get("compiler")
            if dll is None
            else "Provided compiled DLL; see build output"
        ),
        "default_requested_real_game_processes": json.loads(
            (ROOT / "config.example.json").read_text()
        )["workers"],
        "learned_duration_frames": list(DURATIONS),
        "plastic_model": PLASTIC_SCHEMA,
        "dll_sha256": hashlib.sha256(binary).hexdigest(),
        "graph_sha256": digest(ROOT / "Python/data/mcns_subnetwork.npz"),
        "source_sha256": sources,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    # Match .gitattributes so checkout does not invalidate Windows-script hashes.
    for script in ROOT.glob("*.cmd"):
        text = script.read_bytes().replace(b"\r\n", b"\n")
        script.write_bytes(text.replace(b"\n", b"\r\n"))
    hashes = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in public_files()
        if path.name != "CONTENTS.sha256.json"
    }
    (ROOT / "CONTENTS.sha256.json").write_text(
        json.dumps({"version": VERSION, "files": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in public_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Packaged {VERSION}: {output} ({output.stat().st_size:,} bytes)")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dll", type=Path, help="Newly built mod; omit only if C# source is unchanged"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compiler", help="Compiler description from the actual build")
    args = parser.parse_args()
    package(args.dll, args.output, args.compiler)


if __name__ == "__main__":
    main()
