"""Check public-file hashes and compiled mod identity without running Celeste."""

import hashlib
import json
import sys
import zipfile

from release_files import ROOT, digest, public_files, source_hashes

sys.path.insert(0, str(ROOT / "Python"))
from protocol import VERSION


def main():
    manifest = json.loads((ROOT / "CONTENTS.sha256.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Release manifest version differs from protocol")
    current = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in public_files()
        if path.name != "CONTENTS.sha256.json"
    }
    if current != manifest["files"]:
        changed = sorted(
            key
            for key in set(current) | set(manifest["files"])
            if current.get(key) != manifest["files"].get(key)
        )
        raise ValueError(
            "Release files changed; rebuild/package as documented: "
            + ", ".join(changed)
        )
    info = json.loads((ROOT / "prebuilt/BUILD_INFO.json").read_text())
    if info["version"] != VERSION or info["source_sha256"] != source_hashes():
        raise ValueError("Build source identity differs from this release")
    if info["graph_sha256"] != digest(ROOT / "Python/data/mcns_subnetwork.npz"):
        raise ValueError("Selected MCNS graph hash differs from BUILD_INFO")
    dependency = ROOT / "prebuilt/SpeedrunTool.zip"
    dependency_info = json.loads((ROOT / "prebuilt/SpeedrunTool-INFO.json").read_text())
    if digest(dependency) != dependency_info["sha256"]:
        raise ValueError("Bundled SpeedrunTool hash differs from its upstream record")
    with zipfile.ZipFile(dependency) as archive:
        if archive.testzip() is not None:
            raise ValueError("Bundled SpeedrunTool ZIP has a CRC error")
    with zipfile.ZipFile(ROOT / "prebuilt/FlyBrainCeleste.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Compiled mod ZIP has a CRC error")
        if f"Version: {VERSION}" not in archive.read("everest.yaml").decode("utf-8"):
            raise ValueError("Compiled mod manifest version differs from protocol")
        binary = archive.read("bin/FlyBrainCeleste.dll")
        if (
            binary[:2] != b"MZ"
            or hashlib.sha256(binary).hexdigest() != info["dll_sha256"]
        ):
            raise ValueError("Compiled mod DLL hash differs from BUILD_INFO")
        loose_dll = ROOT / "prebuilt/FlyBrainCeleste.dll"
        if not loose_dll.is_file() or loose_dll.read_bytes() != binary:
            raise ValueError(
                "The loose bridge DLL must match the DLL inside the mod ZIP"
            )
    print(
        f"Release {VERSION} verified: {len(current)} public files, graph, source hashes, compiled mod."
    )


if __name__ == "__main__":
    main()
