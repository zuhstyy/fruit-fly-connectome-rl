"""Check required publishing files and local Markdown links without network access."""

import json
import re
from urllib.parse import unquote, urlsplit

from release_files import ROOT, public_files

REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".gitignore",
    ".github/workflows/checks.yml",
    "config.example.json",
    "Source/BridgeServer.cs",
    "Source/everest.yaml",
    "Python/data/DATA_LICENSE.md",
    "Python/data/mcns_subnetwork.npz",
    "prebuilt/FlyBrainCeleste.zip",
    "prebuilt/FlyBrainCeleste.dll",
    "prebuilt/README.md",
    "prebuilt/BUILD_INFO.json",
    "prebuilt/SpeedrunTool.zip",
    "prebuilt/SpeedrunTool-LICENSE.txt",
    "docs/README.md",
    "docs/PUBLISHING.md",
    "docs/VALIDATION.md",
}


def check(root=ROOT):
    files = public_files(root)
    names = {p.relative_to(root).as_posix() for p in files}
    errors = [f"Missing public file: {name}" for name in sorted(REQUIRED - names)]
    launchers = {p.name for p in files if p.parent == root and p.suffix == ".cmd"}
    if launchers != {"START.cmd", "STOP.cmd", "MONITOR.cmd", "TOOLS.cmd"}:
        errors.append("Expected exactly START, STOP, MONITOR, and TOOLS launchers")
    example = root / "config.example.json"
    if example.exists() and json.loads(example.read_text())["game"] != "":
        errors.append("config.example.json must not contain a personal game path")
    documents = [p for p in files if p.suffix == ".md"]
    links = 0
    for path in documents:
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.split(' "', 1)[0].strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            links += 1
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                errors.append(
                    f"Broken local link: {path.relative_to(root)} -> {target}"
                )
    if errors:
        raise ValueError("\n".join(errors))
    print(
        f"Repository checks passed: {len(documents)} Markdown files, "
        f"{links} local links, required publishing files, and example settings."
    )


if __name__ == "__main__":
    check()
