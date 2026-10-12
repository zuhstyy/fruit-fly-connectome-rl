"""Find FlyBrain by manifest identity, including ZIPs renamed by Windows.

Repairs move whole packages outside Mods. No saves or unrelated mods are edited.
"""

from pathlib import Path
from datetime import datetime
import json
import re
import shutil
import zipfile

LIMIT = 256 * 1024
MANIFESTS = ("everest.yaml", "everest.yml")


def manifest_names(text):
    # Everest's manifest is a sequence of module dictionaries. Match Name keys
    # at the dictionary's top level, not Name keys inside Dependencies.
    text = text.lstrip("\ufeff")
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, list):
        return [d.get("Name") for d in data if isinstance(d, dict)]
    lines = text.splitlines()
    entries = [re.match(r"^( *)-\s+\S", line) for line in lines]
    indent = min((len(m[1]) for m in entries if m), default=None)
    if indent is None:
        return []
    names = []
    for line in lines:
        m = re.match(r"^( *)(-\s+)?Name\s*:\s*([^#]+)", line)
        if not m:
            continue
        depth = len(m[1])
        if (m[2] and depth == indent) or (not m[2] and depth == indent + 2):
            names.append(m[3].strip().strip("\"'"))
    return names


def module_names(path):
    path = Path(path)
    if path.is_symlink():
        return []
    if path.is_dir():
        for name in MANIFESTS:
            manifest = path / name
            if (
                manifest.is_file()
                and not manifest.is_symlink()
                and manifest.stat().st_size <= LIMIT
            ):
                return manifest_names(manifest.read_text(encoding="utf-8-sig"))
    elif path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                # Everest uses the first root manifest when both extensions exist.
                for item in archive.infolist():
                    if item.filename in MANIFESTS and item.file_size <= LIMIT:
                        return manifest_names(archive.read(item).decode("utf-8-sig"))
        except (zipfile.BadZipFile, UnicodeError):
            pass
    return []


def find_flybrain(game):
    mods = Path(game) / "Mods"
    if not mods.is_dir():
        return []
    return [p for p in sorted(mods.iterdir()) if "FlyBrainCeleste" in module_names(p)]


def move_to_backup(paths, backup_root, log=print):
    paths = list(paths)
    if not paths:
        return None
    backup = Path(backup_root) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.move(str(path), str(backup / path.name))
        log(f"Disabled old package: {path.name}")
    log(f"Packages preserved in: {backup}")
    return backup


def repair_original(game, log=print):
    game = Path(game)
    paths = find_flybrain(game)
    backup = move_to_backup(paths, game / "FlyBrain-disabled", log)
    if not paths:
        log("No FlyBrain packages in the normal game Mods folder.")
    return backup
