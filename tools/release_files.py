"""Choose public project files without collecting local games or training."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FOLDERS = {"Python", "Source", "Tests", "tools", "docs", ".github", "prebuilt"}
PRIVATE_FOLDERS = {
    "__pycache__",
    ".git",
    ".venv",
    ".venv-learn",
    "bin",
    "obj",
    "training",
    "workers",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}
PRIVATE_FILENAMES = {
    "config.json",
    "config.local.json",
    ".DS_Store",
    "Thumbs.db",
    "PAUSE",
    "STOP",
    "FNA.dll",
    "MMHOOK_Celeste.dll",
    "neurons.csv.gz",
    "connections_princeton.csv.gz",
}
PRIVATE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".bak",
    ".user",
    ".suo",
    ".celeste",
    ".pem",
    ".key",
}
PUBLIC_ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    "CITATION.cff",
    "config.example.json",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".clang-format",
    "app.py",
    "launch.py",
    "CONTENTS.sha256.json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_files(root=ROOT):
    files = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in PRIVATE_FOLDERS for part in relative.parts):
            continue
        if (
            path.suffix.lower() in PRIVATE_SUFFIXES
            or path.name in PRIVATE_FILENAMES
            or path.name == ".env"
            or path.name.startswith(".env.")
            or (
                path.name.startswith("Celeste")
                and path.suffix.lower() in {".dll", ".exe"}
            )
        ):
            continue
        if len(relative.parts) == 1:
            allowed = path.name in PUBLIC_ROOT_FILES or path.suffix == ".cmd"
        else:
            allowed = relative.parts[0] in PUBLIC_FOLDERS
        if allowed:
            files.append(path)
    return files


def source_hashes(root=ROOT):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in public_files(root)
        if path.relative_to(root).parts[0] == "Source"
    }
