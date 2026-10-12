"""Guard against local settings and game files leaking into public archives."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from release_files import public_files
import package_release


class ReleaseFilesTests(unittest.TestCase):
    def selected(self, names):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test fixture", encoding="utf-8")
            return {p.relative_to(root).as_posix() for p in public_files(root)}

    def test_private_files_inside_public_folders_are_excluded(self):
        names = [
            "docs/.env",
            "Python/.env.local",
            "docs/config.json",
            "tools/config.local.json",
            "docs/run.log",
            "Source/test.pem",
            "Python/secret.key",
            "docs/save.celeste",
            "docs/.DS_Store",
            "Source/FNA.dll",
            "Source/Celeste.dll",
            "docs/neurons.csv.gz",
            "Python/connections_princeton.csv.gz",
            "Python/training/latest.npz",
            "docs/workers/game.txt",
            "docs/.vscode/settings.json",
            "Python/__pycache__/test.pyc",
        ]
        self.assertEqual(self.selected(names), set())

    def test_required_code_graph_and_nested_mods_remain_public(self):
        names = [
            "README.md",
            "SECURITY.md",
            "config.example.json",
            "START.cmd",
            "Python/data/mcns_subnetwork.npz",
            "Source/BridgeServer.cs",
            "prebuilt/FlyBrainCeleste.zip",
            "prebuilt/FlyBrainCeleste.dll",
            "prebuilt/SpeedrunTool.zip",
            "prebuilt/SpeedrunTool-LICENSE.txt",
            ".github/workflows/checks.yml",
            "docs/evidence/check.txt",
        ]
        self.assertEqual(self.selected(names), set(names))

    def test_local_root_and_unknown_folders_are_excluded(self):
        names = [
            "config.json",
            ".env",
            "training/latest.npz",
            "workers/worker-01/Celeste.dll",
            "dist/release.zip",
            "data-original/neurons.csv.gz",
            "private/notes.md",
            "notes.tmp",
        ]
        self.assertEqual(self.selected(names), set())

    def test_archive_output_cannot_overwrite_public_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / "README.md"
            protected.write_text("keep this source", encoding="utf-8")
            with patch.object(package_release, "ROOT", root):
                for output in [
                    protected,
                    root / "docs/release.zip",
                    root / "prebuilt/mod.zip",
                ]:
                    with self.subTest(output=output):
                        with self.assertRaisesRegex(ValueError, "under dist/"):
                            package_release.package(output=output)
            self.assertEqual(protected.read_text(), "keep this source")
            self.assertEqual(list(root.iterdir()), [protected])


if __name__ == "__main__":
    unittest.main()
