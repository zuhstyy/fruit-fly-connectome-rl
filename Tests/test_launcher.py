import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from workers import prepare_copy
from mod_install import find_flybrain, repair_original, manifest_names
from controls import Controls

MANIFEST = "- Name: FlyBrainCeleste\n  Version: 3.0.0\n  DLL: bin/FlyBrainCeleste.dll\n"


class LauncherTests(unittest.TestCase):
    def test_game_copy_has_independent_files_and_saves(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            game = root / "game"
            game.mkdir()
            (game / "Celeste.dll").write_bytes(b"original-game")
            (game / "Celeste.exe").write_bytes(b"exe")
            (game / "Saves").mkdir()
            (game / "Saves/0.celeste").write_bytes(b"personal-save")
            (game / "Mods").mkdir()
            (game / "Mods/old-mod.zip").write_bytes(b"old")
            (game / "everest-env.txt").write_text("OTHER_CONFIG=1")
            (game / "nextLaunchIsVanilla.txt").touch()
            mod = root / "FlyBrainCeleste.zip"
            with zipfile.ZipFile(mod, "w") as z:
                z.writestr("everest.yaml", MANIFEST)
            save = prepare_copy(game, root / "worker", mod, lambda m: None)
            self.assertEqual(list(save.iterdir()), [])
            self.assertFalse((root / "worker/Mods/old-mod.zip").exists())
            self.assertFalse((root / "worker/everest-env.txt").exists())
            self.assertEqual(
                (root / "worker/steam_appid.txt").read_text().strip(), "504230"
            )
            self.assertFalse((game / "steam_appid.txt").exists())
            self.assertFalse((root / "worker/nextLaunchIsVanilla.txt").exists())
            self.assertTrue((game / "nextLaunchIsVanilla.txt").exists())
            (root / "worker/Celeste.dll").write_bytes(b"worker-change")
            self.assertEqual((game / "Celeste.dll").read_bytes(), b"original-game")
            self.assertEqual((game / "Saves/0.celeste").read_bytes(), b"personal-save")

    def test_unowned_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            game = root / "game"
            game.mkdir()
            (game / "Celeste.dll").touch()
            dest = root / "worker"
            dest.mkdir()
            (dest / "keep.txt").write_text("keep")
            with self.assertRaises(RuntimeError):
                prepare_copy(game, dest, root / "none")
            self.assertEqual((dest / "keep.txt").read_text(), "keep")

    def test_changing_only_executable_refreshes_owned_worker(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            game = root / "game"
            game.mkdir()
            (game / "Celeste.dll").write_bytes(b"unchanged")
            (game / "Celeste.exe").write_bytes(b"old")
            mod = root / "mod.zip"
            with zipfile.ZipFile(mod, "w") as z:
                z.writestr("everest.yaml", MANIFEST)
            dest = root / "worker"
            prepare_copy(game, dest, mod, lambda m: None)
            (game / "Celeste.exe").write_bytes(b"new executable")
            prepare_copy(game, dest, mod, lambda m: None)
            self.assertEqual((dest / "Celeste.exe").read_bytes(), b"new executable")

    def test_duplicate_repair_finds_renamed_zip_and_folder_preserves_other_mods(self):
        with tempfile.TemporaryDirectory() as t:
            game = Path(t)
            mods = game / "Mods"
            mods.mkdir()
            for name in ("FlyBrainCeleste.zip", "Download (2).zip"):
                with zipfile.ZipFile(mods / name, "w") as z:
                    z.writestr("everest.yaml", MANIFEST)
            (mods / "extracted").mkdir()
            (mods / "extracted/everest.yml").write_text(MANIFEST)
            unrelated = mods / "helper.zip"
            with zipfile.ZipFile(unrelated, "w") as z:
                z.writestr(
                    "everest.yaml",
                    "- Name: AnotherMod\n  Dependencies:\n    - Name: FlyBrainCeleste\n      Version: 1.0\n",
                )
            keep = unrelated.read_bytes()
            (game / "Saves").mkdir()
            (game / "Saves/0.celeste").write_text("personal-save")
            self.assertEqual(len(find_flybrain(game)), 3)
            backup = repair_original(game, lambda m: None)
            self.assertEqual(len(list(backup.iterdir())), 3)
            self.assertEqual(find_flybrain(game), [])
            self.assertEqual(unrelated.read_bytes(), keep)
            self.assertEqual((game / "Saves/0.celeste").read_text(), "personal-save")
            self.assertIsNone(repair_original(game, lambda m: None))

    def test_reused_worker_cannot_retain_duplicate_copy(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            game = root / "game"
            game.mkdir()
            (game / "Celeste.dll").write_bytes(b"game")
            mod = root / "mod.zip"
            with zipfile.ZipFile(mod, "w") as z:
                z.writestr("everest.yaml", MANIFEST)
            dest = root / "worker"
            saves = prepare_copy(game, dest, mod, lambda m: None)
            (dest / "Mods/old.zip").write_bytes(mod.read_bytes())
            (dest / "Mods/loose").mkdir()
            (dest / "Mods/loose/everest.yaml").write_text(MANIFEST)
            (saves / "0.celeste").write_bytes(b"training-save")
            prepare_copy(game, dest, mod, lambda m: None)
            self.assertEqual(find_flybrain(dest), [dest / "Mods/FlyBrainCeleste.zip"])
            self.assertEqual((saves / "0.celeste").read_bytes(), b"training-save")
            self.assertEqual(saves, dest / "training-data/Saves")

    def test_manifest_name_can_follow_version_and_dependency_is_not_a_module(self):
        text = '- Version: 3.0.0\n  Name: "FlyBrainCeleste"\n  Dependencies:\n    - Name: EverestCore\n      Version: 1.0\n'
        self.assertEqual(manifest_names(text), ["FlyBrainCeleste"])

    def test_shared_controls_allow_manual_pause_and_clear_previous_stop(self):
        with tempfile.TemporaryDirectory() as t:
            gui = Controls(t)
            worker = Controls(t)
            gui.new_run()
            self.assertFalse(worker.paused)
            gui.pause()
            self.assertTrue(worker.paused)
            worker.resume()
            self.assertFalse(gui.paused)
            worker.stop()
            self.assertTrue(gui.paused)
            self.assertTrue(gui.stop_file.exists())
            gui.new_run()
            self.assertFalse(worker.paused)
            self.assertFalse(worker.stop_file.exists())


if __name__ == "__main__":
    unittest.main()
