"""Shared controls: the GUI and each worker can release bot control immediately."""

from pathlib import Path


class Controls:
    def __init__(self, root):
        self.root = Path(root)
        self.pause_file = self.root / "PAUSE"
        self.stop_file = self.root / "STOP"

    @property
    def paused(self):
        return self.pause_file.exists()

    def pause(self):
        self.pause_file.touch()

    def resume(self):
        self.pause_file.unlink(missing_ok=True)

    def stop(self):
        self.pause()
        self.stop_file.touch()

    def new_run(self):
        self.stop_file.unlink(missing_ok=True)
        self.resume()
