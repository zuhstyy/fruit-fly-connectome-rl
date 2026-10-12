"""Time gradients on a copy, without launching games or changing a saved model."""

from pathlib import Path
import argparse
import json
from accelerator import choose_backend
from plastic import PlasticPolicy
from runtime import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def run(root=ROOT, sequences=8):
    root = Path(root)
    policy = PlasticPolicy()
    checkpoint = root / "training/latest.npz"
    if checkpoint.exists():
        policy.load(checkpoint, policy.graph_hash)
    report = choose_backend(policy, "auto", count=sequences)
    report.update(native_game_benchmarked=False, saved_weights_modified=False)
    atomic_json(root / "training/gpu-check.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(
        "CUDA accelerates model gradients. Celeste physics remains on the CPU.",
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, choices=range(1, 65), default=8)
    run(sequences=parser.parse_args().sequences)
