"""Optional CUDA gradients for the SAME fly wiring and checkpoint format.

Game inference stays on the CPU: transferring one tiny action at a time to
the GPU costs more than it saves. CUDA batches independent recurrent sequences.
"""

import time
import numpy as np
from protocol import HEADS, SENSOR_COUNT


class TorchGradients:
    def __init__(self, policy, device="cuda", precision="float32"):
        import torch

        self.torch, self.policy = torch, policy
        self.device = torch.device(device)
        self.dtype = getattr(torch, precision)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check the NVIDIA driver and GPU install."
            )
        self.inputs = self.tensor(policy.inputs)
        self.base = self.tensor(policy.base)
        self.indices = self.tensor(
            policy.target * policy.nodes + policy.source, torch.long
        )

    def tensor(self, value, dtype=None):
        return self.torch.as_tensor(
            value, dtype=dtype or self.dtype, device=self.device
        )

    def gradients(self, segments, entropy=0.02, objective="ppo"):
        torch, policy = self.torch, self.policy
        count, length = len(segments), max(len(s["observations"]) for s in segments)
        observations = np.zeros((count, length, SENSOR_COUNT))
        resets, valid = np.ones((count, length), bool), np.zeros((count, length), bool)
        for i, segment in enumerate(segments):
            n = len(segment["actions"])
            observations[i, :n] = policy.pad_observations(segment["observations"])
            resets[i, :n], valid[i, :n] = segment["resets"], True
        observations, resets, valid = (
            self.tensor(observations),
            self.tensor(resets, torch.bool),
            self.tensor(valid, torch.bool),
        )
        parameters = [
            self.tensor(p).detach().clone().requires_grad_() for p in policy.parameters
        ]
        actor, value, theta = parameters
        weights = self.base * theta.exp()
        matrix = torch.zeros(
            policy.nodes * policy.nodes, device=self.device, dtype=self.dtype
        )
        matrix = matrix.index_add(0, self.indices, weights).reshape(
            policy.nodes, policy.nodes
        )
        h = self.tensor(np.asarray([s["initial"] for s in segments]))
        drive = observations @ self.inputs.T
        activities = []
        for t in range(length):
            h = h * (~resets[:, t]).unsqueeze(1)
            for _ in range(3):
                h = 0.65 * h + 0.35 * torch.tanh(h @ matrix.T + drive[:, t])
            activities.append(h)
        activity = torch.stack(activities, dim=1)[valid]
        features = torch.cat([activity, torch.ones_like(activity[:, :1])], dim=1)

        def joined(key, dtype=None):
            return self.tensor(np.concatenate([s[key] for s in segments]), dtype)

        actions = joined("actions", torch.long)
        logits = features @ actor
        if objective == "ppo":
            logits = logits / joined("temperatures").reshape(-1, 1)
        logp = torch.zeros(len(actions), device=self.device, dtype=self.dtype)
        ent = torch.zeros_like(logp)
        offset = 0
        for head, choices in enumerate(HEADS):
            logs = torch.log_softmax(logits[:, offset : offset + choices], dim=1)
            logp = logp + logs.gather(1, actions[:, head : head + 1]).squeeze(1)
            ent = ent - (logs.exp() * logs).sum(1)
            offset += choices
        if objective == "imitation":
            loss = -logp.mean() / len(HEADS)
            # Keep the zero value gradient compatible with NumPy Adam state.
            loss = loss + value.sum() * 0
            kl = torch.zeros((), device=self.device)
        else:
            old, advantages, returns = (
                joined("old_logp"),
                joined("advantages"),
                joined("returns"),
            )
            ratio = (logp - old).clamp(-20, 20).exp()
            loss = -torch.minimum(
                ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
            ).mean()
            loss = (
                loss
                - entropy * ent.mean()
                + 0.5 * ((features @ value - returns) ** 2).mean()
            )
            kl = (old - logp).mean()
        loss.backward()
        # The small actor/edge gradients return once per minibatch. Checkpoint
        # arrays and Adam remain NumPy, so CPU and GPU saves are interchangeable.
        gradients = [p.grad.detach().cpu().double().numpy() for p in parameters]
        return (
            float(loss.detach()),
            gradients,
            float(kl.detach()),
            float(ent.mean().detach()),
        )


def probe_segments(policy, count=8, length=32):
    """A fixed gradient workload. It does not play or teach Celeste."""
    rng = np.random.default_rng(410017)
    return [
        dict(
            observations=rng.uniform(-1, 1, (length, SENSOR_COUNT)),
            resets=np.arange(length) == 0,
            initial=np.zeros(policy.nodes),
            actions=np.column_stack([rng.integers(n, size=length) for n in HEADS]),
            old_logp=np.full(length, -9.0),
            advantages=rng.normal(size=length),
            returns=rng.normal(size=length),
            temperatures=np.ones(length),
        )
        for _ in range(count)
    ]


def choose_backend(policy, mode="auto", count=8, log=print):
    """Auto uses CUDA only after parity and timings show a worthwhile win."""
    report = dict(
        requested=mode,
        selected="numpy",
        cuda_available=False,
        graph_sha256=policy.graph_hash,
    )
    if mode == "cpu":
        report["reason"] = "CPU requested"
        return report
    try:
        import torch

        report.update(
            torch_version=torch.__version__, cuda_available=torch.cuda.is_available()
        )
        if not report["cuda_available"]:
            raise RuntimeError(
                "No working CUDA device; use TOOLS > Install GPU support if needed"
            )
        report["gpu"] = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = TorchGradients(policy)
        sample = probe_segments(policy, count=count)
        reference = policy.numpy_sequence_gradients(sample)
        candidate = backend.gradients(sample)
        for expected, actual in zip(reference[1], candidate[1]):
            if not np.allclose(expected, actual, rtol=0.004, atol=2e-5):
                raise RuntimeError("CUDA gradient parity check failed; CPU retained")
        # Include transfer, Python and synchronization overhead in both timings.
        timings = {}
        for name, function in [
            ("numpy", policy.numpy_sequence_gradients),
            ("cuda", backend.gradients),
        ]:
            times = []
            for _ in range(3):
                started = time.perf_counter()
                function(sample)
                times.append(time.perf_counter() - started)
            timings[name] = float(np.median(times))
        report.update(seconds=timings, sequences=count, precision="float32")
        if mode == "cuda" or timings["cuda"] < 0.9 * timings["numpy"]:
            policy.gradient_backend = backend
            report.update(
                selected="cuda",
                reason=(
                    "CUDA requested"
                    if mode == "cuda"
                    else "CUDA measured at least 10% faster"
                ),
            )
        else:
            report["reason"] = "CPU was faster for this small workload"
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        report["reason"] = str(error)
        if mode == "cuda":
            raise RuntimeError(
                "GPU requested but unavailable: " + str(error)
            ) from error
    log("Learner backend: " + report["selected"] + ". " + report["reason"])
    return report
