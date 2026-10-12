# Sources and credits

## Connectome data

Male CNS v1.0 comes from FlyEM at HHMI Janelia, the University of Cambridge,
the MRC Laboratory of Molecular Biology, Google Research, and collaborators.
I used the MCNS CSV exports in FlyWire Codex, the connectome explorer
developed at Princeton Neuroscience Institute.

- [Male CNS project](https://male-cns.janelia.org/)
- [Male CNS downloads and CC BY license](https://male-cns.janelia.org/download/)
- [FlyWire Codex, MCNS](https://codex.flywire.ai/?dataset=mcns)
- [Data adaptations and license](../Python/data/DATA_LICENSE.md)

## Learning methods

- Schulman, Wolski, Dhariwal, Radford, and Klimov. *Proximal Policy Optimization
  Algorithms* (2017). [Original paper](https://arxiv.org/abs/1707.06347).
- Schulman, Moritz, Levine, Jordan, and Abbeel. *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation* (2015).
  [Original paper](https://arxiv.org/abs/1506.02438).
- Oh, Guo, Singh, and Lee. *Self-Imitation Learning* (2018).
  [Original paper](https://arxiv.org/abs/1806.05635).
- Ecoffet and collaborators. *Go-Explore: a New Approach for Hard-Exploration
  Problems* (2019). [Original paper](https://arxiv.org/abs/1901.10995).
- Ng, Harada, and Russell. *Policy Invariance Under Reward Transformations*
  (1999). [Original paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

I use a NumPy recurrent PPO controller. I chose the sensor projection,
edge-gain bounds, rate dynamics, action durations, and Celeste reward shaping
for this experiment; they are not claims about the original papers or fly biology.
I drew inspiration from this research for successful-route behavior cloning and
entry/frontier practice, without reproducing the full SIL or Go-Explore algorithms.

## Software

- [Celeste](https://www.celestegame.com/): the game; not distributed here.
- [Everest](https://everestapi.github.io/): Celeste's mod loader and hooks.
- [Everest source](https://github.com/EverestAPI/Everest): upstream lifecycle and
  API reference used for integration.
- [NumPy](https://numpy.org/): model calculations and checkpoint arrays.
- [Published Celeste player controller](https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs):
  primary movement/state reference; no game controller source is bundled.
- [SpeedrunTool](https://github.com/DemoJameson/Celeste.SpeedrunTool): native
  save/load dependency, bundled version **3.27.21** under MIT. Its license is in
  `prebuilt/SpeedrunTool-LICENSE.txt`; download, version, and digest are recorded
  in `prebuilt/SpeedrunTool-INFO.json`. Credit its upstream maintainers. This
  dependency is separate from FlyBrain's original code and installed only in
  owned workers.
- Python/Tkinter: local launcher and monitor.
- Microsoft .NET and Win32 documentation: compilation, process identity, memory,
  and owned-window controls.

## My contributions

I chose the project goal and dataset, started with the Pygame experiment, and
provided Windows gameplay feedback that shaped the implementation. I developed
the project with assistance from Codex. I credit the upstream teams for the
connectome data, game, mod loader, and practice-state machinery. I am not
affiliated with the game or dataset creators and do not claim their endorsement.
