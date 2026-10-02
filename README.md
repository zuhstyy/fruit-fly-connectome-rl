# Fruit Fly Connectome RL Project

## Project Goal

Build a reinforcement learning agent inspired by a real fruit fly connectome and see if I can eventually get it to perform basic platforming tasks.

The long term idea is to use real fruit fly connectome data as part of the agent instead of just using a normal neural network.

Right now I am starting extremely simple and making sure I understand the basic reinforcement learning loop before I touch the actual fly brain stuff.

## Current Tools

- Python 3.11
- VS Code
- Pygame
- NumPy
- MCNS v1.0 connectome data
- FlyWire Codex

## Dataset

I am using the MCNS v1.0 dataset.

MCNS stands for Male Central Nervous System and contains the adult male fruit fly brain + nerve cord.

There is currently no neural network data being used at all in this project.

I want to get the basic learning system working first before I start messing with millions of neural connections.

[FlyWire Codex - MCNS](https://codex.flywire.ai/?dataset=mcns)

## Progress Log

### Day 1 - Basic Environment

#### Project Setup

- Created the basic project folder structure.
- Created a Python virtual environment using `venv`.
- Installed Pygame and NumPy.
- Added the MCNS neuron and connection datasets.
- Made a basic Pygame testing environment.

#### First Simulation

The environment is currently extremely simple.

There is:

- a yellow circle representing the fly
- a green circle representing the goal
- a gameplay window
- basic window boundaries
- a 60 FPS game loop

The fly originally used keyboard controls just so I could make sure the movement worked.

After that I removed the keyboard controls and gave it 3 possible actions:

- `0` = move left
- `1` = do nothing
- `2` = move right

Right now the actions are picked randomly.

The fly holds an action for 15 frames before choosing another one, mostly so I can actually see what it is doing instead of it changing direction every frame and bugging out.

#### Reward System

I added a very basic reward system.

If the fly moves closer to the goal:

`+0.1 reward`

If it moves farther away:

`-0.1 reward`

If it does nothing:

`0 reward`

If the fly reaches the green goal:

`+10 reward`

When it reaches the goal, the terminal prints the total episode reward and the fly gets reset back to the starting position.

Example:

```text
GOAL REACHED
Episode reward: 29.2

### The fly is not currently learning anything and is purely random. This is a template used as a foundation to plug things into later.
```
