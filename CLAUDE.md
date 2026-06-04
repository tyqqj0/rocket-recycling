# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rocket-recycling simulation using reinforcement learning. A 2D rigid-body rocket learns to hover or land (belly-flop maneuver) via an Actor-Critic (A2C) policy gradient method. Originally by Zhengxia Zou.

## Commands

```bash
# Install dependencies
pip install numpy torch torchvision opencv-python matplotlib

# Train an agent (edit `task` variable inside for 'hover' or 'landing')
python example_train.py

# Run inference with a trained checkpoint
python example_inference.py
```

Checkpoints save to `{task}_ckpt/` every 100 episodes. Training renders every 100 episodes via OpenCV window.

## Architecture

Four source files, no package structure:

- **`rocket.py`** — `Rocket` class: the environment. Implements physics simulation (rigid body + air resistance + thrust-vectoring nozzle), reward calculation, state transitions, and OpenCV rendering. Follows a gym-like API (`reset()`, `step(action)`, `render()`). State is an 8-dim vector (x, y, vx, vy, theta, vtheta, t, phi) divided by 100 for normalization. Action space is 9 discrete actions (3 thrust levels × 3 nozzle angular velocities).

- **`policy.py`** — `ActorCritic` class: the RL agent. Uses a shared `MLP` backbone with positional encoding (NeRF-style sinusoidal mapping, L=7) feeding into separate actor (policy) and critic (value) heads. Optimizer is RMSprop at 5e-5. `get_action()` returns action, log_prob, value. `update_ac()` performs the A2C update.

- **`utils.py`** — Geometry helpers (polygon creation, 4×4 transformation matrices) and image utilities (background loading, moving average for reward plots).

- **`example_train.py` / `example_inference.py`** — Entry points. Set `task = 'hover'` or `task = 'landing'` to switch tasks.

## Key Design Details

- The positional mapping in the MLP (`PositionalMapping`) maps each input dimension to `2L+1` features using sin/cos at increasing frequencies — this is critical for learning fine-grained control.
- Discount factor gamma=0.999 (high, because episodes can be long — up to 800 steps).
- The environment normalizes state by dividing by 100 in `flatten()`.
- Rocket types 'falcon' and 'starship' only affect rendering polygons, not physics.
- Background images (`hover.jpg`, `landing.jpg`) are required for rendering.
