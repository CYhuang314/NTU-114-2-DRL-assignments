"""Environment wrappers for DRL HW4 Q2/Q3 (DMC humanoid-walk / humanoid-run).

The DM Control Suite — exposed to us via `dmc.make_dmc_env(...)` — yields
float64 observations and expects float64 actions. Our networks are float32
on GPU. To prevent silent float64 leakage from the env into the replay
buffer (which would defeat the buffer's float32 dtype contract and inflate
its memory footprint by 2x), we cast at the env <-> agent boundary here.

This module provides two idiomatic Gymnasium wrappers and a helper:

    Float32ObservationWrapper  : casts obs float64 -> float32 on reset/step
    Float32ActionWrapper       : casts action float32 -> float64 before step
    make_float32_env(env)      : applies both, in the conventional order

Wrap order is `Float32ActionWrapper(Float32ObservationWrapper(env))`:
the outer (action) wrapper is closest to the agent, inner (obs) wrapper
is closest to the env. The two transformations are independent.

Design notes
------------
* This wrapper does *only* dtype casting. Reward scaling, observation
  normalization, frame stacking, action repeat, etc. are deliberately
  out of scope and live in (future) separate wrappers.
* The wrapped `observation_space` and `action_space` report float32, so
  downstream code (ReplayBuffer asserts, sanity checks) sees a consistent
  view. Action bounds remain [-1, 1]; observation bounds remain
  (-inf, +inf) — float32(±inf) is exactly ±inf, no precision loss.
* `dmc.py` is read-only (TA-provided), so wrapping happens externally
  in `train.py`. `student_agent.py` does NOT use this wrapper — at eval
  time, the TA's eval scripts call `make_dmc_env(...)` directly and pass
  float64 observations to `Agent.act()`, which handles the cast internally.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Float32ObservationWrapper(gym.ObservationWrapper):
    """Cast observations from float64 to float32.

    Asserts (one-time, in __init__) that the wrapped env's observation
    space is a Box with float64 dtype — guards against silent regressions
    if the underlying env ever changes. Per-call asserts in `observation`
    verify the return contract (dtype + shape) at near-zero cost.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box), (
            f"Float32ObservationWrapper expects Box observation_space, "
            f"got {type(env.observation_space).__name__}"
        )
        assert env.observation_space.dtype == np.float64, (
            f"Float32ObservationWrapper expects float64 observation_space "
            f"(DMC default), got {env.observation_space.dtype}"
        )

        # Build a new float32 Box with the same (possibly infinite) bounds.
        # np.float32(±inf) == ±inf, so unbounded spaces survive the cast.
        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        # Cache for hot-path asserts.
        self._obs_shape = env.observation_space.shape

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        assert obs.dtype == np.float32, f"obs dtype {obs.dtype} != float32"
        assert obs.shape == self._obs_shape, (
            f"obs shape {obs.shape} != {self._obs_shape}"
        )
        return obs


class Float32ActionWrapper(gym.ActionWrapper):
    """Accept float32 actions from the agent; pass float64 to the env.

    The actor network produces float32 actions on GPU. MuJoCo / dm_control
    expects float64. This wrapper is the asymmetric counterpart to the
    observation wrapper: obs is cast *down* (f64 -> f32) on the way out;
    action is cast *up* (f32 -> f64) on the way in.

    Asserts (one-time, in __init__) that the wrapped env's action space
    is a Box with float64 dtype and bounds [-1, 1]. Per-call asserts in
    `action` verify the agent honored the float32 contract — silent
    float64 leakage from the agent into the buffer is exactly the failure
    mode this whole wrapper exists to prevent.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box), (
            f"Float32ActionWrapper expects Box action_space, "
            f"got {type(env.action_space).__name__}"
        )
        assert env.action_space.dtype == np.float64, (
            f"Float32ActionWrapper expects float64 action_space "
            f"(DMC default), got {env.action_space.dtype}"
        )
        assert np.all(env.action_space.low == -1.0) and np.all(
            env.action_space.high == 1.0
        ), (
            f"Float32ActionWrapper expects action bounds [-1, 1] "
            f"(DMC convention), got "
            f"low={env.action_space.low.min()}..{env.action_space.low.max()}, "
            f"high={env.action_space.high.min()}..{env.action_space.high.max()}"
        )

        # Re-expose the action space as float32 with bounds [-1, 1].
        # np.float32(±1.0) is exactly ±1.0 — no precision loss at the bounds.
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

        # Cache for hot-path asserts.
        self._action_shape = env.action_space.shape

    def action(self, action: np.ndarray) -> np.ndarray:
        # Verify what the agent handed us.
        assert isinstance(action, np.ndarray), (
            f"action must be ndarray, got {type(action).__name__}"
        )
        assert action.dtype == np.float32, (
            f"action dtype {action.dtype} != float32 — "
            f"silent float64 leakage from agent to buffer"
        )
        assert action.shape == self._action_shape, (
            f"action shape {action.shape} != {self._action_shape}"
        )
        # Cast up to float64 for the underlying MuJoCo env.
        return action.astype(np.float64)


def make_float32_env(env: gym.Env) -> gym.Env:
    """Apply both float32 wrappers in the conventional order.

    Outer wrapper (closest to agent) handles actions; inner wrapper
    (closest to env) handles observations. The two transformations are
    independent, so order is conventional rather than semantic.
    """
    env = Float32ObservationWrapper(env)
    env = Float32ActionWrapper(env)
    return env
