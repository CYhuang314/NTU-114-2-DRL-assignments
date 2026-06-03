"""Student agent for Pendulum-v1: deterministic PPO policy at inference time.

Loads weights from disk (saved by train_ppo.py).
The constructor takes no arguments other than self.

Loading order:
  1. If $PPO_CKPT env var is set, use that path.
  2. Otherwise try ppo_pendulum_final.pt (final weights).
  3. Otherwise try ppo_pendulum.pt (best by rolling training return).
  4. If neither exists, the agent runs with random init weights.

The "final" checkpoint is preferred because the policy continues to sharpen
(log_std anneal) past the point where rolling TRAINING return peaks; the
deterministic mean continues to improve even when training return plateaus.
"""
import os
import gymnasium as gym
import numpy as np
import torch

from ppo_model import (
    ActorCritic, preprocess_obs,
    OBS_DIM, ACT_DIM, ACTION_SCALE,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_ckpt_path():
    env_path = os.environ.get("PPO_CKPT")
    if env_path and os.path.exists(env_path):
        return env_path
    for name in ("ppo_pendulum.pt", "ppo_pendulum_final.pt"):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            return p
    return None


class Agent(object):
    """PPO-trained agent for Pendulum-v1. Deterministic at inference."""

    def __init__(self):
        # Required by the spec — used for type info; we do not sample from it.
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        # CPU at inference for portability.
        self.device = torch.device("cpu")
        self.model = ActorCritic(OBS_DIM, ACT_DIM, hidden=64).to(self.device)

        ckpt_path = _resolve_ckpt_path()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if (
                isinstance(ckpt, dict) and "model_state_dict" in ckpt
            ) else ckpt
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def act(self, observation):
        """Deterministic action: policy mean, clipped to [-2, 2]."""
        obs_n = preprocess_obs(np.asarray(observation, dtype=np.float32))
        with torch.no_grad():
            t = torch.as_tensor(obs_n, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            mean = self.model.act_deterministic(t).cpu().numpy()[0]
        return np.clip(mean, -ACTION_SCALE, ACTION_SCALE).astype(np.float32)
