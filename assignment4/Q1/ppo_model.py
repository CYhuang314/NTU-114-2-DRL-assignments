"""
PPO model definitions for Pendulum-v1.

Shared between training (train_ppo.py) and inference (student_agent.py).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ----- State preprocessing -----
# Pendulum obs: [cos(theta), sin(theta), theta_dot]
# theta_dot in [-8, 8]; the others are already in [-1, 1].
# Normalize theta_dot to [-1, 1] for stable NN inputs.
THETA_DOT_SCALE = 8.0
ACTION_SCALE = 2.0  # action space is [-2, 2]
OBS_DIM = 3
ACT_DIM = 1


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """obs: shape (..., 3). Normalize theta_dot by 8."""
    obs = np.asarray(obs, dtype=np.float32)
    out = obs.copy()
    out[..., 2] = out[..., 2] / THETA_DOT_SCALE
    return out


def _orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs. State-independent log-std for action.

    Actor outputs mean mu(s); action ~ Normal(mu, exp(log_std)).
    For env interaction, action is clipped to [-2, 2].
    For PPO ratio, log-prob is computed on the raw (un-clipped) sample.

    The effective log_std is clamped to [log_std_min, log_std_max] so that the
    trainer can ANNEAL EXPLORATION by progressively lowering log_std_max.
    This addresses the failure mode where log_std plateaus at a high value
    (large eval-time variance even though deterministic mean is good).
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 64, init_log_std: float = 0.0,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        # Actor body (outputs mean)
        self.actor = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, hidden), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, act_dim), gain=0.01),
        )
        # State-independent log-std (per-action-dim learnable parameter).
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        # Bounds (buffers so they save with state_dict).
        self.register_buffer("log_std_min", torch.tensor(float(log_std_min)))
        self.register_buffer("log_std_max", torch.tensor(float(log_std_max)))

        # Critic
        self.critic = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, hidden), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, 1), gain=1.0),
        )

    def set_log_std_max(self, value: float) -> None:
        """Lower the upper bound on log_std (for scheduled exploration decay)."""
        self.log_std_max.fill_(float(value))

    def effective_log_std(self) -> torch.Tensor:
        """log_std after clamping to [min, max]."""
        return self.log_std.clamp(min=self.log_std_min.item(),
                                  max=self.log_std_max.item())

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.effective_log_std().exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """Used during rollout collection and PPO update.

        Returns:
            action: sampled (raw, un-clipped) action of shape (..., act_dim)
            log_prob: summed over action dims, shape (...)
            entropy: summed over action dims, shape (...)
            value: shape (...)
        """
        dist = self.get_dist(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """For evaluation: return the policy mean (no exploration)."""
        return self.actor(obs)
