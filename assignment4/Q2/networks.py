"""
networks.py

Actor and Critic architectures for SAC on DMC humanoid (state-based).

- Actor: squashed Gaussian policy with tanh action bound, optionally
  preceded by an ObsNorm input normalizer.
- TwinCritic: two Q-networks bundled for SAC's clipped double-Q,
  optionally with LayerNorm in hidden layers, and optionally sharing
  the actor's ObsNorm.
- ObsNorm: per-dimension running mean/std (Welford), updated only on
  rollout (single observations from env interaction), never on training
  batches (which would reflect the buffer's stale memory distribution
  rather than current state visitation).

Backward compatibility
----------------------
With `use_obs_norm=False` and `use_layer_norm=False` (both defaults),
the architecture is byte-identical to the previous version:
  - same module names, same parameter names, same buffer set (none)
  - `state_dict()` is identical, so old `best_actor.pt` files load
    cleanly with `strict=True`
  - `act_deterministic` / `sample` produce numerically identical output
This was verified against the prior networks.py before this rewrite.

Pure architecture; no RL update logic, no environment coupling.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical bounds on log std of the pre-squash Gaussian.
# - Lower bound -10 -> std ~= 4.5e-5, allows highly confident (near-deterministic)
#   per-joint actions for postural control without inviting log-prob blowup.
# - Upper bound 2  -> std ~= 7.4, broad but bounded; prevents pathological
#   early-training variance.
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

# Init range for final-layer weights/biases of policy heads and critic output.
# Keeps initial action means near 0 (uniform-random post-tanh) and initial
# Q-values near 0, avoiding biased starts.
FINAL_INIT_RANGE = 3e-3


def _init_hidden_(layer: nn.Linear) -> None:
    """Fan-in uniform init for hidden Linear layers (DDPG/TD3/SAC convention)."""
    fan_in = layer.weight.size(1)
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _init_final_(layer: nn.Linear, init_range: float = FINAL_INIT_RANGE) -> None:
    """Small uniform init for final output layers."""
    nn.init.uniform_(layer.weight, -init_range, init_range)
    nn.init.uniform_(layer.bias, -init_range, init_range)


# ===========================================================================
# ObsNorm: per-dimension running mean/std (Welford), rollout-only updates
# ===========================================================================
class ObsNorm(nn.Module):
    """Per-dimension running mean/std normalizer with strict update separation.

    State (all `register_buffer` so they save/load with state_dict):
        mean:  (obs_dim,) float32  -- running mean
        m2:    (obs_dim,) float32  -- sum of squared deviations (Welford M2)
        count: ()         float64  -- # observations seen (float64 to avoid
                                      precision loss past ~1e7 updates)

    Variance is computed on the fly as `var = m2 / (count - 1)` (Bessel-
    corrected sample variance). Storing M2 rather than var keeps the
    Welford recurrence numerically stable across millions of updates.

    Critical separation:
        forward(x)   -- normalizes x using current (mean, var). NEVER updates.
                        Called from every actor/critic forward pass, including
                        SAC update batches at training time and eval rollouts.
        update(x)    -- updates (mean, m2, count) with one rollout observation.
                        Called from train.py's rollout loop after every env
                        step. NEVER called from the SAC update path.

    Why the separation matters: updating on sampled training batches would
    inflate `count` by batch_size x per env step (256x at our defaults) and
    track the replay buffer's stale-memory distribution rather than the
    current state-visitation distribution. New states discovered late in
    training (e.g., the moment the policy learns to walk) would arrive
    with weight ~1/(count*256), effectively invisible to the normalizer.
    """

    def __init__(self, obs_dim: int, eps: float = 1e-8):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("m2", torch.zeros(obs_dim, dtype=torch.float32))
        # float64 count: 5M-step training fits comfortably in float32 (mantissa
        # holds integers up to 2^24 ~= 1.6e7), but float64 is free and robust
        # to long fine-tuning runs and any future vectorization.
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x. Pure read; never modifies state. Safe to call in
        any context: SAC update, eval rollout, target-critic computation."""
        # First observation (or no observations yet): identity. Matches
        # CleanRL/SB3 RunningMeanStd boot behavior. The very-first network
        # forward pass thus sees raw obs, which is fine because:
        #   - at warmup time the policy is random anyway
        #   - by SAC update #1 (step 10001 with our defaults), count >> 1
        if self.count.item() < 2:
            return x
        var = self.m2 / (self.count.to(torch.float32) - 1.0)
        return (x - self.mean) / torch.sqrt(var + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Welford update with a single rollout observation.

        Accepts shape (obs_dim,) or (1, obs_dim). The single-obs constraint
        is intentional and asserted: this method must NOT be called with
        sampled training batches (see class docstring for reasoning).
        """
        if x.dim() == 2:
            assert x.size(0) == 1, (
                f"ObsNorm.update expects single rollout observation, "
                f"got batch of size {x.size(0)}. Updating on sampled "
                f"training batches violates the IID assumption -- use "
                f"forward() (read-only) inside the SAC update path."
            )
            x = x.squeeze(0)
        assert x.dim() == 1 and x.size(0) == self.obs_dim, (
            f"ObsNorm.update expected shape ({self.obs_dim},), got {tuple(x.shape)}"
        )
        x = x.to(self.mean.device, dtype=torch.float32)

        # Standard Welford (Knuth, Vol 2):
        #   count <- count + 1
        #   delta  <- x - mean
        #   mean   <- mean + delta / count
        #   delta2 <- x - mean   (uses UPDATED mean -- this is correct)
        #   m2     <- m2 + delta * delta2
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count.to(torch.float32)
        delta2 = x - self.mean
        self.m2 += delta * delta2


# ===========================================================================
# Actor
# ===========================================================================
class Actor(nn.Module):
    """
    Squashed Gaussian policy: a = tanh(u), u ~ N(mean(s), std(s)).

    Architecture (state-based humanoid, defaults):
        obs (67,) -> [ObsNorm if use_obs_norm]
                  -> Linear(67, 256) -> ReLU
                  -> Linear(256, 256) -> ReLU
                  -> [mean head:    Linear(256, 21)]
                     [log_std head: Linear(256, 21)]

    obs_norm sharing
    ----------------
    If `use_obs_norm=True` and `obs_norm` is provided, that exact
    ObsNorm instance is used (registered as `self.obs_norm`). Pass the
    same instance into TwinCritic so actor and critic share normalization
    statistics -- avoids stat drift between the two networks and means
    the rollout-side `obs_norm.update()` call updates a single shared
    accumulator.

    If `use_obs_norm=True` and `obs_norm` is None, a fresh ObsNorm is
    constructed (useful for standalone Actor tests).

    If `use_obs_norm=False`, no ObsNorm submodule exists at all -- the
    state_dict is byte-identical to the pre-ObsNorm Actor for backward
    compatibility with old checkpoints.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        use_obs_norm: bool = False,
        obs_norm: Optional[ObsNorm] = None,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes

        # Optional input normalization. Constructed BEFORE the linear
        # layers so it shows up first in state_dict iteration order
        # (cosmetic; load_state_dict is order-insensitive).
        if use_obs_norm:
            if obs_norm is None:
                obs_norm = ObsNorm(obs_dim)
            assert obs_norm.obs_dim == obs_dim, (
                f"shared ObsNorm has obs_dim={obs_norm.obs_dim}, "
                f"actor expects {obs_dim}"
            )
            self.obs_norm = obs_norm
        else:
            self.obs_norm = None

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.mean_head = nn.Linear(h2, action_dim)
        self.log_std_head = nn.Linear(h2, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        _init_hidden_(self.fc1)
        _init_hidden_(self.fc2)
        _init_final_(self.mean_head)
        _init_final_(self.log_std_head)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the pre-squash Gaussian."""
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)  # read-only; never updates
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample with tanh squashing.

        Returns:
            action:   tanh-squashed action in [-1, 1], shape (..., action_dim)
            log_prob: log pi(a|s), reduced over the action dim, shape (...)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        action = torch.tanh(u)

        # log pi(a|s) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)
        # Numerically stable: log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        log_prob = normal.log_prob(u)  # (..., action_dim)
        log_prob -= 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation: tanh(mean)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# ===========================================================================
# Critic
# ===========================================================================
class _QNetwork(nn.Module):
    """
    Single Q-network: Q(s, a). Internal building block for TwinCritic.

    Architecture (with use_layer_norm=False, the default):
        concat(obs, action) -> Linear -> ReLU
                            -> Linear -> ReLU
                            -> Linear(_, 1)

    With use_layer_norm=True (TD7-style):
        concat(obs, action) -> Linear -> LayerNorm -> ReLU
                            -> Linear -> LayerNorm -> ReLU
                            -> Linear(_, 1)

    LayerNorm placement is BEFORE ReLU (TD7 convention). It is NOT applied
    to the output Q head -- Q values must be free to grow with the reward
    scale and discount. LayerNorm in the hidden layers regularizes the
    Q-function's per-state Lipschitz behavior, mitigating the late-stage
    Q-divergence we observed (Q1 max climbing from ~30 at 1M steps to ~230
    at 4M steps in the previous walk run).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        use_obs_norm: bool = False,
        obs_norm: Optional[ObsNorm] = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes

        # Optional shared obs normalization. Same semantics as Actor:
        # if use_obs_norm=False, no submodule -> state_dict identical to
        # pre-ObsNorm checkpoint shape.
        if use_obs_norm:
            if obs_norm is None:
                obs_norm = ObsNorm(obs_dim)
            self.obs_norm = obs_norm
        else:
            self.obs_norm = None

        self.fc1 = nn.Linear(obs_dim + action_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.q_head = nn.Linear(h2, 1)

        # Optional LayerNorm. Constructed conditionally so state_dict
        # has no `ln*` keys when disabled -- keeps backward compat with
        # pre-LN checkpoints.
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(h1)
            self.ln2 = nn.LayerNorm(h2)
        else:
            self.ln1 = None
            self.ln2 = None

        _init_hidden_(self.fc1)
        _init_hidden_(self.fc2)
        _init_final_(self.q_head)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q(s, a) with leading batch dims preserved; output shape (...,)."""
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)  # read-only
        x = torch.cat([obs, action], dim=-1)
        x = self.fc1(x)
        if self.ln1 is not None:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.ln2 is not None:
            x = self.ln2(x)
        x = F.relu(x)
        q = self.q_head(x)
        return q.squeeze(-1)


class TwinCritic(nn.Module):
    """
    Twin Q-networks for SAC's clipped double-Q.

    Bundles two independent Q-networks under one Module so that:
      - one optimizer covers both critics' parameters
      - one Polyak update covers both target critics
      - one state_dict() saves/loads both critics

    The two internal Q-networks have identical architecture but different
    random initializations. They SHARE the obs_norm instance (same Python
    object reference) -- there is exactly one ObsNorm per agent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        use_obs_norm: bool = False,
        obs_norm: Optional[ObsNorm] = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        # If caller asked for obs_norm but didn't supply one, create here
        # so both q1 and q2 share the same instance. (For the agent-shared
        # case, caller supplies the actor's ObsNorm and both q heads
        # reference it -- same Python object, same buffers, same updates.)
        if use_obs_norm and obs_norm is None:
            obs_norm = ObsNorm(obs_dim)
        self.q1 = _QNetwork(obs_dim, action_dim, hidden_sizes,
                            use_obs_norm=use_obs_norm, obs_norm=obs_norm,
                            use_layer_norm=use_layer_norm)
        self.q2 = _QNetwork(obs_dim, action_dim, hidden_sizes,
                            use_obs_norm=use_obs_norm, obs_norm=obs_norm,
                            use_layer_norm=use_layer_norm)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1(s, a), Q2(s, a))."""
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience: min(Q1, Q2). Used at bootstrap and (optionally) actor loss."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
