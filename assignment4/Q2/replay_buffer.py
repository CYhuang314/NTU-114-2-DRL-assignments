"""
replay_buffer.py

Uniform random replay buffer for off-policy RL.

Stores transitions as pre-allocated contiguous numpy arrays on CPU.
Returns sampled batches as torch tensors moved to the requested device.

Critical correctness note:
    `terminated` and `truncated` are stored separately. SAC's value bootstrap
    must mask only on `terminated`; it must NOT mask on `truncated`. DMC
    humanoid never terminates, only truncates at step 1000. Conflating the
    two would teach the critic a finite-horizon problem and cripple learning.
"""

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Uniform-sampled, ring-buffer replay storage for off-policy RL.

    Args:
        capacity:   Maximum number of transitions stored.
        obs_dim:    Dimensionality of observation vector.
        action_dim: Dimensionality of action vector.
        device:     torch device for sampled batches (e.g. 'cuda', 'cpu').
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device

        # Pre-allocated storage. Float32 throughout; bool flags for masks.
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._terminated = np.zeros((self.capacity,), dtype=np.bool_)
        self._truncated = np.zeros((self.capacity,), dtype=np.bool_)

        # Ring-buffer state.
        self._ptr = 0   # next write index, wraps at capacity
        self._size = 0  # current number of valid entries, capped at capacity

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert a single transition. Inputs must be float32 numpy arrays.

        We assert dtype to prevent silent float64 leaks from un-cast env outputs.
        """
        assert obs.dtype == np.float32, f"obs dtype must be float32, got {obs.dtype}"
        assert action.dtype == np.float32, f"action dtype must be float32, got {action.dtype}"
        assert next_obs.dtype == np.float32, f"next_obs dtype must be float32, got {next_obs.dtype}"
        assert obs.shape == (self.obs_dim,), f"obs shape {obs.shape} != ({self.obs_dim},)"
        assert action.shape == (self.action_dim,), f"action shape {action.shape} != ({self.action_dim},)"

        i = self._ptr
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._terminated[i] = terminated
        self._truncated[i] = truncated

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniformly sample a batch of transitions as torch tensors on `self.device`.

        Returns dict with keys:
            obs:        (B, obs_dim)  float32
            actions:    (B, action_dim) float32
            rewards:    (B,) float32
            next_obs:   (B, obs_dim)  float32
            terminated: (B,) float32  -- use this for the value-bootstrap mask
            truncated:  (B,) float32  -- exposed for logging; do NOT mask values with it
        """
        assert self._size > 0, "Cannot sample from empty buffer."
        assert batch_size <= self._size, (
            f"batch_size={batch_size} exceeds buffer size={self._size}; "
            "increase warmup steps or wait for buffer to fill."
        )

        idx = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "obs": torch.from_numpy(self._obs[idx]),
            "actions": torch.from_numpy(self._actions[idx]),
            "rewards": torch.from_numpy(self._rewards[idx]),
            "next_obs": torch.from_numpy(self._next_obs[idx]),
            # Cast bool -> float32 here so the agent can do `(1 - terminated)` cleanly.
            "terminated": torch.from_numpy(self._terminated[idx].astype(np.float32)),
            "truncated": torch.from_numpy(self._truncated[idx].astype(np.float32)),
        }

        # Single CPU->device transfer per sample call.
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def state_dict(self) -> Dict[str, np.ndarray]:
        """For checkpointing. Returns VIEWS into internal storage (no copy)
        to avoid the ~600MB peak-RSS spike at full capacity.

        WARNING: The returned arrays share memory with the buffer. Do not
        retain the dict across subsequent `add()` calls; pass directly to
        `torch.save()` and discard. If you need a snapshot for any other
        purpose, deep-copy at the call site.
        """
        return {
            "obs": self._obs[: self._size],
            "actions": self._actions[: self._size],
            "rewards": self._rewards[: self._size],
            "next_obs": self._next_obs[: self._size],
            "terminated": self._terminated[: self._size],
            "truncated": self._truncated[: self._size],
            "ptr": self._ptr,
            "size": self._size,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore from checkpoint. Capacity must match."""
        size = int(state["size"])
        assert size <= self.capacity, "Loaded buffer larger than capacity."
        self._obs[:size] = state["obs"]
        self._actions[:size] = state["actions"]
        self._rewards[:size] = state["rewards"]
        self._next_obs[:size] = state["next_obs"]
        self._terminated[:size] = state["terminated"]
        self._truncated[:size] = state["truncated"]
        self._ptr = int(state["ptr"])
        self._size = size
