"""
replay_buffer.py

Uniform random replay buffer for off-policy RL, with optional n-step returns.

Stores transitions as pre-allocated contiguous numpy arrays on CPU. Returns
sampled batches as torch tensors moved to the requested device.

n-step returns
--------------
Storage is always single-step (one transition per slot). The n-step
target is constructed at *sample time* by walking forward from each
sampled start index and accumulating discounted rewards. With
`n_step=1` the math collapses exactly to the 1-step SAC target, so
old configs train numerically identically (verified by inspection of
the math; see docstring of `sample`).

Returned batch schema (n-step aware)
------------------------------------
    obs       : (B, obs_dim)     float32   start state s_i
    actions   : (B, action_dim)  float32   action a_i taken at s_i
    returns   : (B,)             float32   sum_{k=0..n_eff-1} gamma^k * r_{i+k}
    next_obs  : (B, obs_dim)     float32   bootstrap state s_{i+n_eff}
                                           (= next_obs of the last consumed slot)
    discount  : (B,)             float32   gamma^{n_eff} if bootstrap alive, else 0
    truncated : (B,)             float32   logging parity only; NOT used in target

`discount` already folds in the "kill bootstrap on real termination" logic:
the SAC update reduces to  `target = returns + discount * V_bootstrap`.

Critical correctness points
---------------------------
1. The bootstrap is killed ONLY by `terminated`, never by `truncated`.
   On `truncated`, the chunk shortens but the bootstrap stays alive — the
   truncating slot's `next_obs` is the genuine pre-reset terminal observation,
   which is what we bootstrap from. DMC humanoid never sets `terminated`,
   only `truncated` at step 1000; conflating them would teach a 1000-step
   finite-horizon problem and cripple learning.

2. The reward at the stopping step is INCLUDED. We stop *after* consuming
   the terminating/truncating transition's reward, then bootstrap from
   that transition's `next_obs`. This matches Sutton & Barto's standard
   n-step return definition.

3. Ring-buffer wrap safety: when the buffer is full, the (n-1) indices
   immediately *behind* the write head `_ptr` are unsafe as start indices
   (their forward walk would step ONTO `_ptr`, which holds the oldest
   live data — i.e., data from BEFORE the wrap, breaking temporal
   contiguity). These are excluded from the sampleable set.
"""

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Uniform-sampled, ring-buffer replay storage with optional n-step returns.

    Args:
        capacity:   Maximum number of transitions stored.
        obs_dim:    Dimensionality of observation vector.
        action_dim: Dimensionality of action vector.
        device:     torch device for sampled batches (e.g. 'cuda', 'cpu').
        gamma:      Discount factor used to construct n-step returns at
                    sample time. Must equal the agent's gamma; train.py
                    asserts this after constructing both.
        n_step:     Number of steps to accumulate per sample. n_step=1
                    reproduces standard 1-step SAC. n_step >= 1.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float,
        n_step: int = 1,
    ) -> None:
        assert int(n_step) >= 1, f"n_step must be >= 1, got {n_step}"
        assert 0.0 < float(gamma) < 1.0, f"gamma must be in (0, 1), got {gamma}"

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = float(gamma)
        self.n_step = int(n_step)

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

        # Precomputed gamma powers for the n-step accumulation.
        # Shape (n_step,): [gamma^0, gamma^1, ..., gamma^{n-1}]. Used to weight
        # rewards inside a chunk. The bootstrap discount gamma^{n_eff} is
        # computed per-sample (since n_eff varies) using the same base.
        self._gamma_powers = (self.gamma ** np.arange(self.n_step, dtype=np.float64)).astype(np.float32)

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

    # =====================================================================
    # n-step sampling
    # =====================================================================
    def _sample_start_indices(self, batch_size: int) -> np.ndarray:
        """Draw `batch_size` valid n-step start indices.

        Two regimes (see module docstring point 3):

        Regime A — buffer not yet full (size < capacity):
            All data lives at [0, size). Valid starts are [0, size - n + 1).
            No wraparound to worry about (we haven't written past size-1 yet).

        Regime B — buffer full (size == capacity):
            Data is a ring; `_ptr` is the next-write slot, currently holding
            the OLDEST live transition. Forward-walking from a start index
            must not step ONTO `_ptr` (that would cross the temporal seam
            from new data back to old data). Equivalently: exclude the n-1
            indices immediately BEHIND `_ptr`, i.e.,
                {(_ptr - 1) mod C, ..., (_ptr - n + 1) mod C}
            from the sampleable set. The starting slot itself may be `_ptr`
            (it's the oldest valid start).

            For n=1 the exclusion set is empty; this collapses to ordinary
            uniform sampling over [0, C), which matches the prior behavior.
        """
        n = self.n_step
        N = self._size
        C = self.capacity

        assert N >= n, (
            f"buffer has {N} transitions, cannot sample with n_step={n}; "
            f"increase warmup_steps or wait for buffer to fill."
        )

        if N < C:
            # Regime A: contiguous [0, N), exclude tail (n-1) slots.
            num_valid = N - n + 1
            return np.random.randint(0, num_valid, size=batch_size)

        # Regime B: full buffer.
        # Exclude {(_ptr - 1) mod C, ..., (_ptr - n + 1) mod C}, that is, n-1 slots.
        # For n=1, no exclusion -> direct uniform.
        if n == 1:
            return np.random.randint(0, C, size=batch_size)

        # Sample uniformly from a contiguous range of length (C - (n-1)),
        # then remap so the excluded slots are "skipped over". Concretely:
        # Let raw be in [0, C - n + 1). Define i = (raw + _ptr) mod C.
        # As raw ranges over its domain, i ranges over the (C - n + 1) slots
        # starting at _ptr and going forward — which is exactly all slots
        # EXCEPT {(_ptr - 1) mod C, ..., (_ptr - n + 1) mod C}.
        # (Sanity: starting at _ptr is included; starting at _ptr - 1 is not,
        #  since raw = -1 is out of range; etc.)
        raw = np.random.randint(0, C - n + 1, size=batch_size)
        return ((raw + self._ptr) % C).astype(np.int64)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniformly sample n-step transitions as torch tensors on `self.device`.

        Returns dict with keys (see module docstring for full schema):
            obs, actions, returns, next_obs, discount, truncated

        With n_step=1 this collapses exactly to the standard 1-step target:
            returns  = rewards
            discount = gamma * (1 - terminated)
            next_obs = next_obs (unchanged)
        """
        assert self._size > 0, "Cannot sample from empty buffer."
        assert batch_size <= self._size, (
            f"batch_size={batch_size} exceeds buffer size={self._size}; "
            "increase warmup steps or wait for buffer to fill."
        )

        n = self.n_step
        C = self.capacity
        B = batch_size

        # ---- Phase 1: start indices (per regime). ----
        start = self._sample_start_indices(B)  # (B,) int64

        # ---- Phase 2: build (B, n) chunk index matrix. ----
        # chunk_idx[b, k] = (start[b] + k) mod C
        offsets = np.arange(n, dtype=np.int64)               # (n,)
        chunk_idx = (start[:, None] + offsets[None, :]) % C  # (B, n)

        # ---- Phase 3: gather chunk fields. ----
        rew = self._rewards[chunk_idx]            # (B, n) float32
        term = self._terminated[chunk_idx]        # (B, n) bool
        trunc = self._truncated[chunk_idx]        # (B, n) bool

        # ---- Phase 4: find effective chunk length n_eff per sample. ----
        # `stop[b, k]` = True iff the chunk should stop AT step k
        # (i.e., we consume reward k, then halt; we do NOT consume k+1).
        stop = term | trunc                       # (B, n) bool

        # n_eff = index of first True in `stop` along axis=1, plus 1.
        # If no True anywhere in the row, n_eff = n.
        # np.argmax returns 0 for all-False rows (which would be wrong),
        # so guard with `any(...)`.
        any_stop = stop.any(axis=1)               # (B,) bool
        first_stop_k = np.where(
            any_stop,
            stop.argmax(axis=1),                  # k of first stop (only valid if any_stop)
            n - 1,                                # placeholder; will produce n_eff = n below
        )                                         # (B,) int
        # n_eff in {1, ..., n}. When any_stop is False, n_eff = n.
        # When any_stop is True at k, n_eff = k + 1 (we consumed steps 0..k).
        n_eff = np.where(any_stop, first_stop_k + 1, n).astype(np.int64)  # (B,)

        # ---- Phase 5: accumulate returns with the "consume up to and including
        #               the stopping step" rule. ----
        # `consumed[b, k]` = True iff reward at step k is included in returns
        # for sample b. That is True iff k < n_eff[b].
        # Equivalently: consumed = (offsets[None, :] < n_eff[:, None]).
        consumed = offsets[None, :] < n_eff[:, None]   # (B, n) bool
        # Discount weights (gamma^k) are precomputed.
        weighted = rew * consumed.astype(np.float32) * self._gamma_powers[None, :]
        returns = weighted.sum(axis=1)            # (B,) float32

        # ---- Phase 6: bootstrap state and discount. ----
        # Bootstrap from the LAST CONSUMED slot's next_obs. The last consumed
        # k is (n_eff - 1), so the buffer index is chunk_idx[b, n_eff[b] - 1].
        last_consumed_idx = chunk_idx[np.arange(B), n_eff - 1]   # (B,) int64
        boot_obs = self._next_obs[last_consumed_idx]             # (B, obs_dim)

        # Bootstrap is killed iff the chunk stopped because of `terminated`.
        # `truncated` alone shortens the chunk but leaves the bootstrap alive
        # (truncation is a time-limit artifact, not a real terminal).
        # If any_stop is False, bootstrap is alive (n_eff == n, full chunk walked).
        # If any_stop is True at k, bootstrap is killed iff term[b, k] is True.
        stop_was_terminal = np.where(
            any_stop,
            self._terminated[last_consumed_idx],
            False,
        )                                          # (B,) bool
        alive = ~stop_was_terminal                 # (B,) bool

        # Final discount: gamma^{n_eff} if alive, else 0.
        # Compute gamma^{n_eff} in float32. n_eff is small (<=n), so direct
        # exponentiation is fine; keep float64 intermediate for safety.
        discount = np.where(
            alive,
            (self.gamma ** n_eff.astype(np.float64)).astype(np.float32),
            np.float32(0.0),
        )

        # ---- Phase 7: gather start state and action; expose `truncated`
        #               (parity flag, not used in target). ----
        obs_out = self._obs[start]                 # (B, obs_dim)
        act_out = self._actions[start]             # (B, action_dim)
        # `truncated` parity field: was THIS sample's chunk cut by truncation
        # (and not by termination)? Useful for logging "what fraction of
        # samples had a short chunk". Not consumed by the SAC update.
        trunc_flag = (any_stop & ~stop_was_terminal).astype(np.float32)

        batch = {
            "obs": torch.from_numpy(obs_out),
            "actions": torch.from_numpy(act_out),
            "returns": torch.from_numpy(returns),
            "next_obs": torch.from_numpy(boot_obs),
            "discount": torch.from_numpy(discount),
            "truncated": torch.from_numpy(trunc_flag),
        }

        # Single CPU->device transfer per sample call.
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # =====================================================================
    # Checkpointing
    # =====================================================================
    def state_dict(self) -> Dict[str, np.ndarray]:
        """For checkpointing. Returns VIEWS into internal storage (no copy)
        to avoid the ~600MB peak-RSS spike at full capacity.

        WARNING: The returned arrays share memory with the buffer. Do not
        retain the dict across subsequent `add()` calls; pass directly to
        `torch.save()` and discard. If you need a snapshot for any other
        purpose, deep-copy at the call site.

        Includes `gamma` and `n_step` so a resume can sanity-check that the
        buffer's discount/horizon settings match the new agent config.
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
            "gamma": self.gamma,
            "n_step": self.n_step,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore from checkpoint. Capacity must match.

        `gamma` and `n_step` are checked against the live buffer's config
        and a mismatch raises — silently changing either across a resume
        would invalidate the n-step targets retroactively for the loaded
        transitions (in practice it's fine since storage is single-step,
        but the target you'd compute on the next sample would mix two
        n's, so we forbid it).

        For backward compat with checkpoints predating the n-step rewrite,
        absent `gamma`/`n_step` keys are tolerated (assumed to match).
        """
        size = int(state["size"])
        assert size <= self.capacity, "Loaded buffer larger than capacity."
        if "gamma" in state:
            assert float(state["gamma"]) == self.gamma, (
                f"Buffer gamma mismatch on resume: ckpt={state['gamma']}, "
                f"live={self.gamma}"
            )
        if "n_step" in state:
            assert int(state["n_step"]) == self.n_step, (
                f"Buffer n_step mismatch on resume: ckpt={state['n_step']}, "
                f"live={self.n_step}"
            )
        self._obs[:size] = state["obs"]
        self._actions[:size] = state["actions"]
        self._rewards[:size] = state["rewards"]
        self._next_obs[:size] = state["next_obs"]
        self._terminated[:size] = state["terminated"]
        self._truncated[:size] = state["truncated"]
        self._ptr = int(state["ptr"])
        self._size = size
