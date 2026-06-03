"""
student_agent.py

Inference-time agent for DRL HW4 Q2 (humanoid-walk) and Q3 (humanoid-run).

The TA's eval scripts (eval_walk.py / eval_run.py) load this file via
importlib, instantiate `Agent()` with no arguments, and call `act(obs)`
in a tight loop. The env they pass observations from is the *unwrapped*
DMC env from `dmc.make_dmc_env(...)`, so:

    obs   in:  float64 numpy, shape (67,)
    action out: float64 numpy, shape (21,), in [-1, 1]

This module bridges the dtype boundary: cast obs float64 -> float32 on the
way in, cast action float32 -> float64 on the way out. Internally the
actor runs in float32, matching how it was trained.

Submission layout (this file + these two siblings in the same directory):

    student_agent.py
    networks.py
    best_actor.pt    <- saved by train.py at the eval-best checkpoint

Selecting the architecture variant
----------------------------------
Two architectural switches must match the training run that produced
`best_actor.pt`. Set them below:

    USE_OBS_NORM   - True if the actor was trained with the ObsNorm input
                     normalizer (the input running-stats buffers are then
                     part of best_actor.pt and load along with the weights).
                     False for vanilla checkpoints.
    USE_LAYER_NORM - Critic-only flag; the actor weights are unaffected by
                     critic LayerNorm. Kept here for clarity and so the
                     submission directory is self-documenting.

If you mix these incorrectly with the checkpoint, load_state_dict raises
a clear key-mismatch error rather than silently producing garbage actions.

If the architecture in `networks.py` changes between training and eval
(e.g. hidden_sizes bumped to 512), the constructor args below MUST be
updated to match, or `load_state_dict` will throw a shape mismatch.
"""

from pathlib import Path
import os

import gymnasium as gym
import numpy as np
import torch

from networks import Actor


# ---- Architecture constants ------------------------------------------------
# Must match what train.py / sac_agent.py used. If you change one, change
# the other.
OBS_DIM = 67
ACTION_DIM = 21
HIDDEN_SIZES = (256, 256)

# ---- A/B switch: which training arm produced best_actor.pt? ---------------
# Flip these to match the checkpoint. Mismatch -> load_state_dict raises.
#   USE_OBS_NORM=False, USE_LAYER_NORM=False : vanilla SAC (old checkpoint)
#   USE_OBS_NORM=True,  USE_LAYER_NORM=True  : new arm (ObsNorm + critic LN)
# Note: USE_LAYER_NORM applies only to the critic, not the actor; this
# constant exists for documentation / future symmetry, and is unused by
# `Actor` itself.
USE_OBS_NORM = True
USE_LAYER_NORM = True

# ---- Env-var override (no CLI seam — eval scripts call Agent() with no args)
# Set STUDENT_USE_OBS_NORM=1 (or 0) to override USE_OBS_NORM at eval time
# without editing the file. Convenient for A/B-ing two best_actor.pt files
# with the same student_agent.py:
#     STUDENT_USE_OBS_NORM=0 python eval_walk.py    # ARM A checkpoint
#     STUDENT_USE_OBS_NORM=1 python eval_walk.py    # ARM B checkpoint
# Accepted values for True: "1", "true", "yes" (case-insensitive).
# Anything else -> False. Unset -> use the USE_OBS_NORM constant above.
_env_override = os.environ.get("STUDENT_USE_OBS_NORM")
if _env_override is not None:
    USE_OBS_NORM = _env_override.strip().lower() in ("1", "true", "yes")

# ---- Weights file ----------------------------------------------------------
# Resolved relative to this file, so the eval script's working directory
# doesn't matter.
ACTOR_WEIGHTS_PATH = Path(__file__).parent / "best_actor.pt"


class Agent(object):
    """Deterministic SAC actor wrapped to match the TA's eval interface.

    Do NOT modify `__init__` signature or `act` signature — the eval
    scripts depend on them being argument-free / single-argument.
    """

    def __init__(self):
        # Exposed for parity with the TA's placeholder; not used internally.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float64)

        # Prefer GPU if present; CPU works fine too (actor is small).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the actor with the SAME architecture as training.
        # use_obs_norm must match the training arm (see USE_OBS_NORM above).
        # When True, the ObsNorm running-stats buffers are loaded from
        # best_actor.pt as part of state_dict; no separate file needed.
        self.actor = Actor(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=HIDDEN_SIZES,
            use_obs_norm=USE_OBS_NORM,
        ).to(self.device)

        # Load weights. We let FileNotFoundError surface with a clear hint
        # rather than silently using random init.
        if not ACTOR_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"Actor weights not found at {ACTOR_WEIGHTS_PATH}. "
                f"Place 'best_actor.pt' next to student_agent.py."
            )
        state_dict = torch.load(
            ACTOR_WEIGHTS_PATH,
            map_location=self.device,
            weights_only=False,
        )
        # strict=True (default) is intentional: a missing/extra key here
        # almost always means USE_OBS_NORM is set wrong for this checkpoint.
        # Fail loudly rather than silently running on an un-normalized
        # actor that was trained on normalized inputs (or vice versa).
        self.actor.load_state_dict(state_dict)

        # eval() disables any train-mode behavior (none in current Actor /
        # ObsNorm, but defensive and free).
        self.actor.eval()

    @torch.no_grad()
    def act(self, observation):
        """Deterministic policy: a = tanh(mean(s)).

        Args:
            observation: float64 numpy array, shape (67,), from the
                unwrapped DMC env.

        Returns:
            float64 numpy array, shape (21,), in [-1, 1]. float64 because
            MuJoCo / dm_control consumes float64 internally.
        """
        # float64 -> float32 torch tensor on device, add batch dim.
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Deterministic action; tanh keeps it strictly in [-1, 1].
        # ObsNorm (if present) normalizes inside actor.forward; running
        # stats are frozen at whatever values were saved in best_actor.pt
        # (we never call obs_norm.update at eval time).
        action_t = self.actor.act_deterministic(obs_t)

        # (1, 21) torch float32 -> (21,) numpy float64 for the env.
        return action_t.squeeze(0).cpu().numpy().astype(np.float64)
