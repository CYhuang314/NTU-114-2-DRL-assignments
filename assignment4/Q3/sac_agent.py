"""
sac_agent.py

Soft Actor-Critic (SAC) update logic and target-network bookkeeping for
state-based DMC humanoid (Haarnoja et al. 2018a, "Soft Actor-Critic").

This module owns:
    - Actor + TwinCritic + target TwinCritic (no target actor; SAC has none)
    - Their optimizers
    - Device placement of all networks
    - One SAC update step per `update(batch)` call
    - Action selection (stochastic for training, deterministic for eval)

This module does NOT own:
    - The training loop, warmup, env, or replay buffer instance
    - Logging, checkpointing to disk, eval-episode rollouts
    - Update-to-data ratio (UTD) — caller decides how often to invoke update()

Critical correctness notes
--------------------------
1. The TD target is built from a pre-aggregated n-step batch produced by
   the replay buffer. The buffer hands us:
       returns  : sum_{k=0..n_eff-1} gamma^k * r_{i+k}
       next_obs : bootstrap state s_{i+n_eff} (= last consumed slot's next_obs)
       discount : gamma^{n_eff} if bootstrap alive, else 0
   The target reduces to one clean line:
       target = returns + discount * V_bootstrap(next_obs)
   With n_step=1 this is byte-identical to the prior 1-step SAC target
       target = rewards + gamma * (1 - terminated) * V_bootstrap(next_obs)
   (since returns=rewards and discount=gamma*(1-terminated) for n=1).

   The "bootstrap killed only by `terminated`, never by `truncated`" rule
   that this code used to enforce locally is now enforced INSIDE the
   buffer's sample() — `discount` is already 0 iff the chunk stopped on
   a real `terminated` flag; `truncated` only shortens the chunk and
   leaves the bootstrap alive. Do not re-apply termination masking here;
   it would double-count.

   The buffer also exposes `truncated` as a per-sample float32 flag for
   logging parity (fraction of chunks cut by truncation rather than
   termination). It is not used in the target.

2. Target networks: critic-only (no target actor). Initialized by hard-copy
   from the online critic; updated via Polyak (soft) averaging at every
   update step with coefficient `tau`. Targets stay in `eval()` mode and
   have `requires_grad=False`. They are only used inside `torch.no_grad()`
   contexts when computing the TD target.

3. Gradient flow:
   - Critic update: target action, target log_prob, and target Q values are
     all under `torch.no_grad()`. The critic's two MSE losses are summed
     (not averaged across heads) and a single .backward() updates both Q
     heads via the shared TwinCritic optimizer.
   - Actor update: action is reparameterized-sampled from the actor (grad
     flows back through the policy). Q values from the *online* critic are
     used to score this action — gradients flow through the critic into
     the actor. The critic optimizer is NOT stepped on the actor loss; only
     the actor optimizer's .step() is called. This is the standard SAC
     pattern; matches Haarnoja's original code.

4. Alpha is auto-tuned (Haarnoja et al. 2018b, "SAC and Applications").
   `cfg["alpha"]` is the *initial* value of α; `log_alpha` is then an
   `nn.Parameter` updated by its own Adam optimizer. The alpha loss is
   `-log_alpha * (log_pi + target_entropy).detach()`, minimized w.r.t.
   log_alpha. With `target_entropy = -action_dim` (CleanRL convention),
   α automatically rises if the policy is too peaked and falls if it
   is too uniform. This was previously deferred and was activated after
   the initial fixed-α=0.2 run hit max-entropy policy collapse on
   humanoid (entropy stuck at the tanh-squashed Gaussian ceiling, Q
   inflating without policy improvement).
"""

import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Actor, ObsNorm, TwinCritic


class SACAgent:
    """SAC agent: networks + optimizers + update step + action selection.

    Constructor takes a single `cfg` dict for flexibility. Required keys:
        obs_dim:        int   (e.g. 67)
        action_dim:     int   (e.g. 21)
        device:         torch.device or str
        gamma:          float (discount)
        tau:            float (Polyak coefficient for target update)
        alpha:          float (INITIAL value of α; auto-tuned thereafter)
        actor_lr:       float
        critic_lr:      float
        alpha_lr:       float (LR for the log_alpha parameter)
        hidden_sizes:   tuple[int, int] (e.g. (256, 256))

    Optional keys:
        target_entropy: float | None. None defaults to -action_dim
                        (Haarnoja heuristic for continuous actions).
        use_obs_norm:   bool. Default False. If True, builds a shared
                        ObsNorm input normalizer used by actor + both
                        critics + target critic. Rollout-side updates
                        must be driven by `agent.obs_norm_update(obs)`
                        from the training loop after every env step.
        use_layer_norm: bool. Default False. If True, both Q heads use
                        LayerNorm in their hidden layers (TD7 style).
                        Targets late-stage Q-divergence; safe to combine
                        with use_obs_norm.

    Example:
        cfg = {
            "obs_dim": 67, "action_dim": 21,
            "device": "cuda", "gamma": 0.99, "tau": 0.005, "alpha": 0.2,
            "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
            "hidden_sizes": (256, 256),
            "use_obs_norm": True, "use_layer_norm": True,
        }
        agent = SACAgent(cfg)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["device"])

        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])

        obs_dim = int(cfg["obs_dim"])
        action_dim = int(cfg["action_dim"])
        hidden_sizes = tuple(cfg["hidden_sizes"])

        # Optional architectural flags. Default False to preserve the prior
        # vanilla SAC behavior; flip to True via cfg for the new arms.
        use_obs_norm = bool(cfg.get("use_obs_norm", False))
        use_layer_norm = bool(cfg.get("use_layer_norm", False))
        self.use_obs_norm = use_obs_norm
        self.use_layer_norm = use_layer_norm

        # --- ObsNorm (single shared instance) -------------------------------
        # Built before the networks so we can pass the same Python object to
        # actor, critic, and target critic. All three then call into the same
        # buffers; rollout-side `agent.obs_norm_update(obs)` updates ONE
        # accumulator that all three see.
        #
        # When use_obs_norm=False, self.obs_norm = None and the networks
        # construct themselves with no obs_norm submodule (state_dict
        # byte-identical to pre-ObsNorm checkpoints -- backward compat).
        if use_obs_norm:
            self.obs_norm = ObsNorm(obs_dim).to(self.device)
        else:
            self.obs_norm = None

        # --- Networks -------------------------------------------------------
        self.actor = Actor(
            obs_dim, action_dim, hidden_sizes,
            use_obs_norm=use_obs_norm,
            obs_norm=self.obs_norm,
        ).to(self.device)
        self.critic = TwinCritic(
            obs_dim, action_dim, hidden_sizes,
            use_obs_norm=use_obs_norm,
            obs_norm=self.obs_norm,
            use_layer_norm=use_layer_norm,
        ).to(self.device)

        # Target critic: hard-copy of online critic at init, no grads, eval mode.
        # IMPORTANT: target critic shares the SAME ObsNorm instance as the
        # online critic. The target's normalization stats are therefore not
        # Polyak-lagged -- they always reflect the latest rollout statistics.
        # This is intentional and correct: ObsNorm is a *property of the
        # observation space*, not a learnable component, so there is no
        # "target normalization" the way there is a target Q-function.
        # CleanRL/SB3 handle obs normalization the same way (single shared
        # accumulator, applied identically to all forward passes).
        self.critic_target = TwinCritic(
            obs_dim, action_dim, hidden_sizes,
            use_obs_norm=use_obs_norm,
            obs_norm=self.obs_norm,    # SAME instance, not a clone
            use_layer_norm=use_layer_norm,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_target.eval()

        # --- Auto-α (Haarnoja 2018b) ---------------------------------------
        # `target_entropy` is the constraint we hold the policy entropy at.
        # Standard heuristic: -action_dim. For our 21-dim humanoid, -21.
        # Note: tanh-squashed policies can have entropy outside [-action_dim,
        # action_dim*log(2)] because the Jacobian correction term in log_pi
        # is unbounded; -action_dim is empirically what works.
        target_entropy = cfg.get("target_entropy", None)
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        # log_alpha is a single learnable scalar. Initialized so that
        # exp(log_alpha) == cfg["alpha"]. We optimize log_alpha (not alpha
        # directly) to keep alpha strictly positive automatically.
        init_alpha = float(cfg["alpha"])
        assert init_alpha > 0.0, "Initial alpha must be > 0"
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(init_alpha), dtype=torch.float32, device=self.device)
        )

        # --- Optimizers -----------------------------------------------------
        # Adam defaults match Haarnoja: betas=(0.9, 0.999), eps=1e-8.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg["actor_lr"])
        )
        # One optimizer covers BOTH Q heads (TwinCritic.parameters() yields
        # all params from q1 and q2). One backward(), one step().
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg["critic_lr"])
        )
        # Alpha optimizer: single-parameter, but Adam still has its
        # m/v running averages, so it's not equivalent to plain SGD.
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(cfg["alpha_lr"])
        )

        # Bookkeeping for users that want it (not strictly needed by update()).
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self._update_count = 0

    @property
    def alpha(self) -> float:
        """Current α value (read-only). For tensor-side use in losses,
        access `self.log_alpha.exp()` directly with .detach() as needed."""
        return float(self.log_alpha.exp().item())

    # =====================================================================
    # ObsNorm rollout-side update (no-op when use_obs_norm=False)
    # =====================================================================
    def obs_norm_update(self, obs: np.ndarray) -> None:
        """Feed one rollout observation into the ObsNorm Welford accumulator.

        Call this from the training loop AFTER every env step (including
        warmup random-action steps). Does nothing when use_obs_norm=False.

        The single-obs constraint is intentional: see ObsNorm.update() for
        why batch-mode updates on sampled training data would corrupt
        the running stats.
        """
        if self.obs_norm is None:
            return
        # numpy float32 (env wrapper guarantee) -> torch tensor on device.
        # ObsNorm.update handles the dtype/device coercion defensively too.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.obs_norm.update(obs_t)

    # =====================================================================
    # Action selection
    # =====================================================================
    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Return a float32 numpy action for one observation.

        Args:
            obs: float32 numpy array, shape (obs_dim,). The training env
                wrapper guarantees float32; for eval (`student_agent.py`),
                the caller must cast before passing in.
            deterministic: True at eval, False during training rollouts.

        Returns:
            action: float32 numpy array, shape (action_dim,), in [-1, 1].
        """
        # Add batch dim, move to device.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if deterministic:
            action_t = self.actor.act_deterministic(obs_t)
        else:
            action_t, _ = self.actor.sample(obs_t)

        # (1, action_dim) -> (action_dim,) numpy float32.
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)

    # =====================================================================
    # SAC update step
    # =====================================================================
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run one SAC update on a sampled tensor batch.

        Expects `batch` already on `self.device` with keys (n-step schema):
            obs:       (B, obs_dim)    float32
            actions:   (B, action_dim) float32
            returns:   (B,)            float32  -- n-step return
                                                   sum_{k=0..n_eff-1} gamma^k * r_{i+k}
            next_obs:  (B, obs_dim)    float32  -- bootstrap state s_{i+n_eff}
            discount:  (B,)            float32  -- gamma^{n_eff} if bootstrap alive,
                                                   else 0 (kills bootstrap on real
                                                   `terminated`; truncation does NOT
                                                   kill it). USE THIS as the bootstrap
                                                   coefficient — do not re-apply
                                                   gamma or termination masks.
            truncated: (B,)            float32  -- logging parity flag; NOT used here.

        With n_step=1 the schema is mathematically identical to the previous
        1-step SAC target.

        Returns:
            dict of scalar python floats for logging:
                critic_loss, actor_loss, alpha_loss,
                q1_mean, q2_mean, log_prob_mean, entropy, alpha
        """
        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        next_obs = batch["next_obs"]
        discount = batch["discount"]   # NOTE: pre-folded gamma^{n_eff} * (1 - terminated_at_stop).
                                       # Do not multiply by self.gamma or (1 - terminated) again.

        # ----- Critic update ----------------------------------------------
        # Target: y = returns + discount * (min Q'(s', a') - alpha * log pi(a'|s'))
        # All target-side computation under no_grad — gradients must NOT flow
        # back into the actor or the target critic from this branch.
        # `alpha` here is detached: critic update does not see log_alpha grads.
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            next_q1_t, next_q2_t = self.critic_target(next_obs, next_action)
            next_q_min = torch.min(next_q1_t, next_q2_t)
            # Soft value: V(s') = min Q' - alpha * log pi
            next_v = next_q_min - alpha * next_log_prob
            # n-step TD target. `discount` already encodes both gamma^{n_eff}
            # and the "kill on terminated" mask; see buffer docs.
            target_q = returns + discount * next_v

        # Online Q estimates of the actually-taken action.
        q1, q2 = self.critic(obs, actions)
        # Sum the two MSE losses (not 0.5*mean) — matches Haarnoja's reference.
        # The factor only affects effective LR; we keep a clean sum.
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor update ----------------------------------------------
        # Sample fresh action from current policy (reparameterized — grad flows).
        # Q values come from ONLINE critic (not target). Critic params get
        # gradients here too, but we never call critic_optimizer.step() on
        # this loss, so they are silently discarded. Standard SAC pattern.
        # `alpha` is detached: actor sees alpha as a constant scalar; the
        # log_alpha gradient comes from the alpha update below.
        new_action, log_prob = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        q_pi_min = torch.min(q1_pi, q2_pi)

        # Actor loss: maximize E[Q - alpha * log pi]  =>  minimize alpha*log pi - Q
        actor_loss = (alpha * log_prob - q_pi_min).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Alpha update (Haarnoja 2018b) -----------------------------
        # Maintains the entropy constraint H(π) ≥ -target_entropy via
        # Lagrangian dual ascent on α. With target_entropy = -action_dim,
        # the equilibrium is at H(π) = action_dim (a positive lower bound
        # on policy entropy).
        #
        # Formula matches CleanRL exactly:
        #     alpha_loss = (-α · (log_pi + target_entropy)).mean()
        # where α = exp(log_alpha) and log_pi is detached. Note this is
        # `-α · …`, not `-log_alpha · …` — they have different gradients
        # w.r.t. log_alpha (the former scales by α, slowing α as it
        # approaches 0, which is desirable for stability).
        #
        # We reuse `log_prob` from the actor update (detached) — saves a
        # forward pass and matches CleanRL's implementation choice.
        alpha_loss = (-self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----- Polyak target update --------------------------------------
        self._soft_update_target()

        self._update_count += 1

        # ----- Logging scalars (all detached, .item()'d on caller side
        # only when they leave this dict) ---------------------------------
        with torch.no_grad():
            entropy = -log_prob.mean()  # MC entropy estimate from the same sample
            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "q1_mean": q1.mean().item(),
                "q2_mean": q2.mean().item(),
                "log_prob_mean": log_prob.mean().item(),
                "entropy": entropy.item(),
                "alpha": self.log_alpha.exp().item(),
            }

    # =====================================================================
    # Internal: Polyak soft update of target critic
    # =====================================================================
    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """target <- tau * online + (1 - tau) * target, parameter-wise."""
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            # In-place: p_t.mul_(1 - tau).add_(p, alpha=tau)
            # equivalent to: p_t = (1 - tau) * p_t + tau * p
            p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    # =====================================================================
    # Checkpointing
    # =====================================================================
    def state_dict(self) -> Dict[str, Any]:
        """Return everything needed to resume training.

        Excludes the replay buffer (caller's job to checkpoint separately).
        """
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore training state. Architectures must match."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        # Auto-α state. Older checkpoints (pre-auto-α) won't have these
        # keys; fall through silently and keep the freshly-init log_alpha
        # so resume from old checkpoints still works.
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
        if "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self._update_count = int(state.get("update_count", 0))
        # Restore frozen / eval state on target (load_state_dict doesn't
        # touch requires_grad or training mode).
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_target.eval()

    def actor_state_dict(self) -> Dict[str, torch.Tensor]:
        """Convenience: just the actor weights, for inference-only checkpoints
        consumed by `student_agent.py` at eval time."""
        return self.actor.state_dict()
