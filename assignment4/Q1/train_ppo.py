"""
PPO trainer for Pendulum-v1.

Key implementation details:
  - Vectorized envs (SyncVectorEnv) for decorrelated samples.
  - GAE(lambda) advantage estimation.
  - Truncation-aware bootstrapping: Pendulum has no termination, only
    truncation at step 200; we always bootstrap with V(s_{t+1}).
  - Action sampled from Normal(mu, sigma) with state-independent log-std,
    clipped to [-2, 2] for the env, log-prob computed on raw sample.
  - PPO clip + value clip + entropy bonus + grad clip.
  - Per-batch advantage normalization.
  - Saves checkpoint to ppo_pendulum.pt for student_agent.py to load.

Usage:
    python train_ppo.py
"""
import os
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from ppo_model import (
    ActorCritic, preprocess_obs,
    OBS_DIM, ACT_DIM, ACTION_SCALE,
)


@dataclass
class PPOConfig:
    # Env (matched to RL-Zoo / SB3 PPO Pendulum-v1 defaults)
    env_id: str = "Pendulum-v1"
    num_envs: int = 4               # SB3: n_envs=4
    rollout_len: int = 1024         # SB3: n_steps=1024  -> batch = 4*1024 = 4096
    seed: int = 1

    # Optimization
    total_timesteps: int = 600_000  # SB3 uses 100k; we give 3x margin
    lr: float = 1e-3                # SB3: lr=1e-3
    anneal_lr: bool = True

    # PPO
    gamma: float = 0.9              # SB3: gamma=0.9 (short horizon)
    gae_lambda: float = 0.95
    update_epochs: int = 10         # SB3: n_epochs=10
    minibatch_size: int = 64        # SB3: batch_size=64 -> 4096/64 = 64 minibatches
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Exploration / log_std schedule
    # Anneal log_std_max linearly from log_std_max_start to log_std_max_end
    # over the fraction [anneal_start_frac, anneal_end_frac] of training.
    # This forces the policy to sharpen near the end so eval-time variance is low.
    init_log_std: float = 0.0           # start sigma = 1.0
    log_std_max_start: float = 2.0      # initially un-clamped (we want it free to learn)
    log_std_max_end: float = -2.3       # end sigma = exp(-2.3) ~= 0.10
    anneal_start_frac: float = 0.30     # start tightening after 30% of training
    anneal_end_frac: float = 0.90       # finish tightening at 90%
    log_std_min: float = -5.0           # floor: sigma >= exp(-5) ~= 0.007

    # Logging / checkpointing
    log_interval: int = 1
    ckpt_interval: int = 10         # save best every N updates
    ckpt_path: str = "ppo_pendulum.pt"           # best by rolling return
    final_ckpt_path: str = "ppo_pendulum_final.pt"  # last weights at end of training


def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env
    return thunk


def _make_vec_env(env_id, seed, num_envs):
    """Create a SyncVectorEnv with SAME_STEP autoreset so that info contains
    'final_obs' on truncation/termination steps. This matches the bootstrap
    semantics of pre-1.0 gym and CleanRL-style PPO code.
    """
    fns = [make_env(env_id, seed, i) for i in range(num_envs)]
    # SAME_STEP autoreset: on the truncation step itself, obs becomes the reset
    # obs; the true final obs lives in info["final_obs"].
    try:
        from gymnasium.vector import AutoresetMode
        return gym.vector.SyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP)
    except (ImportError, TypeError):
        # Older gymnasium: SAME_STEP was the only mode and info has
        # 'final_observation' / '_final_observation' keys.
        return gym.vector.SyncVectorEnv(fns)


def train(cfg: PPOConfig):
    # Seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}, num_envs={cfg.num_envs}, rollout_len={cfg.rollout_len}")

    # Vectorized envs (SAME_STEP autoreset for correct info["final_obs"] handling)
    envs = _make_vec_env(cfg.env_id, cfg.seed, cfg.num_envs)

    # Model & optimizer
    model = ActorCritic(
        OBS_DIM, ACT_DIM, hidden=64,
        init_log_std=cfg.init_log_std,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max_start,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # Storage
    N, T = cfg.num_envs, cfg.rollout_len
    obs_buf = torch.zeros((T, N, OBS_DIM), dtype=torch.float32, device=device)
    act_buf = torch.zeros((T, N, ACT_DIM), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    val_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    # done_buf[t]: 1 if obs_buf[t] is the start of a new episode (i.e., the
    # previous step ended with TRUE termination -- value should not propagate
    # backward across this boundary). In Pendulum no true termination occurs,
    # so this is always 0. Kept for generality.
    done_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    # bootstrap_value[t]: V(s_{t+1}) used in GAE. Normally this is val_buf[t+1],
    # but when the env was truncated/reset between step t and t+1, val_buf[t+1]
    # is V of a FRESH episode, not V(s_truncation+1). We override this with the
    # value of the true final_observation reported in the info dict.
    bootstrap_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    truncated_buf = torch.zeros((T, N), dtype=torch.float32, device=device)

    # Initial obs
    obs, _ = envs.reset(seed=cfg.seed)
    obs = preprocess_obs(obs)
    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(N, dtype=torch.float32, device=device)

    batch_size = N * T
    minibatch_size = cfg.minibatch_size
    assert batch_size % minibatch_size == 0, \
        f"batch_size ({batch_size}) must be divisible by minibatch_size ({minibatch_size})"
    num_minibatches = batch_size // minibatch_size
    num_updates = cfg.total_timesteps // batch_size
    print(f"[init] batch_size={batch_size}, minibatch_size={minibatch_size}, "
          f"num_minibatches={num_minibatches}, num_updates={num_updates}")

    global_step = 0
    total_episodes = 0       # global episode counter
    best_return = -1e9
    start_time = time.time()

    # Episode return tracking (per env)
    ep_returns = np.zeros(N, dtype=np.float64)
    ep_lengths = np.zeros(N, dtype=np.int64)
    recent_returns = []   # last 100 completed episodes

    def log_std_max_at(update_idx):
        """Linear anneal of log_std_max from start to end value over the
        configured fraction of training. Before anneal_start_frac: start value.
        After anneal_end_frac: end value. In between: linearly interpolate.
        """
        progress = (update_idx - 1) / max(num_updates - 1, 1)
        if progress <= cfg.anneal_start_frac:
            return cfg.log_std_max_start
        if progress >= cfg.anneal_end_frac:
            return cfg.log_std_max_end
        span = cfg.anneal_end_frac - cfg.anneal_start_frac
        local = (progress - cfg.anneal_start_frac) / span
        return cfg.log_std_max_start + local * (cfg.log_std_max_end - cfg.log_std_max_start)

    for update in range(1, num_updates + 1):
        # LR anneal
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for pg in optimizer.param_groups:
                pg["lr"] = frac * cfg.lr

        # log_std_max schedule (anneal exploration upper bound)
        model.set_log_std_max(log_std_max_at(update))

        # ---- Rollout ----
        for t in range(T):
            global_step += N
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                action, logp, _, value = model.get_action_and_value(next_obs)
            val_buf[t] = value
            act_buf[t] = action
            logp_buf[t] = logp

            # Clip action for env, but store raw action for PPO ratio.
            action_np = action.cpu().numpy()
            action_clipped = np.clip(action_np, -ACTION_SCALE, ACTION_SCALE)

            obs_raw, reward, terminated, truncated, info = envs.step(action_clipped)
            done = np.logical_or(terminated, truncated)

            # If any env was truncated (not terminated), grab its true terminal
            # obs from info and compute V(s_truncation+1) NOW, before obs_raw
            # has been reset by the vector env's auto-reset.
            #
            # gym.vector returns the final obs of truncated/terminated episodes
            # in info; key layout depends on the gymnasium version:
            #   - SyncVectorEnv (gym >= 0.27): info has "final_observation"
            #     (object array) and "_final_observation" (bool mask)
            #   - older: info["final_observation"] is a list aligned with envs.
            trunc_now = np.logical_and(truncated, np.logical_not(terminated))
            trunc_t = torch.as_tensor(trunc_now.astype(np.float32), device=device)
            truncated_buf[t] = trunc_t

            if trunc_now.any():
                # Build a (N, OBS_DIM) array of final obs (zeros where not truncated).
                final_obs_np = np.zeros((N, OBS_DIM), dtype=np.float32)
                # gym >= 1.1 uses "final_obs"; older uses "final_observation".
                fo = info.get("final_obs", info.get("final_observation", None))
                fo_mask = info.get("_final_obs", info.get("_final_observation", None))
                if fo is not None:
                    if fo_mask is not None:
                        for i in range(N):
                            if fo_mask[i] and fo[i] is not None:
                                final_obs_np[i] = np.asarray(fo[i], dtype=np.float32)
                    else:
                        # No mask: assume fo is aligned with envs and may have None entries
                        for i in range(N):
                            if trunc_now[i] and fo[i] is not None:
                                final_obs_np[i] = np.asarray(fo[i], dtype=np.float32)
                final_obs_pp = preprocess_obs(final_obs_np)
                with torch.no_grad():
                    fo_val = model.get_value(
                        torch.as_tensor(final_obs_pp, dtype=torch.float32, device=device)
                    )
                bootstrap_buf[t] = torch.where(trunc_t.bool(), fo_val, torch.zeros_like(fo_val))

            # Track episode stats
            ep_returns += reward
            ep_lengths += 1
            for i in range(N):
                if done[i]:
                    recent_returns.append(float(ep_returns[i]))
                    total_episodes += 1
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    if len(recent_returns) > 100:
                        recent_returns = recent_returns[-100:]

            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            obs_pp = preprocess_obs(obs_raw)
            next_obs = torch.as_tensor(obs_pp, dtype=torch.float32, device=device)
            # Pendulum has no true termination. Truncation -> we should still
            # bootstrap with V(s'). So next_done = terminated (NOT done).
            next_done = torch.as_tensor(terminated.astype(np.float32), device=device)

        # ---- GAE (truncation-aware) ----
        # For step t, V(s_{t+1}) comes from one of:
        #   (a) bootstrap_buf[t] if env was truncated at step t (true final obs)
        #   (b) val_buf[t+1] otherwise (next obs is real continuation)
        #   (c) next_value (after end of rollout) for t == T-1
        # next_non_terminal masks across TRUE termination only (always 1 here).
        with torch.no_grad():
            next_value = model.get_value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(N, dtype=torch.float32, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - next_done
                    next_v = next_value
                else:
                    next_non_terminal = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                # Override next_v where this step was truncated:
                trunc_mask = truncated_buf[t]
                next_v = trunc_mask * bootstrap_buf[t] + (1.0 - trunc_mask) * next_v
                # When truncated, the GAE recursion should also reset (we're at
                # an episode boundary even if not terminal). Effectively the next
                # transition belongs to a NEW episode, so don't propagate gae.
                non_boundary = (1.0 - trunc_mask) * next_non_terminal
                delta = rew_buf[t] + cfg.gamma * next_v * next_non_terminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * non_boundary * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1, ACT_DIM)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        # ---- PPO update ----
        idx = np.arange(batch_size)
        clip_fracs = []
        approx_kls = []
        pg_losses = []
        v_losses = []
        entropies = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = idx[start:end]

                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                # Per-minibatch advantage normalization
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO clipped policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                if cfg.clip_vloss:
                    v_unclipped = (new_val - b_ret[mb]).pow(2)
                    v_clipped = b_val[mb] + torch.clamp(
                        new_val - b_val[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped_loss = (v_clipped - b_ret[mb]).pow(2)
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * (new_val - b_ret[mb]).pow(2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent_loss.item())

            if cfg.target_kl is not None and np.mean(approx_kls[-num_minibatches:]) > cfg.target_kl:
                break

        # ---- Logging ----
        if update % cfg.log_interval == 0:
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            std_log_raw = float(model.log_std.detach().mean().item())
            std_log_eff = float(model.effective_log_std().detach().mean().item())
            log_std_cap = float(model.log_std_max.item())
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            print(f"[upd {update:4d} / step {global_step:7d} / ep {total_episodes:5d}] "
                  f"ret(100)={mean_ret:8.2f}  "
                  f"pg={np.mean(pg_losses):+.4f}  v={np.mean(v_losses):.3f}  "
                  f"ent={np.mean(entropies):+.3f}  KL={np.mean(approx_kls):.4f}  "
                  f"clipfrac={np.mean(clip_fracs):.3f}  "
                  f"log_std(raw/eff/cap)={std_log_raw:+.2f}/{std_log_eff:+.2f}/{log_std_cap:+.2f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  sps={sps}")

        # ---- Checkpoint: best by rolling training return ----
        if update % cfg.ckpt_interval == 0 or update == num_updates:
            if recent_returns:
                rolling = float(np.mean(recent_returns))
                if rolling > best_return:
                    best_return = rolling
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "global_step": global_step,
                        "total_episodes": total_episodes,
                        "rolling_return": rolling,
                    }, cfg.ckpt_path)
                    print(f"  >>> saved BEST to {cfg.ckpt_path} "
                          f"(rolling return = {rolling:.2f})")

    # ---- Final checkpoint (last weights) ----
    final_rolling = float(np.mean(recent_returns)) if recent_returns else float("nan")
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "global_step": global_step,
        "total_episodes": total_episodes,
        "rolling_return": final_rolling,
    }, cfg.final_ckpt_path)
    print(f"  >>> saved FINAL to {cfg.final_ckpt_path} "
          f"(rolling return = {final_rolling:.2f})")

    envs.close()
    print(f"\n[done] best rolling return = {best_return:.2f}, "
          f"final rolling return = {final_rolling:.2f}")


if __name__ == "__main__":
    # Defaults below are placeholders; the dataclass defines the real defaults.
    # Override on CLI as needed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-len", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="ppo_pendulum.pt")
    parser.add_argument("--final-ckpt", type=str, default="ppo_pendulum_final.pt")
    parser.add_argument("--log-std-end", type=float, default=-0.5,
                        help="Target log_std_max at end of annealing (default exp(-2.3)~=0.10)")
    args = parser.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        ckpt_path=args.ckpt,
        final_ckpt_path=args.final_ckpt,
        log_std_max_end=args.log_std_end,
    )
    train(cfg)
