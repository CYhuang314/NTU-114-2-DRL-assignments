"""
train.py

Main training loop for SAC on DMC humanoid (walk / run).

Usage
-----
Fresh training run (humanoid-walk, 3M steps, seed 0):
    python train.py --task humanoid-walk --seed 0 --total_steps 3000000

Resume from a saved full-state checkpoint:
    python train.py --task humanoid-walk --checkpoint_path runs/walk_s0/latest.pt

Warm-start humanoid-run from a trained walk actor (Q3 transfer plan):
    python train.py --task humanoid-run --init_actor_path runs/walk_s0/best_actor.pt

Flags
-----
    --task              "humanoid-walk" or "humanoid-run"
    --seed              int seed for numpy/torch/env (default 0)
    --total_steps       total environment steps to train for
    --device            "cuda" or "cpu" (default "cuda" if available)
    --log_dir           output directory for logs + checkpoints
    --checkpoint_path   resume full state (agent + buffer + RNG + step)
    --init_actor_path   warm-start actor weights ONLY (fresh critic / buffer)

Outputs (in --log_dir)
---------------------
    metrics.jsonl   one JSON object per logged event (episode | train | eval)
    latest.pt       full-state checkpoint, overwritten periodically
    best_actor.pt   actor state_dict, overwritten when eval mean improves
                    (this is what student_agent.py loads at evaluation time)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from dmc import make_dmc_env
from env_wrappers import make_float32_env
from replay_buffer import ReplayBuffer
from sac_agent import SACAgent


# ============================================================================
# Default hyperparameter config. Edit here for anything not on the CLI.
# ============================================================================
DEFAULT_CFG: Dict[str, Any] = {
    # --- Environment dims (DMC humanoid; do not touch unless task changes) --
    "obs_dim": 67,
    "action_dim": 21,

    # --- SAC core -----------------------------------------------------------
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,                 # INITIAL value of α; auto-tuned thereafter
    "alpha_lr": 3e-4,             # LR for log_alpha (auto-α tuning)
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_sizes": (256, 256),
    # target_entropy: omitted -> defaults to -action_dim = -21 inside SACAgent

    # --- Architectural flags (parallel A/B for walk experiment) ------------
    # Set both to False to reproduce the prior vanilla-SAC arm; flip both to
    # True for the obs-norm + critic-LayerNorm arm. They are intended to be
    # used together (see analysis: obs-norm targets the actor's bimodal
    # collapse from input-distribution drift; critic LayerNorm targets the
    # observed late-stage Q-divergence). Architecture is byte-identical to
    # the prior version when both are False, so old `best_actor.pt` files
    # remain loadable via the unchanged `student_agent.py` interface.
    "use_obs_norm": True,
    "use_layer_norm": True,

    # --- Replay buffer ------------------------------------------------------
    "buffer_capacity": 1_000_000,
    "batch_size": 256,

    # --- Loop schedule ------------------------------------------------------
    "warmup_steps": 10_000,       # random-action steps before any update()
    "updates_per_step": 1,        # UTD ratio; bump to 2-4 if walk plateaus

    # --- LR decay (actor + critic only; alpha LR held constant) ------------
    # Linear decay of actor_lr and critic_lr from their initial value down to
    # `lr_decay_final_factor * initial`, starting at `lr_decay_start_frac` of
    # total_steps and finishing at total_steps. Set lr_decay_final_factor=1.0
    # to disable.
    #
    # Rationale: late in training (last ~40% of steps), Q1 max spikes grow
    # (critic step too large vs Polyak bandwidth) and eval std blows up
    # (actor step too large for a near-converged policy). Slowing both
    # optimizers in the final stretch trades final mean for variance
    # reduction, which directly improves the (mean - std) grading metric.
    # See log analysis: WS5 Q1 max climbed 32 -> 138 -> 229 across 0.5M ->
    # 3.5M -> 4M; eval std avg climbed 26 -> 108 in the same window.
    # Alpha LR is intentionally NOT decayed: alpha is driven by the entropy
    # constraint (target_entropy = -action_dim) and slowing its controller
    # would let the policy peak more freely, working against stability.
    "lr_decay_start_frac": 0.6,   # start decay at 60% of total_steps
    "lr_decay_final_factor": 1.0 / 3.0,  # end at 1/3 of initial LR (3e-4 -> 1e-4)

    # --- Logging / eval / checkpoint cadence -------------------------------
    "train_log_every_updates": 1_000,
    "eval_every_steps": 50_000,
    # eval_episodes bumped 10 -> 30: the (mean - std) selection metric needs
    # a precise std estimate, and 10 episodes with ~20% bimodal-collapse
    # probability has ~10% chance of catching >= 3 collapses, which inflates
    # std and corrupts checkpoint selection. 30 episodes ~ tripled precision
    # of std estimate. Marginal cost: ~15s/eval * 100 evals ~ 25 min over a
    # 5M run, negligible vs total training time.
    "eval_episodes": 30,
    "checkpoint_every_steps": 100_000,
}


# ============================================================================
# Utilities
# ============================================================================
def set_seed(seed: int) -> None:
    """Seed numpy + torch (CPU & CUDA) + python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    """Capture numpy + torch RNG state for checkpointing."""
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from checkpoint."""
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class JsonlLogger:
    """Append-only JSONL writer. Each call to log() writes one line."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append mode survives crashes mid-run and supports
        # resume-then-continue without truncation.
        self._f = open(path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._f.close()


# ============================================================================
# Build helpers
# ============================================================================
def build_env(task: str, seed: int):
    """Construct a single training env, wrapped to float32."""
    raw = make_dmc_env(task, seed, flatten=True, use_pixels=False)
    return make_float32_env(raw)


def build_eval_env(task: str, seed: int):
    """Separate env instance for periodic eval. Different seed namespace."""
    raw = make_dmc_env(task, seed, flatten=True, use_pixels=False)
    return make_float32_env(raw)


def build_agent(cfg: Dict[str, Any], device: torch.device) -> SACAgent:
    agent_cfg = {
        "obs_dim": cfg["obs_dim"],
        "action_dim": cfg["action_dim"],
        "device": device,
        "gamma": cfg["gamma"],
        "tau": cfg["tau"],
        "alpha": cfg["alpha"],
        "alpha_lr": cfg["alpha_lr"],
        "actor_lr": cfg["actor_lr"],
        "critic_lr": cfg["critic_lr"],
        "hidden_sizes": cfg["hidden_sizes"],
        "use_obs_norm": cfg["use_obs_norm"],
        "use_layer_norm": cfg["use_layer_norm"],
    }
    return SACAgent(agent_cfg)


def linear_lr_factor(step: int, total_steps: int,
                     start_frac: float, final_factor: float) -> float:
    """Compute the LR scaling factor at `step` for a linear decay schedule.

    Decay is identity (factor = 1.0) until step == start_frac * total_steps,
    then linearly interpolates down to `final_factor` at step == total_steps.
    Returns final_factor for step beyond total_steps (defensive; shouldn't
    happen in normal use).

    Set final_factor = 1.0 to disable decay (factor will be 1.0 throughout).
    """
    if final_factor >= 1.0:
        return 1.0
    start_step = int(start_frac * total_steps)
    if step <= start_step:
        return 1.0
    if step >= total_steps:
        return float(final_factor)
    # Linear from 1.0 at start_step down to final_factor at total_steps.
    frac = (step - start_step) / max(total_steps - start_step, 1)
    return 1.0 + (final_factor - 1.0) * frac


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set lr for all param groups in `optimizer` (in-place)."""
    for g in optimizer.param_groups:
        g["lr"] = lr


def build_buffer(cfg: Dict[str, Any], device: torch.device) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=cfg["buffer_capacity"],
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        device=device,
    )


# ============================================================================
# Evaluation
# ============================================================================
def evaluate(
    agent: SACAgent,
    eval_env,
    n_episodes: int,
    eval_seed_base: int,
) -> Dict[str, float]:
    """Run n_episodes deterministic episodes; return mean / std / min / max."""
    returns = []
    for i in range(n_episodes):
        obs, _ = eval_env.reset(seed=eval_seed_base + i)
        ep_return = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "n_episodes": int(n_episodes),
    }


# ============================================================================
# Main training loop
# ============================================================================
def train(args: argparse.Namespace) -> None:
    cfg = dict(DEFAULT_CFG)  # shallow copy; never mutate the global default

    # CLI overrides for architectural flags. None means "user didn't pass it,
    # keep DEFAULT_CFG"; True/False explicitly override. We apply BEFORE
    # build_agent and BEFORE config.json is written so the persisted config
    # reflects the values actually used at runtime.
    if args.use_obs_norm is not None:
        cfg["use_obs_norm"] = bool(args.use_obs_norm)
    if args.use_layer_norm is not None:
        cfg["use_layer_norm"] = bool(args.use_layer_norm)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_dir / "metrics.jsonl")
    latest_ckpt_path = log_dir / "latest.pt"
    best_actor_path = log_dir / "best_actor.pt"

    # --- Seed BEFORE building envs / agent. We may overwrite RNG state below
    # if resuming, but networks have already been initialized with this seed.
    set_seed(args.seed)

    device = torch.device(args.device)

    # --- Construct env, agent, buffer
    env = build_env(args.task, seed=args.seed)
    eval_env = build_eval_env(args.task, seed=args.seed + 100_000)
    agent = build_agent(cfg, device)
    buffer = build_buffer(cfg, device)

    # --- Resume / warm-start handling
    start_step = 0
    # `best_eval_score` is the highest eval (mean - std) seen so far. We
    # select on (mean - std) rather than mean alone because the grading
    # metric is (mean - std) and the eval distribution is heavy-tailed (a
    # ~20% bimodal-collapse rate inflates std and hides truly stable
    # snapshots if we naively take the highest-mean checkpoint).
    best_eval_score = -float("inf")

    if args.checkpoint_path:
        # Full-state resume: agent + buffer + RNG + step counter.
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        buffer.load_state_dict(ckpt["buffer"])
        set_rng_state(ckpt["rng"])
        start_step = int(ckpt["step"])
        # Backward compat: older checkpoints stored `best_eval_mean` (mean-only
        # selection). New runs store `best_eval_score` (mean-std selection).
        # Resuming from an old ckpt: take whichever key exists; if it's the
        # old `best_eval_mean`, treat it as if it were a score floor of -inf
        # since the values aren't comparable across selection metrics — that
        # way the first new eval will overwrite best_actor.pt and we re-anchor
        # on the new metric. Print a warning so the user knows what happened.
        if "best_eval_score" in ckpt:
            best_eval_score = float(ckpt["best_eval_score"])
        elif "best_eval_mean" in ckpt:
            print(f"[resume] WARNING: old checkpoint has best_eval_mean="
                  f"{ckpt['best_eval_mean']:.2f} (mean-only metric). "
                  f"New runs select on (mean - std); resetting best_eval_score "
                  f"to -inf so the first new eval re-anchors best_actor.pt.")
            best_eval_score = -float("inf")
        print(f"[resume] loaded {args.checkpoint_path} at step {start_step:,}, "
              f"best_eval_score={best_eval_score:.2f}")
    elif args.init_actor_path:
        # Warm-start: actor weights ONLY. Fresh critic, target, buffer.
        actor_state = torch.load(args.init_actor_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(actor_state)
        print(f"[warm-start] loaded actor weights from {args.init_actor_path} "
              f"(critic / buffer fresh)")

    # --- Persist run config for reproducibility
    with open(log_dir / "config.json", "w") as f:
        run_cfg = {
            "task": args.task,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "device": str(device),
            "checkpoint_path": args.checkpoint_path,
            "init_actor_path": args.init_actor_path,
            **{k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        }
        json.dump(run_cfg, f, indent=2)

    # --- Training state
    obs, _ = env.reset(seed=args.seed)
    # Feed the very first rollout observation into ObsNorm. Subsequent
    # observations (next_obs from env.step, and reset obs at episode
    # boundaries) are fed inside the loop below. No-op when use_obs_norm
    # is False. See SACAgent.obs_norm_update / networks.ObsNorm for why
    # this single-obs-per-env-step pattern is correct (vs. updating on
    # sampled training batches, which would track stale buffer stats).
    agent.obs_norm_update(obs)
    episode_return = 0.0
    episode_length = 0
    episode_idx = 0
    update_count = 0
    train_log_accum: Dict[str, float] = {}  # rolling sum for periodic update logging
    train_log_n = 0

    # Initial LRs captured here so we can reapply scaled values without
    # accumulating floating-point drift across many small updates.
    initial_actor_lr = float(cfg["actor_lr"])
    initial_critic_lr = float(cfg["critic_lr"])
    last_lr_factor = 1.0  # cache to skip apply_lr() when factor hasn't changed

    t0 = time.time()
    last_print_step = start_step
    last_print_time = t0

    print(f"[train] task={args.task} seed={args.seed} device={device} "
          f"start_step={start_step:,} total_steps={args.total_steps:,}")
    print(f"[train] warmup_steps={cfg['warmup_steps']:,} "
          f"buffer_capacity={cfg['buffer_capacity']:,} "
          f"batch_size={cfg['batch_size']}")
    print(f"[train] arch flags: use_obs_norm={cfg['use_obs_norm']} "
          f"use_layer_norm={cfg['use_layer_norm']}")
    print(f"[train] lr_decay: start_frac={cfg['lr_decay_start_frac']} "
          f"final_factor={cfg['lr_decay_final_factor']:.4f} "
          f"(actor_lr {initial_actor_lr:.2e} -> "
          f"{initial_actor_lr * cfg['lr_decay_final_factor']:.2e}, "
          f"alpha_lr held constant at {cfg['alpha_lr']:.2e})")

    try:
        for step in range(start_step + 1, args.total_steps + 1):
            # --- Action selection: random during warmup, policy after.
            if step <= cfg["warmup_steps"]:
                action = env.action_space.sample()  # float32 by wrapper contract
            else:
                action = agent.select_action(obs, deterministic=False)

            # --- Env step
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # --- ObsNorm rollout-side update (no-op when use_obs_norm=False).
            # Done HERE — after env.step, before storing — so that:
            #   1. Warmup random-action observations are included (the stats
            #      need to be populated before the first SAC update at
            #      step warmup_steps + 1, which forwards through the
            #      normalizer).
            #   2. The transition stored in the buffer has its `next_obs`
            #      reflected in the running stats from this step onward.
            #   3. We update on raw rollout observations only (single obs,
            #      sequential), never on sampled buffer batches. See
            #      networks.ObsNorm class docstring for why this matters.
            agent.obs_norm_update(next_obs)

            # --- Store transition. Bootstrap mask uses `terminated` only;
            #     `truncated` is stored separately for completeness / logging.
            buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            # --- Episode boundary
            if terminated or truncated:
                logger.log({
                    "event": "episode",
                    "step": step,
                    "episode": episode_idx,
                    "return": episode_return,
                    "length": episode_length,
                })
                # Episode-level stdout: keep tight, one line per episode.
                print(f"[ep {episode_idx:5d} | step {step:>9,}] "
                      f"return={episode_return:7.2f} length={episode_length}")
                episode_idx += 1
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()
                # Episode-start observations are statistically distinct from
                # mid-episode `next_obs` returns (they're drawn from the env's
                # initial-state distribution, not from policy-driven transitions).
                # Feed them into ObsNorm too so the running stats reflect both.
                # No-op when use_obs_norm=False.
                agent.obs_norm_update(obs)

            # --- SAC updates: only after warmup, exactly UTD per env step.
            if step > cfg["warmup_steps"]:
                # Update LR per current decay schedule. We compute the factor
                # once per env-step (cheap), and only call apply_lr() when it
                # actually changes — most consecutive steps share the same
                # factor up to floating point, so this is a no-op the vast
                # majority of the time.
                lr_factor = linear_lr_factor(
                    step=step,
                    total_steps=args.total_steps,
                    start_frac=float(cfg["lr_decay_start_frac"]),
                    final_factor=float(cfg["lr_decay_final_factor"]),
                )
                if lr_factor != last_lr_factor:
                    apply_lr(agent.actor_optimizer, initial_actor_lr * lr_factor)
                    apply_lr(agent.critic_optimizer, initial_critic_lr * lr_factor)
                    last_lr_factor = lr_factor

                for _ in range(cfg["updates_per_step"]):
                    batch = buffer.sample(cfg["batch_size"])
                    info = agent.update(batch)
                    update_count += 1

                    # Accumulate for periodic logging.
                    for k, v in info.items():
                        train_log_accum[k] = train_log_accum.get(k, 0.0) + v
                    train_log_n += 1

                    if update_count % cfg["train_log_every_updates"] == 0:
                        avg = {k: v / train_log_n for k, v in train_log_accum.items()}
                        # Tag the current LR factor in train logs so post-hoc
                        # analysis can attribute Q/loss/entropy changes to
                        # the schedule. Stored as a scalar; downstream parsers
                        # that don't know about it will just ignore the field.
                        record = {"event": "train", "step": step,
                                  "update": update_count,
                                  "lr_factor": lr_factor,
                                  **avg}
                        logger.log(record)

                        # Throughput print: env-steps/sec since last print.
                        now = time.time()
                        sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
                        last_print_step = step
                        last_print_time = now
                        print(f"[upd {update_count:>8,} | step {step:>9,}] "
                              f"critic={avg['critic_loss']:7.3f} "
                              f"actor={avg['actor_loss']:7.3f} "
                              f"q1={avg['q1_mean']:7.2f} "
                              f"ent={avg['entropy']:6.3f} "
                              f"alpha={avg['alpha']:6.4f} "
                              f"lrf={lr_factor:5.3f} "
                              f"sps={sps:5.0f}")

                        train_log_accum.clear()
                        train_log_n = 0

            # --- Periodic eval
            if step % cfg["eval_every_steps"] == 0 and step > cfg["warmup_steps"]:
                eval_stats = evaluate(
                    agent=agent,
                    eval_env=eval_env,
                    n_episodes=cfg["eval_episodes"],
                    eval_seed_base=args.seed + 1_000_000 + step,
                )
                # Score that we both grade on AND select on. Computed once,
                # logged alongside the raw stats so downstream analysis
                # doesn't need to recompute it.
                eval_score = eval_stats["mean_return"] - eval_stats["std_return"]
                logger.log({"event": "eval", "step": step,
                            "score": eval_score, **eval_stats})
                print(f"[eval | step {step:>9,}] "
                      f"mean={eval_stats['mean_return']:7.2f} "
                      f"std={eval_stats['std_return']:6.2f} "
                      f"min={eval_stats['min_return']:7.2f} "
                      f"max={eval_stats['max_return']:7.2f} "
                      f"score(m-s)={eval_score:7.2f}")

                # Best-actor checkpoint: select on (mean - std), not mean.
                # See module docstring at best_eval_score initialization.
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    torch.save(agent.actor_state_dict(), best_actor_path)
                    print(f"[eval] new best score(m-s)={best_eval_score:.2f} "
                          f"-> saved {best_actor_path}")

            # --- Periodic full checkpoint
            if step % cfg["checkpoint_every_steps"] == 0:
                save_full_checkpoint(
                    path=latest_ckpt_path,
                    agent=agent,
                    buffer=buffer,
                    step=step,
                    best_eval_score=best_eval_score,
                )
                print(f"[ckpt | step {step:>9,}] saved {latest_ckpt_path}")

    except KeyboardInterrupt:
        print("\n[interrupt] flushing final checkpoint before exit...")
        save_full_checkpoint(
            path=latest_ckpt_path,
            agent=agent,
            buffer=buffer,
            step=step,
            best_eval_score=best_eval_score,
        )
        print(f"[interrupt] saved {latest_ckpt_path} at step {step:,}")
        logger.close()
        env.close()
        eval_env.close()
        sys.exit(0)

    # --- Final checkpoint at clean termination
    save_full_checkpoint(
        path=latest_ckpt_path,
        agent=agent,
        buffer=buffer,
        step=args.total_steps,
        best_eval_score=best_eval_score,
    )
    # Also save the final actor weights, regardless of whether it beat
    # best_eval_score. This gives the user a second candidate at submission
    # time: best_actor.pt is the highest-scoring snapshot during training;
    # final_actor.pt is wherever training ended up. With LR decay enabled,
    # the final policy often has lower variance than mid-training peaks
    # even at slightly lower mean, which can win on (mean - std).
    final_actor_path = log_dir / "final_actor.pt"
    torch.save(agent.actor_state_dict(), final_actor_path)
    print(f"[done] saved {final_actor_path}")
    elapsed = time.time() - t0
    print(f"[done] total_steps={args.total_steps:,} elapsed={elapsed/3600:.2f}h "
          f"best_eval_score={best_eval_score:.2f}")
    logger.close()
    env.close()
    eval_env.close()


# ============================================================================
# Checkpoint I/O
# ============================================================================
def save_full_checkpoint(
    path: Path,
    agent: SACAgent,
    buffer: ReplayBuffer,
    step: int,
    best_eval_score: float,
) -> None:
    """Atomic-ish: write to a tmp file then rename. Avoids corrupted ckpts
    if the process dies mid-write.

    Note: stores `best_eval_score` (selection metric: mean - std). Older
    checkpoints stored `best_eval_mean` (mean only). The resume path in
    train() handles both keys; new writes always use the new key.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "agent": agent.state_dict(),
            "buffer": buffer.state_dict(),  # views, not copies; written directly to disk
            "rng": get_rng_state(),
            "step": int(step),
            "best_eval_score": float(best_eval_score),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC training for DMC humanoid (walk/run)")
    p.add_argument("--task", type=str, default="humanoid-walk",
                   choices=["humanoid-walk", "humanoid-run"],
                   help="DMC task name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=3_000_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_dir", type=str, default="runs/default",
                   help="Output directory for metrics.jsonl, checkpoints, config.json")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Resume full state (agent + buffer + RNG + step) from this path")
    p.add_argument("--init_actor_path", type=str, default=None,
                   help="Warm-start actor weights ONLY (fresh critic / buffer). "
                        "Mutually exclusive with --checkpoint_path.")

    # Architectural flags. BooleanOptionalAction gives us `--use_obs_norm`
    # AND `--no-use_obs_norm` for explicit enable/disable; default=None
    # means "don't override DEFAULT_CFG", so omitting both flags keeps
    # the cfg defaults intact (currently False / False).
    p.add_argument("--use_obs_norm", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Enable ObsNorm input normalizer (shared by actor + "
                        "both critics + target critic). Welford updates "
                        "happen rollout-side only. Default: use DEFAULT_CFG.")
    p.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Enable LayerNorm in critic hidden layers (TD7 style). "
                        "Actor unaffected. Default: use DEFAULT_CFG.")

    args = p.parse_args()

    if args.checkpoint_path and args.init_actor_path:
        p.error("--checkpoint_path and --init_actor_path are mutually exclusive")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
