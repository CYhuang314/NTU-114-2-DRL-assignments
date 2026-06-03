#!/usr/bin/env python3
"""
Train a Double Dueling DQN with PER and n-step returns
on LevDoom Seek and Slay (curriculum across levels 0-4).

Usage:
    # Fresh training
    python train.py

    # Resume from checkpoint
    python train.py --resume checkpoints/weights.pth --resume-step 2000000 --steps 1000000

    # Override learning rate
    python train.py --lr 3e-5
"""

import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path

import gymnasium
import levdoom  # noqa: F401 – registers envs

# ─────────────────────── Hyperparameters ───────────────────────

TOTAL_STEPS       = 3_000_000
LEARNING_RATE     = 5e-5       # base LR (boosted during curriculum transitions)
LR_BOOST          = 1e-4       # boosted LR during re-annealing windows
GAMMA             = 0.99
BATCH_SIZE        = 64
REPLAY_SIZE       = 100_000
N_STEP            = 3
TARGET_UPDATE     = 1000       # steps between hard target updates
EPS_END           = 0.005
LEARNING_STARTS   = 10_000
TRAIN_FREQ        = 4          # learn every N env steps
FRAME_STACK       = 4
IMG_SIZE          = 84

# PER
PER_ALPHA         = 0.6
PER_BETA_START    = 0.4
PER_BETA_END      = 1.0

# Seed strategy: 75% random seeds, 25% eval seed (1234)
EVAL_SEED         = 1234
EVAL_SEED_RATIO   = 0.25       # probability of using eval seed per episode

# Curriculum: (step_threshold, list_of_(env_id, weight) pairs)
CURRICULUM = [
    (0,        [("SeekAndSlayLevel0-v0", 1.0),
                ("SeekAndSlayLevel1_6-v0", 1.0)]),
    (400_000,  [("SeekAndSlayLevel0-v0", 0.8),
                ("SeekAndSlayLevel1_6-v0", 0.8),
                ("SeekAndSlayLevel3_1-v0", 2.0)]),
    (800_000,  [("SeekAndSlayLevel0-v0", 0.6),
                ("SeekAndSlayLevel1_6-v0", 0.6),
                ("SeekAndSlayLevel3_1-v0", 1.0),
                ("SeekAndSlayLevel2_3-v0", 2.0)]),
    (1_200_000,[("SeekAndSlayLevel0-v0", 0.5),
                ("SeekAndSlayLevel1_6-v0", 0.5),
                ("SeekAndSlayLevel3_1-v0", 1.0),
                ("SeekAndSlayLevel2_3-v0", 1.0),
                ("SeekAndSlayLevel4-v0", 2.0)]),
    (2_000_000,[("SeekAndSlayLevel0-v0", 1.0),
                ("SeekAndSlayLevel1_6-v0", 1.0),
                ("SeekAndSlayLevel3_1-v0", 1.0),
                ("SeekAndSlayLevel2_3-v0", 1.0),
                ("SeekAndSlayLevel4-v0", 1.0)]),
]

# I/O
SAVE_DIR          = Path("checkpoints")
SAVE_INTERVAL     = 10_000     # steps
LOG_INTERVAL      = 1_000      # steps
SEED              = 42

# ─────────────────────── Preprocessing ─────────────────────────

import cv2

def preprocess(obs: np.ndarray) -> np.ndarray:
    """Convert RGB (240,320,3) -> grayscale float32 (84,84)."""
    if obs.ndim == 3 and obs.shape[-1] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs.squeeze()
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """Maintains a deque of the last `k` preprocessed frames."""
    def __init__(self, k=FRAME_STACK):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        frame = preprocess(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get()

    def step(self, obs):
        self.frames.append(preprocess(obs))
        return self._get()

    def _get(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)  # (k, 84, 84)


# ─────────────────────── Dueling DQN Network ───────────────────

class DuelingDQN(nn.Module):
    def __init__(self, in_channels=FRAME_STACK, n_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # After convolutions: 64 * 7 * 7 = 3136
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feat = self.features(x)
        val = self.value_stream(feat)                       # (B, 1)
        adv = self.advantage_stream(feat)                   # (B, n_actions)
        q = val + adv - adv.mean(dim=1, keepdim=True)      # (B, n_actions)
        return q


# ─────────────────────── Prioritized Replay Buffer ─────────────

class SumTree:
    """Binary sum-tree for O(log n) priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self.min_priority = 1e-6

    def add(self, transition):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)

            # Retry until we get a valid (non-None) entry
            data = None
            for _attempt in range(20):
                s = random.uniform(lo, hi)
                idx, prio, data = self.tree.get(s)
                if data is not None:
                    break
                # Widen search range on retry
                lo = 0
                hi = self.tree.total()

            if data is None:
                # Last resort: pick a random valid index
                valid_start = self.tree.capacity - 1
                valid_end = valid_start + self.tree.size
                rand_idx = random.randint(valid_start, valid_end - 1)
                idx = rand_idx
                prio = self.tree.tree[idx]
                data = self.tree.data[idx - self.tree.capacity + 1]

            batch.append(data)
            indices.append(idx)
            priorities.append(max(prio, self.min_priority))

        priorities = np.array(priorities, dtype=np.float64) + self.min_priority
        probs = priorities / max(self.tree.total(), self.min_priority)
        weights = (self.tree.size * probs) ** (-beta)
        weights /= max(weights.max(), 1e-8)

        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            prio = (abs(td) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)

    def __len__(self):
        return self.tree.size


# ─────────────────────── N-step Return Helper ──────────────────

class NStepBuffer:
    """Accumulates n-step transitions before pushing to main replay."""
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def reset(self):
        self.buffer.clear()

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self):
        return len(self.buffer) == self.n

    def get(self):
        """Return the n-step transition (s_0, a_0, R_n, s_n, done_n)."""
        R = 0.0
        for i in range(self.n):
            s, a, r, ns, d = self.buffer[i]
            R += (self.gamma ** i) * r
            if d:
                return self.buffer[0][0], self.buffer[0][1], R, ns, True
        last_ns = self.buffer[-1][3]
        last_done = self.buffer[-1][4]
        return self.buffer[0][0], self.buffer[0][1], R, last_ns, last_done

    def flush(self):
        """Flush remaining transitions at episode end."""
        transitions = []
        while len(self.buffer) > 0:
            R = 0.0
            for i, (s, a, r, ns, d) in enumerate(self.buffer):
                R += (self.gamma ** i) * r
                if d:
                    transitions.append((self.buffer[0][0], self.buffer[0][1], R, ns, True))
                    break
            else:
                last_ns = self.buffer[-1][3]
                last_done = self.buffer[-1][4]
                transitions.append((self.buffer[0][0], self.buffer[0][1], R, last_ns, last_done))
            self.buffer.popleft()
        return transitions


# ─────────────────────── Curriculum Manager ────────────────────

class CurriculumManager:
    def __init__(self, curriculum):
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_stage = 0
        self.envs = {}  # env_id -> gymnasium.Env

    def get_env_weights(self, step):
        """Return the active (env_id, weight) pairs for the current step."""
        for i, (threshold, _) in enumerate(self.curriculum):
            if step >= threshold:
                self.current_stage = i
        return self.curriculum[self.current_stage][1]

    def sample_env_id(self, step):
        """Sample an environment ID according to curriculum weights."""
        pairs = self.get_env_weights(step)
        ids, weights = zip(*pairs)
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        return np.random.choice(ids, p=weights)

    def get_env(self, env_id):
        """Lazily create and cache environments."""
        if env_id not in self.envs:
            self.envs[env_id] = gymnasium.make(env_id, render=False)
        return self.envs[env_id]

    def close_all(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()


# ─────────────────────── Epsilon & LR Schedules ────────────────

# Epsilon re-annealing: bump eps when new levels are introduced, decay again.
# LR boost: use higher LR during the same re-annealing windows.
EPS_REANNEALING = {
    # step_threshold: (eps_reset_to, decay_over_n_steps)
    0:         (1.0,  300_000),   # initial: 1.0 -> EPS_END over 300K
    400_000:   (0.3,  100_000),   # L2 introduced
    800_000:   (0.3,  100_000),   # L3 introduced
    1_200_000: (0.2,  100_000),   # L4 introduced
}

def epsilon(step):
    """Curriculum-aware epsilon with re-annealing at stage transitions."""
    active_threshold = 0
    for threshold in sorted(EPS_REANNEALING.keys()):
        if step >= threshold:
            active_threshold = threshold

    eps_start, decay_steps = EPS_REANNEALING[active_threshold]
    steps_since = step - active_threshold
    frac = min(1.0, steps_since / decay_steps)
    return eps_start + frac * (EPS_END - eps_start)


def get_lr_for_step(step, base_lr):
    """Return boosted LR during re-annealing windows, base LR otherwise.

    During the first 100K steps after a curriculum transition (400K, 800K, 1.2M),
    the LR is boosted to LR_BOOST so the agent can adapt faster to new levels.
    The initial training phase (0-300K) always uses base_lr since it starts
    from scratch and the base decay window is already 300K steps.
    """
    for threshold in sorted(EPS_REANNEALING.keys(), reverse=True):
        if threshold == 0:
            break  # initial phase uses base_lr
        if step >= threshold:
            _, decay_steps = EPS_REANNEALING[threshold]
            steps_since = step - threshold
            if steps_since < decay_steps:
                # Inside a re-annealing window: use boosted LR
                return LR_BOOST
            break
    return base_lr


def pick_episode_seed():
    """Return a seed for env.reset(): eval seed 25% of the time, None otherwise."""
    if random.random() < EVAL_SEED_RATIO:
        return EVAL_SEED
    return None


# ─────────────────────── Main Training Loop ────────────────────

def train(resume_path=None, resume_step=0, total_steps=None, lr=None):
    """
    Main training loop.

    Args:
        resume_path:  Path to a .pth weights file to resume training from.
        resume_step:  Global step number to resume from. This offsets the
                      epsilon schedule and curriculum so they pick up where
                      the previous run left off.
        total_steps:  How many *new* steps to train (default: TOTAL_STEPS).
        lr:           Override base LEARNING_RATE if provided.
    """
    actual_total_steps = total_steps if total_steps is not None else TOTAL_STEPS
    actual_lr = lr if lr is not None else LEARNING_RATE

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Detect action space size from env
    tmp_env = gymnasium.make("SeekAndSlayLevel0-v0", render=False)
    n_actions = tmp_env.action_space.n
    tmp_env.close()
    print(f"Action space: {n_actions}")

    # Networks
    policy_net = DuelingDQN(FRAME_STACK, n_actions).to(device)
    target_net = DuelingDQN(FRAME_STACK, n_actions).to(device)

    # Resume from checkpoint if provided
    if resume_path is not None:
        resume_path = Path(resume_path)
        print(f"Resuming from: {resume_path} at global step {resume_step}")
        state_dict = torch.load(resume_path, map_location=device, weights_only=True)
        policy_net.load_state_dict(state_dict)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=actual_lr)

    final_global_step = resume_step + actual_total_steps
    print(f"Training for {actual_total_steps} new steps "
          f"(global {resume_step} -> {final_global_step}), base_lr={actual_lr}")

    # Replay buffer
    replay = PrioritizedReplayBuffer(REPLAY_SIZE)

    # Curriculum
    curriculum = CurriculumManager(CURRICULUM)

    # Logging
    log_data = {
        "episode_rewards": [],
        "episode_kills": [],
        "episode_lengths": [],
        "losses": [],
        "env_ids": [],
    }

    # ── Episode state ──
    frame_stack = FrameStack(FRAME_STACK)
    nstep_buf = NStepBuffer(N_STEP, GAMMA)

    # Start first episode with mixed seed strategy
    env_id = curriculum.sample_env_id(resume_step)
    env = curriculum.get_env(env_id)
    ep_seed = pick_episode_seed()
    obs, info = env.reset(seed=ep_seed)
    state = frame_stack.reset(obs)
    nstep_buf.reset()

    ep_reward = 0.0
    ep_kills = 0
    ep_len = 0
    ep_count = 0
    total_loss = 0.0
    loss_count = 0
    prev_kills = 0.0
    prev_health = float(info.get("health", 100))

    start_time = time.time()

    for step in range(1, actual_total_steps + 1):
        global_step = step + resume_step

        # ── Select action ──
        eps = epsilon(global_step)
        if random.random() < eps:
            action = random.randrange(n_actions)
        else:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q = policy_net(s_t)
                action = q.argmax(dim=1).item()

        # ── Step environment ──
        next_obs, raw_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Shaped reward from info ──
        # The env returns reward=0 always. We craft our own from info.
        current_kills = float(info.get("kills", 0))
        current_health = float(info.get("health", 0))
        current_ammo = float(info.get("ammo", 0))
        movement = float(info.get("movement", 0))

        # Guard against nan (movement is nan on first frame)
        if np.isnan(movement):
            movement = 0.0

        # Kill reward (dominant signal)
        kill_delta = current_kills - prev_kills
        reward = kill_delta * 1.0

        # Small movement bonus to encourage exploration
        reward += 0.001 * movement

        # Small penalty for losing health (encourages dodging)
        health_delta = current_health - prev_health
        if health_delta < 0:
            reward += 0.01 * health_delta  # negative

        # Update trackers
        prev_kills = current_kills
        prev_health = current_health

        next_state = frame_stack.step(next_obs)

        # ── N-step accumulation ──
        nstep_buf.add(state, action, reward, next_state, done)
        if nstep_buf.is_ready():
            transition = nstep_buf.get()
            replay.add(transition)

        state = next_state
        ep_reward += reward
        ep_kills = current_kills  # cumulative kill count from env
        ep_len += 1

        # ── Episode done ──
        if done:
            # Flush remaining n-step transitions
            for trans in nstep_buf.flush():
                replay.add(trans)

            ep_count += 1
            log_data["episode_rewards"].append(ep_reward)
            log_data["episode_kills"].append(ep_kills)
            log_data["episode_lengths"].append(ep_len)
            log_data["env_ids"].append(env_id)

            # Start new episode (possibly different env via curriculum)
            env_id = curriculum.sample_env_id(global_step)
            env = curriculum.get_env(env_id)
            ep_seed = pick_episode_seed()
            obs, info = env.reset(seed=ep_seed)
            state = frame_stack.reset(obs)
            nstep_buf.reset()
            ep_reward = 0.0
            ep_kills = 0
            ep_len = 0
            prev_kills = 0.0
            prev_health = float(info.get("health", 100))

        # ── Learn ──
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(replay) >= BATCH_SIZE:
            # Adaptive LR: boost during curriculum transition windows
            current_lr = get_lr_for_step(global_step, actual_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (global_step / final_global_step)

            batch, indices, weights = replay.sample(BATCH_SIZE, beta)
            weights = weights.to(device)

            # Unpack batch
            states  = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
            actions = torch.LongTensor([t[1] for t in batch]).to(device)
            rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
            next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
            dones   = torch.FloatTensor([float(t[4]) for t in batch]).to(device)

            # Current Q
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN target
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(dim=1)
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rewards + (GAMMA ** N_STEP) * next_q * (1 - dones)

            td_errors = (q_values - target).detach().cpu().numpy()
            loss = (weights * nn.functional.smooth_l1_loss(q_values, target, reduction='none')).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            optimizer.step()

            replay.update_priorities(indices, td_errors)

            total_loss += loss.item()
            loss_count += 1

        # ── Target update ──
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ── Logging ──
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            fps = step / max(elapsed, 1)
            avg_loss = total_loss / max(loss_count, 1)
            current_lr = get_lr_for_step(global_step, actual_lr)

            recent_rewards = log_data["episode_rewards"][-20:] if log_data["episode_rewards"] else [0]
            recent_kills = log_data["episode_kills"][-20:] if log_data["episode_kills"] else [0]

            stage_envs = curriculum.get_env_weights(global_step)
            stage_names = [e[0].replace("SeekAndSlay", "SS") for e in stage_envs]

            print(f"Step {global_step:>8d}/{final_global_step} | "
                  f"Eps {eps:.3f} | "
                  f"LR {current_lr:.1e} | "
                  f"AvgKills(20) {np.mean(recent_kills):.1f} | "
                  f"AvgRew(20) {np.mean(recent_rewards):.2f} | "
                  f"Loss {avg_loss:.4f} | "
                  f"Episodes {ep_count} | "
                  f"FPS {fps:.0f} | "
                  f"Stage {stage_names}")
            total_loss = 0.0
            loss_count = 0

        # ── Save checkpoint ──
        if step % SAVE_INTERVAL == 0:
            ckpt_path = SAVE_DIR / f"weights_{global_step}.pth"
            torch.save(policy_net.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

            # Also save as latest (submission-ready name)
            torch.save(policy_net.state_dict(), SAVE_DIR / "weights.pth")

            # Save log
            with open(SAVE_DIR / "train_log.json", "w") as f:
                json.dump(log_data, f)

    # ── Final save ──
    torch.save(policy_net.state_dict(), SAVE_DIR / "weights_final.pth")
    torch.save(policy_net.state_dict(), SAVE_DIR / "weights.pth")
    with open(SAVE_DIR / "train_log.json", "w") as f:
        json.dump(log_data, f)

    curriculum.close_all()
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN on LevDoom Seek and Slay")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to weights .pth file to resume training from")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Global step number to resume from "
                             "(affects epsilon schedule and curriculum)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of NEW steps to train (default: 3000000)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override base learning rate (default: 5e-5)")
    args = parser.parse_args()
    train(resume_path=args.resume, resume_step=args.resume_step,
          total_steps=args.steps, lr=args.lr)
