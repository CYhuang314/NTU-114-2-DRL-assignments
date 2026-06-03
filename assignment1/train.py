"""
Q-Learning Training Script for DynamicTaxi Environment

Usage:
    python3 train.py

Output:
    q_table.pkl  — serialized Q-table for use in student_agent.py
"""

import numpy as np
import random
import pickle
import time
from collections import defaultdict

from env import DynamicTaxiEnv
from get_state import get_state
from get_state import find_goal


# ============================================================
# Hyperparameters
# ============================================================
NUM_EPISODES = 600000
GAMMA = 0.99                  # discount factor
ALPHA_START = 0.1            
# ALPHA_END = 0.05              
# ALPHA_DECAY = 0.99999

EPSILON_START = 1.0           # initial exploration rate
EPSILON_END = 0.01            # final exploration rate
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / 200000        

NUM_ACTIONS = 8               # actions 0-7
GRID_SIZE_MIN = 5             # min grid size for training
GRID_SIZE_MAX = 10            # max grid size for training
FUEL_LIMIT = 500              # match test-time fuel limit
MAX_STEPS_PER_EPISODE = 4000

# Logging
LOG_INTERVAL = 100           # print stats every N episodes
SAVE_INTERVAL = 50000         # save checkpoint every N episodes


# ============================================================
# Q-Table: defaultdict with zero initialization
# ============================================================
Q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS))


# ============================================================
# Epsilon-greedy action selection
# ============================================================
def select_action(state, epsilon):
    """Select action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return int(np.argmax(Q_table[state]))


# ============================================================
# Training loop
# ============================================================
def train():
    global Q_table

    alpha = ALPHA_START
    epsilon = EPSILON_START

    # Statistics tracking
    episode_rewards = []
    episode_env_rewards = []
    episode_steps = []
    best_avg_env_reward = -float('inf')

    total_steps = 0
    start_time = time.time()

    for episode in range(NUM_EPISODES):
        # Randomize grid size for generalization
        grid_size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
        
        env = DynamicTaxiEnv(grid_size=grid_size, fuel_limit=FUEL_LIMIT)
        obs, _ = env.reset()
        state = get_state(obs)

        episode_reward = 0.0
        total_env_reward = 0.0
        done = False
        step = 0
        
        prev_obs = obs

        has_entered_highway = False

        while not done and step < MAX_STEPS_PER_EPISODE:
            # 1. Select action
            action = select_action(state, epsilon)

            # 2. Take action
            obs, env_reward, done, info = env.step(action)

            # 3. Compute shaped reward
            shaped_reward = 0.0
            prev_zone, prev_carry_n, prev_fuel_enough, _, _, prev_dist_bin, prev_cell_up, prev_cell_left, prev_cell_cur, prev_cell_right = get_state(prev_obs)
            zone, carry_n, fuel_enough, _, _, dist_bin, cell_up, cell_left, cell_cur, cell_right = get_state(obs)
            prev_goal_x, prev_goal_y = find_goal(prev_obs)
            goal_x, goal_y = find_goal(obs)
            if prev_fuel_enough:
                if prev_carry_n:
                    prev_task = 3
                else:
                    prev_task = 2
            else:
                prev_task = 1

            if fuel_enough:
                if carry_n:
                    task = 3
                else:
                    task = 2
            else:
                task = 1

            if prev_task != task:
                has_entered_highway = False
            # --- Milestone bonus ---
            
            # pickup at zone 1
            if prev_zone == 1 and action == 3 and carry_n == prev_carry_n + 1:
                shaped_reward += 10.0
            
            # enter correct highway
            if prev_dist_bin == 0 and action == 5 and prev_zone != zone and (not has_entered_highway):
                shaped_reward += 15.0
                has_entered_highway = True

            # refuel completed
            if (not prev_fuel_enough) and fuel_enough:
                shaped_reward += 10.0

            # --- Continuous bonus ---

            # distance to goal
            prev_dist = abs(prev_goal_x - prev_obs[0]) + abs(prev_goal_y - prev_obs[1])
            dist = abs(goal_x - obs[0]) + abs(goal_y - obs[1])
            if prev_zone == zone:
                if prev_dist > dist:
                    shaped_reward += 0.35
                elif prev_dist < dist:
                    shaped_reward -= 0.35

            if prev_cell_cur == -1 and action == 7:
                shaped_reward += 0.05
            
            # --- Penalty ---
            if prev_dist_bin and action == 5:
                shaped_reward -= 1.0
            if action == 3 and prev_carry_n == carry_n:
                shaped_reward -= 0.5
            if action == 4 and prev_carry_n == 0:
                shaped_reward -= 0.5

            # 4. Get next state
            next_state = get_state(obs)

            # 5. Q-learning update (Bellman equation)
            if done:
                td_target = shaped_reward + env_reward
            else:
                td_target = shaped_reward + env_reward + GAMMA * np.max(Q_table[next_state])

            td_error = td_target - Q_table[state][action]
            Q_table[state][action] += alpha * td_error

            # 6. Transition
            prev_obs = obs
            state = next_state
            episode_reward += env_reward + shaped_reward
            total_env_reward += env_reward
            step += 1
            total_steps += 1

        # Episode done — decay alpha and epsilon per episode
        # alpha = max(ALPHA_END, alpha * ALPHA_DECAY)
        epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

        episode_env_rewards.append(total_env_reward)
        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        # Logging
        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_rewards[-LOG_INTERVAL:]
            recent_env = episode_env_rewards[-LOG_INTERVAL:]
            avg_reward = np.mean(recent)
            avg_env_reward = np.mean(episode_env_rewards[-LOG_INTERVAL:])
            avg_steps = np.mean(episode_steps[-LOG_INTERVAL:])
            max_reward = np.max(recent)
            min_reward = np.min(recent)
            max_env_reward = np.max(recent_env)
            min_env_reward = np.min(recent_env)
            elapsed = time.time() - start_time

            print(
                f"Ep {episode+1:>6d}/{NUM_EPISODES} | "
                f"Avg R: {avg_reward:>8.2f} | "
                f"Avg env_R: {avg_env_reward:>8.2f}|"
                f"Max R: {max_reward:>8.2f} | "
                f"Min R: {min_reward:>8.2f} | "
                f"Avg Steps: {avg_steps:>7.1f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Alpha: {alpha:.5f} | "
                f"Q-size: {len(Q_table):>8d} | "
                f"Time: {elapsed:>7.1f}s"
            )

            # Track best performance
            if avg_env_reward > best_avg_env_reward:
                best_avg_env_reward = avg_env_reward
                save_qtable("q_table_best.pkl")
                print(f"  >> New best avg_env reward: {best_avg_env_reward:.2f}, saved q_table_best.pkl")

        # Periodic checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_qtable(f"q_table_checkpoint_{episode+1}.pkl")

    # Final save
    save_qtable("q_table.pkl")
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total episodes: {NUM_EPISODES}")
    print(f"  Total steps: {total_steps}")
    print(f"  Q-table size: {len(Q_table)} states")
    print(f"  Best avg env reward (per {LOG_INTERVAL} eps): {best_avg_env_reward:.2f}")
    print(f"  Final epsilon: {epsilon:.4f}")
    print(f"  Final alpha: {alpha:.5f}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Saved: q_table.pkl, q_table_best.pkl")


# ============================================================
# Save / Load Q-table
# ============================================================
def save_qtable(filename):
    """Save Q-table as a regular dict (pickle-friendly)."""
    save_dict = {k: v.tolist() for k, v in Q_table.items()}
    with open(filename, "wb") as f:
        pickle.dump(save_dict, f)


def load_qtable(filename):
    """Load Q-table from file into defaultdict."""
    global Q_table
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
    Q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    for k, v in loaded.items():
        Q_table[k] = np.array(v)
    print(f"Loaded Q-table from {filename}: {len(Q_table)} states")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DynamicTaxi Q-Learning Training")
    print("=" * 60)
    print(f"  Episodes:    {NUM_EPISODES}")
    print(f"  Gamma:       {GAMMA}")
    print(f"  Alpha:       {ALPHA_START}")
    print(f"  Epsilon:     {EPSILON_START} -> {EPSILON_END}")
    print(f"  Grid sizes:  {GRID_SIZE_MIN}-{GRID_SIZE_MAX}")
    print(f"  Fuel limit:  {FUEL_LIMIT}")
    print("=" * 60)
    print()

    train()
