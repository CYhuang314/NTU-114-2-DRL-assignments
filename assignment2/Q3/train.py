"""
Training script for Fib-2584 agent using TD-Afterstate learning with N-Tuple Networks.

Fixed: recomputes V(s'_prev) fresh before TD update (not stale cached value).
Optimized: batch evaluate for action selection, pure-Python evaluate/update for TD step.

Usage:
  python train.py [--num_games 200000] [--alpha 0.0025] [--save_path weights]
"""

import argparse
import time
import os
import numpy as np

from fib2584_env import Fib2584Env
from ntuple_network import NTupleNetwork


def train_one_game(env, net, alpha):
    """Play one game with TD-Afterstate learning."""
    env.reset()
    if env.done:
        return 0, 0, 0

    evaluate = net.evaluate
    update = net.update
    evaluate_batch = net.evaluate_batch
    clamp = NTupleNetwork.clamp_codes
    step_reward_codes = env.step_reward_codes
    simulate_all = env.simulate_all_steps_codes

    # --- First move: select best action ---
    all_after, all_rewards, all_moved = simulate_all()
    all_after_np = np.asarray(all_after, dtype=np.int64)
    all_rewards_np = np.asarray(all_rewards, dtype=np.int64)
    all_moved_np = np.asarray(all_moved, dtype=np.bool_)

    legal_mask = all_moved_np
    if not legal_mask.any():
        return 0, 0, 0

    values = evaluate_batch(all_after_np)
    action_values = np.where(legal_mask, all_rewards_np + values, -1e18)
    best_action = int(np.argmax(action_values))

    prev_after = clamp(all_after_np[best_action].tolist())

    step_reward_codes(best_action)
    num_steps = 1

    while not env.done:
        all_after, all_rewards, all_moved = simulate_all()
        all_after_np = np.asarray(all_after, dtype=np.int64)
        all_rewards_np = np.asarray(all_rewards, dtype=np.int64)
        all_moved_np = np.asarray(all_moved, dtype=np.bool_)

        legal_mask = all_moved_np
        if not legal_mask.any():
            break

        values = evaluate_batch(all_after_np)
        action_values = np.where(legal_mask, all_rewards_np + values, -1e18)
        best_action = int(np.argmax(action_values))

        best_reward = int(all_rewards_np[best_action])
        curr_after = clamp(all_after_np[best_action].tolist())

        # TD update: recompute V(s'_prev) fresh (not stale cached value)
        v_prev = evaluate(prev_after)
        v_curr = float(values[best_action])
        td_error = best_reward + v_curr - v_prev
        update(prev_after, alpha * td_error)

        step_reward_codes(best_action)
        num_steps += 1

        prev_after = curr_after

    # Terminal update: recompute V(s'_last) fresh
    v_last = evaluate(prev_after)
    update(prev_after, alpha * (0 - v_last))

    return env.score, num_steps, env.max_tile


def train(num_games, alpha, save_path, save_interval=10000, log_interval=1000):
    """Main training loop."""
    net = NTupleNetwork()

    if os.path.exists(save_path + '.npz'):
        print(f"Loading existing weights from {save_path}.npz")
        net.load(save_path)
    else:
        print("Starting fresh training")

    net.info()
    print(f"\nTraining: {num_games} games, alpha={alpha}, save={save_path}\n")

    env = Fib2584Env(0)

    scores = []
    steps_list = []
    max_tiles = []
    total_start = time.time()
    interval_start = time.time()

    for game_idx in range(1, num_games + 1):
        score, steps, max_tile = train_one_game(env, net, alpha)
        scores.append(score)
        steps_list.append(steps)
        max_tiles.append(max_tile)

        if game_idx % log_interval == 0:
            rs = scores[-log_interval:]
            rt = max_tiles[-log_interval:]
            elapsed = time.time() - interval_start
            gps = log_interval / elapsed
            eta = (num_games - game_idx) / gps / 60 if gps > 0 else 0

            tc = {}
            for t in rt:
                tc[t] = tc.get(t, 0) + 1
            ts = " | ".join(f"{v}:{c/len(rt)*100:.0f}%" for v, c in
                           sorted(tc.items(), reverse=True)[:5])

            print(f"G {game_idx:>7d}/{num_games} | "
                  f"Avg:{np.mean(rs):>9.0f} | Max:{np.max(rs):>8d} | "
                  f"Steps:{np.mean(steps_list[-log_interval:]):>5.0f} | "
                  f"{gps:>5.1f}g/s | ETA:{eta:>5.1f}m | {ts}")
            interval_start = time.time()

        if game_idx % save_interval == 0:
            cp = f"{save_path}_{game_idx:06d}"
            net.save(cp)
            net.save(save_path)
            print(f"  [Saved {cp}.npz]")

    net.save(save_path)
    net.save(f"{save_path}_{num_games:06d}")
    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"Done! {num_games} games in {total_time/60:.1f}min ({total_time/3600:.2f}h)")
    print(f"Last {log_interval} avg: {np.mean(scores[-log_interval:]):.0f}")
    print(f"{'='*70}")

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fib-2584 agent")
    parser.add_argument("--num_games", type=int, default=200000)
    parser.add_argument("--alpha", type=float, default=0.0025)
    parser.add_argument("--save_path", type=str, default="weights")
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    train(args.num_games, args.alpha, args.save_path, args.save_interval, args.log_interval)
