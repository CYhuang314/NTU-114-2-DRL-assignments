"""
Train an N-Tuple Network for Battle Fib-2584 via TD(0) afterstate self-play.

Optimized: minimises redundant board.clone()/move()/legal_actions() calls.

Usage:
    python train.py --episodes 200000 --lr 0.001 --save-path weights.npz
"""

import argparse
import random
import time
import sys
import os

import numpy as np

from ntuple_network import NTupleNetwork, MAX_CODE, NUM_CODES
from fib2584_attack_env_py import Fib2584AttackEnv, Fib2584Board

TERMINAL_REWARD = 50.0

_CLAMP = [min(i, MAX_CODE) for i in range(32)]


def clamp_board_codes(board: Fib2584Board):
    cl = _CLAMP
    raw = board.raw
    bits = Fib2584Board.CELL_BITS
    mask = Fib2584Board.CELL_MASK
    return [cl[(raw >> (i * bits)) & mask] for i in range(16)]


def evaluate_with_placement(net, base_codes, pos, tile_code):
    codes = list(base_codes)
    codes[pos] = min(tile_code, MAX_CODE)
    return net.evaluate(codes)


def choose_action_fast(net, env, rng, epsilon, place_explore):
    """
    Returns (slide, place_a, place_b, own_codes_after, opp_codes_after).
    own/opp_codes_after are the afterstate codes from the acting player's view.
    """
    cur = env.current_player
    board_a = env.board_a
    board_b = env.board_b

    own_board_orig = board_a if cur == 0 else board_b
    legal = own_board_orig.legal_actions()

    # Slide selection: epsilon-greedy
    if rng.random() < epsilon:
        slide = rng.choice(legal)
    else:
        best_score = -1e18
        slide = legal[0]
        for action in legal:
            ba = board_a.clone()
            bb = board_b.clone()
            ba.move(action)
            bb.move(action)
            codes_a = clamp_board_codes(ba)
            codes_b = clamp_board_codes(bb)
            va = net.evaluate(codes_a)
            vb = net.evaluate(codes_b)
            score = (va - vb) if cur == 0 else (vb - va)
            if score > best_score:
                best_score = score
                slide = action

    # Apply slide to cloned boards
    ba = board_a.clone()
    bb = board_b.clone()
    ba.move(slide)
    bb.move(slide)

    if cur == 0:
        own_board, opp_board = ba, bb
    else:
        own_board, opp_board = bb, ba

    own_codes_base = clamp_board_codes(own_board)
    opp_codes_base = clamp_board_codes(opp_board)

    # Own board: maximise V
    own_empty = own_board.empty_positions()
    if not own_empty:
        own_place = None
        own_codes_final = own_codes_base
    elif rng.random() < place_explore:
        pos = rng.choice(own_empty)
        tc = rng.choice((1, 2))
        own_place = {"position": pos, "tile_code": tc}
        own_codes_final = list(own_codes_base)
        own_codes_final[pos] = min(tc, MAX_CODE)
    else:
        best_val = -1e18
        best_pos = own_empty[0]
        best_tc = 1
        for pos in own_empty:
            for tc in (1, 2):
                v = evaluate_with_placement(net, own_codes_base, pos, tc)
                if v > best_val:
                    best_val = v
                    best_pos = pos
                    best_tc = tc
        own_place = {"position": best_pos, "tile_code": best_tc}
        own_codes_final = list(own_codes_base)
        own_codes_final[best_pos] = min(best_tc, MAX_CODE)

    # Opponent board: minimise V
    opp_empty = opp_board.empty_positions()
    if not opp_empty:
        opp_place = None
        opp_codes_final = opp_codes_base
    elif rng.random() < place_explore:
        pos = rng.choice(opp_empty)
        tc = rng.choice((1, 2))
        opp_place = {"position": pos, "tile_code": tc}
        opp_codes_final = list(opp_codes_base)
        opp_codes_final[pos] = min(tc, MAX_CODE)
    else:
        best_val = 1e18
        best_pos = opp_empty[0]
        best_tc = 1
        for pos in opp_empty:
            for tc in (1, 2):
                v = evaluate_with_placement(net, opp_codes_base, pos, tc)
                if v < best_val:
                    best_val = v
                    best_pos = pos
                    best_tc = tc
        opp_place = {"position": best_pos, "tile_code": best_tc}
        opp_codes_final = list(opp_codes_base)
        opp_codes_final[best_pos] = min(best_tc, MAX_CODE)

    if cur == 0:
        return slide, own_place, opp_place, own_codes_final, opp_codes_final
    else:
        return slide, opp_place, own_place, own_codes_final, opp_codes_final


def train(args):
    net = NTupleNetwork()
    net.info()

    load_file = args.load_path
    if load_file:
        lf = load_file if load_file.endswith('.npz') else load_file + '.npz'
        if os.path.exists(lf):
            print(f"Loading weights from {load_file}")
            net.load(load_file)

    rng = random.Random(args.seed)
    lr = args.lr
    epsilon = args.epsilon
    place_explore = args.place_explore

    total_episodes = args.episodes
    log_interval = args.log_interval
    save_interval = args.save_interval

    num_features = len(net._syms_6) + len(net._syms_4)
    alpha = lr / num_features

    print(f"\nTraining config:")
    print(f"  Episodes: {total_episodes}")
    print(f"  LR: {lr} -> alpha/feature: {alpha:.6f}")
    print(f"  Epsilon: {epsilon}, Place explore: {place_explore}")
    print(f"  Terminal reward: {TERMINAL_REWARD}")
    print(f"  Features: {num_features}")
    print(f"  Seed: {args.seed}")
    print()

    wins_p0 = 0
    wins_p1 = 0
    total_turns = 0
    t_start = time.perf_counter()

    for ep in range(total_episodes):
        env_seed = rng.randint(0, 2**31 - 1)
        env = Fib2584AttackEnv(seed=env_seed)

        # History: (own_codes, opp_codes) from acting player's perspective
        history = []

        while not env.done:
            cur = env.current_player
            slide, place_a, place_b, own_final, opp_final = \
                choose_action_fast(net, env, rng, epsilon, place_explore)
            env.step_turn(slide, place_a=place_a, place_b=place_b)
            history.append((own_final, opp_final))

        winner = env.winner
        total_turns += env.turn_index
        if winner == 0:
            wins_p0 += 1
        else:
            wins_p1 += 1

        # TD(0) backward updates
        n = len(history)
        if n == 0:
            continue

        # Terminal update
        last_player = (n - 1) % 2  # player 0 starts
        own_last, opp_last = history[-1]
        terminal_target = TERMINAL_REWARD if last_player == winner else -TERMINAL_REWARD

        v_game = net.evaluate(own_last) - net.evaluate(opp_last)
        td_error = terminal_target - v_game
        net.update(own_last, alpha * td_error)
        net.update(opp_last, alpha * (-td_error))

        # Non-terminal backward
        for t in range(n - 2, -1, -1):
            own_t, opp_t = history[t]
            own_next, opp_next = history[t + 1]

            v_game_t = net.evaluate(own_t) - net.evaluate(opp_t)
            v_game_next = net.evaluate(own_next) - net.evaluate(opp_next)

            # Negamax: target = -v_game_next (next is opponent's perspective)
            td_error = -v_game_next - v_game_t
            net.update(own_t, alpha * td_error)
            net.update(opp_t, alpha * (-td_error))

        # Logging
        if (ep + 1) % log_interval == 0:
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            speed = (ep + 1) / elapsed
            avg_turns = total_turns / (ep + 1)
            eta_h = (total_episodes - ep - 1) / speed / 3600

            print(
                f"ep={ep + 1:>8d}/{total_episodes}  "
                f"p0={wins_p0} p1={wins_p1} "
                f"(p0r={wins_p0 / (ep + 1):.3f})  "
                f"avg_t={avg_turns:.1f}  "
                f"{speed:.1f}g/s  "
                f"{elapsed / 60:.1f}min  "
                f"ETA={eta_h:.1f}h"
            )

        if (ep + 1) % save_interval == 0:
            net.save(args.save_path)
            print(f"  -> saved to {args.save_path}")

    net.save(args.save_path)
    elapsed = (time.perf_counter() - t_start)
    print(f"\nDone. {total_episodes} ep in {elapsed / 60:.1f}min ({elapsed / 3600:.1f}h)")
    print(f"p0={wins_p0} p1={wins_p1} (p0r={wins_p0 / max(total_episodes, 1):.3f})")
    print(f"Saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--place-explore", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="weights.npz")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=10000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
