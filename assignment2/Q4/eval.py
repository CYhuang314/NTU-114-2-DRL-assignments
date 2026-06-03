import argparse
import secrets
import shlex
import sys
from typing import List, Tuple

from match_core import (
    SubprocessJSONAgent,
    build_python_agent_command,
    load_env_class,
    play_match,
)


DEFAULT_GAMES = 20
DEFAULT_MIN_WINS = 13


def build_agent_command(agent_spec: str) -> List[str]:
    return build_python_agent_command(agent_spec)


def run_match_series(
    student_spec: str,
    baseline_spec: str,
    games: int,
    seed: int,
    timeout: float,
    startup_timeout: float,
    env_backend: str = "auto",
    verbose: bool = False,
    min_wins: int = DEFAULT_MIN_WINS,
) -> Tuple[int, int, str, str]:
    env_class, env_path = load_env_class(env_backend)

    student_wins = 0
    baseline_wins = 0
    stop_reason = "completed_all_games"

    student_cmd = build_agent_command(student_spec)
    baseline_cmd = build_agent_command(baseline_spec)

    for g in range(games):
        swap = (g % 2 == 1)

        if swap:
            cmd_a = baseline_cmd
            cmd_b = student_cmd
            side_to_agent_index = (1, 0)
        else:
            cmd_a = student_cmd
            cmd_b = baseline_cmd
            side_to_agent_index = (0, 1)

        agent_a = SubprocessJSONAgent(cmd_a, "side-a", startup_timeout_sec=startup_timeout)
        agent_b = SubprocessJSONAgent(cmd_b, "side-b", startup_timeout_sec=startup_timeout)
        try:
            result = play_match(
                agent_for_side_a=agent_a,
                agent_for_side_b=agent_b,
                env_class=env_class,
                seed=seed + g,
                verbose=verbose,
                side_to_agent_index=side_to_agent_index,
                action_timeout_sec=timeout,
            )
        finally:
            agent_a.close()
            agent_b.close()

        if result.winner_agent_index == 0:
            student_wins += 1
            winner = "student"
        else:
            baseline_wins += 1
            winner = "baseline"

        side_a_name = "baseline" if swap else "student"
        side_b_name = "student" if swap else "baseline"
        suffix = ""
        if result.ended_by_forfeit:
            suffix = f" FORFEIT side={'A' if result.forfeit_side == 0 else 'B'} reason={result.forfeit_reason}"
        print(
            f"game={g:02d} A={side_a_name} B={side_b_name} winner={winner} "
            f"turns={result.turns} student_wins={student_wins} baseline_wins={baseline_wins}{suffix}"
        )

        games_played = g + 1
        remaining_games = games - games_played

        if student_wins >= min_wins:
            stop_reason = f"early_pass_reached_{student_wins}_wins_after_{games_played}_games"
            break

        max_possible_student_wins = student_wins + remaining_games
        if max_possible_student_wins < min_wins:
            stop_reason = (
                f"early_fail_only_{max_possible_student_wins}_max_possible_wins_after_{games_played}_games"
            )
            break

    return student_wins, baseline_wins, env_path, stop_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="student_agent.py")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--min-wins", type=int, default=DEFAULT_MIN_WINS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--env-backend", choices=["auto", "cpp", "py"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = secrets.randbits(31)

    print(f"base_seed={args.seed}")

    student_wins, baseline_wins, env_path, stop_reason = run_match_series(
        student_spec=args.student,
        baseline_spec=args.baseline,
        games=args.games,
        seed=args.seed,
        timeout=args.timeout,
        startup_timeout=args.startup_timeout,
        env_backend=args.env_backend,
        verbose=args.verbose,
        min_wins=args.min_wins,
    )

    print()
    print(f"env_module={env_path}")
    print(f"student_cmd={' '.join(shlex.quote(x) for x in build_agent_command(args.student))}")
    print(f"baseline_cmd={' '.join(shlex.quote(x) for x in build_agent_command(args.baseline))}")
    print(f"student_wins={student_wins} baseline_wins={baseline_wins}")
    print(f"pass_threshold={args.min_wins}")
    print(f"stop_reason={stop_reason}")

    if student_wins >= args.min_wins:
        print("RESULT: PASS")
        sys.exit(0)

    print("RESULT: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
