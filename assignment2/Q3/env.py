import argparse
import gc
import importlib.util
import os
import random
import signal
import sys
import traceback
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import requests
import secrets

try:
    from fib2584_env import Fib2584Env
except ImportError as exc:
    raise ImportError(
        "Failed to import fib2584_env. Build the pybind11 module first and make sure "
        "it is available on PYTHONPATH before running evaluation."
    ) from exc


DEFAULT_MAX_ACTIVE_GAMES_PER_WORKER = 4
DEFAULT_ACTION_TIMEOUT_SEC = 0.06
DEFAULT_PROGRESS_LOG_EVERY_STEPS = 2000
ACTION_TIMEOUT_SEC = float(os.environ.get("ACTION_TIMEOUT_SEC", str(DEFAULT_ACTION_TIMEOUT_SEC)))
PROGRESS_LOG_EVERY_STEPS = int(
    os.environ.get("PROGRESS_LOG_EVERY_STEPS", str(DEFAULT_PROGRESS_LOG_EVERY_STEPS))
)


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)

    def contains(self, x) -> bool:
        try:
            value = int(x)
        except Exception:
            return False
        return 0 <= value < self.n


class Game2584Env:
    """Single-game wrapper around the C++/pybind Fib2584Env."""

    def __init__(self, seed=None):
        init_seed = 0 if seed is None else int(seed)
        self._env = Fib2584Env(init_seed)
        self.size = 4
        self.action_space = _DiscreteSpace(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True
        self.reset(seed=seed)

    def reset(self, seed=None):
        state = self._env.reset() if seed is None else self._env.reset(int(seed))
        self.last_move_valid = True
        return np.asarray(state, dtype=np.int64)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        state, score, done, info = self._env.step(int(action))
        info = dict(info)
        self.last_move_valid = bool(info.get("moved", True))
        return np.asarray(state, dtype=np.int64), int(score), bool(done), info

    def step_reward(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        state, reward, done, info = self._env.step_reward(int(action))
        info = dict(info)
        self.last_move_valid = bool(info.get("moved", True))
        return np.asarray(state, dtype=np.int64), int(reward), bool(done), info

    def simulate_step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        state, reward, moved = self._env.simulate_step(int(action))
        return np.asarray(state, dtype=np.int64), int(reward), bool(moved)

    def is_move_legal(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        return bool(self._env.is_move_legal(int(action)))

    def legal_actions(self):
        return [int(a) for a in self._env.legal_actions()]

    def get_board(self):
        return np.asarray(self._env.get_state(), dtype=np.int64)

    @property
    def score(self):
        return int(self._env.score)

    @property
    def done(self):
        return bool(self._env.done)

    @property
    def max_tile(self):
        return int(self._env.max_tile)

    @property
    def empty_count(self):
        return int(self._env.empty_count)

    def is_game_over(self):
        return self.done

    def render(self, mode="human", action=None):
        return self._env.render()


Game2048Env = Game2584Env


def load_student_agent(agent_file):
    if "student_agent" in sys.modules:
        del sys.modules["student_agent"]
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)
    return student_agent


def split_evenly(items, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append(item)
    return [chunk for chunk in chunks if chunk]


def _start_game(seed, trial_id):
    env = Game2584Env(seed=seed)
    state = env.reset(seed=seed)
    return {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "env": env,
        "state": state,
        "step_count": 0,
        "timeout_count": 0,
    }


def _call_get_action_with_timeout(agent_module, state, score, timeout_sec):
    if timeout_sec is None or timeout_sec <= 0:
        return agent_module.get_action(state, score)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"get_action exceeded {timeout_sec:.2f} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return agent_module.get_action(state, score)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _get_action_or_fallback(agent_module, game, timeout_sec):
    env = game["env"]
    state = game["state"]
    score = env.score
    timeout_fallback = False

    try:
        action = int(_call_get_action_with_timeout(agent_module, state, score, timeout_sec))
    except TimeoutError:
        legal = env.legal_actions()
        if not legal:
            raise
        action = int(random.choice(legal))
        timeout_fallback = True
    except Exception:
        raise

    if action not in (0, 1, 2, 3):
        raise ValueError(f"Invalid action from agent: {action}")
    if not env.is_move_legal(action):
        raise ValueError(f"Illegal action from agent: {action}")

    return action, timeout_fallback


def _maybe_log_progress(game, current_score, progress_log_every_steps):
    if progress_log_every_steps <= 0:
        return
    if game["step_count"] <= 0:
        return
    if game["step_count"] % progress_log_every_steps != 0:
        return

    trial_no = game["trial_id"] + 1
    timeout_suffix = (
        f" | timeout_fallbacks: {game['timeout_count']}" if game.get("timeout_count", 0) > 0 else ""
    )
    print(
        f"Trial {trial_no:02d} | Progress | Steps: {game['step_count']:5d} | "
        f"Score: {int(current_score)}{timeout_suffix}",
        flush=True,
    )



def run_interleaved_games(
    agent_module,
    worker_jobs,
    max_active_games=DEFAULT_MAX_ACTIVE_GAMES_PER_WORKER,
    action_timeout_sec=ACTION_TIMEOUT_SEC,
    progress_log_every_steps=PROGRESS_LOG_EVERY_STEPS,
):
    if not worker_jobs:
        return []

    max_active_games = max(1, int(max_active_games))
    pending_jobs = list(worker_jobs)
    active_games = []
    finished = []

    def maybe_launch_new_games():
        while pending_jobs and len(active_games) < max_active_games:
            trial_id, seed = pending_jobs.pop(0)
            active_games.append(_start_game(seed=seed, trial_id=trial_id))

    maybe_launch_new_games()

    while active_games:
        next_active_games = []
        for game in active_games:
            env = game["env"]
            trial_id = game["trial_id"]
            try:
                action, timeout_fallback = _get_action_or_fallback(
                    agent_module=agent_module,
                    game=game,
                    timeout_sec=action_timeout_sec,
                )
                if timeout_fallback:
                    game["timeout_count"] += 1

                next_state, score, done, _ = env.step(action)
                game["state"] = next_state
                game["step_count"] += 1
                _maybe_log_progress(
                    game=game,
                    current_score=score,
                    progress_log_every_steps=progress_log_every_steps,
                )
                if done:
                    finished.append({
                        "trial_id": trial_id,
                        "seed": game["seed"],
                        "step_count": game["step_count"],
                        "score": int(score),
                        "timeout_count": int(game["timeout_count"]),
                        "error": None,
                    })
                else:
                    next_active_games.append(game)
            except Exception as exc:
                finished.append({
                    "trial_id": trial_id,
                    "seed": game["seed"],
                    "step_count": game["step_count"],
                    "score": 0,
                    "timeout_count": int(game["timeout_count"]),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        active_games = next_active_games
        maybe_launch_new_games()

    finished.sort(key=lambda x: x["trial_id"])
    return finished



def worker_process(agent_file, worker_id, worker_jobs, max_active_games, action_timeout_sec, progress_log_every_steps, q):
    try:
        student_agent = load_student_agent(agent_file)
        results = run_interleaved_games(
            agent_module=student_agent,
            worker_jobs=worker_jobs,
            max_active_games=max_active_games,
            action_timeout_sec=action_timeout_sec,
            progress_log_every_steps=progress_log_every_steps,
        )
        q.put({"worker_id": worker_id, "results": results, "error": None})
    except Exception:
        q.put({"worker_id": worker_id, "results": [], "error": traceback.format_exc()})



def parse_arguments():
    parser = argparse.ArgumentParser(description="HW2 2584 evaluation")
    parser.add_argument("--token", default="", type=str)
    return parser.parse_args()



def eval_score():
    start_dt = datetime.now()
    args = parse_arguments()



    total_score = 0
    num_trials = 8

    cpu_count = os.cpu_count() or 1
    num_workers = max(1, min(cpu_count, num_trials))
    worker_chunks = split_evenly(
        [(trial_id, secrets.randbits(31)) for trial_id in range(num_trials)],
        num_workers,
    )
    max_active_games_per_worker = min(
        DEFAULT_MAX_ACTIVE_GAMES_PER_WORKER,
        max(len(chunk) for chunk in worker_chunks) if worker_chunks else 1,
    )

    print(f"Evaluation start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {num_trials} Fib-2584 games using {num_workers} worker process(es).")
    print(f"Each worker interleaves up to {max_active_games_per_worker} active game(s).")
    print(f"Per-step action timeout: {ACTION_TIMEOUT_SEC:.2f} sec (timeout -> random legal move)")
    if PROGRESS_LOG_EVERY_STEPS > 0:
        print(f"Progress log every {PROGRESS_LOG_EVERY_STEPS} step(s) per trial.")
    else:
        print("Progress log disabled.")

    queues = []
    processes = []
    for worker_id, worker_jobs in enumerate(worker_chunks):
        gc.collect()
        q = Queue()
        p = Process(
            target=worker_process,
            args=(
                "student_agent.py",
                worker_id,
                worker_jobs,
                max_active_games_per_worker,
                ACTION_TIMEOUT_SEC,
                PROGRESS_LOG_EVERY_STEPS,
                q,
            ),
        )
        p.start()
        queues.append(q)
        processes.append((p, worker_jobs))

    all_results = []
    for (p, worker_jobs), q in zip(processes, queues):
        payload = q.get()
        p.join()
        if payload["error"] is not None:
            print(f"Worker {payload['worker_id']} failed:\n{payload['error']}")
            for trial_id, seed in worker_jobs:
                all_results.append({
                    "trial_id": trial_id,
                    "seed": seed,
                    "step_count": 0,
                    "score": 0,
                    "timeout_count": 0,
                    "error": "worker_crashed",
                })
            continue
        all_results.extend(payload["results"])

    all_results.sort(key=lambda x: x["trial_id"])
    scores = [int(result["score"]) for result in all_results]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    for result in all_results:
        trial_no = result["trial_id"] + 1
        timeout_suffix = f" | timeout_fallbacks: {result['timeout_count']}" if result.get("timeout_count", 0) > 0 else ""
        if result["error"] is None:
            print(
                f"Trial {trial_no:02d} | Steps: {result['step_count']:4d} | "
                f"Score: {result['score']}{timeout_suffix}"
            )
        else:
            print(
                f"Trial {trial_no:02d} | FAILED | Score: {result['score']} | "
                f"{result['error']}{timeout_suffix}"
            )

    print(f"\nFinal Average Score over {len(scores)} runs: {avg_score}")
    end_dt = datetime.now()

    print(f"Evaluation End time: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")




if __name__ == "__main__":
    eval_score()
