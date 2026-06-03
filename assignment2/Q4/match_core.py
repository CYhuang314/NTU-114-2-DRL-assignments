import argparse
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import secrets


ACTION_NAMES = ["up", "down", "left", "right"]
MODULE_EXTENSIONS = (".so", ".pyd", ".dll", ".dylib")


class AgentError(RuntimeError):
    """Base class for agent execution errors."""


class AgentTimeoutError(AgentError):
    """Raised when an agent does not respond before the deadline."""


class AgentActionError(AgentError):
    """Raised when an agent returns an invalid or malformed action."""


@dataclass
class MatchResult:
    winner_side: int
    loser_side: int
    winner_agent_index: int
    loser_agent_index: int
    turns: int
    ended_by_forfeit: bool = False
    forfeit_side: Optional[int] = None
    forfeit_reason: Optional[str] = None


class SubprocessJSONAgent:
    def __init__(self, command: Union[str, Sequence[str]], name: str, startup_timeout_sec: Optional[float] = 15.0):
        self.command = list(command) if not isinstance(command, str) else shlex.split(command)
        self.name = name
        self.startup_timeout_sec = startup_timeout_sec
        self._first_request = True
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def _readline_with_timeout(self, timeout_sec: Optional[float]) -> str:
        if self.proc.stdout is None:
            raise RuntimeError(f"agent {self.name} has no stdout pipe")
        if timeout_sec is None or timeout_sec <= 0:
            return self.proc.stdout.readline()

        def _timeout_handler(signum, frame):
            raise AgentTimeoutError(f"agent {self.name} exceeded {float(timeout_sec):.2f} seconds")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
        try:
            return self.proc.stdout.readline()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _read_stderr_tail(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read() or ""
        except Exception:
            return ""

    def get_action(self, request: Dict[str, Any], timeout_sec: Optional[float]) -> Dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"agent {self.name} has no stdin/stdout pipe")
        if self.proc.poll() is not None:
            err = self._read_stderr_tail()
            raise RuntimeError(
                f"agent {self.name} exited early with code {self.proc.returncode}. stderr:\n{err}"
            )

        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = self._read_stderr_tail()
            raise RuntimeError(f"agent {self.name} broke its stdin pipe. stderr:\n{err}") from exc

        effective_timeout = self.startup_timeout_sec if self._first_request else timeout_sec

        try:
            line = self._readline_with_timeout(effective_timeout)
        finally:
            self._first_request = False

        if not line:
            err = self._read_stderr_tail()
            raise RuntimeError(f"agent {self.name} terminated without a response. stderr:\n{err}")

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentActionError(
                f"agent {self.name} returned non-JSON output on stdout: {line.strip()}"
            ) from exc

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=1.0)
        if self.proc.stderr is not None:
            try:
                _ = self.proc.stderr.read()
            except Exception:
                pass


def load_env_class(backend: str):
    last_err = None
    tried: List[str] = []
    if backend in ("auto", "cpp"):
        tried.append("fib2584_attack_env")
        try:
            mod = importlib.import_module("fib2584_attack_env")
            return mod.Fib2584AttackEnv, getattr(mod, "__file__", "<unknown>")
        except Exception as exc:
            last_err = exc
            if backend == "cpp":
                raise
    if backend in ("auto", "py"):
        tried.append("fib2584_attack_env_py")
        try:
            mod = importlib.import_module("fib2584_attack_env_py")
            return mod.Fib2584AttackEnv, getattr(mod, "__file__", "<unknown>")
        except Exception as exc:
            last_err = exc
            if backend == "py":
                raise
    raise ImportError(f"failed to import env backend from {tried}: {last_err}")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _to_jsonable(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return obj


def build_turn_request(env) -> Dict[str, Any]:
    obs = _to_jsonable(env.observation())
    if bool(obs.get("done", False)):
        raise RuntimeError("cannot build turn request after game is done")

    legal = [int(a) for a in obs["legal_slide_actions"]]
    action_previews: List[Dict[str, Any]] = []
    for action in legal:
        preview = _to_jsonable(env.preview_turn(action))
        preview["slide_action"] = int(preview["slide_action"])
        preview["slide_name"] = ACTION_NAMES[action]
        preview["empty_positions_a"] = [int(x) for x in preview["empty_positions_a"]]
        preview["empty_positions_b"] = [int(x) for x in preview["empty_positions_b"]]
        preview["must_skip_place_a"] = bool(preview["must_skip_place_a"])
        preview["must_skip_place_b"] = bool(preview["must_skip_place_b"])
        preview["slide_reward_a"] = int(preview["slide_reward_a"])
        preview["slide_reward_b"] = int(preview["slide_reward_b"])
        preview["slide_moved_a"] = bool(preview["slide_moved_a"])
        preview["slide_moved_b"] = bool(preview["slide_moved_b"])
        action_previews.append(preview)

    return {
        "request_type": "act",
        "game": "fib2584_attack",
        "version": 2,
        "turn_index": int(obs["turn_index"]),
        "current_player": int(obs["current_player"]),
        "current_player_name": str(obs["current_player_name"]),
        "board_a": obs["board_a"],
        "board_b": obs["board_b"],
        "board_a_codes": obs["board_a_codes"],
        "board_b_codes": obs["board_b_codes"],
        "legal_tile_codes": [int(x) for x in obs["legal_tile_codes"]],
        "legal_slide_actions": legal,
        "legal_slide_action_names": [ACTION_NAMES[a] for a in legal],
        "action_previews": action_previews,
    }


def _preview_by_action(request: Dict[str, Any], action: int) -> Dict[str, Any]:
    for preview in request["action_previews"]:
        if int(preview["slide_action"]) == int(action):
            return preview
    raise AgentActionError(f"slide action {action} is not one of the advertised legal actions")


def _normalize_placement(
    name: str,
    placement: Optional[Dict[str, Any]],
    preview: Dict[str, Any],
    legal_tile_codes: List[int],
) -> Optional[Dict[str, int]]:
    empty_key = "empty_positions_a" if name == "place_a" else "empty_positions_b"
    skip_key = "must_skip_place_a" if name == "place_a" else "must_skip_place_b"
    empty_positions = [int(x) for x in preview[empty_key]]
    must_skip = bool(preview[skip_key])

    if must_skip:
        if placement is not None:
            raise AgentActionError(
                f"{name} must be null because that board has no empty position after slide"
            )
        return None

    if placement is None:
        raise AgentActionError(
            f"{name} must be provided because that board still has empty positions"
        )

    if not isinstance(placement, dict):
        raise AgentActionError(f"{name} must be a dict or null")
    if "position" not in placement or "tile_code" not in placement:
        raise AgentActionError(f"{name} must contain both position and tile_code")

    position = int(placement["position"])
    tile_code = int(placement["tile_code"])
    if position not in empty_positions:
        raise AgentActionError(
            f"{name}.position={position} is not in legal empty positions {empty_positions}"
        )
    if tile_code not in legal_tile_codes:
        raise AgentActionError(
            f"{name}.tile_code={tile_code} is not in legal tile codes {legal_tile_codes}"
        )
    return {"position": position, "tile_code": tile_code}


def validate_turn_action(request: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict):
        raise AgentActionError("agent response must be a dict")
    if "slide_action" not in response:
        raise AgentActionError("agent response must contain slide_action")

    action = int(response["slide_action"])
    if action not in [int(x) for x in request["legal_slide_actions"]]:
        raise AgentActionError(f"slide_action={action} is illegal for this turn")

    preview = _preview_by_action(request, action)
    legal_tile_codes = [int(x) for x in request["legal_tile_codes"]]
    place_a = _normalize_placement("place_a", response.get("place_a"), preview, legal_tile_codes)
    place_b = _normalize_placement("place_b", response.get("place_b"), preview, legal_tile_codes)
    return {"slide_action": action, "place_a": place_a, "place_b": place_b}


def forfeit_result(
    loser_side: int,
    turns: int,
    side_to_agent_index: Tuple[int, int],
    reason: str,
) -> MatchResult:
    winner_side = 1 - int(loser_side)
    return MatchResult(
        winner_side=winner_side,
        loser_side=int(loser_side),
        winner_agent_index=side_to_agent_index[winner_side],
        loser_agent_index=side_to_agent_index[loser_side],
        turns=int(turns),
        ended_by_forfeit=True,
        forfeit_side=int(loser_side),
        forfeit_reason=reason,
    )


def play_match(
    agent_for_side_a: SubprocessJSONAgent,
    agent_for_side_b: SubprocessJSONAgent,
    env_class,
    seed: int = 0,
    verbose: bool = False,
    side_to_agent_index: Tuple[int, int] = (0, 1),
    action_timeout_sec: Optional[float] = 3.0,
) -> MatchResult:
    env = env_class(seed=seed)
    agents = [agent_for_side_a, agent_for_side_b]

    while not env.done:
        cur_side = int(env.current_player)
        request = build_turn_request(env)

        try:
            response = agents[cur_side].get_action(request, timeout_sec=action_timeout_sec)
            action = validate_turn_action(request, response)
        except (AgentTimeoutError, AgentActionError, TimeoutError, ValueError, TypeError, KeyError) as exc:
            return forfeit_result(
                loser_side=cur_side,
                turns=int(request["turn_index"]),
                side_to_agent_index=side_to_agent_index,
                reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return forfeit_result(
                loser_side=cur_side,
                turns=int(request["turn_index"]),
                side_to_agent_index=side_to_agent_index,
                reason=f"Unhandled {type(exc).__name__}: {exc}",
            )

        env.step_turn(action["slide_action"], place_a=action["place_a"], place_b=action["place_b"])

        if verbose:
            print(
                f"turn={request['turn_index']} side={'A' if cur_side == 0 else 'B'} "
                f"agent={'student' if side_to_agent_index[cur_side] == 0 else 'baseline'} "
                f"slide={ACTION_NAMES[action['slide_action']]} "
                f"place_a={action['place_a']} place_b={action['place_b']}"
            )
            print(env.render())
            print()

    winner_side = int(env.winner)
    loser_side = int(env.loser)
    return MatchResult(
        winner_side=winner_side,
        loser_side=loser_side,
        winner_agent_index=side_to_agent_index[winner_side],
        loser_agent_index=side_to_agent_index[loser_side],
        turns=int(env.turn_index),
        ended_by_forfeit=False,
        forfeit_side=None,
        forfeit_reason=None,
    )


def build_python_agent_command(agent_spec: str) -> List[str]:
    spec = str(agent_spec)
    if os.path.isfile(spec):
        lower = spec.lower()
        if lower.endswith(".py"):
            return [sys.executable, "-u", spec]
        if lower.endswith(MODULE_EXTENSIONS):
            bridge_path = os.path.join(os.path.dirname(__file__), "agent_bridge.py")
            return [sys.executable, "-u", bridge_path, "--agent", spec]
        return shlex.split(spec)
    return shlex.split(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fib2584 attack match between two subprocess agents")
    parser.add_argument("--agent-a", required=True, help="Command or agent script for side A")
    parser.add_argument("--agent-b", required=True, help="Command or agent script for side B")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--env-backend", choices=["auto", "cpp", "py"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    print(f"base_seed={args.seed}")
    env_class, env_path = load_env_class(args.env_backend)
    print(f"env_module={env_path}")

    wins_a = 0
    wins_b = 0
    total_turns = 0

    for game_idx in range(args.games):
        agent_a = SubprocessJSONAgent(build_python_agent_command(args.agent_a), "agent-a", startup_timeout_sec=args.startup_timeout)
        agent_b = SubprocessJSONAgent(build_python_agent_command(args.agent_b), "agent-b", startup_timeout_sec=args.startup_timeout)
        try:
            result = play_match(
                agent_for_side_a=agent_a,
                agent_for_side_b=agent_b,
                env_class=env_class,
                seed=args.seed + game_idx,
                verbose=args.verbose,
                side_to_agent_index=(0, 1),
                action_timeout_sec=args.timeout,
            )
        finally:
            agent_a.close()
            agent_b.close()

        total_turns += result.turns
        if result.winner_agent_index == 0:
            wins_a += 1
        else:
            wins_b += 1
        if result.ended_by_forfeit:
            print(
                f"game={game_idx:02d} winner={'A' if result.winner_agent_index == 0 else 'B'} "
                f"turns={result.turns} FORFEIT side={'A' if result.forfeit_side == 0 else 'B'} "
                f"reason={result.forfeit_reason}"
            )
        else:
            print(
                f"game={game_idx:02d} winner={'A' if result.winner_agent_index == 0 else 'B'} "
                f"turns={result.turns}"
            )

    avg_turns = total_turns / max(args.games, 1)
    print(f"wins_A={wins_a} wins_B={wins_b} avg_turns={avg_turns:.2f}")


if __name__ == "__main__":
    main()
