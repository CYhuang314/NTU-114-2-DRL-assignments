import random
import numpy as np
import os


def _can_merge(a: int, b: int) -> bool:
    if a == 0 or b == 0:
        return False
    if a == 1 and b == 1:
        return True
    # Fibonacci-2584 adjacency rule.
    return a + b in (2, 3) or abs(a - b) in (1, 2) and max(a, b) > 2 and (a + b) > max(a, b)


def _build_fib_set(limit: int = 1000000):
    fibs = [1, 1]
    while fibs[-1] < limit:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


_FIBS = _build_fib_set()
_FIB_PAIRS = set()
for i in range(len(_FIBS) - 1):
    a, b = _FIBS[i], _FIBS[i + 1]
    _FIB_PAIRS.add((a, b))
    _FIB_PAIRS.add((b, a))


def _mergeable(a: int, b: int) -> bool:
    if a == 0 or b == 0:
        return False
    if a == 1 and b == 1:
        return True
    return (int(a), int(b)) in _FIB_PAIRS


def _simulate_line_left(line: np.ndarray) -> np.ndarray:
    nonzero = [int(x) for x in line if int(x) != 0]
    merged = []
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and _mergeable(nonzero[i], nonzero[i + 1]):
            merged.append(nonzero[i] + nonzero[i + 1])
            i += 2
        else:
            merged.append(nonzero[i])
            i += 1
    out = np.zeros(4, dtype=np.int64)
    out[:len(merged)] = merged
    return out


def _apply_action(board: np.ndarray, action: int) -> np.ndarray:
    b = np.asarray(board, dtype=np.int64).reshape(4, 4)
    out = np.zeros_like(b)

    if action == 0:  # up
        for j in range(4):
            out[:, j] = _simulate_line_left(b[:, j])
    elif action == 1:  # down
        for j in range(4):
            out[:, j] = _simulate_line_left(b[::-1, j])[::-1]
    elif action == 2:  # left
        for i in range(4):
            out[i, :] = _simulate_line_left(b[i, :])
    elif action == 3:  # right
        for i in range(4):
            out[i, :] = _simulate_line_left(b[i, ::-1])[::-1]
    else:
        raise ValueError(f"Invalid action: {action}")

    return out


def get_legal_actions(state):
    board = np.asarray(state, dtype=np.int64).reshape(4, 4)
    legal = []
    for action in range(4):
        next_board = _apply_action(board, action)
        if not np.array_equal(next_board, board):
            legal.append(action)
    return legal


# ─────────────────────────────────────────────────────────────
# N-Tuple Network Agent
# ─────────────────────────────────────────────────────────────

NUM_CODES = 21

def _rotate90(idx):
    r, c = divmod(idx, 4)
    return c * 4 + (3 - r)

def _mirror_h(idx):
    r, c = divmod(idx, 4)
    return r * 4 + (3 - c)

def _generate_symmetries(positions):
    symmetries = []
    current = list(positions)
    for _ in range(4):
        symmetries.append(tuple(current))
        mirrored = tuple(_mirror_h(idx) for idx in current)
        symmetries.append(mirrored)
        current = [_rotate90(idx) for idx in current]
    seen = set()
    unique = []
    for sym in symmetries:
        if sym not in seen:
            seen.add(sym)
            unique.append(sym)
    return unique

_SYMS_6A = _generate_symmetries((0, 1, 2, 4, 5, 6))
_SYMS_6B = _generate_symmetries((4, 5, 6, 8, 9, 10))
_SYMS_4C = _generate_symmetries((0, 1, 2, 3))
_SYMS_4D = _generate_symmetries((4, 5, 6, 7))

_TUPLE_GROUPS = [
    (_SYMS_6A, 6),
    (_SYMS_6B, 6),
    (_SYMS_4C, 4),
    (_SYMS_4D, 4),
]

_POWERS = {
    4: np.array([NUM_CODES ** i for i in range(4)], dtype=np.int64),
    6: np.array([NUM_CODES ** i for i in range(6)], dtype=np.int64),
}

_SYM_POSITIONS = []
for _syms, _size in _TUPLE_GROUPS:
    _group = []
    for _sym in _syms:
        _group.append(np.array(_sym, dtype=np.int32))
    _SYM_POSITIONS.append(_group)

_LUTS = None


def _evaluate(codes):
    """Evaluate board from flat int64 codes array using loaded LUTs."""
    if _LUTS is None:
        return 0.0
    # Clamp codes to [0, 20] to prevent index overflow for very high tiles
    codes = np.clip(codes, 0, NUM_CODES - 1)
    total = 0.0
    for group_idx in range(len(_TUPLE_GROUPS)):
        lut = _LUTS[group_idx]
        powers = _POWERS[_TUPLE_GROUPS[group_idx][1]]
        for positions in _SYM_POSITIONS[group_idx]:
            index = int(np.dot(codes[positions], powers))
            total += lut[index]
    return total


def _load_weights():
    """Load weights from weights.npz in the same directory as this script."""
    global _LUTS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "weights.npz")
    if os.path.exists(path):
        try:
            data = np.load(path)
            luts = []
            for i in range(len(_TUPLE_GROUPS)):
                key = f'lut_{i}'
                if key not in data:
                    _LUTS = None
                    return
                luts.append(data[key].astype(np.float32))
            _LUTS = luts
        except Exception:
            _LUTS = None
    else:
        _LUTS = None


# Load weights once at import time
_load_weights()

# Create a helper env for exact afterstate/reward computation during inference
_action_env = None
try:
    from fib2584_env import Fib2584Env as _Fib2584Env
    _action_env = _Fib2584Env(0)
except ImportError:
    _action_env = None


def get_action(state, score):
    """
    Select the best action using the trained N-Tuple Network.

    Policy: argmax_a [ reward(a) + V(afterstate(a)) ]
    Falls back to random legal action if weights are not loaded.

    Args:
        state: 4x4 numpy array of tile values
        score: current game score

    Returns:
        int: action (0=up, 1=down, 2=left, 3=right)
    """
    # Fallback if no weights loaded
    if _LUTS is None:
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return 0
        return random.choice(legal_actions)

    # Use C++ env for exact reward and afterstate computation
    if _action_env is not None:
        return _get_action_with_env(state, score)

    # Fallback without C++ env
    return _get_action_without_env(state, score)


def _get_action_with_env(state, score):
    """Select best action using C++ env for exact afterstate and reward."""
    _action_env.set_state(np.asarray(state, dtype=np.int64), int(score))

    best_action = None
    best_value = -float('inf')

    for action in range(4):
        if not _action_env.is_move_legal(action):
            continue
        after_codes, reward, moved = _action_env.simulate_step_codes(action)
        if not moved:
            continue
        after_codes = np.asarray(after_codes, dtype=np.int64)
        value = reward + _evaluate(after_codes)
        if value > best_value:
            best_value = value
            best_action = action

    if best_action is not None:
        return best_action

    # Should not reach here, but fallback
    legal_actions = get_legal_actions(state)
    return random.choice(legal_actions) if legal_actions else 0


def _get_action_without_env(state, score):
    """Fallback: select best action using Python simulation (no exact reward)."""
    fib_vals = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                987, 1597, 2584, 4181, 6765, 10946]
    v2c = {v: i for i, v in enumerate(fib_vals)}

    board_flat = np.asarray(state, dtype=np.int64).ravel()
    best_action = None
    best_value = -float('inf')

    for action in range(4):
        next_board = _apply_action(state, action)
        next_flat = next_board.ravel()
        if np.array_equal(next_flat, board_flat):
            continue
        codes = np.zeros(16, dtype=np.int64)
        for i in range(16):
            codes[i] = v2c.get(int(next_flat[i]), 0)
        value = _evaluate(codes)
        if value > best_value:
            best_value = value
            best_action = action

    if best_action is not None:
        return best_action

    legal_actions = get_legal_actions(state)
    return random.choice(legal_actions) if legal_actions else 0
