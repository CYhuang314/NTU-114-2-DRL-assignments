import random
from typing import Dict, List, Optional


class Fib2584Board:
    SIZE = 4
    CELL_BITS = 5
    CELL_MASK = (1 << CELL_BITS) - 1

    def __init__(self):
        self.raw = 0

    def clone(self) -> "Fib2584Board":
        b = Fib2584Board()
        b.raw = self.raw
        return b

    def clear(self) -> None:
        self.raw = 0

    def at_code(self, idx: int) -> int:
        return (self.raw >> (idx * self.CELL_BITS)) & self.CELL_MASK

    def set_code(self, idx: int, code: int) -> None:
        mask = self.CELL_MASK << (idx * self.CELL_BITS)
        self.raw = (self.raw & ~mask) | ((code & self.CELL_MASK) << (idx * self.CELL_BITS))

    @staticmethod
    def tile_value_from_code(code: int) -> int:
        fib = [0, 1, 2]
        while len(fib) <= max(code, 2):
            fib.append(fib[-1] + fib[-2])
        return fib[code]

    def tile_value(self, idx: int) -> int:
        return self.tile_value_from_code(self.at_code(idx))

    def empty_count(self) -> int:
        return sum(1 for i in range(16) if self.at_code(i) == 0)

    def empty_positions(self) -> List[int]:
        return [i for i in range(16) if self.at_code(i) == 0]

    def to_values_flat(self) -> List[int]:
        return [self.tile_value(i) for i in range(16)]

    def to_values_grid(self) -> List[List[int]]:
        vals = self.to_values_flat()
        return [vals[i * 4:(i + 1) * 4] for i in range(4)]

    def to_codes_flat(self) -> List[int]:
        return [self.at_code(i) for i in range(16)]

    def init(self, rng: random.Random) -> None:
        self.clear()
        self.popup(rng)
        self.popup(rng)

    def popup(self, rng: random.Random) -> bool:
        empty = self.empty_positions()
        if not empty:
            return False
        pos = rng.choice(empty)
        code = 1 if rng.random() < 0.9 else 2
        self.set_code(pos, code)
        return True

    def place_tile(self, pos: int, code: int) -> bool:
        if pos < 0 or pos >= 16:
            raise ValueError("position must be in [0, 15]")
        if code not in (1, 2):
            raise ValueError("tile code must be 1 or 2")
        if self.at_code(pos) != 0:
            return False
        self.set_code(pos, code)
        return True

    @staticmethod
    def _can_merge(a: int, b: int) -> bool:
        if a == 0 or b == 0:
            return False
        if a == 1 and b == 1:
            return True
        return abs(a - b) == 1

    @staticmethod
    def _merged_code(a: int, b: int) -> int:
        if a == 1 and b == 1:
            return 2
        return max(a, b) + 1

    @classmethod
    def _move_left_codes(cls, row: List[int]) -> int:
        out = [0, 0, 0, 0]
        top = 0
        hold = 0
        score = 0
        for tile in row:
            if tile == 0:
                continue
            if hold == 0:
                hold = tile
                continue
            if cls._can_merge(hold, tile):
                merged = cls._merged_code(hold, tile)
                out[top] = merged
                top += 1
                score += cls.tile_value_from_code(merged)
                hold = 0
            else:
                out[top] = hold
                top += 1
                hold = tile
        if hold != 0:
            out[top] = hold
        row[:] = out
        return score

    def move_left(self) -> int:
        prev = self.raw
        score = 0
        for r in range(4):
            row = [self.at_code(r * 4 + c) for c in range(4)]
            score += self._move_left_codes(row)
            for c in range(4):
                self.set_code(r * 4 + c, row[c])
        return -1 if self.raw == prev else score

    def move_right(self) -> int:
        prev = self.raw
        score = 0
        for r in range(4):
            row = [self.at_code(r * 4 + c) for c in range(4)][::-1]
            score += self._move_left_codes(row)
            row = row[::-1]
            for c in range(4):
                self.set_code(r * 4 + c, row[c])
        return -1 if self.raw == prev else score

    def move_up(self) -> int:
        prev = self.raw
        score = 0
        for c in range(4):
            row = [self.at_code(r * 4 + c) for r in range(4)]
            score += self._move_left_codes(row)
            for r in range(4):
                self.set_code(r * 4 + c, row[r])
        return -1 if self.raw == prev else score

    def move_down(self) -> int:
        prev = self.raw
        score = 0
        for c in range(4):
            row = [self.at_code(r * 4 + c) for r in range(4)][::-1]
            score += self._move_left_codes(row)
            row = row[::-1]
            for r in range(4):
                self.set_code(r * 4 + c, row[r])
        return -1 if self.raw == prev else score

    def move(self, action: int) -> int:
        if action == 0:
            return self.move_up()
        if action == 1:
            return self.move_down()
        if action == 2:
            return self.move_left()
        if action == 3:
            return self.move_right()
        raise ValueError("action must be one of {0,1,2,3}")

    def is_move_legal(self, action: int) -> bool:
        b = self.clone()
        return b.move(action) >= 0

    def legal_actions(self) -> List[int]:
        return [a for a in range(4) if self.is_move_legal(a)]

    def has_legal_move(self) -> bool:
        return any(self.is_move_legal(a) for a in range(4))

    def to_string(self) -> str:
        rows = ["+----------------------------+"]
        for r in range(4):
            row = "|" + "".join(f"{self.tile_value(r * 4 + c):7d}" for c in range(4)) + "|"
            rows.append(row)
        rows.append("+----------------------------+")
        return "\n".join(rows)


class Fib2584AttackEnv:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.board_a = Fib2584Board()
        self.board_b = Fib2584Board()
        self.reset()

    def clone(self) -> "Fib2584AttackEnv":
        env = Fib2584AttackEnv(0)
        env.rng.setstate(self.rng.getstate())
        env.board_a = self.board_a.clone()
        env.board_b = self.board_b.clone()
        env.current_player = self.current_player
        env.turn_index = self.turn_index
        env.done = self.done
        env.winner = self.winner
        env.loser = self.loser
        env.last_slide_action = self.last_slide_action
        env.last_slide_reward_a = self.last_slide_reward_a
        env.last_slide_reward_b = self.last_slide_reward_b
        env.last_slide_moved_a = self.last_slide_moved_a
        env.last_slide_moved_b = self.last_slide_moved_b
        return env

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.board_a.init(self.rng)
        self.board_b.init(self.rng)
        self.current_player = 0
        self.turn_index = 0
        self.done = False
        self.winner = -1
        self.loser = -1
        self.last_slide_action = -1
        self.last_slide_reward_a = 0
        self.last_slide_reward_b = 0
        self.last_slide_moved_a = False
        self.last_slide_moved_b = False
        self._resolve_terminal_before_turn()
        return self.observation()

    def _current_board(self) -> Fib2584Board:
        return self.board_a if self.current_player == 0 else self.board_b

    def _other_board(self) -> Fib2584Board:
        return self.board_b if self.current_player == 0 else self.board_a

    def legal_slide_actions(self) -> List[int]:
        if self.done:
            return []
        return self._current_board().legal_actions()

    def observation(self) -> Dict:
        return {
            "board_a": self.board_a.to_values_grid(),
            "board_b": self.board_b.to_values_grid(),
            "board_a_codes": self.board_a.to_codes_flat(),
            "board_b_codes": self.board_b.to_codes_flat(),
            "current_player": self.current_player,
            "current_player_name": "A" if self.current_player == 0 else "B",
            "phase": "done" if self.done else "slide",
            "phase_id": 1 if self.done else 0,
            "turn_index": self.turn_index,
            "done": self.done,
            "winner": self.winner,
            "winner_name": None if self.winner == -1 else ("A" if self.winner == 0 else "B"),
            "loser": self.loser,
            "loser_name": None if self.loser == -1 else ("A" if self.loser == 0 else "B"),
            "last_slide_action": self.last_slide_action,
            "last_slide_reward_a": self.last_slide_reward_a,
            "last_slide_reward_b": self.last_slide_reward_b,
            "last_slide_moved_a": self.last_slide_moved_a,
            "last_slide_moved_b": self.last_slide_moved_b,
            "empty_positions_a": self.board_a.empty_positions(),
            "empty_positions_b": self.board_b.empty_positions(),
            "legal_slide_actions": self.legal_slide_actions(),
            "legal_tile_codes": [1, 2],
        }

    def preview_turn(self, action: int) -> Dict:
        if self.done:
            raise RuntimeError("cannot preview after game is done")
        if action not in self.legal_slide_actions():
            raise ValueError("slide action is illegal on current player's board")

        ba = self.board_a.clone()
        bb = self.board_b.clone()
        reward_a = ba.move(action)
        moved_a = reward_a >= 0
        if not moved_a:
            reward_a = 0
        reward_b = bb.move(action)
        moved_b = reward_b >= 0
        if not moved_b:
            reward_b = 0

        return {
            "slide_action": action,
            "board_a_after_slide": ba.to_values_grid(),
            "board_b_after_slide": bb.to_values_grid(),
            "board_a_codes_after_slide": ba.to_codes_flat(),
            "board_b_codes_after_slide": bb.to_codes_flat(),
            "empty_positions_a": ba.empty_positions(),
            "empty_positions_b": bb.empty_positions(),
            "must_skip_place_a": ba.empty_count() == 0,
            "must_skip_place_b": bb.empty_count() == 0,
            "slide_reward_a": reward_a,
            "slide_reward_b": reward_b,
            "slide_moved_a": moved_a,
            "slide_moved_b": moved_b,
        }

    def _apply_placement(self, board: Fib2584Board, placement) -> None:
        if board.empty_count() == 0:
            if placement is not None:
                raise ValueError("placement must be None when target board has no empty cells")
            return
        if placement is None:
            raise ValueError("placement cannot be None when target board still has empty cells")
        position = int(placement["position"])
        tile_code = int(placement["tile_code"])
        if not board.place_tile(position, tile_code):
            raise ValueError("invalid placement")

    def step_turn(self, action: int, place_a=None, place_b=None):
        if self.done:
            raise RuntimeError("cannot act after game is done")
        if action not in self.legal_slide_actions():
            raise ValueError("slide action is illegal on current player's board")

        self.last_slide_action = action
        self.last_slide_reward_a = self.board_a.move(action)
        self.last_slide_moved_a = self.last_slide_reward_a >= 0
        if not self.last_slide_moved_a:
            self.last_slide_reward_a = 0
        self.last_slide_reward_b = self.board_b.move(action)
        self.last_slide_moved_b = self.last_slide_reward_b >= 0
        if not self.last_slide_moved_b:
            self.last_slide_reward_b = 0

        self._apply_placement(self.board_a, place_a)
        self._apply_placement(self.board_b, place_b)
        self._finalize_turn()
        return self.observation()

    def _resolve_terminal_before_turn(self):
        cur_can = self._current_board().has_legal_move()
        other_can = self._other_board().has_legal_move()
        if cur_can or other_can:
            return
        self.done = True
        self.winner = 1 - self.current_player
        self.loser = self.current_player

    def _finalize_turn(self):
        a_can = self.board_a.has_legal_move()
        b_can = self.board_b.has_legal_move()
        if not a_can and not b_can:
            self.done = True
            self.winner = 1 - self.current_player
            self.loser = self.current_player
            return
        if a_can and not b_can:
            self.done = True
            self.winner = 0
            self.loser = 1
            return
        if not a_can and b_can:
            self.done = True
            self.winner = 1
            self.loser = 0
            return
        self.current_player = 1 - self.current_player
        self.turn_index += 1
        self._resolve_terminal_before_turn()

    def render(self) -> str:
        lines = [f"Turn: {self.turn_index} | Current player: {'A' if self.current_player == 0 else 'B'}"]
        if self.done:
            lines[0] += f" | Winner: {'A' if self.winner == 0 else 'B'}"
        lines.append("[Board A]")
        lines.append(self.board_a.to_string())
        lines.append("")
        lines.append("[Board B]")
        lines.append(self.board_b.to_string())
        return "\n".join(lines)
