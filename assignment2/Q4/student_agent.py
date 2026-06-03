"""
Battle Fib-2584 Agent using N-Tuple Network.

Reads JSON requests from stdin, outputs JSON actions to stdout.
Uses 1-ply search: for each legal slide, evaluates all placement combos,
picks the action that maximises V(own_board) - V(opp_board).
"""

import json
import sys
import os
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Inline N-Tuple Network (to avoid import issues in eval environment)
# ---------------------------------------------------------------------------

NUM_CODES = 18
MAX_CODE = NUM_CODES - 1

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

TUPLE_GROUPS = [
    (_generate_symmetries((0, 1, 2, 4, 5, 6)), 6),
    (_generate_symmetries((4, 5, 6, 8, 9, 10)), 6),
    (_generate_symmetries((0, 1, 2, 3)), 4),
    (_generate_symmetries((4, 5, 6, 7)), 4),
]

_CLAMP = [min(i, MAX_CODE) for i in range(32)]


class NTupleNetwork:
    def __init__(self):
        self.luts = []
        for syms, size in TUPLE_GROUPS:
            self.luts.append(np.zeros(NUM_CODES ** size, dtype=np.float32))
        self._NC = NUM_CODES
        self._rebuild_sym_refs()

    def _rebuild_sym_refs(self):
        self._syms_6 = []
        self._syms_4 = []
        for g, (syms, size) in enumerate(TUPLE_GROUPS):
            for sym in syms:
                if size == 6:
                    self._syms_6.append((self.luts[g], sym[0], sym[1], sym[2], sym[3], sym[4], sym[5]))
                else:
                    self._syms_4.append((self.luts[g], sym[0], sym[1], sym[2], sym[3]))

    def evaluate(self, state_codes):
        c = state_codes
        NC = self._NC
        total = 0.0
        for lut, p0, p1, p2, p3, p4, p5 in self._syms_6:
            total += lut[c[p0] + NC * (c[p1] + NC * (c[p2] + NC * (c[p3] + NC * (c[p4] + NC * c[p5]))))]
        for lut, p0, p1, p2, p3 in self._syms_4:
            total += lut[c[p0] + NC * (c[p1] + NC * (c[p2] + NC * c[p3]))]
        return total

    def load(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        data = np.load(filepath)
        for i in range(len(self.luts)):
            key = f'lut_{i}'
            if key in data:
                self.luts[i] = data[key].astype(np.float32)
        self._rebuild_sym_refs()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def clamp_codes(codes):
    cl = _CLAMP
    return [cl[codes[0]], cl[codes[1]], cl[codes[2]], cl[codes[3]],
            cl[codes[4]], cl[codes[5]], cl[codes[6]], cl[codes[7]],
            cl[codes[8]], cl[codes[9]], cl[codes[10]], cl[codes[11]],
            cl[codes[12]], cl[codes[13]], cl[codes[14]], cl[codes[15]]]


def codes_with_tile(codes_list, pos, tile_code):
    """Return a new clamped code list with tile_code placed at pos."""
    new = list(codes_list)
    new[pos] = min(tile_code, MAX_CODE)
    return new


class Agent:
    def __init__(self):
        self.net = NTupleNetwork()
        # Load weights from same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "weights.npz")
        if os.path.exists(weights_path):
            self.net.load(weights_path)

    def get_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        current_player = request["current_player"]
        legal_slides = request["legal_slide_actions"]
        legal_tiles = request["legal_tile_codes"]
        previews = request["action_previews"]

        best_score = -1e18
        best_result = None

        for preview in previews:
            slide_action = preview["slide_action"]

            # Board codes after slide (before placement)
            codes_a_after = clamp_codes(preview["board_a_codes_after_slide"])
            codes_b_after = clamp_codes(preview["board_b_codes_after_slide"])

            empty_a = preview["empty_positions_a"]
            empty_b = preview["empty_positions_b"]
            skip_a = preview["must_skip_place_a"]
            skip_b = preview["must_skip_place_b"]

            # Determine which board is ours vs opponent's
            if current_player == 0:
                own_codes = codes_a_after
                opp_codes = codes_b_after
                own_empty = empty_a
                opp_empty = empty_b
                own_skip = skip_a
                opp_skip = skip_b
            else:
                own_codes = codes_b_after
                opp_codes = codes_a_after
                own_empty = empty_b
                opp_empty = empty_a
                own_skip = skip_b
                opp_skip = skip_a

            # Find best placement for own board (maximise V)
            if own_skip:
                best_own_val = self.net.evaluate(own_codes)
                best_own_place = None
            else:
                best_own_val = -1e18
                best_own_place = {"position": own_empty[0], "tile_code": legal_tiles[0]}
                for pos in own_empty:
                    for tc in legal_tiles:
                        c = codes_with_tile(own_codes, pos, tc)
                        v = self.net.evaluate(c)
                        if v > best_own_val:
                            best_own_val = v
                            best_own_place = {"position": pos, "tile_code": tc}

            # Find worst placement for opponent board (minimise V)
            if opp_skip:
                best_opp_val = self.net.evaluate(opp_codes)
                best_opp_place = None
            else:
                best_opp_val = 1e18
                best_opp_place = {"position": opp_empty[0], "tile_code": legal_tiles[0]}
                for pos in opp_empty:
                    for tc in legal_tiles:
                        c = codes_with_tile(opp_codes, pos, tc)
                        v = self.net.evaluate(c)
                        if v < best_opp_val:
                            best_opp_val = v
                            best_opp_place = {"position": pos, "tile_code": tc}

            # Score = V(own after placement) - V(opp after placement)
            # For opp, we picked the placement that minimises V(opp),
            # so best_opp_val is the minimised value. From our perspective
            # we want V(own) - V(opp) to be maximised, so lower V(opp) is better for us.
            # Wait -- lower V(opp) means opp board is worse, which is GOOD for us.
            # score = best_own_val - best_opp_val  =>  max own, min opp => CORRECT
            # Actually: -V(opp) is maximised when V(opp) is minimised. So:
            score = best_own_val - best_opp_val

            if score > best_score:
                best_score = score
                if current_player == 0:
                    best_result = {
                        "slide_action": int(slide_action),
                        "place_a": best_own_place,
                        "place_b": best_opp_place,
                    }
                else:
                    best_result = {
                        "slide_action": int(slide_action),
                        "place_a": best_opp_place,
                        "place_b": best_own_place,
                    }

        return best_result


def main() -> None:
    agent = Agent()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        response = agent.get_action(request)
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()

