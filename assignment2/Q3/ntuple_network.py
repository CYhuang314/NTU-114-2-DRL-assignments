"""
N-Tuple Network for Fib-2584.
- Clamps tile codes to [0, MAX_CODE] once at input to prevent index overflow
- Pure-Python Horner evaluate/update for speed
- Batch evaluate for action selection
"""

import numpy as np

NUM_CODES = 21
MAX_CODE = NUM_CODES - 1  # = 20

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

# Precompute clamp table: maps any code 0..31 to min(code, MAX_CODE)
_CLAMP = [min(i, MAX_CODE) for i in range(32)]


class NTupleNetwork:
    def __init__(self):
        self.luts = []
        for syms, size in TUPLE_GROUPS:
            self.luts.append(np.zeros(NUM_CODES ** size, dtype=np.float32))

        NC = NUM_CODES

        # Precompute for fast single evaluate/update (pure Python)
        self._syms_6 = []
        self._syms_4 = []
        for g, (syms, size) in enumerate(TUPLE_GROUPS):
            for sym in syms:
                if size == 6:
                    self._syms_6.append((self.luts[g], sym[0], sym[1], sym[2], sym[3], sym[4], sym[5]))
                else:
                    self._syms_4.append((self.luts[g], sym[0], sym[1], sym[2], sym[3]))

        self._NC = NC

        # Precompute for batch evaluate (numpy vectorized)
        self._gpos = []
        self._gpow = []
        for syms, size in TUPLE_GROUPS:
            self._gpos.append(np.array(syms, dtype=np.int32))
            self._gpow.append(np.array([NC ** i for i in range(size)], dtype=np.int64))

    @staticmethod
    def clamp_codes(state_codes):
        """Clamp a list of 16 codes to [0, MAX_CODE]. Fast via lookup table."""
        cl = _CLAMP
        return [cl[state_codes[0]], cl[state_codes[1]], cl[state_codes[2]], cl[state_codes[3]],
                cl[state_codes[4]], cl[state_codes[5]], cl[state_codes[6]], cl[state_codes[7]],
                cl[state_codes[8]], cl[state_codes[9]], cl[state_codes[10]], cl[state_codes[11]],
                cl[state_codes[12]], cl[state_codes[13]], cl[state_codes[14]], cl[state_codes[15]]]

    def evaluate(self, state_codes):
        """Fast single-state evaluation. state_codes: list of 16 ints (already clamped)."""
        c = state_codes
        NC = self._NC
        total = 0.0
        for lut, p0, p1, p2, p3, p4, p5 in self._syms_6:
            total += lut[c[p0] + NC*(c[p1] + NC*(c[p2] + NC*(c[p3] + NC*(c[p4] + NC*c[p5]))))]
        for lut, p0, p1, p2, p3 in self._syms_4:
            total += lut[c[p0] + NC*(c[p1] + NC*(c[p2] + NC*c[p3]))]
        return total

    def evaluate_batch(self, states_codes):
        """Evaluate multiple states. states_codes: numpy (N, 16) int array."""
        clamped = np.clip(states_codes, 0, MAX_CODE)
        values = np.zeros(clamped.shape[0], dtype=np.float64)
        for g in range(len(TUPLE_GROUPS)):
            gathered = clamped[:, self._gpos[g]]
            indices = gathered @ self._gpow[g]
            values += self.luts[g][indices].sum(axis=1)
        return values

    def update(self, state_codes, delta):
        """Update LUT weights. state_codes: list of 16 ints (already clamped)."""
        c = state_codes
        NC = self._NC
        for lut, p0, p1, p2, p3, p4, p5 in self._syms_6:
            lut[c[p0] + NC*(c[p1] + NC*(c[p2] + NC*(c[p3] + NC*(c[p4] + NC*c[p5]))))] += delta
        for lut, p0, p1, p2, p3 in self._syms_4:
            lut[c[p0] + NC*(c[p1] + NC*(c[p2] + NC*c[p3]))] += delta

    def save(self, filepath):
        np.savez_compressed(filepath, **{f'lut_{i}': l for i, l in enumerate(self.luts)})

    def load(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        data = np.load(filepath)
        for i in range(len(self.luts)):
            key = f'lut_{i}'
            if key in data:
                self.luts[i] = data[key].astype(np.float32)
            else:
                raise KeyError(f"Missing {key} in {filepath}")
        self._syms_6 = []
        self._syms_4 = []
        for g, (syms, size) in enumerate(TUPLE_GROUPS):
            for sym in syms:
                if size == 6:
                    self._syms_6.append((self.luts[g], sym[0], sym[1], sym[2], sym[3], sym[4], sym[5]))
                else:
                    self._syms_4.append((self.luts[g], sym[0], sym[1], sym[2], sym[3]))

    def memory_usage_mb(self):
        return sum(l.nbytes for l in self.luts) / (1024**2)

    def info(self):
        print("N-Tuple Network Architecture:")
        print(f"  Codes per cell: {NUM_CODES} (max code: {MAX_CODE})")
        tw = 0
        for i, (syms, size) in enumerate(TUPLE_GROUPS):
            nw = NUM_CODES ** size
            tw += nw
            print(f"  Group {i}: {size}-tuple, {len(syms)} syms, {nw:,} weights ({nw*4/1024**2:.1f} MB)")
        print(f"  Total: {tw:,} weights, {self.memory_usage_mb():.1f} MB, {sum(len(s) for s,_ in TUPLE_GROUPS)} lookups/eval")


if __name__ == "__main__":
    import time
    net = NTupleNetwork()
    net.info()

    codes = list(np.random.randint(0, NUM_CODES, size=16))
    N = 100000

    t0 = time.perf_counter()
    for _ in range(N):
        net.evaluate(codes)
    t1 = time.perf_counter()
    print(f"\nevaluate: {(t1-t0)/N*1e6:.1f} us/call")

    t0 = time.perf_counter()
    for _ in range(N):
        net.update(codes, 0.001)
    t1 = time.perf_counter()
    print(f"update:   {(t1-t0)/N*1e6:.1f} us/call")

    # Test clamping
    codes_overflow = list(np.random.randint(0, 25, size=16))
    clamped = NTupleNetwork.clamp_codes(codes_overflow)
    try:
        v = net.evaluate(clamped)
        print(f"\nClamping test: OK (no crash)")
    except Exception as e:
        print(f"\nClamping test FAILED: {e}")

    batch_overflow = np.random.randint(0, 25, size=(4, 16), dtype=np.int64)
    try:
        v = net.evaluate_batch(batch_overflow)
        print(f"Batch clamping test: OK")
    except Exception as e:
        print(f"Batch clamping test FAILED: {e}")
