# Battle Fib-2584 Agent

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cqklpiQh)

## Overview

This project implements an agent for **Battle Fib-2584**, a two-player adversarial variant of the Fibonacci-based 2048 game. Each player has their own 4×4 board. On each turn, the current player chooses a slide direction that is applied to both boards simultaneously, then places a new tile (1 or 2) on each board. The goal is to make the opponent run out of legal moves before you do.

The agent uses an **N-Tuple Network** trained via **TD(0) afterstate learning** through self-play. At inference time, it performs a 1-ply search to select the slide direction and tile placements that maximize the value gap between its own board and the opponent's board.

## File Structure

```
.
├── student_agent.py        # Inference agent (submitted for evaluation)
├── weights.npz             # Trained N-Tuple network weights
├── ntuple_network.py       # N-Tuple network definition (used by train.py)
├── train.py                # Self-play TD(0) training script
├── eval.py                 # Local evaluation runner (provided)
├── match_core.py           # Match infrastructure (provided)
├── fib2584_attack_env_py.py # Game environment (provided)
└── README.md
```

## Dependencies

- Python 3.8+
- NumPy

Install NumPy if not already available:

```bash
pip install numpy
```

No GPU or PyTorch is required. The N-Tuple network uses only NumPy arrays as lookup tables.

## How to Reproduce Training

### Step 1: Ensure all files are in the same directory

Place the following files in a single working directory:

- `ntuple_network.py`
- `train.py`
- `fib2584_attack_env_py.py`

### Step 2: Run the training script

```bash
python train.py --episodes 200000 --lr 0.0025
```

**Expected training time:** approximately 2–3 hours on a standard CPU.

**Expected output** (example log):

```
N-Tuple Network for Battle 2584:
  Codes per cell: 18 (max code: 17)
  Total: 68,234,400 weights, 260.3 MB, 32 lookups/eval

Config: episodes=200000 lr=0.0025 alpha/feat=0.000078
  eps=0.1->0.01 place_exp=0.1->0.01 terminal=50.0
  features=32 seed=42

ep=    1000/200000  p0=2518 p1=2482 (p0r=0.504)  avg_t=52.3  eps=0.0978  25.1g/s  3.3min  ETA=2.2h
ep=    2000/200000  ...
...
ep=  200000/200000  ...

Done. 200000 ep in 130.5min (2.2h)
Saved to weights.npz
```

The training produces a `weights.npz` file (~260 MB uncompressed, ~17 MB compressed).

### Step 3: Verify the trained agent

Place `student_agent.py` and `weights.npz` in the same directory as the evaluation files, then run:

```bash
python eval.py --student student_agent.py --baseline baselines/baseline_1.py --games 20 --min-wins 14 --timeout 5 --startup-timeout 15
```
