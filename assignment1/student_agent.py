"""
Student Agent for DynamicTaxi Environment.
Loads a pre-trained Q-table and selects actions using get_state.
"""

import numpy as np
import pickle
import os

# ============================================================
# Import get_state (same module used during training)
# ============================================================
from get_state import get_state

# ============================================================
# Load Q-table
# ============================================================
NUM_ACTIONS = 8

# Try loading best model first, fallback to regular
_q_table_file = None
for fname in ["q_table_best.pkl", "q_table.pkl"]:
    if os.path.exists(fname):
        _q_table_file = fname
        break

Q_table = {}
with open(_q_table_file, "rb") as f:
    Q_table = pickle.load(f)

# ============================================================
# Action selection
# ============================================================
def get_action(obs):
    """
    Select the best action given the current observation.
    Falls back to uniform random if state not in Q-table.
    """
    
    state = get_state(obs)

    if state in Q_table:
        return int(np.argmax(Q_table[state]))
    else:
        return np.random.randint(0, NUM_ACTIONS)
