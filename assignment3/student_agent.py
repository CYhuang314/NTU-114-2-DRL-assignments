from pathlib import Path
_DIR = Path(__file__).parent

import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque

# ─────────────────────── Network (must match train.py) ─────────

class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feat = self.features(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

# ─────────────────────── Preprocessing ─────────────────────────

IMG_SIZE = 84
FRAME_STACK_K = 4

def preprocess(obs: np.ndarray) -> np.ndarray:
    """Convert RGB (240,320,3) -> grayscale float32 (84,84)."""
    if obs.ndim == 3 and obs.shape[-1] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs.squeeze()
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

# Implement your agent logic here
class StudentAgent:
    def __init__(self, action_space):
        # Called once. Load weights here.
        self.action_space = action_space
        self.device = torch.device("cpu")

        self.model = DuelingDQN(in_channels=FRAME_STACK_K, n_actions=action_space.n)
        weights_path = _DIR / "weights.pth"
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.frames = deque(maxlen=FRAME_STACK_K)

    def reset(self):
        # Called before each episode. Clear internal state here.
        self.frames.clear()

    def act(self, obs) -> int:
        # Called every timestep. Return an integer action.
        frame = preprocess(obs)

        # Initialize frame stack if empty (first step of episode)
        if len(self.frames) == 0:
            for _ in range(FRAME_STACK_K):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        stacked = np.stack(self.frames, axis=0)  # (4, 84, 84)
        state_t = torch.FloatTensor(stacked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_t)
            action = q_values.argmax(dim=1).item()

        return action
