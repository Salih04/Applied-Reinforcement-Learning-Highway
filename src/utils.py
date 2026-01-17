from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional
import numpy as np
import gymnasium as gym

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_env(env: gym.Env, seed: int) -> None:
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

def repo_root() -> Path:
    # project root
    return Path(__file__).resolve().parent.parent

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
