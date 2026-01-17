from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class TrainConfig:
    # repo paths
    repo_root: str = "."

    # env
    env_id: str = "highway-fast-v0"
    env_config: Dict[str, Any] = None  # set in build_train_config()

    # reproducibility
    seed: int = 42

    # timesteps
    total_timesteps: int = 300_000
    half_timesteps: int = 150_000

    # DQN hyperparams
    learning_rate: float = 5e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05

    # MLP architecture
    net_arch_1: int = 256
    net_arch_2: int = 256

    # reward shaping parameters (your wrapper)
    alpha_speed: float = 1.0
    beta_right_lane: float = 0.2
    gamma_crash: float = 2.0
    delta_unsafe: float = 0.5
    lambda_lane_change: float = 0.05
    unsafe_distance_m: float = 10.0


def build_train_config() -> TrainConfig:
    # default highway-env config (fast with CPU)
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-30, 30], "vy": [-30, 30]},
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "vehicles_count": 30,
        "duration": 40,
        "reward_speed_range": [20, 30],
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
    }

    return TrainConfig(
        repo_root=str(Path(".").resolve()),
        env_config=env_config,
    )
