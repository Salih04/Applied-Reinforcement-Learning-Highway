from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import DQN

from src.config import build_train_config
from src.rewards import RewardConfig, RewardShapingWrapper
from src.utils import repo_root, seed_env


def make_env(cfg, seed: int) -> gym.Env:
    env = gym.make(cfg.env_id, config=cfg.env_config)
    r_cfg = RewardConfig(
        alpha_speed=cfg.alpha_speed,
        beta_right_lane=cfg.beta_right_lane,
        gamma_crash=cfg.gamma_crash,
        delta_unsafe=cfg.delta_unsafe,
        lambda_lane_change=cfg.lambda_lane_change,
        unsafe_distance_m=cfg.unsafe_distance_m,
        reward_speed_range=tuple(cfg.env_config.get("reward_speed_range", [20, 30])),
    )
    env = RewardShapingWrapper(env, r_cfg)
    seed_env(env, seed)
    return env


def evaluate(model_path: Path, episodes: int = 5, seed: int = 42) -> None:
    cfg = build_train_config()
    env = make_env(cfg, seed=seed)

    model = DQN.load(str(model_path), env=env)

    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        terminated = truncated = False
        ep_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)

        returns.append(ep_return)

    env.close()
    arr = np.array(returns, dtype=np.float32)
    print(f"Model: {model_path.name}")
    print(f"Episodes: {episodes}")
    print(f"Mean return: {arr.mean():.2f} | Std: {arr.std():.2f}")


def main() -> None:
    root = repo_root()
    model_path = root / "models" / "dqn_full.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    evaluate(model_path)


if __name__ == "__main__":
    main()
