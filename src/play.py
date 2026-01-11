from __future__ import annotations

import argparse
import random

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import DQN

from src.config import build_train_config
from src.wrappers.reward_wrapper import HighwayRewardWrapper, RewardParams


def seed_env(env: gym.Env, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass


def make_env(seed: int) -> gym.Env:
    cfg = build_train_config()

    env = gym.make(cfg.env_id, config=cfg.env_config, render_mode="human")

    params = RewardParams(
        alpha_speed=cfg.alpha_speed,
        beta_right_lane=cfg.beta_right_lane,
        gamma_crash=cfg.gamma_crash,
        delta_unsafe=cfg.delta_unsafe,
        lambda_lane_change=cfg.lambda_lane_change,
        unsafe_distance_m=cfg.unsafe_distance_m,
        reward_speed_range=tuple(cfg.env_config.get("reward_speed_range", [20, 30])),
    )
    env = HighwayRewardWrapper(env, params=params)

    seed_env(env, seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    env = make_env(seed=args.seed)

    if args.model:
        model = DQN.load(args.model, env=env)
    else:
        model = None  # untrained: random actions

    obs, _ = env.reset()
    done = truncated = False

    while not (done or truncated):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, done, truncated, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
