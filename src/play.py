from __future__ import annotations
import argparse
from pathlib import Path
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from src.config import build_train_config
from src.rewards import wrap_with_shaping
from src.utils import seed_env, repo_root


def make_env(cfg, seed: int, record: bool, model_path: str | None) -> gym.Env:
    render_mode = "rgb_array" if record else "human"
    env = gym.make(cfg.env_id, config=cfg.env_config, render_mode=render_mode)

    # reward shaping
    env = wrap_with_shaping(env, cfg)
    seed_env(env, seed)

    if record:
        # decide stage name
        stage = "untrained"
        if model_path:
            if "half" in model_path:
                stage = "half_trained"
            elif "full" in model_path:
                stage = "fully_trained"

        video_dir = repo_root() / "assets" / "videos" / stage
        video_dir.mkdir(parents=True, exist_ok=True)

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    cfg = build_train_config()
    env = make_env(
        cfg,
        seed=args.seed,
        record=args.record,
        model_path=args.model,
    )
    model = None
    if args.model:
        model = DQN.load(args.model, env=env)
    obs, _ = env.reset()
    done = truncated = False

    while not (done or truncated):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)

        env.render()

    env.close()

    if args.record:
        print("[OK] Video saved under assets/videos/")


if __name__ == "__main__":
    main()
