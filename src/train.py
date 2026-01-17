from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from src.config import TrainConfig, build_train_config
from src.rewards import RewardConfig, RewardShapingWrapper
from src.utils import ensure_dirs, repo_root, seed_env, seed_everything


class EpisodeRewardLogger(BaseCallback):
    def __init__(self, log_path: Path):
        super().__init__(verbose=0)
        self.log_path = log_path
        self._file = None
        self.episode_counter = 0

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                self.episode_counter += 1
                record = {
                    "episode": self.episode_counter,
                    "reward": float(ep["r"]),
                    "length": int(ep["l"]),
                }
                self._file.write(json.dumps(record) + "\n")
        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()

def make_env(cfg: TrainConfig, seed: int) -> gym.Env:
    env = gym.make(cfg.env_id, config=cfg.env_config)
    env = Monitor(env)
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

def train(cfg: TrainConfig) -> Tuple[Path, Path]:
    seed_everything(cfg.seed)

    root = repo_root()
    models_dir = root / "models"
    runs_dir = root / "runs"
    ensure_dirs(models_dir, runs_dir)
    env = make_env(cfg, seed=cfg.seed)
    policy_kwargs = {"net_arch": [cfg.net_arch_1, cfg.net_arch_2]}
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
    )
    # Save config
    (runs_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )
    rewards_log = runs_dir / "episode_rewards.jsonl"
    callback = EpisodeRewardLogger(rewards_log)

    half_path = models_dir / "dqn_half.zip"
    full_path = models_dir / "dqn_full.zip"

    # Half
    if half_path.exists() and half_path.stat().st_size > 0:
        print(f"[INFO] Resuming from half model: {half_path}")
        model = DQN.load(str(half_path), env=env)
    else:
        print("[INFO] Training half model...")
        model.learn(total_timesteps=cfg.half_timesteps, callback=callback, progress_bar=False)
        model.save(str(half_path))
        print(f"[OK] Saved half model: {half_path}")

    # Full
    remaining = max(cfg.total_timesteps - cfg.half_timesteps, 0)
    print(f"[INFO] Training full model, remaining steps: {remaining}")
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callback, progress_bar=False)

    model.save(str(full_path))
    print(f"[OK] Saved full model: {full_path}")
    env.close()
    return half_path, full_path

def main() -> None:
    cfg = build_train_config()
    half_path, full_path = train(cfg)
    print(f"\nHalf: {half_path}\nFull: {full_path}\n")

if __name__ == "__main__":
    main()
