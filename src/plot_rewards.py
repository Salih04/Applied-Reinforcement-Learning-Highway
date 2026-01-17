from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from src.utils import repo_root


def load_rewards(jsonl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    episodes: List[int] = []
    rewards: List[float] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            episodes.append(int(obj["episode"]))
            rewards.append(float(obj["reward"]))
    return np.array(episodes, dtype=np.int32), np.array(rewards, dtype=np.float32)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")

def main() -> None:
    root = repo_root()
    runs_dir = root / "runs"
    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = runs_dir / "episode_rewards.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing rewards log: {jsonl_path}")
    
    ep, r = load_rewards(jsonl_path)
    downsample = max(len(r) // 2000, 1)
    ep_ds = ep[::downsample]
    r_ds = r[::downsample]
    window = 100
    r_ma = moving_average(r, window=window)
    ep_ma = ep[window - 1 :]

    plt.figure(figsize=(10, 5))
    plt.plot(ep_ds, r_ds, linewidth=1.0, alpha=0.35, label="Episode reward (downsampled)")
    plt.plot(ep_ma, r_ma, linewidth=2.0, label=f"Moving average (window={window})")
    plt.title("Training Curve: Episode Reward vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()

    out_path = assets_dir / "reward_curve.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
