from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np


@dataclass
class RewardConfig:
    alpha_speed: float = 1.0
    beta_right_lane: float = 0.2
    gamma_crash: float = 2.0
    delta_unsafe: float = 0.5
    lambda_lane_change: float = 0.05
    unsafe_distance_m: float = 10.0
    reward_speed_range: Tuple[float, float] = (20.0, 30.0)


class RewardShapingWrapper(gym.Wrapper):
    """
    Simple reward shaping wrapper for highway-env.

    Adds:
    - speed reward (normalized)
    - right lane reward
    - crash penalty
    - unsafe distance penalty (based on closest front vehicle)
    - lane change penalty
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self._last_lane_index: Optional[int] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_lane_index = self._get_lane_index()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped = 0.0
        shaped += self.cfg.alpha_speed * self._speed_reward()
        shaped += self.cfg.beta_right_lane * self._right_lane_reward()
        shaped -= self.cfg.gamma_crash * (1.0 if self._crashed(info) else 0.0)
        shaped -= self.cfg.delta_unsafe * (1.0 if self._unsafe_gap() else 0.0)
        shaped -= self.cfg.lambda_lane_change * (1.0 if self._lane_changed() else 0.0)

        # keep original reward too (optional, but helps stability)
        total = float(reward) + float(shaped)
        return obs, total, terminated, truncated, info

    def _vehicle(self):
        # highway-env exposes `vehicle` on unwrapped envs
        return getattr(self.env.unwrapped, "vehicle", None)

    def _crashed(self, info: dict) -> bool:
        v = self._vehicle()
        if v is not None and hasattr(v, "crashed"):
            return bool(v.crashed)
        return bool(info.get("crashed", False))

    def _speed_reward(self) -> float:
        v = self._vehicle()
        if v is None or not hasattr(v, "speed"):
            return 0.0
        vmin, vmax = self.cfg.reward_speed_range
        if vmax <= vmin:
            return 0.0
        return float(np.clip((v.speed - vmin) / (vmax - vmin), 0.0, 1.0))

    def _get_lane_index(self) -> Optional[int]:
        v = self._vehicle()
        if v is None:
            return None
        lane_index = getattr(v, "lane_index", None)
        if lane_index is None:
            return None
        # lane_index can be tuple like (road_id, lane_id, index)
        if isinstance(lane_index, tuple) and len(lane_index) >= 3:
            return int(lane_index[2])
        if isinstance(lane_index, int):
            return int(lane_index)
        return None

    def _lane_changed(self) -> bool:
        current = self._get_lane_index()
        if current is None:
            return False
        changed = (self._last_lane_index is not None) and (current != self._last_lane_index)
        self._last_lane_index = current
        return changed

    def _right_lane_reward(self) -> float:
        """
        Encourage right-most lane (higher reward when lane index is smaller/rightmost)
        Note: depending on map, lane indexing direction can differ, but for highway-fast it works reasonably.
        """
        lane_idx = self._get_lane_index()
        road = getattr(self.env.unwrapped, "road", None)
        if lane_idx is None or road is None:
            return 0.0

        # Try to estimate number of lanes
        lanes_count = getattr(self.env.unwrapped, "config", {}).get("lanes_count", None)
        if lanes_count is None:
            # fallback: count lanes from road network if possible
            try:
                lanes_count = len(road.network.graph[0][1])
            except Exception:
                lanes_count = 4

        lanes_count = max(int(lanes_count), 1)
        # right-most -> reward 1.0, left-most -> 0.0
        return float(np.clip((lanes_count - 1 - lane_idx) / max(lanes_count - 1, 1), 0.0, 1.0))

    def _unsafe_gap(self) -> bool:
        """
        Penalize if nearest front vehicle is closer than unsafe_distance_m.
        This is a heuristic; highway-env has utilities but we keep it student-simple.
        """
        v = self._vehicle()
        road = getattr(self.env.unwrapped, "road", None)
        if v is None or road is None:
            return False

        try:
            # find vehicles ahead in same lane (approx)
            my_x = float(v.position[0])
            my_lane = v.lane_index
            min_front = None

            for other in road.vehicles:
                if other is v:
                    continue
                if getattr(other, "lane_index", None) != my_lane:
                    continue
                dx = float(other.position[0]) - my_x
                if dx > 0:  # in front
                    if (min_front is None) or (dx < min_front):
                        min_front = dx

            if min_front is None:
                return False
            return bool(min_front < float(self.cfg.unsafe_distance_m))
        except Exception:
            return False
