"""MyZero 通用 schema 的轻量 Gymnasium 纵切环境。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class _FiniteToyEnv(gym.Env):
    """固定短 horizon 的公共实现。"""

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 8) -> None:
        super().__init__()
        self.horizon = horizon
        self.step_count = 0

    def _finish(self, reward: float):
        self.step_count += 1
        terminated = self.step_count >= self.horizon
        return reward, terminated


class MultiDiscreteToyEnv(_FiniteToyEnv):
    """固定结构 MultiDiscrete([4,4,16])。"""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([4, 4, 16])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        reward, terminated = self._finish(float(action.tolist() == [1, 2, 3]))
        obs = np.full(6, self.step_count / self.horizon, dtype=np.float32)
        return obs, reward, terminated, False, {}


class Continuous2DToyEnv(_FiniteToyEnv):
    """两维连续 Box，用 categorical bins 验证联合动作 codec。"""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -2.0], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        reward, terminated = self._finish(-float(np.square(action).sum()))
        obs = np.array([*action, self.step_count / self.horizon, 1.0], dtype=np.float32)
        return obs, reward, terminated, False, {}


class ImageDenseToyEnv(_FiniteToyEnv):
    """矩形 HWC 图像 + 稠密辅助状态的 Dict observation。"""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    0.0, 1.0, shape=(16, 24, 3), dtype=np.float32
                ),
                "aux": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(3)

    def _obs(self):
        value = self.step_count / self.horizon
        image = np.fromfunction(
            lambda y, x, c: (c * 200.0 + y * 10.0 + x) / 600.0,
            (16, 24, 3),
            dtype=np.float32,
        ).astype(np.float32)
        image = np.clip(image + value * 0.01, 0.0, 1.0)
        return {
            "image": image,
            "aux": np.array([value, 0.0, 1.0, -1.0, 0.5], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        reward, terminated = self._finish(float(int(action) == 1))
        return self._obs(), reward, terminated, False, {}


class TokenToyEnv(_FiniteToyEnv):
    """环境直接提供固定长度 token IDs；tokenizer 不属于 RL 内核。"""

    LENGTH = 6
    VOCAB_SIZE = 32

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            0.0,
            float(self.VOCAB_SIZE - 1),
            shape=(self.LENGTH,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def _obs(self):
        start = self.step_count % (self.VOCAB_SIZE - self.LENGTH)
        return np.arange(start, start + self.LENGTH, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        reward, terminated = self._finish(float(int(action) == self.step_count % 2))
        return self._obs(), reward, terminated, False, {}
