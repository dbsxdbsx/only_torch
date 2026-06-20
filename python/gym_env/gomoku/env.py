"""
五子棋 Gymnasium 环境（薄包装层）

两种变体：
- GomokuSelfPlayEnv：双方外部落子，无内置对手（self-play 自对弈训练）
- GomokuEnv：内置 naive 对手（评测 / 人机）

职责：reset/step/obs + 持有 Board 实例。
不含 MCTS/UCB 搜索逻辑。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from gym_env.gomoku.board import Board, BLACK, WHITE
from gym_env.gomoku.opponents import make_opponent


class GomokuSelfPlayEnv(gym.Env):
    """自对弈五子棋环境（self-play 自对弈训练用）

    step(action) 只执行当前方落子并切换 to_play，不自动触发对手。
    MCTS 搜索时直接操作 env.board（Board 实例），不经 step()。

    观察空间：Box(0,1, (3, size, size), int8) — 当前方视角
    动作空间：Discrete(size²)
    奖励：当前方视角 +1/-1/0
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, board_size: int = 9, win_length: int = 5,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.board = Board(board_size, win_length)
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, board_size, board_size),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(board_size * board_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self.board.observation(), {}

    def step(self, action: int):
        player_before = self.board.to_play
        reward, done = self.board.step(action)
        obs = self.board.observation()
        return obs, reward, done, False, {
            "player": player_before,
            "winner": self.board.winner,
        }

    def render(self):
        return self.board.render()

    def close(self):
        pass


class GomokuEnv(gym.Env):
    """带内置对手的五子棋环境（评测 / 人机用）

    玩家始终为 player_color（默认 BLACK），step(action) 先落子再自动触发对手回应。
    观察空间与 SelfPlayEnv 一致。
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, board_size: int = 9, win_length: int = 5,
                 opponent: str = "naive2", player_color: str = "black",
                 render_mode: Optional[str] = None):
        super().__init__()
        self.board = Board(board_size, win_length)
        self.render_mode = render_mode
        color_map = {"black": BLACK, "white": WHITE}
        assert player_color in color_map
        self.player_color = color_map[player_color]
        self.opponent_fn = make_opponent(opponent)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, board_size, board_size),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(board_size * board_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        if self.player_color != self.board.to_play:
            opp_action = self.opponent_fn(self.board, self.np_random)
            self.board.step(opp_action)
        return self._get_obs(), {}

    def step(self, action: int):
        if self.board.done:
            return self._get_obs(), 0.0, True, False, {}

        if not self.board.is_valid(action):
            self.board.done = True
            self.board.winner = 1 - self.player_color
            return self._get_obs(), -1.0, True, False, {"reason": "illegal_move"}

        reward, done = self.board.step(action)
        if done:
            final_reward = self._player_reward()
            return self._get_obs(), final_reward, True, False, {"winner": self.board.winner}

        opp_action = self.opponent_fn(self.board, self.np_random)
        if opp_action is None:
            self.board.done = True
            self.board.winner = -1
            return self._get_obs(), 0.0, True, False, {"winner": -1}

        _, done = self.board.step(opp_action)
        if done:
            final_reward = self._player_reward()
            return self._get_obs(), final_reward, True, False, {"winner": self.board.winner}

        return self._get_obs(), 0.0, False, False, {}

    def render(self):
        return self.board.render()

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        if self.player_color == BLACK:
            return self.board.state.copy()
        obs = np.empty_like(self.board.state)
        obs[0] = self.board.state[1]
        obs[1] = self.board.state[0]
        obs[2] = self.board.state[2]
        return obs

    def _player_reward(self) -> float:
        if self.board.winner == self.player_color:
            return 1.0
        elif self.board.winner == 1 - self.player_color:
            return -1.0
        return 0.0
