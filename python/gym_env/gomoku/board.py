"""
五子棋棋盘（纯规则层）

职责边界：
- 合法着法、落子、终局判断、当前方
- clone / restore（供 MCTS 树搜索回滚）
- legal_mask（供 MCTS 候选动作）
- 不含 UCB / 树节点 / backup 等搜索逻辑

性能关键路径：
- check_winner 使用增量检查（只查最后落子四方向），~6μs/call
- legal_mask 使用 numpy 向量化，~0.8μs/call
- B2 基准：9×9 × 800 sims 环境开销 ≈ 20ms（含 pyo3）
"""
import numpy as np
from typing import Optional, Tuple, List

BLACK = 0
WHITE = 1


class Board:
    """五子棋棋盘

    状态表示：(3, board_size, board_size) int8 数组
      通道 0：黑子（1=有棋子）
      通道 1：白子（1=有棋子）
      通道 2：空位（1=空）

    落子约定：action = row * board_size + col
    """

    __slots__ = ("size", "win_length", "state", "to_play",
                 "last_action", "done", "winner", "_move_count")

    def __init__(self, board_size: int = 9, win_length: int = 5):
        assert board_size >= 5, f"棋盘大小至少为 5，当前: {board_size}"
        assert win_length <= board_size, f"获胜长度不能超过棋盘大小"
        self.size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        """重置为空盘，黑先"""
        self.state = np.zeros((3, self.size, self.size), dtype=np.int8)
        self.state[2, :, :] = 1
        self.to_play = BLACK
        self.last_action: Optional[int] = None
        self.done = False
        self.winner: Optional[int] = None  # None=未结束, BLACK/WHITE=胜方, -1=平局
        self._move_count = 0

    # ======================== 核心操作 ========================

    def step(self, action: int) -> Tuple[float, bool]:
        """落子（不触发对手，纯规则）

        返回 (reward_for_current_player, terminal)。
        reward: +1=当前方胜, 0=继续或平局, -1=非法着（同时 terminal=True）。
        MCTS 内部调用此方法，不经 env.step()。
        """
        assert not self.done, "游戏已结束"
        if not self.is_valid(action):
            self.done = True
            self.winner = 1 - self.to_play
            return -1.0, True

        row, col = divmod(action, self.size)
        self.state[self.to_play, row, col] = 1
        self.state[2, row, col] = 0
        self.last_action = action
        self._move_count += 1

        if self._check_winner_incremental(self.to_play, row, col):
            self.done = True
            self.winner = self.to_play
            reward = 1.0
        elif self._move_count >= self.size * self.size:
            self.done = True
            self.winner = -1  # 平局
            reward = 0.0
        else:
            reward = 0.0

        self.to_play = 1 - self.to_play
        return reward, self.done

    def is_valid(self, action: int) -> bool:
        if action < 0 or action >= self.size * self.size:
            return False
        row, col = divmod(action, self.size)
        return self.state[2, row, col] == 1

    def is_terminal(self) -> bool:
        return self.done

    def current_player(self) -> int:
        return self.to_play

    # ======================== MCTS 接口 ========================

    def legal_mask(self) -> np.ndarray:
        """合法着法掩码，shape=(board_size²,)，dtype=bool"""
        return self.state[2].flatten().astype(bool)

    def legal_actions(self) -> List[int]:
        """合法着法列表"""
        return np.where(self.state[2].flatten() == 1)[0].tolist()

    def clone(self) -> "Board":
        """深拷贝（供 MCTS snapshot）"""
        b = Board.__new__(Board)
        b.size = self.size
        b.win_length = self.win_length
        b.state = self.state.copy()
        b.to_play = self.to_play
        b.last_action = self.last_action
        b.done = self.done
        b.winner = self.winner
        b._move_count = self._move_count
        return b

    def get_snapshot(self) -> dict:
        """获取轻量快照（dict），比 clone 整个对象更快"""
        return {
            "state": self.state.copy(),
            "to_play": self.to_play,
            "last_action": self.last_action,
            "done": self.done,
            "winner": self.winner,
            "move_count": self._move_count,
        }

    def restore(self, snap: dict):
        """从快照恢复（必须 copy，否则后续 step 会污染原快照）"""
        self.state = snap["state"].copy()
        self.to_play = snap["to_play"]
        self.last_action = snap["last_action"]
        self.done = snap["done"]
        self.winner = snap["winner"]
        self._move_count = snap["move_count"]

    # ======================== 观察 ========================

    def observation(self) -> np.ndarray:
        """当前方视角的观察（3 通道 int8）

        通道 0：当前方棋子
        通道 1：对方棋子
        通道 2：空位
        """
        if self.to_play == BLACK:
            return self.state.copy()
        else:
            obs = np.empty_like(self.state)
            obs[0] = self.state[1]
            obs[1] = self.state[0]
            obs[2] = self.state[2]
            return obs

    def observation_flat(self) -> np.ndarray:
        """展平为 f32 向量，供神经网络输入"""
        return self.observation().flatten().astype(np.float32)

    # ======================== 增量获胜检查 ========================

    def _check_winner_incremental(self, color: int, row: int, col: int) -> bool:
        """只检查最后落子位置的四个方向（水平/垂直/主对角/副对角）"""
        b = self.state[color]
        wl = self.win_length
        d = self.size
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                for i in range(1, wl):
                    r, c = row + dr * i * sign, col + dc * i * sign
                    if 0 <= r < d and 0 <= c < d and b[r, c] == 1:
                        count += 1
                    else:
                        break
            if count >= wl:
                return True
        return False

    # ======================== 显示 ========================

    def render(self) -> str:
        d = self.size
        lines = [f"To play: {'black' if self.to_play == BLACK else 'white'} | "
                 f"moves: {self._move_count}"]
        col_labels = "     " + " ".join(f"{j+1:2d}" for j in range(d))
        lines.append(col_labels)
        lines.append("     " + "+" + "---" * d + "+")
        for i in range(d):
            row_str = f" {i+1:2d}  |"
            for j in range(d):
                if self.state[2, i, j] == 1:
                    row_str += " . "
                elif self.state[0, i, j] == 1:
                    row_str += " X "
                else:
                    row_str += " O "
            row_str += "|"
            lines.append(row_str)
        lines.append("     " + "+" + "---" * d + "+")
        out = "\n".join(lines)
        print(out)
        return out

    def __repr__(self):
        return f"Board({self.size}×{self.size}, moves={self._move_count}, to_play={'B' if self.to_play == BLACK else 'W'})"
