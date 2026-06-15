"""
五子棋 naive 对手策略（评测用）

策略列表：
  random  — 纯随机
  naive0  — 只防守
  naive1  — 搜索 3 连珠扩展
  naive2  — 搜索 2 连珠扩展
  naive3  — 搜索 1 连珠扩展（最积极）

所有策略签名：(board: Board, rng: np.random.Generator) -> Optional[int]
"""
import numpy as np
from typing import Optional, Callable

from gym_env.gomoku.board import Board


def make_opponent(name: str) -> Callable:
    """根据名称创建对手策略函数"""
    if name == "random":
        return _random_policy
    if name.startswith("naive"):
        level = int(name[-1]) if name[-1].isdigit() else 0
        return lambda board, rng: _naive_policy(board, rng, level)
    raise ValueError(f"未知对手类型: {name}")


def _random_policy(board: Board, rng: np.random.Generator) -> Optional[int]:
    actions = board.legal_actions()
    if not actions:
        return None
    return int(rng.choice(actions))


def _naive_policy(board: Board, rng: np.random.Generator,
                  level: int) -> Optional[int]:
    d = board.size
    wl = board.win_length
    me = board.to_play
    opp = 1 - me

    actions = board.legal_actions()
    if not actions:
        return None

    # 第一手占中心
    if board._move_count <= 1:
        center = d // 2 * d + d // 2
        if board.is_valid(center):
            return center

    # 检查我方是否有一步获胜
    for a in actions:
        r, c = divmod(a, d)
        if _would_win(board.state, me, r, c, d, wl):
            return a

    # 检查对方是否有一步获胜，阻挡
    for a in actions:
        r, c = divmod(a, d)
        if _would_win(board.state, opp, r, c, d, wl):
            return a

    # 按 level 搜索扩展
    for target_len in range(wl - 2, max(0, wl - 2 - level), -1):
        for a in actions:
            r, c = divmod(a, d)
            if _extends_line(board.state, me, r, c, d, target_len):
                return a

    # 在最后一步附近落子
    if board.last_action is not None:
        lr, lc = divmod(board.last_action, d)
        nearby = [a for a in actions
                  if abs(a // d - lr) <= 2 and abs(a % d - lc) <= 2]
        if nearby:
            return int(rng.choice(nearby))

    return int(rng.choice(actions))


def _would_win(state: np.ndarray, color: int, row: int, col: int,
               d: int, wl: int) -> bool:
    """假设在 (row, col) 落子后是否形成 wl 连珠"""
    b = state[color]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
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


def _extends_line(state: np.ndarray, color: int, row: int, col: int,
                  d: int, target_len: int) -> bool:
    """假设在 (row, col) 落子后是否延长至少 target_len 连珠"""
    b = state[color]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            for i in range(1, target_len + 1):
                r, c = row + dr * i * sign, col + dc * i * sign
                if 0 <= r < d and 0 <= c < d and b[r, c] == 1:
                    count += 1
                else:
                    break
        if count >= target_len:
            return True
    return False
