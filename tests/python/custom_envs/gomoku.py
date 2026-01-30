#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五子棋 Gymnasium 环境

基于 Gymnasium API 实现的五子棋环境，支持多种难度级别的 AI 对手。
参考：tongzou/gym-gomoku 的策略逻辑

特性：
- 支持任意棋盘大小（默认 15x15）
- 支持任意获胜长度（默认 5 连珠）
- 5 个难度级别：random, naive0, naive1, naive2, naive3
- 3 通道观察空间：[我方棋子, 对方棋子, 空位]
- 兼容 Gymnasium 现代 API
"""
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List, Any


class GomokuEnv(gym.Env):
    """
    五子棋环境
    
    观察空间：Box(0, 1, (3, board_size, board_size), int8)
        - 通道 0: 黑子位置
        - 通道 1: 白子位置
        - 通道 2: 空位标记
    
    动作空间：Discrete(board_size^2)
        - 动作 i 对应位置 (i // board_size, i % board_size)
    
    奖励：
        - +1: 玩家获胜
        - -1: 对手获胜 或 非法落子
        - 0: 游戏进行中 或 平局
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 1,
    }
    
    # 玩家常量
    BLACK = 0
    WHITE = 1
    
    def __init__(
        self,
        board_size: int = 15,
        win_length: int = 5,
        opponent: str = 'naive2',
        player_color: str = 'black',
        render_mode: Optional[str] = None,
    ):
        """
        初始化五子棋环境
        
        Args:
            board_size: 棋盘大小（默认 15）
            win_length: 获胜所需连珠数（默认 5）
            opponent: 对手策略 ('random', 'naive0', 'naive1', 'naive2', 'naive3')
            player_color: 玩家颜色 ('black' 或 'white')
            render_mode: 渲染模式 ('human', 'ansi', None)
        """
        super().__init__()
        
        assert board_size >= 5, f"棋盘大小至少为 5，当前: {board_size}"
        assert win_length >= 3, f"获胜长度至少为 3，当前: {win_length}"
        assert win_length <= board_size, f"获胜长度不能超过棋盘大小"
        
        self.board_size = board_size
        self.win_length = win_length
        self.render_mode = render_mode
        
        # 玩家颜色
        color_map = {'black': self.BLACK, 'white': self.WHITE}
        assert player_color in color_map, f"player_color 必须是 'black' 或 'white'"
        self.player_color = color_map[player_color]
        
        # 对手策略
        self.opponent_type = opponent
        self.opponent_policy = self._make_opponent_policy(opponent)
        
        # 动作空间：棋盘上的每个位置
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 观察空间：3 通道（黑子、白子、空位）
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, board_size, board_size),
            dtype=np.int8
        )
        
        # 游戏状态
        self.state = None
        self.to_play = None
        self.done = False
        self.last_action = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 初始化棋盘状态
        self.state = np.zeros((3, self.board_size, self.board_size), dtype=np.int8)
        self.state[2, :, :] = 1  # 所有位置初始为空
        self.to_play = self.BLACK  # 黑棋先行
        self.done = False
        self.last_action = None
        
        # 如果玩家是白棋，让对手（黑棋）先下
        if self.player_color != self.to_play:
            opponent_action = self.opponent_policy(self.state, None, None)
            self._make_move(opponent_action, self.BLACK)
            self.to_play = self.WHITE
            self.last_action = opponent_action
        
        return self.state.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步动作
        
        Returns:
            observation: 新的观察
            reward: 奖励
            terminated: 是否终止（游戏结束）
            truncated: 是否截断（超时等）
            info: 额外信息
        """
        assert self.to_play == self.player_color, "不是玩家的回合"
        
        # 如果游戏已结束
        if self.done:
            return self.state.copy(), 0.0, True, False, {}
        
        # 检查动作合法性
        if not self._is_valid_move(action):
            self.done = True
            return self.state.copy(), -1.0, True, False, {'reason': 'illegal_move'}
        
        # 玩家落子
        prev_state = self.state.copy()
        self._make_move(action, self.player_color)
        self.last_action = action
        
        # 检查玩家是否获胜
        if self._check_winner(self.player_color):
            self.done = True
            return self.state.copy(), 1.0, True, False, {'winner': 'player'}
        
        # 检查是否平局
        if self._is_board_full():
            self.done = True
            return self.state.copy(), 0.0, True, False, {'winner': 'draw'}
        
        # 对手落子
        opponent_color = 1 - self.player_color
        opponent_action = self.opponent_policy(self.state, prev_state, action)
        
        if opponent_action is None:
            # 对手无法落子（平局）
            self.done = True
            return self.state.copy(), 0.0, True, False, {'winner': 'draw'}
        
        self._make_move(opponent_action, opponent_color)
        self.last_action = opponent_action
        
        # 检查对手是否获胜
        if self._check_winner(opponent_color):
            self.done = True
            return self.state.copy(), -1.0, True, False, {'winner': 'opponent'}
        
        # 检查是否平局
        if self._is_board_full():
            self.done = True
            return self.state.copy(), 0.0, True, False, {'winner': 'draw'}
        
        # 游戏继续
        return self.state.copy(), 0.0, False, False, {}
    
    def render(self):
        """渲染棋盘"""
        # 无论 render_mode 如何，都输出棋盘（方便调试）
        return self._render_board()
    
    def _render_board(self) -> str:
        """生成棋盘的字符串表示"""
        d = self.board_size
        lines = []
        
        # 标题行
        lines.append(f"To play: {'black' if self.to_play == self.BLACK else 'white'}")
        
        # 列标签
        if d > 9:
            tens_line = ' ' * 7
            for j in range(d):
                if j + 1 >= 10:
                    tens_line += str((j + 1) // 10) + ' '
                else:
                    tens_line += '  '
            lines.append(tens_line)
        
        col_labels = ' ' * 6
        for j in range(d):
            col_labels += str((j + 1) % 10) + ' '
        lines.append(col_labels)
        
        # 上边界
        lines.append(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+')
        
        # 棋盘内容
        for i in range(d):
            row_str = f"{i + 1:3d} | "
            for j in range(d):
                action = i * d + j
                if self.state[2, i, j] == 1:  # 空位
                    row_str += '. '
                elif self.state[0, i, j] == 1:  # 黑子
                    if self.last_action == action:
                        row_str += 'X)'
                    else:
                        row_str += 'X '
                else:  # 白子
                    if self.last_action == action:
                        row_str += 'O)'
                    else:
                        row_str += 'O '
            row_str += '|'
            lines.append(row_str)
        
        # 下边界
        lines.append(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+')
        
        output = '\n'.join(lines)
        print(output)
        return output
    
    def _make_move(self, action: int, color: int):
        """落子"""
        row, col = divmod(action, self.board_size)
        self.state[2, row, col] = 0  # 不再是空位
        self.state[color, row, col] = 1  # 放置棋子
    
    def _is_valid_move(self, action: int) -> bool:
        """检查落子是否合法"""
        if action < 0 or action >= self.board_size ** 2:
            return False
        row, col = divmod(action, self.board_size)
        return self.state[2, row, col] == 1  # 必须是空位
    
    def _is_board_full(self) -> bool:
        """检查棋盘是否已满"""
        return np.sum(self.state[2]) == 0
    
    def _check_winner(self, color: int) -> bool:
        """检查指定颜色是否获胜"""
        board = self.state[color]
        pattern = '1' * self.win_length
        return self._search_pattern(board, pattern) is not None
    
    def _search_pattern(self, board: np.ndarray, pattern: str) -> Optional[Tuple]:
        """在棋盘上搜索模式"""
        d = self.board_size
        
        # 水平搜索
        for i in range(d):
            row_str = ''.join(map(str, board[i]))
            if re.search(pattern, row_str):
                return ('horizontal', i)
        
        # 垂直搜索
        for j in range(d):
            col_str = ''.join(map(str, board[:, j]))
            if re.search(pattern, col_str):
                return ('vertical', j)
        
        # 对角线搜索
        for k in range(-d + self.win_length, d - self.win_length + 1):
            # 主对角线方向
            diag = np.diag(board, k)
            if len(diag) >= self.win_length:
                diag_str = ''.join(map(str, diag))
                if re.search(pattern, diag_str):
                    return ('diagonal', k)
            
            # 副对角线方向
            anti_diag = np.diag(np.fliplr(board), k)
            if len(anti_diag) >= self.win_length:
                anti_diag_str = ''.join(map(str, anti_diag))
                if re.search(pattern, anti_diag_str):
                    return ('anti_diagonal', k)
        
        return None
    
    def get_valid_actions(self) -> List[int]:
        """获取所有合法动作"""
        valid = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state[2, i, j] == 1:
                    valid.append(i * self.board_size + j)
        return valid
    
    @staticmethod
    def action_to_coord(board_size: int, action: int) -> Tuple[int, int]:
        """动作转坐标"""
        return divmod(action, board_size)
    
    @staticmethod
    def coord_to_action(board_size: int, row: int, col: int) -> int:
        """坐标转动作"""
        return row * board_size + col
    
    # ==================== 对手策略 ====================
    
    def _make_opponent_policy(self, opponent_type: str):
        """创建对手策略"""
        if opponent_type == 'random':
            return self._random_policy
        elif opponent_type.startswith('naive'):
            level = int(opponent_type[-1]) if opponent_type[-1].isdigit() else 0
            return lambda state, prev_state, prev_action: self._naive_policy(
                state, prev_state, prev_action, level
            )
        else:
            raise ValueError(f"未知的对手类型: {opponent_type}")
    
    def _random_policy(self, state: np.ndarray, prev_state, prev_action) -> Optional[int]:
        """随机策略"""
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[2, i, j] == 1:
                    valid_actions.append(i * self.board_size + j)
        
        if not valid_actions:
            return None
        
        return self.np_random.choice(valid_actions)
    
    def _naive_policy(
        self, 
        state: np.ndarray, 
        prev_state: Optional[np.ndarray], 
        prev_action: Optional[int],
        level: int
    ) -> Optional[int]:
        """
        启发式策略
        
        level:
            0 - 只防守，不主动扩展
            1 - 搜索 3 连珠扩展
            2 - 搜索 2 连珠扩展
            3 - 搜索 1 连珠扩展（最积极）
        """
        d = self.board_size
        win_len = self.win_length
        
        # 获取可用位置
        valid_coords = [(i, j) for i in range(d) for j in range(d) if state[2, i, j] == 1]
        if not valid_coords:
            return None
        
        # 确定对手颜色（AI 是玩家的对手）
        opponent_color = 1 - self.player_color
        player_color = self.player_color  # 玩家（AI 的对手）
        
        # 第一步：占据中心
        if prev_state is None:
            center = d // 2
            return center * d + center
        
        # 第二步：如果是后手，在对手棋子对角线位置落子
        empty_count = np.sum(state[2])
        if empty_count == d * d - 1:
            prev_row, prev_col = divmod(prev_action, d)
            dx = 1 if prev_row <= d // 2 else -1
            dy = 1 if prev_col <= d // 2 else -1
            new_row, new_col = prev_row + dx, prev_col + dy
            if 0 <= new_row < d and 0 <= new_col < d and state[2, new_row, new_col] == 1:
                return new_row * d + new_col
        
        # 构建 AI 视角的棋盘（1=对手, 2=空, 3=自己）
        my_board = np.zeros((d, d), dtype=int)
        my_board[state[2] == 1] = 2  # 空位
        my_board[state[opponent_color] == 1] = 3  # AI 自己
        my_board[state[player_color] == 1] = 1  # 玩家（AI 的对手）
        
        # 1. 检查 AI 是否有获胜机会
        move = self._search_winning_move(my_board, '3', win_len)
        if move is not None:
            return move[0] * d + move[1]
        
        # 2. 检查玩家是否有获胜威胁，阻挡
        move = self._search_winning_move(my_board, '1', win_len)
        if move is not None:
            return move[0] * d + move[1]
        
        # 3. 检查 AI 是否有活 win_len-2（例如活三）
        if win_len >= 4:
            pattern = '2' + '3' * (win_len - 2) + '2'
            move = self._search_pattern_move(my_board, pattern, len(pattern))
            if move is not None:
                return move[0] * d + move[1]
        
        # 4. 检查玩家是否有活 win_len-2，阻挡
        if win_len >= 4:
            pattern = '2' + '1' * (win_len - 2) + '2'
            move = self._search_pattern_move(my_board, pattern, len(pattern))
            if move is not None:
                return move[0] * d + move[1]
        
        # 5. 根据 level 搜索扩展机会
        for i in range(2, level + 2):
            if win_len - i < 1:
                break
            
            # 搜索 AI 的连珠扩展
            pattern = '23{' + str(win_len - i) + '}'
            move = self._search_pattern_move(my_board, pattern, win_len - i + 1)
            if move is not None:
                return move[0] * d + move[1]
            
            pattern = '3{' + str(win_len - i) + '}2'
            move = self._search_pattern_move(my_board, pattern, win_len - i + 1, begin=False)
            if move is not None:
                return move[0] * d + move[1]
        
        # 6. 如果都没找到，在上一步附近随机落子
        if prev_action is not None:
            prev_row, prev_col = divmod(prev_action, d)
            nearby = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = prev_row + dr, prev_col + dc
                    if 0 <= nr < d and 0 <= nc < d and state[2, nr, nc] == 1:
                        nearby.append((nr, nc))
            if nearby:
                r, c = nearby[self.np_random.integers(len(nearby))]
                return r * d + c
        
        # 7. 最后随机选择
        r, c = valid_coords[self.np_random.integers(len(valid_coords))]
        return r * d + c
    
    def _search_winning_move(
        self, 
        board: np.ndarray, 
        color_char: str, 
        win_len: int
    ) -> Optional[Tuple[int, int]]:
        """搜索获胜落子位置"""
        # 检查是否有 win_len-1 连珠且有空位可以补齐
        for i in range(win_len):
            pattern = color_char * i + '2' + color_char * (win_len - i - 1)
            move = self._search_pattern_move(board, pattern, win_len, begin=True, offset=i)
            if move is not None:
                return move
        return None
    
    def _search_pattern_move(
        self, 
        board: np.ndarray, 
        pattern: str, 
        size: int,
        begin: bool = True,
        offset: int = 0
    ) -> Optional[Tuple[int, int]]:
        """搜索模式并返回落子位置"""
        d = self.board_size
        
        # 水平搜索
        for i in range(d):
            row_str = ''.join(map(str, board[i]))
            match = re.search(pattern, row_str)
            if match:
                j = match.start() + (0 if begin else size - 1) + offset
                if 0 <= j < d and board[i, j] == 2:
                    return (i, j)
        
        # 垂直搜索
        for j in range(d):
            col_str = ''.join(map(str, board[:, j]))
            match = re.search(pattern, col_str)
            if match:
                i = match.start() + (0 if begin else size - 1) + offset
                if 0 <= i < d and board[i, j] == 2:
                    return (i, j)
        
        # 主对角线搜索
        for k in range(-d + size, d - size + 1):
            diag = np.diag(board, k)
            if len(diag) >= size:
                diag_str = ''.join(map(str, diag))
                match = re.search(pattern, diag_str)
                if match:
                    idx = match.start() + (0 if begin else size - 1) + offset
                    if k >= 0:
                        i, j = idx, idx + k
                    else:
                        i, j = idx - k, idx
                    if 0 <= i < d and 0 <= j < d and board[i, j] == 2:
                        return (i, j)
        
        # 副对角线搜索
        flipped = np.fliplr(board)
        for k in range(-d + size, d - size + 1):
            diag = np.diag(flipped, k)
            if len(diag) >= size:
                diag_str = ''.join(map(str, diag))
                match = re.search(pattern, diag_str)
                if match:
                    idx = match.start() + (0 if begin else size - 1) + offset
                    if k >= 0:
                        i, j = idx, d - 1 - (idx + k)
                    else:
                        i, j = idx - k, d - 1 - idx
                    if 0 <= i < d and 0 <= j < d and board[i, j] == 2:
                        return (i, j)
        
        return None
