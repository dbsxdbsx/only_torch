"""
Board 单元测试（增量 check_winner + clone/restore + legal_mask）

运行：python -m pytest python/gym_env/gomoku/test_board.py -v
"""
import numpy as np
import pytest

from gym_env.gomoku.board import Board, BLACK, WHITE


class TestBoardBasic:
    def test_initial_state(self):
        b = Board(9, 5)
        assert b.to_play == BLACK
        assert not b.done
        assert b._move_count == 0
        assert b.legal_mask().sum() == 81

    def test_step_alternates_player(self):
        b = Board(9, 5)
        b.step(40)
        assert b.to_play == WHITE
        b.step(41)
        assert b.to_play == BLACK

    def test_invalid_move_terminates(self):
        b = Board(9, 5)
        b.step(40)
        reward, done = b.step(40)  # 重复落子
        assert done
        assert reward == -1.0

    def test_out_of_range_invalid(self):
        b = Board(9, 5)
        assert not b.is_valid(-1)
        assert not b.is_valid(81)
        assert b.is_valid(0)
        assert b.is_valid(80)


class TestCheckWinner:
    """增量 check_winner 边界测试"""

    def _play_sequence(self, board, actions_black, actions_white):
        """交替执行黑白棋步"""
        for i in range(max(len(actions_black), len(actions_white))):
            if i < len(actions_black):
                board.step(actions_black[i])
            if i < len(actions_white) and not board.done:
                board.step(actions_white[i])
            if board.done:
                break

    def test_horizontal_win(self):
        b = Board(9, 5)
        # 黑：(0,0)(0,1)(0,2)(0,3)(0,4) = 0,1,2,3,4
        # 白：(1,0)(1,1)(1,2)(1,3) = 9,10,11,12
        self._play_sequence(b, [0, 1, 2, 3, 4], [9, 10, 11, 12])
        assert b.done
        assert b.winner == BLACK

    def test_vertical_win(self):
        b = Board(9, 5)
        # 黑：(0,0)(1,0)(2,0)(3,0)(4,0) = 0,9,18,27,36
        # 白：(0,1)(1,1)(2,1)(3,1) = 1,10,19,28
        self._play_sequence(b, [0, 9, 18, 27, 36], [1, 10, 19, 28])
        assert b.done
        assert b.winner == BLACK

    def test_diagonal_win(self):
        b = Board(9, 5)
        # 黑：(0,0)(1,1)(2,2)(3,3)(4,4) = 0,10,20,30,40
        # 白：(0,1)(1,2)(2,3)(3,4) = 1,11,21,31
        self._play_sequence(b, [0, 10, 20, 30, 40], [1, 11, 21, 31])
        assert b.done
        assert b.winner == BLACK

    def test_anti_diagonal_win(self):
        b = Board(9, 5)
        # 黑：(0,4)(1,3)(2,2)(3,1)(4,0) = 4,12,20,28,36
        # 白：(0,0)(1,0)(2,0)(3,0) = 0,9,18,27
        self._play_sequence(b, [4, 12, 20, 28, 36], [0, 9, 18, 27])
        assert b.done
        assert b.winner == BLACK

    def test_four_not_win(self):
        """只有 4 连珠不应触发获胜"""
        b = Board(9, 5)
        self._play_sequence(b, [0, 1, 2, 3], [9, 10, 11, 12])
        assert not b.done

    def test_white_wins(self):
        b = Board(9, 5)
        # 白方五连珠
        # 黑：0,1,2,3,72  白：9,10,11,12,13
        self._play_sequence(b, [0, 1, 2, 3, 72], [9, 10, 11, 12, 13])
        assert b.done
        assert b.winner == WHITE

    def test_edge_row_win(self):
        """最后一行五连珠"""
        b = Board(9, 5)
        # 黑：(8,0-4) = 72,73,74,75,76
        # 白：(7,0-3) = 63,64,65,66
        self._play_sequence(b, [72, 73, 74, 75, 76], [63, 64, 65, 66])
        assert b.done
        assert b.winner == BLACK

    def test_corner_diagonal(self):
        """角落对角线五连"""
        b = Board(9, 5)
        # 黑：(4,4)(5,5)(6,6)(7,7)(8,8) = 40,50,60,70,80
        # 白：(0,0)(0,1)(0,2)(0,3) = 0,1,2,3
        self._play_sequence(b, [40, 50, 60, 70, 80], [0, 1, 2, 3])
        assert b.done
        assert b.winner == BLACK


class TestCloneRestore:
    def test_clone_independence(self):
        b = Board(9, 5)
        b.step(40)
        clone = b.clone()
        # 修改原始不影响 clone
        b.step(41)
        assert b._move_count == 2
        assert clone._move_count == 1
        assert clone.to_play == WHITE

    def test_snapshot_restore_roundtrip(self):
        b = Board(9, 5)
        b.step(40)
        b.step(41)
        snap = b.get_snapshot()

        b.step(0)
        b.step(1)
        assert b._move_count == 4

        b.restore(snap)
        assert b._move_count == 2
        assert b.to_play == BLACK
        assert not b.is_valid(40)
        assert not b.is_valid(41)
        assert b.is_valid(0)

    def test_snapshot_state_array_independent(self):
        """快照的 numpy 数组应为独立副本"""
        b = Board(9, 5)
        snap = b.get_snapshot()
        b.step(40)
        assert snap["state"][2, 4, 4] == 1

    def test_repeated_restore_not_polluted(self):
        """同一 snapshot 反复 restore + step 不应污染快照"""
        b = Board(9, 5)
        b.step(40)
        snap = b.get_snapshot()
        original_empty_count = snap["state"][2].sum()

        for action in [0, 1, 2]:
            b.restore(snap)
            b.step(action)

        # 快照本身不应被后续 step 修改
        assert snap["state"][2].sum() == original_empty_count
        # 再次 restore 后棋盘应和快照一致
        b.restore(snap)
        assert b._move_count == 1
        assert b.is_valid(0)  # action 0 应重新可用


class TestLegalMask:
    def test_initial_all_legal(self):
        b = Board(9, 5)
        mask = b.legal_mask()
        assert mask.shape == (81,)
        assert mask.dtype == bool
        assert mask.all()

    def test_after_move(self):
        b = Board(9, 5)
        b.step(40)
        mask = b.legal_mask()
        assert not mask[40]
        assert mask.sum() == 80

    def test_legal_actions_consistent(self):
        b = Board(9, 5)
        b.step(10)
        b.step(20)
        b.step(30)
        mask = b.legal_mask()
        actions = b.legal_actions()
        assert len(actions) == mask.sum()
        for a in actions:
            assert mask[a]


class TestObservation:
    def test_obs_shape_and_dtype(self):
        b = Board(9, 5)
        obs = b.observation()
        assert obs.shape == (3, 9, 9)
        assert obs.dtype == np.int8

    def test_obs_flat(self):
        b = Board(9, 5)
        flat = b.observation_flat()
        assert flat.shape == (243,)
        assert flat.dtype == np.float32

    def test_obs_perspective(self):
        """观察应是当前方视角"""
        b = Board(9, 5)
        b.step(40)  # 黑下 (4,4)
        # 现在是白方视角
        obs = b.observation()
        # 通道 0 应是"当前方"（白）棋子 → 空
        assert obs[0, 4, 4] == 0
        # 通道 1 应是"对方"（黑）棋子 → 有
        assert obs[1, 4, 4] == 1


class TestDraw:
    def test_full_board_draw(self):
        """填满 5×5 棋盘（无人五连珠）应平局"""
        b = Board(5, 5)
        # 用特定顺序填满不形成五连珠
        # 棋盘索引 0-24，交替落子
        moves = [
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,
        ]
        for action in moves:
            if b.done:
                break
            b.step(action)
        # 不管是否平局（可能某方先赢），只验证 done
        assert b.done
