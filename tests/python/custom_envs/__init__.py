#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义 Gymnasium 环境

导入此模块时自动注册所有自定义环境。
"""
from gymnasium.envs.registration import register

# 注册五子棋环境（5 个难度级别，15x15 棋盘）
_gomoku_levels = ['random', 'naive0', 'naive1', 'naive2', 'naive3']

for level in _gomoku_levels:
    register(
        id=f'Gomoku-{level}-v0',
        entry_point='tests.python.custom_envs.gomoku:GomokuEnv',
        kwargs={
            'board_size': 15,
            'win_length': 5,
            'opponent': level,
        },
        max_episode_steps=225,  # 15x15 棋盘最多 225 步
    )
