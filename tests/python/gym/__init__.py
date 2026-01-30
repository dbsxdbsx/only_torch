# -*- coding: utf-8 -*-
"""
Gymnasium 环境测试模块

测试脚本说明:
- test_01_basic_discrete.py: 基础离散动作环境 (CartPole, Acrobot)
- test_02_basic_continuous.py: 基础连续动作环境 (Pendulum, MountainCarContinuous)
- test_03_box2d.py: Box2D 环境 (LunarLander, BipedalWalker) [需要 gymnasium[box2d]]
- test_04_mujoco.py: MuJoCo 环境 (Ant, HalfCheetah) [需要 gymnasium[mujoco]]
- test_05_atari.py: Atari 环境 (Breakout, Pong) [需要 gymnasium[atari]]
- test_06_minari.py: 离线 RL 数据集 [需要 minari[hf,hdf5]]

运行方式:
    python -m tests.python.gym.test_01_basic_discrete
    # 或直接运行
    python tests/python/gym/test_01_basic_discrete.py
    # 或运行所有测试
    python tests/python/gym/run_all_tests.py
"""
