#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试自定义 Gymnasium 五子棋环境

测试内容：
1. 环境创建与基本功能
2. 15x15 棋盘各难度级别
3. 对局演示与渲染
4. 各难度级别对比

环境位置：tests/python/custom_envs/gomoku.py
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# 添加项目根目录到路径，确保可以导入 tests.python.custom_envs
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np


def test_gymnasium_import():
    """测试 Gymnasium 导入"""
    print("=" * 60)
    print("测试 1: Gymnasium 导入检查")
    print("=" * 60)

    try:
        import gymnasium as gym
        print(f"✓ Gymnasium 导入成功，版本: {gym.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Gymnasium 未安装: {e}")
        print("  请运行: pip install gymnasium")
        return False


def test_custom_env_import():
    """测试自定义环境导入"""
    print("\n" + "=" * 60)
    print("测试 2: 自定义环境导入")
    print("=" * 60)

    try:
        # 导入自定义环境模块（自动注册环境）
        import tests.python.custom_envs
        print("✓ 自定义环境模块导入成功")

        # 导入环境类
        from tests.python.custom_envs.gomoku import GomokuEnv
        print("✓ GomokuEnv 类导入成功")

        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_env_basic():
    """测试基本环境功能"""
    print("\n" + "=" * 60)
    print("测试 3: 基本环境功能 (15x15 棋盘)")
    print("=" * 60)

    from tests.python.custom_envs.gomoku import GomokuEnv

    # 创建环境
    env = GomokuEnv(board_size=15, win_length=5, opponent='random')
    print(f"✓ 环境创建成功")
    print(f"  棋盘大小: {env.board_size}x{env.board_size}")
    print(f"  获胜条件: {env.win_length} 连珠")
    print(f"  对手类型: {env.opponent_type}")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"  初始观察形状: {obs.shape}")
    print(f"  初始观察类型: {obs.dtype}")

    # 渲染
    print("\n初始棋盘:")
    env.render()

    env.close()
    return True


def test_env_registration():
    """测试环境注册"""
    print("\n" + "=" * 60)
    print("测试 4: 环境注册与 gymnasium.make()")
    print("=" * 60)

    import gymnasium as gym
    import tests.python.custom_envs  # 触发注册

    env_ids = [
        'Gomoku-random-v0',
        'Gomoku-naive0-v0',
        'Gomoku-naive1-v0',
        'Gomoku-naive2-v0',
        'Gomoku-naive3-v0',
    ]

    for env_id in env_ids:
        try:
            env = gym.make(env_id)
            print(f"  ✓ {env_id}: 观察空间 {env.observation_space.shape}")
            env.close()
        except Exception as e:
            print(f"  ✗ {env_id}: {e}")

    return True


def test_random_game(opponent='random', max_steps=100):
    """测试随机对局（15x15 棋盘）"""
    print("\n" + "=" * 60)
    print(f"测试 5: 随机对局演示 (15x15, {opponent})")
    print("=" * 60)

    from tests.python.custom_envs.gomoku import GomokuEnv

    env = GomokuEnv(board_size=15, win_length=5, opponent=opponent)
    obs, info = env.reset(seed=42)
    total_reward = 0

    for step in range(max_steps):
        # 随机选择合法动作
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print(f"  步骤 {step + 1}: 无合法动作")
            break

        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"\n最终棋盘状态 (步骤 {step + 1}):")
            env.render()

            winner = info.get('winner', 'unknown')
            if winner == 'player':
                print(f"✓ 玩家获胜！总奖励: {total_reward}")
            elif winner == 'opponent':
                print(f"✗ 对手获胜！总奖励: {total_reward}")
            elif winner == 'draw':
                print(f"= 平局！总奖励: {total_reward}")
            else:
                print(f"游戏结束，原因: {info.get('reason', 'unknown')}")
            break
    else:
        print(f"\n达到最大步数 {max_steps}，游戏未结束")
        env.render()

    env.close()
    return True


def test_difficulty_levels(num_games=10):
    """测试各难度级别对手"""
    print("\n" + "=" * 60)
    print(f"测试 6: 各难度级别对手对比 ({num_games} 局/级别, 15x15 棋盘)")
    print("=" * 60)

    from tests.python.custom_envs.gomoku import GomokuEnv

    levels = [
        ('random', 'Random (随机)'),
        ('naive0', 'Naive-0 (最弱)'),
        ('naive1', 'Naive-1'),
        ('naive2', 'Naive-2'),
        ('naive3', 'Naive-3 (最强)'),
    ]

    results = {}

    for opponent, level_name in levels:
        print(f"\n  测试 {level_name}...", end="", flush=True)
        wins, losses, draws = 0, 0, 0
        total_steps = 0

        for game in range(num_games):
            env = GomokuEnv(board_size=15, win_length=5, opponent=opponent)
            obs, _ = env.reset(seed=game)
            terminated = False
            steps = 0
            reward = 0

            while not terminated and steps < 225:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

            total_steps += steps

            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1

            env.close()

        avg_steps = total_steps / num_games
        win_rate = wins / num_games * 100
        results[level_name] = (wins, losses, draws, avg_steps)
        print(f" 胜/负/平: {wins}/{losses}/{draws}, 胜率: {win_rate:.0f}%, 平均步数: {avg_steps:.1f}")

    return results


def test_15x15_demo():
    """15x15 棋盘演示对局"""
    print("\n" + "=" * 60)
    print("测试 7: 15x15 棋盘演示对局 (naive2 对手)")
    print("=" * 60)

    from tests.python.custom_envs.gomoku import GomokuEnv

    env = GomokuEnv(board_size=15, win_length=5, opponent='naive2')
    obs, _ = env.reset(seed=123)

    print("\n初始棋盘:")
    env.render()

    # 完整对局
    terminated = False
    steps = 0
    reward = 0

    while not terminated and steps < 225:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    print(f"\n游戏结束 (共 {steps} 步):")
    env.render()

    winner = info.get('winner', 'unknown')
    if winner == 'player':
        print("结果: 玩家获胜！")
    elif winner == 'opponent':
        print("结果: 对手获胜！")
    else:
        print("结果: 平局！")

    env.close()
    return True


def test_env_info():
    """显示环境详细信息"""
    print("\n" + "=" * 60)
    print("测试 8: 环境详细信息")
    print("=" * 60)

    from tests.python.custom_envs.gomoku import GomokuEnv

    env = GomokuEnv(board_size=15, win_length=5, opponent='naive2')

    print(f"  棋盘大小: {env.board_size}x{env.board_size}")
    print(f"  获胜条件: {env.win_length} 连珠")
    print(f"  观察空间形状: {env.observation_space.shape}")
    print(f"  动作空间大小: {env.action_space.n}")

    print("\n  观察空间编码 (3 通道):")
    print("    通道 0: 黑子位置 (1=有棋子)")
    print("    通道 1: 白子位置 (1=有棋子)")
    print("    通道 2: 空位标记 (1=空)")

    print("\n  难度级别:")
    print("    random : 纯随机落子")
    print("    naive0 : 只防守，不主动扩展")
    print("    naive1 : 搜索 3 连珠扩展")
    print("    naive2 : 搜索 2 连珠扩展")
    print("    naive3 : 搜索 1 连珠扩展（最积极）")

    print("\n  奖励设计:")
    print("    +1 = 玩家获胜")
    print("    -1 = 对手获胜 / 非法落子")
    print("     0 = 游戏进行中或平局")

    print("\n  API 返回值:")
    print("    obs, info = env.reset(seed=...)")
    print("    obs, reward, terminated, truncated, info = env.step(action)")

    env.close()
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("自定义 Gymnasium 五子棋环境测试 (15x15)")
    print("=" * 60)

    # 测试 1: Gymnasium 导入
    if not test_gymnasium_import():
        return False

    # 测试 2: 自定义环境导入
    if not test_custom_env_import():
        return False

    # 测试 3: 基本功能
    test_env_basic()

    # 测试 4: 环境注册
    test_env_registration()

    # 测试 5: 随机对局
    test_random_game(opponent='random', max_steps=100)

    # 测试 6: 各难度级别对比
    test_difficulty_levels(num_games=10)

    # 测试 7: 15x15 演示
    test_15x15_demo()

    # 测试 8: 环境详细信息
    test_env_info()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

    return True


if __name__ == '__main__':
    main()
