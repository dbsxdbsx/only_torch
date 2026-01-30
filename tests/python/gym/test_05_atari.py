# -*- coding: utf-8 -*-
"""
Atari 环境测试
- ALE/Breakout-v5: 打砖块 (图像输入 + 离散动作)
- ALE/Pong-v5: 乒乓球 (图像输入 + 离散动作)

需要安装: pip install "gymnasium[atari]" "gymnasium[accept-rom-license]"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
import numpy as np

# 必须导入 ale_py 以注册 ALE 环境
import ale_py
gym.register_envs(ale_py)


def test_breakout():
    """测试 Breakout 图像输入 + 离散动作环境"""
    print("=" * 60)
    print("测试 ALE/Breakout-v5 (图像输入 + 离散动作)")
    print("=" * 60)
    
    env = gym.make("ALE/Breakout-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 类型: Box (图像)")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 数据类型: {env.observation_space.dtype}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 类型: Discrete")
    print(f"  - 动作数: {env.action_space.n}")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察形状: {obs.shape}")
    print(f"初始观察范围: [{obs.min()}, {obs.max()}]")
    print()
    
    # 执行几步
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={action}, reward={reward}, obs_shape={obs.shape}")
    
    print(f"\n累计奖励: {total_reward}")
    env.close()
    print("✅ ALE/Breakout-v5 测试通过!\n")


def test_pong():
    """测试 Pong 图像输入 + 离散动作环境"""
    print("=" * 60)
    print("测试 ALE/Pong-v5 (图像输入 + 离散动作)")
    print("=" * 60)
    
    env = gym.make("ALE/Pong-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 动作数: {env.action_space.n}")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察形状: {obs.shape}")
    print()
    
    # 执行几步
    print("执行 5 步随机动作:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: action={action}, reward={reward}")
    
    env.close()
    print("✅ ALE/Pong-v5 测试通过!\n")


def test_space_invaders():
    """测试 SpaceInvaders 图像输入环境"""
    print("=" * 60)
    print("测试 ALE/SpaceInvaders-v5 (图像输入 + 离散动作)")
    print("=" * 60)
    
    env = gym.make("ALE/SpaceInvaders-v5")
    
    print(f"观察空间形状: {env.observation_space.shape}")
    print(f"动作数: {env.action_space.n}")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察形状: {obs.shape}")
    
    # 执行几步
    print("执行 5 步随机动作:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: action={action}, reward={reward}")
    
    env.close()
    print("✅ ALE/SpaceInvaders-v5 测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("Gymnasium Atari 环境测试 (图像输入)")
    print("=" * 60 + "\n")
    
    test_breakout()
    test_pong()
    test_space_invaders()
    
    print("=" * 60)
    print("🎉 所有 Atari 环境测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
