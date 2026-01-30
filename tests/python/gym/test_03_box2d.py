# -*- coding: utf-8 -*-
"""
Box2D 环境测试
- LunarLander-v3: 月球着陆器 (离散动作)
- LunarLanderContinuous-v3: 月球着陆器 (多维连续动作)
- BipedalWalker-v3: 双足行走器 (多维连续动作)

需要安装: pip install "gymnasium[box2d]"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
import numpy as np


def test_lunar_lander_discrete():
    """测试 LunarLander-v3 离散动作环境"""
    print("=" * 60)
    print("测试 LunarLander-v3 (离散动作)")
    print("=" * 60)
    
    env = gym.make("LunarLander-v3")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 含义: [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 动作数: {env.action_space.n}")
    print(f"  - 含义: 0=无操作, 1=左引擎, 2=主引擎, 3=右引擎")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 (8维): {obs[:4]}... (前4维)")
    print()
    
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={action}, reward={reward:.4f}")
    
    print(f"\n累计奖励: {total_reward:.4f}")
    env.close()
    print("✅ LunarLander-v3 测试通过!\n")


def test_lunar_lander_continuous():
    """测试 LunarLanderContinuous-v3 多维连续动作环境"""
    print("=" * 60)
    print("测试 LunarLanderContinuous-v3 (多维连续动作)")
    print("=" * 60)
    
    env = gym.make("LunarLander-v3", continuous=True)
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: low={env.action_space.low}, high={env.action_space.high}")
    print(f"  - 含义: [主引擎推力, 侧向引擎推力]")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs[:4]}... (前4维)")
    print()
    
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action=[{action[0]:.3f}, {action[1]:.3f}], reward={reward:.4f}")
    
    print(f"\n累计奖励: {total_reward:.4f}")
    env.close()
    print("✅ LunarLanderContinuous-v3 测试通过!\n")


def test_bipedal_walker():
    """测试 BipedalWalker-v3 多维连续动作环境"""
    print("=" * 60)
    print("测试 BipedalWalker-v3 (多维连续动作)")
    print("=" * 60)
    
    env = gym.make("BipedalWalker-v3")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 含义: 24维状态 (躯干角度/速度、腿部关节角度/速度、地面接触等)")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 4个关节的力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 (24维): {obs[:6]}... (前6维)")
    print()
    
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_str = ", ".join([f"{a:.2f}" for a in action])
        print(f"  Step {step+1}: action=[{action_str}], reward={reward:.4f}")
    
    print(f"\n累计奖励: {total_reward:.4f}")
    env.close()
    print("✅ BipedalWalker-v3 测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("Gymnasium Box2D 环境测试")
    print("=" * 60 + "\n")
    
    test_lunar_lander_discrete()
    test_lunar_lander_continuous()
    test_bipedal_walker()
    
    print("=" * 60)
    print("🎉 所有 Box2D 环境测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
