# -*- coding: utf-8 -*-
"""
基础连续动作环境测试
- Pendulum-v1: 倒立摆问题 (单维连续动作)
- MountainCarContinuous-v0: 山地车问题 (单维连续动作)

这些环境只需要基础的 gymnasium 包，无需额外依赖。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
import numpy as np


def test_pendulum():
    """测试 Pendulum-v1 连续动作环境"""
    print("=" * 60)
    print("测试 Pendulum-v1 (连续动作)")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    
    # 打印环境信息
    print(f"观察空间: {env.observation_space}")
    print(f"  - 类型: Box")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 类型: Box (连续)")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 施加的力矩")
    print()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs}")
    print(f"  - cos(theta): {obs[0]:.4f}")
    print(f"  - sin(theta): {obs[1]:.4f}")
    print(f"  - angular_velocity: {obs[2]:.4f}")
    print()
    
    # 执行几步随机动作
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={action[0]:.4f}, reward={reward:.4f}")
    
    print(f"\n累计奖励: {total_reward:.4f}")
    env.close()
    print("✅ Pendulum-v1 测试通过!\n")


def test_mountain_car_continuous():
    """测试 MountainCarContinuous-v0 连续动作环境"""
    print("=" * 60)
    print("测试 MountainCarContinuous-v0 (连续动作)")
    print("=" * 60)
    
    env = gym.make("MountainCarContinuous-v0")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 含义: [位置, 速度]")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs}")
    print(f"  - position: {obs[0]:.4f}")
    print(f"  - velocity: {obs[1]:.4f}")
    print()
    
    # 执行几步
    print("执行 5 步随机动作:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: action={action[0]:.4f}, reward={reward:.4f}")
    
    env.close()
    print("✅ MountainCarContinuous-v0 测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("Gymnasium 基础连续动作环境测试")
    print("=" * 60 + "\n")
    
    test_pendulum()
    test_mountain_car_continuous()
    
    print("=" * 60)
    print("🎉 所有基础连续环境测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
