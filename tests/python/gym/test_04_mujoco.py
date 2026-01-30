# -*- coding: utf-8 -*-
"""
MuJoCo 环境测试
- Ant-v5: 四足蚂蚁行走 (高维连续动作)
- HalfCheetah-v5: 半猎豹奔跑 (高维连续动作)
- Hopper-v5: 单腿跳跃器 (连续动作)
- Walker2d-v5: 双腿行走器 (连续动作)

需要安装: pip install "gymnasium[mujoco]"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
import numpy as np


def test_ant():
    """测试 Ant-v5 高维连续控制环境"""
    print("=" * 60)
    print("测试 Ant-v5 (高维连续动作)")
    print("=" * 60)
    
    env = gym.make("Ant-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 含义: 躯干位置/速度、关节角度/速度、外部力等")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 8个关节的力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 ({obs.shape[0]}维): {obs[:6]}... (前6维)")
    print()
    
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_str = ", ".join([f"{a:.2f}" for a in action[:4]])
        print(f"  Step {step+1}: action=[{action_str}, ...], reward={reward:.4f}")
    
    print(f"\n累计奖励: {total_reward:.4f}")
    env.close()
    print("✅ Ant-v5 测试通过!\n")


def test_half_cheetah():
    """测试 HalfCheetah-v5 高维连续控制环境"""
    print("=" * 60)
    print("测试 HalfCheetah-v5 (高维连续动作)")
    print("=" * 60)
    
    env = gym.make("HalfCheetah-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 6个关节的力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 ({obs.shape[0]}维): {obs[:6]}... (前6维)")
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
    print("✅ HalfCheetah-v5 测试通过!\n")


def test_hopper():
    """测试 Hopper-v5 连续控制环境"""
    print("=" * 60)
    print("测试 Hopper-v5 (连续动作)")
    print("=" * 60)
    
    env = gym.make("Hopper-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 3个关节的力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 ({obs.shape[0]}维): {obs[:6]}... (前6维)")
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
    print("✅ Hopper-v5 测试通过!\n")


def test_walker2d():
    """测试 Walker2d-v5 连续控制环境"""
    print("=" * 60)
    print("测试 Walker2d-v5 (连续动作)")
    print("=" * 60)
    
    env = gym.make("Walker2d-v5")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 形状: {env.action_space.shape}")
    print(f"  - 动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  - 含义: 6个关节的力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 ({obs.shape[0]}维): {obs[:6]}... (前6维)")
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
    print("✅ Walker2d-v5 测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("Gymnasium MuJoCo 环境测试")
    print("=" * 60 + "\n")
    
    test_ant()
    test_half_cheetah()
    test_hopper()
    test_walker2d()
    
    print("=" * 60)
    print("🎉 所有 MuJoCo 环境测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
