# -*- coding: utf-8 -*-
"""
基础离散动作环境测试
- CartPole-v1: 经典的杆平衡问题
- Acrobot-v1: 双连杆摆动问题

这些环境只需要基础的 gymnasium 包，无需额外依赖。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym


def test_cartpole():
    """测试 CartPole-v1 离散动作环境"""
    print("=" * 60)
    print("测试 CartPole-v1 (离散动作)")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    
    # 打印环境信息
    print(f"观察空间: {env.observation_space}")
    print(f"  - 类型: Box")
    print(f"  - 形状: {env.observation_space.shape}")
    print(f"  - 范围: [{env.observation_space.low}, {env.observation_space.high}]")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 类型: Discrete")
    print(f"  - 动作数: {env.action_space.n}")
    print(f"  - 动作含义: 0=向左推, 1=向右推")
    print()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs}")
    print(f"  - cart_position: {obs[0]:.4f}")
    print(f"  - cart_velocity: {obs[1]:.4f}")
    print(f"  - pole_angle: {obs[2]:.4f}")
    print(f"  - pole_angular_velocity: {obs[3]:.4f}")
    print()
    
    # 执行几步随机动作
    print("执行 5 步随机动作:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={action}, reward={reward}, terminated={terminated}")
    
    print(f"\n累计奖励: {total_reward}")
    env.close()
    print("✅ CartPole-v1 测试通过!\n")


def test_acrobot():
    """测试 Acrobot-v1 离散动作环境"""
    print("=" * 60)
    print("测试 Acrobot-v1 (离散动作)")
    print("=" * 60)
    
    env = gym.make("Acrobot-v1")
    
    print(f"观察空间: {env.observation_space}")
    print(f"  - 形状: {env.observation_space.shape}")
    print()
    print(f"动作空间: {env.action_space}")
    print(f"  - 动作数: {env.action_space.n}")
    print(f"  - 动作含义: 0=负力矩, 1=无力矩, 2=正力矩")
    print()
    
    obs, info = env.reset(seed=42)
    print(f"初始观察 (6维): {obs}")
    print()
    
    # 执行几步
    print("执行 5 步随机动作:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: action={action}, reward={reward}")
    
    env.close()
    print("✅ Acrobot-v1 测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("Gymnasium 基础离散动作环境测试")
    print("=" * 60 + "\n")
    
    test_cartpole()
    test_acrobot()
    
    print("=" * 60)
    print("🎉 所有基础离散环境测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
