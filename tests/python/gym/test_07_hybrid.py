# -*- coding: utf-8 -*-
"""
Platform-v0 混合动作空间测试（替代已弃用的 gym-hybrid Moving/Sliding）

使用 hybrid-platform 包的 Platform-v0 环境：
- 观察空间: Tuple(Box(9,), Discrete(200))
- 动作空间: Tuple(Discrete(3), Tuple(Box(1,), Box(1,), Box(1,)))
- 离散：平台选择（0, 1, 2）
- 连续：跳跃力度 [0,30]、横向速度 [0,720]、跳跃时机 [0,430]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
import gym_platform


def test_platform_v0():
    """测试 Platform-v0 环境基本功能"""
    print("=" * 60)
    print("测试 Platform-v0 (混合动作空间)")
    print("=" * 60)

    env = gym.make('Platform-v0')
    obs, info = env.reset(seed=42)

    print(f"环境: Platform-v0")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始观察: {obs}")

    # Tuple obs 结构
    assert isinstance(obs, tuple), "obs 应为 tuple"
    assert len(obs) == 2, "obs 应有 2 个子空间"
    assert len(obs[0]) == 9, "Box 子空间长度应为 9"
    assert isinstance(obs[1], (int,)), f"Discrete 子空间应为 int, got {type(obs[1])}"

    # 执行几步
    total_reward = 0.0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        print(f"  Step {i+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break

    env.close()
    print(f"  累计奖励: {total_reward:.4f}")
    print("PASS: Platform-v0 测试通过!\n")
    return True


def test_action_space_structure():
    """测试 Platform-v0 动作空间结构"""
    print("=" * 60)
    print("测试 Platform-v0 动作空间结构")
    print("=" * 60)

    env = gym.make('Platform-v0')
    action_space = env.action_space

    print(f"动作空间类型: {type(action_space).__name__}")

    # Tuple(Discrete(3), Tuple(Box, Box, Box))
    assert len(action_space.spaces) == 2, "动作空间应有 2 个顶层子空间"

    discrete = action_space.spaces[0]
    print(f"  离散部分: {discrete}")
    assert discrete.n == 3, "离散动作数应为 3"

    cont_tuple = action_space.spaces[1]
    print(f"  连续部分: {cont_tuple}")
    assert len(cont_tuple.spaces) == 3, "连续参数应有 3 个子空间"

    # 验证连续参数范围
    ranges = [(0, 30), (0, 720), (0, 430)]
    for i, (low, high) in enumerate(ranges):
        sub = cont_tuple.spaces[i]
        assert sub.low[0] == low, f"连续参数[{i}] 下界应为 {low}"
        assert sub.high[0] == high, f"连续参数[{i}] 上界应为 {high}"
        print(f"    param[{i}]: [{low}, {high}]")

    # 验证采样
    for _ in range(3):
        sampled = action_space.sample()
        print(f"  采样动作: {sampled}")
        assert isinstance(sampled, tuple)

    env.close()
    print("PASS: 动作空间结构验证通过!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("Platform-v0 混合动作空间测试")
    print("=" * 60 + "\n")

    results = []
    results.append(("Platform-v0 基本功能", test_platform_v0()))
    results.append(("动作空间结构", test_action_space_structure()))

    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n通过: {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
