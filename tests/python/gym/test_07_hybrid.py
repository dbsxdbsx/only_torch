# -*- coding: utf-8 -*-
"""
gym-hybrid 混合动作空间测试

通过 DI-engine 安装的 gym-hybrid 环境，用于测试离散+连续混合动作空间。

环境特点：
- 动作空间: Tuple(Discrete(3), Box(2,))
- 3 个离散动作: 加速(0)、转向(1)、刹车(2)
- 2 个连续参数: acceleration, rotation
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gym
import gym_hybrid


def test_moving_v0():
    """测试 Moving-v0 环境（不考虑惯性）"""
    print("=" * 60)
    print("测试 Moving-v0 (混合动作空间)")
    print("=" * 60)
    
    env = gym.make('Moving-v0')
    obs = env.reset()
    
    print(f"环境: Moving-v0")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始观察形状: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    
    # 动作空间结构
    discrete_space = env.action_space[0]
    continuous_space = env.action_space[1]
    print(f"  离散动作数: {discrete_space.n}")
    print(f"  连续参数维度: {continuous_space.shape}")
    
    # 测试不同动作
    actions = [
        (0, [0.5, 0.0]),   # 加速
        (1, [0.0, 0.3]),   # 转向
        (2, []),           # 刹车（无参数）
    ]
    
    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: 动作={action}, 奖励={reward:.4f}, done={done}")
        if done:
            break
    
    env.close()
    print(f"  累计奖励: {total_reward:.4f}")
    print("✅ Moving-v0 测试通过!\n")
    return True


def test_sliding_v0():
    """测试 Sliding-v0 环境（考虑惯性，更真实）"""
    print("=" * 60)
    print("测试 Sliding-v0 (混合动作空间，带惯性)")
    print("=" * 60)
    
    env = gym.make('Sliding-v0')
    obs = env.reset()
    
    print(f"环境: Sliding-v0")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 执行随机动作
    total_reward = 0.0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: 随机动作={action}, 奖励={reward:.4f}")
        if done:
            break
    
    env.close()
    print(f"  累计奖励: {total_reward:.4f}")
    print("✅ Sliding-v0 测试通过!\n")
    return True


def test_action_space_structure():
    """测试混合动作空间结构"""
    print("=" * 60)
    print("测试混合动作空间结构")
    print("=" * 60)
    
    env = gym.make('Moving-v0')
    
    # 验证动作空间类型
    action_space = env.action_space
    print(f"动作空间类型: {type(action_space).__name__}")
    
    # 验证是 Tuple 类型
    assert hasattr(action_space, '__len__'), "动作空间应为 Tuple 类型"
    assert len(action_space) == 2, "动作空间应有 2 个子空间"
    
    # 验证离散部分
    discrete = action_space[0]
    print(f"  离散部分: {discrete}")
    assert hasattr(discrete, 'n'), "第一个子空间应为 Discrete"
    assert discrete.n == 3, "离散动作数应为 3"
    
    # 验证连续部分
    continuous = action_space[1]
    print(f"  连续部分: {continuous}")
    assert hasattr(continuous, 'shape'), "第二个子空间应为 Box"
    assert continuous.shape == (2,), "连续参数维度应为 2"
    
    # 验证采样
    for _ in range(3):
        sampled = action_space.sample()
        print(f"  采样动作: {sampled}")
        assert isinstance(sampled, tuple), "采样结果应为 tuple"
        assert len(sampled) == 2, "采样结果应有 2 个元素"
    
    env.close()
    print("✅ 动作空间结构验证通过!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("gym-hybrid 混合动作空间测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Moving-v0", test_moving_v0()))
    results.append(("Sliding-v0", test_sliding_v0()))
    results.append(("动作空间结构", test_action_space_structure()))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有 gym-hybrid 测试通过!")
    else:
        print("\n⚠️ 部分测试未通过")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
