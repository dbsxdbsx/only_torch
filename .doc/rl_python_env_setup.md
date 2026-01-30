# Windows 下强化学习 Python 环境搭建指南

> 本文档记录如何在 Windows 上搭建强化学习所需的 Python 环境，为后续 Rust 桥接做准备。

## 背景

在开发 Rust 强化学习模块之前，需要先确保 Python 侧的 Gym 环境能够正常运行。由于 Python RL 生态在 Windows 上历来存在各种兼容性问题，采用**渐进式验证**策略：

1. 先在纯 Python 层面验证环境可用
2. 再进行 Rust pyo3 桥接
3. 最后对接 only_torch Tensor

## 核心工具选型

### Gymnasium（推荐，替代 OpenAI Gym）

| 对比项 | OpenAI Gym | Gymnasium |
|--------|-----------|-----------|
| 维护状态 | 2022 年后停止维护 | Farama Foundation 活跃维护 |
| 推荐程度 | 已过时 | **官方推荐替代品** |
| 官网 | - | https://gymnasium.farama.org/ |

**结论**：直接使用 **Gymnasium**，不要用老的 OpenAI Gym。

### MuJoCo

- 2021 年被 DeepMind 收购后**完全开源免费**（Apache 2.0 协议）
- **支持 Windows**，最新版专门优化了 Windows 栈大小
- 官网：https://mujoco.org/

### Minari（离线 RL 数据集）

- Farama Foundation 维护（与 Gymnasium 同一团队）
- D4RL 数据集的官方继任者
- 支持 Python 3.8 - 3.12
- 官网：https://minari.farama.org/

## 环境分类与优先级

### 按输入/动作类型分类

| 输入类型 | 动作类型 | 环境示例 | 难度 |
|---------|---------|---------|------|
| 简单数字 | 离散 | CartPole-v1 | ⭐ |
| 简单数字 | 连续 | Pendulum-v1 | ⭐ |
| 简单数字 | 多维连续 | LunarLanderContinuous-v3 | ⭐⭐ |
| 简单数字 | 多维连续 | Ant-v5, HalfCheetah-v5 | ⭐⭐⭐ |
| 图像 | 离散 | ALE/Breakout-v5 | ⭐⭐⭐ |
| 图像 | 多维连续 | CarRacing-v2 | ⭐⭐⭐⭐ |

### 按学习范式分类

| 范式 | 工具 | 说明 |
|-----|------|------|
| Online RL | Gymnasium | 实时与环境交互 |
| Off-policy RL | Gymnasium + 经验回放 | 可复用历史数据 |
| Offline RL | Minari | 纯离线数据集学习 |

## 安装步骤

### 前置条件

- Python 3.8 - 3.12（推荐 3.10+）
- pip 最新版

```bash
python --version
pip install --upgrade pip
```

### 第一批：基础环境（必须先跑通）

```bash
pip install gymnasium
```

验证环境：
- `CartPole-v1` - 离散动作
- `Pendulum-v1` - 连续动作
- `Acrobot-v1` - 离散动作
- `MountainCarContinuous-v0` - 连续动作

### 第二批：Box2D 环境

```bash
pip install gymnasium[box2d]
```

> **注意**：Gymnasium 1.2.3+ 使用 `box2d` 包替代了旧的 `box2d-py`。

验证环境：
- `LunarLander-v3` - 离散动作
- `LunarLander-v3` (continuous=True) - 多维连续
- `BipedalWalker-v3` - 多维连续
- `CarRacing-v2` - 图像输入 + 多维连续

### 第三批：MuJoCo 环境

```bash
pip install gymnasium[mujoco]
```

验证环境：
- `Ant-v5` - 高维连续控制
- `HalfCheetah-v5` - 高维连续控制
- `Hopper-v5` - 连续控制
- `Walker2d-v5` - 连续控制
- `Humanoid-v5` - 超高维连续控制

### 第四批：Atari 环境（可选）

```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

验证环境：
- `ALE/Breakout-v5` - 图像输入 + 离散动作
- `ALE/Pong-v5` - 图像输入 + 离散动作

### 第五批：离线 RL 数据集（可选）

```bash
pip install minari
# 或完整安装
pip install "minari[all]"
```

```bash
# 查看可用数据集
minari list remote

# 下载数据集
minari download D4RL/pointmaze/umaze-v2
```

## 验证脚本

在 `tests/` 目录下创建 Python 测试脚本验证环境。

### 基础验证脚本

```python
# tests/test_gym_basic.py
import gymnasium as gym

def test_cartpole():
    """测试离散动作环境"""
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"CartPole-v1:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始观察: {obs}")

    # 执行一步
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  动作: {action}, 奖励: {reward}")
    env.close()

def test_pendulum():
    """测试连续动作环境"""
    env = gym.make("Pendulum-v1")
    obs, info = env.reset()
    print(f"\nPendulum-v1:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  动作范围: [{env.action_space.low}, {env.action_space.high}]")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  动作: {action}, 奖励: {reward}")
    env.close()

if __name__ == "__main__":
    test_cartpole()
    test_pendulum()
    print("\n✅ 基础环境测试通过！")
```

### Box2D 验证脚本

```python
# tests/test_gym_box2d.py
import gymnasium as gym

def test_lunar_lander_continuous():
    """测试多维连续动作环境"""
    env = gym.make("LunarLanderContinuous-v2")
    obs, info = env.reset()
    print(f"LunarLanderContinuous-v2:")
    print(f"  观察空间: {env.observation_space} (shape={env.observation_space.shape})")
    print(f"  动作空间: {env.action_space} (shape={env.action_space.shape})")
    print(f"  动作范围: low={env.action_space.low}, high={env.action_space.high}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  动作: {action}, 奖励: {reward:.4f}")
    env.close()

if __name__ == "__main__":
    test_lunar_lander_continuous()
    print("\n✅ Box2D 环境测试通过！")
```

### MuJoCo 验证脚本

```python
# tests/test_gym_mujoco.py
import gymnasium as gym

def test_ant():
    """测试高维连续控制环境"""
    env = gym.make("Ant-v4")
    obs, info = env.reset()
    print(f"Ant-v4:")
    print(f"  观察空间: {env.observation_space} (shape={env.observation_space.shape})")
    print(f"  动作空间: {env.action_space} (shape={env.action_space.shape})")
    print(f"  动作维度: {env.action_space.shape[0]}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  奖励: {reward:.4f}")
    env.close()

if __name__ == "__main__":
    test_ant()
    print("\n✅ MuJoCo 环境测试通过！")
```

### Minari 离线数据集验证

```python
# tests/test_minari.py
import minari

def test_minari_dataset():
    """测试离线 RL 数据集"""
    # 列出本地已下载的数据集
    local_datasets = minari.list_local_datasets()
    print(f"本地数据集: {local_datasets}")

    # 如果有数据集，加载并检查
    if local_datasets:
        dataset_id = list(local_datasets.keys())[0]
        dataset = minari.load_dataset(dataset_id)
        print(f"\n数据集 {dataset_id}:")
        print(f"  总 episode 数: {dataset.total_episodes}")
        print(f"  总 step 数: {dataset.total_steps}")

        # 采样一个 episode
        episode = dataset.sample_episodes(1)[0]
        print(f"  Episode 长度: {len(episode.observations)}")

if __name__ == "__main__":
    test_minari_dataset()
```

## Windows 常见问题

### Box2D 安装失败

如果 `pip install gymnasium[box2d]` 失败，尝试：

```bash
# 方法 1：使用 conda
conda install -c conda-forge box2d-py

# 方法 2：安装预编译 wheel
pip install box2d
```

### MuJoCo 渲染问题

如果 MuJoCo 环境渲染失败：

```bash
# 确保安装了渲染后端
pip install gymnasium[mujoco]

# 使用 rgb_array 模式代替 human 模式
env = gym.make("Ant-v4", render_mode="rgb_array")
```

### Atari ROM 许可证

首次使用 Atari 环境需要接受 ROM 许可证：

```bash
pip install gymnasium[accept-rom-license]
```

## 验证清单

按顺序验证，确保每一步都通过后再进行下一步：

```bash
# 运行所有测试
python tests/python/gym/run_all_tests.py

# 或分步运行
python tests/python/gym/test_01_basic_discrete.py    # 基础离散环境
python tests/python/gym/test_02_basic_continuous.py  # 基础连续环境
python tests/python/gym/test_03_box2d.py             # Box2D 环境
python tests/python/gym/test_04_mujoco.py            # MuJoCo 环境
```

- [x] **第一批**：基础离散/连续环境 ✅
- [x] **第二批**：Box2D 环境 ✅
- [x] **第三批**：MuJoCo 环境 ✅
- [ ] **第四批**：Atari 环境（可选）
- [ ] **第五批**：Minari 离线数据集（可选）

## 后续步骤

Python 环境验证通过后，进入 Rust 桥接阶段：

1. 使用 `pyo3` 创建 Rust-Python 绑定
2. 参考 [RustRL 项目](https://github.com/dbsxdbsx/rustRL) 的 `gym_env.rs` 实现
3. 将 `tch::Tensor` 替换为 only_torch 的 Tensor

## 参考资料

- [Gymnasium 官方文档](https://gymnasium.farama.org/)
- [MuJoCo 官网](https://mujoco.org/)
- [Minari 官方文档](https://minari.farama.org/)
- [Farama Foundation](https://farama.org/)
