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

### gym-hybrid（混合动作空间）

- 提供离散+连续混合动作空间的测试环境
- 包含 `Moving-v0` 和 `Sliding-v0` 两个环境
- **推荐通过 DI-engine 安装**（内置集成，维护更好）
- DI-engine GitHub：https://github.com/opendilab/DI-engine（3.5k+ stars）
- gym-hybrid 文档：https://di-engine-docs.readthedocs.io/en/latest/13_envs/gym_hybrid.html

## 安装与测试环境一览

### 前置条件

```bash
python --version   # 需要 Python 3.8 - 3.12（推荐 3.10+）
pip install --upgrade pip
```

### 环境总表

| 批次 | 安装命令 | 环境名称 | 观察空间 | 动作空间 | 测试场景 |
|:----:|---------|---------|---------|---------|---------|
| 1 | `pip install gymnasium` | CartPole-v1 | Box(4,) | Discrete(2) | 基础离散 |
| 1 | | Acrobot-v1 | Box(6,) | Discrete(3) | 基础离散 |
| 1 | | Pendulum-v1 | Box(3,) | Box(1,) [-2,2] | 基础连续 |
| 1 | | MountainCarContinuous-v0 | Box(2,) | Box(1,) [-1,1] | 基础连续 |
| 2 | `pip install gymnasium[box2d]` | LunarLander-v3 | Box(8,) | Discrete(4) | Box2D 离散 |
| 2 | | BipedalWalker-v3 | Box(24,) | Box(4,) [-1,1] | Box2D 多维连续 |
| 3 | `pip install gymnasium[mujoco]` | Ant-v5 | Box(27,) | Box(8,) [-1,1] | MuJoCo 高维控制 |
| 3 | | HalfCheetah-v5 | Box(17,) | Box(6,) [-1,1] | MuJoCo 高维控制 |
| 3 | | Hopper-v5 | Box(11,) | Box(3,) [-1,1] | MuJoCo 连续控制 |
| 3 | | Walker2d-v5 | Box(17,) | Box(6,) [-1,1] | MuJoCo 连续控制 |
| 4 | `pip install gymnasium[atari]`<br>`pip install gymnasium[accept-rom-license]` | ALE/Breakout-v5 | Box(210,160,3) | Discrete(4) | Atari 图像+离散 |
| 4 | | ALE/Pong-v5 | Box(210,160,3) | Discrete(6) | Atari 图像+离散 |
| 4 | | ALE/SpaceInvaders-v5 | Box(210,160,3) | Discrete(6) | Atari 图像+离散 |
| 5 | `pip install minari` | D4RL/pointmaze/umaze-v2 | 离线数据集 | 离线数据集 | Offline RL |
| 6 | `pip install DI-engine` | Moving-v0 | Box(10,) | Tuple(Discrete(3), Box(2,)) | 混合动作空间 |
| 6 | | Sliding-v0 | Box(10,) | Tuple(Discrete(3), Box(2,)) | 混合动作空间 |

> **说明**：
> - Box(n,) 表示 n 维连续向量；Discrete(n) 表示 n 选 1 离散动作
> - Atari 观察空间为 HWC 格式图像 (高度 × 宽度 × 通道)
> - Minari 用于离线 RL，通过 `minari download <dataset_id>` 下载数据集
> - gym-hybrid 的动作格式为 `(action_id, [param1, param2])`，action_id 对应加速/转向/刹车

### 按学习范式分类

| 范式 | 工具 | 说明 |
|-----|------|------|
| Online RL | Gymnasium | 实时与环境交互 |
| Off-policy RL | Gymnasium + 经验回放 | 可复用历史数据 |
| Offline RL | Minari | 纯离线数据集学习 |

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

```bash
# 运行所有测试
just py-gym                # 或 python tests/python/gym/run_all_tests.py

# 分步运行
just py-gym-basic          # test_01 + test_02: 基础离散/连续环境
just py-gym-box2d          # test_03: Box2D 环境
just py-gym-mujoco         # test_04: MuJoCo 环境
just py-gym-atari          # test_05: Atari 环境
just py-gym-minari         # test_06: Minari 离线数据集
just py-gym-hybrid         # test_07: 混合动作空间
```

- [x] **批次 1**：基础离散/连续环境 ✅
- [x] **批次 2**：Box2D 环境 ✅
- [x] **批次 3**：MuJoCo 环境 ✅
- [x] **批次 4**：Atari 环境 ✅
- [x] **批次 5**：Minari 离线数据集 ✅
- [x] **批次 6**：混合动作空间（gym-hybrid）✅

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
- [gym-hybrid](https://github.com/thomashirtz/gym-hybrid) - 混合动作空间环境
- [DI-engine](https://github.com/opendilab/DI-engine) - 生产级 RL 框架（含 PDQN/MPDQN/HPPO 算法）
- [DI-engine gym-hybrid 文档](https://di-engine-docs.readthedocs.io/en/latest/13_envs/gym_hybrid.html)