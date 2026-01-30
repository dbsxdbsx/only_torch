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

## Rust 测试并行问题

### 问题

pyo3 在多线程并行测试时，Python 模块导入存在竞争条件，导致 `circular import` 等错误。

### 解决方案

使用 `serial_test` crate 控制测试串行执行：

```toml
# Cargo.toml
[dev-dependencies]
serial_test = "3"
```

```rust
use serial_test::serial;

#[test]
#[serial]  // 带此属性的测试会串行执行
fn test_gym_env() {
    Python::attach(|py| { ... });
}
```

### 说明

- **这不是 GIL 性能问题**：Python 3.13+ 的 free-threading 模式无法解决此问题
- **本质是模块导入竞争**：多线程同时初始化/导入 Python 模块时的竞争条件
- 所有 `src/rl/tests/` 下的测试已添加 `#[serial]` 属性
- 运行命令：`cargo test rl::tests`（无需 `--test-threads=1`）

## 自定义环境支持

### Gymnasium 自定义环境

Gymnasium 完全支持自定义环境，这是创建新环境的推荐方式。

#### 创建自定义环境

```python
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

class GomokuEnv(gym.Env):
    """五子棋自定义环境示例"""
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, board_size=15, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        
        # 动作空间：选择棋盘上的一个位置 (0 ~ board_size^2 - 1)
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 观察空间：棋盘状态 (-1=对手, 0=空, 1=自己)
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(board_size, board_size), 
            dtype=np.int8
        )
        
        self.board = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return self.board.copy(), {}
    
    def step(self, action):
        row, col = divmod(action, self.board_size)
        # ... 落子逻辑、胜负判断 ...
        reward = 0.0
        terminated = False
        truncated = False
        return self.board.copy(), reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            # 打印棋盘
            pass
    
    def close(self):
        pass

# 注册环境
register(
    id="Gomoku-v0",
    entry_point="custom_envs.gomoku:GomokuEnv",
    max_episode_steps=225,  # 15x15 棋盘最多 225 步
)
```

#### 项目中的自定义环境组织

建议在 `tests/python/custom_envs/` 目录下组织自定义环境：

```
tests/python/
├── gym/                    # 标准环境测试
│   ├── test_01_basic_discrete.py
│   └── ...
└── custom_envs/            # 自定义环境
    ├── __init__.py         # 导入时自动注册所有环境
    ├── gomoku.py           # 五子棋环境
    └── ...
```

`__init__.py` 示例：
```python
# 导入时自动注册所有自定义环境
from gymnasium.envs.registration import register

register(id="Gomoku-v0", entry_point="tests.python.custom_envs.gomoku:GomokuEnv")
# 添加更多自定义环境...
```

### Rust 端使用自定义环境

```rust
Python::attach(|py| {
    // 确保自定义环境模块被导入（触发注册）
    py.import("tests.python.custom_envs").expect("导入自定义环境模块失败");
    
    // 然后正常使用
    let env = GymEnv::new(py, "Gomoku-v0");
    env.print_env_basic_info();
});
```

## Rust 端智能环境加载

### 设计目标

Rust 层的 `GymEnv` 对用户透明地处理各种 Python 环境来源：

- **gymnasium 环境**：CartPole-v1, Pendulum-v1, MuJoCo 环境等
- **gym 环境**：gym-hybrid 的 Moving-v0, Sliding-v0 等（仅支持老 gym）
- **自定义环境**：注册到 gymnasium 的用户自定义环境

### 加载策略

```
用户调用: GymEnv::new(py, "Moving-v0")
                    ↓
         1. 尝试 gymnasium.make("Moving-v0")
                    ↓ (失败: 环境未注册)
         2. 尝试 gym.make("Moving-v0")
                    ↓ (成功)
         3. 返回环境，用户无感知
```

### gym 与 gymnasium 的 API 差异

| 方面 | gymnasium | gym (legacy) |
|------|-----------|--------------|
| reset() 返回值 | `(obs, info)` | `obs` 或 `(obs, info)` |
| step() 返回值 | `(obs, reward, terminated, truncated, info)` | `(obs, reward, done, info)` |
| seed 设置 | `reset(seed=42)` | `env.seed(42)` |

这些差异由 Rust 端的 `GymEnv` 内部统一处理，对用户完全透明。

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Rust 层 (only_torch)                     │
├─────────────────────────────────────────────────────────────┤
│  GymEnv                                                      │
│  ├── new(py, "CartPole-v1")     → 自动选择 gymnasium         │
│  ├── new(py, "Moving-v0")       → 自动回退到 gym             │
│  ├── new(py, "Gomoku-v0")       → 自定义环境 (gymnasium)     │
│  │                                                          │
│  ├── reset(seed) → Vec<Vec<f32>>                           │
│  ├── step(action) → (obs, reward, done)                    │
│  └── sample_action() → Vec<f32>                            │
├─────────────────────────────────────────────────────────────┤
│                     Python 层                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  gymnasium  │  │     gym     │  │  自定义环境模块      │ │
│  │  (主要)     │  │  (兼容)     │  │  (注册到gymnasium)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         ↑                ↑                    ↑             │
│         │                │                    │             │
│  CartPole-v1      Moving-v0             Gomoku-v0          │
│  Pendulum-v1      Sliding-v0            ...                │
│  MuJoCo envs...                                            │
└─────────────────────────────────────────────────────────────┘
```

## 参考资料

- [Gymnasium 官方文档](https://gymnasium.farama.org/)
- [Gymnasium 自定义环境教程](https://gymnasium.farama.org/introduction/create_custom_env/)
- [MuJoCo 官网](https://mujoco.org/)
- [Minari 官方文档](https://minari.farama.org/)
- [Farama Foundation](https://farama.org/)
- [gym-hybrid](https://github.com/thomashirtz/gym-hybrid) - 混合动作空间环境
- [DI-engine](https://github.com/opendilab/DI-engine) - 生产级 RL 框架（含 PDQN/MPDQN/HPPO 算法）
- [DI-engine gym-hybrid 文档](https://di-engine-docs.readthedocs.io/en/latest/13_envs/gym_hybrid.html)