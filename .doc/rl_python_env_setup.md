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

**结论**：项目 **仅支持 Gymnasium**；不要安装或使用 OpenAI Gym（`pip install gym`）。`GymEnv` 不再回退到老 gym。

### 老 gym / 其他库环境怎么办：Python 侧 shimmy 适配

`GymEnv` 在 Rust 侧**只认 `gymnasium.make`**，永不 `import gym`。万一某个环境只在老 gym（或 dm_control / PettingZoo 等）注册，**不要回到 Rust 层恢复 gym 分支**，而是在 Python 侧用 Farama 官方 [`shimmy`](https://shimmy.farama.org/environments/gym/) 把它包成标准 Gymnasium 环境（与五子棋自定义环境同一套 `gymnasium.register` 思路，Rust `GymEnv` 零改动）：

```python
# pip install shimmy[gym-v21]   # 老 gym v0.21 风格；v0.26 风格用 shimmy[gym-v26]
import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

gym.register(
    id="LegacyFoo-v0",
    entry_point=lambda **kw: GymV21CompatibilityV0(env_id="Foo-v0", **kw),
)
```

```rust
let env = GymEnv::new(py, "LegacyFoo-v0"); // 对 Rust 就是普通 Gymnasium 环境
// step() 已是 terminated/truncated 五元组：shimmy 用 info["TimeLimit.truncated"] 正确还原
```

**有损警告**：单一 `done` → `terminated`/`truncated` 的还原依赖环境暴露 `info["TimeLimit.truncated"]`；无此字段则只能当 `terminated`（信息有损）。这也是项目坚持 Gymnasium-only 的理由之一。

**前提警告（2026-06-07 实测）**：shimmy 能用的前提是**底层老 gym 能在当前环境加载**。本机实测 `numpy 2.4.1` + 老 `gym 0.25.2`：老 gym 内部 `np.bool8` 已被 numpy 2.0 移除 → shimmy 直接 `AttributeError: module 'numpy' has no attribute 'bool8'`。即 **numpy ≥ 2.0 下，未维护的老 gym 环境往往根本加载不了，shimmy 也救不回**；此时唯一出路是把环境迁到 Gymnasium 原生（或为该环境单独建一个钉住 `numpy<2` 的 Python 环境）。

**报错约定（Phase 0）**：`GymEnv` 加载失败（id 未在 gymnasium 注册）时**一律 panic + 中文友好提示**——①装 `gymnasium` ②确认 id 已注册到 gymnasium ③老 gym 专属环境用上面的 shimmy 适配；**不** `import gym` 自动回退。`#[should_panic(expected=…)]` + `#[serial]` 单测验证 panic 文案含指引；v0.20 **不**为此引入 `try_new` / `Result`（YAGNI，维持 panic 为主）。

### MuJoCo

- 2021 年被 DeepMind 收购后**完全开源免费**（Apache 2.0 协议）
- **支持 Windows**，最新版专门优化了 Windows 栈大小
- 官网：https://mujoco.org/

### Minari（离线 RL 数据集）

- Farama Foundation 维护（与 Gymnasium 同一团队）
- D4RL 数据集的官方继任者
- 支持 Python 3.8 - 3.12
- 官网：https://minari.farama.org/

### 混合动作：`Platform-v0`（[`hybrid-platform`](https://pypi.org/project/hybrid-platform/)）

- **取代** gym-hybrid / Moving-v0 / Sliding-v0（后者依赖老 `gym`，项目不再使用）
- 安装：`pip install hybrid-platform`（依赖 `gymnasium`、`numpy`、`pygame`）
- 用法：`import gymnasium as gym` + **`import gym_platform`** → `gym.make("Platform-v0")`
- 任务：横版跳台（run / hop / leap + 连续参数）；论文 [Masson et al. 2016](https://arxiv.org/abs/1509.01644)
- 示意图：[gym-platform platform_domain.png](https://github.com/cycraig/gym-platform/blob/master/img/platform_domain.png)

## 生态分层（Gymnasium / 扩展包 / Minari）

| 层级 | 是什么 | 安装 | only_torch 对接 |
|------|--------|------|-----------------|
| **核心** | Gymnasium API + 内置环境 ID（CartPole、Pendulum 等） | `pip install gymnasium` | 见下 **CartPole 版本约定** |
| **扩展环境** | 同一 API，额外物理/任务包 | `gymnasium[mujoco]`、`[box2d]`、`[atari]` 等 | 同上，换 env id |
| **自定义环境** | 用户 `gymnasium.register` | 项目自有 Python 包 | 同上 |
| **离线数据** | **不是**在线 `env.step` 循环 | `pip install minari` + `minari download …` | `MinariDataset`（`src/rl/env/minari.rs`） |

### CartPole 版本约定（2026-05-27）

| 用途 | 环境 ID | 说明 |
|------|---------|------|
| **SAC / MuZero / PPO 架构跑通** | **`CartPole-v0`** | 满分 200；Gym solved = **单局或 100 局均值 ≥ 195** |
| **EfficientZero V2（EZ-V2）终极调优** | **`CartPole-v1`** | 满分 500；性能指标按 EZ-V2 / v1 任务单独定义 |
| **GymEnv 单元测试** | `CartPole-v1` 等 | 与算法验收无关，仅测桥接 API |

**关于「Gymnasium 是否包含老 gym 的一切」**：

- Gymnasium 是 OpenAI Gym 的**官方继任者**（维护方、API 规范），**不是**数学意义上的「老 gym 环境全集 ⊂ Gymnasium」。
- **标准环境**（CartPole、Pendulum、MuJoCo 等）应使用 Gymnasium 及其 extras。
- **混合动作**：用第三方 **`hybrid-platform`**（Platform-v0），不装 gym-hybrid；Rust 不回退老 `gym`。

**离线 RL**：数据在 **Minari**（D4RL 继任），与 `gymnasium` 是**并列包**；装 Gymnasium **不会**自动带上 Minari 数据集。

## 安装与测试环境一览

### 前置条件

```bash
python --version   # 需要 Python 3.10+（Gymnasium 1.3 要求 >=3.10）
pip install --upgrade pip
```

### Gymnasium 版本约定（2026-06-15 锁定）

| 约束 | 值 | 理由 |
|------|-----|------|
| **下界** | `>= 1.3.0` | v1.0+ 为稳定 API（`terminated`/`truncated` 五元组）；1.3.0 为截至 2026-06-15 最新稳定版，实测全链路通过 |
| **上界** | `< 2.0` | Farama 在 v1.0.0 release notes 声明"v1.0 是核心 API 变更的终点"，2.0 才可能有 breaking change |
| **推荐安装** | `pip install "gymnasium>=1.3.0,<2.0"` | 开发与 CI 统一 |

**CartPole-v0 存续状态**：Gymnasium 1.3.0 main 分支仍注册（`max_episode_steps=200, reward_threshold=195.0`），每次 `make` 会打 DeprecationWarning 建议升 v1。翻遍 v0.26–v1.3.0 全部 release notes **无任何移除计划**；被标记 deprecated + 计划移除的只有 MuJoCo v2/v3（基于老 mujoco-py）。v0.20–v0.23 用 v0 做架构跑通标准（200 步 truncation 教学价值），v0.24 EZ-V2 全切 `-v1`。

**实测记录（2026-06-15，Windows + Python 3.11.3）**：Gymnasium 1.3.0 下 CartPole-v0 / CartPole-v1 / Pendulum-v1 / Platform-v0 全链路通过；`just test-serial` RL 测试 17 项全绿（含 hybrid action、gomoku 自定义环境）。

### 环境总表

| 批次 | 安装命令 | 环境名称 | 观察空间 | 动作空间 | 测试场景 |
|:----:|---------|---------|---------|---------|---------|
| 1 | `pip install gymnasium` | CartPole-v0 / CartPole-v1 | Box(4,) | Discrete(2) | v0：SAC/MZ/PPO（≥195）；v1：EfficientZero V2 |
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
| 6 | `pip install hybrid-platform` | Platform-v0 | Tuple(Box(9,), Discrete(200)) | Tuple(Discrete(3), Tuple(Box×3)) | 混合动作；需 `import gym_platform` |
| 7 | 自定义环境 | Gomoku-random-v0 | Box(3,15,15) | Discrete(225) | 五子棋-随机对手 |
| 7 | | Gomoku-naive0-v0 | Box(3,15,15) | Discrete(225) | 五子棋-Naive0 |
| 7 | | Gomoku-naive1-v0 | Box(3,15,15) | Discrete(225) | 五子棋-Naive1 |
| 7 | | Gomoku-naive2-v0 | Box(3,15,15) | Discrete(225) | 五子棋-Naive2 |
| 7 | | Gomoku-naive3-v0 | Box(3,15,15) | Discrete(225) | 五子棋-Naive3 |

> **说明**：
> - Box(n,) 表示 n 维连续向量；Discrete(n) 表示 n 选 1 离散动作
> - Atari 观察空间为 HWC 格式图像 (高度 × 宽度 × 通道)
> - Minari 用于离线 RL，通过 `minari download <dataset_id>` 下载数据集
> - Platform-v0：离散选 run(0)/hop(1)/leap(2)，连续参数见 [hybrid-platform PyPI](https://pypi.org/project/hybrid-platform/)
> - **Tuple obs 处理（Rust `GymEnv` 约定，机制不政策）**：`GymEnv` 按子空间**暴露结构**（不钦定展平）；需要单向量时由上层 flatten helper **按 space 原生顺序拼接**（不赌"谁前谁后"——实测 Platform obs 是 Box 在前 Discrete 在后）、Discrete 编码 one-hot / 标量可配，训练/推理用同一套即可；图像 obs 保 `(C,H,W)` 不展平。action 维持按 `action_space` 原生结构递归处理

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
just py-gym-platform       # test_07: Platform-v0 混合动作（待 justfile 从 py-gym-hybrid 改名）
just py-gym-gomoku         # test_08: 五子棋自定义环境
```

- [x] **批次 1**：基础离散/连续环境 ✅
- [x] **批次 2**：Box2D 环境 ✅
- [x] **批次 3**：MuJoCo 环境 ✅
- [x] **批次 4**：Atari 环境 ✅
- [x] **批次 5**：Minari 离线数据集 ✅
- [x] **批次 6**：混合动作空间（**Platform-v0 / hybrid-platform**）✅ 2026-06-07 实测通过：obs `Tuple(Box(9),Discrete(200))`、action `Tuple(Discrete(3),Box×3)`、step 5 元组
- [x] **批次 7**：五子棋自定义环境 ✅

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

### 五子棋环境（Gomoku）

> **TODO v0.22**：将 `tests/python/custom_envs/gomoku.py` 迁入 **`python/gym_env/gomoku/`**（`pip install -e python/gym_env`）。**扁平布局**：`python/gym_env/` 即 `import gym_env` 的包根（`__init__.py` 与 `pyproject.toml` 同级，**不**建 `gym_env/gym_env/`）；`pyproject.toml` 用 `[tool.setuptools.package-dir] "gym_env" = "."`。子模块：`board.py`（规则 + `legal_mask` + `clone`/`restore`，**无 MCTS**）+ `env.py`（薄 Gym 包装）。注册 **`Gomoku-selfplay-v0`**（训练）与 **`Gomoku-naive*-v0`**（评测）。Rust：**MCTS 在 `src/rl/mcts/`**；`GymEnv` 仅桥接 Board；**无** `GomokuRust`。详见 [RL 主线实施计划 v0.22](../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md)。

项目已实现基于 Gymnasium 的五子棋环境，位于 `tests/python/custom_envs/gomoku.py`。

#### 环境特性

| 特性 | 说明 |
|------|------|
| 棋盘大小 | 可配置（默认 15x15） |
| 获胜条件 | 可配置（默认 5 连珠） |
| 观察空间 | `Box(0, 1, (3, 15, 15), int8)` - 3 通道 |
| 动作空间 | `Discrete(225)` |
| 难度级别 | 5 级：random, naive0~3 |

#### 难度级别说明

| 级别 | 策略 | 强度 |
|------|------|------|
| `random` | 纯随机落子 | 最弱 |
| `naive0` | 只防守，不主动扩展 | 弱 |
| `naive1` | 搜索 3 连珠扩展 | 中等 |
| `naive2` | 搜索 2 连珠扩展 | 较强 |
| `naive3` | 搜索 1 连珠扩展（最积极） | 最强 |

#### 测试结果（随机玩家 vs AI，10 局/级别）

| 对手 | 随机玩家胜率 | 平均步数 |
|------|--------------|----------|
| Random | ~80% | ~59 |
| Naive-0 | 0% | ~28 |
| Naive-1 | 0% | ~27 |
| Naive-2 | 0% | ~10 |
| Naive-3 | 0% | ~5 |

#### 使用方式

```python
import gymnasium as gym
import tests.python.custom_envs  # 触发环境注册

# 方式 1: 通过 gymnasium.make()（5 个难度级别可选）
env = gym.make('Gomoku-random-v0')  # 随机对手
env = gym.make('Gomoku-naive0-v0')  # 最弱
env = gym.make('Gomoku-naive1-v0')
env = gym.make('Gomoku-naive2-v0')  # 推荐
env = gym.make('Gomoku-naive3-v0')  # 最强

# 方式 2: 直接实例化
from tests.python.custom_envs.gomoku import GomokuEnv
env = GomokuEnv(board_size=15, win_length=5, opponent='naive2')

# 标准 Gymnasium API
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
env.render()  # 打印棋盘
```

#### 观察空间编码

```
通道 0: 黑子位置 (1=有棋子, 0=无)
通道 1: 白子位置 (1=有棋子, 0=无)
通道 2: 空位标记 (1=空, 0=已占)
```

#### 奖励设计

```
+1 = 玩家获胜
-1 = 对手获胜 / 非法落子
 0 = 游戏进行中 / 平局
```

### 自定义环境目录结构

**当前（v0.20）**：

```
tests/python/
├── __init__.py
├── gym/                    # 标准环境测试
│   ├── test_01_basic_discrete.py
│   ├── ...
│   └── test_08_gomoku.py   # 五子棋测试
└── custom_envs/            # 自定义环境
    ├── __init__.py         # 导入时自动注册所有环境
    └── gomoku.py           # 五子棋环境实现
```

**目标（v0.22，`python/gym_env/` 扁平包）**：

```
python/
└── gym_env/                # pip install -e python/gym_env；import gym_env
    ├── pyproject.toml      # package-dir: "gym_env" = "."
    ├── __init__.py         # gymnasium.register
    └── gomoku/
        ├── __init__.py
        ├── board.py
        ├── env.py
        └── opponents.py
```

### Rust 端使用自定义环境

```rust
Python::attach(|py| {
    // 确保自定义环境模块被导入（触发注册）
    py.import("tests.python.custom_envs").expect("导入自定义环境模块失败");

    // 使用五子棋环境（5 个难度可选）
    let env = GymEnv::new(py, "Gomoku-naive2-v0");  // 推荐
    env.print_env_basic_info();
});
```

## Rust 端环境加载（Gymnasium-only）

### 设计目标（v0.20.0+）

`GymEnv` **只**通过 `gymnasium.make(env_id)` 创建环境：

- **内置 / extras 环境**：CartPole-v1、Pendulum-v1、MuJoCo、Atari 等（需对应 `pip install` extras）
- **自定义环境**：已 `gymnasium.register` 的包（如五子棋）
- **不支持**：OpenAI Gym（`import gym`）、gym-hybrid（Moving-v0 / Sliding-v0）
- **混合动作**：`Platform-v0`（先 `import gym_platform`）

### 加载策略

```
GymEnv::new(py, "CartPole-v1")
        ↓
gymnasium.make("CartPole-v1")
        ↓
成功 → 统一 reset/step API（step 透出 terminated + truncated 两个信号，不再合并 done）
失败 → 明确报错（缺 extra / 未 register），不尝试 gym
```

### API（仅 Gymnasium）

| 方面 | 约定 |
|------|------|
| reset | `(obs, info)`，`reset(seed=…)` |
| step | `(obs, reward, terminated, truncated, info)` |
| step（Rust 侧） | 透出 `(obs, reward, terminated, truncated)`，**不合并 `done`**；便捷 `Transition::is_episode_end() = terminated \|\| truncated` |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Rust 层 (only_torch)                     │
├─────────────────────────────────────────────────────────────┤
│  GymEnv                                                      │
│  ├── new(py, "CartPole-v1")     → gymnasium                  │
│  ├── new(py, "Gomoku-v0")       → 自定义 (gymnasium.register) │
│  ├── new(py, "Platform-v0")     → hybrid-platform（import gym_platform）│
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
│  CartPole-v1      Platform-v0           Gomoku-*-v0   │
│  Pendulum-v1      (hybrid-platform)     (5 难度级别)        │
│  MuJoCo envs...                                            │
└─────────────────────────────────────────────────────────────┘
```

## 参考资料

- [Gymnasium 官方文档](https://gymnasium.farama.org/)
- [Gymnasium 自定义环境教程](https://gymnasium.farama.org/introduction/create_custom_env/)
- [MuJoCo 官网](https://mujoco.org/)
- [Minari 官方文档](https://minari.farama.org/)
- [Farama Foundation](https://farama.org/)
- [hybrid-platform](https://pypi.org/project/hybrid-platform/) - Platform-v0 混合动作（Gymnasium）
- [gym-platform](https://github.com/cycraig/gym-platform) - Platform 原实现与论文截图
- [Masson et al. 2016](https://arxiv.org/abs/1509.01644) - Parameterized Actions / Platform domain