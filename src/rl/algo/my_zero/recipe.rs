//! 按环境内置算法配方（内部；用户 API 不暴露组件开关）。
//!
//! 团队 promote 组件时只改此模块；公开入口 [`MyZero::new`](super::my_zero::MyZero::new) 自动套用。

use super::component::Components;
use super::config::ActionPlan;
use super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS;

/// CartPole 已验收的 MyZero 组件栈（consistency + reconstruction + Sampled）。
fn cartpole_stack() -> Components {
    let mut c = Components::base();
    c.consistency = true;
    c.reconstruction = true;
    c.sampled = true;
    // K 由 sampled_params 按 N、sims 自动算
    // reanalyze / completedQ / Gumbel：CartPole promote 暂缓 → .issue/items/
    c.reanalyze = false;
    c
}

/// Pendulum 当前诊断栈：先复用 CartPole 已验收组件，但不把 Pendulum 裁决标记为已通过。
fn pendulum_diagnostic_stack() -> Components {
    cartpole_stack()
}

/// 图像（ALE/Atari）base 栈：consistency ON + reconstruction OFF。
/// consistency ON（图像是其 native 场景）+ reconstruction OFF（decoder 重建高维像素
/// 代价大，A/B 臂另行开启）+ two-hot + raw [0,1] obs。**未 promote**，基准与消融进行中。
fn image_base_stack() -> Components {
    let mut c = Components::base();
    c.consistency = true;
    c
}

/// 棋盘（Gomoku）栈 = **base 组件全关**（2026-07-05 定型）。
///
/// 九臂消融 3-seed 裁决（账本见 `examples/my_zero/gomoku/README.md`）：
/// Gumbel+completedQ / consistency / reconstruction / CNN 表征 / 预算×5 /
/// replay×8 / lr 3e-3 全部中性或偏害，唯一弱阳性 = D4 对称增广
/// （naive0 中位 0.10 → 0.15，未达 promote 线，复核入口见
/// `.issue/items/gomoku_naive0_tactical_wall.md`）。base = 双门槛 3/3 达标配置。
fn board_stack() -> Components {
    Components::base()
}

/// 大联合动作空间的接口纵切：只开启 Sampled 候选，避免每节点展开完整 catalog。
fn structured_action_stack() -> Components {
    let mut c = Components::base();
    c.sampled = true;
    c
}

/// MinAtar 基础栈：consistency ON（空间观测原生场景）+ reconstruction OFF（10×10 二值像素无需重建）。
/// 与 Atari image_base_stack 同逻辑，但不经过 ImagePipe，走 Board 路径。
fn minatar_base_stack() -> Components {
    let mut c = Components::base();
    c.consistency = true;
    c
}

/// 给定 Gymnasium `env_id` 返回当前内置组件组合。
pub(crate) fn components_for(env_id: &str) -> Components {
    if env_id.starts_with("MinAtar/") {
        return minatar_base_stack();
    }
    if env_id.starts_with("ALE/") {
        return image_base_stack();
    }
    if env_id.starts_with("Gomoku-") {
        return board_stack();
    }
    match env_id {
        "CartPole-v1" => cartpole_stack(),
        "Pendulum-v1" => pendulum_diagnostic_stack(),
        "Platform-v0" | "MyZero-MultiDiscrete-v0" => structured_action_stack(),
        _ => Components::base(),
    }
}

/// 给定 `env_id` 返回默认动作方案（连续 env 用 Sampled MuZero 默认 B=7；离散 env 用 Auto）。
pub(crate) fn action_plan_for(env_id: &str) -> ActionPlan {
    match env_id {
        "Pendulum-v1" => ActionPlan::Discretize {
            buckets: DEFAULT_CONTINUOUS_BUCKETS,
        },
        _ => ActionPlan::Auto,
    }
}
