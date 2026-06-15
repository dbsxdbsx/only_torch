//! On-policy rollout 单步数据（PPO / A2C 族）

/// 单步 on-policy 采集数据
///
/// 存储 PPO 训练所需的全部字段：
/// - `log_prob`：采集时行为策略的 log π(a|s)，PPO ratio 的分母，必须 detach
/// - `value`：采集时 critic 的 V(s)，GAE 计算用
/// - `terminated` / `truncated`：镜像 Gymnasium 双信号（GAE 只 mask terminated，truncated 仍 bootstrap）
#[derive(Debug, Clone)]
pub struct RolloutStep {
    pub obs: Vec<f32>,
    pub action: Vec<f32>,
    pub log_prob: f32,
    pub value: f32,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}
