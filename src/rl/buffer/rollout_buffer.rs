//! On-policy rollout 缓冲区（PPO / A2C 族，用完即弃，不 impl BufferItem）

use super::rollout::RolloutStep;

/// 固定 n_steps 的 on-policy 采集缓冲区
///
/// 语义：采满 → 计算 GAE → 训练若干 epoch → `clear()` 清空 → 重新采集。
/// 与 `ReplayBuffer` 的区别：不做有放回随机采样，不 impl `BufferItem`。
pub struct RolloutBuffer {
    steps: Vec<RolloutStep>,
    capacity: usize,
}

impl RolloutBuffer {
    /// 创建指定容量的 rollout 缓冲区
    pub fn new(capacity: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// 添加一步采集数据
    ///
    /// 若已满则 panic（调用方应在 `is_full()` 时先处理）
    pub fn push(&mut self, step: RolloutStep) {
        assert!(
            !self.is_full(),
            "RolloutBuffer 已满（capacity={}），push 前应先检查 is_full() 并处理",
            self.capacity
        );
        self.steps.push(step);
    }

    /// 是否已采满
    pub fn is_full(&self) -> bool {
        self.steps.len() >= self.capacity
    }

    /// 当前步数
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// 容量
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 获取所有步骤的只读引用（供 GAE / batch 构建使用）
    pub fn steps(&self) -> &[RolloutStep] {
        &self.steps
    }

    /// 清空缓冲区（on-policy：每次更新后清空，重新采集）
    pub fn clear(&mut self) {
        self.steps.clear();
    }
}
