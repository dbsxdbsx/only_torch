//! 经验回放缓冲区

use super::BufferItem;
use rand::Rng;
use std::collections::VecDeque;

/// 泛型经验回放缓冲区。
///
/// 存储 `T: BufferItem` 的 FIFO 队列，容量满时淘汰最老元素。
/// `sample` 为**有放回**随机抽样（直接 `gen_range`，不建全长索引）。
///
/// # 设计边界
/// - `sample` = 按**存储单位** `T` 随机有放回抽样，**不是**训练采样器
/// - position 级等更细粒度的采样由上层 helper 负责
pub struct ReplayBuffer<T: BufferItem> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T: BufferItem> ReplayBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.min(10_000)),
            capacity,
        }
    }

    /// 压入一条数据，容量满时淘汰最老元素。
    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    /// 有放回随机抽样 `batch_size` 条（逐次 `gen_range`，不建全长索引）。
    ///
    /// - `batch_size == 0` → 返回空 Vec
    /// - `batch_size > len` → 仍返回 `batch_size` 条（有放回允许重复）
    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<T> {
        if batch_size == 0 || self.buffer.is_empty() {
            return Vec::new();
        }
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| self.buffer[rng.gen_range(0..len)].clone())
            .collect()
    }

    /// 有放回随机抽样，额外返回每个样本在 buffer 中的**存储下标**。
    ///
    /// 与 [`sample`](Self::sample) 的唯一区别是带回下标，使「完整 reanalyze 写回」
    /// 「PER priority 更新」等未来能力能定位回原条目（配合 [`update_at`](Self::update_at)）。
    ///
    /// v0.24 主路径仍是 **batch-time reanalyze**（改采样副本、不写回），本方法的下标用于
    /// 形状预留；完整回写式 reanalyze 推 v0.25。
    pub fn sample_indexed(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<(usize, T)> {
        if batch_size == 0 || self.buffer.is_empty() {
            return Vec::new();
        }
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..len);
                (idx, self.buffer[idx].clone())
            })
            .collect()
    }

    /// 按存储下标**原地回写**一个元素（完整 reanalyze 写回 / 原地刷新预留形状）。
    ///
    /// ⚠️ `ReplayBuffer` 是 FIFO：长时间运行后下标会随 `push` 淘汰而漂移。调用方须保证
    /// `idx` 仍指向期望条目（如「采样→写回」之间未发生淘汰）。越界则忽略（no-op）。
    /// v0.24 主路径不用本方法（batch-time reanalyze 只读副本）；完整 reanalyze（v0.25）启用。
    pub fn update_at(&mut self, idx: usize, item: T) {
        if idx < self.buffer.len() {
            self.buffer[idx] = item;
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}
