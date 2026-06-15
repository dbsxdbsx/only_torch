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
/// - v0.22 `SelfPlayGame` 的 position 级采样由上层 helper 负责
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
