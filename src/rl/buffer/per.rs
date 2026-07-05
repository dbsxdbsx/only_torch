//! 位置级优先经验回放（PER）伴生采样器（Schaul et al. 2016 · arXiv:1511.05952；
//! MuZero 口径优先级 `p = |ν − z|`，Schrittwieser et al. 2020 附录 G）。
//!
//! # 设计
//!
//! 不替换 [`ReplayBuffer`](super::ReplayBuffer)，而是做它的**伴生结构**：FIFO 语义
//! 逐槽位镜像（`push_game` 满容淘汰最老局），槽位 `slot` 恒对应 buffer 同下标的局。
//! 采样返回 `(slot, pos)` 位置对，按 `p^α` 比例加权（前缀和 + 二分，O(N) 构建 +
//! O(log N)/draw；万级位置量下可忽略）。
//!
//! # 口径（首臂预注册，2026-07-05）
//!
//! - **优先级在入库时一次性计算**（`p = |search 根价值 − MC 回报| + ε`），不做
//!   采样后在线刷新——网络自评刷新是 ③ ROSMO 臂已证的弱网噪声源，staleness
//!   风险留档、正信号后再单变量加刷新；
//! - **无 IS 重要性修正**（等效 β=0）：逐样本 loss 加权需动计算图，且「改写消费
//!   分布」正是本组件在棋盘域的干预目的（课程的零领域知识替代物）——偏差即疗法。
//!   若正信号伴随不稳，IS 加权为后续单变量臂。

use rand::Rng;
use std::collections::VecDeque;

/// 优先级下限（防零概率位置永不被采样；也兜底「全零优先级 → 均匀采样」退化）。
pub const PER_EPS: f32 = 1e-2;

/// 位置级 PER 采样器（`ReplayBuffer<SelfPlayGame>` 的伴生结构）。
#[derive(Debug, Clone)]
pub struct PerPriorities {
    /// 逐局位置优先级（已含 `^α` 变换后的采样权重），槽位镜像 buffer FIFO。
    per_game: VecDeque<Vec<f32>>,
    capacity: usize,
    alpha: f32,
}

impl PerPriorities {
    /// `capacity` 须与伴生 `ReplayBuffer` 一致；`alpha` 为优先化强度
    /// （0 = 均匀，1 = 全比例；Schaul 默认 0.6）。
    pub fn new(capacity: usize, alpha: f32) -> Self {
        Self {
            per_game: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
        }
    }

    /// 压入一局的原始优先级（与伴生 buffer 的 `push` 同步调用；满容淘汰最老局）。
    ///
    /// 内部存 `(p + ε)^α`；空局（<2 步，训练路径会跳过）传空 Vec = 零采样质量。
    pub fn push_game(&mut self, raw_priorities: Vec<f32>) {
        if self.per_game.len() >= self.capacity {
            self.per_game.pop_front();
        }
        let weighted = raw_priorities
            .into_iter()
            .map(|p| (p.max(0.0) + PER_EPS).powf(self.alpha))
            .collect();
        self.per_game.push_back(weighted);
    }

    /// 有放回按权采样 `batch_size` 个 `(slot, pos)` 位置对。
    ///
    /// 总质量为 0（无局或全空局）时返回空 Vec（调用方按无样本处理）。
    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<(usize, usize)> {
        if batch_size == 0 || self.per_game.is_empty() {
            return Vec::new();
        }
        // 前缀和（逐位置展平；万级规模每次重建可忽略）
        let mut cumsum = Vec::new();
        let mut index = Vec::new();
        let mut acc = 0.0f64;
        for (slot, ps) in self.per_game.iter().enumerate() {
            for (pos, &w) in ps.iter().enumerate() {
                acc += f64::from(w);
                cumsum.push(acc);
                index.push((slot, pos));
            }
        }
        if acc <= 0.0 {
            return Vec::new();
        }
        (0..batch_size)
            .map(|_| {
                let r = rng.gen_range(0.0..acc);
                let i = cumsum.partition_point(|&c| c <= r).min(index.len() - 1);
                index[i]
            })
            .collect()
    }

    /// 当前镜像的局数（应与伴生 buffer `len()` 一致）。
    pub fn len_games(&self) -> usize {
        self.per_game.len()
    }
}
