//! MyZero 各尾缀产出的分数报告（与 train / eval / run 一一对应）。

use std::path::PathBuf;

/// [`MyZeroBuilder::train`](super::builder::MyZeroBuilder::train) 产出。
#[derive(Debug, Clone, PartialEq)]
pub struct TrainReport {
    pub seed: u64,
    pub wall_secs: f32,
    /// 训末 latest 权重上的 greedy eval（与返回实例一致；磁盘 best 见 [`best_greedy`](Self::best_greedy)）
    pub final_greedy: f32,
    /// 训练内首次 periodic greedy ≥ solved 时的 env-step；未达标为 `None`。
    pub solved_at_steps: Option<u64>,
    pub solved_threshold: f32,
    /// 训练过程 periodic greedy 最高分
    pub best_greedy: f32,
    pub best_at_episode: Option<usize>,
    /// best 模型基名（无 `.otm` 后缀），如 `models/my_zero/CartPole-v1/seed_42/best`
    pub model_path: Option<PathBuf>,
}

/// [`MyZero::eval`](super::my_zero::MyZero::eval) 产出。
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    pub n_episodes: usize,
    pub mean_return: f32,
    pub episode_returns: Vec<f32>,
}

/// [`MyZero::run`](super::my_zero::MyZero::run) 产出。
#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    /// 请求局数：`None` = 无限循环（正常不会返回）。
    pub episodes_requested: Option<usize>,
    pub episodes_completed: usize,
    pub episode_returns: Vec<f32>,
    pub episode_lengths: Vec<usize>,
    /// 已完成局的 return 均值（无局时为 0）
    pub mean_return: f32,
}
