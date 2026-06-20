//! MyZero 链式配置。
//!
//! # 必填 vs 可选
//!
//! | 类别 | 字段 | 说明 |
//! |------|------|------|
//! | **必填** | [`MyZero::new`](super::my_zero::MyZero::new) 的 `env_id` | Gymnasium 环境 ID |
//! | **必填（仅 train）** | [`.solved`](MyZeroBuilder::solved) | greedy eval 达标线 |
//! | **必填（仅 train）** | [`.max_episodes`](MyZeroBuilder::max_episodes) | 训练局数上限 |
//! | **特殊动作时才写** | [`.discretize`](MyZeroBuilder::discretize) 等 | 默认 [`ActionPlan::Auto`] |
//! | **非默认时常写** | [`.reward_scale`](MyZeroBuilder::reward_scale) | 如 Pendulum 的 `0.1` |
//! | **推理** | [`.load_model`](MyZeroBuilder::load_model) | 加载 `path.otm`（path 不含后缀） |
//!
//! 训练期 best 模型默认写入 `models/my_zero/{env_id}/seed_{seed}/best.otm`（`SMOKE` 跳过；`MODEL_DIR` 可覆盖根目录）。
//! **同一实例**训后直接 `eval` / `run` 使用 **latest** 训末权重；要用 best 须显式 [`load_model`](MyZeroBuilder::load_model)。

use super::component::Components;
use super::config::{ActionPlan, EvalSettings, MyZeroConfig, TrainSettings};
use super::my_zero::MyZero;
use super::runner::train_all_seeds;
use crate::nn::GraphError;
use std::path::Path;

/// 链式配置；尾缀 [`train`](Self::train) / [`load_model`](Self::load_model) 物化 [`MyZero`]。
#[derive(Debug, Clone)]
pub struct MyZeroBuilder {
    pub(crate) cfg: MyZeroConfig,
    pub(crate) solved_set: bool,
    pub(crate) max_episodes_set: bool,
}

impl MyZeroBuilder {
    fn ensure_train_required(&self) -> Result<(), GraphError> {
        if !self.solved_set {
            return Err(GraphError::InvalidOperation(
                "MyZero: 必须调用 .solved(门槛) 指定 greedy eval 达标线".into(),
            ));
        }
        if !self.max_episodes_set {
            return Err(GraphError::InvalidOperation(
                "MyZero: 必须调用 .max_episodes(n) 指定训练局数上限".into(),
            ));
        }
        Ok(())
    }

    // ---- env ----

    pub fn reward_scale(mut self, v: f32) -> Self {
        self.cfg.env.reward_scale = v;
        self
    }

    /// 覆盖默认动作方案（默认 [`ActionPlan::Auto`]，一般不必调用）。
    pub fn action(mut self, plan: ActionPlan) -> Self {
        self.cfg.env.action = plan;
        self
    }

    /// 连续动作 env：将力矩/控制量均匀离散为 `buckets` 档 MCTS 候选（**须显式声明**）。
    pub fn discretize(mut self, buckets: usize) -> Self {
        self.cfg.env.action = ActionPlan::Discretize { buckets };
        self
    }

    // ---- model ----

    pub fn latent_dim(mut self, dim: usize) -> Self {
        self.cfg.model.latent_dim = dim;
        self
    }

    // ---- train（可选）----

    pub fn gamma(mut self, v: f32) -> Self {
        self.cfg.train.gamma = v;
        self
    }

    pub fn lr(mut self, v: f32) -> Self {
        self.cfg.train.lr = v;
        self
    }

    pub fn num_simulations(mut self, n: u32) -> Self {
        self.cfg.train.num_simulations = n;
        self
    }

    pub fn reanalyze_fraction(mut self, f: f32) -> Self {
        self.cfg.train.reanalyze_fraction = f.clamp(0.0, 1.0);
        self
    }

    pub fn train_settings(mut self, train: TrainSettings) -> Self {
        self.cfg.train = train;
        self
    }

    // ---- components（可选）----

    pub fn consistency(mut self) -> Self {
        self.cfg.components.consistency = true;
        self
    }

    pub fn value_prefix(mut self) -> Self {
        self.cfg.components.value_prefix = true;
        self
    }

    pub fn target_net(mut self) -> Self {
        self.cfg.components.target_net = true;
        self
    }

    pub fn completed_q(mut self) -> Self {
        self.cfg.components.completed_q_target = true;
        self
    }

    pub fn sve(mut self, weight: f32) -> Self {
        self.cfg.components.sve_weight = weight;
        self
    }

    pub fn gumbel(mut self) -> Self {
        self.cfg.components.gumbel = true;
        self
    }

    pub fn components(mut self, c: Components) -> Self {
        self.cfg.components = c;
        self
    }

    // ---- eval ----

    /// greedy eval 达标门槛（**train 必填**）。
    pub fn solved(mut self, threshold: f32) -> Self {
        self.cfg.eval.solved = threshold;
        self.solved_set = true;
        self
    }

    /// 训练局数上限（**train 必填**；`SMOKE=1` 时运行期仍强制 3 局）。
    pub fn max_episodes(mut self, n: usize) -> Self {
        self.cfg.eval.max_episodes = n;
        self.max_episodes_set = true;
        self
    }

    /// 随机种子（训练 + eval + run + 环境 reset 派生；默认 42）。
    pub fn seed(mut self, seed: u64) -> Self {
        self.cfg.eval.seed = seed;
        self
    }

    pub fn eval_every(mut self, n: usize) -> Self {
        self.cfg.eval.eval_every = n;
        self
    }

    pub fn eval_settings(mut self, eval: EvalSettings) -> Self {
        self.solved_set = true;
        self.max_episodes_set = true;
        self.cfg.eval = eval;
        self
    }

    /// 仅构建配置（测试 / 高级用法；须已填 train 契约项）。
    pub fn build(self) -> Result<MyZeroConfig, GraphError> {
        self.ensure_train_required()?;
        Ok(self.cfg)
    }

    /// 完整训练 + 内置周期性 eval，返回训练后的 [`MyZero`]。
    pub fn train(self) -> Result<MyZero, GraphError> {
        self.ensure_train_required()?;
        train_all_seeds(self.cfg)
    }

    /// 从 `.otm` 加载模型（须先 `new` 声明 env / action 契约；不要求 `solved` / `max_episodes`）。
    pub fn load_model(self, path: impl AsRef<Path>) -> Result<MyZero, GraphError> {
        MyZero::materialize_from_cfg(&self.cfg, self.cfg.eval.seed)?.load_model(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_env_defaults_to_auto_action() {
        let cfg = MyZero::new("CartPole-v1")
            .solved(475.0)
            .max_episodes(2000)
            .build()
            .unwrap();
        assert_eq!(cfg.env.action, ActionPlan::Auto);
    }

    #[test]
    fn seed_sets_config() {
        let cfg = MyZero::new("CartPole-v1")
            .solved(475.0)
            .max_episodes(100)
            .seed(99)
            .build()
            .unwrap();
        assert_eq!(cfg.eval.seed, 99);
    }

    #[test]
    fn missing_solved_is_error_on_train() {
        let r = MyZero::new("CartPole-v1").max_episodes(2000).train();
        assert!(matches!(r, Err(GraphError::InvalidOperation(_))));
    }

    #[test]
    fn missing_max_episodes_is_error_on_train() {
        let r = MyZero::new("CartPole-v1").solved(475.0).train();
        assert!(matches!(r, Err(GraphError::InvalidOperation(_))));
    }
}
