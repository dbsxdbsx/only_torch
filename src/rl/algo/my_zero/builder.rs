//! MyZero 链式配置。
//!
//! # 必填 vs 可选
//!
//! | 类别 | 字段 | 说明 |
//! |------|------|------|
//! | **必填** | [`MyZero::new`](super::my_zero::MyZero::new) 的 `env_id` | Gymnasium 环境 ID（内置算法配方） |
//! | **必填（仅 train）** | [`.solved`](MyZeroBuilder::solved) | greedy eval 达标线 |
//! | **必填（仅 train）** | [`.max_episodes`](MyZeroBuilder::max_episodes) | 训练局数上限 |
//! | **特殊动作时才写** | [`.discretize`](MyZeroBuilder::discretize) 等 | 默认 [`ActionPlan::Auto`] |
//! | **非默认时常写** | [`.reward_scale`](MyZeroBuilder::reward_scale) | 如 Pendulum 的 `0.1` |
//! | **eval 创新高时落盘** | [`.save_model_when_eval(path)`](MyZeroBuilder::save_model_when_eval) | 默认**不写**；path 为 `.otm` 基名（不含后缀） |
//! | **推理** | [`.load_model_if_exists`](MyZeroBuilder::load_model_if_exists) | 必填 path（不含 `.otm` 后缀） |
//!
//! **权重语义**：`.train()` 返回 **latest** 训末权重；eval 前若要用磁盘 best，须显式
//! [`.load_model_if_exists(path)`](super::my_zero::MyZero::load_model_if_exists)（`path` 见 [`TrainReport::model_path`](super::report::TrainReport::model_path)）。

use super::config::{ActionPlan, EvalSettings, MyZeroConfig, TrainSettings};
use super::my_zero::MyZero;
use super::runner::train_all_seeds;
use crate::nn::GraphError;
use std::path::Path;

/// 链式配置；尾缀 [`train`](Self::train) / [`load_model_if_exists`](Self::load_model_if_exists) 物化 [`MyZero`]。
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

    /// 开启 completedQ 策略训练目标（默认关；CartPole recipe 未 promote，供 A/B 用）。
    pub fn completed_q_target(mut self, enabled: bool) -> Self {
        self.cfg.components.completed_q_target = enabled;
        self
    }

    /// 开启 Gumbel MuZero 标准根搜索（Sequential Halving + Gumbel-Top-k）。
    pub fn gumbel(mut self, enabled: bool) -> Self {
        self.cfg.components.gumbel = enabled;
        self
    }

    /// 论文标准 bundle：Gumbel-root + completedQ 训练 target。
    pub fn gumbel_standard(mut self) -> Self {
        self.cfg.components.gumbel = true;
        self.cfg.components.completed_q_target = true;
        self
    }

    pub fn train_batch_size(mut self, n: usize) -> Self {
        self.cfg.train.train_batch_size = n.max(1);
        self
    }

    pub fn train_settings(mut self, train: TrainSettings) -> Self {
        self.cfg.train = train;
        self
    }

    // ---- eval ----

    /// greedy eval 达标门槛（**train 必填**）。
    pub fn solved(mut self, threshold: f32) -> Self {
        self.cfg.eval.solved = threshold;
        self.solved_set = true;
        self
    }

    /// 训练局数上限（**train 必填**；`.smoke()` 时运行期仍强制 3 局）。
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

    /// 多 seed 回归（benchmark 用；默认 1）。
    pub fn seeds(mut self, n: u64) -> Self {
        self.cfg.eval.seed_runs = n.max(1);
        self
    }

    pub fn eval_every(mut self, n: usize) -> Self {
        self.cfg.eval.eval_every = n;
        self
    }

    /// 管线自检（3 局 self-play + 1 次训练，不验收敛；通常由 example 在 `SMOKE=1` 时调用）。
    pub fn smoke(mut self) -> Self {
        self.cfg.eval.smoke = true;
        self
    }

    /// dynamics 诊断（对比 model 想象 vs 真实 reward/value）。
    pub fn diagnose(mut self) -> Self {
        self.cfg.eval.diagnose = true;
        self
    }

    /// periodic greedy eval 分数创新高时写入 `{path}.otm`。
    ///
    /// `path` 为完整基名（含目录与文件名，**不含** `.otm` 后缀），无默认路径。
    /// 多 seed（`.seeds(n)`，`n>1`）时在 `path` 的父目录下自动插入 `seed_{seed}/` 子目录。
    pub fn save_model_when_eval(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cfg.eval.checkpoint.enabled = true;
        self.cfg.eval.checkpoint.best_base = Some(path.into());
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

    /// 完整训练 + 内置周期性 eval，返回训练后的 [`MyZero`]（**latest** 权重）。
    pub fn train(self) -> Result<MyZero, GraphError> {
        self.ensure_train_required()?;
        train_all_seeds(self.cfg)
    }

    /// 物化空权重实例并从磁盘加载（若 `path.otm` 存在）。
    pub fn load_model_if_exists(self, path: impl AsRef<Path>) -> Result<MyZero, GraphError> {
        MyZero::materialize_from_cfg(&self.cfg, self.cfg.eval.seed)?.load_model_if_exists(path)
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
    fn cartpole_recipe_has_consistency_and_reconstruction() {
        let cfg = MyZero::new("CartPole-v1")
            .solved(475.0)
            .max_episodes(2000)
            .build()
            .unwrap();
        assert!(cfg.components.consistency);
        assert!(cfg.components.reconstruction);
        assert!(!cfg.components.reanalyze);
        assert!(!cfg.components.completed_q_target);
    }

    #[test]
    fn save_model_when_eval_sets_path() {
        let path = std::path::PathBuf::from("models/my_zero/CartPole-v1/seed_42/best");
        let cfg = MyZero::new("CartPole-v1")
            .solved(475.0)
            .max_episodes(2000)
            .save_model_when_eval(&path)
            .build()
            .unwrap();
        assert!(cfg.eval.checkpoint.enabled);
        assert_eq!(cfg.eval.checkpoint.best_base.as_ref(), Some(&path));
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
