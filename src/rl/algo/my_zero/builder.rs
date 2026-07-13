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

use super::config::{ActionPlan, EvalSettings, MyZeroConfig, ObservationPlan, TrainSettings};
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

    fn ensure_component_compatibility(&self) -> Result<(), GraphError> {
        let c = &self.cfg.components;
        if c.gumbel && c.sampled {
            return Err(GraphError::InvalidOperation(
                "MyZero: Gumbel root 与 Sampled root Dirichlet 语义暂不兼容；\
                 请关闭 Sampled 或不要开启 Gumbel"
                    .into(),
            ));
        }
        if c.rosmo && c.reanalyze {
            return Err(GraphError::InvalidOperation(
                "MyZero: ROSMO（一步现算刷新）与 reanalyze（全树重搜写回）互斥；\
                 二者是同一 target 刷新职责的两个消融臂，请只开一个"
                    .into(),
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

    /// 配置原生图像预处理的矩形尺寸与帧历史。
    pub fn image_observation(mut self, height: usize, width: usize, history: usize) -> Self {
        self.cfg.env.observation = ObservationPlan::Image {
            height,
            width,
            history,
        };
        self
    }

    /// 固定长度 token observation；环境须直接返回 token IDs。
    pub fn token_observation(mut self, length: usize, vocab_size: usize, embed_dim: usize) -> Self {
        self.cfg.env.observation = ObservationPlan::Tokens {
            length,
            vocab_size,
            embed_dim,
            pad_id: 0,
        };
        self
    }

    /// 与 [`token_observation`](Self::token_observation) 相同，但显式指定 padding token。
    pub fn token_observation_with_padding(
        mut self,
        length: usize,
        vocab_size: usize,
        embed_dim: usize,
        pad_id: usize,
    ) -> Self {
        self.cfg.env.observation = ObservationPlan::Tokens {
            length,
            vocab_size,
            embed_dim,
            pad_id,
        };
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

    /// 覆盖 n-step bootstrap 步数（默认见 [`TrainSettings::td_steps`]）。
    pub fn td_steps(mut self, n: usize) -> Self {
        self.cfg.train.td_steps = n.max(1);
        self
    }

    /// 开启 completedQ 策略训练目标（默认关；CartPole recipe 未 promote，供 A/B 用）。
    pub fn completed_q_target(mut self, enabled: bool) -> Self {
        self.cfg.components.completed_q_target = enabled;
        self
    }

    /// 开启 ROSMO 式一步 target 刷新（reanalyze 复活阶梯一；默认关，供 A/B 消融）。
    ///
    /// 采样时现算 policy/value target + 优势过滤行为正则，不写回 buffer；
    /// 与 [`reanalyze`](super::component::Components::reanalyze) 互斥。见 [`super::rosmo`]。
    pub fn rosmo(mut self, enabled: bool) -> Self {
        self.cfg.components.rosmo = enabled;
        self
    }

    /// 开启 KL 自适应 lr（「用户不调 lr」去旋钮机制；默认关，供 A/B 消融）。
    ///
    /// 每局训练块前后测探针局面 policy KL 位移，自动调 lr 乘子（[0.1,10]）。
    /// 棋盘域 ⑨ 臂已验证无害；单智能体 promote 须过 CartPole「等效不劣」闸门。
    pub fn kl_adaptive_lr(mut self, enabled: bool) -> Self {
        self.cfg.train.kl_adaptive_lr = enabled;
        self
    }

    /// 开启 Gumbel MuZero 标准根搜索（Sequential Halving + Gumbel-Top-k）。
    pub fn gumbel(mut self, enabled: bool) -> Self {
        self.cfg.components.gumbel = enabled;
        if enabled {
            // Gumbel root 自带探索噪声；当前 Sampled root 会在采样前注入 Dirichlet，
            // 两者组合语义尚未定义，开启 Gumbel 时默认退出 Sampled 路径。
            self.cfg.components.sampled = false;
        }
        self
    }

    /// 论文标准 bundle：Gumbel-root + completedQ 训练 target。
    pub fn gumbel_standard(mut self) -> Self {
        self.cfg.components.gumbel = true;
        self.cfg.components.completed_q_target = true;
        self.cfg.components.sampled = false;
        self
    }

    /// 高级消融开关：覆盖 recipe 中的 Sampled MuZero 搜索路径。
    ///
    /// Gumbel root 与 Sampled root Dirichlet 组合暂未定义；若手动把二者同时打开，
    /// [`build`](Self::build) / [`train`](Self::train) 会显式报错。
    pub fn sampled(mut self, enabled: bool) -> Self {
        self.cfg.components.sampled = enabled;
        self
    }

    /// 启用 recurrent posterior（GRU 状态估计器，POMDP-lite）。
    ///
    /// 关闭时（默认）为无记忆 MDP 路径；开启后 self-play 和训练均走
    /// posterior burn-in + hidden state 管理。配合 [`obs_mask`](Self::obs_mask)
    /// 可验证"单帧失败、history posterior 成功"。
    pub fn recurrent_posterior(mut self, enabled: bool) -> Self {
        self.cfg.components.recurrent_posterior = enabled;
        self
    }

    /// 观测遮蔽（POMDP-lite 验证用）。
    ///
    /// `indices` 中的维度在 `ObsAdapter::reset/step` 后被置零。
    /// 例如 CartPole `&[1, 3]` 遮蔽速度维度。
    pub fn obs_mask(mut self, indices: &[usize]) -> Self {
        self.cfg.env.obs_mask = indices.to_vec();
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
        self.ensure_component_compatibility()?;
        Ok(self.cfg)
    }

    /// 完整训练 + 内置周期性 eval，返回训练后的 [`MyZero`]（**latest** 权重）。
    pub fn train(self) -> Result<MyZero, GraphError> {
        self.ensure_train_required()?;
        self.ensure_component_compatibility()?;
        train_all_seeds(self.cfg)
    }

    /// 物化空权重实例并从磁盘加载（若 `path.otm` 存在）。
    pub fn load_model_if_exists(self, path: impl AsRef<Path>) -> Result<MyZero, GraphError> {
        self.ensure_component_compatibility()?;
        MyZero::materialize_from_cfg(&self.cfg, self.cfg.eval.seed)?.load_model_if_exists(path)
    }
}
