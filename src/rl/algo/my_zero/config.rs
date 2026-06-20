//! MyZero 统一配置（5 层，按关注点分组）
//!
//! 一个 config 适配所有环境。分层原则：
//! - [`EnvConfig`]：env I/O 适配（env_id / reward_scale / 动作方案）——**事实**让库问 env，
//!   **选择**才让用户填（动作的离散/连续/范围是 env 事实，库自动推断，见 [`crate::rl::algo::my_zero::action`]）。
//! - [`ModelConfig`]：网络容量（latent_dim ...）。
//! - [`TrainConfig`]：训练超参（gamma/lr/k_unroll/sims/buffer ...），MyZero 自有。
//! - [`ComponentConfig`]：消融组件开关。
//! - [`RunConfig`]：评测 / 跑法（solved/seeds/max_episodes/eval_every/smoke/diagnose）。
//!
//! 判据（字段归层）：**改了它训练出的 agent 会不会变？** 会变→`train`；不会变（只改这次怎么跑/怎么量）→`run`。

use super::component::ComponentConfig;

/// 连续 / 混合动作的处理方案（仅当 env 非纯离散时生效；纯离散 env 忽略）。
///
/// 动作"是离散还是连续、几档、范围"是 **env 事实**，库自动推断；
/// "连续怎么近似"是 **算法选择**，env 不知道，故由此枚举表达。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionPlan {
    /// 离散 env：库按 env 动作数自动枚举（`Discrete(0..n)`）。
    Auto,
    /// 连续 / 混合：每个连续维均匀离散成 `buckets` 档。
    Discretize { buckets: usize },
    // 未来：Sampled { k } / Gumbel —— 新增 ActionSampler 实现即可，main.rs 不动。
}

/// env I/O 适配层
#[derive(Debug, Clone, PartialEq)]
pub struct EnvConfig {
    /// Gymnasium 环境 ID（如 `"CartPole-v1"` / `"Pendulum-v1"`）
    pub env_id: &'static str,
    /// reward 缩放（1.0 = 原样；Pendulum 用 0.1 使累计 value 落入 categorical support 域）
    pub reward_scale: f32,
    /// 动作方案（默认 `Auto`：离散 env 自动；连续/混合 env 需指定）
    pub action: ActionPlan,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            env_id: "CartPole-v1",
            reward_scale: 1.0,
            action: ActionPlan::Auto,
        }
    }
}

/// 网络容量层
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    /// learned latent（隐状态）维度。低维 env 64 足够；图像 obs 需更大/带结构。
    pub latent_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self { latent_dim: 64 }
    }
}

/// 训练超参层（MyZero 自有）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    /// 折扣因子 γ
    pub gamma: f32,
    /// K 步 unroll 展开步数
    pub k_unroll: usize,
    /// n-step bootstrap 的步数（value target）
    pub td_steps: usize,
    /// 每步 MCTS 模拟数（按环境调：棋类 ~800、向量/Atari ~50）
    pub num_simulations: u32,
    /// 学习率
    pub lr: f32,
    /// 每次训练采样的整局数
    pub batch_games: usize,
    /// 每个 episode 结束后的训练次数
    pub trains_per_episode: usize,
    /// replay buffer 容量（按整局计）
    pub buffer_capacity: usize,
    /// 开始训练前需累积的局数
    pub start_training_after: usize,
    /// reanalyze 比例 ∈ [0,1]（0.0 = 关闭，CPU 友好）
    pub reanalyze_fraction: f32,
}

impl Default for TrainConfig {
    /// CartPole 级（低维向量 obs / CPU only）合理默认；换环境覆盖字段。
    fn default() -> Self {
        Self {
            gamma: 0.997,
            k_unroll: 5,
            td_steps: 50,
            num_simulations: 50,
            lr: 0.02,
            batch_games: 8,
            trains_per_episode: 8,
            buffer_capacity: 1000,
            start_training_after: 10,
            reanalyze_fraction: 0.0,
        }
    }
}

/// 评测 / 跑法层（改这里**不改算法**，只改这次怎么驱动/判定）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunConfig {
    /// greedy(temp=0) eval 达标门槛
    pub solved: f32,
    /// 多 seed 数（取中位数；从 base_seed=42 起递增）
    pub seeds: u64,
    /// 最大训练局数（smoke 时强制为 3）
    pub max_episodes: usize,
    /// 每 N 局 greedy eval 一次
    pub eval_every: usize,
    /// 每次 greedy eval 的局数
    pub eval_episodes: usize,
    /// 管线自检（3 局 self-play + 1 次训练，不验收敛）
    pub smoke: bool,
    /// dynamics 诊断（对比 model 想象 vs 真实 reward/value）
    pub diagnose: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            solved: 475.0,
            seeds: 1,
            max_episodes: 2000,
            eval_every: 25,
            eval_episodes: 10,
            smoke: false,
            diagnose: false,
        }
    }
}

/// MyZero 统一配置（5 层）
#[derive(Debug, Clone, PartialEq)]
pub struct MyZeroConfig {
    /// env I/O 适配
    pub env_config: EnvConfig,
    /// 网络容量
    pub model_config: ModelConfig,
    /// 训练超参
    pub train_config: TrainConfig,
    /// 消融组件
    pub component_config: ComponentConfig,
    /// 评测 / 跑法
    pub run_config: RunConfig,
}

impl Default for MyZeroConfig {
    /// CartPole-v1 级 CPU 友好起点（组件全关 = base）。
    fn default() -> Self {
        Self {
            env_config: EnvConfig::default(),
            model_config: ModelConfig::default(),
            train_config: TrainConfig::default(),
            component_config: ComponentConfig::default(),
            run_config: RunConfig::default(),
        }
    }
}

impl MyZeroConfig {
    /// 从环境变量统一覆盖（消融 / 调参 / 跑法旋钮集中在此一处）。
    ///
    /// 支持：
    /// - 组件：`CONSISTENCY` / `VALUE_PREFIX` / `TARGET_NET` / `SVE=<f32>` / `CQ` / `CQ_SCALE=<f32>` / `CQ_VISIT=<f32>`
    /// - 训练：`SIMS=<u32>` / `REANALYZE=<f32>` / `GAMMA=<f32>` / `LR=<f32>`
    /// - 跑法：`MAX_EP=<usize>` / `SEEDS=<u64>` / `SMOKE` / `DIAG` / `SOLVED=<f32>`
    /// - 动作 / env：`NUM_ACTIONS=<usize>`（仅 `Discretize`）/ `RSCALE=<f32>`
    pub fn apply_env_overrides(&mut self) {
        use std::env::var;

        // ---- 组件 ----
        if var("CONSISTENCY").is_ok() {
            self.component_config.consistency = true;
        }
        if var("VALUE_PREFIX").is_ok() {
            self.component_config.value_prefix = true;
        }
        if var("TARGET_NET").is_ok() {
            self.component_config.target_net = true;
        }
        if let Ok(v) = var("SVE")
            && let Ok(w) = v.parse::<f32>()
        {
            self.component_config.sve_weight = w;
        }
        if var("CQ").is_ok() {
            self.component_config.completed_q_target = true;
        }
        if let Ok(v) = var("CQ_SCALE")
            && let Ok(s) = v.parse::<f32>()
        {
            self.component_config.cq_c_scale = s;
        }
        if let Ok(v) = var("CQ_VISIT")
            && let Ok(s) = v.parse::<f32>()
        {
            self.component_config.cq_c_visit = s;
        }

        // ---- 训练 ----
        if let Ok(v) = var("SIMS")
            && let Ok(n) = v.parse::<u32>()
        {
            self.train_config.num_simulations = n;
        }
        if let Ok(v) = var("REANALYZE")
            && let Ok(f) = v.parse::<f32>()
        {
            self.train_config.reanalyze_fraction = f.clamp(0.0, 1.0);
        }
        if let Ok(v) = var("GAMMA")
            && let Ok(g) = v.parse::<f32>()
        {
            self.train_config.gamma = g;
        }
        if let Ok(v) = var("LR")
            && let Ok(lr) = v.parse::<f32>()
            && lr.is_finite()
            && lr > 0.0
        {
            self.train_config.lr = lr;
        }

        // ---- 跑法 ----
        if let Ok(v) = var("MAX_EP")
            && let Ok(n) = v.parse::<usize>()
        {
            self.run_config.max_episodes = n;
        }
        if let Ok(v) = var("SEEDS")
            && let Ok(n) = v.parse::<u64>()
        {
            self.run_config.seeds = n.max(1);
        }
        if var("SMOKE").is_ok() {
            self.run_config.smoke = true;
        }
        if var("DIAG").is_ok() {
            self.run_config.diagnose = true;
        }
        if let Ok(v) = var("SOLVED")
            && let Ok(s) = v.parse::<f32>()
        {
            self.run_config.solved = s;
        }

        // ---- 动作 / env ----
        if let Ok(v) = var("NUM_ACTIONS")
            && let Ok(n) = v.parse::<usize>()
            && n >= 1
            && let ActionPlan::Discretize { buckets } = &mut self.env_config.action
        {
            *buckets = n;
        }
        if let Ok(v) = var("RSCALE")
            && let Ok(s) = v.parse::<f32>()
            && s.is_finite()
            && s > 0.0
        {
            self.env_config.reward_scale = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_base() {
        let cfg = MyZeroConfig::default();
        assert_eq!(cfg.component_config, ComponentConfig::base(), "默认组件全关 = base");
        assert_eq!(cfg.train_config, TrainConfig::default());
        assert_eq!(cfg.env_config.env_id, "CartPole-v1");
        assert!((cfg.env_config.reward_scale - 1.0).abs() < 1e-6);
        assert_eq!(cfg.env_config.action, ActionPlan::Auto);
        assert_eq!(cfg.model_config.latent_dim, 64);
    }

    #[test]
    fn train_default_is_cartpole_friendly() {
        let t = TrainConfig::default();
        assert_eq!(t.num_simulations, 50);
        assert_eq!(t.k_unroll, 5);
        assert!((t.gamma - 0.997).abs() < 1e-6);
        assert_eq!(t.reanalyze_fraction, 0.0);
    }

    #[test]
    fn single_component_override() {
        let cfg = MyZeroConfig {
            component_config: ComponentConfig {
                consistency: true,
                ..ComponentConfig::default()
            },
            ..MyZeroConfig::default()
        };
        assert!(cfg.component_config.consistency);
        assert!(!cfg.component_config.value_prefix, "只开 consistency，其余不变");
    }
}
