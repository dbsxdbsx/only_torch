//! MyZero 统一配置（5 层，按关注点分组）
//!
//! 一个 config 适配所有环境。分层原则：
//! - [`EnvSettings`]：env I/O 适配（env_id / reward_scale / 动作方案）——**事实**让库问 env，
//!   **选择**才让用户填（动作的离散/连续/范围是 env 事实，库自动推断，见 [`crate::rl::algo::my_zero::action`]）。
//! - [`ModelSettings`]：网络容量（latent_dim ...）。
//! - [`TrainSettings`]：训练超参（gamma/lr/k_unroll/sims/buffer ...），MyZero 自有。
//! - [`Components`]：内部组件开关（由 [`recipe`](super::recipe) 按 env 注入，用户 API 不暴露）。
//! - [`EvalSettings`]：评测 / 跑法（solved / seed / max_episodes / …）。
//!
//! 判据（字段归层）：**改了它训练出的 agent 会不会变？** 会变→`train`；不会变（只改这次怎么跑/怎么量）→`eval`。

use super::component::Components;

/// 默认随机种子（训练 + eval + run + 环境 reset 派生自此）。
pub const DEFAULT_SEED: u64 = 42;

/// 与 [`DEFAULT_SEED`] 相同；保留别名以免外部引用断裂。
pub const DEFAULT_TRAIN_SEED: u64 = DEFAULT_SEED;
pub const DEFAULT_EVAL_SEED: u64 = DEFAULT_SEED;
pub const DEFAULT_ROLLOUT_SEED: u64 = DEFAULT_SEED;

/// greedy eval / run 第 `i` 局的 `env.reset` 种子。
#[inline]
pub(crate) fn greedy_episode_seed(base: u64, episode_index: u64) -> u64 {
    base.wrapping_add(episode_index)
}

/// self-play 训练第 `ep` 局（从 0 计）的 `env.reset` 种子；与 greedy 序列错开命名空间。
#[inline]
pub(crate) fn self_play_episode_seed(base: u64, ep: usize) -> u64 {
    base.wrapping_add(1_000_000 + ep as u64)
}

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
pub struct EnvSettings {
    /// Gymnasium 环境 ID（如 `"CartPole-v1"` / `"Pendulum-v1"`）
    pub env_id: &'static str,
    /// reward 缩放（1.0 = 原样；Pendulum 用 0.1 使累计 value 落入 categorical support 域）
    pub reward_scale: f32,
    /// 动作方案（默认 `Auto`：离散 env 自动；连续/混合 env 需指定）
    pub action: ActionPlan,
}

impl Default for EnvSettings {
    /// 库内默认（测试 / `MyZeroConfig::default`）；公开 API 请用 [`crate::rl::algo::my_zero::MyZero::new`]。
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
pub struct ModelSettings {
    /// learned latent（隐状态）维度。低维 env 64 足够；图像 obs 需更大/带结构。
    pub latent_dim: usize,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self { latent_dim: 64 }
    }
}

/// 训练超参层（MyZero 自有）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainSettings {
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
    /// 每次 optimizer step 采样的训练 position 数（从 buffer 抽局 + 局内随机起点，对齐 MuZero minibatch）
    pub train_batch_size: usize,
    /// 每个 episode 结束后的训练次数
    pub trains_per_episode: usize,
    /// replay buffer 容量（按整局计）
    pub buffer_capacity: usize,
    /// 开始训练前需累积的局数
    pub start_training_after: usize,
}

impl Default for TrainSettings {
    /// CartPole 级（低维向量 obs / CPU only）合理默认；换环境覆盖字段。
    fn default() -> Self {
        Self {
            gamma: 0.997,
            k_unroll: 5,
            td_steps: 50,
            num_simulations: 50,
            lr: 0.02,
            train_batch_size: 8,
            trains_per_episode: 8,
            buffer_capacity: 1000,
            start_training_after: 10,
        }
    }
}

/// 训练期 best 模型落盘（须显式指定路径；仅在 periodic greedy eval 创新高时写入）
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointSettings {
    /// 是否落盘（默认关；链式 [`.save_model_when_eval(path)`](super::builder::MyZeroBuilder::save_model_when_eval)）
    pub enabled: bool,
    /// best `.otm` 基名（不含后缀）；须显式设定，无默认路径
    pub best_base: Option<std::path::PathBuf>,
    /// greedy 均值至少提升这么多才覆盖 best（默认 0 = 创新高即存）
    pub min_delta: f32,
    /// 训练结束是否额外写 `last`（默认否）
    pub save_last: bool,
}

impl Default for CheckpointSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            best_base: None,
            min_delta: 0.0,
            save_last: false,
        }
    }
}

/// 评测 / 跑法层（改这里**不改算法**，只改这次怎么驱动/判定）
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSettings {
    /// greedy(temp=0) eval 达标门槛
    pub solved: f32,
    /// 随机种子：权重初始化、self-play / MCTS、greedy eval / run、环境 `reset` 均由此派生
    pub seed: u64,
    /// 多 seed 回归跑法重复次数（[`.seeds(n)`](super::builder::MyZeroBuilder::seeds)）
    pub(crate) seed_runs: u64,
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
    /// best-only 模型落盘
    pub checkpoint: CheckpointSettings,
}

impl Default for EvalSettings {
    fn default() -> Self {
        Self {
            solved: 475.0,
            seed: DEFAULT_SEED,
            seed_runs: 1,
            max_episodes: 2000,
            eval_every: 25,
            eval_episodes: 10,
            smoke: false,
            diagnose: false,
            checkpoint: CheckpointSettings::default(),
        }
    }
}

/// MyZero 统一配置（5 层）
#[derive(Debug, Clone, PartialEq)]
pub struct MyZeroConfig {
    /// env I/O 适配
    pub env: EnvSettings,
    /// 网络容量
    pub model: ModelSettings,
    /// 训练超参
    pub train: TrainSettings,
    /// 消融组件（内部；[`MyZero::new`](super::my_zero::MyZero::new) 按 env 自动注入）
    pub(crate) components: Components,
    /// 评测 / 跑法
    pub eval: EvalSettings,
}

impl Default for MyZeroConfig {
    /// CartPole-v1 级 CPU 友好起点（组件全关 = base）。
    fn default() -> Self {
        Self {
            env: EnvSettings::default(),
            model: ModelSettings::default(),
            train: TrainSettings::default(),
            components: Components::default(),
            eval: EvalSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_base() {
        let cfg = MyZeroConfig::default();
        assert_eq!(cfg.components, Components::base(), "默认组件全关 = base");
        assert_eq!(cfg.train, TrainSettings::default());
        assert_eq!(cfg.env.env_id, "CartPole-v1");
        assert!((cfg.env.reward_scale - 1.0).abs() < 1e-6);
        assert_eq!(cfg.env.action, ActionPlan::Auto);
        assert_eq!(cfg.model.latent_dim, 64);
        assert!(!cfg.eval.checkpoint.enabled, "默认不落盘");
    }

    #[test]
    fn train_default_is_cartpole_friendly() {
        let t = TrainSettings::default();
        assert_eq!(t.num_simulations, 50);
        assert_eq!(t.k_unroll, 5);
        assert!((t.gamma - 0.997).abs() < 1e-6);
        assert_eq!(t.train_batch_size, 8);
    }

    #[test]
    fn eval_default_seed() {
        let e = EvalSettings::default();
        assert_eq!(e.seed, DEFAULT_SEED);
        assert_eq!(e.seed_runs, 1);
    }

    #[test]
    fn component_toggle() {
        let cfg = MyZeroConfig {
            components: Components {
                consistency: true,
                ..Components::default()
            },
            ..MyZeroConfig::default()
        };
        assert!(cfg.components.consistency);
        assert!(!cfg.components.value_prefix, "只开 consistency，其余不变");
    }

    #[test]
    fn new_cartpole_applies_recipe() {
        use super::super::my_zero::MyZero;
        let cfg = MyZero::new("CartPole-v1")
            .solved(475.0)
            .max_episodes(100)
            .build()
            .unwrap();
        assert!(cfg.components.consistency);
        assert!(!cfg.components.reanalyze);
        assert!(!cfg.components.completed_q_target);
    }
}
