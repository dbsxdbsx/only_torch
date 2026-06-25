//! MCTS 核心数据类型定义

/// 动作载荷：支持离散、连续及混合动作空间
#[derive(Debug, Clone, PartialEq)]
pub enum ActionPayload {
    Discrete(usize),
    Continuous(Vec<f32>),
    Hybrid {
        discrete: usize,
        continuous: Vec<f32>,
    },
}

/// 搜索 / 训练目标使用的稳定动作身份。
///
/// `ActionPayload` 描述如何执行动作，`ActionId` 描述它在策略 head / target 向量里的槽位。
/// 对离散动作与当前 Pendulum 离散桶，二者数值相同；后续连续 / 混合动作可保持 payload
/// 任意复杂，但仍用稳定 id 做蒸馏与回放。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub usize);

impl ActionId {
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for ActionId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

/// 单个候选动作及其策略先验。
#[derive(Debug, Clone, PartialEq)]
pub struct ActionCandidate {
    pub id: ActionId,
    pub payload: ActionPayload,
    /// 当前候选进入树搜索时使用的 prior。
    ///
    /// 普通 MuZero 中等于网络策略 prior；Sampled MuZero 中可被修正为 `π̂_β`。
    pub policy_prior: f32,
    /// proposal β；`None` 表示与 `policy_prior` 同源。
    pub proposal_prior: Option<f32>,
}

impl ActionCandidate {
    pub fn new(id: ActionId, payload: ActionPayload, policy_prior: f32) -> Self {
        Self {
            id,
            payload,
            policy_prior,
            proposal_prior: None,
        }
    }

    pub fn with_policy_prior(mut self, policy_prior: f32) -> Self {
        self.policy_prior = policy_prior;
        self
    }
}

/// 某节点可展开的候选动作集合。
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSet {
    pub candidates: Vec<ActionCandidate>,
}

impl CandidateSet {
    pub fn empty() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn from_actions_and_priors(actions: Vec<ActionPayload>, priors: Vec<f32>) -> Self {
        let n = actions.len();
        if n == 0 {
            return Self::empty();
        }
        let aligned = if priors.len() == n {
            priors
        } else {
            vec![1.0 / n as f32; n]
        };
        let candidates = actions
            .into_iter()
            .zip(aligned)
            .enumerate()
            .map(|(idx, (payload, prior))| ActionCandidate::new(ActionId(idx), payload, prior))
            .collect();
        Self { candidates }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn policy_priors(&self) -> Vec<f32> {
        self.candidates.iter().map(|c| c.policy_prior).collect()
    }
}

/// root 推理输出
#[derive(Debug, Clone)]
pub struct RootOut<S> {
    /// 隐状态
    pub state: S,
    /// 价值估计
    pub value: f32,
    /// 候选动作集合
    pub candidates: CandidateSet,
    /// 当前玩家编号（单智能体为 0）
    pub to_play: u8,
}

/// recurrent 推理输出
#[derive(Debug, Clone)]
pub struct RecurrentOut<S> {
    /// 隐状态
    pub state: S,
    /// 即时奖励
    pub reward: f32,
    /// 价值估计
    pub value: f32,
    /// 候选动作集合
    pub candidates: CandidateSet,
    /// 是否终止
    pub terminal: bool,
    /// 当前玩家编号
    pub to_play: u8,
    /// 折扣因子（双人零和时为 1.0）
    pub discount: f32,
}

/// 子节点统计信息（暴露给 SearchPolicy）
#[derive(Debug, Clone)]
pub struct ChildStat {
    /// 策略 / target 使用的稳定动作 id
    pub action_id: ActionId,
    /// 对应动作
    pub action: ActionPayload,
    /// 访问次数
    pub visit_count: u32,
    /// 累计价值（子节点视角）
    pub value_sum: f32,
    /// 先验概率
    pub prior: f32,
    /// 从父到此子的即时奖励
    pub reward: f32,
    /// 子节点的 to_play（用于视角翻转：与父同 → +1，与父异 → -1）
    pub to_play: u8,
    /// 子节点的折扣因子
    pub discount: f32,
}

/// 搜索结果
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 根节点所有子节点的统计
    pub children: Vec<ChildStat>,
    /// 推荐动作
    pub recommended: ActionPayload,
    /// 学习用策略目标（visit count 归一化）
    pub learn_policy: Vec<f32>,
    /// root 推理时 value network 对当前状态的估计 `vπ`（Gumbel MuZero completedQ 未访问动作回填）
    pub network_value: f32,
}

impl SearchResult {
    /// 根节点 value 估计：visit 加权的子节点动作价值
    /// `Q(a) = reward(a) + discount(a)·V(child(a))`，`V(child) = value_sum/visit_count`。
    ///
    /// 这是 MuZero self-play 记录 `root_value` 与 reanalyze 重算 value 目标共用的口径，
    /// 抽到此处避免两边公式漂移。
    ///
    /// 单智能体口径（不翻转视角）；双人零和的 negamax 视角翻转待 AlphaZero 用到时再扩展。
    /// 无子节点或零访问时返回 `0.0`。
    pub fn root_value(&self) -> f32 {
        let total_visits: u32 = self.children.iter().map(|c| c.visit_count).sum();
        if total_visits == 0 {
            return 0.0;
        }
        self.children
            .iter()
            .filter(|c| c.visit_count > 0)
            .map(|c| {
                let child_v = c.value_sum / c.visit_count as f32;
                let q = c.reward + c.discount * child_v;
                q * c.visit_count as f32
            })
            .sum::<f32>()
            / total_visits as f32
    }
}

/// MCTS 搜索配置
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MctsConfig {
    /// 模拟次数
    pub num_simulations: u32,
    /// PUCT 公式 c_base 参数
    pub pb_c_base: f32,
    /// PUCT 公式 c_init 参数
    pub pb_c_init: f32,
    /// 根节点 Dirichlet 噪声 alpha
    pub root_dirichlet_alpha: f32,
    /// 根节点探索噪声混合比例
    pub root_exploration_fraction: f32,
    /// 动作选择温度
    pub temperature: f32,
    /// 折扣因子（单智能体使用）
    pub discount: f32,
    /// Sampled MuZero：每节点展开采 K 个候选；`None` = 全量枚举（标准 MuZero）。
    pub sampled_k: Option<usize>,
}

/// 搜索预算与通用运行参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchBudget {
    pub num_simulations: u32,
    pub temperature: f32,
    pub discount: f32,
}

/// PUCT 选择公式参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PuctConfig {
    pub pb_c_base: f32,
    pub pb_c_init: f32,
}

/// 根 Dirichlet 探索参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDirichletConfig {
    pub alpha: f32,
    pub exploration_fraction: f32,
}

/// Sampled MuZero 候选展开参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampledConfig {
    pub k: usize,
}

/// Zero-family MCTS recipe：把预算、选择、根探索与候选展开分层装配。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MctsRecipe {
    pub budget: SearchBudget,
    pub puct: PuctConfig,
    pub root_dirichlet: RootDirichletConfig,
    pub sampled: Option<SampledConfig>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 20,
            pb_c_base: 19652.0,
            pb_c_init: 1.25,
            root_dirichlet_alpha: 0.3,
            root_exploration_fraction: 0.25,
            temperature: 1.0,
            discount: 1.0,
            sampled_k: None,
        }
    }
}

impl MctsConfig {
    pub fn budget(&self) -> SearchBudget {
        SearchBudget {
            num_simulations: self.num_simulations,
            temperature: self.temperature,
            discount: self.discount,
        }
    }

    pub fn puct(&self) -> PuctConfig {
        PuctConfig {
            pb_c_base: self.pb_c_base,
            pb_c_init: self.pb_c_init,
        }
    }

    pub fn root_dirichlet(&self) -> RootDirichletConfig {
        RootDirichletConfig {
            alpha: self.root_dirichlet_alpha,
            exploration_fraction: self.root_exploration_fraction,
        }
    }

    pub fn sampled(&self) -> Option<SampledConfig> {
        self.sampled_k.map(|k| SampledConfig { k })
    }

    pub fn recipe(&self) -> MctsRecipe {
        MctsRecipe {
            budget: self.budget(),
            puct: self.puct(),
            root_dirichlet: self.root_dirichlet(),
            sampled: self.sampled(),
        }
    }

    pub fn from_recipe(recipe: MctsRecipe) -> Self {
        Self {
            num_simulations: recipe.budget.num_simulations,
            temperature: recipe.budget.temperature,
            discount: recipe.budget.discount,
            pb_c_base: recipe.puct.pb_c_base,
            pb_c_init: recipe.puct.pb_c_init,
            root_dirichlet_alpha: recipe.root_dirichlet.alpha,
            root_exploration_fraction: recipe.root_dirichlet.exploration_fraction,
            sampled_k: recipe.sampled.map(|s| s.k),
        }
    }
}

impl Default for MctsRecipe {
    fn default() -> Self {
        MctsConfig::default().recipe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_set_keeps_action_id_separate_from_payload() {
        let candidates = CandidateSet {
            candidates: vec![ActionCandidate::new(
                ActionId(7),
                ActionPayload::Continuous(vec![0.25]),
                0.8,
            )],
        };
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates.candidates[0].id, ActionId(7));
        assert_eq!(candidates.policy_priors(), vec![0.8]);
        assert!(matches!(
            candidates.candidates[0].payload,
            ActionPayload::Continuous(_)
        ));
    }

    #[test]
    fn mcts_recipe_roundtrip_preserves_legacy_config() {
        let cfg = MctsConfig {
            num_simulations: 32,
            pb_c_base: 100.0,
            pb_c_init: 2.0,
            root_dirichlet_alpha: 0.2,
            root_exploration_fraction: 0.15,
            temperature: 0.5,
            discount: 0.97,
            sampled_k: Some(5),
        };
        let roundtrip = MctsConfig::from_recipe(cfg.recipe());
        assert_eq!(roundtrip, cfg);
    }
}
