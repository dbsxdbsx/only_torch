//! MCTS 核心数据类型定义

/// 动作载荷：支持离散、连续及混合动作空间
#[derive(Debug, Clone, PartialEq)]
pub enum ActionPayload {
    Discrete(usize),
    Continuous(Vec<f32>),
    Hybrid { discrete: usize, continuous: Vec<f32> },
}

/// root 推理输出
#[derive(Debug, Clone)]
pub struct RootOut<S> {
    /// 隐状态
    pub state: S,
    /// 先验概率分布
    pub prior: Vec<f32>,
    /// 价值估计
    pub value: f32,
    /// 候选动作列表
    pub candidate_actions: Vec<ActionPayload>,
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
    /// 先验概率分布
    pub prior: Vec<f32>,
    /// 候选动作列表
    pub candidate_actions: Vec<ActionPayload>,
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
}

/// MCTS 搜索配置
#[derive(Debug, Clone)]
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
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 50,
            pb_c_base: 19652.0,
            pb_c_init: 1.25,
            root_dirichlet_alpha: 0.3,
            root_exploration_fraction: 0.25,
            temperature: 1.0,
            discount: 1.0,
        }
    }
}
