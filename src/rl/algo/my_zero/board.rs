//! 棋盘双人 self-play 训练闭环（Gomoku M1，收口规划 §3）。
//!
//! # 与单智能体 runner 的三点差异
//! - **双人 negamax 搜索**：[`BoardMctsModel`] 把 `to_play` 藏进 `MctsModel::State`
//!   逐层轮转——内核 backup / PUCT 的视角翻转（`perspective = ±1`）随之自动生效，
//!   树内全程 learned dynamics（canonical MuZero 口径，万金油铁律）。
//! - **根节点 legal_mask**：真环境仅在 self-play/eval 走子、根节点合法掩码、终局判定
//!   三处出场；树内不 mask（learned dynamics 不知规则）。
//! - **negamax MC value target**：棋盘无中间 reward、γ=1，value target 直接回传终局
//!   `G_t = r_t − G_{t+1}`（MuZero 论文棋类口径 = bootstrap 到终局）；经
//!   [`PreparedBatch::Refreshed`](super::runner::PreparedBatch) 通道喂给既有
//!   [`train_batch`](super::runner::train_batch)，单智能体路径零改动。
//!
//! # 视角约定（与 `python/gym_env/gomoku` 契约一致）
//! - obs 恒为**当前执子方**视角（通道 0=己方，1=对方，2=空）；
//! - `board_step` 返回的 reward 是**落子方**视角（+1 胜 / 0 其余）；
//! - value 头学的语义是「当前待走方的局面价值」，与树内 negamax 翻转自洽。
//!
//! # 入口与配方（M4 收口，2026-07-05）
//! 组件组合唯一事实源 = [`recipe::components_for`](super::recipe::components_for)
//! （棋盘 = base 全关，裁决依据见该处注释与棋盘账本）。当前入口为手动档 bench
//! （`tests/gomoku_m*_bench.rs`）与 `just smoke-my-zero-gomoku` 冒烟关卡；
//! 公开 builder 接入随象棋支柱需求再定（`allow(dead_code)` 因入口均在 cfg(test) 保留）。
#![allow(dead_code)]

use super::component::Components;
use super::network::{MyZeroModel, ObsSpec};
use super::rosmo::RosmoTargets;
use super::runner::{self_play_temperature, train_batch, unroll_len_at};
use super::search_policy::MyZeroSearchPolicy;
use super::target::mcts_policy_target;
use super::target_net::hard_update;
use crate::nn::{Adam, Graph, GraphError};
use crate::rl::mcts::{
    ActionCandidate, ActionId, ActionPayload, CandidateSet, ChildStat, Dynamics, MctsConfig,
    MctsModel, RecurrentOut, RootOut, mcts_search,
};
use crate::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ============================================================================
// 双人 MctsModel 适配器
// ============================================================================

/// 双人零和棋盘的 `MctsModel` 适配器（每步 search 构造一次，轻量）。
///
/// `State = (latent, to_play)`：`to_play` 随 recurrent 逐层轮转（0↔1），
/// 内核的 negamax backup / PUCT 视角翻转由此自动生效。
/// 根节点候选按 `legal_mask` 过滤（`ActionId` = 棋盘位置，policy target 经
/// [`scatter_policy_target`](super::target::scatter_policy_target) 投射回全长向量）；
/// 树内节点全量候选（learned dynamics 不知规则，不 mask）。
pub(crate) struct BoardMctsModel<'a> {
    model: &'a MyZeroModel,
    /// 根节点合法掩码（真环境三处出场之一）
    root_legal: Vec<bool>,
    /// 根节点执子方（0 黑 / 1 白）
    root_player: u8,
    action_dim: usize,
}

impl<'a> BoardMctsModel<'a> {
    pub fn new(
        model: &'a MyZeroModel,
        root_legal: Vec<bool>,
        root_player: u8,
        action_dim: usize,
    ) -> Self {
        debug_assert_eq!(root_legal.len(), action_dim);
        Self {
            model,
            root_legal,
            root_player,
            action_dim,
        }
    }

    /// 全量候选（树内展开用；prior 直接取网络 softmax）。
    fn full_candidates(&self, prior: &[f32]) -> CandidateSet {
        CandidateSet::from_actions_and_priors(
            (0..self.action_dim).map(ActionPayload::Discrete).collect(),
            prior.to_vec(),
        )
    }
}

impl MctsModel for BoardMctsModel<'_> {
    type State = (Vec<f32>, u8);

    fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
        let (latent, prior, value) = Dynamics::initial_state(&self.model, obs);

        // 合法候选：ActionId = 棋盘位置（保证 target scatter 槽位正确），prior 归一化
        let mut candidates: Vec<ActionCandidate> = (0..self.action_dim)
            .filter(|&a| self.root_legal[a])
            .map(|a| {
                ActionCandidate::new(
                    ActionId(a),
                    ActionPayload::Discrete(a),
                    prior.get(a).copied().unwrap_or(0.0).max(0.0),
                )
            })
            .collect();
        let z: f32 = candidates.iter().map(|c| c.policy_prior).sum();
        if z > 1e-12 && z.is_finite() {
            for c in &mut candidates {
                c.policy_prior /= z;
            }
        } else if !candidates.is_empty() {
            let u = 1.0 / candidates.len() as f32;
            for c in &mut candidates {
                c.policy_prior = u;
            }
        }

        RootOut {
            state: (latent, self.root_player),
            value,
            candidates: CandidateSet { candidates },
            to_play: self.root_player,
        }
    }

    fn recurrent(&self, state: &Self::State, action: &ActionPayload) -> RecurrentOut<Self::State> {
        let (latent, player) = state;
        let out = Dynamics::recurrent(&self.model, latent, action);
        let next_player = 1 - player;
        RecurrentOut {
            state: (out.next_state, next_player),
            reward: out.reward,
            value: out.value,
            candidates: if out.terminal {
                CandidateSet::empty()
            } else {
                self.full_candidates(&out.prior)
            },
            terminal: out.terminal,
            to_play: next_player,
            // 棋盘 γ=1；终止边 0（与单智能体 DynamicsModel 同口径）
            discount: if out.terminal { 0.0 } else { 1.0 },
        }
    }
}

// ============================================================================
// 8 重对称增广（D4 二面体群：旋转 90°×4 × 镜像 ×2）
// ============================================================================

/// 生成对称变换 `sym ∈ [0,8)` 的棋盘格点置换表：`perm[旧格点] = 新格点`。
///
/// `sym = flip·4 + rot`：先转 `rot` 次 90°（(r,c) → (c, n−1−r)），
/// `flip` 再水平镜像（(r,c) → (r, n−1−c)）。`sym=0` 为恒等。
pub(crate) fn symmetry_perm(side: usize, sym: usize) -> Vec<usize> {
    let rot = sym % 4;
    let flip = sym / 4 == 1;
    (0..side * side)
        .map(|idx| {
            let (mut r, mut c) = (idx / side, idx % side);
            for _ in 0..rot {
                let (nr, nc) = (c, side - 1 - r);
                r = nr;
                c = nc;
            }
            if flip {
                c = side - 1 - c;
            }
            r * side + c
        })
        .collect()
}

/// 对一局按同一置换做同构变换：obs 三平面 / action 格点 / policy target 81 维
/// 全部套 `perm`；reward / value / continuation / player 为对称不变量。
///
/// **整局套同一变换**是正确性关键：unroll 窗口内 obs 序列与 action 序列必须
/// 在同一坐标系下自洽，否则 dynamics 学到的是"棋盘会瞬移"的假物理。
pub(crate) fn augment_game(game: &SelfPlayGame, perm: &[usize], side: usize) -> SelfPlayGame {
    let plane = side * side;
    let steps = game
        .steps
        .iter()
        .map(|s| {
            let raw = s.obs.as_f32();
            debug_assert_eq!(raw.len(), 3 * plane, "棋盘 obs 应为 3 平面");
            let mut obs = vec![0.0f32; 3 * plane];
            for ch in 0..3 {
                let (src, dst) = (&raw[ch * plane..(ch + 1) * plane], ch * plane);
                for (old, &new) in perm.iter().enumerate() {
                    obs[dst + new] = src[old];
                }
            }
            let mut policy_target = vec![0.0f32; s.policy_target.len()];
            for (old, &new) in perm.iter().enumerate() {
                policy_target[new] = s.policy_target[old];
            }
            let action_idx = s.action.first().copied().unwrap_or(0.0) as usize;
            SelfPlayStep {
                obs: obs.into(),
                action: vec![perm.get(action_idx).copied().unwrap_or(action_idx) as f32],
                policy_target,
                ..s.clone()
            }
        })
        .collect();
    SelfPlayGame {
        steps,
        outcome: game.outcome,
    }
}

// ============================================================================
// negamax 目标
// ============================================================================

/// negamax MC 回报：`G_t = r_t − G_{t+1}`（`G_{end}=0`），视角 = 位置 `pos` 的执子方。
///
/// 棋盘 reward 约定为落子方视角（+1 胜 / 0 平与中途），故胜局回报沿轨迹交替 ±1、
/// 平局全 0。γ=1、无中间 reward 时等价于「bootstrap 到终局」的 MuZero 棋类口径。
pub(crate) fn negamax_mc_return(steps: &[SelfPlayStep], pos: usize) -> f32 {
    let mut g = 0.0;
    for step in steps[pos..].iter().rev() {
        g = step.reward - g;
    }
    g
}

/// 根节点 negamax value：visit 加权 `Q(a) = r(a) + discount·perspective·V(child)`。
///
/// 与 [`SearchResult::root_value`](crate::rl::mcts::SearchResult::root_value) 的
/// 单智能体口径的唯一差异是 perspective 翻转（子节点 value 是对方视角）。
fn negamax_root_value(children: &[ChildStat], root_player: u8) -> f32 {
    let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
    if total_visits == 0 {
        return 0.0;
    }
    children
        .iter()
        .filter(|c| c.visit_count > 0)
        .map(|c| {
            let v = c.value_sum / c.visit_count as f32;
            let perspective = if c.to_play == root_player { 1.0 } else { -1.0 };
            let q = c.reward + c.discount * perspective * v;
            q * c.visit_count as f32
        })
        .sum::<f32>()
        / total_visits as f32
}

/// 一条样本 unroll 窗口的棋盘现算 target（借 [`RosmoTargets`] 槽位结构走 Refreshed 通道）。
///
/// - policy：读存量 MCTS target（棋盘 M1 不刷新）；越界槽位 uniform（同 legacy padding）。
/// - value：negamax MC 回报；越界 0。
/// - `bc_weights` 全 0（BC 项零开销跳过，见 `train_unroll_batch`）。
fn board_targets(
    steps: &[SelfPlayStep],
    start: usize,
    actual_k: usize,
    action_dim: usize,
) -> RosmoTargets {
    let len = steps.len();
    let uniform = vec![1.0 / action_dim as f32; action_dim];
    let mut policies = Vec::with_capacity(actual_k + 1);
    let mut values = Vec::with_capacity(actual_k + 1);
    for j in 0..=actual_k {
        let pos = start + j;
        if pos < len {
            policies.push(steps[pos].policy_target.clone());
            values.push(negamax_mc_return(steps, pos));
        } else {
            policies.push(uniform.clone());
            values.push(0.0);
        }
    }
    RosmoTargets {
        policies,
        values,
        bc_weights: vec![0.0; actual_k],
    }
}

// ============================================================================
// self-play / eval
// ============================================================================

fn board_mcts_config(
    num_simulations: u32,
    temperature: f32,
    root_exploration_fraction: f32,
) -> MctsConfig {
    MctsConfig {
        num_simulations,
        temperature,
        discount: 1.0,
        // AlphaZero 系口径 alpha ≈ 10/|A|（9×9 → ~0.12）；M1 取 0.15 常数档
        root_dirichlet_alpha: 0.15,
        root_exploration_fraction,
        ..MctsConfig::default()
    }
}

/// 自对弈一局（双方同一网络；真环境只在走子/根 mask/终局三处出场）。
fn board_self_play_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    components: &Components,
    mcts_cfg: &MctsConfig,
    action_dim: usize,
    reset_seed: u64,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    env.reset(Some(reset_seed));
    let mut steps: Vec<SelfPlayStep> = Vec::new();

    loop {
        let obs = env.board_observation_flat();
        let player = env.current_player();
        let legal = env.legal_mask();

        let board_model = BoardMctsModel::new(model, legal, player, action_dim);
        let policy = MyZeroSearchPolicy::from_components(components);
        let result = mcts_search(&board_model, &policy, &obs, mcts_cfg, rng);

        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let root_value = negamax_root_value(&result.children, player);
        // completedQ 臂：改进策略目标（negamax 视角翻转在 target.rs 内按 root_to_play 处理）
        let cq = components
            .completed_q_target
            .then_some((components.cq_c_visit, components.cq_c_scale));
        let policy_target = mcts_policy_target(&result, cq, action_dim, player);

        let (reward, terminal) = env.board_step(action_idx);
        steps.push(SelfPlayStep {
            obs: obs.into(),
            action: vec![action_idx as f32],
            policy_target,
            player,
            reward,
            root_value: Some(root_value),
            terminated: terminal,
            truncated: false,
            continuation: if terminal { 0.0 } else { 1.0 },
            extras: Default::default(),
        });

        if terminal {
            return steps;
        }
    }
}

/// 对内置对手 env（如 `Gomoku-random-v0`）greedy 评测 `n` 局，返回 (胜率, 局均 return)。
///
/// 我方恒执黑先行（env 契约）；每步搜索带根 legal_mask，temperature=0、无 Dirichlet。
fn board_eval_vs(
    env: &GymEnv,
    model: &MyZeroModel,
    components: &Components,
    action_dim: usize,
    num_simulations: u32,
    n_episodes: usize,
    eval_seed: u64,
) -> (f32, f32) {
    let eval_cfg = board_mcts_config(num_simulations, 0.0, 0.0);
    let mut wins = 0usize;
    let mut total_return = 0.0f32;

    for i in 0..n_episodes {
        let seed = eval_seed.wrapping_add(i as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        env.reset(Some(seed));
        loop {
            let obs = env.board_observation_flat();
            let player = env.current_player();
            let legal = env.legal_mask();
            let board_model = BoardMctsModel::new(model, legal, player, action_dim);
            let policy = MyZeroSearchPolicy::from_components(components);
            let result = mcts_search(&board_model, &policy, &obs, &eval_cfg, &mut rng);
            let action_idx = match &result.recommended {
                ActionPayload::Discrete(idx) => *idx,
                _ => 0,
            };
            // env.step 内部执行我方落子 + 对手回应
            let (_, reward, terminated, truncated) = env.step(&[action_idx as f32]);
            if terminated || truncated {
                total_return += reward;
                if reward > 0.5 {
                    wins += 1;
                }
                break;
            }
        }
    }
    let n = n_episodes.max(1) as f32;
    (wins as f32 / n, total_return / n)
}

/// 已启用组件标签（棋盘线日志用；只列棋盘消融关心的开关）。
fn tags_of(c: &Components) -> Vec<&'static str> {
    let mut tags = Vec::new();
    if c.gumbel {
        tags.push("Gumbel");
    }
    if c.completed_q_target {
        tags.push("completedQ");
    }
    if c.consistency {
        tags.push("consistency");
    }
    if c.reconstruction {
        tags.push("reconstruction");
    }
    tags
}

/// 权重快照：同构新图上物化第二个模型实例，逐参数硬拷贝（gating 对弈用）。
fn snapshot_model(
    model: &MyZeroModel,
    spec: ObsSpec,
    action_dim: usize,
    latent_dim: usize,
    seed: u64,
) -> Result<MyZeroModel, GraphError> {
    let graph = Graph::new_with_seed(seed ^ 0x5AFE);
    let snap = MyZeroModel::new_with_spec(&graph, spec, action_dim, latent_dim)?;
    // 两实例构造路径完全相同 → parameters() 顺序一致，硬拷贝安全
    hard_update(&model.parameters(), &snap.parameters());
    Ok(snap)
}

/// 双模型对弈一局（黑白各自网络，greedy、无噪声）。返回胜方 player id（None = 平局）。
///
/// `opening_plies` 步**随机合法开局**（由 `seed` 决定，双方各半）：纯贪心对弈整局
/// 确定，无开局多样性时 n 局退化为同一盘棋重复（M2 首裁实测教训）；随机开局是
/// AlphaZero 系评测对局的标准做法，同一 `seed` 的开局可跨色镜像复用保证公平。
fn board_duel_episode(
    env: &GymEnv,
    black: &MyZeroModel,
    white: &MyZeroModel,
    components: &Components,
    action_dim: usize,
    num_simulations: u32,
    opening_plies: usize,
    seed: u64,
) -> Option<u8> {
    let eval_cfg = board_mcts_config(num_simulations, 0.0, 0.0);
    let mut rng = StdRng::seed_from_u64(seed);
    env.reset(Some(seed));

    for _ in 0..opening_plies {
        let legal = env.legal_mask();
        let choices: Vec<usize> = (0..action_dim).filter(|&a| legal[a]).collect();
        if choices.is_empty() {
            return None;
        }
        let player = env.current_player();
        let action = choices[rng.gen_range(0..choices.len())];
        let (reward, terminal) = env.board_step(action);
        if terminal {
            // 开局撞终局的概率可忽略（9×9 需 ≥9 步才可能连五），防御性处理
            return (reward > 0.5).then_some(player);
        }
    }

    loop {
        let obs = env.board_observation_flat();
        let player = env.current_player();
        let legal = env.legal_mask();
        let model = if player == 0 { black } else { white };
        let board_model = BoardMctsModel::new(model, legal, player, action_dim);
        let policy = MyZeroSearchPolicy::from_components(components);
        let result = mcts_search(&board_model, &policy, &obs, &eval_cfg, &mut rng);
        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let (reward, terminal) = env.board_step(action_idx);
        if terminal {
            return (reward > 0.5).then_some(player);
        }
    }
}

/// `candidate` 对 `opponent` 的比赛分（胜 1 / 平 0.5，gating ≥55% 准入线用此分数）。
///
/// `n_games/2` 个随机开局（各 2 步），每个开局镜像打两盘（candidate 执黑/执白各一），
/// 同开局同 seed——先手优势与开局运气跨色对消。
fn board_duel_score(
    env: &GymEnv,
    candidate: &MyZeroModel,
    opponent: &MyZeroModel,
    components: &Components,
    action_dim: usize,
    num_simulations: u32,
    n_games: usize,
    seed: u64,
) -> f32 {
    const OPENING_PLIES: usize = 2;
    let n_pairs = (n_games / 2).max(1);
    let mut score = 0.0f32;
    for pair in 0..n_pairs {
        let opening_seed = seed.wrapping_add(pair as u64);
        for cand_is_black in [true, false] {
            let (black, white) = if cand_is_black {
                (candidate, opponent)
            } else {
                (opponent, candidate)
            };
            let winner = board_duel_episode(
                env,
                black,
                white,
                components,
                action_dim,
                num_simulations,
                OPENING_PLIES,
                opening_seed,
            );
            let cand_color = u8::from(!cand_is_black);
            score += match winner {
                Some(p) if p == cand_color => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
        }
    }
    score / (n_pairs * 2) as f32
}

// ============================================================================
// 训练入口
// ============================================================================

/// 棋盘训练配置（默认值 = M2 达标口径；组件默认走 recipe，消融臂按需覆盖）。
#[derive(Debug, Clone)]
pub(crate) struct BoardTrainConfig {
    /// self-play 环境（双方外部落子）
    pub env_id: &'static str,
    /// 评测环境（内置对手，如 `Gomoku-random-v0`）
    pub eval_env_id: &'static str,
    pub latent_dim: usize,
    pub num_simulations: u32,
    pub lr: f32,
    pub k_unroll: usize,
    pub train_batch_size: usize,
    pub trains_per_episode: usize,
    /// buffer 容量（整局计）
    pub buffer_capacity: usize,
    pub start_training_after: usize,
    pub temp_hold_episodes: usize,
    pub temp_decay_episodes: usize,
    pub max_episodes: usize,
    pub eval_every: usize,
    pub eval_episodes: usize,
    /// vs 评测对手的达标胜率（如 0.95）
    pub solved_win_rate: f32,
    pub seed: u64,
    /// 周期 eval 命中 `solved_win_rate` 即提前收工（M1 口径 true；M2 满预算跑完 false）
    pub early_stop: bool,
    /// 该局结束后拍权重快照（M2 gating 对弈的"旧 checkpoint"；None = 不拍）
    pub snapshot_at_episode: Option<usize>,
    /// 训后终局闸门局数（vs random 正裁 + gating 对弈各用此数；0 = 跳过终局闸门）
    pub gate_games: usize,
    /// 训后 naive 梯队观察性评测（每档 `eval_episodes` 局，不设门槛）
    pub naive_ladder: bool,
    /// 组件开关（M3 消融臂注入口；默认 base 全关）
    pub components: Components,
    /// representation 编码器用 CNN（`ObsSpec::Image{channels:3, side}`，复用 Phase 1
    /// conv 栈）；false = Flat MLP（M1/M2 基线）。M3 表征臂开关。
    pub cnn_repr: bool,
    /// 8 重对称增广（D4）：训练采样时每条样本随机套一个对称变换（整局同构）。
    pub augment: bool,
}

impl Default for BoardTrainConfig {
    /// Gomoku 9×9 M1 口径。
    fn default() -> Self {
        Self {
            env_id: "Gomoku-selfplay-v0",
            eval_env_id: "Gomoku-random-v0",
            latent_dim: 64,
            num_simulations: 100,
            lr: 0.02,
            k_unroll: 5,
            train_batch_size: 16,
            trains_per_episode: 4,
            buffer_capacity: 200,
            start_training_after: 10,
            temp_hold_episodes: 100,
            temp_decay_episodes: 100,
            max_episodes: 400,
            eval_every: 25,
            eval_episodes: 20,
            solved_win_rate: 0.95,
            seed: 42,
            early_stop: true,
            snapshot_at_episode: None,
            gate_games: 0,
            naive_ladder: false,
            components: super::recipe::components_for("Gomoku-selfplay-v0"),
            cnn_repr: false,
            augment: false,
        }
    }
}

/// 棋盘训练结果（M1 口径：胜率闸门）。
#[derive(Debug, Clone)]
pub(crate) struct BoardTrainReport {
    pub final_win_rate: f32,
    pub best_win_rate: f32,
    /// 首次胜率 ≥ solved 时的累计 env-steps（None = 未达标）
    pub solved_at_steps: Option<u64>,
    pub total_env_steps: u64,
    pub wall_secs: f32,
    /// 终局闸门：vs random 正裁胜率（`gate_games` 局；None = 未跑）
    pub gate_vs_random: Option<f32>,
    /// 终局闸门：vs 半程快照 gating 比赛分（胜 1 / 平 0.5；None = 未拍快照或未跑）
    pub gate_vs_checkpoint: Option<f32>,
    /// naive 梯队观察性胜率（档名, 胜率）
    pub naive_win_rates: Vec<(&'static str, f32)>,
}

/// 棋盘双人训练闭环（单 seed）：self-play → negamax target 训练 → vs 对手评测。
pub(crate) fn train_board(cfg: &BoardTrainConfig) -> Result<BoardTrainReport, GraphError> {
    Python::attach(|py| {
        let wall_t0 = std::time::Instant::now();
        let mut components = cfg.components.clone();
        // Refreshed 通道会把 rosmo_alpha 当 BC 系数；棋盘的 bc_weights 恒 0，
        // 此处显式归零以杜绝语义误触发（防御性，见 train_batch 的 bc_coef 推导）
        components.rosmo_alpha = 0.0;
        if !tags_of(&components).is_empty() {
            println!("[MyZero-board] 组件: {}", tags_of(&components).join(" + "));
        }

        let env = GymEnv::new(py, cfg.env_id);
        let eval_env = GymEnv::new(py, cfg.eval_env_id);
        env.reset(Some(cfg.seed));
        let action_dim = env.legal_mask().len();
        let obs_len = env.board_observation_flat().len();
        let side = (action_dim as f32).sqrt() as usize;
        debug_assert_eq!(side * side, action_dim, "棋盘动作数应为平方数");
        debug_assert_eq!(obs_len, 3 * action_dim, "棋盘 obs 应为 3 平面");

        println!(
            "[MyZero-board {}] obs={obs_len}(3×{side}²) |A|={action_dim} sims={} γ=1.0 negamax",
            cfg.env_id, cfg.num_simulations,
        );

        // base = Flat MLP 编码器（243 维三平面直通）；`cnn_repr` 臂 = stride-1 棋盘卷积塔。
        let obs_spec = if cfg.cnn_repr {
            println!("[MyZero-board] repr=CNN（stride-1 棋盘塔）");
            ObsSpec::Board { channels: 3, side }
        } else {
            ObsSpec::Flat(obs_len)
        };
        if cfg.augment {
            println!("[MyZero-board] 8 重对称增广开启（D4，采样时随机套变换）");
        }
        let perms: Vec<Vec<usize>> = (0..8).map(|s| symmetry_perm(side, s)).collect();
        let graph = Graph::new_with_seed(cfg.seed);
        let model = MyZeroModel::new_with_spec(&graph, obs_spec, action_dim, cfg.latent_dim)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), cfg.lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(cfg.buffer_capacity);
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let mut total_steps: u64 = 0;
        let mut best_win_rate = 0.0f32;
        let mut final_win_rate = 0.0f32;
        let mut hit_solved: Option<u64> = None;
        let mut snapshot: Option<MyZeroModel> = None;

        for ep in 0..cfg.max_episodes {
            let t0 = std::time::Instant::now();
            let temperature =
                self_play_temperature(ep, cfg.temp_hold_episodes, cfg.temp_decay_episodes);
            let mcts_cfg = board_mcts_config(
                cfg.num_simulations,
                temperature,
                MctsConfig::default().root_exploration_fraction,
            );

            let steps = board_self_play_episode(
                &env,
                &model,
                &components,
                &mcts_cfg,
                action_dim,
                cfg.seed.wrapping_add(1_000_000 + ep as u64),
                &mut rng,
            );
            let ep_len = steps.len();
            total_steps += ep_len as u64;
            let last = steps.last().expect("棋局至少一步");
            let outcome = if last.reward > 0.5 {
                GameOutcome::Win(last.player)
            } else {
                GameOutcome::Draw
            };
            buffer.push(SelfPlayGame { steps, outcome });

            // 训练：零克隆采样 + 现算 negamax target（走 Refreshed 通道，不写回）；
            // augment 开启时每条样本随机套 D4 对称（整局同构变换，clone 为代价）
            let mut avg_loss = 0.0;
            if buffer.len() >= cfg.start_training_after {
                let mut loss_sum = 0.0;
                for _ in 0..cfg.trains_per_episode {
                    let mut picks: Vec<(usize, usize)> = Vec::with_capacity(cfg.train_batch_size);
                    for idx in buffer.sample_indices(cfg.train_batch_size, &mut rng) {
                        let len = buffer
                            .get_ref(idx)
                            .map(|g| g.steps.len())
                            .unwrap_or_default();
                        if len < 2 {
                            continue;
                        }
                        picks.push((idx, rng.gen_range(0..len)));
                    }
                    let augmented: Vec<SelfPlayGame> = if cfg.augment {
                        picks
                            .iter()
                            .map(|&(idx, _)| {
                                let game = buffer.get_ref(idx).expect("buffer 下标应有效");
                                let sym = rng.gen_range(0..8);
                                augment_game(game, &perms[sym], side)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let game_at = |i: usize, idx: usize| -> &SelfPlayGame {
                        if cfg.augment {
                            &augmented[i]
                        } else {
                            buffer.get_ref(idx).expect("buffer 下标应有效")
                        }
                    };
                    let targets: Vec<RosmoTargets> = picks
                        .iter()
                        .enumerate()
                        .map(|(i, &(idx, start))| {
                            let game = game_at(i, idx);
                            let actual_k = unroll_len_at(&game.steps, start, cfg.k_unroll);
                            board_targets(&game.steps, start, actual_k, action_dim)
                        })
                        .collect();
                    let train_view: Vec<(&SelfPlayGame, usize)> = picks
                        .iter()
                        .enumerate()
                        .map(|(i, &(idx, start))| (game_at(i, idx), start))
                        .collect();
                    loss_sum += train_batch(
                        &model,
                        &mut optimizer,
                        &train_view,
                        Some(&targets),
                        cfg.k_unroll,
                        // td_steps 在 Refreshed 通道不参与 value target（现算 negamax）
                        cfg.k_unroll,
                        1.0,
                        &components,
                        None,
                    )?;
                }
                avg_loss = loss_sum / cfg.trains_per_episode as f32;
            }

            let winner_tag = match outcome {
                GameOutcome::Win(0) => "黑胜",
                GameOutcome::Win(_) => "白胜",
                _ => "平局",
            };
            println!(
                "Ep {:4}: len={ep_len:3} {winner_tag} loss={avg_loss:.4} temp={temperature:.2} total_env_steps={total_steps} t={:.2}s",
                ep + 1,
                t0.elapsed().as_secs_f32()
            );

            if (ep + 1) % cfg.eval_every == 0 && buffer.len() >= cfg.start_training_after {
                let (win_rate, mean_ret) = board_eval_vs(
                    &eval_env,
                    &model,
                    &components,
                    action_dim,
                    cfg.num_simulations,
                    cfg.eval_episodes,
                    cfg.seed.wrapping_add(9_000_000 + ep as u64),
                );
                final_win_rate = win_rate;
                best_win_rate = best_win_rate.max(win_rate);
                println!(
                    "  eval vs {}: win_rate={win_rate:.2} mean_return={mean_ret:.2}（{} 局，total_env_steps={total_steps}）",
                    cfg.eval_env_id, cfg.eval_episodes,
                );
                if win_rate >= cfg.solved_win_rate && hit_solved.is_none() {
                    hit_solved = Some(total_steps);
                    println!("  ✅ 达标 ep={} total_env_steps={total_steps}", ep + 1);
                    if cfg.early_stop {
                        break;
                    }
                }
            }

            if cfg.snapshot_at_episode == Some(ep + 1) {
                snapshot = Some(snapshot_model(
                    &model,
                    obs_spec,
                    action_dim,
                    cfg.latent_dim,
                    cfg.seed,
                )?);
                println!("  📸 已拍权重快照（ep={}，gating 对弈基准）", ep + 1);
            }
        }

        // ---- 训后终局闸门（M2）----
        let mut gate_vs_random = None;
        let mut gate_vs_checkpoint = None;
        let mut naive_win_rates = Vec::new();

        if cfg.gate_games > 0 {
            let (wr, _) = board_eval_vs(
                &eval_env,
                &model,
                &components,
                action_dim,
                cfg.num_simulations,
                cfg.gate_games,
                cfg.seed.wrapping_add(77_000_000),
            );
            gate_vs_random = Some(wr);
            println!(
                "[闸门] vs random（{} 局）: win_rate={wr:.3}",
                cfg.gate_games
            );

            if let Some(snap) = &snapshot {
                let score = board_duel_score(
                    &env,
                    &model,
                    snap,
                    &components,
                    action_dim,
                    cfg.num_simulations,
                    cfg.gate_games,
                    cfg.seed.wrapping_add(88_000_000),
                );
                gate_vs_checkpoint = Some(score);
                println!(
                    "[闸门] vs 半程快照（{} 局，黑白各半）: score={score:.3}",
                    cfg.gate_games
                );
            }
        }

        if cfg.naive_ladder {
            const LADDER: [(&str, &str); 4] = [
                ("naive0", "Gomoku-naive0-v0"),
                ("naive1", "Gomoku-naive1-v0"),
                ("naive2", "Gomoku-naive2-v0"),
                ("naive3", "Gomoku-naive3-v0"),
            ];
            for (name, env_id) in LADDER {
                let ladder_env = GymEnv::new(py, env_id);
                let (wr, _) = board_eval_vs(
                    &ladder_env,
                    &model,
                    &components,
                    action_dim,
                    cfg.num_simulations,
                    cfg.eval_episodes,
                    cfg.seed.wrapping_add(99_000_000),
                );
                ladder_env.close();
                naive_win_rates.push((name, wr));
                println!(
                    "[梯队] vs {name}（{} 局，观察性）: win_rate={wr:.2}",
                    cfg.eval_episodes
                );
            }
        }

        env.close();
        eval_env.close();
        let wall_secs = wall_t0.elapsed().as_secs_f32();
        println!(
            "📈 {} win_rate={final_win_rate:.2}（best {best_win_rate:.2}）| 门槛 {} | {wall_secs:.1}s",
            cfg.env_id, cfg.solved_win_rate,
        );
        Ok(BoardTrainReport {
            final_win_rate,
            best_win_rate,
            solved_at_steps: hit_solved,
            total_env_steps: total_steps,
            wall_secs,
            gate_vs_random,
            gate_vs_checkpoint,
            naive_win_rates,
        })
    })
}
