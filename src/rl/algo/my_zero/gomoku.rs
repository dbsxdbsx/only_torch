//! 五子棋领域插件（棋盘管线 `board.rs` 的 Gomoku 专用件）。
//!
//! # 定位（2026-07-05 拆分）
//! `board.rs` 是**通用棋盘双人管线**（self-play / negamax target / 训练闭环 /
//! 增广 / 评测），不含任何具体棋种知识；本模块承载五子棋的领域内容，通过
//! 三个插拔点接入：
//! - **规则层** [`RulesBoard`]：树内真规则推演（`true_rules_tree` 开关）；
//! - **树内转移适配器** [`TrueRulesBoardModel`]：AlphaZero 式真规则树；
//! - **开局课程生成器** [`tactical_opening`]：定向战术开局
//!   （`tactical_opening_fraction` 开关）。
//!
//! 未来象棋支柱按同样的三件套另立 `xiangqi.rs`，`board.rs` 零改动是拆分的
//! 验收标准（当前 `train_board` 内 naive 梯队 env 名仍硬编码 Gomoku，属评测
//! 配置项，随象棋支柱提入 `BoardTrainConfig`）。

use super::network::MyZeroModel;
use crate::rl::mcts::{
    ActionCandidate, ActionId, ActionPayload, CandidateSet, Dynamics, MctsModel, RecurrentOut,
    RootOut,
};
use rand::Rng;
use rand::rngs::StdRng;

// ============================================================================
// 规则层（naive0 issue §四-② 诊断臂进库；可插拔，默认关）
// ============================================================================

/// 五连胜利长度（与 `python/gym_env/gomoku/board.py` 的 `win_length=5` 契约一致）。
const WIN_LENGTH: usize = 5;

/// 纯 Rust 棋盘规则（树内推演专用）：与 Python `Board` 同规则、零 pyo3 调用。
///
/// 只服务 [`TrueRulesBoardModel`] 的树内 clone/step——真环境（走子/根 mask/终局）
/// 仍走 Python env，不改三处出场契约。规则等价性由 `tests/board.rs` 的
/// 「随机对局 vs Python 板逐步对照」单测锁死。
#[derive(Clone)]
pub(crate) struct RulesBoard {
    side: usize,
    /// 黑（0）/ 白（1）子平面，1 = 有子
    stones: [Vec<u8>; 2],
    to_play: u8,
    move_count: usize,
    done: bool,
}

impl RulesBoard {
    /// 从「当前方视角 obs（3 平面）+ 执子方」重建绝对局面（根节点入口）。
    ///
    /// obs 契约（同 Python `Board.observation`）：通道 0 = 己方、1 = 对方、2 = 空。
    pub fn from_obs(obs: &[f32], player: u8, side: usize) -> Self {
        let plane = side * side;
        debug_assert_eq!(obs.len(), 3 * plane, "棋盘 obs 应为 3 平面");
        let read = |ch: usize| -> Vec<u8> {
            obs[ch * plane..(ch + 1) * plane]
                .iter()
                .map(|&v| u8::from(v > 0.5))
                .collect()
        };
        let (own, opp) = (read(0), read(1));
        let move_count = own
            .iter()
            .chain(opp.iter())
            .map(|&v| v as usize)
            .sum::<usize>();
        let stones = if player == 0 { [own, opp] } else { [opp, own] };
        Self {
            side,
            stones,
            to_play: player,
            move_count,
            done: false,
        }
    }

    pub fn to_play(&self) -> u8 {
        self.to_play
    }

    /// 落子（同 Python `Board.step` 语义）：返回（落子方视角 reward, terminal）。
    ///
    /// 树内候选已按合法掩码过滤，非法着走防御分支（判负终局，与 Python 一致）。
    pub fn step(&mut self, action: usize) -> (f32, bool) {
        debug_assert!(!self.done, "终局后不应再落子");
        let plane = self.side * self.side;
        if action >= plane || self.stones[0][action] == 1 || self.stones[1][action] == 1 {
            self.done = true;
            return (-1.0, true);
        }
        let color = self.to_play as usize;
        self.stones[color][action] = 1;
        self.move_count += 1;

        let reward = if self.wins_at(color, action) {
            self.done = true;
            1.0
        } else if self.move_count >= plane {
            self.done = true; // 平局
            0.0
        } else {
            0.0
        };
        self.to_play = 1 - self.to_play;
        (reward, self.done)
    }

    /// 合法掩码（空位即合法）。
    pub fn legal_mask(&self) -> Vec<bool> {
        (0..self.side * self.side)
            .map(|i| self.stones[0][i] == 0 && self.stones[1][i] == 0)
            .collect()
    }

    /// 当前方视角展平观察（3 平面，同 Python `Board.observation_flat`）。
    pub fn observation_flat(&self) -> Vec<f32> {
        let plane = self.side * self.side;
        let (own, opp) = (
            &self.stones[self.to_play as usize],
            &self.stones[1 - self.to_play as usize],
        );
        let mut obs = vec![0.0f32; 3 * plane];
        for i in 0..plane {
            obs[i] = own[i] as f32;
            obs[plane + i] = opp[i] as f32;
            obs[2 * plane + i] = f32::from(own[i] == 0 && opp[i] == 0);
        }
        obs
    }

    /// 假设 `color` 在空位 `action` 落子是否立即成五（`wins_at` 不读 `b[action]`
    /// 本身，天然支持假设性探测——一步胜威胁检查入口）。
    pub fn would_win(&self, color: u8, action: usize) -> bool {
        debug_assert!(
            self.stones[0][action] == 0 && self.stones[1][action] == 0,
            "would_win 只应探测空位"
        );
        self.wins_at(color as usize, action)
    }

    /// `color` 是否存在一步胜着（对手视角 = 必挡局面）。
    pub fn has_win_in_1(&self, color: u8) -> bool {
        (0..self.side * self.side)
            .any(|a| self.stones[0][a] == 0 && self.stones[1][a] == 0 && self.would_win(color, a))
    }

    /// 增量胜利检查（只查最后落子四方向，同 Python `_check_winner_incremental`）。
    fn wins_at(&self, color: usize, action: usize) -> bool {
        let d = self.side as isize;
        let (row, col) = ((action / self.side) as isize, (action % self.side) as isize);
        let b = &self.stones[color];
        for (dr, dc) in [(0isize, 1isize), (1, 0), (1, 1), (1, -1)] {
            let mut count = 1;
            for sign in [1isize, -1] {
                for i in 1..WIN_LENGTH as isize {
                    let (r, c) = (row + dr * i * sign, col + dc * i * sign);
                    if r >= 0 && r < d && c >= 0 && c < d && b[(r * d + c) as usize] == 1 {
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            if count >= WIN_LENGTH {
                return true;
            }
        }
        false
    }
}

// ============================================================================
// 树内真规则适配器
// ============================================================================

/// 树内真规则的双人 `MctsModel` 适配器（AlphaZero 式）。
///
/// 与 `BoardMctsModel`（learned dynamics）的唯一差异 = **树内转移函数**：
/// - 转移/终局：[`RulesBoard`] 真规则推演（终局 reward 直接 backup，「一步胜/必挡」
///   在树内结构性可见）；
/// - 叶子先验/价值：仍由网络 `initial_state`（representation + prediction）在**真实
///   obs** 上给出——不用 dynamics/reward 头；
/// - 树内候选按真规则 legal mask 过滤（learned 路径树内不 mask，因 dynamics 不知规则）。
///
/// 训练侧完全不变（loss 仍含 dynamics unroll），单变量 = 树内转移，经
/// `BoardTrainConfig::true_rules_tree` 开关插拔，默认关。
pub(crate) struct TrueRulesBoardModel<'a> {
    model: &'a MyZeroModel,
    root_board: RulesBoard,
    action_dim: usize,
}

impl<'a> TrueRulesBoardModel<'a> {
    /// `obs`/`player` 为根局面（真环境读出），`side` 由 `action_dim` 开方而来。
    pub fn new(model: &'a MyZeroModel, obs: &[f32], player: u8, side: usize) -> Self {
        Self {
            model,
            root_board: RulesBoard::from_obs(obs, player, side),
            action_dim: side * side,
        }
    }

    /// 按棋盘合法掩码过滤候选并归一化 prior（根与树内同一路径）。
    fn masked_candidates(&self, board: &RulesBoard, prior: &[f32]) -> CandidateSet {
        let legal = board.legal_mask();
        let mut candidates: Vec<ActionCandidate> = (0..self.action_dim)
            .filter(|&a| legal[a])
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
        CandidateSet { candidates }
    }
}

impl MctsModel for TrueRulesBoardModel<'_> {
    type State = RulesBoard;

    fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
        let (_latent, prior, value) = Dynamics::initial_state(&self.model, obs);
        RootOut {
            state: self.root_board.clone(),
            value,
            candidates: self.masked_candidates(&self.root_board, &prior),
            to_play: self.root_board.to_play(),
        }
    }

    fn recurrent(&self, state: &Self::State, action: &ActionPayload) -> RecurrentOut<Self::State> {
        let action_idx = match action {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let mut board = state.clone();
        let (reward, terminal) = board.step(action_idx);
        let next_player = board.to_play();
        let (value, candidates) = if terminal {
            // 终局：价值已由 reward 完全表达，无后续候选
            (0.0, CandidateSet::empty())
        } else {
            let obs = board.observation_flat();
            let (_latent, prior, value) = Dynamics::initial_state(&self.model, &obs);
            (value, self.masked_candidates(&board, &prior))
        };
        RecurrentOut {
            state: board,
            reward,
            value,
            candidates,
            terminal,
            to_play: next_player,
            discount: if terminal { 0.0 } else { 1.0 },
        }
    }
}

// ============================================================================
// 定向战术开局课程（naive0 issue §四-⑥；可插拔，默认关）
// ============================================================================

/// 战术开局课题（`tactical_opening` 的课程内容枚举）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TacticalCourse {
    /// 一步胜威胁（四连留胜点）：防守方必挡否则对手下一手成五——防守课题（⑥臂进库）。
    WinIn1,
    /// 活三（三连两端开放）：放任则下一手成活四（双头必胜）——进攻/预防课题
    /// （naive1+ 梯队的核心技能：制造与扼杀活三）。
    OpenThree,
}

/// 战术开局生成：构造指定课题的中局局面，返回从空盘起的交替落子序列。
///
/// 机制通用（Leela 开局库 / KataGo forced openings 同族：向 self-play 起始分布
/// 注入决策关键局面，补小预算下自然覆盖的抽签方差）；内容领域特定（五子棋的
/// 威胁线构造），故生成器住在领域插件层，与 [`RulesBoard`] 同级。
///
/// 构造式而非随机推演（随机对局连子成线概率过低）：随机取一条盘内 5 格窗口，
/// 威胁方按课题占格（[`TacticalCourse::WinIn1`] = 占 4 留 1 胜点；
/// [`TacticalCourse::OpenThree`] = 占中间 3、两端留空），陪跑方在窗口外落随机
/// 散点；威胁方黑/白随机（攻防两种视角都进训练分布）。终检双保险：
/// replay 无终局 + 课题不变量成立（WinIn1 = 威胁方有一步胜着且防守方无；
/// OpenThree = 双方均无一步胜着且三连两端空）。
pub(crate) fn tactical_opening(
    side: usize,
    course: TacticalCourse,
    rng: &mut StdRng,
) -> Option<Vec<usize>> {
    const MAX_TRIES: usize = 20;
    let plane = side * side;
    let empty_obs = vec![0.0f32; 3 * plane];

    'retry: for _ in 0..MAX_TRIES {
        // 1) 随机威胁线：方向 × 起点，5 格窗口全部在盘内
        let (dr, dc) = [(0isize, 1isize), (1, 0), (1, 1), (1, -1)][rng.gen_range(0..4)];
        let (r0, c0) = {
            let d = side as isize;
            let (rlo, rhi) = if dr >= 0 {
                (0, d - 1 - dr * 4)
            } else {
                (4, d - 1)
            };
            let (clo, chi) = if dc >= 0 {
                (0, d - 1 - dc * 4)
            } else {
                (4, d - 1)
            };
            if rhi < rlo || chi < clo {
                continue 'retry;
            }
            (rng.gen_range(rlo..=rhi), rng.gen_range(clo..=chi))
        };
        let window: Vec<usize> = (0..5)
            .map(|i| ((r0 + dr * i) * side as isize + (c0 + dc * i)) as usize)
            .collect();
        // 威胁方按课题占格，落子顺序随机
        let mut threat_cells: Vec<usize> = match course {
            TacticalCourse::WinIn1 => {
                let hole = rng.gen_range(0..5);
                (0..5).filter(|&i| i != hole).map(|i| window[i]).collect()
            }
            // 中间三格（窗口两端即活三的两个开放端，构造上保证在盘内）
            TacticalCourse::OpenThree => (1..4).map(|i| window[i]).collect(),
        };
        for i in (1..threat_cells.len()).rev() {
            threat_cells.swap(i, rng.gen_range(0..=i));
        }

        // 2) 陪跑方随机散点（窗口外），黑白身份随机；
        //    黑先手守恒：黑方子数 = 白方子数 或 白方子数 + 1，且收尾后轮到防守方
        let a_is_black = rng.gen_bool(0.5);
        let filler_n = if a_is_black {
            threat_cells.len() - 1
        } else {
            threat_cells.len()
        };
        let mut filler_cells: Vec<usize> = Vec::with_capacity(filler_n);
        while filler_cells.len() < filler_n {
            let cell = rng.gen_range(0..plane);
            if !window.contains(&cell) && !filler_cells.contains(&cell) {
                filler_cells.push(cell);
            }
        }

        // 3) 黑白交替组装落子序列并 replay 验证
        let mut moves: Vec<usize> = Vec::with_capacity(threat_cells.len() + filler_n);
        let (first, second): (&[usize], &[usize]) = if a_is_black {
            (&threat_cells, &filler_cells) // 黑=威胁方先行
        } else {
            (&filler_cells, &threat_cells) // 白=威胁方后行
        };
        for i in 0..first.len().max(second.len()) {
            if let Some(&m) = first.get(i) {
                moves.push(m);
            }
            if let Some(&m) = second.get(i) {
                moves.push(m);
            }
        }
        let mut board = RulesBoard::from_obs(&empty_obs, 0, side);
        for &m in &moves {
            let (_, done) = board.step(m);
            if done {
                continue 'retry; // 陪跑散点意外成五（极罕见）
            }
        }
        let threat_color = u8::from(!a_is_black);
        let defender = 1 - threat_color;
        if board.to_play() != defender {
            continue 'retry;
        }
        let invariant_ok = match course {
            TacticalCourse::WinIn1 => {
                board.has_win_in_1(threat_color) && !board.has_win_in_1(defender)
            }
            TacticalCourse::OpenThree => {
                !board.has_win_in_1(threat_color) && !board.has_win_in_1(defender)
            }
        };
        if invariant_ok {
            return Some(moves);
        }
    }
    None
}
