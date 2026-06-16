//! EfficientZero 五子棋 learned-model 双人 self-play 示例（Phase 3）
//!
//! **smoke 级实现**：首次跑通**双人 learned-model**路径——MCTS 全程在 learned latent 推演，
//! 真实棋盘（`python/gym_env/gomoku/board.py`）仅用于交互 + 终局 + reward。两人交替通过
//! 自定义 `MctsModel` 在 State 尾部携带 `to_play`、每次 `recurrent` 翻转，触发内核 negamax backup。
//! 忠实达标（vs naive3 胜率）留后续，本文件只验「self-play 闭环、loss 有限、无 panic」。
//!
//! ```bash
//! pip install -e python/gym_env   # 或确保 python/ 在 sys.path
//! cargo run --example efficientzero_gomoku --release
//! SMOKE=1 cargo run --example efficientzero_gomoku  # 管线验证（小棋盘 3 局 + 1 训练）
//! ```

#[path = "../cartpole/model.rs"]
mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::mcts::{
    ActionPayload, Dynamics, MctsConfig, MctsModel, PuctPolicy, RecurrentOut, RootOut, mcts_search,
};
use only_torch::rl::{GameOutcome, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

const BOARD_SIZE: usize = 6;
const WIN_LENGTH: usize = 5;
const NUM_ACTIONS: usize = BOARD_SIZE * BOARD_SIZE;
const OBS_DIM: usize = 3 * BOARD_SIZE * BOARD_SIZE;

/// 双人 learned-model 适配器：State = `[latent || to_play]`，`recurrent` 翻转 to_play。
///
/// 与库层 `DynamicsModel`（写死 `to_play=0`）的区别：本适配器让 `to_play` 按搜索深度奇偶
/// 翻转，从而触发 `ChildStat.to_play != parent` 的 negamax 视角翻转（零和博弈）。
struct GomokuTwoPlayerModel<'a> {
    model: &'a EzModel,
    latent_dim: usize,
    /// 当前真实根局面的合法掩码（长度 NUM_ACTIONS）：root 处把非法着法 prior 置 0。
    root_legal: Vec<bool>,
}

impl<'a> GomokuTwoPlayerModel<'a> {
    fn all_actions() -> Vec<ActionPayload> {
        (0..NUM_ACTIONS).map(ActionPayload::Discrete).collect()
    }
}

impl MctsModel for GomokuTwoPlayerModel<'_> {
    type State = Vec<f32>;

    fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
        let (latent, mut prior, value) = self.model.initial_state(obs);
        // root：非法着法 prior 置 0 并归一化（树内深层不再依赖真实合法性）。
        let mut sum = 0.0;
        for (i, p) in prior.iter_mut().enumerate() {
            if !self.root_legal.get(i).copied().unwrap_or(true) {
                *p = 0.0;
            }
            sum += *p;
        }
        if sum > 1e-8 {
            for p in prior.iter_mut() {
                *p /= sum;
            }
        }
        let mut state = latent;
        state.push(0.0); // to_play = 0（根方）
        RootOut {
            state,
            prior,
            value,
            candidate_actions: Self::all_actions(),
            to_play: 0,
        }
    }

    fn recurrent(&self, state: &Self::State, action: &ActionPayload) -> RecurrentOut<Self::State> {
        let parent_to_play = state.get(self.latent_dim).copied().unwrap_or(0.0) as u8;
        let child_to_play = 1 - parent_to_play;
        let out = self.model.recurrent(&state[..self.latent_dim], action);
        let mut next_state = out.next_state;
        next_state.push(child_to_play as f32);
        RecurrentOut {
            state: next_state,
            reward: out.reward,
            value: out.value,
            prior: out.prior,
            candidate_actions: Self::all_actions(),
            terminal: false,
            to_play: child_to_play,
            discount: 1.0, // 零和博弈：不折扣，negamax 翻转视角
        }
    }
}

/// 真实棋盘（pyo3）封装：仅做交互 / 终局 / reward，不参与 MCTS 树内推演。
struct Board<'py> {
    obj: Bound<'py, PyAny>,
}

impl<'py> Board<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        // 确保 python/ 在 sys.path（无需 pip install -e 也能 import）。
        let sys = py.import("sys")?;
        sys.getattr("path")?.call_method1("insert", (0, "python"))?;
        let board_mod = py.import("gym_env.gomoku.board")?;
        let obj = board_mod
            .getattr("Board")?
            .call1((BOARD_SIZE, WIN_LENGTH))?;
        Ok(Self { obj })
    }

    fn reset(&self) {
        self.obj.call_method0("reset").unwrap();
    }

    fn obs(&self) -> Vec<f32> {
        self.obj
            .call_method0("observation_flat")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap()
    }

    fn legal_mask(&self) -> Vec<bool> {
        self.obj
            .call_method0("legal_mask")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap()
    }

    fn to_play(&self) -> u8 {
        self.obj
            .getattr("to_play")
            .unwrap()
            .extract::<i64>()
            .unwrap() as u8
    }

    /// 落子，返回 (reward_for_mover, terminal)。
    fn step(&self, action: usize) -> (f32, bool) {
        self.obj
            .call_method1("step", (action as i64,))
            .unwrap()
            .extract()
            .unwrap()
    }

    fn winner(&self) -> Option<i64> {
        self.obj.getattr("winner").unwrap().extract().unwrap()
    }
}

/// 自对弈一局：交替落子，返回带 player 标注的轨迹（value target = 零和 outcome z）。
fn self_play_one_game(
    board: &Board,
    model: &EzModel,
    latent_dim: usize,
    mcts_cfg: &MctsConfig,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    board.reset();
    let mut steps: Vec<SelfPlayStep> = Vec::new();

    loop {
        let obs = board.obs();
        let mover = board.to_play();
        let two_player = GomokuTwoPlayerModel {
            model,
            latent_dim,
            root_legal: board.legal_mask(),
        };
        let result = mcts_search(&two_player, &PuctPolicy::new(), &obs, mcts_cfg, rng);
        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        steps.push(SelfPlayStep {
            obs,
            action: vec![action_idx as f32],
            policy_target: result.learn_policy,
            player: mover,
            reward: 0.0,
            root_value: Some(0.0),
            terminated: false,
            extras: Default::default(),
        });

        let (reward, terminal) = board.step(action_idx);
        let last = steps.last_mut().unwrap();
        last.reward = reward;
        last.terminated = terminal;
        if terminal {
            break;
        }
    }

    // 终局后回填零和 value target z（从每步落子方视角）。
    let winner = board.winner();
    for s in steps.iter_mut() {
        let z = match winner {
            Some(w) if w >= 0 => {
                if w as u8 == s.player {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => 0.0, // 平局 / 未定
        };
        s.root_value = Some(z);
    }
    steps
}

/// batch 训练：value target 用零和 outcome z（存于 root_value），reward 用落子方即时奖励。
fn train_batch(
    model: &EzModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let valid: Vec<&SelfPlayGame> = games.iter().filter(|g| g.steps.len() >= 2).collect();
    if valid.is_empty() {
        return Ok(0.0);
    }
    let batch_size = valid.len() as f32;
    let mut total = 0.0;
    optimizer.zero_grad()?;

    for game in &valid {
        let steps = &game.steps;
        let len = steps.len();
        let t = rng.gen_range(0..len);
        let actual_k = k_unroll.min(len - 1 - t).max(0);

        let uniform = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let target_policies: Vec<Vec<f32>> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].policy_target.clone()
                } else {
                    uniform.clone()
                }
            })
            .collect();
        let target_values: Vec<f32> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].root_value.unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .collect();
        let target_rewards: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].reward
                } else {
                    0.0
                }
            })
            .collect();
        let actions: Vec<usize> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].action[0] as usize
                } else {
                    0
                }
            })
            .collect();
        let next_obs: Vec<Option<Vec<f32>>> = (0..actual_k).map(|_| None).collect();

        let obs_t = &steps[t].obs;
        let loss = model.train_unroll(
            obs_t,
            &actions,
            &target_policies,
            &target_values,
            &target_rewards,
            &next_obs,
            0.0,
        )? * (1.0 / batch_size);
        total += loss.backward()?;
    }

    optimizer.step()?;
    Ok(total)
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let latent_dim = 64;
    let lr = 0.02;
    let num_sim = if smoke { 16 } else { 60 };
    let max_games = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500)
    };

    Python::attach(|py| {
        let board = Board::new(py).expect(
            "创建 Gomoku Board 失败：请确保 python/ 在 sys.path（或 pip install -e python/gym_env）",
        );
        println!(
            "[EZ Gomoku smoke] {BOARD_SIZE}x{BOARD_SIZE} win={WIN_LENGTH} obs_dim={OBS_DIM} actions={NUM_ACTIONS} num_sim={num_sim}（双人 learned-model + negamax）"
        );

        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, OBS_DIM, NUM_ACTIONS, latent_dim, false)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(2000);
        let mut rng = StdRng::seed_from_u64(42);
        let mut len_hist: VecDeque<usize> = VecDeque::with_capacity(50);

        for g in 0..max_games {
            let t0 = std::time::Instant::now();
            let mcts_cfg = MctsConfig {
                num_simulations: num_sim,
                temperature: 1.0,
                discount: 1.0,
                ..MctsConfig::default()
            };
            let steps = self_play_one_game(&board, &model, latent_dim, &mcts_cfg, &mut rng);
            let game_len = steps.len();
            let winner = board.winner();
            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            let mut avg_loss = 0.0;
            if buffer.len() >= 2 {
                let n_trains = if smoke { 1 } else { 8 };
                let mut loss_sum = 0.0;
                for _ in 0..n_trains {
                    let games = buffer.sample(4, &mut rng);
                    loss_sum += train_batch(&model, &mut optimizer, &games, 3, &mut rng)?;
                }
                avg_loss = loss_sum / n_trains as f32;
                if smoke {
                    assert!(avg_loss.is_finite(), "SMOKE: loss={avg_loss} 非有限");
                }
            }

            len_hist.push_back(game_len);
            if len_hist.len() > 50 {
                len_hist.pop_front();
            }
            println!(
                "Game {:4}: len={:3} winner={:?} loss={:.4} t={:.2}s",
                g + 1,
                game_len,
                winner,
                avg_loss,
                t0.elapsed().as_secs_f32()
            );
        }

        if smoke {
            println!("[SMOKE] EZ Gomoku 双人 learned-model 管线验证通过");
        }
        Ok(())
    })
}
