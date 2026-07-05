//! `board.rs` 单元测试：negamax MC 回报 + 双人 MctsModel 根 mask / to_play 轮转
//! + 树内真规则（RulesBoard 规则等价性 / TrueRulesBoardModel 适配器）。

use crate::nn::Graph;
use crate::rl::algo::my_zero::board::{
    BoardMctsModel, RulesBoard, TrueRulesBoardModel, augment_game, board_rosmo_policy,
    negamax_mc_return, symmetry_perm, tactical_opening,
};
use crate::rl::algo::my_zero::network::{MyZeroModel, ObsSpec};
use crate::rl::mcts::{ActionPayload, MctsModel};
use crate::rl::{GameOutcome, GymEnv, SelfPlayGame, SelfPlayStep};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serial_test::serial;

fn mk_step(player: u8, reward: f32, terminated: bool) -> SelfPlayStep {
    SelfPlayStep {
        obs: vec![0.0f32; 12].into(),
        action: vec![0.0],
        policy_target: vec![0.25; 4],
        player,
        reward,
        root_value: Some(0.0),
        terminated,
        truncated: false,
        continuation: if terminated { 0.0 } else { 1.0 },
        extras: Default::default(),
    }
}

/// 黑（步 0,2）胜局：negamax 回报应沿轨迹交替 ±1。
#[test]
fn negamax_mc_return_alternates_signs_on_win() {
    // 黑 → 白 → 黑（终局，黑胜 reward=+1，落子方视角）
    let steps = vec![
        mk_step(0, 0.0, false),
        mk_step(1, 0.0, false),
        mk_step(0, 1.0, true),
    ];
    // pos=2：黑刚落子获胜 → +1
    assert_eq!(negamax_mc_return(&steps, 2), 1.0);
    // pos=1：白视角，下一手黑胜 → −1
    assert_eq!(negamax_mc_return(&steps, 1), -1.0);
    // pos=0：黑视角 → +1
    assert_eq!(negamax_mc_return(&steps, 0), 1.0);
}

/// 平局：全轨迹回报 0。
#[test]
fn negamax_mc_return_draw_is_zero() {
    let steps = vec![
        mk_step(0, 0.0, false),
        mk_step(1, 0.0, false),
        mk_step(0, 0.0, true), // 平局终局 reward=0
    ];
    for pos in 0..3 {
        assert_eq!(negamax_mc_return(&steps, pos), 0.0);
    }
}

/// D4 对称置换：恒等 / 双射 / 旋转 90° 的格点映射正确性（3×3 便于手算）。
#[test]
fn symmetry_perm_identity_bijection_and_rotation() {
    let side = 3;
    // sym=0 恒等
    assert_eq!(symmetry_perm(side, 0), (0..9).collect::<Vec<_>>());
    // 全部 8 个变换都是双射
    for sym in 0..8 {
        let mut seen = [false; 9];
        for &p in &symmetry_perm(side, sym) {
            assert!(!seen[p], "sym={sym} 置换应为双射");
            seen[p] = true;
        }
    }
    // sym=1 = 旋转 90°：(r,c) → (c, 2−r)。格点 (0,0)→(0,2)=2、(0,1)→(1,2)=5
    let rot = symmetry_perm(side, 1);
    assert_eq!(rot[0], 2);
    assert_eq!(rot[1], 5);
    // 中心不动
    assert_eq!(rot[4], 4);
}

/// 整局同构变换：obs 平面 / action / policy target 同套一个置换，
/// 不变量（reward / player / continuation / outcome）原样保留。
#[test]
fn augment_game_transforms_consistently() {
    let side = 3;
    let plane = side * side;
    // 黑在格点 1 落子：obs 通道 0 的格点 1 = 1、空位通道对应清零
    let mut obs = vec![0.0f32; 3 * plane];
    obs[1] = 1.0; // 己方平面
    for i in 0..plane {
        obs[2 * plane + i] = if i == 1 { 0.0 } else { 1.0 };
    }
    let mut policy = vec![0.0f32; plane];
    policy[1] = 0.8;
    policy[4] = 0.2;
    let game = SelfPlayGame {
        steps: vec![SelfPlayStep {
            obs: obs.into(),
            action: vec![1.0],
            policy_target: policy,
            player: 0,
            reward: 1.0,
            root_value: Some(0.5),
            terminated: true,
            truncated: false,
            continuation: 0.0,
            extras: Default::default(),
        }],
        outcome: GameOutcome::Win(0),
    };

    let perm = symmetry_perm(side, 1); // 旋转 90°：格点 1 → 5，格点 4 → 4
    let aug = augment_game(&game, &perm, side);
    let s = &aug.steps[0];
    let raw = s.obs.as_f32();
    assert_eq!(raw[5], 1.0, "己方棋子应随旋转移动到格点 5");
    assert_eq!(raw[1], 0.0);
    assert_eq!(raw[2 * plane + 5], 0.0, "空位平面应同步变换");
    assert_eq!(s.action[0] as usize, 5, "action 应同套置换");
    assert!((s.policy_target[5] - 0.8).abs() < 1e-6);
    assert!((s.policy_target[4] - 0.2).abs() < 1e-6, "中心不动");
    // 不变量
    assert_eq!(s.reward, 1.0);
    assert_eq!(s.player, 0);
    assert_eq!(aug.outcome, GameOutcome::Win(0));
}

/// RulesBoard 手算金测试：横向五连获胜、reward 落子方视角、to_play 轮转。
#[test]
fn rules_board_horizontal_win_and_turn_taking() {
    let side = 9;
    let empty = {
        let plane = side * side;
        let mut obs = vec![0.0f32; 3 * plane];
        obs[2 * plane..].fill(1.0);
        obs
    };
    let mut b = RulesBoard::from_obs(&empty, 0, side);
    assert_eq!(b.to_play(), 0);
    // 黑走 0..4 横排，白走第二行陪跑；黑第 5 手（action=4）应五连胜
    for i in 0..4 {
        let (r, t) = b.step(i); // 黑
        assert_eq!((r, t), (0.0, false));
        assert_eq!(b.to_play(), 1);
        let (r, t) = b.step(9 + i); // 白
        assert_eq!((r, t), (0.0, false));
        assert_eq!(b.to_play(), 0);
    }
    let (r, t) = b.step(4);
    assert_eq!((r, t), (1.0, true), "黑五连应胜");
    // 已占位在 legal_mask 中应为 false
    let legal = b.legal_mask();
    assert!(!legal[0] && !legal[4] && !legal[9]);
    assert!(legal[80]);
}

/// RulesBoard::from_obs 视角重建：白方视角 obs（通道 0=己方白）应还原绝对局面。
#[test]
fn rules_board_from_obs_white_perspective() {
    let side = 9;
    let plane = side * side;
    // 绝对局面：黑在 0，白在 1；当前轮白 → obs 通道 0=白(1)、通道 1=黑(0)
    let mut obs = vec![0.0f32; 3 * plane];
    obs[1] = 1.0; // 己方（白）在格点 1
    obs[plane] = 1.0; // 对方（黑）在格点 0
    for i in 0..plane {
        obs[2 * plane + i] = f32::from(i > 1);
    }
    let mut b = RulesBoard::from_obs(&obs, 1, side);
    assert_eq!(b.to_play(), 1);
    // 白落 10 后轮黑；黑视角 obs 通道 0 应含格点 0（黑子）
    b.step(10);
    assert_eq!(b.to_play(), 0);
    let next_obs = b.observation_flat();
    assert_eq!(next_obs[0], 1.0, "黑视角己方平面应含格点 0");
    assert_eq!(next_obs[plane + 1], 1.0, "黑视角对方平面应含格点 1");
    assert_eq!(next_obs[plane + 10], 1.0, "黑视角对方平面应含新落的 10");
}

/// would_win / has_win_in_1 手算金测试：黑四连的两个延长点均为一步胜，白无胜着。
#[test]
fn rules_board_would_win_detection() {
    let side = 9;
    let plane = side * side;
    let empty = {
        let mut obs = vec![0.0f32; 3 * plane];
        obs[2 * plane..].fill(1.0);
        obs
    };
    let mut b = RulesBoard::from_obs(&empty, 0, side);
    // 黑走 1..4 横排（格点 1,2,3,4）→ 四连活两头；白散点陪跑（不成线）
    for (i, w) in [(0usize, 20usize), (1, 40), (2, 60), (3, 78)] {
        b.step(1 + i); // 黑
        b.step(w); // 白
    }
    assert!(b.would_win(0, 0), "黑在 0 应成五");
    assert!(b.would_win(0, 5), "黑在 5 应成五");
    assert!(!b.would_win(0, 6), "黑在 6 不成五");
    assert!(b.has_win_in_1(0), "黑应有一步胜着");
    assert!(!b.has_win_in_1(1), "白不应有一步胜着");
}

/// tactical_opening 契约：生成的前缀 replay 后未终局、且「刚落子方」有一步胜着
/// （= 当前待走方处于必挡局面）；多 seed 覆盖生成器稳定性。
#[test]
fn tactical_opening_yields_must_block_position() {
    let side = 9;
    let plane = side * side;
    let empty = {
        let mut obs = vec![0.0f32; 3 * plane];
        obs[2 * plane..].fill(1.0);
        obs
    };
    for seed in 0..10u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let moves = tactical_opening(side, &mut rng)
            .unwrap_or_else(|| panic!("seed={seed} 生成器应在重试预算内出局面"));
        let mut b = RulesBoard::from_obs(&empty, 0, side);
        for (i, &a) in moves.iter().enumerate() {
            assert_eq!(b.to_play() as usize, i % 2, "前缀应黑白交替");
            let (_, done) = b.step(a);
            assert!(!done, "前缀 replay 不应终局");
        }
        let threat_maker = 1 - b.to_play();
        assert!(
            b.has_win_in_1(threat_maker),
            "seed={seed}: 刚落子方应有一步胜着（必挡局面）"
        );
    }
}

/// 规则等价性金测试：随机对局全程与 Python `Board` 逐步对照
/// （reward / terminal / legal_mask / observation_flat / current_player 五项全等）。
#[test]
#[serial]
fn rules_board_matches_python_board() {
    pyo3::Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        let mut rng = StdRng::seed_from_u64(2026_0705);
        for game in 0..3u64 {
            env.reset(Some(game));
            let side = 9;
            let obs0 = env.board_observation_flat();
            let mut rust_board = RulesBoard::from_obs(&obs0, env.current_player(), side);
            loop {
                // 双板状态对照
                let py_legal = env.legal_mask();
                assert_eq!(rust_board.legal_mask(), py_legal, "legal_mask 应一致");
                assert_eq!(rust_board.to_play(), env.current_player(), "执子方应一致");
                assert_eq!(
                    rust_board.observation_flat(),
                    env.board_observation_flat(),
                    "obs 应一致"
                );
                // 随机合法落子，双板同步走
                let choices: Vec<usize> = py_legal
                    .iter()
                    .enumerate()
                    .filter_map(|(a, &ok)| ok.then_some(a))
                    .collect();
                let action = choices[rng.gen_range(0..choices.len())];
                let (py_r, py_t) = env.board_step(action);
                let (rs_r, rs_t) = rust_board.step(action);
                assert_eq!(
                    (rs_r, rs_t),
                    (py_r, py_t),
                    "step 返回应一致（action={action}）"
                );
                if py_t {
                    break;
                }
            }
        }
        env.close();
    });
}

/// TrueRulesBoardModel：根/树内候选均按真规则 mask、to_play 轮转、
/// 连五终局边 terminal + reward=1 + discount=0（learned 路径对照见下一测试）。
#[test]
fn true_rules_model_masks_and_terminal() {
    let side = 9;
    let action_dim = side * side;
    let plane = action_dim;
    let graph = Graph::new_with_seed(7);
    let model = MyZeroModel::new_with_spec(&graph, ObsSpec::Flat(3 * plane), action_dim, 16)
        .expect("模型构造失败");

    // 局面：黑已有 0..4 横排差一子（0..=3），白 9..12；轮黑，黑走 4 即连五
    let mut obs = vec![0.0f32; 3 * plane];
    for i in 0..4 {
        obs[i] = 1.0; // 己方（黑）
        obs[plane + 9 + i] = 1.0; // 对方（白）
    }
    for i in 0..plane {
        let occupied = i < 4 || (9..13).contains(&i);
        obs[2 * plane + i] = f32::from(!occupied);
    }

    let bm = TrueRulesBoardModel::new(&model, &obs, 0, side);
    let root = bm.root(&obs);
    assert_eq!(root.to_play, 0);
    assert_eq!(
        root.candidates.len(),
        plane - 8,
        "根候选应 = 空位数（真规则 mask）"
    );
    let prior_sum: f32 = root.candidates.policy_priors().iter().sum();
    assert!((prior_sum - 1.0).abs() < 1e-5, "根 prior 应归一化");

    // 树内走 4：黑连五 → 终局边
    let rec = bm.recurrent(&root.state, &ActionPayload::Discrete(4));
    assert!(rec.terminal, "连五应终局");
    assert_eq!(rec.reward, 1.0, "落子方胜 reward=+1");
    assert_eq!(rec.discount, 0.0, "终局边 discount=0");
    assert!(rec.candidates.is_empty(), "终局无候选");

    // 树内走 80（无关处）：轮白、候选按真规则 mask（又少一空位）
    let rec2 = bm.recurrent(&root.state, &ActionPayload::Discrete(80));
    assert!(!rec2.terminal);
    assert_eq!(rec2.to_play, 1, "recurrent 应轮转执子方");
    assert_eq!(
        rec2.candidates.len(),
        plane - 9,
        "树内候选应按真规则 mask（非全量）"
    );
}

/// 棋盘 ROSMO 刷新分布：合法支撑（非法位恒 0）、归一化、有限值。
///
/// 用 2×2 迷你棋盘（|A|=4 < top-m）覆盖「全合法位 look-ahead」路径；
/// 语义正确性（negamax q 翻转）由 ③ 臂实测裁决，这里锁契约不锁数值。
#[test]
fn board_rosmo_policy_legal_support_and_normalized() {
    let action_dim = 4usize;
    let plane = action_dim;
    let graph = Graph::new_with_seed(11);
    let model = MyZeroModel::new_with_spec(&graph, ObsSpec::Flat(3 * plane), action_dim, 16)
        .expect("模型构造失败");

    // 位置 1 有己方子、2 有对方子 → 合法位 {0, 3}
    let mut obs = vec![0.0f32; 3 * plane];
    obs[1] = 1.0;
    obs[plane + 2] = 1.0;
    for i in 0..plane {
        obs[2 * plane + i] = f32::from(i == 0 || i == 3);
    }

    let policy = board_rosmo_policy(&model, &obs, action_dim);
    assert_eq!(policy.len(), action_dim);
    assert_eq!(policy[1], 0.0, "已占位（己方）应为 0");
    assert_eq!(policy[2], 0.0, "已占位（对方）应为 0");
    let z: f32 = policy.iter().sum();
    assert!((z - 1.0).abs() < 1e-5, "应归一化，got {z}");
    assert!(policy.iter().all(|p| p.is_finite() && *p >= 0.0));
    assert!(policy[0] > 0.0 && policy[3] > 0.0, "合法位应有非零质量");
}

/// 根节点候选应只含合法动作、ActionId = 棋盘位置、prior 归一化；
/// recurrent 应轮转 to_play 且树内不 mask（全量候选）。
#[test]
fn board_model_masks_root_and_alternates_to_play() {
    let action_dim = 4usize; // 2×2 迷你"棋盘"
    let obs_len = 3 * action_dim;
    let graph = Graph::new_with_seed(7);
    let model = MyZeroModel::new_with_spec(&graph, ObsSpec::Flat(obs_len), action_dim, 16)
        .expect("模型构造失败");

    // 位置 1、3 已占
    let legal = vec![true, false, true, false];
    let bm = BoardMctsModel::new(&model, legal, 1, action_dim);
    let obs = vec![0.0f32; obs_len];

    let root = bm.root(&obs);
    assert_eq!(root.to_play, 1, "根 to_play 应为传入执子方");
    assert_eq!(root.state.1, 1);
    assert_eq!(root.candidates.len(), 2, "根候选应只含合法位");
    let ids: Vec<usize> = root
        .candidates
        .candidates
        .iter()
        .map(|c| c.id.index())
        .collect();
    assert_eq!(ids, vec![0, 2], "ActionId 应保持棋盘位置槽位");
    let prior_sum: f32 = root.candidates.policy_priors().iter().sum();
    assert!(
        (prior_sum - 1.0).abs() < 1e-5,
        "根 prior 应归一化，got {prior_sum}"
    );

    // 树内推演：to_play 轮转，候选全量（learned dynamics 不 mask）
    let rec = bm.recurrent(&root.state, &ActionPayload::Discrete(0));
    assert_eq!(rec.to_play, 0, "recurrent 应轮转执子方");
    assert_eq!(rec.state.1, 0);
    if !rec.terminal {
        assert_eq!(rec.candidates.len(), action_dim, "树内候选应全量");
    }
    let rec2 = bm.recurrent(&rec.state, &ActionPayload::Discrete(1));
    assert_eq!(rec2.to_play, 1, "二层应轮转回原执子方");
}
