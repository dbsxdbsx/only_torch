//! `board.rs` 单元测试：negamax MC 回报 + 双人 MctsModel 根 mask / to_play 轮转。

use crate::nn::Graph;
use crate::rl::algo::my_zero::board::{
    BoardMctsModel, augment_game, negamax_mc_return, symmetry_perm,
};
use crate::rl::algo::my_zero::network::{MyZeroModel, ObsSpec};
use crate::rl::mcts::{ActionPayload, MctsModel};
use crate::rl::{GameOutcome, SelfPlayGame, SelfPlayStep};

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
