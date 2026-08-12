//! # Gomoku M0 风险 spike：棋盘形状 CNN × MCTS 单步 wall-clock 实测
//!
//! **目的**（对照 CPU 风险 issue 棋盘格；为训练闭环定 sims 预算）：
//! 实测「CPU-only × 棋盘 CNN root × sims × MLP recurrent」的单步 acting 成本，
//! 给 M1 训练闭环定 sims 预算。纯计时，不训练收敛、不接 env。
//!
//! 与 Pong spike（`spike_cnn_mcts_bench.rs`）的差异：
//! - obs = [1, 3, S, S]（黑/白/空三平面，当前方视角），S ∈ {9, 15}；
//! - |A| = S²（81 / 225），|A|≫sims 正是 Gumbel 复裁的 native 场景；
//! - 棋盘 CNN 不做 stride 降采样（棋盘空间结构不可丢），conv s1 ×3 → flatten → flat latent。
//!
//! ## 运行（手动档 bench，与其他 `*_bench.rs` 同约定）
//! ```bash
//! cargo test --release --features blas-mkl spike_gomoku_mcts -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::{Conv2d, Graph, GraphError, Linear, Module, Var, VarActivationOps, VarShapeOps};
use crate::rl::mcts::{
    ActionPayload, Dynamics, DynamicsModel, DynamicsOutput, MctsConfig, PuctPolicy, mcts_search,
};
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// 棋盘观测通道数（黑 / 白 / 空）
const OBS_C: usize = 3;
/// flat latent 维度（对齐 MyZero CartPole 口径）
const LATENT_DIM: usize = 64;
/// categorical support 原子数（对齐 MyZero SUPPORT = 2*20+1）
const SUPPORT_SIZE: usize = 41;

#[test]
#[ignore = "manual: Gomoku M0 风险 spike（棋盘 CNN × MCTS 计时）"]
fn spike_gomoku_mcts() -> Result<(), GraphError> {
    println!("=== Gomoku M0 spike：棋盘 CNN × MCTS 单步 wall-clock ===");
    println!(
        "口径:{} build,warmup 后取中位数\n",
        if cfg!(debug_assertions) {
            "debug(⚠️ 结果无效,请用 --release)"
        } else {
            "release"
        }
    );

    for &side in &[9usize, 15] {
        let action_dim = side * side;
        let model = BoardModel::new(side)?;
        println!("[棋盘 {side}×{side}，|A|={action_dim}]");
        let ms_root = model.root.bench(20, 100);
        let ms_rec = model.bench_recurrent(20, 100);
        println!(
            "  CNN root 前向 : {ms_root:8.3} ms（参数 {}）",
            model.root.param_count()
        );
        println!("  MLP recurrent : {ms_rec:8.3} ms（latent {LATENT_DIM}, hidden 128）");

        println!("  完整 acting 单步（mcts_search，中位数 over 30 步）:");
        for &sims in &[16u32, 50, 100, 200] {
            let ms = bench_full_step(&model, sims, side);
            let per_move_budget = ms / 1000.0;
            println!(
                "    sims={sims:>4} → {ms:>9.3} ms/步（一局 {} 步 ≈ {:.1} s）",
                action_dim / 2,
                per_move_budget * (action_dim / 2) as f64
            );
        }
        println!();
    }
    Ok(())
}

// ============================================================================
// 计时工具
// ============================================================================

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// 完整 acting 单步：真实 mcts_search（每步换一个假棋盘，防缓存作弊）
fn bench_full_step(model: &BoardModel, sims: u32, side: usize) -> f64 {
    let action_dim = side * side;
    let cfg = MctsConfig {
        num_simulations: sims,
        discount: 1.0,
        ..MctsConfig::default()
    };
    let policy = PuctPolicy::new();
    let mut rng = StdRng::seed_from_u64(42);
    let obs_len = OBS_C * side * side;
    let dyn_model = DynamicsModel::new(
        model,
        (0..action_dim).map(ActionPayload::Discrete).collect(),
        1.0,
    );

    let mut run = |n: usize, record: bool, out: &mut Vec<f64>| {
        for i in 0..n {
            let obs = fake_board(obs_len, i as u64);
            let t = Instant::now();
            let r = mcts_search(&dyn_model, &policy, &obs, &cfg, &mut rng);
            std::hint::black_box(&r.recommended);
            if record {
                out.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
    };
    let mut samples = Vec::new();
    run(3, false, &mut samples);
    run(30, true, &mut samples);
    median(samples)
}

/// 确定性假棋盘（0/1 平面近似）
fn fake_board(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| if rng.r#gen::<f32>() > 0.7 { 1.0 } else { 0.0 })
        .collect()
}

/// categorical logits → 标量（softmax 期望，对齐 MyZero 解码成本）
fn decode_categorical(logits: &Tensor) -> f32 {
    let probs = logits.softmax(1);
    let data = probs.data_as_slice();
    let half = (SUPPORT_SIZE / 2) as f32;
    data.iter()
        .enumerate()
        .map(|(i, p)| p * (i as f32 - half))
        .sum()
}

fn softmax_vec(logits: &Tensor) -> Vec<f32> {
    logits.softmax(1).data_as_slice().to_vec()
}

// ============================================================================
// 棋盘 root（h+f）：conv s1 ×3 → flat latent → 预测头
// ============================================================================

struct BoardRoot {
    graph: Graph,
    obs_in: Var,
    latent: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
    params: Vec<Var>,
    side: usize,
}

impl BoardRoot {
    /// 栈：conv3x3 s1 p1（3→32）→ conv（32→64）→ conv（64→64）→ flatten → linear → latent64
    fn new(side: usize) -> Result<Self, GraphError> {
        let action_dim = side * side;
        let graph = Graph::new_with_seed(42);
        graph.inference();
        let obs_in = graph.input(&Tensor::zeros(&[1, OBS_C, side, side]))?;

        let c1 = Conv2d::new(
            &graph,
            OBS_C,
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            "c1",
        )?;
        let c2 = Conv2d::new(&graph, 32, 64, (3, 3), (1, 1), (1, 1), (1, 1), true, "c2")?;
        let c3 = Conv2d::new(&graph, 64, 64, (3, 3), (1, 1), (1, 1), (1, 1), true, "c3")?;

        let mut h = c1.forward(&obs_in).relu();
        h = c2.forward(&h).relu();
        h = c3.forward(&h).relu();
        let mut params = [c1.parameters(), c2.parameters(), c3.parameters()].concat();
        let flat = h.flatten()?;
        let flat_dim = flat.value_expected_shape()[1];
        let fc_latent = Linear::new(&graph, flat_dim, LATENT_DIM, true, "fc_latent")?;
        let latent = fc_latent.forward(&flat);
        // 预测头 f（与 MyZero PredictionNet 同构）
        let fc1 = Linear::new(&graph, LATENT_DIM, 128, true, "pred_fc1")?;
        let fc_p = Linear::new(&graph, 128, action_dim, true, "pred_policy")?;
        let fc_v = Linear::new(&graph, 128, SUPPORT_SIZE, true, "pred_value")?;
        let ph = fc1.forward(&latent).relu();
        let policy = fc_p.forward(&ph);
        let value_logits = fc_v.forward(&ph);
        let sink = Var::concat(&[&latent, &policy, &value_logits], 1)?;
        params.extend(fc_latent.parameters());
        params.extend(fc1.parameters());
        params.extend(fc_p.parameters());
        params.extend(fc_v.parameters());

        Ok(Self {
            graph,
            obs_in,
            latent,
            policy,
            value_logits,
            sink,
            params,
            side,
        })
    }

    fn param_count(&self) -> usize {
        self.params
            .iter()
            .map(|p| {
                p.value()
                    .ok()
                    .flatten()
                    .map(|t| t.shape().iter().product::<usize>())
                    .unwrap_or(0)
            })
            .sum()
    }

    fn infer(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        self.obs_in
            .set_value(&Tensor::new(obs, &[1, OBS_C, self.side, self.side]))
            .expect("set obs 失败");
        self.graph.forward(&self.sink).expect("root forward 失败");
        let latent = self
            .latent
            .value()
            .unwrap()
            .unwrap()
            .data_as_slice()
            .to_vec();
        let policy = softmax_vec(&self.policy.value().unwrap().unwrap());
        let value = decode_categorical(&self.value_logits.value().unwrap().unwrap());
        (latent, policy, value)
    }

    fn bench(&self, warmup: usize, reps: usize) -> f64 {
        let obs_len = OBS_C * self.side * self.side;
        let mut samples = Vec::with_capacity(reps);
        for i in 0..(warmup + reps) {
            let obs = fake_board(obs_len, i as u64);
            let t = Instant::now();
            let out = self.infer(&obs);
            std::hint::black_box(&out);
            if i >= warmup {
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        median(samples)
    }
}

// ============================================================================
// MLP recurrent（对齐 MyZero DynamicsNet/PredictionNet，动作 onehot 维度 = S²）
// ============================================================================

struct MlpRec {
    graph: Graph,
    latent_in: Var,
    action_in: Var,
    next_latent: Var,
    reward_logits: Var,
    continuation_logit: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
    action_dim: usize,
}

/// latent min-max 归一化（照抄 MyZero network.rs 口径，成本一致）
fn min_max_normalize(latent: &Var) -> Result<Var, GraphError> {
    use crate::nn::VarReduceOps;
    let batch = latent.value_expected_shape()[0];
    let min_v = latent.amin(1).reshape(&[batch, 1])?;
    let max_v = latent.amax(1).reshape(&[batch, 1])?;
    let range = (&max_v - &min_v) + 1e-5_f32;
    Ok(&(latent - &min_v) / &range)
}

impl MlpRec {
    fn new(graph: &Graph, action_dim: usize) -> Result<Self, GraphError> {
        let latent_in = graph.input(&Tensor::zeros(&[1, LATENT_DIM]))?;
        let action_in = graph.input(&Tensor::zeros(&[1, action_dim]))?;
        // dynamics g
        let fc1 = Linear::new(graph, LATENT_DIM + action_dim, 128, true, "dyn_fc1")?;
        let fc_latent = Linear::new(graph, 128, LATENT_DIM, true, "dyn_latent")?;
        let fc_reward = Linear::new(graph, 128, SUPPORT_SIZE, true, "dyn_reward")?;
        let fc_cont = Linear::new(graph, 128, 1, true, "dyn_cont")?;
        let input = Var::concat(&[&latent_in, &action_in], 1)?;
        let h = fc1.forward(&input).relu();
        let next_latent = min_max_normalize(&fc_latent.forward(&h))?;
        let reward_logits = fc_reward.forward(&h);
        let continuation_logit = fc_cont.forward(&h);
        // prediction f
        let pfc1 = Linear::new(graph, LATENT_DIM, 128, true, "rec_pred_fc1")?;
        let pfc_p = Linear::new(graph, 128, action_dim, true, "rec_pred_policy")?;
        let pfc_v = Linear::new(graph, 128, SUPPORT_SIZE, true, "rec_pred_value")?;
        let ph = pfc1.forward(&next_latent).relu();
        let policy = pfc_p.forward(&ph);
        let value_logits = pfc_v.forward(&ph);
        let sink = Var::concat(
            &[
                &next_latent,
                &reward_logits,
                &continuation_logit,
                &policy,
                &value_logits,
            ],
            1,
        )?;
        Ok(Self {
            graph: graph.clone(),
            latent_in,
            action_in,
            next_latent,
            reward_logits,
            continuation_logit,
            policy,
            value_logits,
            sink,
            action_dim,
        })
    }

    fn step(&self, state: &[f32], action_idx: usize) -> DynamicsOutput {
        self.latent_in
            .set_value(&Tensor::new(state, &[1, LATENT_DIM]))
            .expect("set latent 失败");
        let mut oh = vec![0.0; self.action_dim];
        oh[action_idx.min(self.action_dim - 1)] = 1.0;
        self.action_in
            .set_value(&Tensor::new(&oh, &[1, self.action_dim]))
            .expect("set action 失败");
        self.graph
            .forward(&self.sink)
            .expect("recurrent forward 失败");

        let next_state = self
            .next_latent
            .value()
            .unwrap()
            .unwrap()
            .data_as_slice()
            .to_vec();
        let reward = decode_categorical(&self.reward_logits.value().unwrap().unwrap());
        let value = decode_categorical(&self.value_logits.value().unwrap().unwrap());
        let prior = softmax_vec(&self.policy.value().unwrap().unwrap());
        let cont_logit = self
            .continuation_logit
            .value()
            .unwrap()
            .unwrap()
            .data_as_slice()[0];
        let continuation = (1.0 / (1.0 + (-(cont_logit + 5.0)).exp())).clamp(0.0, 1.0);
        DynamicsOutput {
            next_state,
            reward,
            prior,
            value,
            terminal: continuation <= 0.05,
            continuation,
        }
    }
}

struct BoardModel {
    root: BoardRoot,
    rec: MlpRec,
}

impl BoardModel {
    fn new(side: usize) -> Result<Self, GraphError> {
        let root = BoardRoot::new(side)?;
        let rec = MlpRec::new(&root.graph, side * side)?;
        Ok(Self { root, rec })
    }

    fn bench_recurrent(&self, warmup: usize, reps: usize) -> f64 {
        let mut rng = StdRng::seed_from_u64(7);
        let mut samples = Vec::with_capacity(reps);
        for i in 0..(warmup + reps) {
            let state: Vec<f32> = (0..LATENT_DIM).map(|_| rng.r#gen::<f32>()).collect();
            let t = Instant::now();
            let out = self.rec.step(&state, i % self.rec.action_dim);
            std::hint::black_box(&out.reward);
            if i >= warmup {
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        median(samples)
    }
}

impl Dynamics for &BoardModel {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        self.root.infer(obs)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let ActionPayload::Discrete(idx) = action else {
            panic!("benchmark 只支持离散动作");
        };
        self.rec.step(state, *idx)
    }
}
