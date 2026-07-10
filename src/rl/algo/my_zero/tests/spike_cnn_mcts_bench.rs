//! # Phase 1 风险 spike:CNN 前向 × MCTS sims 单步 wall-clock 实测
//!
//! **目的**(预注册协议见 `.doc/design/rl_phase1_image_plan.md` §S0):
//! 实测「CPU-only × 图像 CNN × MCTS」的单步 acting 成本,对照实时预算触发线
//! (16–33ms/步,见 `.issue/items/cpu_only_mcts_image_realtime_risk.md`),
//! 产出**去/留/改道**三态裁决。纯计时,不训练收敛、不接 env。
//!
//! ## 两条架构臂(卡出风险上下界)
//! - **flat-latent 臂**(现实路径 = MyZero 现架构):CNN 只进 representation h,
//!   dynamics/prediction 维持 flat MLP → acting 单步 = 1×CNN root + sims×MLP recurrent。
//! - **conv-recurrent 臂**(悲观路径 = EfficientZero 忠实版空间 latent):
//!   dynamics 也是卷积 → acting 单步 = 1×CNN root + sims×conv recurrent。
//!
//! ## 运行（手动档 bench，与其他 `*_bench.rs` 同约定）
//! ```bash
//! cargo test --release --features blas-mkl spike_cnn_mcts -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::{
    Adam, Conv2d, Graph, GraphError, Linear, Module, Optimizer, Var, VarActivationOps, VarLossOps,
    VarReduceOps, VarShapeOps,
};
use crate::rl::mcts::{
    ActionPayload, Dynamics, DynamicsModel, DynamicsOutput, MctsConfig, PuctPolicy, mcts_search,
};
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// 动作数(对齐 ALE/Pong-v5 的 Discrete(6))
const ACTION_DIM: usize = 6;
/// flat latent 维度(对齐 MyZero CartPole 口径)
const LATENT_DIM: usize = 64;
/// categorical support 原子数(对齐 MyZero SUPPORT = 2*20+1)
const SUPPORT_SIZE: usize = 41;
/// 帧堆叠数
const STACK: usize = 4;
/// 悲观臂空间 latent:[C, H, W]
const SP_C: usize = 32;
const SP_H: usize = 6;
const SP_W: usize = 6;
const SP_LEN: usize = SP_C * SP_H * SP_W;

#[test]
#[ignore = "manual: v0.26 Phase 1 风险 spike（CNN × MCTS 计时）"]
fn spike_cnn_mcts() -> Result<(), GraphError> {
    println!("=== Phase 1 spike:CNN × MCTS 单步 wall-clock(预注册协议 §S0)===");
    println!(
        "口径:{} build,warmup 后取中位数;触发线 16–33ms/步\n",
        if cfg!(debug_assertions) {
            "debug(⚠️ 结果无效,请用 --release)"
        } else {
            "release"
        }
    );

    // ------------------------------------------------------------------
    // 1. 组件级计时
    // ------------------------------------------------------------------
    println!("[1/4] 组件级前向中位数(warmup 20 + 计时 100)");
    let root84 = CnnRoot::new(84)?;
    let root42 = CnnRoot::new(42)?;
    let ms_root84 = root84.bench(20, 100);
    let ms_root42 = root42.bench(20, 100);
    println!(
        "  CNN root 前向  84×84×4: {ms_root84:8.3} ms(参数 {})",
        root84.param_count()
    );
    println!(
        "  CNN root 前向  42×42×4: {ms_root42:8.3} ms(参数 {})",
        root42.param_count()
    );

    let flat = FlatModel::new(84)?;
    let ms_mlp = flat.bench_recurrent(20, 100);
    println!("  MLP recurrent(g+f)   : {ms_mlp:8.3} ms(latent {LATENT_DIM}, hidden 128)");

    let conv_arm = ConvModel::new(84)?;
    let ms_convrec = conv_arm.bench_recurrent(20, 100);
    println!("  conv recurrent(g+f)  : {ms_convrec:8.3} ms(空间 latent {SP_C}×{SP_H}×{SP_W})");

    // ------------------------------------------------------------------
    // 2. 完整 acting 单步(真实 mcts_search,含树簿记)
    // ------------------------------------------------------------------
    println!("\n[2/4] 完整 acting 单步 = mcts_search(root + sims×recurrent + 树簿记)");
    println!("  中位数 over 30 步(warmup 3),单位 ms/步\n");
    println!(
        "  {:>6} | {:>14} | {:>14}",
        "sims", "flat-latent 臂", "conv-recurrent 臂"
    );
    println!("  {:->6}-+-{:->14}-+-{:->14}", "", "", "");

    let sims_grid: [u32; 4] = [2, 4, 20, 50];
    let mut flat_ms = Vec::new();
    let mut conv_ms = Vec::new();
    for &sims in &sims_grid {
        let f = bench_full_step(&flat, sims, 84);
        let c = bench_full_step(&conv_arm, sims, 84);
        println!("  {sims:>6} | {f:>14.3} | {c:>14.3}");
        flat_ms.push(f);
        conv_ms.push(c);
    }

    // 42×42 降档臂(仅 flat,判黄/红时的退档参考)
    let flat42 = FlatModel::new(42)?;
    let f42_s2 = bench_full_step(&flat42, 2, 42);
    let f42_s20 = bench_full_step(&flat42, 20, 42);
    println!("\n  42×42 降档(flat):sims=2 → {f42_s2:.3} ms,sims=20 → {f42_s20:.3} ms");

    // ------------------------------------------------------------------
    // 3. 训练吞吐(batch=8 CNN root 前向+反向+step)
    // ------------------------------------------------------------------
    println!("\n[3/4] 训练吞吐(batch=8,CNN root + policy/value CE,fwd+bwd+Adam step)");
    let ms_train84 = bench_train_step(84, 8)?;
    let ms_train42 = bench_train_step(42, 8)?;
    println!("  84×84: {ms_train84:8.2} ms/optimizer-step");
    println!("  42×42: {ms_train42:8.2} ms/optimizer-step");

    // ------------------------------------------------------------------
    // 4. 预注册判决(协议 §S0 判决规则,机器可读输出)
    // ------------------------------------------------------------------
    println!("\n[4/4] 预注册判决");
    let budget = 33.0_f64; // 触发线上限(商业游戏一帧 16–33ms 取宽口径)
    let flat_s20 = flat_ms[2];
    let flat_s4 = flat_ms[1];

    // 离线训练吞吐:120k env-steps,假设 1 train-step / env-step(MuZero 口径)
    // 单 env-step 总成本 ≈ acting(sims=20) + train step
    let per_step_ms = flat_s20 + ms_train84;
    let hours_120k = per_step_ms * 120_000.0 / 3_600_000.0;
    println!("  离线口径:120k env-steps ≈ {hours_120k:.1} h(acting sims=20 + 1 train/step,84×84)");

    let verdict = if flat_s20 <= budget {
        "GO(绿):flat 臂 sims=20 在实时预算内,图像线全速推进"
    } else if flat_s4 <= budget || hours_120k <= 24.0 {
        "STAY(黄):实时需 Gumbel 少 sim / acting-reanalyze 解耦,离线训练可行"
    } else if f42_s2 > budget && hours_120k > 72.0 {
        "REROUTE(红):触发条款一——载体收缩 Gomoku + planning-free 退路评估"
    } else {
        "STAY(黄,边缘):42×42 降档可行,按解耦架构记账"
    };
    println!("\n  [spike-verdict] {verdict}");
    println!(
        "  [spike-data] root84={ms_root84:.3} root42={ms_root42:.3} mlp_rec={ms_mlp:.3} \
         conv_rec={ms_convrec:.3} flat_s2={:.3} flat_s4={:.3} flat_s20={:.3} flat_s50={:.3} \
         conv_s2={:.3} conv_s20={:.3} train84={ms_train84:.2} train42={ms_train42:.2}",
        flat_ms[0], flat_ms[1], flat_ms[2], flat_ms[3], conv_ms[0], conv_ms[2]
    );
    Ok(())
}

// ============================================================================
// 计时工具
// ============================================================================

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// 完整 acting 单步:真实 mcts_search(每步换一张假图像,防缓存作弊)
fn bench_full_step<M>(model: &M, sims: u32, side: usize) -> f64
where
    for<'a> &'a M: Dynamics,
{
    let cfg = MctsConfig {
        num_simulations: sims,
        discount: 0.997,
        ..MctsConfig::default()
    };
    let policy = PuctPolicy::new();
    let mut rng = StdRng::seed_from_u64(42);
    let obs_len = STACK * side * side;
    let dyn_model = DynamicsModel::new(
        model,
        (0..ACTION_DIM).map(ActionPayload::Discrete).collect(),
        0.997,
    );

    let mut run = |n: usize, record: bool, out: &mut Vec<f64>| {
        for i in 0..n {
            let obs = fake_frame(obs_len, i as u64);
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

/// 确定性假图像([0,1] 均匀)
fn fake_frame(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.r#gen::<f32>()).collect()
}

/// categorical logits → 标量(softmax 期望,对齐 MyZero 解码成本)
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
// CNN root(h+f):EfficientZero-lite 卷积栈 → flat latent → 预测头
// ============================================================================

/// 持久化 root 推理子图:input [1,4,S,S] → conv 栈 → flatten → latent64 → policy/value
struct CnnRoot {
    graph: Graph,
    obs_in: Var,
    latent: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
    params: Vec<Var>,
    side: usize,
}

impl CnnRoot {
    /// side ∈ {84, 42}。栈:conv3x3s2p1(4→32) → conv3x3s2p1(32→64) → conv3x3s2p1(64→64)
    /// → flatten → linear→64;84 边长再加一层 s2 压空间(84→42→21→11→6)。
    fn new(side: usize) -> Result<Self, GraphError> {
        let graph = Graph::new_with_seed(42);
        graph.inference();
        let obs_in = graph.input(&Tensor::zeros(&[1, STACK, side, side]))?;

        let c1 = Conv2d::new(
            &graph,
            STACK,
            32,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            "c1",
        )?;
        let c2 = Conv2d::new(&graph, 32, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "c2")?;
        let c3 = Conv2d::new(&graph, 64, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "c3")?;

        let mut h = c1.forward(&obs_in).relu();
        h = c2.forward(&h).relu();
        h = c3.forward(&h).relu();
        let mut params = [c1.parameters(), c2.parameters(), c3.parameters()].concat();
        if side == 84 {
            // 84→42→21→11 后再压到 6
            let c4 = Conv2d::new(&graph, 64, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "c4")?;
            h = c4.forward(&h).relu();
            params.extend(c4.parameters());
        }
        let flat = h.flatten()?;
        let flat_dim = flat.value_expected_shape()[1];
        let fc_latent = Linear::new(&graph, flat_dim, LATENT_DIM, true, "fc_latent")?;
        let latent = fc_latent.forward(&flat);
        // 预测头 f(与 MyZero PredictionNet 同构)
        let fc1 = Linear::new(&graph, LATENT_DIM, 128, true, "pred_fc1")?;
        let fc_p = Linear::new(&graph, 128, ACTION_DIM, true, "pred_policy")?;
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
            .set_value(&Tensor::new(obs, &[1, STACK, self.side, self.side]))
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
        let obs_len = STACK * self.side * self.side;
        let mut samples = Vec::with_capacity(reps);
        for i in 0..(warmup + reps) {
            let obs = fake_frame(obs_len, i as u64);
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
// flat-latent 臂:CNN root + MLP recurrent(对齐 MyZero DynamicsNet/PredictionNet)
// ============================================================================

/// 持久化 MLP recurrent 子图:与 MyZero rec_infer 同构(含 latent min-max 归一化)
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
}

/// latent min-max 归一化(照抄 MyZero network.rs 口径,成本一致)
fn min_max_normalize(latent: &Var) -> Result<Var, GraphError> {
    let batch = latent.value_expected_shape()[0];
    let min_v = latent.amin(1).reshape(&[batch, 1])?;
    let max_v = latent.amax(1).reshape(&[batch, 1])?;
    let range = (&max_v - &min_v) + 1e-5_f32;
    Ok(&(latent - &min_v) / &range)
}

impl MlpRec {
    fn new(graph: &Graph) -> Result<Self, GraphError> {
        let latent_in = graph.input(&Tensor::zeros(&[1, LATENT_DIM]))?;
        let action_in = graph.input(&Tensor::zeros(&[1, ACTION_DIM]))?;
        // dynamics g
        let fc1 = Linear::new(graph, LATENT_DIM + ACTION_DIM, 128, true, "dyn_fc1")?;
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
        let pfc_p = Linear::new(graph, 128, ACTION_DIM, true, "rec_pred_policy")?;
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
        })
    }

    fn step(&self, state: &[f32], action_idx: usize) -> DynamicsOutput {
        self.latent_in
            .set_value(&Tensor::new(state, &[1, LATENT_DIM]))
            .expect("set latent 失败");
        let mut oh = vec![0.0; ACTION_DIM];
        oh[action_idx.min(ACTION_DIM - 1)] = 1.0;
        self.action_in
            .set_value(&Tensor::new(&oh, &[1, ACTION_DIM]))
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

struct FlatModel {
    root: CnnRoot,
    rec: MlpRec,
}

impl FlatModel {
    fn new(side: usize) -> Result<Self, GraphError> {
        let root = CnnRoot::new(side)?;
        let rec = MlpRec::new(&root.graph)?;
        Ok(Self { root, rec })
    }

    fn bench_recurrent(&self, warmup: usize, reps: usize) -> f64 {
        let mut rng = StdRng::seed_from_u64(7);
        let mut samples = Vec::with_capacity(reps);
        for i in 0..(warmup + reps) {
            let state: Vec<f32> = (0..LATENT_DIM).map(|_| rng.r#gen::<f32>()).collect();
            let t = Instant::now();
            let out = self.rec.step(&state, i % ACTION_DIM);
            std::hint::black_box(&out.reward);
            if i >= warmup {
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        median(samples)
    }
}

impl Dynamics for &FlatModel {
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

// ============================================================================
// conv-recurrent 臂(悲观):空间 latent [32,6,6] + 卷积 dynamics
// ============================================================================

/// 持久化 conv recurrent 子图:input [1,32+A,6,6] → conv×2 → 空间 next latent + 各头
struct ConvRec {
    graph: Graph,
    input: Var, // [1, SP_C+ACTION_DIM, SP_H, SP_W](latent 与 action 平面拼好后整体写入)
    next_latent_flat: Var,
    reward_logits: Var,
    continuation_logit: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
}

impl ConvRec {
    fn new(graph: &Graph) -> Result<Self, GraphError> {
        let in_c = SP_C + ACTION_DIM;
        let input = graph.input(&Tensor::zeros(&[1, in_c, SP_H, SP_W]))?;
        let c1 = Conv2d::new(
            graph,
            in_c,
            SP_C,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            "dc1",
        )?;
        let c2 = Conv2d::new(
            graph,
            SP_C,
            SP_C,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            "dc2",
        )?;
        let h = c1.forward(&input).relu();
        let next_latent = c2.forward(&h).relu();
        let next_latent_flat = next_latent.flatten()?; // [1, SP_LEN]
        // 头部(reward / continuation / prediction 均从 flat 出,对齐 EfficientZero 小头)
        let fc_r = Linear::new(graph, SP_LEN, SUPPORT_SIZE, true, "dr")?;
        let fc_c = Linear::new(graph, SP_LEN, 1, true, "dcont")?;
        let pfc1 = Linear::new(graph, SP_LEN, 128, true, "pf1")?;
        let pfc_p = Linear::new(graph, 128, ACTION_DIM, true, "pp")?;
        let pfc_v = Linear::new(graph, 128, SUPPORT_SIZE, true, "pv")?;
        let reward_logits = fc_r.forward(&next_latent_flat);
        let continuation_logit = fc_c.forward(&next_latent_flat);
        let ph = pfc1.forward(&next_latent_flat).relu();
        let policy = pfc_p.forward(&ph);
        let value_logits = pfc_v.forward(&ph);
        let sink = Var::concat(
            &[
                &next_latent_flat,
                &reward_logits,
                &continuation_logit,
                &policy,
                &value_logits,
            ],
            1,
        )?;
        Ok(Self {
            graph: graph.clone(),
            input,
            next_latent_flat,
            reward_logits,
            continuation_logit,
            policy,
            value_logits,
            sink,
        })
    }

    fn step(&self, state: &[f32], action_idx: usize) -> DynamicsOutput {
        // CPU 侧组装 [latent 32ch | action onehot 平面 6ch]
        let plane = SP_H * SP_W;
        let mut buf = vec![0.0f32; (SP_C + ACTION_DIM) * plane];
        buf[..SP_LEN].copy_from_slice(state);
        let a = action_idx.min(ACTION_DIM - 1);
        buf[SP_LEN + a * plane..SP_LEN + (a + 1) * plane].fill(1.0);
        self.input
            .set_value(&Tensor::new(&buf, &[1, SP_C + ACTION_DIM, SP_H, SP_W]))
            .expect("set conv input 失败");
        self.graph
            .forward(&self.sink)
            .expect("conv recurrent forward 失败");

        let next_state = self
            .next_latent_flat
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

/// 悲观臂 root:conv s2 ×4 → 空间 latent [32,6,6] → 预测头
struct ConvRoot {
    graph: Graph,
    obs_in: Var,
    latent_flat: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
    side: usize,
}

impl ConvRoot {
    fn new(side: usize) -> Result<Self, GraphError> {
        let graph = Graph::new_with_seed(43);
        graph.inference();
        let obs_in = graph.input(&Tensor::zeros(&[1, STACK, side, side]))?;
        // 84→42→21→11→6(通道 4→16→32→32→32)
        let c1 = Conv2d::new(
            &graph,
            STACK,
            16,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            "r1",
        )?;
        let c2 = Conv2d::new(&graph, 16, 32, (3, 3), (2, 2), (1, 1), (1, 1), true, "r2")?;
        let c3 = Conv2d::new(&graph, 32, 32, (3, 3), (2, 2), (1, 1), (1, 1), true, "r3")?;
        let c4 = Conv2d::new(&graph, 32, SP_C, (3, 3), (2, 2), (1, 1), (1, 1), true, "r4")?;
        let mut h = c1.forward(&obs_in).relu();
        h = c2.forward(&h).relu();
        h = c3.forward(&h).relu();
        h = c4.forward(&h).relu();
        let latent_flat = h.flatten()?; // [1, SP_LEN]
        let pfc1 = Linear::new(&graph, SP_LEN, 128, true, "rf1")?;
        let pfc_p = Linear::new(&graph, 128, ACTION_DIM, true, "rp")?;
        let pfc_v = Linear::new(&graph, 128, SUPPORT_SIZE, true, "rv")?;
        let ph = pfc1.forward(&latent_flat).relu();
        let policy = pfc_p.forward(&ph);
        let value_logits = pfc_v.forward(&ph);
        let sink = Var::concat(&[&latent_flat, &policy, &value_logits], 1)?;
        Ok(Self {
            graph,
            obs_in,
            latent_flat,
            policy,
            value_logits,
            sink,
            side,
        })
    }

    fn infer(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        self.obs_in
            .set_value(&Tensor::new(obs, &[1, STACK, self.side, self.side]))
            .expect("set obs 失败");
        self.graph
            .forward(&self.sink)
            .expect("conv root forward 失败");
        let latent = self
            .latent_flat
            .value()
            .unwrap()
            .unwrap()
            .data_as_slice()
            .to_vec();
        let policy = softmax_vec(&self.policy.value().unwrap().unwrap());
        let value = decode_categorical(&self.value_logits.value().unwrap().unwrap());
        (latent, policy, value)
    }
}

struct ConvModel {
    root: ConvRoot,
    rec: ConvRec,
}

impl ConvModel {
    fn new(side: usize) -> Result<Self, GraphError> {
        let root = ConvRoot::new(side)?;
        let rec = ConvRec::new(&root.graph)?;
        Ok(Self { root, rec })
    }

    fn bench_recurrent(&self, warmup: usize, reps: usize) -> f64 {
        let mut rng = StdRng::seed_from_u64(9);
        let mut samples = Vec::with_capacity(reps);
        for i in 0..(warmup + reps) {
            let state: Vec<f32> = (0..SP_LEN).map(|_| rng.r#gen::<f32>()).collect();
            let t = Instant::now();
            let out = self.rec.step(&state, i % ACTION_DIM);
            std::hint::black_box(&out.reward);
            if i >= warmup {
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        median(samples)
    }
}

impl Dynamics for &ConvModel {
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

// ============================================================================
// 训练吞吐:batch=8 的 CNN root fwd+bwd+Adam step
// ============================================================================

fn bench_train_step(side: usize, batch: usize) -> Result<f64, GraphError> {
    let graph = Graph::new_with_seed(44);
    graph.train();
    let obs_in = graph.input(&Tensor::zeros(&[batch, STACK, side, side]))?;

    let c1 = Conv2d::new(
        &graph,
        STACK,
        32,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        true,
        "t1",
    )?;
    let c2 = Conv2d::new(&graph, 32, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "t2")?;
    let c3 = Conv2d::new(&graph, 64, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "t3")?;
    let mut h = c1.forward(&obs_in).relu();
    h = c2.forward(&h).relu();
    h = c3.forward(&h).relu();
    let mut params = [c1.parameters(), c2.parameters(), c3.parameters()].concat();
    if side == 84 {
        let c4 = Conv2d::new(&graph, 64, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, "t4")?;
        h = c4.forward(&h).relu();
        params.extend(c4.parameters());
    }
    let flat = h.flatten()?;
    let flat_dim = flat.value_expected_shape()[1];
    let fc_latent = Linear::new(&graph, flat_dim, LATENT_DIM, true, "tl")?;
    let latent = fc_latent.forward(&flat);
    let pfc1 = Linear::new(&graph, LATENT_DIM, 128, true, "tp1")?;
    let pfc_p = Linear::new(&graph, 128, ACTION_DIM, true, "tpp")?;
    let pfc_v = Linear::new(&graph, 128, SUPPORT_SIZE, true, "tpv")?;
    let ph = pfc1.forward(&latent).relu();
    let policy = pfc_p.forward(&ph);
    let value_logits = pfc_v.forward(&ph);
    params.extend(fc_latent.parameters());
    params.extend(pfc1.parameters());
    params.extend(pfc_p.parameters());
    params.extend(pfc_v.parameters());

    // 均匀 policy 目标 + 中间原子 value 目标(纯计时,不关心收敛)
    let tp = Tensor::new(
        vec![1.0 / ACTION_DIM as f32; batch * ACTION_DIM],
        &[batch, ACTION_DIM],
    );
    let mut tv_flat = vec![0.0f32; batch * SUPPORT_SIZE];
    for b in 0..batch {
        tv_flat[b * SUPPORT_SIZE + SUPPORT_SIZE / 2] = 1.0;
    }
    let tv = Tensor::new(&tv_flat, &[batch, SUPPORT_SIZE]);
    let loss = &policy.cross_entropy(&tp)? + &value_logits.cross_entropy(&tv)?;

    let mut optimizer = Adam::new(&graph, &params, 1e-3);
    let obs_len = batch * STACK * side * side;

    let warmup = 3;
    let reps = 20;
    let mut samples = Vec::with_capacity(reps);
    for i in 0..(warmup + reps) {
        let obs = fake_frame(obs_len, 100 + i as u64);
        obs_in.set_value(&Tensor::new(&obs, &[batch, STACK, side, side]))?;
        let t = Instant::now();
        optimizer.zero_grad()?;
        let lv = loss.backward()?;
        optimizer.step()?;
        std::hint::black_box(lv);
        if i >= warmup {
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    Ok(median(samples))
}
