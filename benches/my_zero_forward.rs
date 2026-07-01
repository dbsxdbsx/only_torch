//! MyZero 前向路径微基准：MCTS 搜索里 `recurrent` / `initial_state` 的单次前向开销。
//!
//! 运行：`cargo bench --bench my_zero_forward`
//!
//! # 背景
//! CartPole 哨兵 wall-clock 同语义代码抖动 ±30%（112s vs 152s），不可用于验收。
//! 本 bench 把 model 前向从 env / eval / 训练噪声里隔离出来，给出**低方差 ns/call**，
//! 作为「持久化推理图」等前向优化的稳定验收标尺。
//!
//! # 维度
//! 对齐 CartPole-v1 MyZero：obs=4、action=2、latent=64（`ModelSettings` 默认）。
//! `recurrent` 是搜索热点（`sims` × 每个 env step 各一次），故单列。
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use only_torch::nn::Graph;
use only_torch::rl::algo::my_zero::MyZeroModel;
use only_torch::rl::mcts::{ActionPayload, Dynamics};

const OBS_DIM: usize = 4; // CartPole-v1
const ACTION_DIM: usize = 2;
const LATENT_DIM: usize = 64; // ModelSettings 默认

fn build_model() -> (Graph, MyZeroModel) {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, OBS_DIM, ACTION_DIM, LATENT_DIM).unwrap();
    (graph, model)
}

fn bench_my_zero_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_zero_forward");
    group.sample_size(100);

    // root 前向：obs → (latent, policy, value)。每次搜索 1 次。
    group.bench_function("initial_state_obs4_latent64", |b| {
        let (_g, model) = build_model();
        let m: &MyZeroModel = &model;
        let obs = vec![0.1f32, -0.2, 0.05, 0.3];
        b.iter(|| {
            black_box(m.initial_state(black_box(&obs)));
        });
    });

    // recurrent 前向：(latent, action) → DynamicsOutput。搜索热点（sims × 每步）。
    group.bench_function("recurrent_latent64_action2", |b| {
        let (_g, model) = build_model();
        let m: &MyZeroModel = &model;
        // 合法 latent（dynamics 输出经 min-max 归一化到 [0,1]）
        let state: Vec<f32> = (0..LATENT_DIM)
            .map(|i| i as f32 / LATENT_DIM as f32)
            .collect();
        let action = ActionPayload::Discrete(1);
        b.iter(|| {
            black_box(m.recurrent(black_box(&state), black_box(&action)));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_my_zero_forward);
criterion_main!(benches);
