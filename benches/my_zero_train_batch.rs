//! MyZero 训练侧 batch 微基准：确认「逐样本 forward+backward」vs「batch 一次」在
//! tiny-MLP / CartPole 维度下到底省不省框架开销（measure-first，不改主路径）。
//!
//! 运行：`cargo bench --bench my_zero_train_batch`
//!
//! # 背景
//! 搜索前向优化落地后，`train_step`（~26.9s）成为哨兵最大桶。现状是逐样本：
//! `train_batch_size=8` 个 position 各自 `train_unroll`+`backward`（梯度累积），
//! 全程 `[1,X]` tiny matmul，无 batch 维度向量化。
//!
//! 本 bench 把「同一套 h/g/f 栈」隔离出来，对比：
//! - `per_sample_x8`：8 次 `[1,X]` forward+backward（逐样本累积，= 现状）
//! - `batched_x8`：1 次 `[8,X]` forward+backward（batch 化目标）
//! 给出低方差 ns/iter，判断 batch 化训练是否值得立项。
//!
//! # 覆盖范围（保守下界）
//! h/g/f **base** 栈：repr→pred + K×(dynamics→pred) + categorical CE/MSE loss + backward，
//! 维度对齐 CartPole：obs=4 / action=2 / latent=64 / hidden=128 / support=41 / K=5。
//! **不含** consistency / reconstruction（它们只增加节点数 → batch 化收益更大），
//! 故本 bench 是 batch 化收益的**下界**。
//!
//! # min-max 归一化的 batch 一致性
//! [`min_max`] 是 batch-general 版：`amin(1)/amax(1)` 逐样本沿特征维、`reshape([B,1])` +
//! `repeat([1,dim])`。B=1 时与 `network.rs::min_max_normalize` 的 `reshape([1,1])` 逐 bit 一致，
//! 且各行独立 → batch 前向逐行等价于逐样本。
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use only_torch::nn::{Graph, Linear, Var, VarActivationOps, VarLossOps, VarReduceOps, VarShapeOps};
use only_torch::tensor::Tensor;

const OBS: usize = 4;
const ACTION: usize = 2;
const LATENT: usize = 64;
const HIDDEN: usize = 128;
const SUPPORT: usize = 41; // 2*20+1，对齐 MyZero categorical support
const K: usize = 5; // k_unroll 默认

/// h/g/f base 栈的全部 Linear 层（参数持久，跨迭代复用）。
struct Net {
    repr1: Linear,
    repr2: Linear,
    pred1: Linear,
    pred_p: Linear,
    pred_v: Linear,
    dyn1: Linear,
    dyn_l: Linear,
    dyn_r: Linear,
    dyn_c: Linear,
}

impl Net {
    fn new(g: &Graph) -> Self {
        Self {
            repr1: Linear::new(g, OBS, HIDDEN, true, "repr1").unwrap(),
            repr2: Linear::new(g, HIDDEN, LATENT, true, "repr2").unwrap(),
            pred1: Linear::new(g, LATENT, HIDDEN, true, "pred1").unwrap(),
            pred_p: Linear::new(g, HIDDEN, ACTION, true, "pred_p").unwrap(),
            pred_v: Linear::new(g, HIDDEN, SUPPORT, true, "pred_v").unwrap(),
            dyn1: Linear::new(g, LATENT + ACTION, HIDDEN, true, "dyn1").unwrap(),
            dyn_l: Linear::new(g, HIDDEN, LATENT, true, "dyn_l").unwrap(),
            dyn_r: Linear::new(g, HIDDEN, SUPPORT, true, "dyn_r").unwrap(),
            dyn_c: Linear::new(g, HIDDEN, 1, true, "dyn_c").unwrap(),
        }
    }
}

/// batch-general min-max 归一化（B=1 与 network.rs 原实现逐 bit 一致）。
fn min_max(latent: &Var, batch: usize, dim: usize) -> Var {
    let min_v = latent.amin(1).reshape(&[batch, 1]).unwrap();
    let max_v = latent.amax(1).reshape(&[batch, 1]).unwrap();
    let range = (&max_v - &min_v) + 1e-5f32;
    let min_b = min_v.repeat(&[1, dim]).unwrap();
    let range_b = range.repeat(&[1, dim]).unwrap();
    &(latent - &min_b) / &range_b
}

/// 构建一次 K 步 unroll 的总 loss（batch=`b` 行），返回可 backward 的 loss Var。
fn forward_loss(net: &Net, g: &Graph, b: usize) -> Var {
    // 输入 + 目标（固定值，只为触发真实 forward/backward 计算量）
    let obs = Tensor::new(&vec![0.1f32; b * OBS], &[b, OBS]);
    let tp = Tensor::new(&vec![1.0 / ACTION as f32; b * ACTION], &[b, ACTION]);
    let tv = Tensor::new(&vec![1.0 / SUPPORT as f32; b * SUPPORT], &[b, SUPPORT]);
    let tc = Tensor::new(&vec![1.0f32; b], &[b, 1]);
    let mut oh = vec![0.0f32; b * ACTION];
    for r in 0..b {
        oh[r * ACTION] = 1.0; // 每行 action=0 的 onehot
    }
    let oh_tensor = Tensor::new(&oh, &[b, ACTION]);

    // repr → latent
    let latent = min_max(
        &net.repr2.forward(net.repr1.forward(&obs).relu()),
        b,
        LATENT,
    );

    // pred(latent) → policy/value（k=0）
    let h = net.pred1.forward(&latent).relu();
    let mut total = net.pred_p.forward(&h).cross_entropy(&tp).unwrap();
    total = &total + &net.pred_v.forward(&h).cross_entropy(&tv).unwrap();

    // K 步 unroll：dyn → pred
    let mut lat = latent;
    for _ in 0..K {
        let oh_var = g.input(&oh_tensor).unwrap();
        let inp = Var::concat(&[&lat, &oh_var], 1).unwrap();
        let dh = net.dyn1.forward(&inp).relu();
        let next = min_max(&net.dyn_l.forward(&dh), b, LATENT);
        let reward = net.dyn_r.forward(&dh);
        let cont = net.dyn_c.forward(&dh);

        let ph = net.pred1.forward(&next).relu();
        total = &total + &net.pred_p.forward(&ph).cross_entropy(&tp).unwrap();
        total = &total + &net.pred_v.forward(&ph).cross_entropy(&tv).unwrap();
        total = &total + &reward.cross_entropy(&tv).unwrap();
        total = &total + &cont.sigmoid().mse_loss(&tc).unwrap();

        lat = next.scale_gradient(0.5); // canonical MuZero hidden ×0.5
    }
    total
}

fn bench_train_batch(c: &mut Criterion) {
    let mut grp = c.benchmark_group("my_zero_train_batch");
    grp.sample_size(50);

    let g = Graph::new_with_seed(42);
    let net = Net::new(&g);

    // 现状：8 个样本逐个 forward+backward（梯度累积）
    grp.bench_function("per_sample_x8", |bch| {
        bch.iter(|| {
            g.zero_grad().unwrap();
            for _ in 0..8 {
                let loss = forward_loss(&net, &g, 1);
                black_box(loss.backward().unwrap());
            }
        });
    });

    // 目标：1 次 [8,X] batch forward+backward
    grp.bench_function("batched_x8", |bch| {
        bch.iter(|| {
            g.zero_grad().unwrap();
            let loss = forward_loss(&net, &g, 8);
            black_box(loss.backward().unwrap());
        });
    });

    // 更大 batch，观察规模效应（每次 optimizer step 覆盖更多 position）
    grp.bench_function("per_sample_x32", |bch| {
        bch.iter(|| {
            g.zero_grad().unwrap();
            for _ in 0..32 {
                let loss = forward_loss(&net, &g, 1);
                black_box(loss.backward().unwrap());
            }
        });
    });
    grp.bench_function("batched_x32", |bch| {
        bch.iter(|| {
            g.zero_grad().unwrap();
            let loss = forward_loss(&net, &g, 32);
            black_box(loss.backward().unwrap());
        });
    });

    grp.finish();
}

criterion_group!(benches, bench_train_batch);
criterion_main!(benches);
