/// 反向传播性能基准测试
///
/// 测量不同节点类型在反向传播中的耗时，用于量化 GradResult 重构前后的差异。
/// 运行方式：`cargo bench --bench backward`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::nn::{Graph, Init, Linear, Module, Optimizer, SGD, VarActivationOps, VarLossOps};
use only_torch::tensor::Tensor;

// ===================== Add 链路 benchmark =====================

/// 构建 Add 链路图：p + p + p + ... + p → loss
///
/// 用于测量 Add 节点反向传播的 clone 开销。
/// GradResult::PassThrough 优化后，无广播时应接近零分配。
fn bench_add_chain_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_chain_backward");
    group.sample_size(20);

    // (名称, 形状, 链长)
    let configs: &[(&str, &[usize], usize)] = &[
        ("small_4adds", &[8, 32], 4),
        ("medium_8adds", &[32, 128], 8),
        ("large_16adds", &[64, 256], 16),
    ];

    for &(name, shape, chain_len) in configs {
        let graph = Graph::new();
        let p = graph.parameter(shape, Init::Normal { mean: 0.0, std: 0.1 }, "p").unwrap();
        let target = graph.input(&Tensor::zeros(shape)).unwrap();

        // 构建 Add 链：p + p + p + ...
        let mut result = p.clone();
        for _ in 1..chain_len {
            result = &result + &p;
        }
        let loss = result.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);

        let mut opt = SGD::new(&graph, &[p.clone()], 0.001);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                loss.forward().unwrap();
                opt.zero_grad().unwrap();
                let _ = loss.backward().unwrap();
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

// ===================== Negate 链路 benchmark =====================

/// 构建 Negate 链路图：-(-(-p)) → loss
///
/// 用于测量 Negate 节点反向传播的分配开销。
/// GradResult::Negated 优化后，累加时应为零分配。
fn bench_negate_chain_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("negate_chain_backward");
    group.sample_size(20);

    let configs: &[(&str, &[usize], usize)] = &[
        ("small_4neg", &[8, 32], 4),
        ("medium_8neg", &[32, 128], 8),
        ("large_16neg", &[64, 256], 16),
    ];

    for &(name, shape, chain_len) in configs {
        let graph = Graph::new();
        let p = graph.parameter(shape, Init::Normal { mean: 0.0, std: 0.1 }, "p").unwrap();
        let target = graph.input(&Tensor::zeros(shape)).unwrap();

        // 构建 Negate 链：-(-(-p))
        let mut result = p.clone();
        for _ in 0..chain_len {
            result = -&result;
        }
        let loss = result.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);

        let mut opt = SGD::new(&graph, &[p.clone()], 0.001);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                loss.forward().unwrap();
                opt.zero_grad().unwrap();
                let _ = loss.backward().unwrap();
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

// ===================== Subtract 链路 benchmark =====================

/// 构建 Subtract 链路：p1 - p2 - p3 ... → loss
///
/// Subtract 第一个父节点是 PassThrough，第二个是 Negated。
fn bench_subtract_chain_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtract_chain_backward");
    group.sample_size(20);

    let configs: &[(&str, &[usize], usize)] = &[
        ("small_4sub", &[8, 32], 4),
        ("medium_8sub", &[32, 128], 8),
    ];

    for &(name, shape, chain_len) in configs {
        let graph = Graph::new();
        let params: Vec<_> = (0..chain_len)
            .map(|i| {
                graph
                    .parameter(shape, Init::Normal { mean: 0.0, std: 0.1 }, &format!("p{i}"))
                    .unwrap()
            })
            .collect();
        let target = graph.input(&Tensor::zeros(shape)).unwrap();

        // 构建 Subtract 链：p0 - p1 - p2 - ...
        let mut result = params[0].clone();
        for param in params.iter().skip(1) {
            result = &result - param;
        }
        let loss = result.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);

        let param_refs: Vec<_> = params.iter().collect();
        let mut opt = SGD::new(&graph, &param_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>(), 0.001);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                loss.forward().unwrap();
                opt.zero_grad().unwrap();
                let _ = loss.backward().unwrap();
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

// ===================== MLP backward benchmark =====================

/// MLP 反向传播 benchmark
///
/// Linear 层内部包含 MatMul + Add（bias），综合测量常见训练场景。
struct BenchMLP {
    fc1: Linear,
    fc2: Linear,
}

impl BenchMLP {
    fn new(graph: &Graph, input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            fc1: Linear::new(graph, input_size, hidden_size, true, "fc1").unwrap(),
            fc2: Linear::new(graph, hidden_size, output_size, true, "fc2").unwrap(),
        }
    }
}

impl Module for BenchMLP {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

fn bench_mlp_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_backward");
    group.sample_size(20);

    // (名称, batch, input, hidden, output)
    let configs: &[(&str, usize, usize, usize, usize)] = &[
        ("xor_b8", 8, 2, 4, 1),
        ("small_b32", 32, 32, 64, 10),
        ("medium_b64", 64, 784, 128, 10),
    ];

    for &(name, batch, input, hidden, output) in configs {
        let graph = Graph::new();
        let mlp = BenchMLP::new(&graph, input, hidden, output);
        let x = Tensor::random(0.0, 1.0, &[batch, input]);
        let target = Tensor::random(0.0, 1.0, &[batch, output]);

        let mut opt = SGD::new(&graph, &mlp.parameters(), 0.01);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let h = mlp.fc1.forward(&x).relu();
                let out = mlp.fc2.forward(&h);
                let loss = out.mse_loss(&target).unwrap();
                graph.snapshot_once_from(&[&loss]);
                opt.zero_grad().unwrap();
                let _ = loss.backward().unwrap();
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add_chain_backward,
    bench_negate_chain_backward,
    bench_subtract_chain_backward,
    bench_mlp_backward,
);
criterion_main!(benches);
