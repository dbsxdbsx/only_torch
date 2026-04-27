/// 端到端训练步性能基准测试
///
/// 测量 forward + backward + optimizer.step 完整训练步的耗时。
/// 运行方式：`cargo bench --bench end_to_end`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::nn::{
    Conv2d, Graph, Linear, MaxPool2d, Module, Optimizer, SGD, VarActivationOps, VarLossOps,
    VarShapeOps,
};
use only_torch::tensor::Tensor;

// ===================== MLP 端到端 =====================

struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MLP {
    fn new(graph: &Graph, input: usize, hidden: usize, output: usize) -> Self {
        Self {
            fc1: Linear::new(graph, input, hidden, true, "fc1").unwrap(),
            fc2: Linear::new(graph, hidden, hidden, true, "fc2").unwrap(),
            fc3: Linear::new(graph, hidden, output, true, "fc3").unwrap(),
        }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
        ]
        .concat()
    }
}

/// MLP 完整训练步（forward + backward + step）
fn bench_mlp_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_train_step");
    group.sample_size(20);

    // (名称, batch, input, hidden, output)
    let configs: &[(&str, usize, usize, usize, usize)] = &[
        ("xor_b8", 8, 2, 8, 1),
        ("mnist_linear_b32", 32, 784, 128, 10),
        ("mnist_linear_b64", 64, 784, 128, 10),
    ];

    for &(name, batch, input, hidden, output) in configs {
        let graph = Graph::new();
        let mlp = MLP::new(&graph, input, hidden, output);
        let x = Tensor::random(0.0, 1.0, &[batch, input]);
        let target = Tensor::random(0.0, 1.0, &[batch, output]);

        let mut opt = SGD::new(&graph, &mlp.parameters(), 0.01);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let h = mlp.fc1.forward(&x).relu();
                let h = mlp.fc2.forward(&h).relu();
                let out = mlp.fc3.forward(&h);
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

// ===================== CNN 端到端 =====================

struct SimpleCNN {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc: Linear,
}

impl SimpleCNN {
    fn new(graph: &Graph, in_c: usize) -> Self {
        Self {
            conv1: Conv2d::new(
                graph,
                in_c,
                8,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                "conv1",
            )
            .unwrap(),
            pool1: MaxPool2d::new(graph, (2, 2), None, "pool1"),
            conv2: Conv2d::new(graph, 8, 16, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2")
                .unwrap(),
            pool2: MaxPool2d::new(graph, (2, 2), None, "pool2"),
            // 28x28 → 14x14 → 7x7，展平后 16*7*7 = 784
            fc: Linear::new(graph, 16 * 7 * 7, 10, true, "fc").unwrap(),
        }
    }

    fn forward_pass(&self, x: &Tensor) -> only_torch::nn::Var {
        let h = self.conv1.forward(x).relu();
        let h = self.pool1.forward(&h);
        let h = self.conv2.forward(&h).relu();
        let h = self.pool2.forward(&h);
        let h = h.flatten().unwrap();
        self.fc.forward(&h)
    }
}

impl Module for SimpleCNN {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.fc.parameters(),
        ]
        .concat()
    }
}

/// CNN 完整训练步
fn bench_cnn_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnn_train_step");
    group.sample_size(10);

    // (名称, batch, in_c)
    let configs: &[(&str, usize, usize)] = &[
        ("batch1_1ch", 1, 1),
        ("batch8_1ch", 8, 1),
        ("batch32_1ch", 32, 1),
    ];

    for &(name, batch, in_c) in configs {
        let graph = Graph::new();
        let cnn = SimpleCNN::new(&graph, in_c);
        let x = Tensor::random(0.0, 1.0, &[batch, in_c, 28, 28]);
        let target = Tensor::random(0.0, 1.0, &[batch, 10]);

        let mut opt = SGD::new(&graph, &cnn.parameters(), 0.01);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let out = cnn.forward_pass(&x);
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

criterion_group!(benches, bench_mlp_train_step, bench_cnn_train_step);
criterion_main!(benches);
