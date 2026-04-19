/// Conv2d 前向/反向传播性能基准测试
///
/// 用于量化 im2col 优化前后的性能差异。
/// 运行方式：`cargo bench --bench conv2d`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::nn::{
    Conv2d, Graph, MaxPool2d, Module, Optimizer, SGD, VarActivationOps, VarLossOps,
};
use only_torch::tensor::Tensor;

// ===================== 辅助模型 =====================

/// 单层 Conv2d 网络（隔离测量卷积性能）
struct SingleConvNet {
    conv: Conv2d,
}

impl SingleConvNet {
    fn new(
        graph: &Graph,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        pad: usize,
    ) -> Self {
        Self {
            conv: Conv2d::new(
                graph, in_c, out_c, (kernel, kernel), (1, 1), (pad, pad), (1, 1), false, "conv",
            )
            .unwrap(),
        }
    }
}

impl Module for SingleConvNet {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        self.conv.parameters()
    }
}

/// 双层 Conv+Pool 网络（更接近真实场景）
struct TwoLayerCNN {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
}

impl TwoLayerCNN {
    fn new(graph: &Graph, in_c: usize) -> Self {
        Self {
            conv1: Conv2d::new(graph, in_c, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1").unwrap(),
            pool1: MaxPool2d::new(graph, (2, 2), None, "pool1"),
            conv2: Conv2d::new(graph, 8, 16, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2").unwrap(),
            pool2: MaxPool2d::new(graph, (2, 2), None, "pool2"),
        }
    }

    fn forward_pass(&self, x: &Tensor) -> only_torch::nn::Var {
        let h = self.conv1.forward(x).relu();
        let h = self.pool1.forward(&h);
        let h = self.conv2.forward(&h).relu();
        self.pool2.forward(&h)
    }
}

impl Module for TwoLayerCNN {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [self.conv1.parameters(), self.conv2.parameters()].concat()
    }
}

// ===================== Benchmark =====================

/// 前向传播 benchmark（隔离 Conv2d 卷积计算）
fn bench_conv_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_forward");
    group.sample_size(10);

    // (名称, batch, in_c, h, w, out_c, kernel, pad)
    let configs: &[(&str, usize, usize, usize, usize, usize, usize, usize)] = &[
        ("1x1x28x28_to_4", 1, 1, 28, 28, 4, 3, 1),
        ("32x3x28x28_to_8", 32, 3, 28, 28, 8, 3, 1),
        ("64x8x14x14_to_16", 64, 8, 14, 14, 16, 3, 1),
        ("128x16x14x14_to_32", 128, 16, 14, 14, 32, 3, 1),
    ];

    for &(name, batch, in_c, h, w, out_c, k, pad) in configs {
        let graph = Graph::new();
        let net = SingleConvNet::new(&graph, in_c, out_c, k, pad);
        let input = Tensor::random(0.0, 1.0, &[batch, in_c, h, w]);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let _out = net.conv.forward(&input);
            });
        });
    }
    group.finish();
}

/// 前向 + 反向完整步骤 benchmark
fn bench_conv_full_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_full_step");
    group.sample_size(10);

    let configs: &[(&str, usize, usize, usize, usize, usize, usize, usize)] = &[
        ("1x1x28x28_to_4", 1, 1, 28, 28, 4, 3, 1),
        ("32x3x28x28_to_8", 32, 3, 28, 28, 8, 3, 1),
        ("64x8x14x14_to_16", 64, 8, 14, 14, 16, 3, 1),
    ];

    for &(name, batch, in_c, h, w, out_c, k, pad) in configs {
        let graph = Graph::new();
        let conv =
            Conv2d::new(&graph, in_c, out_c, (k, k), (1, 1), (pad, pad), (1, 1), false, "conv").unwrap();
        let input = Tensor::random(0.0, 1.0, &[batch, in_c, h, w]);
        // 构造 target 用于 mse_loss
        let out_h = (h + 2 * pad - k) / 1 + 1;
        let out_w = (w + 2 * pad - k) / 1 + 1;
        let target = Tensor::random(0.0, 1.0, &[batch, out_c, out_h, out_w]);

        let mut opt = SGD::new(&graph, &conv.parameters(), 0.01);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let out = conv.forward(&input);
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

/// 双层 CNN 实际场景 benchmark（模拟 MNIST/象棋 CNN 前向）
fn bench_two_layer_cnn(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_layer_cnn");
    group.sample_size(10);

    let configs: &[(&str, usize, usize)] = &[
        ("batch1_1ch", 1, 1),
        ("batch32_1ch", 32, 1),
        ("batch32_3ch", 32, 3),
    ];

    for &(name, batch, in_c) in configs {
        let graph = Graph::new();
        let net = TwoLayerCNN::new(&graph, in_c);
        let input = Tensor::random(0.0, 1.0, &[batch, in_c, 28, 28]);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let _out = net.forward_pass(&input);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_conv_forward, bench_conv_full_step, bench_two_layer_cnn);
criterion_main!(benches);
