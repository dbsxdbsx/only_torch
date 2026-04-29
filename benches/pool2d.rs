/// Pool2d 前向/反向传播性能基准测试
///
/// 覆盖 MaxPool2d / AvgPool2d 的常见 CNN 下采样路径。
/// 运行方式：`cargo bench --bench pool2d`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::nn::{AvgPool2d, Graph, Init, MaxPool2d, Optimizer, SGD, VarLossOps};
use only_torch::tensor::Tensor;

fn pool_output_hw(input_hw: usize, kernel: usize, stride: usize) -> usize {
    (input_hw - kernel) / stride + 1
}

fn bench_max_pool2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_pool2d_forward");
    group.sample_size(10);

    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("b8_c8_28x28_k2", 8, 8, 28, 2, 2),
        ("b32_c16_28x28_k2", 32, 16, 28, 2, 2),
        ("b16_c32_14x14_k3", 16, 32, 14, 3, 2),
    ];

    for &(name, batch, channels, hw, kernel, stride) in configs {
        let graph = Graph::new();
        let pool = MaxPool2d::new(&graph, (kernel, kernel), Some((stride, stride)), "max_pool");
        let input = Tensor::random(0.0, 1.0, &[batch, channels, hw, hw]);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                let _ = pool.forward(&input);
            });
        });
    }
    group.finish();
}

fn bench_avg_pool2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_pool2d_forward");
    group.sample_size(10);

    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("b8_c8_28x28_k2", 8, 8, 28, 2, 2),
        ("b32_c16_28x28_k2", 32, 16, 28, 2, 2),
        ("b16_c32_14x14_k3", 16, 32, 14, 3, 2),
    ];

    for &(name, batch, channels, hw, kernel, stride) in configs {
        let graph = Graph::new();
        let pool = AvgPool2d::new(&graph, (kernel, kernel), Some((stride, stride)), "avg_pool");
        let input = Tensor::random(0.0, 1.0, &[batch, channels, hw, hw]);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                let _ = pool.forward(&input);
            });
        });
    }
    group.finish();
}

fn bench_max_pool2d_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_pool2d_backward");
    group.sample_size(10);

    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("b8_c8_28x28_k2", 8, 8, 28, 2, 2),
        ("b32_c16_28x28_k2", 32, 16, 28, 2, 2),
    ];

    for &(name, batch, channels, hw, kernel, stride) in configs {
        let graph = Graph::new();
        let input = graph
            .parameter(
                &[batch, channels, hw, hw],
                Init::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                "input",
            )
            .unwrap();
        let pool = MaxPool2d::new(&graph, (kernel, kernel), Some((stride, stride)), "max_pool");
        let out_hw = pool_output_hw(hw, kernel, stride);
        let target = Tensor::zeros(&[batch, channels, out_hw, out_hw]);
        let out = pool.forward(&input);
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[input], 0.001);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                loss.forward().unwrap();
                opt.zero_grad().unwrap();
                let _ = loss.backward().unwrap();
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_avg_pool2d_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_pool2d_backward");
    group.sample_size(10);

    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("b8_c8_28x28_k2", 8, 8, 28, 2, 2),
        ("b32_c16_28x28_k2", 32, 16, 28, 2, 2),
    ];

    for &(name, batch, channels, hw, kernel, stride) in configs {
        let graph = Graph::new();
        let input = graph
            .parameter(
                &[batch, channels, hw, hw],
                Init::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                "input",
            )
            .unwrap();
        let pool = AvgPool2d::new(&graph, (kernel, kernel), Some((stride, stride)), "avg_pool");
        let out_hw = pool_output_hw(hw, kernel, stride);
        let target = Tensor::zeros(&[batch, channels, out_hw, out_hw]);
        let out = pool.forward(&input);
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[input], 0.001);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                loss.forward().unwrap();
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
    bench_max_pool2d_forward,
    bench_avg_pool2d_forward,
    bench_max_pool2d_backward,
    bench_avg_pool2d_backward,
);
criterion_main!(benches);
