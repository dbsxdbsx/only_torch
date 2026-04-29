/// Normalization 层性能基准测试
///
/// 覆盖 BatchNorm / LayerNorm / RMSNorm / GroupNorm 的 forward + backward 路径。
/// 运行方式：`cargo bench --bench normalization`
use criterion::{Criterion, criterion_group, criterion_main};
use only_torch::nn::{
    BatchNorm, Graph, GroupNorm, Init, LayerNorm, Module, Optimizer, RMSNorm, SGD, VarLossOps,
};
use only_torch::tensor::Tensor;

fn bench_normalization_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization_forward");
    group.sample_size(10);

    group.bench_function("batch_norm_b32_c16_28x28", |bench| {
        let graph = Graph::new();
        let norm = BatchNorm::new(&graph, 16, 1e-5, 0.1, "bn").unwrap();
        let input = Tensor::random(0.0, 1.0, &[32, 16, 28, 28]);

        bench.iter(|| {
            let _ = norm.forward(&input);
        });
    });

    group.bench_function("layer_norm_b32_128", |bench| {
        let graph = Graph::new();
        let norm = LayerNorm::new(&graph, &[128], 1e-5, "ln").unwrap();
        let input = Tensor::random(0.0, 1.0, &[32, 128]);

        bench.iter(|| {
            let _ = norm.forward(&input);
        });
    });

    group.bench_function("rms_norm_b32_128", |bench| {
        let graph = Graph::new();
        let norm = RMSNorm::new(&graph, &[128], 1e-5, "rn").unwrap();
        let input = Tensor::random(0.0, 1.0, &[32, 128]);

        bench.iter(|| {
            let _ = norm.forward(&input);
        });
    });

    group.bench_function("group_norm_b16_c32_14x14", |bench| {
        let graph = Graph::new();
        let norm = GroupNorm::new(&graph, 8, 32, 1e-5, "gn").unwrap();
        let input = Tensor::random(0.0, 1.0, &[16, 32, 14, 14]);

        bench.iter(|| {
            let _ = norm.forward(&input);
        });
    });

    group.finish();
}

fn bench_batch_norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_norm_backward");
    group.sample_size(10);

    let graph = Graph::new();
    let input = graph
        .parameter(
            &[32, 16, 28, 28],
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            "input",
        )
        .unwrap();
    let norm = BatchNorm::new(&graph, 16, 1e-5, 0.1, "bn").unwrap();
    let target = Tensor::zeros(&[32, 16, 28, 28]);
    let out = norm.forward(&input);
    let loss = out.mse_loss(&target).unwrap();
    graph.snapshot_once_from(&[&loss]);
    let mut params = norm.parameters();
    params.push(input);
    let mut opt = SGD::new(&graph, &params, 0.001);

    group.bench_function("b32_c16_28x28", |bench| {
        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
    group.finish();
}

fn bench_layer_norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm_backward");
    group.sample_size(10);

    let graph = Graph::new();
    let input = graph
        .parameter(
            &[32, 128],
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            "input",
        )
        .unwrap();
    let norm = LayerNorm::new(&graph, &[128], 1e-5, "ln").unwrap();
    let target = Tensor::zeros(&[32, 128]);
    let out = norm.forward(&input);
    let loss = out.mse_loss(&target).unwrap();
    graph.snapshot_once_from(&[&loss]);
    let mut params = norm.parameters();
    params.push(input);
    let mut opt = SGD::new(&graph, &params, 0.001);

    group.bench_function("b32_128", |bench| {
        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
    group.finish();
}

fn bench_rms_norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm_backward");
    group.sample_size(10);

    let graph = Graph::new();
    let input = graph
        .parameter(
            &[32, 128],
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            "input",
        )
        .unwrap();
    let norm = RMSNorm::new(&graph, &[128], 1e-5, "rn").unwrap();
    let target = Tensor::zeros(&[32, 128]);
    let out = norm.forward(&input);
    let loss = out.mse_loss(&target).unwrap();
    graph.snapshot_once_from(&[&loss]);
    let mut params = norm.parameters();
    params.push(input);
    let mut opt = SGD::new(&graph, &params, 0.001);

    group.bench_function("b32_128", |bench| {
        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
    group.finish();
}

fn bench_group_norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_norm_backward");
    group.sample_size(10);

    let graph = Graph::new();
    let input = graph
        .parameter(
            &[16, 32, 14, 14],
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            "input",
        )
        .unwrap();
    let norm = GroupNorm::new(&graph, 8, 32, 1e-5, "gn").unwrap();
    let target = Tensor::zeros(&[16, 32, 14, 14]);
    let out = norm.forward(&input);
    let loss = out.mse_loss(&target).unwrap();
    graph.snapshot_once_from(&[&loss]);
    let mut params = norm.parameters();
    params.push(input);
    let mut opt = SGD::new(&graph, &params, 0.001);

    group.bench_function("b16_c32_14x14", |bench| {
        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_normalization_forward,
    bench_batch_norm_backward,
    bench_layer_norm_backward,
    bench_rms_norm_backward,
    bench_group_norm_backward,
);
criterion_main!(benches);
