/// 优化器 step 性能基准测试
///
/// 隔离测量 SGD::step / Adam::step 在不同参数规模下的更新开销。
/// 运行方式：`cargo bench --bench optimizer`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::nn::{Adam, Graph, Init, Optimizer, SGD, Var, VarLossOps};
use only_torch::tensor::Tensor;

fn build_params_with_grads(graph: &Graph, shape: &[usize], count: usize, prefix: &str) -> Vec<Var> {
    let params: Vec<_> = (0..count)
        .map(|idx| {
            graph
                .parameter(
                    shape,
                    Init::Normal {
                        mean: 0.0,
                        std: 0.1,
                    },
                    &format!("{prefix}_{idx}"),
                )
                .unwrap()
        })
        .collect();

    let target = Tensor::zeros(shape);
    let mut total_loss = params[0].mse_loss(&target).unwrap();
    for param in params.iter().skip(1) {
        let loss = param.mse_loss(&target).unwrap();
        total_loss = &total_loss + &loss;
    }
    graph.snapshot_once_from(&[&total_loss]);
    let _ = total_loss.backward().unwrap();

    params
}

fn bench_sgd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd_step");
    group.sample_size(20);

    let configs: &[(&str, &[usize], usize)] = &[
        ("small_16_params_32x32", &[32, 32], 16),
        ("mlp_8_params_128x128", &[128, 128], 8),
        ("cnn_16_params_16x3x3x3", &[16, 3, 3, 3], 16),
    ];

    for &(name, shape, count) in configs {
        let graph = Graph::new();
        let params = build_params_with_grads(&graph, shape, count, "sgd_param");
        let mut opt = SGD::new(&graph, &params, 0.01);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_adam_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_step");
    group.sample_size(20);

    let configs: &[(&str, &[usize], usize)] = &[
        ("small_16_params_32x32", &[32, 32], 16),
        ("mlp_8_params_128x128", &[128, 128], 8),
        ("cnn_16_params_16x3x3x3", &[16, 3, 3, 3], 16),
    ];

    for &(name, shape, count) in configs {
        let graph = Graph::new();
        let params = build_params_with_grads(&graph, shape, count, "adam_param");
        let mut opt = Adam::new(&graph, &params, 0.001);
        opt.step().unwrap(); // 预热 Adam 状态，正式测稳态 step。

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                opt.step().unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sgd_step, bench_adam_step);
criterion_main!(benches);
