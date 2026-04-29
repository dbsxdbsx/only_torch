/// Loss 算子性能基准测试
///
/// 覆盖 MSE / CrossEntropy / BCE / Huber 的 forward + backward 路径。
/// 运行方式：`cargo bench --bench loss`
use criterion::{Criterion, criterion_group, criterion_main};
use only_torch::nn::{Graph, Init, Optimizer, SGD, Var, VarLossOps};
use only_torch::tensor::Tensor;

fn one_hot(batch: usize, classes: usize) -> Tensor {
    let mut data = vec![0.0; batch * classes];
    for row in 0..batch {
        data[row * classes + row % classes] = 1.0;
    }
    Tensor::new(&data, &[batch, classes])
}

fn binary_targets(shape: &[usize]) -> Tensor {
    let len = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|idx| (idx % 2) as f32).collect();
    Tensor::new(&data, shape)
}

fn parameter(graph: &Graph, shape: &[usize], name: &str) -> Var {
    graph
        .parameter(
            shape,
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            name,
        )
        .unwrap()
}

fn bench_loss_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_forward");
    group.sample_size(20);

    group.bench_function("mse_b64_128", |bench| {
        let graph = Graph::new();
        let pred = parameter(&graph, &[64, 128], "mse_pred");
        let target = Tensor::zeros(&[64, 128]);
        let loss = pred.mse_loss(&target).unwrap();

        bench.iter(|| {
            loss.forward().unwrap();
        });
    });

    group.bench_function("cross_entropy_b64_c10", |bench| {
        let graph = Graph::new();
        let logits = parameter(&graph, &[64, 10], "ce_logits");
        let target = one_hot(64, 10);
        let loss = logits.cross_entropy(&target).unwrap();

        bench.iter(|| {
            loss.forward().unwrap();
        });
    });

    group.bench_function("bce_b64_128", |bench| {
        let graph = Graph::new();
        let logits = parameter(&graph, &[64, 128], "bce_logits");
        let target = binary_targets(&[64, 128]);
        let loss = logits.bce_loss(&target).unwrap();

        bench.iter(|| {
            loss.forward().unwrap();
        });
    });

    group.bench_function("huber_b64_128", |bench| {
        let graph = Graph::new();
        let pred = parameter(&graph, &[64, 128], "huber_pred");
        let target = Tensor::zeros(&[64, 128]);
        let loss = pred.huber_loss(&target).unwrap();

        bench.iter(|| {
            loss.forward().unwrap();
        });
    });

    group.finish();
}

fn bench_loss_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_backward");
    group.sample_size(20);

    group.bench_function("mse_b64_128", |bench| {
        let graph = Graph::new();
        let pred = parameter(&graph, &[64, 128], "mse_pred");
        let target = Tensor::zeros(&[64, 128]);
        let loss = pred.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[pred], 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("cross_entropy_b64_c10", |bench| {
        let graph = Graph::new();
        let logits = parameter(&graph, &[64, 10], "ce_logits");
        let target = one_hot(64, 10);
        let loss = logits.cross_entropy(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[logits], 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("bce_b64_128", |bench| {
        let graph = Graph::new();
        let logits = parameter(&graph, &[64, 128], "bce_logits");
        let target = binary_targets(&[64, 128]);
        let loss = logits.bce_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[logits], 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("huber_b64_128", |bench| {
        let graph = Graph::new();
        let pred = parameter(&graph, &[64, 128], "huber_pred");
        let target = Tensor::zeros(&[64, 128]);
        let loss = pred.huber_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut opt = SGD::new(&graph, &[pred], 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_loss_forward, bench_loss_backward);
criterion_main!(benches);
