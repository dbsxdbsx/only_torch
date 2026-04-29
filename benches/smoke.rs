/// 快速性能回归 smoke benchmark
///
/// 覆盖 Tensor / Conv2d / MLP / CNN / Add backward 五条主链路。
/// 运行方式：`cargo bench --bench smoke`
use criterion::{Criterion, criterion_group, criterion_main};
use only_torch::nn::{
    Conv2d, Graph, Init, Linear, MaxPool2d, Module, Optimizer, SGD, VarActivationOps, VarLossOps,
    VarShapeOps,
};
use only_torch::tensor::Tensor;
use std::time::Duration;

fn bench_tensor_add_64x784(c: &mut Criterion) {
    let a = Tensor::random(0.0, 1.0, &[64, 784]);
    let b = Tensor::random(0.0, 1.0, &[64, 784]);

    c.bench_function("smoke_tensor_add_64x784", |bench| {
        bench.iter(|| {
            let _ = &a + &b;
        });
    });
}

fn bench_conv2d_fwd_b32_3x28x28(c: &mut Criterion) {
    let graph = Graph::new();
    let conv = Conv2d::new(&graph, 3, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv").unwrap();
    let input = Tensor::random(0.0, 1.0, &[32, 3, 28, 28]);

    c.bench_function("smoke_conv2d_fwd_b32_3x28x28", |bench| {
        bench.iter(|| {
            let _ = conv.forward(&input);
        });
    });
}

fn bench_conv2d_eval_1x1_b1(c: &mut Criterion) {
    let graph = Graph::new();
    graph.eval();
    let conv = Conv2d::new(&graph, 3, 16, (1, 1), (1, 1), (0, 0), (1, 1), true, "conv").unwrap();
    let input = Tensor::random(0.0, 1.0, &[1, 3, 64, 64]);

    c.bench_function("smoke_conv2d_eval_1x1_b1", |bench| {
        bench.iter(|| {
            let _ = conv.forward(&input);
        });
    });
}

struct SmokeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl SmokeMlp {
    fn new(graph: &Graph) -> Self {
        Self {
            fc1: Linear::new(graph, 2, 8, true, "fc1").unwrap(),
            fc2: Linear::new(graph, 8, 1, true, "fc2").unwrap(),
        }
    }

    fn forward(&self, x: &Tensor) -> only_torch::nn::Var {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h)
    }
}

impl Module for SmokeMlp {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

fn bench_mlp_train_step_xor(c: &mut Criterion) {
    let graph = Graph::new();
    let mlp = SmokeMlp::new(&graph);
    let x = Tensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let target = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);
    let mut opt = SGD::new(&graph, &mlp.parameters(), 0.1);

    c.bench_function("smoke_mlp_train_step_xor", |bench| {
        bench.iter(|| {
            let out = mlp.forward(&x);
            let loss = out.mse_loss(&target).unwrap();
            graph.snapshot_once_from(&[&loss]);
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
}

fn bench_add_chain_backward_8(c: &mut Criterion) {
    let graph = Graph::new();
    let p = graph
        .parameter(
            &[8, 32],
            Init::Normal {
                mean: 0.0,
                std: 0.1,
            },
            "p",
        )
        .unwrap();
    let target = graph.input(&Tensor::zeros(&[8, 32])).unwrap();
    let mut result = p.clone();
    for _ in 1..8 {
        result = &result + &p;
    }
    let loss = result.mse_loss(&target).unwrap();
    graph.snapshot_once_from(&[&loss]);
    let mut opt = SGD::new(&graph, &[p], 0.001);

    c.bench_function("smoke_add_chain_backward_8", |bench| {
        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
}

struct SmokeCnn {
    conv: Conv2d,
    pool: MaxPool2d,
    fc: Linear,
}

impl SmokeCnn {
    fn new(graph: &Graph) -> Self {
        Self {
            conv: Conv2d::new(graph, 1, 4, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv").unwrap(),
            pool: MaxPool2d::new(graph, (2, 2), None, "pool"),
            fc: Linear::new(graph, 4 * 14 * 14, 10, true, "fc").unwrap(),
        }
    }

    fn forward(&self, x: &Tensor) -> only_torch::nn::Var {
        let h = self.conv.forward(x).relu();
        let h = self.pool.forward(&h);
        let h = h.flatten().unwrap();
        self.fc.forward(&h)
    }
}

impl Module for SmokeCnn {
    fn parameters(&self) -> Vec<only_torch::nn::Var> {
        [self.conv.parameters(), self.fc.parameters()].concat()
    }
}

fn bench_cnn_train_step_b4(c: &mut Criterion) {
    let graph = Graph::new();
    let cnn = SmokeCnn::new(&graph);
    let input = Tensor::random(0.0, 1.0, &[4, 1, 28, 28]);
    let target = Tensor::random(0.0, 1.0, &[4, 10]);
    let mut opt = SGD::new(&graph, &cnn.parameters(), 0.01);

    c.bench_function("smoke_cnn_train_step_b4", |bench| {
        bench.iter(|| {
            let out = cnn.forward(&input);
            let loss = out.mse_loss(&target).unwrap();
            graph.snapshot_once_from(&[&loss]);
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    targets =
        bench_tensor_add_64x784,
        bench_conv2d_fwd_b32_3x28x28,
        bench_conv2d_eval_1x1_b1,
        bench_mlp_train_step_xor,
        bench_add_chain_backward_8,
        bench_cnn_train_step_b4
}
criterion_main!(benches);
