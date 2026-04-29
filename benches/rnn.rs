/// 循环层性能基准测试
///
/// 覆盖 RNN / LSTM / GRU 的小规模序列 forward + backward 路径。
/// 运行方式：`cargo bench --bench rnn`
use criterion::{Criterion, criterion_group, criterion_main};
use only_torch::nn::{Graph, Gru, Init, Lstm, Module, Optimizer, Rnn, SGD, Var, VarLossOps};
use only_torch::tensor::Tensor;

const BATCH: usize = 16;
const SEQ_LEN: usize = 16;
const INPUT_SIZE: usize = 16;
const HIDDEN_SIZE: usize = 32;

fn sequence_input() -> Tensor {
    Tensor::random(0.0, 1.0, &[BATCH, SEQ_LEN, INPUT_SIZE])
}

fn trainable_sequence_input(graph: &Graph, name: &str) -> Var {
    graph
        .parameter(
            &[BATCH, SEQ_LEN, INPUT_SIZE],
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            name,
        )
        .unwrap()
}

fn bench_rnn_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_forward");
    group.sample_size(10);

    group.bench_function("rnn_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let rnn = Rnn::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "rnn").unwrap();
        let input = sequence_input();

        bench.iter(|| {
            let out = rnn.forward(&input).unwrap();
            out.forward().unwrap();
        });
    });

    group.bench_function("lstm_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let lstm = Lstm::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "lstm").unwrap();
        let input = sequence_input();

        bench.iter(|| {
            let out = lstm.forward(&input).unwrap();
            out.forward().unwrap();
        });
    });

    group.bench_function("gru_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let gru = Gru::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "gru").unwrap();
        let input = sequence_input();

        bench.iter(|| {
            let out = gru.forward(&input).unwrap();
            out.forward().unwrap();
        });
    });

    group.finish();
}

fn bench_rnn_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_backward");
    group.sample_size(10);

    group.bench_function("rnn_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let rnn = Rnn::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "rnn").unwrap();
        let input = trainable_sequence_input(&graph, "rnn_input");
        let target = Tensor::zeros(&[BATCH, HIDDEN_SIZE]);
        let out = rnn.forward(&input).unwrap();
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut params = rnn.parameters();
        params.push(input);
        let mut opt = SGD::new(&graph, &params, 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("lstm_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let lstm = Lstm::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "lstm").unwrap();
        let input = trainable_sequence_input(&graph, "lstm_input");
        let target = Tensor::zeros(&[BATCH, HIDDEN_SIZE]);
        let out = lstm.forward(&input).unwrap();
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut params = lstm.parameters();
        params.push(input);
        let mut opt = SGD::new(&graph, &params, 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("gru_b16_t16_i16_h32", |bench| {
        let graph = Graph::new();
        let gru = Gru::new(&graph, INPUT_SIZE, HIDDEN_SIZE, "gru").unwrap();
        let input = trainable_sequence_input(&graph, "gru_input");
        let target = Tensor::zeros(&[BATCH, HIDDEN_SIZE]);
        let out = gru.forward(&input).unwrap();
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut params = gru.parameters();
        params.push(input);
        let mut opt = SGD::new(&graph, &params, 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rnn_forward, bench_rnn_backward);
criterion_main!(benches);
