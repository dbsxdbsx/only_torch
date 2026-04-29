/// MultiHeadAttention 性能基准测试
///
/// 覆盖 self-attention / cross-attention 的 forward + backward 路径。
/// 运行方式：`cargo bench --bench attention`
use criterion::{Criterion, criterion_group, criterion_main};
use only_torch::nn::{Graph, Init, Module, MultiHeadAttention, Optimizer, SGD, Var, VarLossOps};
use only_torch::tensor::Tensor;

const BATCH: usize = 4;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const SELF_T: usize = 8;
const CROSS_T_Q: usize = 6;
const CROSS_T_KV: usize = 10;

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

fn bench_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_forward");
    group.sample_size(10);

    group.bench_function("self_b4_t8_d32_h4", |bench| {
        let graph = Graph::new();
        let attn = MultiHeadAttention::new(&graph, EMBED_DIM, NUM_HEADS, "attn").unwrap();
        let input = Tensor::random(0.0, 1.0, &[BATCH, SELF_T, EMBED_DIM]);

        bench.iter(|| {
            let out = attn.forward(&input, &input, &input);
            out.forward().unwrap();
        });
    });

    group.bench_function("cross_b4_tq6_tkv10_d32_h4", |bench| {
        let graph = Graph::new();
        let attn = MultiHeadAttention::new(&graph, EMBED_DIM, NUM_HEADS, "attn").unwrap();
        let query = Tensor::random(0.0, 1.0, &[BATCH, CROSS_T_Q, EMBED_DIM]);
        let kv = Tensor::random(0.0, 1.0, &[BATCH, CROSS_T_KV, EMBED_DIM]);

        bench.iter(|| {
            let out = attn.forward(&query, &kv, &kv);
            out.forward().unwrap();
        });
    });

    group.finish();
}

fn bench_attention_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_backward");
    group.sample_size(10);

    group.bench_function("self_b4_t8_d32_h4", |bench| {
        let graph = Graph::new();
        let attn = MultiHeadAttention::new(&graph, EMBED_DIM, NUM_HEADS, "attn").unwrap();
        let input = parameter(&graph, &[BATCH, SELF_T, EMBED_DIM], "self_input");
        let target = Tensor::zeros(&[BATCH, SELF_T, EMBED_DIM]);
        let out = attn.forward(&input, &input, &input);
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut params = attn.parameters();
        params.push(input);
        let mut opt = SGD::new(&graph, &params, 0.001);

        bench.iter(|| {
            loss.forward().unwrap();
            opt.zero_grad().unwrap();
            let _ = loss.backward().unwrap();
            opt.step().unwrap();
        });
    });

    group.bench_function("cross_b4_tq6_tkv10_d32_h4", |bench| {
        let graph = Graph::new();
        let attn = MultiHeadAttention::new(&graph, EMBED_DIM, NUM_HEADS, "attn").unwrap();
        let query = parameter(&graph, &[BATCH, CROSS_T_Q, EMBED_DIM], "query");
        let kv = parameter(&graph, &[BATCH, CROSS_T_KV, EMBED_DIM], "kv");
        let target = Tensor::zeros(&[BATCH, CROSS_T_Q, EMBED_DIM]);
        let out = attn.forward(&query, &kv, &kv);
        let loss = out.mse_loss(&target).unwrap();
        graph.snapshot_once_from(&[&loss]);
        let mut params = attn.parameters();
        params.push(query);
        params.push(kv);
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

criterion_group!(benches, bench_attention_forward, bench_attention_backward);
criterion_main!(benches);
