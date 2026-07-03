//! 训练 batch 组装 + 入图路径微基准：实证「融合组装 + owned 入图」优化。
//!
//! 运行：`cargo bench --bench obs_batch_assembly`
//!
//! # 背景（2026-07-03 拷贝消除优化）
//! 图像模式（Pong 84²×4 帧堆叠）训练 batch 组装旧路径每样本 obs 数据经手 3 次：
//! 1. `assemble_stacked_obs`：u8 帧反量化 + 4 帧堆叠 → 每样本一个中间 `Vec<f32>`；
//! 2. `stack_obs`：G 个样本 Vec 逐个 `extend_from_slice` 拼进 batch flat `Vec<f32>`；
//! 3. `graph.input(&Tensor)`：Tensor 深拷贝进 `BasicInput` 节点。
//!
//! 新路径把 (1)(2) 融合为「来源直写最终 flat」（物化恰好一次）。
//! (3) 的历史对比对象 `graph.input_owned`（move 语义）已随 **Tensor 存储 Arc/CoW 化**
//! 收敛删除——`input(&t)` 内部 clone 如今是 O(1) 引用计数浅拷贝，owned 双轨失去
//! 存在理由；下方 `graph_input_owned` / `*_input_owned` 两个 case 保留原名
//! （维持 Criterion baseline 连续性），实际同样调用 `input(&t)`。
//! 本 bench 用与生产等价的数据流复现两条路径（维度对齐 Pong：frame=84²、
//! stack=4、G=16），量化真实收益。
//!
//! 注意：两条路径的**反量化算术完全相同**（u8→f32 除 255），差异只在
//! 中间 Vec 物化次数与入图 clone——这正是本次优化改变的全部内容。

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use only_torch::nn::Graph;
use only_torch::tensor::Tensor;

const FRAME: usize = 84 * 84; // 单帧像素数（Atari 口径）
const STACK: usize = 4; // 帧堆叠数
const G: usize = 16; // batch 内样本数
const EP_LEN: usize = 128; // 模拟 episode 长度

const OBS_DIM: usize = STACK * FRAME;

/// 模拟 replay buffer 中一局的量化帧（u8 存储，与 `StoredObs::U8` 同编码）
fn make_episode() -> Vec<Vec<u8>> {
    (0..EP_LEN)
        .map(|t| (0..FRAME).map(|i| ((t * 31 + i * 7) % 256) as u8).collect())
        .collect()
}

/// 反量化追加（与 `StoredObs::append_f32_into` 同口径）
fn append_dequant(frame: &[u8], out: &mut Vec<f32>) {
    out.extend(frame.iter().map(|&b| f32::from(b) / 255.0));
}

/// 旧路径 (1)+(2)：每样本先物化堆叠 Vec，再复制进 batch flat
fn assemble_old(episode: &[Vec<u8>], starts: &[usize]) -> Vec<f32> {
    // (1) per-sample 堆叠 Vec（含帧引用表构造，与旧 train_batch 一致）
    let per_sample: Vec<Vec<f32>> = starts
        .iter()
        .map(|&t| {
            let frames: Vec<&Vec<u8>> = episode.iter().collect();
            let mut v = Vec::with_capacity(OBS_DIM);
            for i in (0..STACK).rev() {
                append_dequant(frames[t.saturating_sub(i)], &mut v);
            }
            v
        })
        .collect();
    // (2) 拼 batch flat
    let mut flat = Vec::with_capacity(G * OBS_DIM);
    for row in &per_sample {
        flat.extend_from_slice(row);
    }
    flat
}

/// 新路径 (1)+(2) 融合：来源直写最终 flat（零中间 Vec）
fn assemble_fused(episode: &[Vec<u8>], starts: &[usize]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(G * OBS_DIM);
    for &t in starts {
        for i in (0..STACK).rev() {
            append_dequant(&episode[t.saturating_sub(i)], &mut flat);
        }
    }
    flat
}

fn bench_assembly(c: &mut Criterion) {
    let episode = make_episode();
    let starts: Vec<usize> = (0..G).map(|s| (s * 13) % EP_LEN).collect();

    // 正确性守门：两条路径逐 bit 一致
    assert_eq!(
        assemble_old(&episode, &starts),
        assemble_fused(&episode, &starts)
    );

    let mut grp = c.benchmark_group("obs_batch_assembly");
    grp.sample_size(60);

    // ---- 组装本体（拷贝 1+2 vs 融合一次物化）----
    grp.bench_function("assemble_old_per_sample_then_stack", |b| {
        b.iter(|| black_box(assemble_old(&episode, &starts)));
    });
    grp.bench_function("assemble_fused_direct_flat", |b| {
        b.iter(|| black_box(assemble_fused(&episode, &starts)));
    });

    // ---- 入图（拷贝 3：clone vs move）----
    let g_old = Graph::new();
    grp.bench_function("graph_input_cloned", |b| {
        b.iter(|| {
            let flat = assemble_fused(&episode, &starts);
            let t = Tensor::new(flat, &[G, OBS_DIM]);
            black_box(g_old.input(&t).unwrap());
        });
    });
    let g_new = Graph::new();
    grp.bench_function("graph_input_owned", |b| {
        b.iter(|| {
            let flat = assemble_fused(&episode, &starts);
            let t = Tensor::new(flat, &[G, OBS_DIM]);
            black_box(g_new.input(&t).unwrap());
        });
    });

    // ---- 端到端（旧全套 vs 新全套）----
    let g_e2e_old = Graph::new();
    grp.bench_function("e2e_old_assemble_plus_input_cloned", |b| {
        b.iter(|| {
            let flat = assemble_old(&episode, &starts);
            let t = Tensor::new(flat, &[G, OBS_DIM]);
            black_box(g_e2e_old.input(&t).unwrap());
        });
    });
    let g_e2e_new = Graph::new();
    grp.bench_function("e2e_new_fused_plus_input_owned", |b| {
        b.iter(|| {
            let flat = assemble_fused(&episode, &starts);
            let t = Tensor::new(flat, &[G, OBS_DIM]);
            black_box(g_e2e_new.input(&t).unwrap());
        });
    });

    grp.finish();
}

criterion_group!(benches, bench_assembly);
criterion_main!(benches);
