//! Rayon 小任务并行阈值定标 bench（热路径审计 Reviewer 方向 2）
//!
//! 对比同一 map-only 写入体在 串行 chunks_mut vs rayon par_chunks_mut 下的耗时,
//! 覆盖 (chunk 数 × 单 chunk 元素数) 网格,用于给 `utils::parallel::PAR_MIN_WORK`
//! 定标:找出并行开始跑赢串行的总工作量交叉点。
//!
//! 运行: cargo bench --bench rayon_threshold --features blas-mkl

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use std::hint::black_box;

/// 模拟逐元素 map 工作体(~4 flop/元素,近似激活/归一化类算子密度)
#[inline]
fn fill_body(i: usize, chunk: &mut [f32]) {
    let base = i as f32 + 0.5;
    for (j, v) in chunk.iter_mut().enumerate() {
        let x = base + j as f32 * 1.000_1;
        *v = x.mul_add(0.999, -0.25) * 0.5 + x * 0.001;
    }
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_threshold");
    // (chunk 数, 单 chunk 元素数):batch=2/8/32 × 小/中/大样本
    let grid: &[(usize, usize)] = &[
        (2, 64),
        (2, 1024),
        (2, 8192),
        (2, 32768),
        (8, 64),
        (8, 1024),
        (8, 8192),
        (32, 64),
        (32, 1024),
        (32, 8192),
    ];
    for &(chunks, size) in grid {
        let label = format!("{chunks}x{size}");
        let mut buf = vec![0.0f32; chunks * size];

        group.bench_with_input(BenchmarkId::new("serial", &label), &(), |b, ()| {
            b.iter(|| {
                buf.chunks_mut(size)
                    .enumerate()
                    .for_each(|(i, ch)| fill_body(i, ch));
                black_box(buf[0]);
            });
        });
        group.bench_with_input(BenchmarkId::new("rayon", &label), &(), |b, ()| {
            b.iter(|| {
                buf.par_chunks_mut(size)
                    .enumerate()
                    .for_each(|(i, ch)| fill_body(i, ch));
                black_box(buf[0]);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
