/// Tensor 基础操作性能基准测试
///
/// 测量 Tensor 层的基础操作耗时，为上层优化提供底层参考数据。
/// 运行方式：`cargo bench --bench tensor_ops`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use only_torch::tensor::Tensor;

// ===================== clone =====================

/// Tensor clone 开销（不同形状）
fn bench_tensor_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_clone");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
        ("32x16x7x7", &[32, 16, 7, 7]),
    ];

    for &(name, shape) in configs {
        let tensor = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let _cloned = tensor.clone();
            });
        });
    }
    group.finish();
}

// ===================== 逐元素操作 =====================

/// 逐元素加法
fn bench_tensor_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
    ];

    for &(name, shape) in configs {
        let a = Tensor::random(0.0, 1.0, shape);
        let b = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b_iter, _| {
            b_iter.iter(|| {
                let _result = &a + &b;
            });
        });
    }
    group.finish();
}

/// 原地加法（AddAssign）
fn bench_tensor_add_assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add_assign");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
    ];

    for &(name, shape) in configs {
        let delta = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            let mut target = Tensor::random(0.0, 1.0, shape);
            b.iter(|| {
                target += &delta;
            });
        });
    }
    group.finish();
}

/// 逐元素乘法
fn bench_tensor_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_mul");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
    ];

    for &(name, shape) in configs {
        let a = Tensor::random(0.0, 1.0, shape);
        let b = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b_iter, _| {
            b_iter.iter(|| {
                let _result = &a * &b;
            });
        });
    }
    group.finish();
}

/// 取反操作（Negate）
fn bench_tensor_negate(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_negate");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
    ];

    for &(name, shape) in configs {
        let a = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                let _result = -&a;
            });
        });
    }
    group.finish();
}

// ===================== 矩阵乘法 =====================

/// 矩阵乘法（不同大小）
fn bench_tensor_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_matmul");
    group.sample_size(20);

    // (名称, M, K, N) → A[M,K] * B[K,N]
    let configs: &[(&str, usize, usize, usize)] = &[
        ("8x4_4x8", 8, 4, 8),         // NEAT 小网络
        ("32x32_32x32", 32, 32, 32),   // 小型 Linear
        ("64x784_784x128", 64, 784, 128), // MNIST Linear
        ("32x128_128x10", 32, 128, 10),   // 分类输出层
    ];

    for &(name, m, k, n) in configs {
        let a = Tensor::random(0.0, 1.0, &[m, k]);
        let b = Tensor::random(0.0, 1.0, &[k, n]);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                let _result = a.mat_mul(&b);
            });
        });
    }
    group.finish();
}

// ===================== where_with_tensor =====================

/// where_with_tensor（ReLU 反向等场景使用）
fn bench_tensor_where(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_where");

    let configs: &[(&str, &[usize])] = &[
        ("8x32", &[8, 32]),
        ("32x128", &[32, 128]),
        ("64x784", &[64, 784]),
    ];

    for &(name, shape) in configs {
        let a = Tensor::random(0.0, 1.0, shape);
        let b = Tensor::random(0.0, 1.0, shape);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                // 模拟 ReLU 反向：condition > 0 时取 a 值，否则取 0
                let _result = a.where_with_tensor(
                    &b,
                    |a_val, _| a_val > 0.0,
                    |a_val, b_val| a_val * b_val,
                    |_, _| 0.0,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_clone,
    bench_tensor_add,
    bench_tensor_add_assign,
    bench_tensor_mul,
    bench_tensor_negate,
    bench_tensor_matmul,
    bench_tensor_where,
);
criterion_main!(benches);
